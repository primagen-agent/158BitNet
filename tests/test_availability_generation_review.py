"""Panel and scoring guards; these fixtures are not memory capability evidence."""
import base64
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python')); sys.path.insert(0,str(ROOT/'scripts'))
from download_training_artifact import recover
from eval_availability_checkpoint import fixed_panel
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from review_availability_generation import paired_summary, qualify_complete_generation
from review_availability_generation import validate_candidate
from eval_availability_checkpoint import BASELINE_SHA256


class GenerationReviewTests(unittest.TestCase):
    def test_role_candidate_cannot_use_legacy_hash_or_protocol(self):
        good={'checkpoint_sha256':'eb6b9a551c616d08a2aa0e8d183a63233c87a3c10e296f290a7b48b99dea716d',
              'baseline_sha256':BASELINE_SHA256,'format':'cg002-fixed-native-checkpoint-v1',
              'route_policy':'predicted_initial_prefill_argmax_fixed_for_reply','predictions':[
                  {'selected_route':0,'backend_cache_policy_verified':True,'every_base_reference_checked':True}]}
        validate_candidate(good,'cg002')
        with self.assertRaises(ValueError):validate_candidate(good,'cg001')
        for key in ('checkpoint_sha256','route_policy','format'):
            with self.assertRaises(ValueError):validate_candidate({**good,key:'changed'},'cg002')
        with self.assertRaises(ValueError):validate_candidate({**good,'predictions':[{'selected_route':True}]},'cg002')

    def test_original_panel_cannot_drop_reorder_or_replace_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus=Path(directory)/'data'
            materialize(ROOT/'training/memory/neural-system/data/CG-001-pilot.json',corpus)
            records=read_rows(corpus/'dev.inputs.jsonl'); index=read_rows(corpus/'dev.index.jsonl')
            worlds=sorted({r['world_id'] for r in index})[:2]
            ids={r['id'] for r in index if r['world_id'] in worlds and r['scenario'] in ('original','swapped_values','empty','ordinary_original')}
            rows=[{'id':r['id'],'input_sha256':digest(r)} for r in records if r['id'] in ids]
            self.assertEqual(len(fixed_panel(records,index,{'predictions':rows})),16)
            for invalid in (rows[:-1],list(reversed(rows)),[{**r,'input_sha256':'0'*64} for r in rows]):
                with self.assertRaises(ValueError): fixed_panel(records,index,{'predictions':invalid})

    def test_pair_requires_both_sides_and_retains_missing_denominator(self):
        rows=[{'id':'a','world_id':'w','language':'en','scenario':'original','status':'pass'},
              {'id':'b','world_id':'w','language':'en','scenario':'swapped_values','status':'fail'}]
        result=paired_summary(rows)
        self.assertEqual((result['value_swap']['total'],result['value_swap']['fail']),(1,1))
        self.assertEqual(result['remove_memory']['needs_review'],1)

    def test_clipped_reply_cannot_count_as_complete_success(self):
        rows=[{'id':s,'world_id':'w','language':'en','scenario':s,'status':'pass','reasons':[]}
              for s in ('original','swapped_values','empty','ordinary_original')]
        score={'cases':rows,'groups':{s['scenario']+'/en':{} for s in rows},'worlds':{'w':{}}}
        report={'predictions':[{'id':s['id'],'truncated':s['id']=='original','utf8_complete':True} for s in rows]}
        result=qualify_complete_generation(score,report)
        self.assertEqual((result['total'],result['pass'],result['fail']),(4,3,1))
        self.assertEqual(result['paired_controls']['value_swap']['pass'],0)
        self.assertEqual(result['worlds']['w']['fail'],1)

    def test_download_preserves_shell_control_prefix_and_rejects_corruption(self):
        nonce=b'a'*32; payload=b'checkpoint\0bytes'; sha=hashlib.sha256(payload).hexdigest()
        wire=(b'command echo\r\n\x1b[?2004l\rCGSTART '+nonce+b'\r\nCGDATA '+nonce+b' 0 '+base64.b64encode(payload)
              +b'\r\nCGEND '+nonce+b' '+sha.encode()+b'\r\nprompt')
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'raw'; target=Path(directory)/'checkpoint'
            source.write_bytes(wire)
            self.assertEqual(recover(source,target,sha),1); self.assertEqual(target.read_bytes(),payload)
            with self.assertRaises(FileExistsError): recover(source,target,sha)
            source.write_bytes(wire.replace(b' 0 ',b' 1 '))
            with self.assertRaisesRegex(ValueError,'sequence'): recover(source,Path(directory)/'bad',sha)


if __name__=='__main__': unittest.main()
