from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from value_teacher_trajectory import TeacherTrajectory,compile_teacher
from fine_span_supervision import ByteTargets
from native_memory_encoder import digest
from neural_memory_contract import ModelBinding


class FixtureCodec:
    backbone_sha256='a'*64;tokenizer_sha256='b'*64;bos_id=0;vocab=10000
    def __init__(self):self.tokenizer=self;self.pieces={999:b'<|im_end|>'}
    def verify_identity(self):pass
    def encode(self,text,add_bos):
        self.pieces[1]=text.encode();return [0,1] if add_bos else [1]
    def encode_value(self,value):
        ids=tuple(100+b for b in value)
        self.pieces.update({100+b:bytes([b]) for b in value});return ids
    def decode_pieces(self,ids):return tuple(self.pieces[t] for t in ids)
    def forbidden(self,t):return t in (0,999)
    def eos(self):return 999


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.codec=FixtureCodec();self.binding=ModelBinding(*[c*64 for c in 'abcd'])
        self.runtime={'id':'fixture','context':[{'role':'user','speaker':'user','text':'Where does A live?'}],
                      'episodes':[{'role':'user','speaker':'user','text':'A currently lives in Riga.'}]}
        self.meta={'id':'fixture','split':'train','language':'en','query_subject':'A','query_relation':'home_city',
                   'source_facts':[{'subject':'A','relation':'home_city','value':'Riga','time':'current','status':'actual'}]}
        self.label={'id':'fixture','input_sha256':digest(self.runtime),'state':'supported','response':'A lives in Riga.'}

    def compile(self,**kw):
        args=dict(runtime=self.runtime,label=self.label,meta=self.meta,codec=self.codec,binding=self.binding,memory_model_sha256='e'*64,purpose='training_diagnostic')
        args.update(kw);return compile_teacher(**args)

    def test_full_teacher_bytes_and_transition_order(self):
        t=self.compile();self.assertEqual(b''.join(self.codec.decode_pieces(t.completion_ids[:-1])),self.label['response'].encode())
        self.assertEqual(t.phases.count('START'),1);self.assertEqual(t.phases.count('END'),1)
        s=t.phases.index('START');e=t.phases.index('END');self.assertEqual(e-s,4)
        self.assertEqual(t.idle_targets[s],1);self.assertTrue(all(x is None for x in t.idle_targets[s+1:e]))
        for i in range(len(t.completion_ids)):self.assertEqual(t.prefix(i,purpose='training_diagnostic'),t.prompt_ids+t.completion_ids[:i])

    def test_stop_can_be_end_transition(self):
        t=self.compile(label={**self.label,'response':'Riga'});self.assertEqual(t.phases[-1],'END');self.assertEqual(t.completion_ids[-1],999)

    def test_no_positive_modes_for_normal_or_negative(self):
        for state in ('insufficient','no_memory_needed'):
            t=self.compile(label={**self.label,'state':state,'response':'Unknown.'})
            self.assertEqual(set(t.phases),{'GENERATE'});self.assertEqual(set(t.idle_targets),{0})

    def test_dev_test_and_inference_rejected(self):
        for split in ('dev','test'):
            with self.assertRaises(ValueError):self.compile(meta={**self.meta,'split':split})
        with self.assertRaises(ValueError):self.compile(purpose='inference')
        with self.assertRaises(ValueError):self.compile().prefix(0,purpose='inference')

    def test_changed_label_identity_rejected(self):
        with self.assertRaises(ValueError):self.compile(label={**self.label,'input_sha256':'0'*64})

    def test_ambiguous_value_and_capacity_not_silently_dropped(self):
        with self.assertRaises(ValueError):self.compile(label={**self.label,'response':'Riga Riga'})
        with self.assertRaises(ValueError):self.compile(capacity=3)

    def test_all_transition_positions_get_direct_reference(self):
        t=self.compile();selected=t.reference_positions()
        self.assertTrue({0,len(t.phases)//2,len(t.phases)-1}.issubset(selected))
        self.assertTrue(all(i in selected for i,p in enumerate(t.phases) if p!='GENERATE'))

    def test_malformed_phases_or_origins_rejected(self):
        t=self.compile()
        with self.assertRaises(ValueError):replace(t,origin='autonomous')
        with self.assertRaises(ValueError):replace(t,phases=('GENERATE',)*len(t.phases))
        with self.assertRaises(ValueError):replace(t,static_targets=ByteTargets(2,None,None,None))


if __name__=='__main__':unittest.main()
