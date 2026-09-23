"""Training transport and guards; no optimizer updates or semantic accuracy."""
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from availability_supervision import ReplyTargets,supervised_loss
from memory_availability import AvailabilityMemoryFusion
from native_memory_encoder import sha_file
from native_prefix_bank import read_prefix_batch,validate_prefix,HIDDEN,VOCAB
from train_availability_memory import ForwardFeatures,forward_example,require_training_ready


class TrainingTests(unittest.TestCase):
    def test_feature_forward_is_target_free_and_head_frozen(self):
        torch.manual_seed(3)
        module=AvailabilityMemoryFusion(8,3,2); head=torch.randn(24,8)
        features=ForwardFeatures(torch.randn(3,8),torch.randn(3,24),torch.empty(0,16))
        before,state=forward_example(module,features,head,1.)
        target=ReplyTargets((1,2),(3,4,0),2,1.,"fixture")
        losses=supervised_loss(before,state,target,state_coefficient=.2); losses["weighted_total"].backward()
        self.assertIsNone(head.grad)
        self.assertGreater(float(module.uncertainty_output.weight.grad.norm()),0)
        after,_=forward_example(module,features,head,1.)
        self.assertTrue(torch.equal(before,after))
        with self.assertRaises(ValueError): forward_example(module,{"labels":target},head,1.)
        with self.assertRaises(ValueError): forward_example(module,features,head.requires_grad_(),1.)

    def test_status_edit_alone_and_cpu_training_are_blocked(self):
        for config,report in (({},{}),({"prerequisites_complete":True},{}),
                              ({"prerequisites_complete":True,"source_manifest":"x","launch_command":"x"},{"passed":True,"device":"cpu"})):
            with self.assertRaises(ValueError): require_training_ready(config,"x",report,None)

    def test_portable_hash_matches_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bytes"; data=b"hash-test"*200000; path.write_bytes(data)
            self.assertEqual(sha_file(path),hashlib.sha256(data).hexdigest())

    def test_prefix_reader_rejects_reuse_and_bad_geometry(self):
        prefixes=[(1,2),(1,2,3)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"output"
            raw=b"BNPO0001"+struct.pack("<III",2,HIDDEN,VOCAB)
            for ids in prefixes: raw+=struct.pack("<III",len(ids),0,len(ids))+np.zeros(HIDDEN+VOCAB,dtype="<f4").tobytes()
            path.write_bytes(raw)
            h,l=read_prefix_batch(path,prefixes); self.assertEqual(h.shape,(2,HIDDEN)); self.assertEqual(l.shape,(2,VOCAB))
            changed=bytearray(raw); struct.pack_into("<I",changed,24,1); path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError,"reuse"): read_prefix_batch(path,prefixes)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError): read_prefix_batch(path,prefixes)

    def test_invalid_tokens_rejected_before_native_process(self):
        for ids in ([1,2],(1,),(-1,2),(1,73448),(True,2)):
            with self.assertRaises(ValueError):validate_prefix(ids)

    @unittest.skipUnless((ROOT/"build/memory_prefix_probe").exists(),"build prefix probe")
    def test_native_probe_rejects_corruption_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,output=Path(tmp)/"input",Path(tmp)/"output"
            source.write_bytes(b"bad"); command=[str(ROOT/"build/memory_prefix_probe"),"missing.gguf",str(source),str(output)]
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            output.write_bytes(b"keep")
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(output.read_bytes(),b"keep")


if __name__=="__main__":unittest.main()
