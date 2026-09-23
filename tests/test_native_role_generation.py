"""Harness invariants; fixtures are not C qualification or semantic scores."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from native_role_generation import NativeRoleBackend
from native_memory_encoder import sha_file


class RoleBackendTests(unittest.TestCase):
    def test_legacy_model_rejected_before_file_access(self):
        with self.assertRaises(ValueError):NativeRoleBackend(fusion=object())

    def test_one_decision_for_actual_prediction_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);probe=root/'probe';probe.write_bytes(b'fixture')
            backend=object.__new__(NativeRoleBackend)
            backend.probe=backend.reference_probe=str(probe);backend.root=root;backend.gguf='fixture'
            backend.identity={'probe_sha256':sha_file(probe),'reference_probe_sha256':sha_file(probe)}
            backend.condition='role_separated_predicted';backend.bound_sources=();backend.prepared=None
            backend.steps=[];backend.previous_prefix=backend.previous_prediction=backend.decision=None
            calls=[]
            def decide(*args):
                calls.append(True);return SimpleNamespace(logits=torch.tensor([[2.,0.,0.]]),probabilities=torch.tensor([[.8,.1,.1]]))
            backend.fusion=SimpleNamespace(decide=decide,selected_residual=lambda h,d:torch.zeros_like(h))
            base={'hidden':np.zeros(1024,dtype=np.float32),'base':np.zeros(73448,dtype=np.float32),'logits':np.zeros(73448,dtype=np.float32)}
            with patch('native_role_generation.native_forward',return_value=base),patch('native_role_generation.forward',return_value=base),patch('native_role_generation.check_trace',return_value={}):
                backend((1,2),());backend((1,2,0),())
                self.assertEqual(len(calls),1)
                self.assertEqual([s['decision_created'] for s in backend.steps],[True,False])
                self.assertTrue(all(s['output_equals_base'] for s in backend.steps))
                with self.assertRaises(ValueError):backend((1,2,0,7),())
                with self.assertRaises(ValueError):backend((1,2,0,0),('different source',))

    def test_prefix_and_source_guards(self):
        b=object.__new__(NativeRoleBackend)
        for prefix in ([1,2],(1,),(1,-1),(True,2)):
            with self.assertRaises(ValueError):b(prefix,())
        b.bound_sources=None
        for sources in (['text'],('a','b'),('',)):
            with self.assertRaises(ValueError):b._bind_sources(sources)


if __name__=='__main__':unittest.main()
