"""Reject edited prefixes, residuals, wire geometry and unsafe audit paths."""
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from audit_joint_generation import continuous_file,reference_file,close,audit_case
from native_memory_encoder import BACKBONE_SHA256


class JointGenerationAuditTests(unittest.TestCase):
    def write_continuous(self,root,delta):
        prefix=(1,2,3);n=len(prefix)
        (root/'x.input').write_bytes(b'BNCI0001'+struct.pack('<I',n)+np.asarray(prefix,dtype='<i4').tobytes()+
                                   struct.pack('<I',1)+delta.astype('<f4').tobytes())
        (root/'x.bin').write_bytes(b'BNCO0001'+struct.pack('<III',n,1024,73448)+np.zeros(1024+3*73448,dtype='<f4').tobytes())
        trace={'initial_position':0,'final_position':n,'eval_calls':1,'prefix_tokens':n}
        (root/'x.log').write_text('[bitnet] cpu tier: arm_neon\nBNC_TRACE '+json.dumps(trace)+'\n')
        return prefix

    def test_continuous_wire_binds_actual_prefix_and_exact_trained_residual(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);delta=np.ones(1024,dtype='<f4');prefix=self.write_continuous(root,delta)
            output=continuous_file(root,'x',prefix,True,torch.from_numpy(delta)[None])
            self.assertEqual(output[0].shape,(1,1024));self.assertEqual(output[3].shape,(1,73448))
            with self.assertRaises(ValueError):continuous_file(root,'x',(1,2,4),True)
            with self.assertRaises(ValueError):continuous_file(root,'x',prefix,True,torch.zeros(1,1024))

    def test_trailing_bytes_and_nonfinite_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prefix=self.write_continuous(root,np.zeros(1024,dtype='<f4'))
            raw=(root/'x.bin').read_bytes();(root/'x.bin').write_bytes(raw+b'x')
            with self.assertRaises(ValueError):continuous_file(root,'x',prefix,True)
            bad=bytearray(raw);struct.pack_into('<f',bad,20,float('nan'));(root/'x.bin').write_bytes(bad)
            with self.assertRaises(ValueError):continuous_file(root,'x',prefix,True)

    def test_reference_rejects_enabled_or_changed_position(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prefix=(1,2)
            wire=b'BNGI0001'+struct.pack('<III',2,1,2)+struct.pack('<III',23,1,0)+np.zeros(1024,dtype='<f4').tobytes()
            (root/'x.input').write_bytes(wire)
            (root/'x.bin').write_bytes(b'BNGO0001'+struct.pack('<IIIII',2,1024,73448,23,1)+np.zeros(3072+73448,dtype='<f4').tobytes())
            h,z=reference_file(root,'x',prefix);self.assertEqual(h.shape,(1,1024));self.assertEqual(z.shape,(1,73448))
            bad=bytearray(wire);struct.pack_into('<I',bad,28,1);(root/'x.input').write_bytes(bad)
            with self.assertRaises(ValueError):reference_file(root,'x',prefix)

    def test_close_requires_top1_even_inside_numeric_tolerance(self):
        a=torch.tensor([[1.,1.0000001]])
        self.assertEqual(close(a,a),0.)
        with self.assertRaises(ValueError):close(a,a.flip(-1))
        with self.assertRaises(ValueError):close(a,torch.full_like(a,float('nan')))

    def test_parent_path_rejected_before_reading_any_tensor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            backend={'cross_step_kv_reuse':False,'prefix_bank_used':False,'gold_answer_prefix_used':False,
                'enabled':True,'route_policy':'joint_aux','identity':{'encoder':'test','backbone_sha256':BACKBONE_SHA256},
                'raw_sha256':{'../outside':'x'}}
            (root/'backend.json').write_text(json.dumps(backend))
            with self.assertRaisesRegex(ValueError,'raw artifact changed'):
                audit_case(root,None,None,None,SimpleNamespace(route_policy='joint_aux'),None,None,{'encoder_identity':'test'},None)


if __name__=='__main__':unittest.main()
