"""Platform preflight fixtures are wiring checks, never recall evidence."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from joint_memory_model import create_joint_model
from memory_role_separated import RoleSeparatedMemory
from memory_token_read import TokenReadInput,TokenReadPrototype
from token_memory_composition import ComposedTokenMemory
from native_joint_generation import NativeJointBackend,check_extension
from qualify_joint_native import fixture_parameters
from joint_cuda_preflight import comparison,move
from joint_training_data import JointForwardFeatures


class JointPlatformTests(unittest.TestCase):
    def test_new_backend_rejects_old_composition_before_file_access(self):
        legacy=ComposedTokenMemory(TokenReadPrototype(8,13,'x'),RoleSeparatedMemory(8,2,2))
        with self.assertRaises(ValueError):
            NativeJointBackend(runtime={},tokenizer=None,tok_probe='',gguf='',probe='',reference_probe='',encoder_probe='',
                               model=legacy,head=None,scale=4.,output='unused')

    def test_nonzero_fixtures_cover_each_neural_route_for_both_policies(self):
        torch.manual_seed(9)
        inputs=TokenReadInput(torch.randn(4,16),torch.randn(5,16),torch.tensor([2,3,4,5,6]),torch.ones(5,dtype=torch.bool),'test')
        for arm in ('joint_aux','joint_product'):
            for route in (0,1,2):
                m=create_joint_model('test',(0,),arm,hidden=8,vocab=13,layers=2,heads=2)
                fixture_parameters(m,route)
                self.assertGreater(int(torch.count_nonzero(m.roles.content.output.weight)),0)
                self.assertGreater(int(torch.count_nonzero(m.roles.uncertainty_output.weight)),0)
                state=m.prefill(inputs);self.assertEqual(int(state.core.state_logits.argmax()),route)

    def test_new_backend_keeps_actual_prefix_guard(self):
        check_extension((1,2),(1,2),3,(1,2,3),('source',),('source',))
        with self.assertRaises(ValueError):check_extension((1,2),(1,2),3,(1,2,4),('source',),('source',))

    def test_comparison_rejects_nonfinite_and_outside_tolerance(self):
        a=torch.tensor([1.,2.]);self.assertTrue(comparison(a,a,1e-5,1e-4)['passed'])
        self.assertFalse(comparison(a,a+.1,1e-5,1e-4)['passed'])
        self.assertFalse(comparison(a,torch.tensor([float('nan'),2.]),1e-5,1e-4)['passed'])

    def test_device_transfer_keeps_labels_outside_forward_features(self):
        memory=TokenReadInput(torch.randn(3,16),torch.empty(0,16),torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.bool),'test')
        original=JointForwardFeatures(memory,torch.randn(2,8),torch.randn(2,13))
        moved=move(original,'cpu')
        self.assertEqual(moved.memory.binding,'test');self.assertTrue(torch.equal(moved.hidden,original.hidden))
        self.assertEqual(set(vars(moved)),{'memory','hidden','base'})


if __name__=='__main__':unittest.main()
