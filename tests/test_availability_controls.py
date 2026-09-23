"""No-model-fixture gates; real C-feature controls are registered as DG-007."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from availability_controls import controlled_read, prefill_decision, initialize_content_output
from memory_availability import AvailabilityMemoryFusion


class ControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(70)
        self.m=AvailabilityMemoryFusion(8,2,2)
        with torch.no_grad():
            self.m.content.layer_gain[1]=.2
            self.m.uncertainty_output.weight.normal_(std=.1)
        self.h=torch.randn(4,8); self.prepared=self.m.prepare(torch.randn(6,16))

    def test_isolation_preserves_values_and_blocks_entire_classifier_gradient(self):
        controller=list(self.m.state_encoder.parameters())+list(self.m.state_head.parameters())
        for source in (None,self.prepared):
            a=self.m.inspect(1,self.h,source)
            b=controlled_read(self.m,1,self.h,source,isolate=True)
            self.assertTrue(torch.equal(a.residual,b.residual))
            gradients=torch.autograd.grad(b.residual.square().sum(),controller,allow_unused=True,retain_graph=True)
            self.assertTrue(all(g is None or not torch.count_nonzero(g) for g in gradients))
            grad=torch.autograd.grad(b.residual.square().sum(),self.m.uncertainty_output.weight,retain_graph=True)[0]
            self.assertGreater(float(grad.norm()),0)
            cls=F.cross_entropy(b.state_logits[:1],torch.tensor([2]))
            self.assertTrue(any(float(g.norm())>0 for g in torch.autograd.grad(cls,controller,retain_graph=True)))

    def test_fixed_decision_is_independent_of_answer_prefix_and_has_initial_state_gradient(self):
        d=prefill_decision(self.m,self.h[:1],self.prepared)
        a=controlled_read(self.m,1,self.h,self.prepared,decision=d)
        b=controlled_read(self.m,1,self.h+12,self.prepared,decision=d)
        self.assertTrue(torch.equal(a.state_probabilities,b.state_probabilities))
        self.assertEqual(a.state_logits.shape,(1,3))
        self.assertTrue(a.state_logits.requires_grad)
        initial=controlled_read(self.m,1,self.h[:1],self.prepared,decision=d)
        self.assertTrue(torch.equal(initial.residual,self.m.inspect(1,self.h[:1],self.prepared).residual))
        with self.assertRaisesRegex(ValueError,'one initial'): prefill_decision(self.m,self.h,self.prepared)

    def test_decision_rejects_different_source_model_parameter_and_tensor_mutation(self):
        d=prefill_decision(self.m,self.h[:1],self.prepared)
        with self.assertRaises(ValueError): controlled_read(self.m,1,self.h,None,decision=d)
        with self.assertRaises(ValueError): controlled_read(AvailabilityMemoryFusion(8,2,2),1,self.h,self.prepared,decision=d)
        with torch.no_grad(): self.m.state_head.bias.add_(1)
        with self.assertRaises(ValueError): controlled_read(self.m,1,self.h,self.prepared,decision=d)
        d=prefill_decision(self.m,self.h[:1],self.prepared)
        with torch.no_grad(): d.probabilities.add_(1)
        with self.assertRaisesRegex(ValueError,'mutated'): controlled_read(self.m,1,self.h,self.prepared,decision=d)

    def test_decision_rejects_in_place_memory_mutation(self):
        d=prefill_decision(self.m,self.h[:1],self.prepared)
        with torch.no_grad(): self.prepared[0].add_(.01)
        with self.assertRaisesRegex(ValueError,'memory changed'): controlled_read(self.m,1,self.h,self.prepared,decision=d)

    def test_isolation_also_blocks_the_fixed_prefill_controller(self):
        d=prefill_decision(self.m,self.h[:1],self.prepared)
        r=controlled_read(self.m,1,self.h,self.prepared,decision=d,isolate=True)
        classifier=list(self.m.state_encoder.parameters())+list(self.m.state_head.parameters())
        gradients=torch.autograd.grad(r.residual.sum(),classifier,allow_unused=True)
        self.assertTrue(all(g is None or not torch.count_nonzero(g) for g in gradients))

    def test_disabled_and_empty_content_remain_exact_zero(self):
        a=controlled_read(self.m,1,self.h,None,isolate=True)
        self.assertEqual(torch.count_nonzero(a.content_residual),0)
        self.assertEqual(torch.count_nonzero(a.state_probabilities[:,1]),0)
        b=controlled_read(self.m,1,self.h,self.prepared,isolate=True,enabled=False)
        self.assertEqual(torch.count_nonzero(b.residual),0)
        with self.assertRaises(ValueError): controlled_read(self.m,1,self.h,self.prepared,decision='gold')

    def test_alternative_starts_zero_with_live_output_gradients_and_rejects_trained_model(self):
        m=AvailabilityMemoryFusion(8,2,2); initialize_content_output(m)
        source=m.prepare(torch.randn(6,16)); r=controlled_read(m,1,self.h,source,isolate=True)
        self.assertEqual(torch.count_nonzero(r.residual),0)
        # Use a nonzero downstream derivative: squared zero loss would be a false dead-gradient test.
        gradients=torch.autograd.grad((r.residual*torch.randn_like(r.residual)).sum(),
                                     (m.content.output.weight,m.uncertainty_output.weight))
        self.assertTrue(all(torch.isfinite(g).all() and float(g.norm())>0 for g in gradients))
        with self.assertRaises(ValueError): initialize_content_output(self.m)


if __name__=='__main__': unittest.main()
