from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from autonomous_value_controller import LiveFrame,FrameOrigin,ActionKind
from joint_optimizer import tensor_digest
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_controller import ResearchUncertaintyController,UncertaintyDecision


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1021);self.model=QueryOnlyUncertainty(4,'fixture').eval();self.head=torch.randn(9,4)
        self.frame=LiveFrame((1,2),torch.randn(4),torch.tensor([-.0,0.,1.,2.,3.,4.,5.,6.,7.]),'fixture',FrameOrigin.SYNTHETIC_TEST)

    def forward(self,**kw):return self.model(self.frame,self.frame.prefix_ids,self.head,4.,**kw)

    def test_zero_identity_preserves_bits_and_has_expected_gradients(self):
        before=tensor_digest(self.model.state_dict());o=self.forward()
        self.assertTrue(torch.equal(o.logits.view(torch.int32),self.frame.logits.view(torch.int32)))
        F.cross_entropy(o.logits[None],torch.tensor([0])).backward()
        self.assertGreater(float(self.model.output.weight.grad.abs().sum()),0)
        self.assertEqual(float(self.model.encoder.weight.grad.abs().sum()),0.)
        self.assertEqual(float(self.model.encoder.bias.grad.abs().sum()),0.)
        self.assertEqual(before,tensor_digest(self.model.state_dict()))

    def test_nonzero_fixture_connects_both_layers(self):
        with torch.no_grad():self.model.output.weight.copy_(torch.eye(4)*.1)
        o=self.forward();self.assertGreater(float(o.correction.detach().abs().sum()),0)
        F.cross_entropy(o.logits[None],torch.tensor([0])).backward()
        for p in self.model.parameters():self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)

    def test_disabled_never_calls_network(self):
        with patch.object(self.model.encoder,'forward',side_effect=AssertionError('disabled encoder called')):
            o=self.forward(enabled=False)
        self.assertIs(o.logits,self.frame.logits);self.assertEqual(int(torch.count_nonzero(o.residual)),0)

    def test_memory_labels_or_extra_inputs_are_not_accepted(self):
        with self.assertRaises(TypeError):self.model(self.frame,self.frame.prefix_ids,self.head,4.,source='secret')
        with self.assertRaises(TypeError):self.model(self.frame,self.frame.prefix_ids,self.head,4.,answer='secret')
        with self.assertRaises(ValueError):self.model({'teacher':'answer'},(1,2),self.head,4.)

    def test_exact_prefix_and_frozen_head(self):
        with self.assertRaises(ValueError):self.model(self.frame,(1,3),self.head,4.)
        with self.assertRaises(ValueError):self.model(self.frame,(1,2),self.head.clone().requires_grad_(),4.)
        with self.assertRaises(ValueError):self.model(replace(self.frame,binding='wrong'),(1,2),self.head,4.)

    def test_stale_output_rejected(self):
        o=self.forward()
        with torch.no_grad():self.model.encoder.weight.add_(1)
        with self.assertRaises(ValueError):o.validate(self.model)
        o=self.forward();self.head.add_(1)
        with self.assertRaises(ValueError):o.validate(self.model)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        import test_autonomous_value_controller as fixture
        self.f=fixture.ControllerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.prediction.update(route=2,mode='generate')
        self.branch=QueryOnlyUncertainty(4,self.f.key).eval();self.head=torch.randn(self.f.codec.vocab,4)

    def wrapper(self,**kw):
        self.core=self.f.controller();return ResearchUncertaintyController(self.core,self.branch,self.head,4.,diagnostic_only=kw.get('diagnostic_only',True))

    def frame(self,wrapper,base=ord('X')+2):
        logits=torch.zeros(self.f.codec.vocab);logits[base]=1
        return LiveFrame(wrapper.prefix,torch.randn(4),logits,self.f.key,FrameOrigin.SYNTHETIC_TEST)

    def test_own_handoff_zero_branch_then_eos(self):
        w=self.wrapper();d=w.propose(self.frame(w));self.assertIsInstance(d,UncertaintyDecision)
        self.assertEqual(d.token,ord('X')+2);self.assertEqual(w.state,'uncertainty_active');w.commit(d,d.token)
        d=w.propose(self.frame(w,1));w.commit(d,1);self.assertEqual(w.state,'done')
        with self.assertRaises(ValueError):w.propose(self.frame(w))

    def test_diagnostic_opt_in_required(self):
        with self.assertRaises(ValueError):self.wrapper(diagnostic_only=False)

    def test_normal_never_calls_uncertainty_branch(self):
        self.f.prediction.update(route=0,mode='generate');w=self.wrapper()
        with patch.object(self.branch,'forward',side_effect=AssertionError('normal called uncertainty')):
            d=w.propose(self.frame(w));self.assertIs(d.kind,ActionKind.BASE);w.commit(d,d.token)

    def test_supported_copy_stays_on_existing_core(self):
        self.f.prediction.update(route=1,mode='start');w=self.wrapper()
        with patch.object(self.branch,'forward',side_effect=AssertionError('copy called uncertainty')):
            d=w.propose(self.frame(w));self.assertIs(d.kind,ActionKind.START);w.commit(d,d.token)

    def test_wrong_frame_commit_and_pending_mutation_fail(self):
        for action in ('copy','wrong_token','mutate_frame'):
            w=self.wrapper();frame=self.frame(w);d=w.propose(frame)
            if action=='mutate_frame':frame.hidden.add_(1)
            with self.assertRaises(ValueError):w.commit(replace(d) if action=='copy' else d,1 if action=='wrong_token' else d.token)
            self.assertEqual(w.state,'failed')

    def test_weights_or_head_cannot_change_after_handoff(self):
        w=self.wrapper();d=w.propose(self.frame(w))
        with torch.no_grad():self.branch.output.weight.add_(.1)
        with self.assertRaises(ValueError):w.commit(d,d.token)

    def test_teacher_frame_and_external_waiting_controller_rejected(self):
        w=self.wrapper()
        with self.assertRaises(ValueError):w.propose({'teacher':'answer'})
        core=self.f.controller();core.propose(self.f.frame(core))
        with self.assertRaises(ValueError):ResearchUncertaintyController(core,self.branch,self.head,4.,diagnostic_only=True)

    def test_budget_and_origin_binding(self):
        core=self.f.controller(max_new_tokens=1);w=ResearchUncertaintyController(core,self.branch,self.head,4.,diagnostic_only=True)
        d=w.propose(self.frame(w));w.commit(d,d.token)
        with self.assertRaises(ValueError):w.propose(self.frame(w))
        w=self.wrapper();d=w.propose(self.frame(w));w.commit(d,d.token)
        with self.assertRaises(ValueError):w.propose(replace(self.frame(w),origin=FrameOrigin.NATIVE_FRESH))


if __name__=='__main__':unittest.main()
