from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from autonomous_value_controller import LiveFrame,FrameOrigin
from joint_optimizer import tensor_digest
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_supervision import (UncertaintyTrajectory,aggregate_loss,disabled_identity,
                                           gradient_structure,prefix_chain,trajectory_losses)


def trajectory(seed=0,n=4,vocab=9,eos=7):
    g=torch.Generator().manual_seed(seed)
    branch=QueryOnlyUncertainty(4,'fixture').eval();head=torch.randn(vocab,4,generator=g)
    prompt=(0,1,2);targets=tuple(3+i%n for i in range(n-1))+(eos,)
    frames=tuple(LiveFrame(prompt+targets[:i],torch.randn(4,generator=g),
                  torch.randn(vocab,generator=g),'fixture',FrameOrigin.SYNTHETIC_TEST) for i in range(n))
    return branch,head,UncertaintyTrajectory('fixture-record',2,prompt,targets,eos,64,frames)


class ChainTests(unittest.TestCase):
    def test_prefix_chain_is_exact_teacher_forcing(self):
        self.assertEqual(prefix_chain((0,1),(5,6,7)),((0,1),(0,1,5),(0,1,5,6)))

    def test_route_guard_rejects_non_uncertainty_targets(self):
        b,h,t=trajectory()
        from dataclasses import replace
        with self.assertRaises(ValueError):replace(t,route_target=0)
        with self.assertRaises(ValueError):replace(t,route_target=1)


class StructureTests(unittest.TestCase):
    def test_missing_eos_or_over_budget_rejected(self):
        b,h,t=trajectory()
        from dataclasses import replace
        with self.assertRaises(ValueError):replace(t,target_ids=t.target_ids[:-1])
        with self.assertRaises(ValueError):replace(t,max_new_tokens=len(t.target_ids)-1)

    def test_frame_prefix_deviation_rejected(self):
        b,h,t=trajectory()
        from dataclasses import replace
        bad=replace(t.frames[1],prefix_ids=t.frames[1].prefix_ids+(99,))
        with self.assertRaises(ValueError):replace(t,frames=(t.frames[0],bad)+t.frames[2:])

    def test_labels_are_not_forward_arguments(self):
        b,h,t=trajectory()
        with self.assertRaises(TypeError):trajectory_losses(b,t,h,4.,answer='secret')
        with self.assertRaises(TypeError):aggregate_loss([t],b,h,4.,target_ids=t.target_ids)


class SupervisionTests(unittest.TestCase):
    def test_zero_init_trajectory_gradients_follow_chain_rule(self):
        b,h,t=trajectory(seed=1)
        before=tensor_digest(b.state_dict());loss,sums=aggregate_loss([t],b,h,4.)
        self.assertEqual(len(sums),1);self.assertTrue(all(torch.isfinite(x) for x in sums));loss.backward()
        s=gradient_structure(b)
        self.assertTrue(s['expected_zero_init_chain_rule']);self.assertFalse(s['connected'])
        self.assertEqual(tensor_digest(b.state_dict()),before)

    def test_fixture_trajectory_connects_both_layers(self):
        b,h,t=trajectory(seed=2)
        with torch.no_grad():b.output.weight.copy_(torch.eye(4)*.001)
        loss,_=aggregate_loss([t],b,h,4.);loss.backward()
        self.assertTrue(gradient_structure(b)['connected'])

    def test_equal_weight_aggregate_and_accumulation(self):
        b,h1,t1=trajectory(seed=3);_,h2,t2=trajectory(seed=4);_,h3,t3=trajectory(seed=5)
        mean,sums=aggregate_loss([t1,t2,t3],b,h1,4.)
        self.assertTrue(torch.isfinite(mean))
        self.assertAlmostEqual(float(mean),float((sums[0]+sums[1]+sums[2])/3),places=6)
        mean.backward()
        self.assertGreater(float(b.output.weight.grad.abs().sum()),0)

    def test_per_position_losses_are_read_after_forward(self):
        b,h,t=trajectory(seed=6)
        losses=trajectory_losses(b,t,h,4.)
        self.assertEqual(len(losses),len(t.target_ids))
        with torch.no_grad():
            zero=b(t.frames[0],t.frames[0].prefix_ids,h,4.)
        self.assertTrue(torch.equal(zero.logits.view(torch.int32),t.frames[0].logits.view(torch.int32)))


class BypassTests(unittest.TestCase):
    def test_disabled_identity_holds_at_every_position(self):
        b,h,t=trajectory(seed=7)
        with torch.no_grad():b.output.weight.copy_(torch.eye(4)*.1)
        self.assertTrue(disabled_identity(t,b,h,4.))


if __name__=='__main__':unittest.main()
