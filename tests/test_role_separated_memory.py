"""CG-002 structural/loss guards. Synthetic fixture numbers are not accuracy."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from availability_supervision import ReplyTargets
from memory_role_separated import RoleSeparatedMemory,RoleOutputs,forward_roles,role_loss
from train_availability_memory import ForwardFeatures


class RoleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17); self.m=RoleSeparatedMemory(8,2,2)
        self.f=ForwardFeatures(torch.randn(3,8),torch.randn(3,13),torch.randn(7,16)); self.head=torch.randn(13,8)

    def target(self,state):
        return ReplyTargets((1,2),(3,4,5),state,2.,'reference')

    def open_outputs(self):
        with torch.no_grad():
            self.m.content.output.weight.normal_(std=.1); self.m.uncertainty_output.weight.normal_(std=.1)
        return forward_roles(self.m,self.f,self.head,4.)

    def test_zero_initialization_and_live_output_gradients(self):
        o=forward_roles(self.m,self.f,self.head,4.)
        self.assertTrue(torch.equal(o.base,o.content) and torch.equal(o.base,o.uncertainty))
        for state,param in ((1,self.m.content.output.weight),(2,self.m.uncertainty_output.weight)):
            loss=role_loss(o,self.target(state))['branch']
            g=torch.autograd.grad(loss,param,retain_graph=True)[0]
            self.assertGreater(float(g.norm()),0)

    def test_losses_do_not_cross_specialists_or_private_classifier(self):
        o=self.open_outputs(); classifier=tuple(self.m.state_encoder.parameters())+tuple(self.m.state_head.parameters())
        for state,other in ((1,self.m.uncertainty_output.weight),(2,self.m.content.output.weight)):
            loss=role_loss(o,self.target(state))
            grads=torch.autograd.grad(loss['branch'],classifier+(other,),allow_unused=True,retain_graph=True)
            self.assertTrue(all(g is None or not torch.count_nonzero(g) for g in grads))
            state_grads=torch.autograd.grad(loss['state'],classifier,retain_graph=True)
            self.assertTrue(any(float(g.norm())>0 for g in state_grads))

    def test_normal_loss_matches_independent_kl_and_ignores_answer_tokens(self):
        o=self.open_outputs(); target=self.target(0)
        actual=role_loss(o,target)['branch']
        p=o.base.double().softmax(-1)
        expected=sum((p*(p.log()-t.double().log_softmax(-1))).sum(-1).mean() for t in (o.content,o.uncertainty))/2
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12))
        self.assertTrue(torch.equal(actual,role_loss(o,replace(target,completion_token_ids=(9,8,7),response='different'))['branch']))
        z=forward_roles(RoleSeparatedMemory(8,2,2),self.f,self.head,4.)
        self.assertEqual(float(role_loss(z,target)['branch'].detach()),0.)

    def test_predicted_normal_and_disabled_are_exact_bypass(self):
        self.open_outputs()
        with torch.no_grad(): self.m.state_head.weight.zero_(); self.m.state_head.bias.copy_(torch.tensor([9.,0.,0.]))
        prepared=self.m.prepare(self.f.source); d=self.m.decide(self.f.hidden[:1],prepared)
        self.assertEqual(torch.count_nonzero(self.m.selected_residual(self.f.hidden,d)),0)
        self.assertEqual(torch.count_nonzero(self.m.selected_residual(self.f.hidden,d,enabled=False)),0)

    def test_uncertainty_cannot_read_source_values(self):
        self.open_outputs()
        with torch.no_grad(): self.m.state_head.weight.zero_(); self.m.state_head.bias.copy_(torch.tensor([0.,0.,9.]))
        a=self.m.decide(self.f.hidden[:1],self.m.prepare(self.f.source))
        b=self.m.decide(self.f.hidden[:1],self.m.prepare(self.f.source*10))
        self.assertTrue(torch.equal(self.m.selected_residual(self.f.hidden,a),self.m.selected_residual(self.f.hidden,b)))

    def test_prefill_is_single_row_and_frozen_for_reply(self):
        source=self.m.prepare(self.f.source); d=self.m.decide(self.f.hidden[:1],source)
        before=d.probabilities.clone(); self.m.selected_residual(self.f.hidden+3,d)
        self.assertTrue(torch.equal(before,d.probabilities))
        with self.assertRaises(ValueError): self.m.decide(self.f.hidden,source)
        with torch.no_grad(): self.m.state_head.bias.add_(.1)
        with self.assertRaises(ValueError): self.m.selected_residual(self.f.hidden,d)

    def test_forward_rejects_labels_and_trainable_backbone(self):
        with self.assertRaises(ValueError): forward_roles(self.m,self.target(1),self.head,4.)
        with self.assertRaises(ValueError): forward_roles(self.m,replace(self.f,hidden=self.f.hidden.requires_grad_()),self.head,4.)

    def test_empty_memory_masks_supported_and_has_zero_content(self):
        o=forward_roles(self.m,replace(self.f,source=torch.empty(0,16)),self.head,4.)
        self.assertTrue(torch.equal(o.content,o.base)); self.assertTrue(torch.isneginf(o.state_logits[0,1]))
        with self.assertRaises(ValueError): role_loss(o,self.target(1))

    def test_empty_normal_preservation_has_no_content_gradient(self):
        self.open_outputs()
        o=forward_roles(self.m,replace(self.f,source=torch.empty(0,16)),self.head,4.)
        term=role_loss(o,self.target(0))['branch']
        a,b=torch.autograd.grad(term,(self.m.content.output.weight,self.m.uncertainty_output.weight),allow_unused=True)
        self.assertIsNone(a); self.assertGreater(float(b.norm()),0)

    def test_draft_and_cpu_cannot_launch_training(self):
        from train_role_separated_memory import require_ready,validate_config
        import json
        config=json.loads((Path(__file__).resolve().parents[1]/'training/memory/neural-system/experiments/CG-002.json').read_text())
        validate_config(config)
        with self.assertRaises(ValueError): require_ready(config,{'passed':True,'device':'cpu'},None,None)
        changed={**config,'first_interval_steps':51}
        with self.assertRaises(ValueError):validate_config(changed)
        changed={**config,'constraints':{'lora':False}}
        with self.assertRaises(ValueError):validate_config(changed)


if __name__=='__main__': unittest.main()
