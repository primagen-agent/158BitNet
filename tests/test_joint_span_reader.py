from dataclasses import replace
from pathlib import Path
import hashlib
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from joint_span_reader import JointSpanReader,SpanFeatures,PrefixFeature,PayloadAlignment,candidate_spans
from joint_span_supervision import SpanTargets,span_loss
from joint_optimizer import tensor_digest
from native_memory_encoder import digest
from neural_memory_contract import ModelBinding
from value_transport import PayloadSnapshot,FactPayload,ReplyBinding,Origin


class SpanTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1017)
        self.binding=ModelBinding(*[c*64 for c in 'abcd']);key=digest(vars(self.binding))
        self.model=JointSpanReader(8,4,key,width=8)
        self.x=SpanFeatures(torch.randn(3,8),torch.randn(4,8),torch.tensor([True,True,True,True]),key,
                            hashlib.sha256(b'abcd').hexdigest(),'f'*64)
        self.prefix=PrefixFeature((1,2,3),torch.randn(4),key)

    def forward(self):return self.model(self.x,self.prefix,self.prefix.token_ids)

    def test_exhaustive_spans_not_punctuation_or_gold(self):
        self.assertEqual(candidate_spans(torch.tensor([True,True,False,True])),((0,1),(0,2),(1,2),(3,4)))
        self.assertEqual(len(self.forward().spans),10)

    def test_joint_distribution_and_containment(self):
        o=self.forward();joint=(o.fact_logp[:,None]+o.value_logp).exp()
        self.assertAlmostEqual(float(joint.detach().sum()),1.,places=6)
        for f,(s,e) in enumerate(o.spans):
            for v,(a,b) in enumerate(o.spans):
                self.assertEqual(bool(torch.isfinite(o.value_logp[f,v])),s<=a<b<=e)

    def test_empty_no_support_no_nan(self):
        self.x=replace(self.x,source=torch.empty(0,8),allowed=torch.empty(0,dtype=torch.bool))
        o=self.forward();self.assertTrue(torch.isneginf(o.route_logits[1]));self.assertNotEqual(self.model.predict(o)['route'],1)
        self.assertTrue(torch.isfinite(span_loss(o,SpanTargets(2,None,None,0))['total']))

    def test_loss_is_post_forward_and_connected(self):
        before=tensor_digest(self.model.state_dict());o=self.forward()
        losses=span_loss(o,SpanTargets(1,(0,4),(1,3),1));losses['total'].backward()
        self.assertEqual(before,tensor_digest(self.model.state_dict()))
        for name,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.parameters()),0)

    def test_gold_labels_not_accepted_in_forward(self):
        with self.assertRaises(TypeError):self.model(self.x,self.prefix,self.prefix.token_ids,SpanTargets(1,(0,4),(0,1),1))
        with self.assertRaises(ValueError):self.model({'labels':'bad'},self.prefix,self.prefix.token_ids)

    def test_targets_cannot_rewrite_predictions(self):
        o=self.forward();before=self.model.predict(o)
        span_loss(o,SpanTargets(1,(0,2),(0,1),1));span_loss(o,SpanTargets(2,None,None,0))
        self.assertEqual(before,self.model.predict(o))

    def test_stale_forward_rejected(self):
        o=self.forward()
        with torch.no_grad():self.x.source.add_(1)
        with self.assertRaises(ValueError):self.model.predict(o)
        o=self.forward()
        with torch.no_grad():next(self.model.parameters()).add_(.1)
        with self.assertRaises(ValueError):span_loss(o,SpanTargets(2,None,None,0))

    def test_exact_prefix_identity_not_only_length(self):
        with self.assertRaises(ValueError):self.model(self.x,self.prefix,(1,2,4))
        with self.assertRaises(ValueError):self.model(self.x,self.prefix,list(self.prefix.token_ids))

    def test_bad_targets_and_capacity_fail(self):
        with self.assertRaises(ValueError):SpanTargets(2,(0,1),None,0)
        with self.assertRaises(ValueError):SpanTargets(1,(0,1),(1,2),1)
        with self.assertRaises(ValueError):span_loss(self.forward(),SpanTargets(1,(0,5),(0,1),1))
        with self.assertRaises(ValueError):candidate_spans(torch.ones(33,dtype=torch.bool))

    def test_frozen_feature_domain(self):
        for q in (self.x.query.double(),self.x.query.clone().requires_grad_(),torch.full((3,8),float('nan'))):
            with self.assertRaises(ValueError):self.model(replace(self.x,query=q),self.prefix,self.prefix.token_ids)

    def test_predicted_handle_binding_or_null(self):
        snapshot=PayloadSnapshot(self.binding,tensor_digest(self.model.state_dict()),'u','m',0,(FactPayload('source',b'abcd'),))
        reply=ReplyBinding('u','r','f'*64)
        alignment=PayloadAlignment(self.x,snapshot,reply,((0,1),(1,2),(2,3),(3,4)))
        o=self.forward();h=self.model.handle(o,alignment)
        self.assertIs(h.origin,Origin.NEURAL_PREDICTION);h.validate(snapshot,reply,allow_oracle=False)
        if h.span:
            pred=self.model.predict(o);self.assertEqual((h.span.start_byte,h.span.end_byte),pred['value'])
        for a in (replace(alignment,reply=replace(reply,context_sha256='9'*64)),
                  replace(alignment,snapshot=replace(snapshot,facts=(FactPayload('source',b'wxyz'),))),
                  replace(alignment,snapshot=replace(snapshot,memory_model_sha256='9'*64))):
            with self.assertRaises(ValueError):self.model.handle(o,a)

    def test_start_handle_uses_predicted_nested_spans(self):
        # Artificial routing fixture only; no optimizer and no learned capability claim.
        with torch.no_grad():
            self.model.route[-1].weight.zero_();self.model.route[-1].bias.copy_(torch.tensor([-10.,10.,-10.]))
            self.model.mode[-1].weight.zero_();self.model.mode[-1].bias.copy_(torch.tensor([-10.,10.]))
        snapshot=PayloadSnapshot(self.binding,tensor_digest(self.model.state_dict()),'u','m',0,(FactPayload('source',b'abcd'),))
        alignment=PayloadAlignment(self.x,snapshot,ReplyBinding('u','r','f'*64),((0,1),(1,2),(2,3),(3,4)))
        o=self.forward();p=self.model.predict(o);h=self.model.handle(o,alignment)
        self.assertEqual(p['mode'],'start');self.assertEqual((h.span.start_byte,h.span.end_byte),p['value'])
        self.assertTrue(p['fact'][0]<=p['value'][0]<p['value'][1]<=p['fact'][1])

    def test_empty_support_target_fails_not_dropped(self):
        self.x=replace(self.x,source=torch.empty(0,8),allowed=torch.empty(0,dtype=torch.bool))
        with self.assertRaises(ValueError):span_loss(self.forward(),SpanTargets(1,(0,1),(0,1),1))

    def test_exact_value_inside_token_is_not_silently_trimmed(self):
        from check_joint_span_interface import exact_span
        with self.assertRaisesRegex(ValueError,'unrepresentable exact byte boundary'):
            exact_span(b'city Aarhus.', 'Aarhus', ((0,4),(4,8),(8,11),(11,12)))


if __name__=='__main__':unittest.main()
