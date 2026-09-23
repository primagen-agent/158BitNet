from dataclasses import replace
import hashlib
import itertools
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from fine_span_reader import ByteLayout,FineSpanReader,boundary_chain
from fine_span_supervision import ByteTargets,fine_loss
from joint_span_reader import SpanFeatures,PrefixFeature
from joint_optimizer import tensor_digest
from native_memory_encoder import digest
from neural_memory_contract import ModelBinding
from value_transport import PayloadSnapshot,FactPayload,ReplyBinding


class ChainTests(unittest.TestCase):
    def test_partition_map_and_gradients_match_exhaustive(self):
        points=((0,1,4),(0,2,5),(1,3,5),(0,3,6))
        valid=[p for p in itertools.product(*points) if p[0]<=p[1]<p[2]<=p[3]]
        for seed in range(4):
            torch.manual_seed(seed);scores=tuple(torch.randn(3,dtype=torch.float64,requires_grad=True) for _ in range(4))
            totals=torch.stack([sum(z[row.index(t)] for z,row,t in zip(scores,points,p)) for p in valid])
            result=boundary_chain(points,scores,valid[0]);expected=totals.logsumexp(0)-totals[0]
            self.assertTrue(torch.allclose(result['logz'],totals.logsumexp(0),atol=1e-12,rtol=1e-12))
            self.assertEqual(result['map'],valid[int(totals.argmax())])
            a=torch.autograd.grad(result['nll'],scores,retain_graph=True);b=torch.autograd.grad(expected,scores)
            for x,y in zip(a,b):self.assertTrue(torch.isfinite(x).all());self.assertTrue(torch.allclose(x,y,atol=1e-12,rtol=1e-12))

    def test_unreachable_nodes_finite_zero_gradient(self):
        p=((0,9),(1,8),(2,7),(3,6));s=tuple(torch.zeros(2,requires_grad=True) for _ in p)
        r=boundary_chain(p,s,(0,1,2,3));r['nll'].backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in s));self.assertEqual(float(s[0].grad[1]),0.)

    def test_no_path_or_invalid_target_fails(self):
        with self.assertRaises(ValueError):boundary_chain(((2,),(0,),(1,),(0,)),tuple(torch.zeros(1) for _ in range(4)))
        p=((0,),(0,),(1,),(1,));s=tuple(torch.zeros(1) for _ in range(4))
        with self.assertRaises(ValueError):boundary_chain(p,s,(0,0,0,1))


class FineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1018);self.binding=ModelBinding(*[c*64 for c in 'abcd']);key=digest(vars(self.binding))
        self.model=FineSpanReader(8,4,key,width=8)
        self.raw=b' AArhus.';self.ranges=((0,3),(3,5),(5,8))
        self.x=SpanFeatures(torch.randn(2,8),torch.randn(3,8),torch.ones(3,dtype=torch.bool),key,
                            hashlib.sha256(self.raw).hexdigest(),'f'*64)
        self.layout=ByteLayout(self.x,self.raw,self.ranges);self.prefix=PrefixFeature((1,2),torch.randn(4),key)

    def forward(self):return self.model(self.x,self.layout,self.prefix,self.prefix.token_ids)

    def test_fact_and_value_can_both_start_end_inside_tokens(self):
        o=self.forward();target=ByteTargets(1,(1,7),(2,6),1)
        before=tensor_digest(self.model.state_dict());loss=fine_loss(o,target);loss['total'].backward()
        self.assertEqual(before,tensor_digest(self.model.state_dict()))
        for name,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(float(self.model.offset.weight.grad.abs().sum()),0)
        self.assertGreater(float(self.model.boundaries[-1].weight.grad.abs().sum()),0)

    def test_masks_include_all_legal_offsets_not_whitespace_rules(self):
        starts,ends=self.layout.endpoints();self.assertEqual(starts[0],(0,1,2));self.assertEqual(ends[0],(1,2,3))
        self.assertEqual(self.layout.covering_span((1,7)),(0,3))

    def test_utf8_character_split_across_tokens(self):
        raw='里加'.encode();self.x=replace(self.x,source=torch.randn(4,8),allowed=torch.ones(4,dtype=torch.bool),payload_sha256=hashlib.sha256(raw).hexdigest())
        self.layout=ByteLayout(self.x,raw,((0,1),(1,3),(3,4),(4,6)))
        starts,ends=self.layout.endpoints();self.assertEqual(starts,((0,),(),(3,),()));self.assertEqual(ends,((),(3,),(),(6,)))
        o=self.forward();self.assertIn((0,4),o.spans);self.assertNotIn((0,1),o.spans)
        fine_loss(o,ByteTargets(1,(0,6),(3,6),1))['total'].backward()
        self.assertTrue(torch.isfinite(self.model.offset.weight.grad).all())
        with self.assertRaises(ValueError):fine_loss(o,ByteTargets(1,(0,6),(1,6),1))

    def test_labels_do_not_change_forward_or_prediction(self):
        o=self.forward();p=self.model.predict(o)
        fine_loss(o,ByteTargets(1,(0,8),(1,7),0));fine_loss(o,ByteTargets(2,None,None,0))
        self.assertEqual(p,self.model.predict(o))
        with self.assertRaises(TypeError):self.model(self.x,self.layout,self.prefix,self.prefix.token_ids,ByteTargets(2,None,None,0))

    def test_empty_route_and_null(self):
        self.x=replace(self.x,source=torch.empty(0,8),allowed=torch.empty(0,dtype=torch.bool),payload_sha256=hashlib.sha256(b'').hexdigest())
        self.layout=ByteLayout(self.x,b'',())
        o=self.forward();self.assertTrue(torch.isneginf(o.route_logits[1]));self.assertIsNone(self.model.predict(o)['value_bytes'])
        self.assertTrue(torch.isfinite(fine_loss(o,ByteTargets(2,None,None,0))['total']))

    def test_changed_features_output_or_weights_rejected(self):
        o=self.forward()
        with torch.no_grad():o.boundary_scores.add_(1)
        with self.assertRaises(ValueError):self.model.predict(o)
        o=self.forward()
        with torch.no_grad():self.x.source.add_(1)
        with self.assertRaises(ValueError):fine_loss(o,ByteTargets(2,None,None,0))
        o=self.forward()
        with torch.no_grad():self.model.offset.weight.add_(1)
        with self.assertRaises(ValueError):self.model.predict(o)

    def test_capacity_bad_layout_and_prefix_rejected(self):
        with self.assertRaises(ValueError):self.model(self.x,self.layout,self.prefix,(1,3))
        with self.assertRaises(ValueError):replace(self.layout,payload=b'wrong').endpoints()
        raw=b'x'*65;x=replace(self.x,source=torch.randn(1,8),allowed=torch.ones(1,dtype=torch.bool),payload_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):ByteLayout(x,raw,((0,65),)).endpoints()

    def test_predicted_handle_not_gold_and_binding_protected(self):
        with torch.no_grad():
            self.model.route[-1].weight.zero_();self.model.route[-1].bias.copy_(torch.tensor([-10.,10.,-10.]))
            self.model.mode[-1].weight.zero_();self.model.mode[-1].bias.copy_(torch.tensor([-10.,10.]))
        o=self.forward();p=self.model.predict(o)
        snapshot=PayloadSnapshot(self.binding,tensor_digest(self.model.state_dict()),'u','m',0,(FactPayload('source',self.raw),));reply=ReplyBinding('u','r','f'*64)
        h=self.model.handle(o,snapshot,reply);self.assertEqual((h.span.start_byte,h.span.end_byte),p['value_bytes'])
        fs,fe=p['fact_bytes'];vs,ve=p['value_bytes'];self.assertTrue(fs<=vs<ve<=fe)
        for s,r in ((replace(snapshot,memory_model_sha256='9'*64),reply),(snapshot,replace(reply,context_sha256='9'*64)),
                    (replace(snapshot,facts=(FactPayload('source',b'wrong'),)),reply)):
            with self.assertRaises(ValueError):self.model.handle(o,s,r)


if __name__=='__main__':unittest.main()
