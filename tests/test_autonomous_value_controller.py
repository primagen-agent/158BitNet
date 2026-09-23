from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from autonomous_value_controller import AutonomousValueController,LiveFrame,FrameOrigin,ActionKind,ControllerState
from fine_span_reader import FineSpanReader,ByteLayout
from joint_span_reader import SpanFeatures
from joint_optimizer import tensor_digest
from native_memory_encoder import digest
from neural_memory_contract import ModelBinding
from value_transport import FactPayload,PayloadSnapshot,ReplyBinding,TransportLimits,Origin


class FixtureCodec:
    vocab=512;bos_id=0;backbone_sha256='a'*64;tokenizer_sha256='b'*64
    def __init__(self):self.tokenizer=self
    def eos(self):return 1
    def verify_identity(self):pass
    def encode_value(self,value):return tuple(2+b for b in value)
    def decode_pieces(self,ids):return tuple(b'<|im_end|>' if t==1 else bytes([t-2]) for t in ids)
    def forbidden(self,t):return t in (0,1)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1020);self.codec=FixtureCodec();self.binding=ModelBinding(*[c*64 for c in 'abcd']);self.key=digest(vars(self.binding))
        self.model=FineSpanReader(8,4,self.key,width=8).eval();self.raw=b'Riga';self.prefix=(0,ord(':')+2)
        self.inputs=SpanFeatures(torch.randn(2,8),torch.randn(1,8),torch.tensor([True]),self.key,hashlib.sha256(self.raw).hexdigest(),'f'*64)
        self.layout=ByteLayout(self.inputs,self.raw,((0,4),))
        self.snapshot=PayloadSnapshot(self.binding,tensor_digest(self.model.state_dict()),'u','m',0,(FactPayload('source',self.raw),))
        self.reply=ReplyBinding('u','request','f'*64)
        # Explicit stub for STATE-MACHINE coverage only, never native model evidence.
        self.prediction={'route':1,'mode':'start','fact':(0,1),'value':(0,1),'fact_bytes':(0,4),'value_bytes':(0,4),'confidence':.8}
        self.patcher=patch.object(self.model,'predict',side_effect=lambda output:dict(self.prediction));self.patcher.start();self.addCleanup(self.patcher.stop)

    def controller(self,**kw):
        args=dict(model=self.model,inputs=self.inputs,layout=self.layout,snapshot=self.snapshot,reply=self.reply,
                  codec=self.codec,initial_prefix=self.prefix,allow_synthetic=True)
        args.update(kw);return AutonomousValueController(**args)

    def frame(self,c,base=ord('X')+2):
        logits=torch.zeros(self.codec.vocab);logits[base]=1
        return LiveFrame(c.prefix,torch.randn(4),logits,self.key,FrameOrigin.SYNTHETIC_TEST)

    def test_start_cursor_end_then_base_eos(self):
        c=self.controller();kinds=[]
        for _ in range(4):
            decision=c.propose(self.frame(c));kinds.append(decision.kind);c.commit(decision,decision.token)
        self.assertEqual(kinds,[ActionKind.START]+[ActionKind.CONTINUE]*3)
        self.assertEqual(b''.join(self.codec.decode_pieces(c.generated_tokens)),b'Riga')
        decision=c.propose(self.frame(c,base=1));self.assertIs(decision.kind,ActionKind.END);self.assertEqual(decision.token,1)
        c.commit(decision,1);self.assertIs(c.state,ControllerState.DONE)
        with self.assertRaises(ValueError):c.propose(self.frame(c))

    def test_active_cursor_ignores_new_predictions(self):
        c=self.controller();d=c.propose(self.frame(c));c.commit(d,d.token)
        self.prediction.update(route=2,mode='generate')
        d=c.propose(self.frame(c));self.assertIs(d.kind,ActionKind.CONTINUE);self.assertEqual(d.token,ord('i')+2)

    def test_normal_identity_and_eos(self):
        self.prediction.update(route=0,mode='generate');c=self.controller()
        for base in (ord('Q')+2,1):
            d=c.propose(self.frame(c,base));self.assertIs(d.kind,ActionKind.BASE);self.assertEqual(d.token,base);c.commit(d,base)
        self.assertIs(c.state,ControllerState.DONE)

    def test_supported_idle_is_base_not_forced_copy(self):
        self.prediction.update(mode='generate');c=self.controller();d=c.propose(self.frame(c))
        self.assertIs(d.kind,ActionKind.BASE);self.assertEqual(d.token,ord('X')+2)

    def test_insufficient_cannot_emit_or_commit(self):
        self.prediction.update(route=2,mode='generate');c=self.controller();d=c.propose(self.frame(c))
        self.assertIs(d.kind,ActionKind.NEEDS_UNCERTAINTY);self.assertIsNone(d.token)
        self.assertEqual(c.generated_tokens,());self.assertIs(c.state,ControllerState.WAIT_UNCERTAINTY)
        with self.assertRaises(ValueError):c.commit(d,ord('X')+2)
        self.assertEqual(c.generated_tokens,())

    def test_wrong_ack_fails_without_advancing(self):
        c=self.controller();d=c.propose(self.frame(c))
        with self.assertRaises(ValueError):c.commit(d,1)
        self.assertIs(c.state,ControllerState.FAILED);self.assertEqual(c.generated_tokens,())

    def test_action_copy_and_double_propose_rejected(self):
        for kind in ('copy','double'):
            c=self.controller();d=c.propose(self.frame(c))
            with self.assertRaises(ValueError):
                if kind=='copy':c.commit(replace(d),d.token)
                else:c.propose(self.frame(c))
            self.assertEqual(c.generated_tokens,())

    def test_stale_or_wrong_prefix_rejected(self):
        c=self.controller();frame=self.frame(c);d=c.propose(frame);c.commit(d,d.token)
        with self.assertRaises(ValueError):c.propose(frame)

    def test_teacher_and_oracle_rejected(self):
        c=self.controller()
        with self.assertRaises(ValueError):c.propose({'origin':'teacher_forcing_oracle','prefix_ids':c.prefix})
        c=self.controller();original=self.model.handle
        with patch.object(self.model,'handle',side_effect=lambda *a:replace(original(*a),origin=Origin.ORACLE_FIXTURE)):
            with self.assertRaises(ValueError):c.propose(self.frame(c))
        self.assertEqual(c.generated_tokens,())

    def test_synthetic_opt_in_required(self):
        c=self.controller(allow_synthetic=False)
        with self.assertRaises(ValueError):c.propose(self.frame(c))

    def test_invalid_live_frame_or_origin_rejected(self):
        for field in ('nan','binding','origin'):
            c=self.controller();f=self.frame(c)
            if field=='nan':f.hidden.fill_(float('nan'))
            elif field=='binding':f=replace(f,binding='wrong')
            else:f=replace(f,origin='teacher_forcing_oracle')
            with self.assertRaises(ValueError):c.propose(f)

    def test_model_mutation_rejected(self):
        c=self.controller()
        with torch.no_grad():self.model.offset.weight.add_(1)
        with self.assertRaises(ValueError):c.propose(self.frame(c))

    def test_source_mutation_rejected(self):
        c=self.controller()
        with torch.no_grad():self.inputs.source.add_(1)
        with self.assertRaises(ValueError):c.propose(self.frame(c))

    def test_codec_mutation_rejected(self):
        c=self.controller();self.codec.tokenizer_sha256='9'*64
        with self.assertRaises(ValueError):c.propose(self.frame(c))

    def test_frame_mutation_before_ack_rejected(self):
        c=self.controller();f=self.frame(c);d=c.propose(f);f.logits.add_(1)
        with self.assertRaises(ValueError):c.commit(d,d.token)

    def test_scope_request_context_model_rebinding_rejected(self):
        for kw in (dict(reply=replace(self.reply,scope='v')),dict(reply=replace(self.reply,context_sha256='9'*64)),
                   dict(snapshot=replace(self.snapshot,memory_model_sha256='9'*64))):
            with self.assertRaises(ValueError):self.controller(**kw)

    def test_value_budget_reserves_end_slot(self):
        for kw in (dict(max_new_tokens=4),dict(limits=TransportLimits(max_value_tokens=3)),dict(limits=TransportLimits(max_total_tokens=6))):
            c=self.controller(**kw)
            with self.assertRaises(ValueError):c.propose(self.frame(c))
            self.assertEqual(c.generated_tokens,())

    def test_repeated_activation_fails_instead_of_silent_suppression(self):
        c=self.controller()
        for _ in range(5):
            d=c.propose(self.frame(c));c.commit(d,d.token)
        self.assertIs(c.state,ControllerState.IDLE)
        with self.assertRaisesRegex(ValueError,'activation budget'):c.propose(self.frame(c))

    def test_runtime_route_change_rejected(self):
        self.prediction.update(mode='generate');c=self.controller();d=c.propose(self.frame(c));c.commit(d,d.token)
        self.prediction.update(route=2)
        with self.assertRaisesRegex(ValueError,'route changed'):c.propose(self.frame(c))

    def test_utf8_value_preserved_across_byte_tokens(self):
        raw='里加🧠'.encode();x=replace(self.inputs,payload_sha256=hashlib.sha256(raw).hexdigest())
        layout=ByteLayout(x,raw,((0,len(raw)),));snapshot=replace(self.snapshot,facts=(FactPayload('source',raw),))
        self.prediction.update(fact_bytes=(0,len(raw)),value_bytes=(0,len(raw)))
        c=self.controller(inputs=x,layout=layout,snapshot=snapshot)
        for _ in range(len(raw)):
            d=c.propose(self.frame(c));c.commit(d,d.token)
        self.assertEqual(b''.join(self.codec.decode_pieces(c.generated_tokens)),raw)
        d=c.propose(self.frame(c,1));self.assertIs(d.kind,ActionKind.END)

    def test_unsafe_value_rejected_before_emission(self):
        self.codec.forbidden=lambda token:token in (0,1,ord('R')+2)
        c=self.controller()
        with self.assertRaises(ValueError):c.propose(self.frame(c))
        self.assertEqual(c.generated_tokens,());self.assertIs(c.state,ControllerState.FAILED)

    def test_normal_generation_budget_exhaustion_is_explicit(self):
        self.prediction.update(route=0,mode='generate');c=self.controller(max_new_tokens=1)
        d=c.propose(self.frame(c));c.commit(d,d.token)
        with self.assertRaisesRegex(ValueError,'budget exhausted'):c.propose(self.frame(c))

    def test_empty_source_has_explicit_no_token_uncertainty(self):
        x=replace(self.inputs,source=torch.empty(0,8),allowed=torch.empty(0,dtype=torch.bool),payload_sha256=hashlib.sha256(b'').hexdigest())
        layout=ByteLayout(x,b'',());snapshot=replace(self.snapshot,facts=())
        self.prediction.update(route=2,mode='generate')
        c=self.controller(inputs=x,layout=layout,snapshot=snapshot);d=c.propose(self.frame(c))
        self.assertIsNone(d.token);self.assertIs(c.state,ControllerState.WAIT_UNCERTAINTY)

    def test_other_request_action_cannot_be_committed(self):
        one=self.controller();two=self.controller(reply=replace(self.reply,request_id='other-request'))
        first=one.propose(self.frame(one));second=two.propose(self.frame(two))
        self.assertEqual(first.token,second.token)
        with self.assertRaises(ValueError):one.commit(second,second.token)
        self.assertEqual(one.generated_tokens,())

    def test_start_during_partial_utf8_prefix_fails_closed(self):
        c=self.controller(initial_prefix=(0,2+0xe9))
        with self.assertRaises(UnicodeDecodeError):c.propose(self.frame(c))
        self.assertEqual(c.generated_tokens,());self.assertIs(c.state,ControllerState.FAILED)


if __name__=='__main__':unittest.main()
