"""DG-020 research controller. No teacher inputs or uncertainty text generator.

Provenance tags prevent accidental interface mixing; they do not authenticate
arbitrary Python callers. The native adapter must independently certify frames.
"""
from dataclasses import dataclass
from enum import Enum

import torch

from append_value_transport import compile_append_value
from fine_span_reader import FineSpanReader,ByteLayout
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures,PrefixFeature
from memory_token_read import versions
from native_memory_encoder import digest
from value_transport import (ActivatedFact,PayloadSnapshot,ReplyBinding,TransportLimits,Origin,require)


class FrameOrigin(str,Enum):
    NATIVE_FRESH='native_fresh'
    SYNTHETIC_TEST='synthetic_test'


class ActionKind(str,Enum):
    BASE='base'
    START='start_value'
    CONTINUE='continue_value'
    END='end_value'
    NEEDS_UNCERTAINTY='needs_uncertainty'


class ControllerState(str,Enum):
    IDLE='idle'
    ACTIVE='active'
    WAIT_UNCERTAINTY='wait_uncertainty'
    DONE='done'
    FAILED='failed'


@dataclass(frozen=True)
class LiveFrame:
    prefix_ids: tuple
    hidden: torch.Tensor
    logits: torch.Tensor
    binding: str
    origin: FrameOrigin


@dataclass(frozen=True)
class Decision:
    kind: ActionKind
    token: int | None
    prefix_ids: tuple
    route: int
    origin: FrameOrigin


class AutonomousValueController:
    """Single-reply state machine; commits acknowledge real emission, not targets."""
    def __init__(self,model,inputs,layout,snapshot,reply,codec,initial_prefix,*,max_new_tokens=64,max_value_reads=1,
                 limits=TransportLimits(),allow_synthetic=False):
        require(type(model) is FineSpanReader and type(inputs) is SpanFeatures and type(layout) is ByteLayout and
                layout.inputs is inputs and type(snapshot) is PayloadSnapshot and type(reply) is ReplyBinding,
                'typed neural runtime inputs required')
        require(not model.training and type(allow_synthetic) is bool,'eval mode and explicit fixture policy required')
        require(type(max_new_tokens) is int and max_new_tokens>0 and type(max_value_reads) is int and max_value_reads>0 and
                type(limits) is TransportLimits,'positive reply budgets required')
        require(type(initial_prefix) is tuple and bool(initial_prefix) and initial_prefix[0]==codec.bos_id and
                all(type(t) is int and 0<=t<codec.vocab for t in initial_prefix) and len(initial_prefix)<limits.max_total_tokens,
                'valid actual initial prefix required')
        require(snapshot.scope==reply.scope and inputs.context_sha256==reply.context_sha256 and
                inputs.binding==model.binding==digest(vars(snapshot.binding)) and
                snapshot.memory_model_sha256==tensor_digest(model.state_dict()),'model/request/scope binding mismatch')
        require((len(snapshot.facts)==1 and snapshot.facts[0].data==layout.payload) or
                (not snapshot.facts and layout.payload==b'' and len(inputs.source)==0),'source snapshot mismatch')
        require(codec.backbone_sha256==snapshot.binding.backbone_sha256 and codec.tokenizer_sha256==snapshot.binding.tokenizer_sha256,
                'codec binding mismatch')
        codec.verify_identity();layout.endpoints()
        self._model=model;self._inputs=inputs;self._layout=layout;self._snapshot=snapshot;self._reply=reply;self._codec=codec
        self._initial=initial_prefix;self._prefix=initial_prefix;self._limit=limits;self._max_new=max_new_tokens;self._max_reads=max_value_reads
        self._allow_synthetic=allow_synthetic;self._state=ControllerState.IDLE;self._route=None;self._cursor=None;self._reads=0
        self._pending=None;self._frame=None;self._frame_versions=None;self._frame_origin=None
        self._bound_versions=versions(model,(inputs.query,inputs.source,inputs.allowed))
        self._bound_key=model.binding
        self._codec_binding=(codec.backbone_sha256,codec.tokenizer_sha256,codec.bos_id,codec.vocab)
        self._eos=codec.tokenizer.eos()
        require(type(self._eos) is int and 0<=self._eos<codec.vocab,'explicit valid EOS required')

    @property
    def state(self):return self._state

    @property
    def prefix(self):return self._prefix

    @property
    def generated_tokens(self):return self._prefix[len(self._initial):]

    def _validate_binding(self):
        require(not self._model.training and self._bound_versions==versions(self._model,(self._inputs.query,self._inputs.source,self._inputs.allowed)),
                'model or source/query tensors changed during reply')
        require(self._model.binding==self._inputs.binding==self._bound_key and
                self._codec_binding==(self._codec.backbone_sha256,self._codec.tokenizer_sha256,self._codec.bos_id,self._codec.vocab),
                'runtime codec/feature binding changed')
        self._codec.verify_identity()

    def _validate_frame(self,frame):
        require(type(frame) is LiveFrame and type(frame.origin) is FrameOrigin,'live typed frame required; teacher input rejected')
        require(frame.origin is FrameOrigin.NATIVE_FRESH or self._allow_synthetic,'synthetic frame requires explicit opt-in')
        require(self._frame_origin is None or self._frame_origin is frame.origin,'frame origin changed within reply')
        require(frame.prefix_ids==self._prefix and type(frame.prefix_ids) is tuple and all(type(t) is int for t in frame.prefix_ids) and
                frame.binding==self._model.binding,'not the actual reply prefix')
        for x,shape in ((frame.hidden,(self._model.hidden_size,)),(frame.logits,(self._codec.vocab,))):
            require(type(x) is torch.Tensor and x.shape==shape and x.dtype==torch.float32 and not x.requires_grad and
                    x.device==next(self._model.parameters()).device and bool(torch.isfinite(x).all()),'invalid frozen C frame')

    def propose(self,frame):
        try:
            require(self._state in (ControllerState.IDLE,ControllerState.ACTIVE) and self._pending is None,'controller is terminal or has uncommitted action')
            self._validate_binding();self._validate_frame(frame)
            remaining=self._max_new-len(self.generated_tokens)
            require(remaining>0 and len(self._prefix)<self._limit.max_total_tokens,'reply token/context budget exhausted')
            base=int(frame.logits.argmax());kind=ActionKind.BASE;token=base
            if self._cursor is not None:
                if self._cursor.done:
                    self._cursor.validate(self._snapshot,self._reply,self._prefix,allow_oracle=False);kind=ActionKind.END
                else:
                    kind=ActionKind.CONTINUE;token=self._cursor.select(base,self._snapshot,self._reply,self._prefix)
            elif self._route!=0:
                with torch.no_grad():
                    prefix=PrefixFeature(self._prefix,frame.hidden,self._model.binding)
                    output=self._model(self._inputs,self._layout,prefix,self._prefix);prediction=self._model.predict(output)
                    route=prediction['route']
                    require(self._route is None or route==self._route,'reply-local route changed')
                    self._route=route
                    if route==2:
                        kind=ActionKind.NEEDS_UNCERTAINTY;token=None
                    elif route==1 and prediction['mode']=='start':
                        require(self._reads<self._max_reads,'value activation budget exhausted')
                        require(remaining>=2 and self._limit.max_total_tokens-len(self._prefix)>=2,'no room for value and END handoff')
                        handle=self._model.handle(output,self._snapshot,self._reply)
                        require(type(handle) is ActivatedFact and handle.origin is Origin.NEURAL_PREDICTION and handle.span is not None and
                                (handle.span.start_byte,handle.span.end_byte)==prediction['value_bytes'],'START requires its own predicted non-NULL handle')
                        effective=TransportLimits(self._limit.max_value_bytes,min(self._limit.max_value_tokens,remaining-1),self._limit.max_total_tokens-1)
                        # Decode ACTUAL tokens; do not rebuild a canonical prompt or read a reference reply.
                        actual_text=b''.join(self._codec.decode_pieces(self._prefix[1:])).decode('utf-8',errors='strict')
                        cursor=compile_append_value(handle,self._snapshot,self._reply,self._prefix,actual_text,self._codec,limits=effective)
                        require(cursor is not None,'unexpected NULL START')
                        self._cursor=cursor;kind=ActionKind.START;token=cursor.select(base,self._snapshot,self._reply,self._prefix)
            require(self._route in (0,1,2),'missing route')
            decision=Decision(kind,token,self._prefix,self._route,frame.origin)
            self._frame_origin=frame.origin
            if kind is ActionKind.NEEDS_UNCERTAINTY:
                # No output token, no canned reply and no path to ordinary commit.
                self._state=ControllerState.WAIT_UNCERTAINTY
                return decision
            self._pending=decision;self._frame=frame;self._frame_versions=(frame.hidden._version,frame.logits._version)
            return decision
        except Exception:
            self._state=ControllerState.FAILED;self._pending=None
            raise

    def commit(self,decision,emitted_token):
        try:
            require(self._state in (ControllerState.IDLE,ControllerState.ACTIVE) and self._pending is not None and
                    decision is self._pending,'unknown, copied, stale or non-emitting decision')
            self._validate_binding()
            require(self._frame_versions==(self._frame.hidden._version,self._frame.logits._version),'C frame changed before acknowledgement')
            require(type(emitted_token) is int and emitted_token==decision.token,'acknowledged token differs from proposed token')
            if decision.kind in (ActionKind.START,ActionKind.CONTINUE):
                self._cursor=self._cursor.commit(emitted_token,self._snapshot,self._reply,self._prefix)
                if decision.kind is ActionKind.START:self._reads+=1
                self._state=ControllerState.ACTIVE
            elif decision.kind is ActionKind.END:
                require(self._cursor is not None and self._cursor.done,'premature END')
                self._cursor=None;self._state=ControllerState.IDLE
            else:self._state=ControllerState.IDLE
            self._prefix+=(emitted_token,)
            if emitted_token==self._eos:self._state=ControllerState.DONE
            self._pending=None;self._frame=None;self._frame_versions=None
            return self._prefix
        except Exception:
            self._state=ControllerState.FAILED;self._pending=None
            raise
