"""Diagnostic-only DG-021 handoff. No approved inference/training entry point."""
from dataclasses import dataclass

import torch

from autonomous_value_controller import AutonomousValueController,ActionKind,ControllerState,LiveFrame,FrameOrigin
from memory_token_read import versions
from query_only_uncertainty import QueryOnlyUncertainty
from value_transport import require


@dataclass(frozen=True)
class UncertaintyDecision:
    token: int
    prefix_ids: tuple
    origin: FrameOrigin
    branch: str = 'query_only_uncertainty_diagnostic'


class ResearchUncertaintyController:
    def __init__(self,core,branch,head,scale,*,diagnostic_only=False):
        require(diagnostic_only is True,'untrained uncertainty branch is not approved for serving')
        require(type(core) is AutonomousValueController and core.state is ControllerState.IDLE and not core.generated_tokens,
                'fresh core controller required; external handoff decisions are not accepted')
        require(type(branch) is QueryOnlyUncertainty and not branch.training and branch.binding==core._bound_key and
                branch.hidden==core._model.hidden_size,'query-only branch identity mismatch')
        require(type(head) is torch.Tensor and head.shape==(core._codec.vocab,branch.hidden) and not head.requires_grad and
                head.dtype==torch.float32 and head.device==next(branch.parameters()).device and bool(torch.isfinite(head).all()),
                'frozen same-domain output head required')
        self._core=core;self._branch=branch;self._head=head;self._scale=scale;self._key=branch.binding
        self._versions=versions(branch,(head,));self._initial=core.prefix;self._prefix=core.prefix
        self._uncertain=False;self._done=False;self._failed=False;self._pending=None;self._output=None

    @property
    def prefix(self):return self._prefix if self._uncertain else self._core.prefix

    @property
    def generated_tokens(self):return self.prefix[len(self._initial):]

    @property
    def state(self):
        if self._failed:return 'failed'
        if self._done:return 'done'
        return 'uncertainty_active' if self._uncertain else self._core.state.value

    @property
    def pending_output(self):return self._output

    def _validate(self):
        self._core._validate_binding()
        require(not self._branch.training and self._branch.binding==self._key and self._versions==versions(self._branch,(self._head,)),
                'uncertainty weights/head changed during reply')

    def propose(self,frame):
        try:
            require(not self._done and not self._failed and self._pending is None,'terminal or uncommitted diagnostic reply')
            self._validate()
            if not self._uncertain:
                action=self._core.propose(frame)
                if action.kind is not ActionKind.NEEDS_UNCERTAINTY:
                    self._pending=action;return action
                # Only our own live neural decision can activate this branch.
                require(self._core.state is ControllerState.WAIT_UNCERTAINTY and action.token is None,'invalid core handoff')
                self._uncertain=True;self._prefix=self._core.prefix
            require(type(frame) is LiveFrame and type(frame.origin) is FrameOrigin and
                    frame.origin is self._core._frame_origin and frame.binding==self._key and frame.prefix_ids==self._prefix,
                    'wrong actual uncertainty prefix/origin')
            require(len(self.generated_tokens)<self._core._max_new and len(self._prefix)<self._core._limit.max_total_tokens,'uncertainty budget exhausted')
            with torch.no_grad():output=self._branch(frame,self._prefix,self._head,self._scale)
            output.validate(self._branch)
            decision=UncertaintyDecision(int(output.logits.argmax()),self._prefix,frame.origin)
            self._output=output;self._pending=decision;return decision
        except Exception:
            self._failed=True;self._pending=None;self._output=None;raise

    def commit(self,decision,emitted_token):
        try:
            require(not self._done and not self._failed and self._pending is not None and decision is self._pending,
                    'unknown/stale uncertainty decision')
            self._validate()
            require(type(emitted_token) is int and emitted_token==decision.token,'wrong acknowledged token')
            if not self._uncertain:
                result=self._core.commit(decision,emitted_token);self._done=self._core.state is ControllerState.DONE
            else:
                require(type(decision) is UncertaintyDecision and self._output is not None,'missing query-only output')
                self._output.validate(self._branch)
                self._prefix+=(emitted_token,);result=self._prefix;self._done=emitted_token==self._core._eos
            self._pending=None;self._output=None;return result
        except Exception:
            self._failed=True;self._pending=None;self._output=None;raise
