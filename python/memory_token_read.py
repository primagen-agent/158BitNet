"""DG-010 untrained, single-supplied-event interface. Not a serving model.

Two learned matching slots are hypotheses, not trained subject/relation labels.
Token IDs are payload only; they never enter address or state computation.
The caller must handle the normal/insufficient routes with its own generator.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TokenReadInput:
    query: torch.Tensor
    source: torch.Tensor
    token_ids: torch.Tensor
    copy_allowed: torch.Tensor
    binding: str


def versions(module, tensors):
    return tuple((id(t), t._version) for t in (*module.parameters(), *tensors))


@dataclass(frozen=True)
class TokenReadState:
    owner: object
    inputs: TokenReadInput
    slots: torch.Tensor
    memory: torch.Tensor
    match_attention: torch.Tensor
    state_logits: torch.Tensor
    version: tuple

    def tensors(self):
        return (self.inputs.query, self.inputs.source, self.inputs.token_ids, self.inputs.copy_allowed,
                self.slots, self.memory, self.match_attention, self.state_logits)

    def validate(self, model):
        if self.owner is not model or self.version != versions(model, self.tensors()):
            raise ValueError('reply-local model or memory changed')


@dataclass(frozen=True)
class TokenReadOutput:
    logits: torch.Tensor
    route: int
    position_probabilities: torch.Tensor
    copy_mass: torch.Tensor


def token_mixture(base, position_logits, gate_logits, token_ids):
    """Null at position zero; duplicate payload IDs SUM mass, never max it."""
    if (base.ndim != 2 or position_logits.shape != (len(base), len(token_ids) + 1)
            or gate_logits.shape != (len(base), 1) or token_ids.ndim != 1 or token_ids.dtype != torch.long
            or not torch.isfinite(base).all() or not torch.isfinite(gate_logits).all()
            or not torch.isfinite(position_logits[:, 0]).all() or torch.isnan(position_logits).any()
            or torch.isposinf(position_logits).any()
            or any(t.device != base.device for t in (position_logits, gate_logits, token_ids))
            or (len(token_ids) and (int(token_ids.min()) < 0 or int(token_ids.max()) >= base.shape[1]))):
        raise ValueError('invalid token-mixture domain')
    logpos = position_logits.double().log_softmax(-1)
    positions = logpos.exp()
    copied = torch.zeros_like(base, dtype=torch.float64).scatter_add(
        1, token_ids[None].expand(len(base), -1), positions[:, 1:])
    logcopy = copied.clamp_min(torch.finfo(torch.float64).tiny).log().masked_fill(copied == 0, -torch.inf)
    loggate = F.logsigmoid(gate_logits.double())
    # Base mass = gate-off OR gate-on-and-null. Stable even at extreme gates.
    logbase = torch.logaddexp(F.logsigmoid(-gate_logits.double()), loggate + logpos[:, :1])
    mixture = torch.logaddexp(base.double().log_softmax(-1) + logbase, logcopy + loggate)
    mass = gate_logits.double().sigmoid() * positions[:, 1:].sum(-1, keepdim=True)
    return mixture.to(base.dtype), positions.to(base.dtype), mass.to(base.dtype)


class TokenReadPrototype(nn.Module):
    def __init__(self, hidden, vocab, binding, blocked_ids=()):
        super().__init__()
        if type(hidden) is not int or hidden <= 0 or type(vocab) is not int or vocab < 3 or not binding:
            raise ValueError('explicit model geometry and identity required')
        if any(type(i) is not int or not 0 <= i < vocab for i in blocked_ids):
            raise ValueError('invalid blocked token ID')
        self.hidden, self.vocab, self.binding = hidden, vocab, binding
        self.blocked_ids = tuple(blocked_ids)
        self.query_encoder = nn.Linear(2 * hidden, hidden, bias=False)
        self.source_encoder = nn.Linear(2 * hidden, hidden, bias=False)
        self.slot_queries = nn.Parameter(torch.randn(2, hidden) / math.sqrt(hidden))
        self.match_query = nn.Linear(hidden, hidden, bias=False)
        self.match_key = nn.Linear(hidden, hidden, bias=False)
        self.null_key = nn.Parameter(torch.zeros(1, hidden))
        self.state_encoder = nn.Linear(8 * hidden + 2, hidden)
        self.state_head = nn.Linear(hidden, 3)
        self.position_query = nn.Linear(3 * hidden, hidden, bias=False)
        self.position_key = nn.Linear(hidden, hidden, bias=False)
        self.copy_gate = nn.Linear(2 * hidden, 1)
        nn.init.constant_(self.copy_gate.bias, -4.)

    def validate_input(self, inputs):
        if type(inputs) is not TokenReadInput or inputs.binding != self.binding:
            raise ValueError('label-free input with bound encoder/tokenizer required')
        tensors = (inputs.query, inputs.source, inputs.token_ids, inputs.copy_allowed)
        device = self.slot_queries.device
        if any(not isinstance(t, torch.Tensor) or t.device != device or t.requires_grad for t in tensors):
            raise ValueError('frozen same-device C inputs required')
        for features in (inputs.query, inputs.source):
            if features.dtype != torch.float32 or features.ndim != 2 or features.shape[1] != 2 * self.hidden or not torch.isfinite(features).all():
                raise ValueError('finite FP32 token features required')
        n = len(inputs.source)
        if not len(inputs.query) or inputs.token_ids.shape != (n,) or inputs.token_ids.dtype != torch.long or inputs.copy_allowed.shape != (n,) or inputs.copy_allowed.dtype != torch.bool:
            raise ValueError('token/feature alignment mismatch')
        if n and (int(inputs.token_ids.min()) < 0 or int(inputs.token_ids.max()) >= self.vocab):
            raise ValueError('token outside bound vocabulary')
        if any(bool(((inputs.token_ids == i) & inputs.copy_allowed).any()) for i in self.blocked_ids):
            raise ValueError('control token cannot be copied')

    def prefill(self, inputs):
        self.validate_input(inputs)
        query = F.layer_norm(self.query_encoder(inputs.query), (self.hidden,))
        memory = F.layer_norm(self.source_encoder(inputs.source), (self.hidden,))
        pooling = (self.slot_queries @ query.T / math.sqrt(self.hidden)).softmax(-1)
        slots = pooling @ query
        keys = torch.cat((self.null_key, self.match_key(memory)))
        attention = (self.match_query(slots) @ keys.T / math.sqrt(self.hidden)).softmax(-1)
        evidence = attention[:, 1:] @ memory
        joined = torch.cat((slots, evidence, slots * evidence, (slots - evidence).abs()), -1)
        logits = self.state_head(self.state_encoder(torch.cat((joined.flatten(), attention[:, 0]))).tanh())
        if not len(memory):
            logits = logits.masked_fill(torch.tensor([False, True, False], device=logits.device), -torch.inf)
        state = TokenReadState(self, inputs, slots, memory, attention, logits, ())
        return TokenReadState(self, inputs, slots, memory, attention, logits, versions(self, state.tensors()))

    def proposal(self, hidden, base, state):
        """Label-free differentiable specialist, independent of target route.

        Training may supervise this proposal AFTER forward, but serving uses read.
        """
        self._validate_step(hidden, base, state)
        if not len(state.memory) or not bool(state.inputs.copy_allowed.any()):
            return self._bypass(base, state)
        h = F.layer_norm(hidden, (self.hidden,))
        slots = state.slots.flatten()[None].expand(len(hidden), -1)
        q = self.position_query(torch.cat((h, slots), -1))
        keys = torch.cat((self.null_key, self.position_key(state.memory)))
        logits = q @ keys.T / math.sqrt(self.hidden)
        invalid = torch.cat((torch.zeros(1, dtype=torch.bool, device=base.device), ~state.inputs.copy_allowed))
        logits = logits.masked_fill(invalid[None], -torch.inf)
        positions = logits.softmax(-1)
        read = positions[:, 1:] @ state.memory
        gate = self.copy_gate(torch.cat((h, read), -1))
        output, positions, mass = token_mixture(base, logits, gate, state.inputs.token_ids)
        return TokenReadOutput(output, int(state.state_logits.argmax()), positions, mass)

    def _validate_step(self, hidden, base, state):
        if type(state) is not TokenReadState: raise ValueError('initial prefill state required')
        state.validate(self)
        self.validate_input(state.inputs)
        if (hidden.ndim != 2 or not len(hidden) or hidden.shape[1] != self.hidden
                or base.shape != (len(hidden), self.vocab)
                or any(t.dtype != torch.float32 or t.device != self.slot_queries.device or t.requires_grad or not torch.isfinite(t).all() for t in (hidden, base))):
            raise ValueError('frozen finite C hidden and logits required')

    def _bypass(self, base, state):
        positions = base.new_zeros((len(base), len(state.memory) + 1)); positions[:, 0] = 1
        return TokenReadOutput(base, int(state.state_logits.argmax()), positions, base.new_zeros((len(base), 1)))

    def read(self, hidden, base, state, *, enabled=True):
        if type(enabled) is not bool: raise ValueError('explicit enable flag required')
        self._validate_step(hidden, base, state)
        if not enabled or int(state.state_logits.argmax()) != 1:
            return self._bypass(base, state)
        return self.proposal(hidden, base, state)
