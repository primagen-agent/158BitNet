"""DG-011 differentiable composition; untrained research interface only."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from continuous_memory import continuous_logits
from memory_token_read import TokenReadPrototype, versions


@dataclass(frozen=True)
class ComposedState:
    owner: object
    pointer: object
    prepared: object
    factor_logits: torch.Tensor
    state_logits: torch.Tensor
    version: tuple

    def tensors(self):
        return (*self.pointer.tensors(), *(self.prepared or ()), self.factor_logits, self.state_logits)

    def validate(self, model):
        self.pointer.validate(model.reader)
        if self.owner is not model or self.version != versions(model, self.tensors()):
            raise ValueError('composed prefill state changed')


@dataclass(frozen=True)
class ComposedOutputs:
    base: torch.Tensor
    supported: torch.Tensor
    uncertainty: torch.Tensor
    positions: torch.Tensor
    copy_mass: torch.Tensor
    factor_logits: torch.Tensor
    state_logits: torch.Tensor


def apply_factor_support(raw_logits, factors):
    """Matched subject AND relation necessary, not sufficient, for support."""
    if raw_logits.shape != (3,) or factors.shape != (2,) or not torch.isfinite(factors).all():
        raise ValueError('factor/state shape mismatch')
    raw = raw_logits.double().log_softmax(-1)
    log_yes = F.logsigmoid(factors.double()).sum()
    # P(not both) = P(not subject) + P(subject)*P(not relation).
    log_no = torch.logaddexp(F.logsigmoid(-factors[0].double()),
                            F.logsigmoid(factors[0].double()) + F.logsigmoid(-factors[1].double()))
    return torch.stack((raw[0], raw[1] + log_yes, torch.logaddexp(raw[2], raw[1] + log_no))).to(raw_logits.dtype)


def mix_content_copy(content, positions, mass, ids):
    """Do not detach content to satisfy the frozen-base pointer interface."""
    if (content.ndim != 2 or positions.shape != (len(content), len(ids) + 1)
            or mass.shape != (len(content), 1) or ids.ndim != 1 or ids.dtype != torch.long
            or any(t.device != content.device for t in (positions, mass, ids))
            or not torch.isfinite(content).all() or not torch.isfinite(positions).all()
            or not torch.isfinite(mass).all() or (mass < 0).any() or (mass > 1).any()
            or (positions < 0).any() or not torch.allclose(positions.sum(-1), torch.ones(len(content), device=content.device), atol=1e-6, rtol=1e-6)):
        raise ValueError('invalid composition domain')
    p = positions[:, 1:].double()
    normalizer = p.sum(-1, keepdim=True)
    if ((normalizer == 0) & (mass > 0)).any(): raise ValueError('copy mass without a source')
    if not len(ids) or not bool(p.any()): return content
    if int(ids.min()) < 0 or int(ids.max()) >= content.shape[1]: raise ValueError('payload outside vocabulary')
    copy = torch.zeros_like(content, dtype=torch.float64).scatter_add(1, ids[None].expand(len(content), -1),
        p / normalizer.clamp_min(torch.finfo(torch.float64).tiny))
    logcopy = copy.clamp_min(torch.finfo(torch.float64).tiny).log().masked_fill(copy == 0, -torch.inf)
    # Endpoint clamp only avoids log(0) gradients after FP32 gate saturation.
    w = mass.double().clamp(torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
    return torch.logaddexp(content.double().log_softmax(-1) + torch.log1p(-w), logcopy + w.log()).to(content.dtype)


class ComposedTokenMemory(nn.Module):
    def __init__(self, reader, roles):
        super().__init__()
        if not isinstance(reader, TokenReadPrototype) or reader.hidden != roles.hidden:
            raise ValueError('matching reader and specialist geometry required')
        self.reader, self.roles = reader, roles
        self.factor_heads = nn.ModuleList(nn.Linear(4 * reader.hidden, 1) for _ in range(2))

    def prefill(self, inputs):
        pointer = self.reader.prefill(inputs)
        evidence = pointer.match_attention[:, 1:] @ pointer.memory
        q = pointer.slots
        features = torch.cat((q, evidence, q * evidence, (q - evidence).abs()), -1)
        factors = torch.cat([head(features[i]) for i, head in enumerate(self.factor_heads)])
        logits = apply_factor_support(pointer.state_logits, factors)
        prepared = self.roles.prepare(inputs.source)
        provisional = ComposedState(self, pointer, prepared, factors, logits, ())
        return ComposedState(self, pointer, prepared, factors, logits, versions(self, provisional.tensors()))

    def branches(self, hidden, base, head, scale, state):
        if type(state) is not ComposedState: raise ValueError('initial composed decision required')
        state.validate(self)
        pointer = self.reader.proposal(hidden, base, state.pointer)
        content, uncertainty = self.roles.branches(hidden, state.prepared)
        content, _ = continuous_logits(base, content, head, scale)
        uncertainty, _ = continuous_logits(base, uncertainty, head, scale)
        supported = mix_content_copy(content, pointer.position_probabilities, pointer.copy_mass, state.pointer.inputs.token_ids)
        return ComposedOutputs(base, supported, uncertainty, pointer.position_probabilities, pointer.copy_mass,
                               state.factor_logits, state.state_logits)

    def read(self, hidden, base, head, scale, state, *, enabled=True):
        if type(enabled) is not bool or type(state) is not ComposedState: raise ValueError('explicit state/enable required')
        state.validate(self)
        self.reader._validate_step(hidden, base, state.pointer)
        route = int(state.state_logits.argmax())
        if not enabled or route == 0: return base
        if route == 2:
            # This path MUST NOT call the content/copy specialists.
            return continuous_logits(base, self.roles.uncertainty(hidden), head, scale)[0]
        return self.branches(hidden, base, head, scale, state).supported
