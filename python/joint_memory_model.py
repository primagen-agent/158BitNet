"""CG-003 route ablation. Preserve DG-011 implementation/certificates unchanged."""
from dataclasses import dataclass

import torch

from continuous_memory import continuous_logits
from memory_role_separated import RoleSeparatedMemory
from memory_token_read import TokenReadPrototype, versions
from token_memory_composition import ComposedTokenMemory, ComposedState


@dataclass(frozen=True)
class JointState:
    core: ComposedState
    policy: str

    def validate(self, model):
        if self.policy != model.route_policy: raise ValueError('route policy changed during reply')
        self.core.validate(model)


class JointTokenMemory(ComposedTokenMemory):
    def __init__(self, reader, roles, route_policy):
        if route_policy not in ('joint_aux', 'joint_product'): raise ValueError('unregistered route policy')
        super().__init__(reader, roles)
        self._route_policy = route_policy
        for name, p in self.roles.named_parameters():
            if name.startswith(('state_encoder.', 'state_head.')): p.requires_grad_(False)

    @property
    def route_policy(self):
        return self._route_policy

    def prefill(self, inputs):
        core = super().prefill(inputs)
        if self.route_policy == 'joint_aux':
            logits = core.pointer.state_logits.log_softmax(-1)
            value = ComposedState(self, core.pointer, core.prepared, core.factor_logits, logits, ())
            core = ComposedState(self, core.pointer, core.prepared, core.factor_logits, logits, versions(self, value.tensors()))
        return JointState(core, self.route_policy)

    def branches(self, hidden, base, head, scale, state):
        if type(state) is not JointState: raise ValueError('bound joint decision required')
        state.validate(self)
        return super().branches(hidden, base, head, scale, state.core)

    def read(self, hidden, base, head, scale, state, *, enabled=True):
        if type(state) is not JointState or type(enabled) is not bool: raise ValueError('explicit bound route required')
        state.validate(self)
        self.reader._validate_step(hidden, base, state.core.pointer)
        route = int(state.core.state_logits.argmax())
        if not enabled or route == 0: return base
        if route == 2: return continuous_logits(base, self.roles.uncertainty(hidden), head, scale)[0]
        return self.branches(hidden, base, head, scale, state).supported


def create_joint_model(binding, blocked_ids, policy, *, seed=1013, hidden=1024, vocab=73448, layers=24, heads=8):
    # Isolate initialization from sampling/application RNG and reproduce both arms.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        roles = RoleSeparatedMemory(hidden, layers, heads)
        reader = TokenReadPrototype(hidden, vocab, binding, blocked_ids)
        return JointTokenMemory(reader, roles, policy)
