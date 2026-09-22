"""Post-forward training-only labels. Never accepted by neural forward/read."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from availability_supervision import ReplyTargets
from token_memory_composition import ComposedOutputs


@dataclass(frozen=True)
class TokenSupervision:
    reply: ReplyTargets
    factors: tuple
    positions: tuple

    def __post_init__(self):
        if type(self.reply) is not ReplyTargets or type(self.factors) is not tuple or len(self.factors) != 2:
            raise ValueError('separate immutable supervision required')
        if any(v is not None and type(v) is not bool for v in self.factors): raise ValueError('factor target must be boolean or unknown')
        if type(self.positions) is not tuple or len(self.positions) != len(self.reply.completion_token_ids):
            raise ValueError('retain all reply positions including uncopyable tokens')
        for positions in self.positions:
            if positions is not None and (type(positions) is not tuple or not positions or
                    len(set(positions)) != len(positions) or any(type(i) is not int or i < 0 for i in positions)
                    or (0 in positions and positions != (0,))):
                raise ValueError('immutable marginal position target required')
        if self.reply.state_index != 1 and any(p is not None for p in self.positions):
            raise ValueError('non-support uses copy-off loss, never positive position labels')


def make_position_targets(completion, state_index, value_offsets, source_ids, allowed, source_value_positions):
    """Gold spans are post-forward supervision, never the structural copy mask."""
    if len(source_ids) != len(allowed): raise ValueError('source alignment mismatch')
    if any(type(i) is not int or not 0 <= i < len(completion) for i in value_offsets): raise ValueError('invalid target value span')
    if any(type(i) is not int or not 0 <= i < len(source_ids) for i in source_value_positions): raise ValueError('invalid source value span')
    if state_index != 1: return (None,) * len(completion)
    result = [None] * len(completion)
    for offset in value_offsets:
        eligible = tuple(i + 1 for i in source_value_positions if allowed[i] and source_ids[i] == completion[offset])
        result[offset] = eligible or (0,)  # Uncopyable remains in CE, teaches NULL.
    return tuple(result)


def supervised_token_loss(outputs, targets):
    if type(outputs) is not ComposedOutputs or type(targets) is not TokenSupervision:
        raise ValueError('separate forward outputs and post-forward supervision required')
    reply = targets.reply
    if outputs.base.requires_grad or outputs.base.ndim != 2 or len(outputs.base) != len(reply.completion_token_ids):
        raise ValueError('frozen base and every reply token required')
    if outputs.supported.shape != outputs.base.shape or outputs.uncertainty.shape != outputs.base.shape or outputs.state_logits.shape != (3,) or outputs.factor_logits.shape != (2,):
        raise ValueError('supervision geometry mismatch')
    device = outputs.base.device
    ids = torch.tensor(reply.completion_token_ids, device=device)
    if ids.max() >= outputs.base.shape[1]: raise ValueError('reply token outside vocabulary')
    state = F.cross_entropy(outputs.state_logits[None], torch.tensor([reply.state_index], device=device))
    labelled = [i for i, v in enumerate(targets.factors) if v is not None]
    zero = outputs.base.new_zeros(())
    factor = (F.binary_cross_entropy_with_logits(outputs.factor_logits[labelled],
        torch.tensor([float(targets.factors[i]) for i in labelled], device=device)) if labelled else zero)
    if reply.state_index == 0:
        reference = outputs.base.double().log_softmax(-1)
        generation = sum(F.kl_div(z.double().log_softmax(-1), reference, log_target=True, reduction='none').sum(-1).mean()
                         for z in (outputs.supported, outputs.uncertainty)) / 2
    else:
        generation = F.cross_entropy(outputs.supported if reply.state_index == 1 else outputs.uncertainty, ids)
    terms = []
    for i, choices in enumerate(targets.positions):
        if choices is not None:
            if max(choices) >= outputs.positions.shape[1]: raise ValueError('position outside source')
            terms.append(-outputs.positions[i, list(choices)].double().sum().clamp_min(1e-30).log())
    position = torch.stack(terms).mean() if terms else zero
    copy_off = (-torch.log1p(-outputs.copy_mass.double().clamp(max=1-1e-12)).mean() if reply.state_index != 1 else zero)
    total = reply.sample_weight * (state + factor + generation + position + copy_off)
    if not torch.isfinite(total): raise ValueError('nonfinite supervised loss')
    return {'state': state, 'factor': factor, 'generation': generation, 'position': position,
            'copy_off': copy_off, 'weighted_total': total, 'reply_tokens': len(ids), 'position_targets': len(terms)}
