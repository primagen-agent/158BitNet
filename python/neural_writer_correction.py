"""Fit-only capped source-occupancy and conditional structure corrections.

The occupancy objective compares the best legal gold-occupancy path with the
best path having any different occupied word. Its boundary inputs are detached.
The structure objective fixes gold occupancy and compares exact gold tags with
every other legal tag path. Common membership is omitted, rather than relying
on cancellation of large floating-point path scores. Neither objective changes
serving or lets a student's prediction choose teacher eligibility.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from neural_writer_anchor import _ordered_scan, strongest_wrong_path


@dataclass(frozen=True)
class WriterCorrectionTeacher:
    target: torch.Tensor
    occupancy: torch.Tensor
    source_eligible: bool
    source_preservation_eligible: bool
    structure_eligible: bool
    source_required_gap: torch.Tensor
    structure_required_gap: torch.Tensor
    parent_source_gap: torch.Tensor
    parent_structure_gap: torch.Tensor
    frozen_action_gap: torch.Tensor
    frozen_action_unreachable: bool


def _aligned(base_logits, boundaries, membership, target):
    n = base_logits.shape[0] if base_logits.ndim else 0
    if (base_logits.ndim != 2 or base_logits.shape != (n, 5) or not n
            or boundaries.shape != (n, 2) or membership.shape != (n,)
            or target.shape != (n,) or not base_logits.is_floating_point()
            or boundaries.dtype != base_logits.dtype or membership.dtype != base_logits.dtype
            or target.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or boundaries.device != base_logits.device or membership.device != base_logits.device
            or target.device != base_logits.device):
        raise ValueError("writer corrections require aligned five-tag, boundary, membership and gold tensors")


def _weight(weight):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("writer correction weight must be finite and nonnegative")


def _frozen_grammar(model):
    # The current trainer isolates residual heads. Silently training the fixed
    # action/grammar parameters would break both gradient-routing guarantees.
    if any(getattr(model, name).requires_grad
           for name in ("plan_start", "plan_transition", "plan_end")):
        raise ValueError("writer corrections require frozen source grammar parameters")


def _logits(base_logits, boundaries, membership=None):
    logits = base_logits.detach() + F.pad(boundaries[:, :1].expand(-1, 3), (1, 1))
    return logits if membership is None else logits + F.pad(membership[:, None].expand(-1, 4), (1, 0))


def _restrict_occupancy(logits, occupancy):
    active = torch.arange(5, device=logits.device) > 0
    return logits.masked_fill(active[None] != occupancy[:, None], -torch.inf)


def _best_score(model, logits, ends):
    start, _ = model.plan_potentials()
    terminal = model.plan_terminal_potentials(ends)
    first = start + logits[0]
    if len(logits) == 1:
        return (first + terminal).amax()
    edges = model.plan_edge_potentials(ends) + logits[1:, None, :]
    product = _ordered_scan(edges)[-1]
    return (first[:, None] + product + terminal[None]).amax()


def _gold_score(model, logits, ends, target):
    start, _ = model.plan_potentials()
    terminal = model.plan_terminal_potentials(ends)
    score = logits.gather(1, target[:, None]).sum() + start[target[0]] + terminal[target[-1]]
    if len(logits) > 1:
        edges = model.plan_edge_potentials(ends)
        score = score + edges[torch.arange(len(logits) - 1, device=logits.device), target[:-1], target[1:]].sum()
    return score


def strongest_wrong_occupancy_path(model, logits, ends, occupancy):
    """Max over every legal path with at least one different occupied word."""
    if (logits.ndim != 2 or logits.shape[1] != 5 or not len(logits)
            or ends.shape != (len(logits),) or occupancy.shape != (len(logits),)
            or occupancy.dtype != torch.bool or ends.dtype != logits.dtype
            or ends.device != logits.device or occupancy.device != logits.device):
        raise ValueError("occupancy rivals require aligned five-tag logits, ends and Boolean occupancy")
    start, _ = model.plan_potentials()
    terminal = model.plan_terminal_potentials(ends)
    first = start + logits[0]
    same = (torch.arange(5, device=logits.device) > 0)[None] == occupancy[:, None]
    if len(logits) == 1:
        return (first + terminal).masked_fill(same[0], -torch.inf).amax()
    edges = model.plan_edge_potentials(ends) + logits[1:, None, :]
    prefixes = _ordered_scan(edges)
    suffixes = _ordered_scan(edges, suffix=True)
    forward = torch.cat((first[None], (first[None, :, None] + prefixes).amax(-2)), 0)
    backward = torch.cat(((suffixes + terminal[None, None, :]).amax(-1), terminal[None]), 0)
    return (forward + backward).masked_fill(same, -torch.inf).amax()


def _source_gap(model, logits, ends, occupancy):
    return (_best_score(model, _restrict_occupancy(logits, occupancy), ends)
            - strongest_wrong_occupancy_path(model, logits, ends, occupancy))


def _structure_gap(model, logits, ends, target, occupancy):
    constrained = _restrict_occupancy(logits, occupancy)
    return (_gold_score(model, constrained, ends, target)
            - strongest_wrong_path(model, constrained, ends, target))


@torch.no_grad()
def build_writer_correction_teacher(model, base_logits, parent_boundaries,
                                    parent_membership, target, *, split, margin=1.0):
    """Capture immutable parent component masks and attainable capped targets.

    Correct full source plans stay with the trainer's existing anchors. Wrong
    occupancy receives margin ``margin``; correct occupancy on a wrong full plan
    preserves min(parent gap, margin). Already-correct conditional structure
    preserves the same capped parent gap; incorrect structure
    learns that margin, capped by any frozen action-only competitor. An action
    path beating gold with identical B/I positions is unreachable by these heads.
    """
    if split != "fit":
        raise ValueError("writer correction teachers are fit-only")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("writer correction margin must be finite and positive")
    _aligned(base_logits, parent_boundaries, parent_membership, target)
    if not all(bool(torch.isfinite(t).all()) for t in (base_logits, parent_boundaries, parent_membership)):
        raise ValueError("writer correction parent tensors must be finite")
    tags = target.detach().cpu().tolist()
    if any(tag < 0 or tag > 4 or (tag == 4 and (i == 0 or tags[i - 1] == 0))
           for i, tag in enumerate(tags)):
        raise ValueError("writer correction gold must obey the current source action/inside grammar")
    target = target.detach().clone().long()
    occupancy = (target > 0).detach()
    # The native-equivalent Viterbi helper deliberately returns CPU labels.
    # Compare cold-path eligibility there while keeping loss tensors on-device.
    cpu_target = target.cpu()
    cpu_occupancy = cpu_target > 0
    full = _logits(base_logits, parent_boundaries, parent_membership)
    ends = parent_boundaries[:, 1]
    prediction = model.plan_labels(full, ends)
    full_correct = bool(torch.equal(prediction, cpu_target))
    source_eligible = not bool(torch.equal(prediction > 0, cpu_occupancy))
    source_preservation_eligible = not full_correct and not source_eligible
    source_gap = _source_gap(model, full, ends, occupancy).detach()
    structural = _logits(base_logits, parent_boundaries)
    constrained = _restrict_occupancy(structural, occupancy)
    conditional = model.plan_labels(constrained, ends)
    structure_correct = bool(torch.equal(conditional, cpu_target))
    zero = base_logits.new_zeros(())
    structure_gap, action_gap, action_unreachable = zero, zero, False
    # Isolated single-word spans have no B/I alternatives. Omitting their
    # redundant structure loss also handles nulls without inf-inf or NaN grads.
    has_bi_alternative = any(a > 0 and b > 0 for a, b in zip(tags, tags[1:]))
    if any(tags):
        structure_gap = _structure_gap(model, structural, ends, target, occupancy).detach()
        states = torch.arange(5, device=target.device)[None]
        begin = ((target > 0) & (target < 4))[:, None]
        action_allowed = torch.where(begin, (states > 0) & (states < 4), states == target[:, None])
        action_logits = structural.masked_fill(~action_allowed, -torch.inf)
        action_gap = (_gold_score(model, action_logits, ends, target)
                      - strongest_wrong_path(model, action_logits, ends, target)).detach()
        action_unreachable = not bool(torch.equal(model.plan_labels(action_logits, ends), cpu_target))
    structure_eligible = not full_correct and has_bi_alternative and not action_unreachable
    cap = base_logits.new_tensor(margin)
    source_required = cap if source_eligible else torch.minimum(source_gap, cap)
    structure_required = (torch.minimum(structure_gap, cap) if structure_correct
                          else torch.minimum(action_gap, cap)) if any(tags) else zero
    return WriterCorrectionTeacher(target=target, occupancy=occupancy,
                                   source_eligible=source_eligible,
                                   source_preservation_eligible=source_preservation_eligible,
                                   structure_eligible=structure_eligible,
                                   source_required_gap=source_required.detach().clone(),
                                   structure_required_gap=structure_required.detach().clone(),
                                   parent_source_gap=source_gap.detach().clone(),
                                   parent_structure_gap=structure_gap.detach().clone(),
                                   frozen_action_gap=action_gap.detach().clone(),
                                   frozen_action_unreachable=action_unreachable)


def source_occupancy_margin_loss(model, base_logits, boundaries, membership, teacher, *, weight=1.0):
    """Squared positive margin violation; gradients enter only membership."""
    _weight(weight)
    _aligned(base_logits, boundaries, membership, teacher.target)
    _frozen_grammar(model)
    if not (teacher.source_eligible or teacher.source_preservation_eligible):
        return membership.sum() * 0
    logits = _logits(base_logits, boundaries.detach(), membership)
    gap = _source_gap(model, logits, boundaries[:, 1].detach(), teacher.occupancy)
    return weight * F.relu(teacher.source_required_gap - gap).square()


def conditional_structure_margin_loss(model, base_logits, boundaries, teacher, *, weight=1.0):
    """Squared conditional B/I/action margin violation; membership is absent."""
    _weight(weight)
    _aligned(base_logits, boundaries, boundaries[:, 0].detach(), teacher.target)
    _frozen_grammar(model)
    if not teacher.structure_eligible:
        return boundaries.sum() * 0
    gap = _structure_gap(model, _logits(base_logits, boundaries), boundaries[:, 1],
                         teacher.target, teacher.occupancy)
    return weight * F.relu(teacher.structure_required_gap - gap).square()


def writer_correction_losses(model, base_logits, boundaries, membership, teacher, *,
                             source_weight=1.0, structure_weight=1.0):
    return (source_occupancy_margin_loss(model, base_logits, boundaries, membership, teacher, weight=source_weight),
            conditional_structure_margin_loss(model, base_logits, boundaries, teacher, weight=structure_weight))
