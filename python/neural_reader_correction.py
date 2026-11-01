"""Fit-only matcher correction with independently frozen reader decisions.

Each nonempty error-positive, error-negative or preservation query contributes
its worst finite-margin violation. These three classes use separate valid-query
means; queries without a member of that error class do not dilute its loss.
Correct-pair protection uses the global worst capped violation, so unchanged
queries cannot dilute a newly lost correct margin.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

PRESERVATION_REDUCTION = "global-worst-parent-correct-pair"


def _finite_float(value, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.device.type == "meta" or not torch.isfinite(value).all()):
        raise ValueError(name + " must be a finite floating tensor")


@dataclass(frozen=True)
class ReaderCorrectionTeacher:
    parent_semantic: torch.Tensor
    parent_activation: torch.Tensor
    fixed_log_gate: torch.Tensor
    gold: torch.Tensor
    reachable_pairs: torch.Tensor
    correct_pairs: torch.Tensor
    wrong_positive_pairs: torch.Tensor
    wrong_negative_pairs: torch.Tensor
    semantic_thresholds: torch.Tensor
    signed_targets: torch.Tensor
    required_margins: torch.Tensor
    margin: float


def snapshot_reader_corrections(parent_semantic, parent_activation, gold, *,
                                fixed_log_gate, split, margin=1., evaluation_only=False):
    """Snapshot reachable errors and every correct pair, including partial sets.

    The supplied actual identity/time log gate determines the semantic decision
    threshold; it is never reconstructed by subtracting rounded reader outputs. An
unreachable pair never trains this matcher. Correct parent pairs retain their
capped actual margin; wrong pairs learn the finite requested margin. Teacher
eligibility is fixed before student updates and never depends on predictions.
"""
    if split != "fit" or evaluation_only:
        raise ValueError("reader correction requires fit labels, never evaluation or runtime")
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin <= 0:
        raise ValueError("reader correction margin must be finite and positive")
    _finite_float(parent_semantic, "parent semantic")
    _finite_float(parent_activation, "parent activation")
    _finite_float(fixed_log_gate, "fixed log gate")
    if (parent_semantic.ndim != 2 or parent_activation.shape != parent_semantic.shape
            or parent_activation.device != parent_semantic.device
            or parent_activation.dtype != parent_semantic.dtype
            or fixed_log_gate.shape != parent_semantic.shape
            or fixed_log_gate.device != parent_semantic.device
            or fixed_log_gate.dtype != parent_semantic.dtype
            or not isinstance(gold, torch.Tensor) or gold.shape != parent_semantic.shape
            or gold.device != parent_semantic.device or gold.is_complex()
            or not torch.isfinite(gold).all() or not ((gold == 0) | (gold == 1)).all()):
        raise ValueError("reader correction needs aligned complete binary support labels")
    semantic, activation = parent_semantic.detach().clone(), parent_activation.detach().clone()
    labels = gold.detach().bool().clone()
    log_gate = fixed_log_gate.detach().clone()
    tolerance = 64 * torch.finfo(semantic.dtype).eps
    if (log_gate > tolerance).any():
        raise ValueError("fixed log gate cannot exceed zero")
    log_gate = log_gate.clamp_max(0)
    activation_log_probability = F.logsigmoid(activation)
    if ((activation_log_probability > F.logsigmoid(semantic) + tolerance).any()
            or (activation_log_probability > log_gate + tolerance).any()):
        raise ValueError("parent activation cannot exceed its semantic or fixed gate probability")
    reachable = log_gate > -math.log(2.)
    difference = (log_gate + math.log(2.)).masked_fill(~reachable, 1.)
    thresholds = -torch.log(torch.expm1(difference))
    thresholds = thresholds.masked_fill(~reachable, 0.)
    signed = labels.to(semantic.dtype) * 2 - 1
    correct = reachable & ((activation > 0) == labels)
    required = torch.where(correct, (signed * (semantic - thresholds)).clamp_max(margin),
                           torch.full_like(semantic, margin)).masked_fill(~reachable, 0.)
    if not torch.isfinite(thresholds).all() or not torch.isfinite(required).all():
        raise ValueError("reader correction threshold and margin arithmetic must remain finite")
    return ReaderCorrectionTeacher(
        semantic, activation, log_gate.detach().clone(), labels,
        reachable.detach().clone(), correct.detach().clone(),
        (reachable & ~correct & labels).detach().clone(),
        (reachable & ~correct & ~labels).detach().clone(),
        thresholds.detach().clone(), signed.detach().clone(), required.detach().clone(), float(margin),
    )


def _query_worst_mean(violations, mask):
    valid = mask.any(-1)
    worst = violations.masked_fill(~mask, 0).amax(-1)
    return worst.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def reader_correction_loss(student_semantic, teacher, weight=1.):
    """Correct each error class and preserve every reachable correct pair.

Positive and negative error classes use independent valid-query worst means.
The worst violation over every eligible parent-correct pair supplies protection,
including correct pairs in initially incomplete queries. Adding unchanged queries
cannot weaken that protection. Squared hinges stop
after their finite target margins are satisfied; serving receives no masks.
"""
    if not isinstance(teacher, ReaderCorrectionTeacher):
        raise ValueError("reader correction needs an independently frozen fit teacher")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("reader correction weight must be finite and nonnegative")
    _finite_float(student_semantic, "student semantic")
    if (student_semantic.shape != teacher.parent_semantic.shape
            or student_semantic.device != teacher.parent_semantic.device
            or student_semantic.dtype != teacher.parent_semantic.dtype):
        raise ValueError("student matcher must match the frozen teacher geometry and precision")
    zero = student_semantic.reshape(-1)[:1].sum() * 0
    if student_semantic.numel() and weight:
        signed = teacher.signed_targets * (student_semantic - teacher.semantic_thresholds)
        difference = (teacher.required_margins - signed).masked_fill(~teacher.reachable_pairs, 0.)
        violations = F.relu(difference).square()
        if not torch.isfinite(violations).all():
            raise ValueError("reader correction loss arithmetic must remain finite")
        positive = _query_worst_mean(violations, teacher.wrong_positive_pairs)
        negative = _query_worst_mean(violations, teacher.wrong_negative_pairs)
        preservation = violations.masked_fill(~teacher.correct_pairs, 0).amax()
    else:
        positive = negative = preservation = zero
    loss = weight * (positive + negative + preservation)
    if not torch.isfinite(loss):
        raise ValueError("reader correction weighted loss must remain finite")
    return loss, dict(
        correction_positive=positive.detach(), correction_negative=negative.detach(),
        preservation=preservation.detach(),
        reachable_pairs=teacher.reachable_pairs.sum().detach(),
        unreachable_pairs=(~teacher.reachable_pairs).sum().detach(),
        unreachable_positive_pairs=(teacher.gold & ~teacher.reachable_pairs).sum().detach(),
        correct_pairs=teacher.correct_pairs.sum().detach(),
        wrong_positive_pairs=teacher.wrong_positive_pairs.sum().detach(),
        wrong_negative_pairs=teacher.wrong_negative_pairs.sum().detach(),
        positive_queries=teacher.wrong_positive_pairs.any(-1).sum().detach(),
        negative_queries=teacher.wrong_negative_pairs.any(-1).sum().detach(),
        protected_queries=teacher.correct_pairs.any(-1).sum().detach(),
        queries=len(teacher.parent_semantic),
    )
