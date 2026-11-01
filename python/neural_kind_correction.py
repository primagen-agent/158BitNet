"""Fit-only capped source-kind correction on an independently frozen parent.

Categorical classification uses the original source label, never a physical
slot status. Correct parent classifications need only retain their capped
gold-versus-best-rival margin. Incorrect parent classifications learn the same
finite target margin; sufficient confidence has exactly zero gradient.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


_LABEL_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _logits(value, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 2 or value.shape[1] != 3 or not len(value)
            or value.device.type == "meta"
            or not torch.isfinite(value).all()):
        raise ValueError(name + " must be finite nonempty floating [N,3] source-kind logits")


def _gold_gap(logits, labels):
    gold = logits.gather(1, labels[:, None]).squeeze(1)
    rivals = logits.masked_fill(F.one_hot(labels, 3).bool(), -torch.inf).max(-1).values
    gap = gold - rivals
    if not torch.isfinite(gap).all():
        raise ValueError("source-kind margin arithmetic must remain finite")
    return gap


@dataclass(frozen=True)
class KindCorrectionTeacher:
    parent_logits: torch.Tensor
    labels: torch.Tensor
    correct: torch.Tensor
    parent_gap: torch.Tensor
    required_gap: torch.Tensor
    class_counts: torch.Tensor
    margin: float


def snapshot_kind_teacher(logits, labels, split="fit", margin=1., evaluation_only=False):
    """Detach independent parent outputs and fixed complete-class fit targets."""
    if split != "fit" or evaluation_only:
        raise ValueError("source-kind correction labels must come from fit, never evaluation or runtime")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("source-kind correction margin must be finite and positive")
    _logits(logits, "parent")
    if (not isinstance(labels, torch.Tensor) or labels.dtype not in _LABEL_DTYPES
            or labels.shape != (len(logits),) or labels.device != logits.device
            or not ((labels >= 0) & (labels < 3)).all()):
        raise ValueError("source-kind correction needs aligned integer labels in 0..2")
    parent = logits.detach().clone()
    targets = labels.detach().to(torch.long).clone()
    correct = (parent.argmax(-1) == targets).detach().clone()
    gap = _gold_gap(parent, targets).detach().clone()
    required = torch.where(correct, gap.clamp(min=0, max=margin),
                           torch.full_like(gap, margin)).detach().clone()
    counts = torch.stack([(targets == kind).sum() for kind in range(3)]).detach().clone()
    return KindCorrectionTeacher(parent, targets, correct, gap, required, counts, float(margin))


def _class_balanced(values, teacher):
    """Use original complete-class denominators for every reported component."""
    per_class = torch.stack([
        values.masked_fill(teacher.labels != kind, 0).sum()
        / teacher.class_counts[kind].clamp_min(1)
        for kind in range(3)
    ])
    present = teacher.class_counts > 0
    return per_class.masked_fill(~present, 0).sum() / present.sum()


def kind_correction_loss(student_logits, teacher):
    """Return capped class-balanced loss and detached fit diagnostics.

Teacher eligibility never depends on the student's prediction. Preservation
and correction use the same complete-class denominator, so a rare old error
does not acquire a second independent normalization. Actual correctness is
diagnostic only and never affects targets or checkpoint selection.
    """
    if not isinstance(teacher, KindCorrectionTeacher):
        raise ValueError("source-kind correction requires a fit-only frozen parent teacher")
    _logits(student_logits, "student")
    if (student_logits.shape != teacher.parent_logits.shape
            or student_logits.device != teacher.parent_logits.device
            or student_logits.dtype != teacher.parent_logits.dtype):
        raise ValueError("student source-kind logits must match the frozen teacher geometry and precision")
    gap = _gold_gap(student_logits, teacher.labels)
    losses = F.relu(teacher.required_gap - gap).square()
    if not torch.isfinite(losses).all():
        raise ValueError("source-kind correction loss arithmetic must remain finite")
    preservation = _class_balanced(losses.masked_fill(~teacher.correct, 0), teacher)
    correction = _class_balanced(losses.masked_fill(teacher.correct, 0), teacher)
    loss = preservation + correction
    actual = (student_logits.detach().argmax(-1) == teacher.labels).sum()
    return loss, dict(
        correct_preservation=preservation.detach(), wrong_correction=correction.detach(),
        samples=teacher.labels.new_tensor(len(teacher.labels)),
        parent_correct_samples=teacher.correct.sum().detach(),
        parent_wrong_samples=(~teacher.correct).sum().detach(),
        class_sample_counts=teacher.class_counts.detach().clone(),
        current_actual_correct=actual.detach(),
    )
