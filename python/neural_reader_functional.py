"""Fit-only behavioral preservation for the complete parent memory reader.

The caller runs an independently frozen parent on the original query, memory
bank and evidence rounds. These teachers never evaluate the parent classifier
on student features. Final activation margins therefore also protect changes
caused by query attention, feedback, identity inputs and temporal inputs.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from neural_slot_model import WIDTH


def _finite_float(value, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or not torch.isfinite(value).all()):
        raise ValueError(name + " must be a finite floating tensor")


def _aligned_outputs(activation, semantic, identity, intent, temporal_gate):
    for value, name in ((activation, "activation logits"), (semantic, "semantic logits"),
                        (identity, "identity logits"), (intent, "temporal intent"),
                        (temporal_gate, "temporal gate")):
        _finite_float(value, name)
    if (activation.ndim != 2 or semantic.shape != activation.shape
            or identity.shape != activation.shape or temporal_gate.shape != activation.shape
            or intent.shape != (len(activation), WIDTH)
            or any(value.device != activation.device
                   or value.dtype != activation.dtype
                   for value in (semantic, identity, intent, temporal_gate))
            or not ((temporal_gate >= 0) & (temporal_gate <= 1)).all()):
        raise ValueError("functional reader outputs require aligned full-bank logits and temporal probabilities")


def _weight(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("functional reader weight must be finite and nonnegative")


@dataclass(frozen=True)
class ReaderFunctionalTeacher:
    parent_activation: torch.Tensor
    parent_semantic: torch.Tensor
    parent_identity: torch.Tensor
    parent_intent: torch.Tensor
    parent_temporal_gate: torch.Tensor
    correct_queries: torch.Tensor
    protected_queries: torch.Tensor
    signed_targets: torch.Tensor
    required_margins: torch.Tensor
    activation_probability: torch.Tensor
    semantic_probability: torch.Tensor
    identity_probability: torch.Tensor
    semantic_mask: torch.Tensor
    identity_mask: torch.Tensor
    temporal_mask: torch.Tensor
    intent_relevance: torch.Tensor
    margin: float


def build_reader_functional_teacher(parent_activation, parent_semantic, parent_identity,
                                    parent_intent, parent_temporal_gate, gold, *,
                                    split, margin=1., evaluation_only=False):
    """Clone detached full-model outputs and immutable fit-only eligibility.

Only a parent's complete correct support set is protected, including complete
null queries. Parent mistakes remain available for correction by supervised
training. The time input is the aggregate gate probability, after all learned
version and temporal-intent factors, rather than a version logit alone.
    """
    if split != "fit" or evaluation_only:
        raise ValueError("functional reader labels must come from fit, never evaluation or runtime")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("functional reader margin must be finite and positive")
    _aligned_outputs(parent_activation, parent_semantic, parent_identity,
                     parent_intent, parent_temporal_gate)
    if (not isinstance(gold, torch.Tensor) or gold.shape != parent_activation.shape
            or gold.device != parent_activation.device or gold.is_complex()
            or not torch.isfinite(gold).all() or not ((gold == 0) | (gold == 1)).all()):
        raise ValueError("functional reader labels require complete binary support sets")
    activation, semantic, identity, intent, temporal = (
        value.detach().clone() for value in
        (parent_activation, parent_semantic, parent_identity, parent_intent, parent_temporal_gate)
    )
    labels = gold.detach().bool().clone()
    probability, semantic_probability, identity_probability = (
        value.sigmoid() for value in (activation, semantic, identity)
    )
    tolerance = 64 * torch.finfo(activation.dtype).eps
    if ((probability > semantic_probability + tolerance).any()
            or (probability > identity_probability + tolerance).any()
            or (probability > temporal + tolerance).any()):
        raise ValueError("parent activation probability cannot exceed any learned gate")
    correct = ((activation > 0) == labels).all(-1)
    protected = correct & (activation.shape[1] > 0)
    signed = labels.to(activation.dtype) * 2 - 1
    required = (signed * activation).clamp(min=0, max=margin)
    # Fixed parent probabilities suppress irrelevant component constraints:
    # a different-person slot must not dominate semantic retention, and a
    # semantically unrelated slot must not dominate identity/time retention.
    eligible = protected[:, None].to(activation.dtype)
    semantic_mask = identity_probability * temporal * eligible
    identity_mask = semantic_probability * temporal * eligible
    temporal_mask = semantic_probability * identity_probability * eligible
    relevance = probability.max(-1).values if activation.shape[1] else activation.new_zeros(len(activation))
    return ReaderFunctionalTeacher(
        activation, semantic, identity, intent, temporal, correct, protected,
        signed, required, probability, semantic_probability, identity_probability,
        semantic_mask, identity_mask, temporal_mask, relevance * protected,
        float(margin),
    )


def _student_activation(student, teacher):
    if not isinstance(teacher, ReaderFunctionalTeacher):
        raise ValueError("functional reader requires a fit-only frozen full-model teacher")
    _finite_float(student, "student activation logits")
    if (student.shape != teacher.parent_activation.shape
            or student.device != teacher.parent_activation.device
            or student.dtype != teacher.parent_activation.dtype):
        raise ValueError("student reader outputs must match the frozen full-bank teacher")


def reader_functional_margin_loss(student_activation, teacher, weight=1.):
    """Protect the worst capped final activation-margin loss per fit query."""
    _weight(weight)
    _student_activation(student_activation, teacher)
    if not student_activation.numel() or not teacher.protected_queries.any() or weight == 0:
        return student_activation.sum() * 0
    signed = teacher.signed_targets * student_activation
    violation = F.relu(teacher.required_margins - signed).square()
    per_query = violation.max(-1).values
    return weight * per_query[teacher.protected_queries].mean()


def reader_functional_distillation_loss(student_activation, student_semantic, student_identity,
                                       student_intent, student_temporal_gate, teacher, weight=1.):
    """Preserve complete-parent behavior with detached probability masks.

Probability differences avoid anchoring saturated semantic or identity logits.
Each pair component uses the largest weighted difference in the query, so an
important changed slot cannot disappear in a large bank. Intent is a 64-wide
feature channel: relative bounded differences preserve its actual representation
without treating all its values as Bernoulli logits. Its relevance comes only
from the frozen parent's final activation probability.
    """
    _weight(weight)
    _student_activation(student_activation, teacher)
    _aligned_outputs(student_activation, student_semantic, student_identity,
                     student_intent, student_temporal_gate)
    zero = sum(value.sum() * 0 for value in
               (student_activation, student_semantic, student_identity,
                student_intent, student_temporal_gate))
    if not student_activation.numel() or not teacher.protected_queries.any() or weight == 0:
        return zero
    activation = (student_activation.sigmoid() - teacher.activation_probability).square()
    semantic = ((student_semantic.sigmoid() - teacher.semantic_probability).square()
                * teacher.semantic_mask)
    identity = ((student_identity.sigmoid() - teacher.identity_probability).square()
                * teacher.identity_mask)
    temporal = ((student_temporal_gate - teacher.parent_temporal_gate).square()
                * teacher.temporal_mask)
    intent_difference = ((student_intent - teacher.parent_intent)
                         / (1 + teacher.parent_intent.abs())).tanh()
    intent = intent_difference.square().mean(-1) * teacher.intent_relevance
    per_query = sum(value.max(-1).values for value in (activation, semantic, identity, temporal)) + intent
    return weight * per_query[teacher.protected_queries].mean() + zero
