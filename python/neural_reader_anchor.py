"""Fit-only preservation for a frozen-input, four-parameter reader matcher.

Serving activates a slot when sigmoid(semantic) * fixed_gate > 1/2.
For fixed_gate > 1/2 this is equivalent to
semantic > -log(2 * fixed_gate - 1). No semantic value can activate a slot
whose fixed identity/temporal gate is at most 1/2.

The teacher protects only queries whose complete parent support set agrees
with fit labels. It caps each signed distance from that semantic decision
threshold, then penalizes the largest lost margin in each protected query.
This keeps complete-set errors visible without preserving saturated confidence
or averaging one critical slot into hundreds of unrelated negatives. All
eligibility comes from the detached parent; student predictions select nothing.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from neural_slot_model import RELATION_HEADS, SEMANTIC, WIDTH


MATCHER_KEYS = ("read.0.weight", "read.0.bias", "read.2.weight", "read.2.bias")
_MATCHER_SHAPES = ((WIDTH, 4 * SEMANTIC), (WIDTH,), (1, WIDTH), (1,))


def _finite_tensor(value, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.device.type == "meta" or not torch.isfinite(value).all()):
        raise ValueError(name + " must be a finite floating tensor")


def _parameters(parameters):
    if set(parameters) != set(MATCHER_KEYS):
        raise ValueError("reader parent requires exactly the four matcher tensors")
    reference = parameters[MATCHER_KEYS[0]]
    for name, shape in zip(MATCHER_KEYS, _MATCHER_SHAPES):
        value = parameters[name]
        _finite_tensor(value, name)
        if tuple(value.shape) != shape or value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError("invalid reader matcher parameter geometry or precision")


def snapshot_reader_matcher(model):
    """Return independent detached read4 tensors, never a registered model copy."""
    state = model.state_dict()
    parameters = {name: state[name].detach().clone() for name in MATCHER_KEYS}
    _parameters(parameters)
    return parameters


def matcher_pair_features(queries, memories, mask, attention_scales):
    """Exact full-bank read inputs with frozen encoders and attention weights."""
    for value, name in ((queries, "queries"), (memories, "memories"),
                        (attention_scales, "attention scales")):
        _finite_tensor(value, name)
        if value.requires_grad:
            raise ValueError("matcher anchors require frozen pair inputs")
    if (queries.ndim != 2 or queries.shape[1] < SEMANTIC
            or memories.ndim != 3 or memories.shape[2] != WIDTH
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(memories.shape[:2])
            or tuple(attention_scales.shape) != (RELATION_HEADS,)
            or not (attention_scales > 0).all()
            or any(value.device != queries.device for value in (memories, mask, attention_scales))
            or any(value.dtype != queries.dtype for value in (memories, attention_scales))):
        raise ValueError("invalid frozen matcher input geometry")
    if len(memories) and not mask.any(-1).all():
        raise ValueError("each memory requires a nonempty token representation")
    if not len(memories):
        return queries.new_empty((len(queries), 0, 4 * SEMANTIC))
    relations = queries[:, :SEMANTIC].reshape(len(queries), RELATION_HEADS, WIDTH)
    attention = torch.einsum("qhd,ntd->qnht", relations, memories) * .125
    attention = attention * attention_scales[None, None, :, None]
    attention = attention.masked_fill(~mask[None, :, None], -torch.inf).softmax(-1)
    attended = torch.einsum("qnht,ntd->qnhd", attention, memories)
    cue = relations[:, None].expand_as(attended)
    return torch.cat((cue, attended, cue * attended, (cue - attended).abs()), -1).flatten(-2)


def matcher_logits_from_parameters(pairs, parameters):
    """Evaluate current or detached parent read4 on the identical fixed inputs."""
    _parameters(parameters)
    _finite_tensor(pairs, "matcher pairs")
    reference = parameters[MATCHER_KEYS[0]]
    if (pairs.ndim != 3 or pairs.shape[-1] != 4 * SEMANTIC
            or pairs.device != reference.device or pairs.dtype != reference.dtype
            or pairs.requires_grad):
        raise ValueError("matcher logits require aligned frozen full-bank pairs")
    hidden = F.linear(pairs, parameters["read.0.weight"], parameters["read.0.bias"]).tanh()
    return F.linear(hidden, parameters["read.2.weight"], parameters["read.2.bias"]).squeeze(-1)


@dataclass(frozen=True)
class ReaderAnchorTeacher:
    correct_queries: torch.Tensor
    protected_queries: torch.Tensor
    eligible_pairs: torch.Tensor
    signed_targets: torch.Tensor
    semantic_thresholds: torch.Tensor
    required_margins: torch.Tensor
    parent_semantic: torch.Tensor
    fixed_log_gate: torch.Tensor
    margin: float


def build_reader_anchor_teacher(parent_semantic, parent_activation, gold, *,
                                fixed_log_gate, split, margin=1., evaluation_only=False):
    """Compile immutable complete support-set teachers from fit data only.

    The actual frozen identity/time log gate determines reachability and the
    semantic decision threshold. Rounded reader outputs never reconstruct this
    gate. A parent's tiny numerical margin is retained as-is so identical
    current parameters start at zero loss.
    """
    if split != "fit" or evaluation_only:
        raise ValueError("reader anchor labels must come from fit, never evaluation or runtime")
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin <= 0:
        raise ValueError("reader anchor margin must be finite and positive")
    _finite_tensor(parent_semantic, "parent semantic logits")
    _finite_tensor(parent_activation, "parent activation logits")
    _finite_tensor(fixed_log_gate, "fixed log gate")
    if (parent_semantic.ndim != 2 or parent_activation.shape != parent_semantic.shape
            or parent_activation.device != parent_semantic.device
            or parent_activation.dtype != parent_semantic.dtype
            or fixed_log_gate.shape != parent_semantic.shape
            or fixed_log_gate.device != parent_semantic.device
            or fixed_log_gate.dtype != parent_semantic.dtype
            or not isinstance(gold, torch.Tensor) or gold.shape != parent_semantic.shape
            or gold.device != parent_semantic.device
            or gold.is_complex() or not torch.isfinite(gold).all()
            or not ((gold == 0) | (gold == 1)).all()):
        raise ValueError("reader anchors require complete binary support sets with aligned logits")
    semantic = parent_semantic.detach().clone()
    activation = parent_activation.detach().clone()
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
    # expm1 is accurate immediately above the gate=1/2 decision boundary.
    # Unreachable entries receive a harmless placeholder and remain excluded.
    difference = (log_gate + math.log(2.)).masked_fill(~reachable, 1.)
    thresholds = -torch.log(torch.expm1(difference))
    thresholds = thresholds.masked_fill(~reachable, 0.)
    correct = ((activation > 0) == labels).all(-1)
    eligible = reachable & correct[:, None]
    targets = labels.to(semantic.dtype) * 2 - 1
    required = (targets * (semantic - thresholds)).clamp_max(margin)
    if not torch.isfinite(thresholds).all() or not torch.isfinite(required).all():
        raise ValueError("reader anchor threshold and margin arithmetic must remain finite")
    return ReaderAnchorTeacher(
        correct_queries=correct,
        protected_queries=correct & eligible.any(-1),
        eligible_pairs=eligible,
        signed_targets=targets,
        semantic_thresholds=thresholds,
        required_margins=required,
        parent_semantic=semantic,
        fixed_log_gate=log_gate,
        margin=float(margin),
    )


def reader_support_anchor_loss(student_semantic, teacher, weight=1.):
    """Mean complete-query maximum squared loss of the capped parent margin."""
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("reader anchor weight must be finite and nonnegative")
    if not isinstance(teacher, ReaderAnchorTeacher):
        raise ValueError("reader anchor requires a fit-only frozen teacher")
    _finite_tensor(student_semantic, "student semantic logits")
    if (student_semantic.shape != teacher.parent_semantic.shape
            or student_semantic.device != teacher.parent_semantic.device
            or student_semantic.dtype != teacher.parent_semantic.dtype):
        raise ValueError("student reader logits must match the frozen full-bank teacher")
    if not student_semantic.numel() or not teacher.protected_queries.any() or weight == 0:
        return student_semantic.reshape(-1)[:1].sum() * 0
    signed = teacher.signed_targets * (student_semantic - teacher.semantic_thresholds)
    difference = (teacher.required_margins - signed).masked_fill(~teacher.eligible_pairs, 0)
    violation = F.relu(difference).square()
    if not torch.isfinite(violation).all():
        raise ValueError("reader anchor loss arithmetic must remain finite")
    per_query = violation.max(-1).values
    loss = weight * per_query[teacher.protected_queries].mean()
    if not torch.isfinite(loss):
        raise ValueError("reader anchor weighted loss must remain finite")
    return loss
