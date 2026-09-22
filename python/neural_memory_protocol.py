"""P1 evaluation boundaries and conservative, independently adjudicated scoring.

This is not an NLP judge. Unknown free-form replies remain unscored until an
independent reviewer supplies claims and discourse assessments bound to the text.
The evaluated model must never produce its own adjudication or receive the rubric.
"""
from dataclasses import dataclass
from enum import Enum
import hashlib

from neural_memory_contract import ConversationContext, MemoryState, require, sequence


class EvaluationLevel(str, Enum):
    ORACLE_COMPONENT = "oracle_component"
    PREDICTED_COMPONENT = "predicted_component"
    END_TO_END = "end_to_end"


@dataclass(frozen=True)
class RuntimeInput:
    context: ConversationContext
    state: MemoryState

    def __post_init__(self):
        require(type(self.context) is ConversationContext and type(self.state) is MemoryState, "runtime types required")
        require(self.context.scope == self.state.scope, "input scope mismatch")
        require(all(e.learned_at_ms <= self.context.at_ms for e in self.state.episodes), "future memory")


@dataclass(frozen=True)
class Claim:
    subject: str
    relation: str
    value: str
    time: str
    status: str

    def __post_init__(self):
        require(all(type(v) is str and bool(v.strip()) for v in vars(self).values()), "complete claim required")


@dataclass(frozen=True)
class AnswerRubric:
    required: tuple[Claim, ...]
    allowed: tuple[Claim, ...]
    evidence_missing: bool
    memory_narration_allowed: bool = False

    def __post_init__(self):
        sequence(self.required, Claim); sequence(self.allowed, Claim)
        require(type(self.evidence_missing) is bool and type(self.memory_narration_allowed) is bool, "invalid rubric flags")
        require(set(self.required).issubset(self.allowed), "required claims must be allowed")
        require(not self.evidence_missing or not self.required, "missing-evidence case cannot demand facts")


def response_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Adjudication:
    response_sha256: str
    reviewer_id: str
    claims: tuple[Claim, ...]
    acknowledges_missing_evidence: bool
    natural_reply: bool
    unsolicited_memory_narration: bool

    def __post_init__(self):
        sequence(self.claims, Claim)
        require(bool(self.reviewer_id), "independent reviewer identity required")
        for flag in (self.acknowledges_missing_evidence, self.natural_reply, self.unsolicited_memory_narration):
            require(type(flag) is bool, "review flags must be explicit booleans")


def score_response(text, rubric, adjudication=None):
    """Return pass/fail/needs_review; never turn unknown text into a substring pass."""
    if not text.strip():
        return {"status": "fail", "reasons": ["empty_reply"]}
    if adjudication is None:
        return {"status": "needs_review", "reasons": ["independent_adjudication_required"]}
    require(adjudication.response_sha256 == response_hash(text), "review belongs to different response")
    actual, required, allowed = set(adjudication.claims), set(rubric.required), set(rubric.allowed)
    reasons = []
    if not required.issubset(actual): reasons.append("missing_required_claim")
    if not actual.issubset(allowed): reasons.append("unsupported_or_misattributed_claim")
    if rubric.evidence_missing and not adjudication.acknowledges_missing_evidence:
        reasons.append("unjustified_certainty")
    if not adjudication.natural_reply: reasons.append("not_natural_reply")
    if adjudication.unsolicited_memory_narration and not rubric.memory_narration_allowed:
        reasons.append("unsolicited_memory_narration")
    return {"status": "fail" if reasons else "pass", "reasons": reasons}


@dataclass(frozen=True)
class InferenceTrace:
    """Required observations, not optional flags defaulting to 'safe'.

    Runtime instrumentation must supply these. Self-report alone is not proof of
    no leakage; causality tests and call-boundary inspection remain mandatory.
    """
    level: EvaluationLevel
    backbone_kv_cache_enabled: bool
    cached_tokens: int
    reused_tokens: int
    label_fields_in_forward: tuple[str, ...]
    future_messages_in_forward: tuple[str, ...]
    oracle_activation: bool
    supplied_write_boundaries: bool
    source_text_in_prompt: bool
    whole_answer_bypass: bool
    generated_by_decoder: bool

    def validate(self, *, generation_required=True):
        require(type(self.level) is EvaluationLevel, "evaluation level required")
        for count in (self.cached_tokens, self.reused_tokens):
            require(type(count) is int and count == 0, "KV reuse invalidates memory measurement")
        for flag in (self.backbone_kv_cache_enabled, self.oracle_activation, self.supplied_write_boundaries,
                     self.source_text_in_prompt, self.whole_answer_bypass, self.generated_by_decoder):
            require(type(flag) is bool, "trace flags must be observed booleans")
        sequence(self.label_fields_in_forward, str); sequence(self.future_messages_in_forward, str)
        require(not self.backbone_kv_cache_enabled, "backbone KV cache forbidden")
        require(not self.label_fields_in_forward and not self.future_messages_in_forward, "label/future leakage")
        require(not self.source_text_in_prompt and not self.whole_answer_bypass, "retrieval/answer bypass forbidden")
        require(not self.oracle_activation or self.level == EvaluationLevel.ORACLE_COMPONENT, "unreported oracle routing")
        if self.level == EvaluationLevel.END_TO_END:
            require(not self.supplied_write_boundaries, "supplied writes are not automatic memory")
        if generation_required:
            require(self.generated_by_decoder, "generation evidence required")
