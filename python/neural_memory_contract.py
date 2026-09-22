"""Research-only P1 contracts; not a model, semantic writer, or persistent store.

Construct runtime objects from application data, never from annotated training rows.
Immutable tuples deliberately preserve event boundaries and snapshot identity.
"""
from dataclasses import dataclass
from enum import Enum
import math
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identifier(value):
    require(isinstance(value, str) and bool(value.strip()), "nonempty identifier required")


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer")


def sequence(value, kind):
    require(type(value) is tuple and all(type(x) is kind for x in value),
            "immutable typed tuple required")


@dataclass(frozen=True)
class ModelBinding:
    backbone_sha256: str
    tokenizer_sha256: str
    encoder_sha256: str
    protocol_sha256: str

    def __post_init__(self):
        for value in vars(self).values():
            require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
                    "SHA-256 identity required")


@dataclass(frozen=True)
class Message:
    scope: str
    conversation_id: str
    message_id: str
    speaker_id: str
    role: str
    sequence: int
    observed_at_ms: int
    text: str

    def __post_init__(self):
        for value in (self.scope, self.conversation_id, self.message_id, self.speaker_id):
            identifier(value)
        require(self.role in ("system", "user", "assistant", "tool"), "invalid role")
        integer(self.sequence); integer(self.observed_at_ms)
        require(type(self.text) is str and bool(self.text), "source text required")
        self.text.encode("utf-8", errors="strict")


@dataclass(frozen=True)
class ConversationContext:
    scope: str
    conversation_id: str
    request_id: str
    at_ms: int
    messages: tuple[Message, ...]

    def __post_init__(self):
        for value in (self.scope, self.conversation_id, self.request_id): identifier(value)
        integer(self.at_ms); sequence(self.messages, Message)
        require(bool(self.messages), "current user input required")
        require(self.messages[-1].role == "user", "context must stop at current user input")
        require(len({m.message_id for m in self.messages}) == len(self.messages), "duplicate message id")
        require(all(a.sequence < b.sequence for a, b in zip(self.messages, self.messages[1:])),
                "messages must be strictly ordered")
        for message in self.messages:
            require(message.scope == self.scope and message.conversation_id == self.conversation_id,
                    "context scope/conversation mismatch")
            require(message.observed_at_ms <= self.at_ms, "future message in context")


@dataclass(frozen=True)
class SourceSpan:
    message: Message
    start_byte: int
    end_byte: int

    def __post_init__(self):
        require(type(self.message) is Message, "source message required")
        integer(self.start_byte); integer(self.end_byte, 1)
        raw = self.message.text.encode("utf-8")
        require(self.start_byte < self.end_byte <= len(raw), "source span out of bounds")
        # Checking both prefixes also rejects a span starting inside a multibyte character.
        raw[:self.start_byte].decode("utf-8", errors="strict")
        raw[:self.end_byte].decode("utf-8", errors="strict")

    @property
    def text(self):
        return self.message.text.encode("utf-8")[self.start_byte:self.end_byte].decode("utf-8")


class FactStatus(str, Enum):
    ACTUAL = "actual"
    NEGATED = "negated"
    HYPOTHETICAL = "hypothetical"
    QUOTED = "quoted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Episode:
    event_id: str
    scope: str
    sources: tuple[SourceSpan, ...]
    learned_at_ms: int
    # Model predictions, NOT gold labels. Unknown semantic time remains None.
    event_time: str | None = None
    fact_status: FactStatus = FactStatus.UNKNOWN
    retracted: bool = False

    def __post_init__(self):
        identifier(self.event_id); identifier(self.scope); integer(self.learned_at_ms)
        sequence(self.sources, SourceSpan)
        require(bool(self.sources), "episode must retain provenance")
        require(type(self.fact_status) is FactStatus and type(self.retracted) is bool, "invalid fact state")
        if self.event_time is not None: identifier(self.event_time)
        for span in self.sources:
            require(span.message.scope == self.scope, "cross-scope episode")
            require(span.message.observed_at_ms <= self.learned_at_ms, "future episode source")


@dataclass(frozen=True)
class MemoryState:
    scope: str
    revision: int
    binding: ModelBinding
    episodes: tuple[Episode, ...]

    def __post_init__(self):
        identifier(self.scope); integer(self.revision)
        require(type(self.binding) is ModelBinding, "model binding required")
        sequence(self.episodes, Episode)
        require(all(e.scope == self.scope for e in self.episodes), "cross-scope memory")
        require(len({e.event_id for e in self.episodes}) == len(self.episodes), "duplicate event id")


class Operation(str, Enum):
    APPEND = "append"
    REPLACE = "replace"
    COEXIST = "coexist"
    RETRACT = "retract"


@dataclass(frozen=True)
class WriteOperation:
    operation: Operation
    candidate: Episode | None
    target_ids: tuple[str, ...] = ()

    def __post_init__(self):
        require(type(self.operation) is Operation, "invalid operation")
        sequence(self.target_ids, str)
        for target in self.target_ids: identifier(target)
        require(len(set(self.target_ids)) == len(self.target_ids), "duplicate target")
        if self.operation == Operation.RETRACT:
            require(self.candidate is None and bool(self.target_ids), "retraction needs only targets")
        else:
            require(type(self.candidate) is Episode and not self.candidate.retracted, "live candidate required")
            if self.operation == Operation.APPEND:
                require(not self.target_ids, "append must not replace an event")
            else:
                require(bool(self.target_ids), "directed relation needs targets")


@dataclass(frozen=True)
class WriteDelta:
    scope: str
    base_revision: int
    request_id: str
    operations: tuple[WriteOperation, ...]

    def __post_init__(self):
        identifier(self.scope); identifier(self.request_id); integer(self.base_revision)
        sequence(self.operations, WriteOperation)

    def validate_against(self, state: MemoryState, context: ConversationContext):
        require(self.scope == state.scope == context.scope, "write scope mismatch")
        require(self.base_revision == state.revision, "stale write snapshot")
        require(self.request_id == context.request_id, "idempotency key must come from application")
        existing = {e.event_id: e for e in state.episodes}
        messages = {m.message_id: m for m in context.messages}
        new_ids, mutated = set(), set()
        for operation in self.operations:
            require(all(t in existing and not existing[t].retracted for t in operation.target_ids),
                    "target missing or retracted")
            if operation.operation in (Operation.REPLACE, Operation.RETRACT):
                require(not mutated.intersection(operation.target_ids), "conflicting mutations")
                mutated.update(operation.target_ids)
            candidate = operation.candidate
            if candidate is not None:
                require(candidate.scope == state.scope, "candidate scope mismatch")
                require(candidate.event_id not in existing and candidate.event_id not in new_ids,
                        "duplicate candidate id")
                new_ids.add(candidate.event_id)
                require(candidate.learned_at_ms == context.at_ms, "write time must come from application")
                for span in candidate.sources:
                    require(messages.get(span.message.message_id) == span.message, "source not in context")
                    require(span.message.role in ("user", "tool"), "automatic assistant self-write forbidden")
        # No mutation here. Atomic commit, replay handling and durable storage belong to P3/P5.


class Availability(str, Enum):
    DISABLED = "disabled"
    NOT_NEEDED = "not_needed"
    MISSING = "missing"
    CONFLICT = "conflict"
    AVAILABLE = "available"


@dataclass(frozen=True)
class Activation:
    event_id: str
    weight: float
    payloads: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        identifier(self.event_id); sequence(self.payloads, SourceSpan)
        require(type(self.weight) in (float, int) and math.isfinite(self.weight) and 0 < self.weight <= 1,
                "activation must be finite and in (0, 1]")


@dataclass(frozen=True)
class ReadContext:
    scope: str
    revision: int
    binding: ModelBinding
    availability: Availability
    activations: tuple[Activation, ...]

    def __post_init__(self):
        identifier(self.scope); integer(self.revision)
        require(type(self.binding) is ModelBinding and type(self.availability) is Availability, "invalid read metadata")
        sequence(self.activations, Activation)
        require(len({a.event_id for a in self.activations}) == len(self.activations), "duplicate activation")
        if self.availability in (Availability.DISABLED, Availability.NOT_NEEDED, Availability.MISSING):
            require(not self.activations, "NULL/disabled read must not expose content")
        else:
            require(bool(self.activations), "evidence required")

    def validate_against(self, state: MemoryState):
        require((self.scope, self.revision, self.binding) == (state.scope, state.revision, state.binding),
                "read snapshot/model mismatch")
        events = {e.event_id: e for e in state.episodes if not e.retracted}
        for activation in self.activations:
            require(activation.event_id in events, "activation references missing/retracted event")
            for payload in activation.payloads:
                require(any(payload.message == source.message and
                            source.start_byte <= payload.start_byte < payload.end_byte <= source.end_byte
                            for source in events[activation.event_id].sources), "payload outside activated event")
