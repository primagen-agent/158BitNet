"""V1 research contract, NOT a neural selector, decoder or serving path.

Only transports an already-selected contiguous value. Oracle provenance survives
every operation and requires explicit opt-in. Correct transport cannot certify
that a fact/span was semantically correct. No query matching or source retrieval.
"""
from dataclasses import dataclass, replace
from enum import Enum
import codecs
import hashlib
import json
import math

from neural_memory_contract import ModelBinding


class TransportError(ValueError):
    pass


class PrefixRetokenizationRequired(TransportError):
    pass


class UnsafeToken(TransportError):
    pass


class TransportBudgetExceeded(TransportError):
    pass


def require(ok, message):
    if not ok: raise TransportError(message)


def nonempty(value):
    require(type(value) is str and bool(value.strip()), 'nonempty identity required')


def sha(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'SHA-256 required')


def number(value, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer')


def text_bytes(value):
    require(type(value) is bytes and bool(value) and b'\0' not in value, 'nonempty immutable non-NUL bytes required')
    try: return value.decode('utf-8', errors='strict')
    except UnicodeDecodeError as e: raise TransportError('invalid UTF-8') from e


class Origin(str, Enum):
    ORACLE_FIXTURE = 'oracle_fixture'
    NEURAL_PREDICTION = 'neural_prediction'  # Declaration, not proof of capability.


@dataclass(frozen=True)
class FactPayload:
    fact_id: str
    data: bytes

    def __post_init__(self):
        nonempty(self.fact_id); text_bytes(self.data)


@dataclass(frozen=True)
class PayloadSnapshot:
    binding: ModelBinding
    memory_model_sha256: str
    scope: str
    memory_id: str
    revision: int
    facts: tuple[FactPayload, ...]

    def __post_init__(self):
        require(type(self.binding) is ModelBinding, 'complete model binding required')
        sha(self.memory_model_sha256); nonempty(self.scope); nonempty(self.memory_id); number(self.revision)
        require(type(self.facts) is tuple and all(type(f) is FactPayload for f in self.facts), 'immutable payload inventory required')
        require(len({f.fact_id for f in self.facts}) == len(self.facts), 'duplicate fact identity')

    @property
    def digest(self):
        fields = [vars(self.binding), self.memory_model_sha256, self.scope, self.memory_id, self.revision,
                  [(f.fact_id, f.data.hex()) for f in self.facts]]
        return hashlib.sha256(json.dumps(fields, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ReplyBinding:
    scope: str
    request_id: str
    context_sha256: str

    def __post_init__(self):
        nonempty(self.scope); nonempty(self.request_id); sha(self.context_sha256)


@dataclass(frozen=True)
class ValueSpan:
    fact_index: int
    start_byte: int
    end_byte: int

    def __post_init__(self):
        number(self.fact_index); number(self.start_byte); number(self.end_byte, 1)
        require(self.start_byte < self.end_byte, 'empty/reversed span')


@dataclass(frozen=True)
class ActivatedFact:
    snapshot_sha256: str
    reply: ReplyBinding
    span: ValueSpan | None
    confidence: float
    origin: Origin

    def __post_init__(self):
        sha(self.snapshot_sha256); require(type(self.reply) is ReplyBinding, 'reply binding required')
        require(self.span is None or type(self.span) is ValueSpan, 'value span or NULL required')
        require(type(self.confidence) is float and math.isfinite(self.confidence) and 0 <= self.confidence <= 1, 'finite confidence required')
        require(type(self.origin) is Origin, 'explicit provenance required')

    def validate(self, snapshot, reply, *, allow_oracle):
        require(type(allow_oracle) is bool, 'explicit oracle policy required')
        require(type(snapshot) is PayloadSnapshot and type(reply) is ReplyBinding, 'typed state required')
        require(self.snapshot_sha256 == snapshot.digest and self.reply == reply and snapshot.scope == reply.scope,
                'stale or cross-model/scope/request/context handle')
        require(self.origin != Origin.ORACLE_FIXTURE or allow_oracle, 'oracle fixture is not autonomous recall')
        if self.span is None: return None
        span = self.span
        require(span.fact_index < len(snapshot.facts), 'fact outside snapshot')
        raw = snapshot.facts[span.fact_index].data
        require(span.end_byte <= len(raw), 'value outside fact')
        # Validate both boundaries, including a start inside a UTF-8 sequence.
        try: raw[:span.start_byte].decode('utf-8'); raw[:span.end_byte].decode('utf-8')
        except UnicodeDecodeError as e: raise TransportError('span cuts a UTF-8 character') from e
        return raw[span.start_byte:span.end_byte]


@dataclass(frozen=True)
class TransportLimits:
    max_value_bytes: int = 256
    max_value_tokens: int = 64
    max_total_tokens: int = 128

    def __post_init__(self):
        for v in vars(self).values(): number(v, 1)


def token_ids(ids, codec):
    require(type(ids) is tuple and bool(ids) and all(type(t) is int and 0 <= t < codec.vocab for t in ids), 'immutable valid token IDs required')


def compile_value(handle, snapshot, reply, prefix_ids, prefix_text, codec, *, limits=TransportLimits(), allow_oracle=False):
    """Append-only canonical C tokenization; fail before emission on instability.

    codec must expose binding identities, vocab, bos_id, encode_with_bos,
    decode_pieces, forbidden and verify_identity. Test codecs are explicit mocks.
    This does not insert source text into a prompt or run the backbone.
    """
    require(type(handle) is ActivatedFact and type(limits) is TransportLimits, 'typed handle/limits required')
    value = handle.validate(snapshot, reply, allow_oracle=allow_oracle)
    if value is None: return None  # No tokenizer/decoder call for NULL.
    if len(value) > limits.max_value_bytes: raise TransportBudgetExceeded('value byte budget exceeded')
    codec.verify_identity()
    require(codec.backbone_sha256 == snapshot.binding.backbone_sha256 and codec.tokenizer_sha256 == snapshot.binding.tokenizer_sha256,
            'codec identity differs from bound model')
    token_ids(prefix_ids, codec)
    if len(prefix_ids) >= limits.max_total_tokens: raise TransportBudgetExceeded('no remaining context capacity')
    require(prefix_ids[0] == codec.bos_id and type(prefix_text) is str and bool(prefix_text) and '\0' not in prefix_text, 'complete prefix required')
    # Prefix text is supplied by the caller and verified, never inferred from metadata.
    try: prefix_text.encode('utf-8')
    except UnicodeEncodeError as e: raise TransportError('invalid prefix UTF-8') from e
    if tuple(codec.encode_with_bos(prefix_text)) != prefix_ids:
        raise PrefixRetokenizationRequired('actual prefix is not canonical under this conservative adapter')
    original_pieces = tuple(codec.decode_pieces(prefix_ids[1:]))
    original = b''.join(original_pieces)
    require(original in (prefix_text.encode(), b' ' + prefix_text.encode()), 'prefix bytes changed')
    combined = tuple(codec.encode_with_bos(prefix_text + text_bytes(value)))
    token_ids(combined, codec)
    if combined[:len(prefix_ids)] != prefix_ids:
        raise PrefixRetokenizationRequired('value would rewrite already emitted tokens')
    suffix = combined[len(prefix_ids):]
    require(bool(suffix), 'empty encoded value')
    pieces = tuple(codec.decode_pieces(suffix))
    require(len(pieces) == len(suffix) and all(type(p) is bytes and p for p in pieces), 'empty or misaligned token piece')
    require(b''.join(pieces) == value, 'encoded suffix does not preserve exact value bytes')
    for t, piece in zip(suffix, pieces):
        if codec.forbidden(t) or b'\0' in piece: raise UnsafeToken('control token in value')
    if len(suffix) > limits.max_value_tokens or len(combined) > limits.max_total_tokens:
        raise TransportBudgetExceeded('token budget exceeded before emission')
    codec.verify_identity()
    return ValueCursor(handle, prefix_ids, suffix, pieces, value, 0)


@dataclass(frozen=True)
class ValueCursor:
    handle: ActivatedFact
    initial_prefix: tuple[int, ...]
    value_tokens: tuple[int, ...]
    pieces: tuple[bytes, ...]
    value: bytes
    offset: int

    def __post_init__(self):
        require(type(self.handle) is ActivatedFact and self.handle.span is not None, 'non-NULL handle required')
        for ids in (self.initial_prefix, self.value_tokens):
            require(type(ids) is tuple and bool(ids) and all(type(t) is int and t >= 0 for t in ids), 'immutable IDs required')
        require(type(self.pieces) is tuple and len(self.pieces) == len(self.value_tokens) and
                all(type(p) is bytes and p for p in self.pieces), 'immutable aligned pieces required')
        text_bytes(self.value); require(b''.join(self.pieces) == self.value, 'cursor bytes differ from selected value')
        number(self.offset); require(self.offset <= len(self.value_tokens), 'cursor outside value')

    @property
    def done(self):
        return self.offset == len(self.value_tokens)

    @property
    def expected_prefix(self):
        return self.initial_prefix + self.value_tokens[:self.offset]

    @property
    def committed_text(self):
        """Complete UTF-8 only; withhold a partial final character, never replace it.

        This is a software view of acknowledged pieces, not an HTTP/SSE adapter.
        """
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        return decoder.decode(b''.join(self.pieces[:self.offset]), final=self.done)

    def validate(self, snapshot, reply, prefix, *, allow_oracle):
        require(self.handle.validate(snapshot, reply, allow_oracle=allow_oracle) == self.value, 'selected bytes changed')
        require(type(prefix) is tuple and prefix == self.expected_prefix, 'not the actual ordered prefix')

    def select(self, base_token, snapshot, reply, prefix, *, allow_oracle=False):
        self.validate(snapshot, reply, prefix, allow_oracle=allow_oracle)
        number(base_token)
        return base_token if self.done else self.value_tokens[self.offset]

    def commit(self, emitted_token, snapshot, reply, prefix, *, allow_oracle=False):
        self.validate(snapshot, reply, prefix, allow_oracle=allow_oracle)
        require(not self.done and type(emitted_token) is int and emitted_token == self.value_tokens[self.offset],
                'skip, premature stop, replacement or commit after value end')
        return replace(self, offset=self.offset + 1)


def select_token(cursor, base_token, snapshot, reply, prefix, *, allow_oracle=False):
    """NORMAL/NULL is a literal passthrough; completed cursors can then be dropped."""
    if cursor is None: return base_token
    require(type(cursor) is ValueCursor, 'bound cursor required')
    return cursor.select(base_token, snapshot, reply, prefix, allow_oracle=allow_oracle)
