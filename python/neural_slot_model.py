"""Current learned memory: exact source plans and full-bank neural activation."""

import hashlib
from pathlib import Path
import struct
import zlib
import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F

FORMAT = "BITNET_NEURAL_SLOTS"
HIDDEN = 2048
LEXICAL = 2048
INPUT = HIDDEN + LEXICAL
BOUNDARY_CONTEXT = 3
BOUNDARY_INPUT = BOUNDARY_CONTEXT * INPUT
WIDTH = 64
READ_CONTEXT_WIDTH = 16
TOKEN_LIMIT = 4096
SOURCE_CONTEXT_WIDTH = 32
SOURCE_CONTEXT_INPUT = 2 * WIDTH + HIDDEN
MEMBERSHIP_INPUT = WIDTH + BOUNDARY_INPUT
RELATION_HEADS = 4
SEMANTIC = RELATION_HEADS * WIDTH
QUERY = SEMANTIC + WIDTH
IDENTITY = LEXICAL + WIDTH
SOURCE_INPUT = HIDDEN + 6
ACTIONS = ("ignore", "write", "update", "delete")
KEYS = (
    "encode.weight",
    "encode.bias",
    "role.0.weight",
    "role.0.bias",
    "role.2.weight",
    "role.2.bias",
    "query_pool.weight",
    "read.0.weight",
    "read.0.bias",
    "read.2.weight",
    "read.2.bias",
    "scope.0.weight",
    "scope.0.bias",
    "scope.2.weight",
    "scope.2.bias",
    "version.0.weight",
    "version.0.bias",
    "version.2.weight",
    "version.2.bias",
    "identity_match.0.weight",
    "identity_match.0.bias",
    "identity.weight",
    "identity.bias",
    "kind.weight",
    "kind.bias",
    "intent.weight",
    "intent.bias",
    "identity_match.2.weight",
    "plan_encode.weight",
    "plan_encode.bias",
    "plan_context.weight",
    "plan_context.bias",
    "plan_tags.weight",
    "plan_tags.bias",
    "plan_need.weight",
    "plan_need.bias",
    "plan_ref_query.weight",
    "plan_ref_query.bias",
    "plan_ref_encode.weight",
    "plan_ref_encode.bias",
    "plan_ref_pair.0.weight",
    "plan_ref_pair.0.bias",
    "plan_ref_pair.2.weight",
    "plan_ref_pair.2.bias",
    "read_update.weight",
    "read_update.bias",
    "plan_transition",
    "plan_start",
    "plan_end",
    "plan_origin.weight",
    "plan_origin.bias",
    "query_need.weight",
    "query_need.bias",
    "query_ref_query.weight",
    "query_ref_query.bias",
    "query_origin.weight",
    "query_origin.bias",
    "plan_local.weight",
    "plan_local.bias",
    "query_ref_encode.weight",
    "query_ref_encode.bias",
    "query_ref_pair.0.weight",
    "query_ref_pair.0.bias",
    "query_ref_pair.2.weight",
    "query_ref_pair.2.bias",
    "query_encode.weight",
    "query_encode.bias",
    "identity_update.weight",
    "identity_update.bias",
    "identity_match.2.bias",
    "memory_attention_logits",
    "plan_boundary.0.weight",
    "plan_boundary.0.bias",
    "plan_boundary.2.weight",
    "plan_boundary.2.bias",
    "plan_boundary_raw.weight",
    "scope_query_encode.weight",
    "scope_query_encode.bias",
    "scope_query_pool.weight",
    "scope_memory_attention_logits",
    "plan_membership_raw.weight",
    "plan_source_context.0.weight",
    "plan_source_context.0.bias",
    "plan_source_context.2.weight",
    "plan_source_context.2.bias",
    "kind_context.weight",
    "kind_context.bias",
    "read_context_memory.weight",
    "read_context_query.weight",
    "read_context_delta.weight",
    "query_context.weight",
)
TENSOR_SHAPES = {
    "encode.weight": (WIDTH, INPUT + 2),
    "encode.bias": (64,),
    "role.0.weight": (WIDTH, HIDDEN + 2),
    "role.0.bias": (64,),
    "role.2.weight": (1, 64),
    "role.2.bias": (1,),
    "query_pool.weight": (RELATION_HEADS, WIDTH),
    "read.0.weight": (WIDTH, 4 * SEMANTIC),
    "read.0.bias": (64,),
    "read.2.weight": (1, 64),
    "read.2.bias": (1,),
    "scope.0.weight": (WIDTH, 4 * SEMANTIC),
    "scope.0.bias": (64,),
    "scope.2.weight": (1, 64),
    "scope.2.bias": (1,),
    "version.0.weight": (64, 68),
    "version.0.bias": (64,),
    "version.2.weight": (1, 64),
    "version.2.bias": (1,),
    "identity_match.0.weight": (8, 2),
    "identity_match.0.bias": (8,),
    "identity.weight": (WIDTH, LEXICAL),
    "identity.bias": (64,),
    "kind.weight": (3, 64),
    "kind.bias": (3,),
    "intent.weight": (WIDTH, 2 * INPUT),
    "intent.bias": (64,),
    "identity_match.2.weight": (1, 8),
    "plan_encode.weight": (WIDTH, INPUT + 2),
    "plan_encode.bias": (64,),
    "plan_context.weight": (WIDTH, 4 * INPUT),
    "plan_context.bias": (64,),
    "plan_tags.weight": (5, 64),
    "plan_tags.bias": (5,),
    "plan_need.weight": (1, 64),
    "plan_need.bias": (1,),
    "plan_ref_query.weight": (64, 64),
    "plan_ref_query.bias": (64,),
    "plan_ref_encode.weight": (WIDTH, SOURCE_INPUT),
    "plan_ref_encode.bias": (64,),
    "plan_ref_pair.0.weight": (64, 128),
    "plan_ref_pair.0.bias": (64,),
    "plan_ref_pair.2.weight": (3, 64),
    "plan_ref_pair.2.bias": (3,),
    "read_update.weight": (QUERY, 4 * QUERY),
    "read_update.bias": (QUERY,),
    "plan_transition": (5, 5),
    "plan_start": (5,),
    "plan_end": (5,),
    "plan_origin.weight": (4, 64),
    "plan_origin.bias": (4,),
    "query_need.weight": (1, 64),
    "query_need.bias": (1,),
    "query_ref_query.weight": (64, 64),
    "query_ref_query.bias": (64,),
    "query_origin.weight": (4, 64),
    "query_origin.bias": (4,),
    "plan_local.weight": (64, 256),
    "plan_local.bias": (64,),
    "query_ref_encode.weight": (WIDTH, SOURCE_INPUT),
    "query_ref_encode.bias": (64,),
    "query_ref_pair.0.weight": (64, 128),
    "query_ref_pair.0.bias": (64,),
    "query_ref_pair.2.weight": (3, 64),
    "query_ref_pair.2.bias": (3,),
    "query_encode.weight": (WIDTH, INPUT + 2),
    "query_encode.bias": (WIDTH,),
    "identity_update.weight": (1, 4 * QUERY),
    "identity_update.bias": (1,),
    "identity_match.2.bias": (1,),
    "memory_attention_logits": (RELATION_HEADS,),
    "plan_boundary.0.weight": (WIDTH, 3 * WIDTH),
    "plan_boundary.0.bias": (WIDTH,),
    "plan_boundary.2.weight": (2, WIDTH),
    "plan_boundary.2.bias": (2,),
    "plan_boundary_raw.weight": (2, BOUNDARY_INPUT),
    "scope_query_encode.weight": (WIDTH, INPUT + 2),
    "scope_query_encode.bias": (WIDTH,),
    "scope_query_pool.weight": (RELATION_HEADS, WIDTH),
    "scope_memory_attention_logits": (RELATION_HEADS,),
    "plan_membership_raw.weight": (1, MEMBERSHIP_INPUT),
    "plan_source_context.0.weight": (SOURCE_CONTEXT_WIDTH, SOURCE_CONTEXT_INPUT),
    "plan_source_context.0.bias": (SOURCE_CONTEXT_WIDTH,),
    "plan_source_context.2.weight": (1, SOURCE_CONTEXT_WIDTH),
    "plan_source_context.2.bias": (1,),
    "kind_context.weight": (3, 2 * HIDDEN),
    "kind_context.bias": (3,),
    "read_context_memory.weight": (READ_CONTEXT_WIDTH, 2 * HIDDEN),
    "read_context_query.weight": (READ_CONTEXT_WIDTH, SEMANTIC),
    "read_context_delta.weight": (WIDTH, READ_CONTEXT_WIDTH),
    "query_context.weight": (WIDTH, 4 * INPUT),
}
PLAN_KEYS = tuple(k for k in KEYS if k.startswith("plan_"))
QUERY_KEYS = tuple(
    k for k in KEYS
    if k.startswith(("query_need.", "query_ref_query.", "query_origin.",
                     "query_ref_encode.", "query_ref_pair.", "query_context."))
)
SEQUENCE_KEYS = ("plan_transition", "plan_start", "plan_end")


def require_parameters(model, state):
    if set(state) != set(KEYS):
        raise ValueError("checkpoint must contain exactly the current memory tensors")
    if any(
        not isinstance(state[k], torch.Tensor)
        or tuple(state[k].shape) != TENSOR_SHAPES[k]
        or not torch.isfinite(state[k]).all()
        for k in KEYS
    ):
        raise ValueError("checkpoint tensor geometry or values are invalid")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint tensor geometry mismatch") from error


def load_parameters(model, state):
    require_parameters(model, state)
    return [], []


def observation_context(speaker=None, timestamp=None, past=""):
    """Literal provenance fields, never inferred subjects/actions. Match C."""
    return f"Speaker: {speaker or 'User'}\nObserved at: {timestamp or 'unspecified'}\n{past}"


def decode_plan(tags, units):
    """O / begin WRITE / begin UPDATE / begin DELETE / inside. Orphan I ignored."""
    if isinstance(tags, torch.Tensor):
        tags = tags.detach().cpu().tolist()
    result, first, action = [], None, None
    for i in range(len(units) + 1):
        tag = int(tags[i]) if i < len(units) else 0
        if first is not None and tag != 4:
            result.append((units[first][0], units[i - 1][1], action, first, i))
            first = None
        if tag in (1, 2, 3):
            first, action = i, tag
    return result


def word_features(text, item):
    """Structural source units, byte-identical to the C observation planner."""
    raw = text.encode("utf-8")
    units, pos = [], 0
    word = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'-"
    while pos < len(raw):
        c = raw[pos]
        if c in b" \n\r\t":
            pos += 1
            continue
        start = pos
        if c in word:
            while pos < len(raw) and raw[pos] in word:
                pos += 1
            while pos < len(raw) and raw[pos] in b".,!?;:":
                pos += 1
        else:
            width = (
                1
                if c < 128
                else (
                    2
                    if c & 224 == 192
                    else (3 if c & 240 == 224 else (4 if c & 248 == 240 else 0))
                )
            )
            if not width or pos + width > len(raw):
                raise ValueError("invalid UTF-8 source")
            pos += width
        units.append((start, pos))
    rows = []
    for a, b in units:
        weights = torch.tensor(
            [max(0, min(b, int(v)) - max(a, int(u))) for u, v in item["bounds"]],
            dtype=torch.float32,
        )
        if not weights.sum():
            raise ValueError("source unit has no C features")
        rows.append((item["x"] * (weights / weights.sum())[:, None]).sum(0))
    return torch.stack(rows), units


def identity_word_features(text, item):
    """Source word units for identity. Possessive punctuation is its own unit;
    entity choice remains the learned role head, never a string matcher."""
    raw = text.encode()
    pos = 0
    units = []
    word = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    while pos < len(raw):
        if raw[pos] in b" \n\r\t":
            pos += 1
            continue
        start = pos
        c = raw[pos]
        if c in word:
            while pos < len(raw) and raw[pos] in word:
                pos += 1
        else:
            width = (
                1
                if c < 128
                else (
                    2
                    if c & 224 == 192
                    else (3 if c & 240 == 224 else (4 if c & 248 == 240 else 0))
                )
            )
            if not width or pos + width > len(raw):
                raise ValueError("invalid UTF-8")
            pos += width
        if raw[start:pos] in [bytes([c]) for c in b".,!?;:'\""]:
            continue
        units.append((start, pos))
    rows = []
    for a, b in units:
        weights = torch.tensor(
            [max(0, min(b, int(v)) - max(a, int(u))) for u, v in item["bounds"]],
            dtype=torch.float32,
        )
        if not weights.sum():
            raise ValueError("unaddressed word")
        rows.append((item["x"] * (weights / weights.sum())[:, None]).sum(0))
    if not rows:
        return torch.zeros((1, INPUT)), [(0, 0)]
    return torch.stack(rows), units


def decode_spans(tags, units):
    if isinstance(tags, torch.Tensor):
        tags = tags.detach().cpu().tolist()
    spans, first = [], None
    for i in range(len(units) + 1):
        tag = int(tags[i]) if i < len(units) else 0
        if first is not None and tag in (0, 1):
            spans.append((units[first][0], units[i - 1][1]))
            first = None
        if tag and first is None:
            first = i
    return spans


class NeuralSlots(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(INPUT + 2, WIDTH)
        self.role = nn.Sequential(
            nn.Linear(HIDDEN + 2, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 1)
        )
        self.query_pool = nn.Linear(WIDTH, RELATION_HEADS, bias=False)
        self.read = nn.Sequential(
            nn.Linear(4 * SEMANTIC, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 1)
        )
        self.scope = nn.Sequential(
            nn.Linear(4 * SEMANTIC, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 1)
        )
        self.version = nn.Sequential(
            nn.Linear(WIDTH + 4, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 1)
        )
        self.identity_match = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        self.identity = nn.Linear(LEXICAL, WIDTH)
        self.kind = nn.Linear(WIDTH, 3)
        self.intent = nn.Linear(2 * INPUT, WIDTH)
        self.plan_encode = nn.Linear(INPUT + 2, WIDTH)
        self.plan_context = nn.Linear(4 * INPUT, WIDTH)
        self.plan_tags = nn.Linear(WIDTH, 5)
        self.plan_need = nn.Linear(WIDTH, 1)
        self.plan_ref_query = nn.Linear(WIDTH, WIDTH)
        self.plan_ref_encode = nn.Linear(SOURCE_INPUT, WIDTH)
        self.plan_ref_pair = nn.Sequential(
            nn.Linear(2 * WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 3)
        )
        self.read_update = nn.Linear(4 * QUERY, QUERY)
        nn.init.zeros_(self.read_update.weight)
        nn.init.constant_(self.read_update.bias, -4.0)
        self.plan_transition = nn.Parameter(torch.zeros(5, 5))
        self.plan_start = nn.Parameter(torch.zeros(5))
        self.plan_end = nn.Parameter(torch.zeros(5))
        self.plan_origin = nn.Linear(WIDTH, 4)
        nn.init.zeros_(self.plan_origin.weight)
        nn.init.zeros_(self.plan_origin.bias)
        self.query_need = nn.Linear(WIDTH, 1)
        self.query_ref_query = nn.Linear(WIDTH, WIDTH)
        self.query_origin = nn.Linear(WIDTH, 4)
        self.plan_local = nn.Linear(4 * WIDTH, WIDTH)
        nn.init.zeros_(self.plan_local.weight)
        nn.init.zeros_(self.plan_local.bias)
        self.query_ref_encode = nn.Linear(SOURCE_INPUT, WIDTH)
        self.query_ref_pair = nn.Sequential(
            nn.Linear(2 * WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 3)
        )
        self.query_need.load_state_dict(self.plan_need.state_dict())
        self.query_ref_query.load_state_dict(self.plan_ref_query.state_dict())
        self.query_origin.load_state_dict(self.plan_origin.state_dict())
        self.query_ref_encode.load_state_dict(self.plan_ref_encode.state_dict())
        self.query_ref_pair.load_state_dict(self.plan_ref_pair.state_dict())
        self.query_encode = nn.Linear(INPUT + 2, WIDTH)
        self.identity_update = nn.Linear(4 * QUERY, 1)
        nn.init.zeros_(self.identity_update.weight)
        nn.init.constant_(self.identity_update.bias, -4.0)
        self.memory_attention_logits = nn.Parameter(torch.zeros(RELATION_HEADS))
        self.plan_boundary = nn.Sequential(
            nn.Linear(3 * WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 2)
        )
        nn.init.zeros_(self.plan_boundary[2].weight)
        nn.init.zeros_(self.plan_boundary[2].bias)
        self.plan_boundary_raw = nn.Linear(BOUNDARY_INPUT, 2, bias=False)
        nn.init.zeros_(self.plan_boundary_raw.weight)
        self.scope_query_encode = nn.Linear(INPUT + 2, WIDTH)
        self.scope_query_pool = nn.Linear(WIDTH, RELATION_HEADS, bias=False)
        self.scope_memory_attention_logits = nn.Parameter(torch.zeros(RELATION_HEADS))
        self.plan_membership_raw = nn.Linear(MEMBERSHIP_INPUT, 1, bias=False)
        nn.init.zeros_(self.plan_membership_raw.weight)
        self.plan_source_context = nn.Sequential(
            nn.Linear(SOURCE_CONTEXT_INPUT, SOURCE_CONTEXT_WIDTH), nn.Tanh(),
            nn.Linear(SOURCE_CONTEXT_WIDTH, 1),
        )
        # Independent channel scales give unit-normalized raw hidden cues a
        # useful initial projection alongside the larger source representations.
        nn.init.normal_(self.plan_source_context[0].weight[:, :2 * WIDTH], std=.05)
        nn.init.normal_(self.plan_source_context[0].weight[:, 2 * WIDTH:], std=.5)
        nn.init.zeros_(self.plan_source_context[2].weight)
        nn.init.zeros_(self.plan_source_context[2].bias)
        self.kind_context = nn.Linear(2 * HIDDEN, 3)
        nn.init.zeros_(self.kind_context.weight)
        nn.init.zeros_(self.kind_context.bias)
        # Initialization consumes no source-model RNG. The memory projection
        # stores its input-width scale in the weights, identically to serving.
        with torch.random.fork_rng(devices=[]):
            self.read_context_memory = nn.Linear(2 * HIDDEN, READ_CONTEXT_WIDTH, bias=False)
            self.read_context_query = nn.Linear(SEMANTIC, READ_CONTEXT_WIDTH, bias=False)
            self.read_context_delta = nn.Linear(READ_CONTEXT_WIDTH, WIDTH, bias=False)
            with torch.no_grad():
                self.read_context_memory.weight.mul_(math.sqrt(2 * HIDDEN))
            nn.init.zeros_(self.read_context_delta.weight)
        with torch.random.fork_rng(devices=[]):
            self.query_context = nn.Linear(4 * INPUT, WIDTH, bias=False)
            nn.init.zeros_(self.query_context.weight)

    def memory_context_latents(self, contexts):
        """Project immutable original 1B hidden summaries into slot sidecars."""
        weight = self.read_context_memory.weight
        if (not isinstance(contexts, torch.Tensor) or not contexts.is_floating_point()
                or contexts.ndim not in (1, 2) or contexts.shape[-1] != 2 * HIDDEN
                or contexts.device != weight.device or contexts.dtype != weight.dtype
                or not torch.isfinite(contexts).all()):
            raise ValueError("memory context requires aligned finite original hidden summaries")
        return self.read_context_memory(contexts).tanh()

    @staticmethod
    def _validate_memory_context(latents, reference, slots):
        if (not isinstance(latents, torch.Tensor) or not latents.is_floating_point()
                or latents.shape != (slots, READ_CONTEXT_WIDTH)
                or latents.device != reference.device or latents.dtype != reference.dtype
                or not torch.isfinite(latents).all() or (latents.abs() > 1 + 1e-6).any()):
            raise ValueError("memory context latents require aligned finite 16-wide slot representations")

    def read_logits_from_pairs(self, pairs, queries, memory_context_latents, *, parameters=None):
        """Correct old matcher hidden states while retaining its token values."""
        self._validate_memory_context(memory_context_latents, queries, pairs.shape[1])
        if parameters is None:
            pre = self.read[0](pairs)
            weight, bias = self.read[2].weight, self.read[2].bias
        else:
            pre = F.linear(pairs, parameters["read.0.weight"], parameters["read.0.bias"])
            weight, bias = parameters["read.2.weight"], parameters["read.2.bias"]
        hidden = pre.tanh()
        old = F.linear(hidden, weight, bias).squeeze(-1)
        gate = self.read_context_query(queries[:, :SEMANTIC]).tanh()
        correction = self.read_context_delta(gate[:, None] * memory_context_latents[None])
        return old + F.linear((pre + correction).tanh() - hidden, weight).squeeze(-1)

    @staticmethod
    def kind_context_inputs(raw_features):
        """Original 1B hidden last/mean summary with one joint L2 scale."""
        if (not isinstance(raw_features, torch.Tensor)
                or not raw_features.is_floating_point()
                or raw_features.ndim != 2 or raw_features.shape[1] != INPUT
                or not len(raw_features) or len(raw_features) >= TOKEN_LIMIT
                or not torch.isfinite(raw_features).all()):
            raise ValueError("kind context requires finite original 1B token rows")
        hidden = raw_features[:, :HIDDEN]
        return F.normalize(torch.cat((hidden[-1], hidden.mean(0))), dim=0)

    def kind_logits_from_mean(self, means, contexts):
        """Preserve the learned 64-wide head and add original hidden context."""
        if (not isinstance(means, torch.Tensor)
                or not isinstance(contexts, torch.Tensor)
                or not means.is_floating_point() or not contexts.is_floating_point()
                or means.ndim not in (1, 2) or means.shape[-1] != WIDTH
                or contexts.shape != means.shape[:-1] + (2 * HIDDEN,)
                or contexts.device != means.device or contexts.dtype != means.dtype
                or not torch.isfinite(means).all() or not torch.isfinite(contexts).all()):
            raise ValueError("kind logits need aligned finite memory means and 1B hidden context")
        return self.kind(means) + self.kind_context(contexts)

    @staticmethod
    def augment(x, hidden_only=False):
        positions = torch.arange(len(x), device=x.device, dtype=x.dtype) / max(
            len(x) - 1, 1
        )
        lengths = torch.full_like(positions, np.log1p(len(x)) / 8)
        return torch.cat(
            (x[:, :HIDDEN] if hidden_only else x, positions[:, None], lengths[:, None]),
            -1,
        )

    def roles(self, words):
        return self.role(self.augment(words, hidden_only=True)).squeeze(-1)

    def prepare(self, features):
        tokens, words = features
        return self.encode(self.augment(tokens)).tanh(), self.reference_identity(words)

    def reference_identity(self, words):
        roles = self.roles(words)
        identity = F.normalize(
            (roles.softmax(0)[:, None] * words[:, HIDDEN:]).sum(0), dim=0
        )
        return torch.cat((identity, F.normalize(self.identity(identity), dim=0)))

    def pool_query(self, encoded, mask=None, *, scope=False):
        """Four independently learned attention pools over query tokens."""
        logits = (self.scope_query_pool if scope else self.query_pool)(encoded)
        if mask is not None:
            logits = logits.masked_fill(~mask[..., None], -torch.inf)
        weights = logits.softmax(-2)
        return torch.einsum("...th,...td->...hd", weights, encoded).flatten(-2)

    def query(self, features, *, scope=False):
        tokens, words = features
        encoder = self.scope_query_encode if scope else self.query_encode
        encoded = encoder(self.augment(tokens)).tanh()
        identity = self.reference_identity(words)
        pooled = self.pool_query(encoded, scope=scope)
        return torch.cat((pooled, self.intent(self.summary(features[0])))), identity

    def score(self, query, identity, memory, memory_identity, metadata, *, memory_context_latents, scope=False):
        intent = query[SEMANTIC:]
        relation = query[:SEMANTIC].reshape(RELATION_HEADS, WIDTH)
        attention = (memory @ relation.T * 0.125 * self.attention_scales(scope=scope)[None]).softmax(0)
        attended = attention.T @ memory
        pair = torch.cat((relation, attended, relation * attended, (relation - attended).abs()), -1).flatten()
        if not isinstance(memory_context_latents, torch.Tensor) or memory_context_latents.shape != (READ_CONTEXT_WIDTH,):
            raise ValueError("single-slot score requires a 16-wide memory context latent")
        self._validate_memory_context(memory_context_latents[None], query, 1)
        semantic = (self.scope(pair).squeeze() if scope else
                    self.read_logits_from_pairs(pair[None, None], query[None], memory_context_latents[None]).squeeze())
        entity = self.identity_logits(identity, memory_identity)
        probability = F.logsigmoid(semantic) + F.logsigmoid(entity)
        if not scope:
            current = self.version(torch.cat((intent, metadata))).squeeze()
            probability = (
                probability
                + F.logsigmoid(current)
                + (F.logsigmoid(intent[:3]) * metadata[:3]).sum()
            )
        probability = probability.clamp_max(-1e-6)
        return probability - torch.log(-torch.expm1(probability))

    @staticmethod
    def query_update_features(original, current):
        return torch.cat(
            (original, current, original * current, (original - current).abs()), -1
        )

    def advance_query(self, original, current):
        gate = self.read_update(self.query_update_features(original, current)).sigmoid()
        return (1 - gate) * original + gate * current

    def advance_identity(self, original_query, current_query, original_identity, current_identity):
        """Learned evidence update anchored to the original neural subject state."""
        gate = self.identity_update(self.query_update_features(original_query, current_query)).sigmoid()
        mixed = (1 - gate) * original_identity + gate * current_identity
        return torch.cat((F.normalize(mixed[..., :LEXICAL], dim=-1),
                          F.normalize(mixed[..., LEXICAL:], dim=-1)), -1)

    def identity_logits(self, queries, memories):
        if memories.ndim == 1:
            raw = (queries[..., :LEXICAL] * memories[:LEXICAL]).sum(-1)
            projected = (queries[..., LEXICAL:] * memories[LEXICAL:]).sum(-1)
        else:
            raw = queries[..., :LEXICAL] @ memories[:, :LEXICAL].T
            projected = queries[..., LEXICAL:] @ memories[:, LEXICAL:].T
        return self.identity_match(torch.stack((raw, projected), -1)).squeeze(-1)

    def attention_scales(self, *, scope=False):
        """Learned per-head concentration, bounded to keep native logits finite."""
        logits = self.scope_memory_attention_logits if scope else self.memory_attention_logits
        return (4 * (logits / 4).tanh()).exp()

    @staticmethod
    def summary(x):
        return F.normalize(torch.cat((x[-1], x.mean(0))), dim=0)

    @staticmethod
    def query_binding_context(current, context):
        """Retain each original current/context last-and-mean feature summary."""
        for features in (current, context):
            if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
                    or features.ndim != 2 or features.shape[1] != INPUT
                    or not 0 < len(features) < TOKEN_LIMIT):
                raise ValueError("query context requires finite original 1B token rows")
        if (current.device != context.device or current.dtype != context.dtype
                or not torch.isfinite(current).all() or not torch.isfinite(context).all()):
            raise ValueError("query context requires aligned finite original 1B token rows")
        return torch.cat((NeuralSlots.summary(current), NeuralSlots.summary(context)))

    def query_binding_candidate(self, candidate, raw_context):
        """Add a learned original-feature residual only to query binding cues."""
        weight = self.query_context.weight
        if (not isinstance(candidate, torch.Tensor) or not candidate.is_floating_point()
                or candidate.ndim not in (1, 2) or candidate.shape[-1] != WIDTH
                or not isinstance(raw_context, torch.Tensor) or not raw_context.is_floating_point()
                or raw_context.shape != candidate.shape[:-1] + (4 * INPUT,)
                or candidate.device != weight.device or candidate.dtype != weight.dtype
                or raw_context.device != weight.device or raw_context.dtype != weight.dtype
                or not torch.isfinite(candidate).all() or not torch.isfinite(raw_context).all()):
            raise ValueError("query binding requires aligned finite candidate and original context")
        return candidate + self.query_context(raw_context).tanh()

    def plan_encoding(self, current, context, body_words):
        """Full-context original-source representation, also retained for references."""
        condition = self.plan_context(
            torch.cat((self.summary(current), self.summary(context)))
        )
        return self._plan_encoding_from_condition(
            self.plan_encode(self.augment(body_words)), condition
        )

    def plan_body_encoding(self, current, body_words):
        """The same learned source encoder with no historical-summary contribution."""
        body_summary = self.summary(current)
        condition = self.plan_context(
            torch.cat((body_summary, torch.zeros_like(body_summary)))
        )
        return self._plan_encoding_from_condition(
            self.plan_encode(self.augment(body_words)), condition
        )

    def _plan_encoding_from_condition(self, projected, condition):
        encoded = (projected + condition).tanh()
        # Learned bidirectional word context. Zero padding denotes document
        # boundaries; neither punctuation nor words trigger semantic rules.
        padded = F.pad(encoded, (0, 0, 2, 2))
        neighbours = torch.cat(
            [
                padded[2 + offset : 2 + offset + len(encoded)]
                for offset in (-2, -1, 1, 2)
            ],
            -1,
        )
        return encoded + self.plan_local(neighbours).tanh()

    def plan(self, current, context, body_words):
        """Source decisions use a learned history residual; references retain full context."""
        encoded, _, logits, ends = self.plan_with_encodings(current, context, body_words)
        return encoded, logits, ends

    def plan_with_encodings(self, current, context, body_words):
        projected = self.plan_encode(self.augment(body_words))
        body_summary = self.summary(current)
        full_condition = self.plan_context(torch.cat((body_summary, self.summary(context))))
        body_condition = self.plan_context(torch.cat((body_summary, torch.zeros_like(body_summary))))
        encoded = self._plan_encoding_from_condition(projected, full_condition)
        body_encoded = self._plan_encoding_from_condition(projected, body_condition)
        boundary = self.plan_boundary_scores(encoded, body_words)
        logits = self.plan_tags(encoded)
        logits = logits + F.pad(boundary[:, :1].expand(-1, 3), (1, 1))
        membership = self.plan_total_membership(encoded, body_encoded, body_words)
        logits = logits + F.pad(membership[:, None].expand(-1, 4), (1, 0))
        return encoded, body_encoded, logits, boundary[:, 1]

    def plan_context_residual(self, full, body, words):
        return self.context_residual_from_parameters(full, body, words, {
            "plan_source_context.0.weight": self.plan_source_context[0].weight,
            "plan_source_context.0.bias": self.plan_source_context[0].bias,
            "plan_source_context.2.weight": self.plan_source_context[2].weight,
            "plan_source_context.2.bias": self.plan_source_context[2].bias,
        })

    @staticmethod
    def context_residual_from_parameters(full, body, words, parameters):
        if (full.ndim != 2 or full.shape[1] != WIDTH or body.shape != full.shape
                or words.shape != (len(full), INPUT)):
            raise ValueError("source context head requires aligned representations")
        hidden = F.linear(torch.cat((body, full - body, F.normalize(words[:, :HIDDEN], dim=-1)), -1),
                          parameters["plan_source_context.0.weight"],
                          parameters["plan_source_context.0.bias"]).tanh()
        return F.linear(hidden, parameters["plan_source_context.2.weight"],
                        parameters["plan_source_context.2.bias"]).squeeze(-1)

    def plan_total_membership(self, full, body, words):
        return self.plan_membership_scores(full, words) + self.plan_context_residual(full, body, words)

    @staticmethod
    def total_membership_from_parameters(full, body, words, parameters):
        return (NeuralSlots.membership_scores_from_parameters(full, words, parameters)
                + NeuralSlots.context_residual_from_parameters(full, body, words, parameters))

    def plan_membership_scores(self, encoded, words):
        """Condition source membership on the complete observation and raw word cues."""
        return self.membership_scores_from_parameters(encoded, words, {
            "plan_membership_raw.weight": self.plan_membership_raw.weight,
        })

    @staticmethod
    def membership_scores_from_parameters(encoded, words, parameters):
        if encoded.shape != (len(words), WIDTH):
            raise ValueError("membership inputs require aligned source representations")
        return F.linear(torch.cat((encoded, NeuralSlots.boundary_raw_inputs(words)), -1),
                        parameters["plan_membership_raw.weight"]).squeeze(-1)

    def plan_boundary_scores(self, encoded, words):
        """Raw 1B cues supplement frozen source summaries only for boundaries."""
        parameters = {
            "plan_boundary.0.weight": self.plan_boundary[0].weight,
            "plan_boundary.0.bias": self.plan_boundary[0].bias,
            "plan_boundary.2.weight": self.plan_boundary[2].weight,
            "plan_boundary.2.bias": self.plan_boundary[2].bias,
            "plan_boundary_raw.weight": self.plan_boundary_raw.weight,
        }
        return self.boundary_scores_from_parameters(encoded, words, parameters)

    @staticmethod
    def boundary_raw_inputs(words):
        """Full previous/current/next word cues; document edges are zero.

        Each word's hidden and lexical channels retain independent L2 scales.
        The fixed backbone rows are neither edited nor truncated to 64 values.
        """
        if words.ndim != 2 or words.shape[1] != INPUT or not len(words):
            raise ValueError("raw boundary inputs require source-word rows")
        raw = torch.cat((F.normalize(words[:, :HIDDEN], dim=-1),
                         F.normalize(words[:, HIDDEN:], dim=-1)), -1)
        padded = F.pad(raw, (0, 0, 1, 1))
        return torch.cat((padded[:-2], raw, padded[2:]), -1)

    @staticmethod
    def boundary_scores_from_parameters(encoded, words, parameters):
        """Shared live/immutable-teacher math, without registering extra weights."""
        if encoded.ndim != 2 or words.shape != (len(encoded), INPUT):
            raise ValueError("boundary inputs require aligned source-word rows")
        padded = F.pad(encoded, (0, 0, 1, 1))
        neighbours = torch.cat((padded[:-2], encoded, padded[2:]), -1)
        # Normalize context and lexical channels independently so neither the
        # hidden-state scale nor embedding scale hides the other input channel.
        raw = NeuralSlots.boundary_raw_inputs(words)
        hidden = F.linear(neighbours, parameters["plan_boundary.0.weight"],
                          parameters["plan_boundary.0.bias"]).tanh()
        previous = F.linear(hidden, parameters["plan_boundary.2.weight"],
                            parameters["plan_boundary.2.bias"])
        # Direct residual logits avoid the old tanh/output-weight range limit.
        return previous + F.linear(raw, parameters["plan_boundary_raw.weight"])

    @staticmethod
    def plan_boundary_potentials(ends):
        # Centering makes zero initialization exactly preserve the old lattice.
        return F.logsigmoid(-ends) + math.log(2), F.logsigmoid(ends) + math.log(2)

    def plan_edge_potentials(self, ends):
        _, transition = self.plan_potentials()
        continuing, stopping = self.plan_boundary_potentials(ends)
        previous_source = (torch.arange(5, device=ends.device) > 0)[:, None]
        inside = (torch.arange(5, device=ends.device) == 4)[None]
        dynamic = torch.where(previous_source[None],
                              torch.where(inside[None], continuing[:-1, None, None], stopping[:-1, None, None]),
                              ends.new_zeros(()))
        return transition[None] + dynamic

    def plan_terminal_potentials(self, ends):
        _, stopping = self.plan_boundary_potentials(ends)
        source = (torch.arange(5, device=ends.device) > 0).to(ends.dtype)
        return self.plan_end + source * stopping[-1]

    @torch.no_grad()
    def plan_labels(self, logits, ends):
        start, _ = self.plan_potentials()
        transitions = self.plan_edge_potentials(ends).cpu()
        terminal = self.plan_terminal_potentials(ends).cpu()
        logits = logits.detach().cpu()
        score = start.cpu() + logits[0]
        back = []
        for row, transition in zip(logits[1:], transitions):
            score, previous = (score[:, None] + transition).max(0)
            score = score + row
            back.append(previous)
        tag = (score + terminal).argmax()
        result = [tag]
        for previous in reversed(back):
            tag = previous[tag]
            result.append(tag)
        return torch.stack(list(reversed(result)))

    def plan_log_partition(self, logits, ends):
        """Parallel log-semiring reduction of the five-state source chain.

        Pairwise matrix composition retains order and the exact CRF grammar,
        while reducing accelerator launches from one per word to log2(words).
        """
        start, _ = self.plan_potentials()
        transitions = self.plan_edge_potentials(ends)
        terminal = self.plan_terminal_potentials(ends)
        return self._plan_partition(logits, start, transitions, terminal)

    @staticmethod
    def _plan_partition(logits, start, transitions, terminal):
        alpha = start + logits[0]
        blocks = transitions + logits[1:, None, :]
        while len(blocks) > 1:
            pairs = len(blocks) // 2
            merged = torch.logsumexp(
                blocks[:2 * pairs:2, :, :, None]
                + blocks[1:2 * pairs:2, None, :, :], dim=-2
            )
            blocks = torch.cat((merged, blocks[-1:]), 0) if len(blocks) % 2 else merged
        if len(blocks):
            alpha = torch.logsumexp(alpha[:, None] + blocks[0], dim=0)
        return torch.logsumexp(alpha + terminal, dim=0)

    def sequence_loss(self, logits, target, ends):
        start, _ = self.plan_potentials()
        transitions = self.plan_edge_potentials(ends)
        terminal = self.plan_terminal_potentials(ends)
        partition = self._plan_partition(logits, start, transitions, terminal)
        chosen = (
            logits.gather(1, target[:, None]).sum()
            + start[target[0]]
            + terminal[target[-1]]
        )
        if len(target) > 1:
            chosen = chosen + transitions[torch.arange(len(target) - 1, device=target.device), target[:-1], target[1:]].sum()
        return (partition - chosen) / len(target)

    def plan_potentials(self):
        # Source grammar: an inside tag requires a begin or previous inside.
        # This constrains the representation only; actions/spans remain learned.
        start = self.plan_start.masked_fill(
            torch.arange(5, device=self.plan_start.device) == 4, -torch.inf
        )
        mask = torch.zeros((5, 5), device=self.plan_transition.device, dtype=torch.bool)
        mask[0, 4] = True
        return start, self.plan_transition.masked_fill(mask, -torch.inf)

    def plan_reference(self, candidate, source_words, source_types=None, need_candidate=None):
        return self._reference(candidate, source_words, source_types, False, need_candidate)

    def query_reference(self, candidate, source_words, source_types, *, raw_context):
        corrected = self.query_binding_candidate(candidate, raw_context)
        return self._reference(corrected, source_words, source_types, True, corrected)

    def _reference(self, candidate, source_words, source_types, query_binding, need_candidate=None):
        need_candidate = candidate if need_candidate is None else need_candidate
        need_head = self.query_need if query_binding else self.plan_need
        query_head = self.query_ref_query if query_binding else self.plan_ref_query
        origin_head = self.query_origin if query_binding else self.plan_origin
        encoder = self.query_ref_encode if query_binding else self.plan_ref_encode
        pair_head = self.query_ref_pair if query_binding else self.plan_ref_pair
        if source_types is None or source_types.shape != (len(source_words), 4):
            raise ValueError("reference selection requires literal source locations")
        inputs = torch.cat(
            (self.augment(source_words, hidden_only=True), source_types), -1
        )
        encoded = encoder(inputs).tanh()
        cue = query_head(candidate).tanh()[None].expand_as(encoded)
        tags = pair_head(torch.cat((cue, encoded), -1))
        origin = source_types @ origin_head(candidate)
        tags = tags + torch.stack((torch.zeros_like(origin), origin, origin), -1)
        return need_head(need_candidate).squeeze(), tags

    def semantic_bank_logits(self, queries, memories, mask, *, memory_context_latents, scope=False):
        """Learned relation logits with the same full-bank attention as serving."""
        self._validate_memory_context(memory_context_latents, queries, len(memories))
        q = queries[:, :SEMANTIC].reshape(len(queries), RELATION_HEADS, WIDTH)
        attention = torch.einsum("qhd,ntd->qnht", q, memories) * 0.125
        attention = attention * self.attention_scales(scope=scope)[None, None, :, None]
        attention = attention.masked_fill(~mask[None, :, None], -torch.inf).softmax(-1)
        attended = torch.einsum("qnht,ntd->qnhd", attention, memories)
        cue = q[:, None].expand_as(attended)
        pairs = torch.cat((cue, attended, cue * attended, (cue - attended).abs()), -1).flatten(-2)
        return (self.scope(pairs).squeeze(-1) if scope else
                self.read_logits_from_pairs(pairs, queries, memory_context_latents))

    def score_bank(
        self,
        queries,
        identities,
        memories,
        memory_identities,
        metadata,
        mask,
        *,
        memory_context_latents,
        scope=False,
    ):
        """Exact vectorization of score over every slot; padding is masked only.

        queries [Q,QUERY], memories [N,T,WIDTH], mask [N,T]. No ranking/candidate
        stage is introduced. Supports ordinary [N,4] and dynamic [Q,N,4] state.
        """
        semantic = self.semantic_bank_logits(queries, memories, mask,
                                             memory_context_latents=memory_context_latents, scope=scope)
        return self.score_bank_from_semantic(semantic, queries, identities,
                                             memory_identities, metadata, scope=scope)

    def score_bank_from_semantic(self, semantic, queries, identities,
                                 memory_identities, metadata, *, scope=False):
        """Apply unchanged learned identity/time gates to live or parent logits."""
        entity = self.identity_logits(identities, memory_identities)
        probability = F.logsigmoid(semantic) + F.logsigmoid(entity)
        if not scope:
            if metadata.ndim == 2:
                metadata = metadata[None].expand(len(queries), -1, -1)
            intent = queries[:, None, SEMANTIC:].expand(-1, semantic.shape[1], -1)
            current = self.version(torch.cat((intent, metadata), -1)).squeeze(-1)
            probability = (
                probability
                + F.logsigmoid(current)
                + (F.logsigmoid(intent[:, :, :3]) * metadata[:, :, :3]).sum(-1)
            )
        probability = probability.clamp_max(-1e-6)
        return probability - torch.log(-torch.expm1(probability))


def pack_memories(prepared):
    memories = torch.nn.utils.rnn.pad_sequence(
        [m for m, _ in prepared], batch_first=True
    )
    identities = torch.stack([i for _, i in prepared])
    lengths = torch.tensor([len(m) for m, _ in prepared], device=memories.device)
    mask = (
        torch.arange(memories.shape[1], device=memories.device)[None] < lengths[:, None]
    )
    return memories, identities, mask


def export(model, backbone_sha, path):
    state = model.cpu().state_dict()
    if set(state) != set(KEYS):
        raise ValueError("unexpected neural memory tensors")
    if any(
        tuple(state[k].shape) != TENSOR_SHAPES[k] or not torch.isfinite(state[k]).all()
        for k in KEYS
    ):
        raise ValueError("invalid model tensor geometry or values")
    payload = [
        np.asarray(state[k].detach().float().numpy(), dtype="<f4").tobytes()
        for k in KEYS
    ]
    header = (
        b"BNSMODEL"
        + struct.pack("<4I", 1, INPUT, WIDTH, len(KEYS))
        + bytes.fromhex(backbone_sha)
    )
    header += b"".join(struct.pack("<2I", len(p), zlib.crc32(p)) for p in payload)
    Path(path).write_bytes(header + b"".join(payload))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
