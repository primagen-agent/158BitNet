"""Learned context over immutable 1B memory-record features.

These operators support component training and validation. The native memory
model remains separately qualified before any serving parameter change.
C feature math supports float32 and float64 precision. Contextual states
are per-operation values: never persist them in records,
otherwise a deleted record could survive inside another record's state.
"""

import torch
from torch import nn
from torch.nn import functional as F


INPUT = 4096
HIDDEN = 2048
WIDTH = 64
TOKEN_TYPES = {
    "body": 0, "reference": 1, "time": 2, "status": 3,
    "speaker": 4, "source": 5, "context": 6, "other_literal": 7,
}
STATES = {"current": 0, "superseded": 1, "deleted": 2, "tentative": 3, "unknown": 4}


class _AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.feed_norm = nn.LayerNorm(width)
        self.feed = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x, valid):
        b, n, width = x.shape
        qkv = self.qkv(self.attention_norm(x)).reshape(b, n, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # SDPA bool masks use True for keys that may be attended to. Every
        # sequence has a valid query header, so no row is entirely masked.
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(b, n, width)
        x = x + self.output(attended)
        x = x + self.feed(self.feed_norm(x))
        return x.masked_fill(~valid[..., None], 0.0)


class BankContextOperator(nn.Module):
    """Two-block width-64 neural record decisions, sharing all parameters.

    ``forward(query_features, bank_features, query_mask, bank_mask, ...)``
    takes [B,Q,4096], [B,R,T,4096], [B,Q], [B,R,T]. Masks are bool with True
    for observed tokens; padding is ignored even when its features are NaN.
    Types have shapes [B,Q] and [B,R,T], states [B,R]. Types default to body
    and states to unknown. Every query has at least one real token. R may be
    zero. A record is visible iff its token mask contains at least one True
    and its physical state is not deleted (C's NS_DELETED=2). Deleted rows
    keep their physical address but are excluded before feature validation,
    encoding and attention, and return zero contexts/logits.

    Returns edit/read logits [B,R], record mask [B,R], final query context
    [B,64], record context [B,R,64], and paired query context [B,R,64]. Padded
    records return zero logits and must be excluded by record_mask in loss
    and metrics. No sigmoid or hard threshold is applied. The edit head
    selects an existing record's mutation scope; the read head selects
    evidence. Writer action and transaction validation remain external.

    Bank order has no embedding: permuting records permutes their logits.
    Positions/counts depend only on observed tokens within each input stream,
    never padded dimensions. Full attention costs O((Q+R*T)^2); independent
    attention costs O(R*(Q+T)^2), with the identical learned parameter count.
    """

    def __init__(self, *, width=WIDTH, heads=4):
        super().__init__()
        if width != WIDTH or width % heads:
            raise ValueError("memory context uses width 64, divisible by attention heads")
        self.width = width
        self.token_projection = nn.Linear(INPUT + 2, width)
        self.token_type = nn.Embedding(len(TOKEN_TYPES), width)
        self.stream_type = nn.Embedding(2, width)
        self.record_state = nn.Embedding(len(STATES), width)
        self.query_header = nn.Parameter(torch.empty(width))
        self.record_header = nn.Parameter(torch.empty(width))
        self.header_pool = nn.Linear(width, 1, bias=False)
        self.header_projection = nn.Linear(width, width)
        self.blocks = nn.ModuleList([_AttentionBlock(width, heads) for _ in range(2)])
        self.final_norm = nn.LayerNorm(width)
        self.decision = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 2))
        for embedding in (self.token_type, self.stream_type, self.record_state):
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.normal_(self.query_header, std=0.02)
        nn.init.normal_(self.record_header, std=0.02)

    @staticmethod
    def _mask(mask, shape, device, name):
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape) or mask.device != device:
            raise ValueError(f"{name} must be aligned bool tensor {tuple(shape)}")

    @staticmethod
    def _codes(codes, shape, limit, default, device, name, valid=None):
        if codes is None:
            return torch.full(shape, default, device=device, dtype=torch.long)
        if not isinstance(codes, torch.Tensor) or codes.dtype != torch.long or tuple(codes.shape) != tuple(shape) or codes.device != device:
            raise ValueError(f"{name} must be aligned int64 tensor {tuple(shape)}")
        active = codes if valid is None else codes[valid]
        if bool(((active < 0) | (active >= limit)).any()):
            raise ValueError(f"{name} codes must be in [0,{limit})")
        return codes if valid is None else codes.masked_fill(~valid, default)

    def _encode(self, features, mask, types, stream):
        x = features.masked_fill(~mask[..., None], 0.0)
        # C already normalizes both halves; re-normalization preserves that
        # contract while avoiding relative hidden/lexical scale shortcuts.
        x = torch.cat((F.normalize(x[..., :HIDDEN], dim=-1), F.normalize(x[..., HIDDEN:], dim=-1)), -1)
        count = mask.sum(-1, keepdim=True).to(x.dtype)
        ordinal = (mask.cumsum(-1) - 1).clamp_min(0).to(x.dtype)
        position = ordinal / (count - 1).clamp_min(1)
        length = torch.log1p(count).expand_as(position) / 8.0
        x = self.token_projection(torch.cat((x, position[..., None], length[..., None]), -1))
        x = x + self.token_type(types) + self.stream_type.weight[stream]
        return x.masked_fill(~mask[..., None], 0.0)

    def _transform(self, sequence, valid):
        for block in self.blocks:
            sequence = block(sequence, valid)
        return self.final_norm(sequence).masked_fill(~valid[..., None], 0.0)

    def _local_header(self, encoded, mask):
        # Nonlinearity must follow the combination of content, type and
        # position: otherwise pooling can erase their binding. This local
        # operation sees only this stream in both visibility modes.
        mixed = F.gelu(encoded)
        scores = self.header_pool(mixed).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(-1).masked_fill(~mask, 0.0)
        pooled = (weights[..., None] * mixed).sum(-2)
        return self.header_projection(pooled)

    def forward(self, query_features, bank_features, query_mask, bank_mask,
                query_token_types=None, bank_token_types=None, record_states=None, *, mode="full"):
        if mode not in ("full", "independent"):
            raise ValueError("mode must be full or independent")
        q, m = query_features, bank_features
        if (not isinstance(q, torch.Tensor) or not isinstance(m, torch.Tensor)
                or not q.is_floating_point() or not m.is_floating_point()
                or q.ndim != 3 or m.ndim != 4 or q.shape[-1] != INPUT or m.shape[-1] != INPUT
                or q.shape[0] != m.shape[0] or not q.shape[0] or not q.shape[1]
                or q.device != m.device or q.dtype != m.dtype
                or q.device != self.query_header.device or q.dtype != self.query_header.dtype):
            raise ValueError("aligned floating 1B C features must be [B,Q,4096], [B,R,T,4096] on model device/dtype")
        if q.dtype not in (torch.float32, torch.float64):
            raise ValueError("memory context supports float32/float64 precision only")
        b, nq, _ = q.shape
        _, nr, nt, _ = m.shape
        self._mask(query_mask, (b, nq), q.device, "query_mask")
        self._mask(bank_mask, (b, nr, nt), q.device, "bank_mask")
        states = self._codes(record_states, (b, nr), len(STATES), STATES["unknown"], q.device,
                             "record_states", bank_mask.any(-1))
        # Deletion is literal physical bank state. Its old content must not
        # reach live records or even feature validation; preserve address rows
        # rather than compacting the caller's bank.
        bank_mask = bank_mask & (states != STATES["deleted"])[..., None]
        if not bool(query_mask.any(-1).all()):
            raise ValueError("each query must contain an observed token")
        if not bool(torch.isfinite(q[query_mask]).all()) or not bool(torch.isfinite(m[bank_mask]).all()):
            raise ValueError("observed C feature rows must be finite")
        record_mask = bank_mask.any(-1)
        qt = self._codes(query_token_types, (b, nq), len(TOKEN_TYPES), 0, q.device, "query_token_types", query_mask)
        mt = self._codes(bank_token_types, (b, nr, nt), len(TOKEN_TYPES), 0, q.device, "bank_token_types", bank_mask)
        qe = self._encode(q, query_mask, qt, 0)
        me = self._encode(m, bank_mask, mt, 1)
        qh = (self.query_header + self.stream_type.weight[0] + self._local_header(qe, query_mask))[:, None]
        mh = self.record_header + self.stream_type.weight[1] + self.record_state(states) + self._local_header(me, bank_mask)
        qs = torch.cat((qh, qe), 1)
        ms = torch.cat((mh[:, :, None], me), 2)
        qvalid = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=q.device), query_mask), 1)
        mvalid = torch.cat((record_mask[:, :, None], bank_mask), 2)
        if mode == "full" or nr == 0:
            sequence = torch.cat((qs, ms.reshape(b, nr * (nt + 1), self.width)), 1)
            valid = torch.cat((qvalid, mvalid.reshape(b, nr * (nt + 1))), 1)
            h = self._transform(sequence, valid)
            query_context = h[:, 0]
            records = h[:, nq + 1:].reshape(b, nr, nt + 1, self.width)[:, :, 0]
            paired = query_context[:, None].expand(-1, nr, -1)
        else:
            pair_query = qs[:, None].expand(-1, nr, -1, -1)
            pair_qvalid = qvalid[:, None].expand(-1, nr, -1)
            sequence = torch.cat((pair_query, ms), 2).reshape(b * nr, nq + nt + 2, self.width)
            valid = torch.cat((pair_qvalid, mvalid), 2).reshape(b * nr, nq + nt + 2)
            h = self._transform(sequence, valid).reshape(b, nr, nq + nt + 2, self.width)
            paired, records = h[:, :, 0], h[:, :, nq + 1]
            # Aggregate is diagnostic only; decisions below use each pair's
            # own query context and never this cross-record mean.
            query_context = (paired * record_mask[..., None]).sum(1) / record_mask.sum(1, keepdim=True).clamp_min(1)
        pair = torch.cat((paired, records, paired * records, (paired - records).abs()), -1)
        logits = self.decision(pair).masked_fill(~record_mask[..., None], 0.0)
        return {
            "edit_logits": logits[..., 0], "read_logits": logits[..., 1], "record_mask": record_mask,
            "query_context": query_context, "record_context": records.masked_fill(~record_mask[..., None], 0.0),
            "paired_query_context": paired.masked_fill(~record_mask[..., None], 0.0),
        }


class _QueryRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8192, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, summary):
        return self.net(summary).softmax(-1)


class FactorizedBankContext(nn.Module):
    """Frozen source/provenance experts plus the learned query probability router.

    forward(q, source_bank, provenance_bank, qmask, smask, pmask,
            record_states=None)

    Shapes are q[B,Q,4096], source_bank[B,R,S,4096], provenance_bank[B,R,P,4096]
    and aligned bool token masks. Both bank views must address the same R
    records; token counts may differ. C physical states [B,R] are passed to
    each expert, where NS_DELETED2 is excluded before validation/attention.
    Whole independently encoded literal streams include their own fields;
    token role/type guesses are neither required nor accepted.

    Returns edit/read logits[B,R], weights[B,2] ordered source/provenance,
    record_mask[B,R], plus source_logits/provenance_logits[B,R,2] ordered
    edit/read for inspection. Fusion is logit(clamp(sum(weights*sigmoid(
    expert_logits)),1e-6,1-1e-6)), applied separately to both heads with the
    same query-derived weights. Padded/deleted outputs are zero. All expert
    bank context is ephemeral and is never written back into record storage.
    """

    def __init__(self):
        super().__init__()
        self.source = BankContextOperator()
        self.provenance = BankContextOperator()
        self.router = _QueryRouter()
        self.source.requires_grad_(False)
        self.provenance.requires_grad_(False)

    @staticmethod
    def query_summary(features, mask):
        """Exactly retain each stream's normalized last and mean C rows.

        Compact only physical padding so reductions match the original router
        on valid C row sequences, including masks containing internal holes.
        Expert input validation runs before this helper in normal forward.
        """
        return torch.stack([
            torch.cat((F.normalize(rows[valid][-1], dim=0), F.normalize(rows[valid].mean(0), dim=0)))
            for rows, valid in zip(features, mask)
        ])

    @staticmethod
    def probability_mix(experts, weights):
        """Mix the final two expert axes without allowing logit-scale vetoes."""
        if experts.dtype not in (torch.float32, torch.float64) or weights.dtype not in (torch.float32, torch.float64):
            raise ValueError("probability mixing supports float32/float64 precision only")
        broadcast = (weights.shape[0],) + (1,) * (experts.ndim - 2) + (2,)
        probability = (experts.sigmoid() * weights.reshape(broadcast)).sum(-1)
        return torch.logit(probability.clamp(1e-6, 1. - 1e-6))

    def forward(self, query_features, source_features, provenance_features,
                query_mask, source_mask, provenance_mask, record_states=None):
        source = self.source(query_features, source_features, query_mask, source_mask,
                             record_states=record_states, mode="full")
        provenance = self.provenance(query_features, provenance_features, query_mask, provenance_mask,
                                     record_states=record_states, mode="independent")
        if not torch.equal(source["record_mask"], provenance["record_mask"]):
            raise ValueError("source/provenance views must address the same live physical records")
        valid = source["record_mask"]
        source_logits = torch.stack((source["edit_logits"], source["read_logits"]), -1)
        provenance_logits = torch.stack((provenance["edit_logits"], provenance["read_logits"]), -1)
        expert_logits = torch.stack((source_logits, provenance_logits), -1)
        weights = self.router(self.query_summary(query_features, query_mask))
        logits = self.probability_mix(expert_logits, weights).masked_fill(~valid[..., None], 0.)
        return {
            "edit_logits": logits[..., 0], "read_logits": logits[..., 1],
            "weights": weights, "record_mask": valid,
            "source_logits": source_logits, "provenance_logits": provenance_logits,
        }
