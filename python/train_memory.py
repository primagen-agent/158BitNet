"""train_memory.py -- Metis memory trainer (32 layers, GQA
reads through frozen backbone q_proj/o_proj).

Mirrors the C runtime (src/metis/metis_file.{h,c}) EXACTLY:
- One memory (M[kv,kv], S[kv]) per backbone layer.
- Read (active tokens, every memory layer): q = frozen backbone q_proj @
  h_normed (PRE-RoPE); per-head L2 normalize (head_dim); groups of
  q_dim/kv_dim heads read the same per-layer M: out_g = (q~ M)/(q~ S),
  NO +1, |q~ S| < 1e-8 -> zeros; concat groups -> [q_dim]; mem_norm =
  RMSNorm(out * mem_norm_w, q_dim, eps 1e-6); fused = frozen bb o_proj @
  mem_normed; attn' = gamma*attn + (1-gamma)*fused.
  (The C side pushes the mem vector through the i8-quantized TQ2 o_proj
  kernel; torch uses the dequantized bf16 o_proj — same accepted deviation
  class as the Phase-A G2 parity, which agreed at cos 0.9997.)
- Write (per exchange chunk, per layer): double-norm with the layer's own
  attn_norm weight, w_agg scores, softmax/tau, AlphaTopP(rho, k_min)
  inclusive-crossing hard select, K = L2normalize(h @ Wk)/sqrt(kv),
  V = h @ Wv (SINGLE-layer projections [kv, d]),
  alpha = sum w sigma(gdu_aw h + gdu_ab) / sum w,
  beta_i = beta_scale sigma(gdu_bw h + gdu_bb),
  GDU: M <- alpha M + sum outer(k, beta (V - alpha (k M_old))); km/ks
  PRE-decay; S <- S + sum k beta (1 - alpha (k S_old)); S never scaled
  by alpha.
- Trainable per layer: Wk, Wv [kv, d], w_agg/gdu_aw/gdu_bw [d],
  mem_norm [q_dim]. Init: Wk/Wv = backbone k/v projections (exact row
  copy, kv_dim matches); w_agg/gdu_* zeros; mem_norm ones; gdu_ab=gdu_bb=
  logit_clamped(1.0)=9.2103 (reference recipe); beta_scale 0.9.
- Export: .bnmem v3 (BNMEM3) via bnmem_v3_export.save_bnmem_v3.

Training protocol (reference-aligned): exchange chunks evaluated as
full-prefix replays with fusion/capture gated to the current chunk
(fuse_start); commit after each chunk; QUERY from a FRESH context
(memory is the only fact source — v3 fix kept). Loss = NLL on query
assistant tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ggw import GGUFWeights
from torch_backbone import TorchBackbone, rms_norm
from train_data import CTokenizer, load_dataset, dataset_order, render_chunk
from bnmem_export import save_bnmem_v3

# Reference gate-bias init (GatedDeltaRuleMixin._logit_clamped): p=1.0
# clamped to 1-1e-4, then log(p/(1-p)) = log(9999) ~ +9.2102 (sigmoid -> ~1).
# (An earlier sign-flipped version produced -9.21 -> beta ~ 0 -> empty M/S.)
LOGIT_CLAMPED_1 = math.log((1.0 - 1e-4) / 1e-4)        # ~ +9.2102


class MetisMemory(nn.Module):
    """32-layer memory module; frozen projections borrowed from the
    backbone (q_proj pre-RoPE read side, o_proj fuse side)."""

    def __init__(self, backbone: TorchBackbone, layer_ids, gamma=0.9,
                 tau=1.0, rho=0.9, k_min=1, beta_scale=0.9,
                 device="cuda", dtype=torch.float32, seed=42):
        super().__init__()
        cfg = backbone.cfg
        self.backbone = backbone
        self.layer_ids = list(layer_ids)
        self.n_layers = len(self.layer_ids)
        self.d_model = cfg.hidden
        self.kv_dim = cfg.kv_dim
        self.q_dim = cfg.q_dim
        self.head_dim = cfg.head_dim
        self.groups = cfg.q_dim // cfg.kv_dim
        self.heads_per_group = cfg.n_heads // self.groups
        self.gamma, self.tau, self.rho, self.k_min = gamma, tau, rho, k_min
        self.beta_scale = beta_scale
        # v8: per-layer TRAINABLE gate biases (v9's trained values sit near
        # sigmoid~0.49/0.5, i.e. logits ~0 — NOT the +9.2 clamp we froze
        # at; the clamp pins alpha≈1.0/beta≈1.0 leaving the gates inert).
        # Init at logit 0 (sigmoid 0.5) like a fresh trained-ish start.
        NL = self.n_layers
        self.gdu_ab = nn.Parameter(torch.zeros(NL, device=device, dtype=dtype))
        self.gdu_bb = nn.Parameter(torch.zeros(NL, device=device, dtype=dtype))
        self.device, self.dtype = device, dtype

        g = torch.Generator(device="cpu").manual_seed(seed)
        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=dtype)
        # trainable
        self.wk = nn.Parameter(zeros(NL, self.kv_dim, self.d_model))
        self.wv = nn.Parameter(zeros(NL, self.kv_dim, self.d_model))
        self.w_agg = nn.Parameter(zeros(NL, self.d_model))
        self.gdu_aw = nn.Parameter(zeros(NL, self.d_model))
        self.gdu_bw = nn.Parameter(zeros(NL, self.d_model))
        self.mem_norm = nn.Parameter(torch.ones(NL, self.q_dim,
                                                device=device, dtype=dtype))
        # v8: TRAINED query path (v4 semantics — the reference trains a
        # low-rank query_proj on the raw residual; low-rank is also a
        # 12GB-necessity: full-rank [32,4096,2560] fp32 + AdamW states =
        # 5.4GB alone. Rank-128 factors: q = (h @ B^T) @ A^T with
        # A [q_dim, r], B [r, d_model]; init from SVD of the backbone
        # q_proj (top-128 singular directions — the reference's
        # svd128-derived init equivalent).
        RANK = 128
        self.q_rank = RANK
        self.query_a = nn.Parameter(zeros(NL, self.q_dim, RANK))
        self.query_b = nn.Parameter(zeros(NL, RANK, self.d_model))
        self.query_norm = nn.Parameter(
            torch.ones(NL, self.head_dim, device=device, dtype=dtype))
        # frozen backbone tensors we borrow every forward (registered as
        # buffers-less plain attrs: they live on the backbone, requires_grad
        # False; keeping them out of state_dict avoids double storage)
        self.attn_norm_w = torch.stack([
            backbone.layers[i]["attn_norm"] for i in self.layer_ids
        ]).to(dtype)                                            # [NL, d] fp32-ish
        self.rms_eps = cfg.rms_eps

        # runtime state (fresh tensors per sample)
        self.M = None   # [NL, kv, kv]
        self.S = None   # [NL, kv]
        self.active = False
        self._captured = None    # list of per-layer [T, d] tensors

    def init_from_backbone(self):
        """Wk/Wv <- backbone k/v projections (exact copy; the reference's
        _fit_rows needs no tiling when kv_dim matches, which holds here).
        v8: query low-rank factors <- SVD-128 of the backbone q_proj
        (q ≈ A @ B reproduces the top singular subspace at init)."""
        with torch.no_grad():
            for s, blk in enumerate(self.layer_ids):
                lw = self.backbone.layers[blk]
                self.wk[s].copy_(lw["k"].to(dtype=self.wk.dtype))
                self.wv[s].copy_(lw["v"].to(dtype=self.wv.dtype))
                q = lw["q"].to(dtype=torch.float32)         # [q_dim, d]
                U, Sig, Vh = torch.linalg.svd(q, full_matrices=False)
                r = self.q_rank
                self.query_a[s].copy_(U[:, :r] * Sig[:r].unsqueeze(0))
                self.query_b[s].copy_(Vh[:r, :])

    # ---- state control ----
    def reset_state(self):
        # drop the old tensors FIRST (frees any autograd graph still attached
        # to M/S before allocating the fresh zeros -- on a fragmented GPU the
        # zeros allocation itself can OOM if the old graph is still resident)
        self.M = None
        self.S = None
        self.M = torch.zeros(self.n_layers, self.kv_dim, self.kv_dim,
                             device=self.device, dtype=self.dtype)
        self.S = torch.zeros(self.n_layers, self.kv_dim,
                             device=self.device, dtype=self.dtype)
        self.active = False
        self._captured = None

    def slot_of(self, blk):
        """Backbone block index -> memory slot (None when not a memory
        layer; all 32 are memory layers but keep the general shape)."""
        try:
            return self._slot_map[blk]
        except AttributeError:
            self._slot_map = {b: s for s, b in enumerate(self.layer_ids)}
            return self._slot_map.get(blk)

    def capture(self, per_layer_hn):
        """per_layer_hn: list of NL tensors-or-None ([T, d] attn-normed rows
        of the current chunk at each memory layer; the backbone fills every
        slot on each call). Accumulates per layer until a commit consumes it
        (C1 semantics). Rows keep their autograd graph so write params get
        credit through the state they leave."""
        if self._captured is None:
            self._captured = list(per_layer_hn)
        else:
            self._captured = [
                None if (a is None and b is None)
                else (a if b is None else (b if a is None
                                           else torch.cat([a, b], dim=0)))
                for a, b in zip(self._captured, per_layer_hn)
            ]

    def take_captured(self):
        return self._captured

    def discard_captured(self):
        self._captured = None

    def commit_all(self):
        """Commit every layer's captured rows (no-grad path, e.g. valid).
        Batched like commit_all_grad_enabled: one stack, not 32."""
        caps = self._captured
        eps = self.rms_eps
        ran = False
        if caps is not None:
            M_parts = list(self.M)
            S_parts = list(self.S)
            with torch.no_grad():
                for slot, rows in enumerate(caps):
                    if rows is not None and rows.shape[0] > 0:
                        nm, ns = self._commit_math(slot, rows.float(), eps)
                        M_parts[slot] = nm
                        S_parts[slot] = ns
                        ran = True
                if ran:
                    self.M = torch.stack(M_parts, dim=0)
                    self.S = torch.stack(S_parts, dim=0)
        self.discard_captured()
        if ran:
            self.active = True
        return ran

    def commit_all_grad_enabled(self):
        """Commit ALL slots from captured rows in ONE batched update.

        Memory-critical design (12GB rule): the naive loop called commit()
        per slot, and each commit() stacked the full [32,kv,kv] M — 32
        stacks per chunk each retaining every slot = 537MB/chunk of graph.
        The batched form computes every slot's new_M/new_S first (graphs on
        per-slot math only), then does ONE stack. Retained per chunk:
        the per-slot math tensors + one 16.7MB stack output.
        """
        caps = self._captured
        eps = self.rms_eps
        ran = False
        if caps is not None:
            new_M_parts = []
            new_S_parts = []
            for slot, rows in enumerate(caps):
                if rows is not None and rows.shape[0] > 0:
                    with torch.enable_grad():
                        nm, ns = self._commit_math(slot,
                                                   rows.detach().float(), eps)
                    new_M_parts.append((slot, nm))
                    new_S_parts.append((slot, ns))
                    ran = True
            if ran:
                M_parts = list(self.M)
                S_parts = list(self.S)
                for slot, nm in new_M_parts:
                    M_parts[slot] = nm
                for slot, ns in new_S_parts:
                    S_parts[slot] = ns
                with torch.enable_grad():
                    self.M = torch.stack(M_parts, dim=0)
                    self.S = torch.stack(S_parts, dim=0)
        self.discard_captured()
        if ran:
            self.active = True
        return ran

    # ---- forward math ----
    def read(self, slot, h_raw):
        """v4 reference read law (matches metis_v6_read_v4 in C):
        q = query_proj(h_raw) [T, q_dim]; per head: RMSNorm(head_dim,
        query_norm, eps 1e-6) then L2 (eps 1e-12); group reads with
        (denom + 1). h_raw is the layer INPUT residual (not normed)."""
        T = h_raw.shape[0]
        hd = self.head_dim
        # low-rank query: q = ((h @ B^T) @ A^T)
        q = F.linear(F.linear(h_raw.float(), self.query_b[slot].float()),
                     self.query_a[slot].float())          # [T, q_dim]
        q = q.view(T, -1, hd)                       # [T, H, hd]
        qh = q * self.query_norm[slot]
        inv = torch.rsqrt(qh.pow(2).mean(-1, keepdim=True) + 1e-6)
        q = qh * inv                                 # per-head RMSNorm
        q = F.normalize(q, dim=-1, eps=1e-12)        # per-head L2
        n_heads = q.shape[1]
        groups = self.q_dim // self.kv_dim
        hpg = n_heads // groups
        qg = q.reshape(T, groups, hpg * hd)          # [T, G, kv]
        M = self.M[slot]                             # [kv, kv]
        S = self.S[slot]                             # [kv]
        num = torch.einsum("tgk,kc->tgc", qg, M)
        denom = torch.einsum("tgk,k->tg", qg, S) + 1.0   # +1 (v4 law)
        out = num / denom.unsqueeze(-1)
        return out.reshape(T, self.q_dim)

    def fuse(self, slot, attn_branch, h_raw):
        """attn' = gamma*attn + (1-gamma)*o_proj(memnorm(read(h_raw))).
        h_raw [T, d] = layer INPUT residual (v4 read input)."""
        blk = self.layer_ids[slot]
        o_w = self.backbone.layers[blk]["o"]                 # [d, q_dim] bf16
        mem = self.read(slot, h_raw)
        memn = mem * self.mem_norm[slot]
        inv = torch.rsqrt(memn.pow(2).mean(-1, keepdim=True) + 1e-6)
        memn = memn * inv
        fused = F.linear(memn.to(o_w.dtype), o_w)            # [T, d]
        return (self.gamma * attn_branch.float()
                + (1.0 - self.gamma) * fused.float()).to(attn_branch.dtype)

    def commit(self, slot, hn_rows, eps):
        """hn_rows [L, d] captured (already attn-normed) rows of ONE layer.
        Single-slot commit: computes new_M/new_S math and stacks once."""
        with torch.enable_grad():
            nm, ns = self._commit_math(slot, hn_rows.float(), eps)
        M_parts = list(self.M)
        S_parts = list(self.S)
        M_parts[slot] = nm
        S_parts[slot] = ns
        self.M = torch.stack(M_parts, dim=0)
        self.S = torch.stack(S_parts, dim=0)
        self.active = True

    def _commit_math(self, slot, hn_rows, eps):
        """Pure math: returns (new_M[slot], new_S[slot]) tensors with graph.
        Caller owns the stacking (batched single-stack for commit_all)."""
        L = hn_rows.shape[0]
        norm_w = self.attn_norm_w[slot]                      # [d]
        x = hn_rows.float() * norm_w.float()
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        h = x * inv                                          # [L, d]
        scores = h @ self.w_agg[slot]                        # [L]
        p = F.softmax(scores / self.tau, dim=-1)
        sorted_p, sorted_idx = torch.sort(p, descending=True, stable=True)
        cum = torch.cumsum(sorted_p, dim=0)
        exceed = cum > self.rho
        if exceed.any():
            k = int(torch.nonzero(exceed)[0].item()) + 1
        else:
            k = L
        k = max(k, self.k_min)
        k = min(k, L)
        sel = torch.zeros(L, dtype=torch.bool, device=p.device)
        sel[sorted_idx[:k]] = True
        mass = p[sel].sum().clamp_min(1e-6)
        w_sel = torch.where(sel, p / mass, torch.zeros_like(p))

        Kall = h @ self.wk[slot].t()                         # [L, kv]
        Vall = h @ self.wv[slot].t()                         # [L, kv]
        Kn = F.normalize(Kall, dim=-1, eps=1e-12) / math.sqrt(self.kv_dim)

        a_pre = h @ self.gdu_aw[slot] + self.gdu_ab[slot]
        b_pre = h @ self.gdu_bw[slot] + self.gdu_bb[slot]
        beta_i = self.beta_scale * torch.sigmoid(b_pre)      # [L]
        alpha_scalar = ((torch.sigmoid(a_pre) * w_sel).sum()
                        / w_sel.sum().clamp_min(1e-6))

        km = Kn @ self.M[slot]                               # [L, kv]
        ks = Kn @ self.S[slot]                               # [L]
        b_eff = beta_i * w_sel                               # [L]
        new_M = (alpha_scalar * self.M[slot]
                 + (Kn.t() * b_eff) @ (Vall - alpha_scalar * km))
        new_S = self.S[slot] + (Kn * (b_eff * (1.0 - alpha_scalar * ks))
                                .unsqueeze(-1)).sum(0)
        # caller (commit / commit_all) owns the stacking so a 32-slot
        # batched commit performs ONE stack instead of 32.
        return new_M, new_S

    # ---- export ----
    def export(self, path):
        wk = self.wk.data
        # v4 trained-query path: the query_proj is TRAINED in our torch
        # space and pairs with Wk in the SAME row order — NO re-interleave.
        # (The old interleave served the frozen pre-rope-backbone-q read
        # where C's q snapshot came in GGUF rotary order; v4's q comes from
        # our trained query_proj, matching v9_to_bnmem_v4's direct-copy wk.)
        save_bnmem_v3(path,
                      layer_ids=self.layer_ids,
                      d_model=self.d_model, kv_dim=self.kv_dim,
                      q_dim=self.q_dim, head_dim=self.head_dim,
                      gamma=self.gamma, tau=self.tau, rho=self.rho,
                      k_min=self.k_min,
                      gdu_ab=self.gdu_ab.data, gdu_bb=self.gdu_bb.data,
                      beta_scale=self.beta_scale,
                      wk=wk, wv=self.wv.data,
                      w_agg=self.w_agg.data, gdu_aw=self.gdu_aw.data,
                      gdu_bw=self.gdu_bw.data, mem_norm=self.mem_norm.data,
                      query_a=self.query_a.data, query_b=self.query_b.data,
                      query_norm=self.query_norm.data)
        print(f'{{"exported":"{path}"}}', flush=True)


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

class MemoryTrainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        self.device = "cuda"

        print('{"phase":"load_backbone"}', flush=True)
        t0 = time.time()
        gw = GGUFWeights(args.gguf, args.lib)
        self.gw = gw
        self.backbone = TorchBackbone(gw, device="cuda",
                                      dtype=torch.bfloat16)
        print(f'{{"phase":"backbone_loaded","sec":{time.time()-t0:.1f},'
              f'"layers":{self.backbone.cfg.n_layers}}}', flush=True)

        n_blk = self.backbone.cfg.n_layers
        if args.layers == "all":
            layer_ids = list(range(n_blk))
        else:
            layer_ids = [int(x) for x in args.layers.split(",")]
        self.mem = MetisMemory(self.backbone, layer_ids,
                                gamma=args.gamma, tau=args.tau,
                                rho=args.rho, k_min=1,
                                beta_scale=args.beta_scale,
                                device="cuda", dtype=torch.float32,
                                seed=args.seed)
        if args.bb_init:
            self.mem.init_from_backbone()
            print('{"phase":"memory_init","source":"backbone_kv"}', flush=True)

        self.tok = CTokenizer(args.tok_probe, args.gguf)
        self.eos = self.tok.eos()

        strata = load_dataset(args.data)
        ov = None
        if args.oversample_distract > 1:
            ov = {"distract": args.oversample_distract,
                  "multi_entity": args.oversample_distract}
        order = dataset_order(strata, args.seed, oversample=ov)
        assert len(order) >= args.samples + args.valid, \
            f"data too small: {len(order)} < {args.samples}+{args.valid}"
        self.strata = strata
        self.train_idx = order[:args.samples]
        self.valid_idx = order[args.samples:args.samples + args.valid]

        self.opt = torch.optim.AdamW(self.mem.parameters(), lr=args.lr,
                                     weight_decay=args.wd)
        self.base_lr = args.lr
        self.warmup = args.warmup
        self.best_valid = float("inf")
        self.patience = 0

    def warmup_lr(self, step):
        if self.warmup <= 0 or step >= self.warmup:
            return self.base_lr
        return self.base_lr * (step + 1) / self.warmup

    def run_sample(self, sample, want_grads, tok_cache):
        """Forward one sample: full-prefix replays for exchanges (fusion
        gated to the current chunk), fresh-context query + teacher-forced
        target scoring. Returns (loss_sum, n_label)."""
        chunks = sample["messages"]
        q = sample.get("query_turn_id", len(chunks) - 1)
        self.mem.reset_state()
        mem, bb = self.mem, self.backbone
        eps = bb.cfg.rms_eps

        prompts = []
        for c in range(len(chunks)):
            is_q = c == q
            text, target = render_chunk(chunks[c], is_q)
            key = ("b" if c == 0 else "n", text)
            ids = tok_cache.get(key)
            if ids is None:
                ids = self.tok.encode(text, add_bos=(c == 0))
                tok_cache[key] = ids
            prompts.append((ids, is_q, target))
        tgt_ids = None
        for ids, is_q, target in prompts:
            if is_q:
                tkey = ("t", target)
                tgt_ids = tok_cache.get(tkey)
                if tgt_ids is None:
                    tgt_ids = self.tok.encode(target, add_bos=False)
                    tgt_ids = tgt_ids + [self.eos]
                    tok_cache[tkey] = tgt_ids
                break
        if tgt_ids is None:
            return None, 0

        ctx = torch.enable_grad() if want_grads else torch.no_grad()
        loss_sum = torch.zeros((), device="cuda")
        n_label = 0
        with ctx:
            for ids, is_q, target in prompts:
                if not is_q:
                    # REFERENCE-ALIGNED protocol: each exchange chunk is a
                    # standalone FRESH-context forward (their training loop
                    # evals chunks independently, use_cache=False; memory is
                    # the only carrier between chunks). Matches the C G2'
                    # driver (reset_context per chunk). 12GB ruling: replays
                    # run without grad; commits still differentiate through
                    # the memory params on the constant rows.
                    with torch.no_grad():
                        toks = torch.tensor(ids, device="cuda")
                        _ = bb(toks, memory=mem, fuse_start=0)
                    mem.commit_all_grad_enabled()
                    continue
                # query: FRESH context (memory is the only fact source).
                # M4(b) ruling: the query chunk is NEVER committed before
                # scoring — deploy commits the query turn AFTER generation
                # (it only affects the NEXT turn). Training with a
                # pre-scoring query commit taught the reads to expect the
                # question tokens inside M/S; at eval they are absent ->
                # out-of-distribution S -> degenerate decode (torch probe
                # proved the collapse is model-side, not C-i8).
                # The query forward still FUSES (reads run on exchange-
                # committed state) but its captures are DISCARDED.
                with torch.no_grad():
                    toks = torch.tensor(ids, device="cuda")
                    _ = bb(toks, memory=mem, fuse_start=0)
                mem.discard_captured()
                # score targets on the same fresh context with fusion active
                full = ids + tgt_ids[:-1]
                toks = torch.tensor(full, device="cuda")
                logits_all = bb(toks, memory=mem, logits_all=True,
                                fuse_start=0)
                lp = logits_all[len(full) - len(tgt_ids): len(full)]
                tgt_t = torch.tensor(tgt_ids, device="cuda")
                nll = F.cross_entropy(lp.float(), tgt_t, reduction="sum")
                loss_sum = loss_sum + nll
                n_label += len(tgt_ids)
        return loss_sum, n_label

    def run_valid(self, n, cursor, tok_cache):
        total, tok, n_ok = 0.0, 0, 0
        for _ in range(n):
            ci = cursor[0] % len(self.valid_idx)
            cursor[0] = (cursor[0] + 1) % len(self.valid_idx)
            s_idx = self.valid_idx[ci]
            line = self.strata[s_idx[0]][1][s_idx[1]]
            sample = json.loads(line)
            try:
                with torch.no_grad():
                    ls, nt = self.run_sample(sample, False, tok_cache)
            except Exception:
                continue
            if ls is None or nt == 0:
                continue
            total += ls.item()
            tok += nt
            n_ok += 1
        return total / max(tok, 1), tok, n_ok

    def export(self, path):
        self.mem.export(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("data")
    ap.add_argument("--output", default="metis_v6.bnmem")
    ap.add_argument("--lib", default=None)
    ap.add_argument("--tok-probe", default="tok_probe")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--beta-scale", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--valid-every", type=int, default=0)
    ap.add_argument("--valid-subset", type=int, default=32)
    ap.add_argument("--bb-init", action="store_true", default=True)
    ap.add_argument("--no-bb-init", dest="bb_init", action="store_false")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--oversample-distract", type=int, default=1,
                    help="rounds per pass for distract strata (v10run: 3)")
    args = ap.parse_args()

    tr = MemoryTrainer(args)
    tok_cache = {}
    cursor = [0]
    vcursor = [0]

    if args.export_only:
        tr.mem.reset_state()
        tr.export(args.output)
        return

    t0 = time.time()
    eval_every = args.valid_every or max(1, args.steps // 20)
    for step in range(args.steps):
        tr.opt.zero_grad(set_to_none=True)
        lr = tr.warmup_lr(step)
        for group in tr.opt.param_groups:
            group["lr"] = lr
        total_loss, total_tok, n_ok = 0.0, 0, 0
        for b in range(args.batch):
            ci = cursor[0] % len(tr.train_idx)
            cursor[0] = (cursor[0] + 1) % len(tr.train_idx)
            s_idx = tr.train_idx[ci]
            line = tr.strata[s_idx[0]][1][s_idx[1]]
            sample = json.loads(line)
            try:
                ls, nt = tr.run_sample(sample, True, tok_cache)
                if ls is not None and nt > 0:
                    # backward PER SAMPLE: the graph frees immediately,
                    # capping activation memory at one sample (12GB rule).
                    (ls / nt).backward()
                    total_loss += ls.item()
                    total_tok += nt
                    n_ok += 1
            except Exception as e:
                print(f'{{"step":{step},"skip_sample":"{type(e).__name__}:'
                      f'"{str(e)[:120]}"}}', flush=True)
                if isinstance(e, torch.OutOfMemoryError):
                    # free the failed sample's graph + cached blocks before
                    # the next allocation; reset_state drops M/S to None
                    # first so the fresh zeros cannot OOM on fragmentation
                    tr.mem.M = None
                    tr.mem.S = None
                    tr.mem.discard_captured()
                    torch.cuda.empty_cache()
                tr.mem.reset_state()
                continue
        if total_tok == 0:
            continue
        torch.nn.utils.clip_grad_norm_(tr.mem.parameters(), args.clip)
        tr.opt.step()
        mean_loss = total_loss / total_tok
        el = time.time() - t0
        sps = (step + 1) / el * 3600 if el > 0 else 0
        print(f'{{"step":{step},"split":"train","loss":{mean_loss:.6f},'
              f'"tokens":{total_tok},"samples":{n_ok},'
              f'"steps_per_hour":{sps:.0f}}}', flush=True)

        if (step + 1) % eval_every == 0:
            v, vt, vn = tr.run_valid(args.valid_subset, vcursor, tok_cache)
            print(f'{{"step":{step},"split":"valid","loss":{v:.6f},'
                  f'"tokens":{vt},"samples":{vn}}}', flush=True)
            if v < tr.best_valid - 1e-6:
                tr.best_valid = v
                tr.patience = 0
            else:
                tr.patience += 1
            if tr.patience >= 5:
                print('{"early_stop":true}', flush=True)
                break

    tr.export(args.output)
    v, vt, vn = tr.run_valid(min(args.valid_subset, 16), vcursor, tok_cache)
    print(f'{{"result":"done","final_valid_nll":{v:.6f},"steps":{args.steps}}}',
          flush=True)


if __name__ == "__main__":
    main()
