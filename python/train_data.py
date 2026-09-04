"""train_data.py -- data loading/tokenizer/render for the Metis memory trainer.

Trains the memory module (two-layer SiLU projections, NO gating search) on
the RTX 5070 with a torch bf16 backbone reimplementing the 158BitNet C
runtime math. Exports .bnmem v2 loadable by 158BitNet unchanged.

SEMANTICS (C runtime LAW, FD-verified; violate = model misbehaves):
- Module at block 31 (LAST block): d_model=2560, dk=dv=256, n_h=512,
  gamma=0.9, tau=1.0, rho=0.9 (configurable), k_min=1.
- Read (every active token): q~ = L2normalize(Wq proj of attn-normed h);
  out = (q~ M) / (q~ S); NO +1; |q~S| < 1e-8 -> zeros.
- Fusion: attn' = gamma*attn + (1-gamma) * fuse_w @ memnorm(out);
  memnorm = RMSNorm(out * qnorm_w) over dv with eps 1e-6 (C hardcodes 1e-6
  in metis_apply_layer).
- Commit per chunk (DOUBLE-NORM): h_normed = rmsnorm(h * attn_norm_w);
  scores = w_agg . h_normed; p = softmax(scores/tau); AlphaTopP(rho,k_min)
  hard select; w_i = p_i / sum(p_selected); K = L2normalize(Wk proj)/sqrt(dk);
  V = Wv proj; alpha = sum w_i sigma(gdu_aw.h_i+gdu_ab) / sum w_i;
  beta_i = sigma(gdu_bw.h_i+gdu_bb); GDU: M <- alpha*M +
  sum_i outer(k_i, beta_i w_i (V_i - alpha*(k_i M_old)));
  S <- S + sum_i k_i beta_i w_i (1 - alpha*(k_i S_old))  [S never scaled
  by alpha; both inner products on PRE-decay state].
- Empty-state behavior: memory fusion remains enabled; the zero readout
  therefore scales the attention branch by gamma before the first commit.
- Grad structure (C trainer ruling-A, simplified for Phase A): blocks 0..30
  frozen (no_grad); block-31 attn branch frozen but grad-TRACKED
  activations (fusion + memory trainable; FFN + output frozen). Loss grads
  flow through reads + the query chunk's commit (all chunks run under
  torch.autograd through the memory module; the earlier commits' write
  params DO get gradient through the state they leave -- richer than the
  C trainer's ruling-A, same direction).

Data: same 12-file JSONL + ChatML rendering as tools/train_metis_memory.c.
Tokenizer: the C runtime's own bitnet_tokenize via a C probe binary (token
ids MUST match; no Python tokenizer approximation is trusted).

Recipe: AdamW lr 2e-4 wd 0.01 warmup 200 constant, batch 4 grad-accum,
clip 1.0, seed 42, token-mean NLL on query-chunk assistant tokens.
"""
from __future__ import annotations

import argparse
import hashlib
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
try:
    # Legacy two-layer trainer/exporter.
    from bnmem_export import save_bnmem, METIS_ACT_SILU
except ImportError:
    # The current per-layer trainer imports this module only for dataset and
    # tokenizer helpers.  Keep the legacy class importable with the v3-v5
    # exporter too; its export path is not used by train_memory.py.
    from bnmem_export import save_bnmem_v3 as save_bnmem
    METIS_ACT_SILU = 1

# ----------------------------------------------------------------------
# Tokenizer bridge: token ids come from the C runtime via a probe binary
# that reads token ids (one per line) from stdin text or files.
# ----------------------------------------------------------------------

class CTokenizer:
    """Persistent tokenizer bridge: one tok_probe --serve-esc process,
    model loaded once; one escaped text per stdin line -> one ids line.
    Newlines are escaped as \\n (rendered prompts contain real newlines).
    Token ids come from the C runtime (no Python approximation)."""
    def __init__(self, probe_bin: str, model_path: str):
        import subprocess
        self.probe = probe_bin
        self.model = model_path
        self._proc = subprocess.Popen(
            [probe_bin, model_path, "--serve-esc"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)

    def encode(self, text: str, add_bos: bool) -> list[int]:
        esc = text.replace("\\", "\\\\").replace("\n", "\\n")
        self._proc.stdin.write(esc.encode("utf-8") + b"\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline().decode()
        ids = [int(x) for x in line.split() if x.strip()]
        if add_bos:
            ids = [self.bos()] + ids
        return ids

    def bos(self) -> int:
        if not hasattr(self, "_bos"):
            self._proc.stdin.write(b"BOS?\n")
            self._proc.stdin.flush()
            self._bos = int(self._proc.stdout.readline().decode().split()[0])
        return self._bos

    def eos(self) -> int:
        self._proc.stdin.write(b"EOS?\n")
        self._proc.stdin.flush()
        return int(self._proc.stdout.readline().decode().split()[0])


class CTokenDecoder:
    """Persistent exact decoder backed by the C runtime tokenizer."""

    def __init__(self, probe_bin: str, model_path: str):
        import subprocess
        self._proc = subprocess.Popen(
            [probe_bin, model_path, "--serve-decode"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)

    def decode(self, token_ids) -> str:
        line = " ".join(str(int(token)) for token in token_ids) + "\n"
        self._proc.stdin.write(line.encode("ascii"))
        self._proc.stdin.flush()
        raw = bytes.fromhex(self._proc.stdout.readline().decode().strip())
        return raw.decode("utf-8", errors="replace")


# ----------------------------------------------------------------------
# Metis memory module in torch (differentiable), C-layout parameters
# ----------------------------------------------------------------------

class MetisMemoryTorch(nn.Module):
    def __init__(self, d_model=2560, dk=256, dv=256, n_h=512,
                 gamma=0.9, tau=1.0, rho=0.9, k_min=1, seed=42,
                 device="cuda", dtype=torch.float32):
        super().__init__()
        self.d_model, self.dk, self.dv, self.n_h = d_model, dk, dv, n_h
        self.gamma, self.tau, self.rho, self.k_min = gamma, tau, rho, k_min
        g = torch.Generator(device="cpu").manual_seed(seed)
        def he(fan_in, fan_out, *shape):
            s = math.sqrt(6.0 / (fan_in + fan_out))
            return (torch.rand(*shape, generator=g) * 2 * s - s)
        # C trainer init: init_tensor_he uniform(-s, s), s=sqrt(6/(fi+fo))
        self.wq_gate = nn.Parameter(he(d_model, n_h, d_model, n_h).to(device=device, dtype=dtype))
        self.wq_mix = nn.Parameter(he(n_h, dk, n_h, dk).to(device=device, dtype=dtype))
        self.wk_gate = nn.Parameter(he(d_model, n_h, d_model, n_h).to(device=device, dtype=dtype))
        self.wk_mix = nn.Parameter(he(n_h, dk, n_h, dk).to(device=device, dtype=dtype))
        self.wv_gate = nn.Parameter(he(d_model, n_h, d_model, n_h).to(device=device, dtype=dtype))
        self.wv_mix = nn.Parameter(he(n_h, dv, n_h, dv).to(device=device, dtype=dtype))
        self.w_agg = nn.Parameter(he(d_model, 1, d_model).to(device=device, dtype=dtype))
        self.gdu_aw = nn.Parameter(he(d_model, 1, d_model).to(device=device, dtype=dtype))
        self.gdu_bw = nn.Parameter(he(d_model, 1, d_model).to(device=device, dtype=dtype))
        self.fuse_w = nn.Parameter(he(dv, d_model, d_model * dv).to(device=device, dtype=dtype))
        self.qnorm_w = nn.Parameter(torch.ones(dk, device=device, dtype=dtype))
        # gdu_ab/gdu_bb: NEVER trained (C trainer convention, serialized v2)
        self.gdu_ab = 2.0
        self.gdu_bb = 2.0
        # runtime state
        self.register_buffer("M", torch.zeros(dk, dv, device=device, dtype=dtype),
                             persistent=False)
        self.register_buffer("S", torch.zeros(dk, device=device, dtype=dtype),
                             persistent=False)
        self.active = False
        # capture buffer for commit rows (attn-normed hiddens)
        self._captured = None
        self.attn_norm_w = None   # [d] fp32 set by trainer (block 31 norm)

    def init_from_backbone(self, layer31, device="cuda", dtype=torch.float32):
        """Paper's original init (analog of the reference fp-init): seed the
        memory read/write projections from the backbone block-31 attention
        projections so K/V/Q live in the backbone's own semantic space from
        step 0 instead of learning one from random noise.

        The backbone projections are single-layer [d_model -> kv_dim(256) or
        q head dim]; our module is two-layer d -> n_h -> dk/dv with a fixed
        SiLU on the hidden layer. We approximate the backbone projection as
        mix @ (small-gain linear regime of SiLU):
          - gate rows: small random (pre-act ~0 => SiLU(0)~x/2 slope at 0 is
            0.5; we fold the 0.5 into mix) scaled so ||gate row|| is small.
          - mix: the backbone projection matrix (k_proj/v_proj [256, d] and
            the first q head [128, d] tiled 2x to reach dk=256).
        Concretely: out(x) = (mix @ SiLU(gate @ x)) with gate = eps*G and
        mix = 2*W_bb approximates W_bb @ x when eps*G@x is small. eps=1e-2.
        """
        with torch.no_grad():
            d, n_h = self.d_model, self.n_h
            eps = 1e-2
            def seed_pair(w_bb, out_dim):
                """w_bb [out_dim, d] backbone projection (row = out feature).
                Module layouts (per _project): w_gate [d, n_h], w_mix [n_h, out].
                First od hidden neurons carry backbone features in SiLU's
                linear regime: gate col j = eps * w_bb[j] (pre-act_j =
                eps * <w_bb_j, x>; SiLU(z) ~ z/2 near 0), mix[j, j] = 2 so the
                composition ~= W_bb @ x."""
                gate = torch.zeros(d, n_h, device=device, dtype=dtype)
                mix = torch.zeros(n_h, out_dim, device=device, dtype=dtype)
                od = min(out_dim, n_h)
                for j in range(od):
                    gate[:, j] = w_bb[j] * eps
                    mix[j, j] = 2.0
                return gate, mix
            k_bb = layer31["k"].detach().to(device=device, dtype=dtype)  # [256, d]
            v_bb = layer31["v"].detach().to(device=device, dtype=dtype)
            q_bb = layer31["q"].detach().to(device=device, dtype=dtype)  # [4096, d]
            def tile_to(w, n):
                """Repeat/tile backbone rows [r, d] up to n rows (dk/dv may
                exceed the 256-row kv projections; fill by cycling heads)."""
                if w.shape[0] >= n:
                    return w[:n]
                reps = (n + w.shape[0] - 1) // w.shape[0]
                return torch.cat([w] * reps, dim=0)[:n]
            k_seed = tile_to(k_bb, self.dk)
            v_seed = tile_to(v_bb, self.dv)
            q_seed = tile_to(q_bb, self.dk)   # q has 4096 rows; slice down
            self.wk_gate.copy_(seed_pair(k_seed, self.dk)[0])
            self.wk_mix.copy_(seed_pair(k_seed, self.dk)[1])
            self.wv_gate.copy_(seed_pair(v_seed, self.dv)[0])
            self.wv_mix.copy_(seed_pair(v_seed, self.dv)[1])
            self.wq_gate.copy_(seed_pair(q_seed, self.dk)[0])
            self.wq_mix.copy_(seed_pair(q_seed, self.dk)[1])

    # ---- state control ----
    def reset_state(self):
        # assign FRESH leaf tensors (zero_() in place would keep the old
        # autograd graph attached and later backwards would hit
        # backward-through-freed-graph errors across samples)
        self.M = torch.zeros_like(self.M)
        self.S = torch.zeros_like(self.S)
        self.active = False
        self._captured = None

    def capture(self, hn: torch.Tensor):
        """hn: [T, d] attn-normed rows from the LAST block. Accumulates
        until a commit consumes it (C1 semantics). Rows keep their autograd
        graph: the trainer's ruling-A flow backprops through the LAST
        commit (query chunk) + all reads; committing detached rows would
        orphan the write-param gradients. The backbone activations are
        constants w.r.t. memory params (frozen weights), so the graph
        through hn only carries memory-param dependence when the rows were
        themselves produced under an ACTIVE fusion (reads) -- exactly the
        chain the C trainer's commit-backward covers."""
        if self._captured is None:
            self._captured = hn.float()
        else:
            self._captured = torch.cat([self._captured, hn.float()], dim=0)

    def take_captured(self):
        c = self._captured
        return c

    def discard_captured(self):
        self._captured = None

    # ---- forward math (all differentiable w.r.t. module params) ----
    def _project(self, x, w_gate, w_mix):
        """x [L, d] -> [L, out]. h = SiLU(x @ w_gate); out = h @ w_mix.
        C layout: w_gate [d, n_h] (row j = neuron j), w_mix [n_h, out].
        x may be bf16 (backbone activations); cast to module dtype (fp32)."""
        x = x.float()
        h = F.silu(x @ w_gate)
        return h @ w_mix

    def read(self, hn: torch.Tensor):
        """hn [T, d] attn-normed rows -> mem_out [T, dv] (NO memnorm here --
        memnorm applied inside fuse like metis_apply_layer)."""
        q = self._project(hn, self.wq_gate, self.wq_mix)      # [T, dk]
        qn = F.normalize(q, dim=-1, eps=1e-12)                # L2 normalize
        denom = qn @ self.S                                   # [T]
        num = qn @ self.M                                     # [T, dv]
        safe = torch.where(denom.abs() < 1e-8,
                           torch.ones_like(denom), denom)     # guard: -> 0 row
        out = num / safe.unsqueeze(-1)
        zero_mask = (denom.abs() < 1e-8).unsqueeze(-1)
        return torch.where(zero_mask, torch.zeros_like(out), out)

    def fuse(self, attn_branch: torch.Tensor, hn: torch.Tensor):
        """attn' = gamma*attn + (1-gamma) * (memnorm(read) @ fuse_w^T per C
        row layout fuse_w[o, :] over dv).  attn_branch [T, d], hn [T, d]."""
        mem = self.read(hn)                                   # [T, dv]
        memn = mem * self.qnorm_w
        inv = torch.rsqrt(memn.pow(2).mean(-1, keepdim=True) + 1e-6)
        memn = memn * inv                                     # [T, dv]
        # C: fused[o] = sum_i fuse_w[o*dv + i] * memn[i]  (row-major d x dv)
        fw = self.fuse_w.view(self.d_model, self.dv)
        fused = memn @ fw.t()                                 # [T, d] fp32
        # keep the caller's dtype (bf16 backbone activations downstream)
        return (self.gamma * attn_branch.float()
                + (1.0 - self.gamma) * fused).to(attn_branch.dtype)

    def commit(self, hn_rows: torch.Tensor, attn_norm_w: torch.Tensor,
               eps: float):
        """hn_rows [L, d] = CAPTURED attn-normed rows. Double-norm:
        h_normed = rmsnorm(h * attn_norm_w) -- C re-norms already-normed
        rows. Updates M/S in place (differentiable through M/S values via
        out-of-place functional update)."""
        L = hn_rows.shape[0]
        x = hn_rows * attn_norm_w
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        h_normed = x * inv                                    # [L, d]
        scores = h_normed @ self.w_agg                        # [L]
        p = F.softmax(scores / self.tau, dim=-1)              # [L]
        # AlphaTopP hard select: sort desc (stable by index), cum > rho cuts
        sorted_p, sorted_idx = torch.sort(p, descending=True, stable=True)
        cum = torch.cumsum(sorted_p, dim=0)
        # C: k = first i+1 where cum > rho (breaks AT the crossing element,
        # inclusive); if never exceeds, k = L
        exceed = cum > self.rho
        if exceed.any():
            k = int(torch.nonzero(exceed)[0].item()) + 1
        else:
            k = L
        k = max(k, self.k_min)
        k = min(k, L)
        sel_mask = torch.zeros(L, dtype=torch.bool, device=p.device)
        sel_mask[sorted_idx[:k]] = True
        mass = p[sel_mask].sum().clamp_min(1e-6)
        w_sel = torch.where(sel_mask, p / mass, torch.zeros_like(p))  # [L]

        Kall = self._project(h_normed, self.wk_gate, self.wk_mix)    # [L, dk]
        Vall = self._project(h_normed, self.wv_gate, self.wv_mix)    # [L, dv]
        Kn = F.normalize(Kall, dim=-1, eps=1e-12) / math.sqrt(self.dk)

        a_pre = h_normed @ self.gdu_aw + self.gdu_ab
        b_pre = h_normed @ self.gdu_bw + self.gdu_bb
        beta_i = torch.sigmoid(b_pre)                                # [L]
        alpha_scalar = (torch.sigmoid(a_pre) * w_sel).sum() / w_sel.sum().clamp_min(1e-6)

        # GDU (km/ks PRE-decay; S never scaled by alpha)
        km = Kn @ self.M                                            # [L, dv]
        ks = Kn @ self.S                                            # [L]
        b_eff = beta_i * w_sel                                      # [L]
        new_M = alpha_scalar * self.M + (Kn.t() * b_eff) @ (Vall - alpha_scalar * km)
        new_S = self.S + (Kn * (b_eff * (1.0 - alpha_scalar * ks)).unsqueeze(-1)).sum(0)
        self.M = new_M
        self.S = new_S
        self.active = True
        self.discard_captured()


# ----------------------------------------------------------------------
# Data: same 12-file JSONL streaming as the C trainer
# ----------------------------------------------------------------------

WANTED_FILES = [
    "reconstruction.jsonl",
    "locomo_cat1.jsonl", "locomo_cat2.jsonl",
    "locomo_cat3.jsonl", "locomo_cat4.jsonl",
    "locomo_cat5.jsonl",
    "remember_explicit.jsonl", "remember_implicit.jsonl",
    "remember_distract.jsonl", "update_explicit.jsonl",
    "update_implicit.jsonl", "update_distract.jsonl",
    "forget_explicit.jsonl", "forget_distract.jsonl",
    "reflect_explicit.jsonl", "reflect_distract.jsonl",
    "multi_entity.jsonl", "post_memory.jsonl",
]


def load_dataset(data_dir):
    """data_dir may be a single directory or a colon-separated list; strata
    from later dirs are appended (deduped by (stem, line) so overlapping
    corpora don't double-count)."""
    dirs = [d for d in str(data_dir).split(":") if d]
    strata = []
    seen = set()
    for d in dirs:
        for name in WANTED_FILES:
            path = os.path.join(d, name)
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                lines = [line.rstrip("\n") for line in f
                         if line.strip() and (name.split(".")[0], line) not in seen
                         and not seen.add((name.split(".")[0], line))]
            if lines:
                strata.append((name.split(".")[0], lines))
    return strata


def dataset_order(strata, seed, oversample=None):
    """Round-robin interleave across strata (v10run: weak-class oversample —
    strata named in `oversample` appear k times per round, biasing the mix
    toward the eval's weak classes without duplicating any single sample
    within one pass)."""
    perms = []
    for stem, lines in strata:
        idx = list(range(len(lines)))
        stable_stem_seed = int.from_bytes(
            hashlib.sha256(stem.encode("utf-8")).digest()[:8], "little")
        rng2 = random.Random((seed + stable_stem_seed) % (2**63))
        rng2.shuffle(idx)
        perms.append(idx)
    reps = [oversample.get(stem, 1) if oversample else 1
            for stem, _ in strata]
    order = []
    pos = [0] * len(strata)
    while True:
        progressed = False
        for s in range(len(strata)):
            for _ in range(reps[s]):
                if pos[s] < len(strata[s][1]):
                    order.append((s, perms[s][pos[s]]))
                    pos[s] += 1
                    progressed = True
        if not progressed:
            break
    return order


def render_chunk(chunk_msgs, is_query):
    """ChatML rendering, reference-protocol (probe2: think blocks destroy
    trained memory recall): no system, NO think block anywhere; assistant
    messages render plain; query renders everything except final assistant
    content, then the plain assistant header."""
    out = []
    last = len(chunk_msgs) - 1
    target = None
    if is_query:
        if chunk_msgs[last]["role"] != "assistant":
            raise ValueError("query chunk must end with assistant")
        target = chunk_msgs[last]["content"]
    upto = last if is_query else len(chunk_msgs)
    for m in chunk_msgs[:upto]:
        out.append(f"<|im_start|>{m['role']}\n")
        out.append(f"{m['content']}<|im_end|>\n")
    if is_query:
        out.append("<|im_start|>assistant\n")
    return "".join(out), target


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

class Trainer:
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

        self.mem = MetisMemoryTorch(
            d_model=self.backbone.cfg.hidden,
            dk=args.dk, dv=args.dv, n_h=args.n_h,
            gamma=args.gamma, tau=args.tau, rho=args.rho,
            k_min=1, seed=args.seed, device="cuda", dtype=torch.float32)
        # block-31 attn_norm weights (f32) for the commit double-norm
        self.mem.attn_norm_w = torch.from_numpy(
            gw.get_f32("blk.31.attn_norm.weight", (gw.hidden,)).copy()).cuda()
        self.rms_eps = self.backbone.cfg.rms_eps
        if args.bb_init:
            # v4: seed memory projections from backbone block-31 k/v/q
            # (paper's original init; reference fp-init analog — their
            # update recall went 30 -> 70 with this class of fix)
            self.mem.init_from_backbone(self.backbone.layers[args.layer],
                                        device="cuda", dtype=torch.float32)
            print('{"phase":"memory_init","source":"backbone_proj"}', flush=True)

        self.tok = CTokenizer(args.tok_probe, args.gguf)
        self.eos = self.tok.eos()

        strata = load_dataset(args.data)
        order = dataset_order(strata, args.seed)
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

    # ---- one sample forward (+ backward when want_grads) ----
    def run_sample(self, sample, want_grads, tok_cache):
        """Returns (loss_sum, n_label, ok). Loss = NLL over query-chunk
        assistant tokens (teacher-forced whole-sequence evals)."""
        chunks = sample["messages"]
        q = sample.get("query_turn_id", len(chunks) - 1)
        self.mem.reset_state()
        loss_sum = torch.zeros((), device="cuda")
        n_label = 0
        tok = self.tok

        ctx_grad = torch.enable_grad() if want_grads else torch.no_grad()
        # NOTE: backbone forward internally splits grad tracking per block
        # (0..30 frozen). We run under enable_grad and rely on the backbone's
        # frozen-weight no_grad wrappers (weights have requires_grad=False so
        # autograd only tracks the memory-module leaves we pass activations
        # through).

        prompts = []
        for c in range(len(chunks)):
            is_q = c == q
            text, target = render_chunk(chunks[c], is_q)
            key = ("b" if c == 0 else "n", text)
            ids = tok_cache.get(key)
            if ids is None:
                ids = tok.encode(text, add_bos=(c == 0))
                tok_cache[key] = ids
            prompts.append((ids, is_q, target))
        # query target tokens
        tgt_ids = None
        for ids, is_q, target in prompts:
            if is_q:
                tkey = ("t", target)
                tgt_ids = tok_cache.get(tkey)
                if tgt_ids is None:
                    tgt_ids = tok.encode(target, add_bos=False)
                    tgt_ids = tgt_ids + [self.eos]
                    tok_cache[tkey] = tgt_ids
                break

        with ctx_grad:
            # The C runtime CONTINUES KV/positions across per-chunk evals.
            # The torch backbone has no cross-call KV cache, so we replay
            # each chunk as a FULL-PREFIX forward: for chunk c we feed
            # tokens of chunks 0..c in one sequence (positions 0..n-1,
            # causal attention over the whole prefix -- identical math to
            # the runtime's incremental evals). The memory module only
            # captures/commits the CURRENT chunk's rows: rows 0..prev are
            # pass-through (inactive memory had no effect when they were
            # first processed... EXCEPT earlier chunks' committed reads
            # would now see fused hidden states -- but fusion only alters
            # the attn branch of tokens AFTER a commit, and in the runtime
            # the earlier tokens' KV was stored PRE-commit. Replaying with
            # an ACTIVE memory would re-fuse earlier tokens differently.
            # Therefore: for each chunk's replay, the memory fusion must be
            # INACTIVE for all previous chunks' tokens and ACTIVE only for
            # the current chunk's tokens (matching the runtime timeline:
            # chunk c's tokens were processed with memory state as of
            # commits 0..c-1). We implement this with a token-range mask.
            # DEPLOYMENT-FAITHFUL QUERY (v3 fix, mirrors Metis eval protocol):
            # commits run on the exchange chunks with use-cache-free forwards;
            # the QUERY is evaluated from a FRESH context containing ONLY the
            # query prompt — the causal-attention path cannot see the earlier
            # chunks' text, so the only source of remembered facts is the
            # memory state (M/S). This matches encode_and_commit_memory +
            # model.generate(query) semantics that the reference training and
            # eval used, and removes the context leak that let the model
            # answer from history instead of learning strong memory reads.
            prefix_ids = []
            for c, (ids, is_q, target) in enumerate(prompts):
                if not is_q:
                    prefix_ids = prefix_ids + ids
                    toks = torch.tensor(prefix_ids, device="cuda")
                    self.backbone(toks, memory=self.mem,
                                  fuse_start=len(prefix_ids) - len(ids))
                    captured_all = self.mem.take_captured()
                    self.mem.discard_captured()
                    current = captured_all[len(prefix_ids) - len(ids):]
                    self.mem.commit(current, self.mem.attn_norm_w, self.rms_eps)
                    continue

                # query chunk: FRESH context, memory-only information path
                toks = torch.tensor(ids, device="cuda")
                logits = self.backbone(toks, memory=self.mem,
                                       fuse_start=0)
                captured_all = self.mem.take_captured()
                self.mem.discard_captured()
                self.mem.commit(captured_all, self.mem.attn_norm_w,
                                self.rms_eps)

                # teacher-forced target scoring on the same fresh context:
                # prompt + targets in one sequence; fusion active for the
                # whole query (positions 0..) since memory was already
                # committed by every exchange before the loss.
                full = ids + tgt_ids[:-1]
                toks = torch.tensor(full, device="cuda")
                logits_all = self.backbone(toks, memory=self.mem,
                                           logits_all=True,
                                           fuse_start=0)
                lp = logits_all[len(full) - len(tgt_ids): len(full)]
                tgt_t = torch.tensor(tgt_ids, device="cuda")
                nll = F.cross_entropy(lp.float(), tgt_t, reduction="sum")
                loss_sum = loss_sum + nll
                n_label += len(tgt_ids)
        ok = True
        val = loss_sum.item() if n_label > 0 else 0.0
        return val, n_label, ok

    def train_step(self, step, cursor, tok_cache):
        self.opt.zero_grad(set_to_none=True)
        total_loss, total_tok, n_ok = 0.0, 0, 0
        lr = self.warmup_lr(step)
        for gi, group in enumerate(self.opt.param_groups):
            group["lr"] = lr
        for b in range(self.args.batch):
            ci = cursor[0] % len(self.train_idx)
            cursor[0] = (cursor[0] + 1) % len(self.train_idx)
            s_idx = self.train_idx[ci]
            line = self.strata[s_idx[0]][1][s_idx[1]]
            sample = json.loads(line)
            try:
                ls, nt, ok = self.run_sample(sample, True, tok_cache)
            except Exception as e:
                print(f'{{"step":{step},"skip_sample":"{type(e).__name__}"}}',
                      flush=True)
                continue
            if not ok or nt == 0:
                continue
            n_ok += 1
            total_loss += ls
            total_tok += nt
        # loss was accumulated as raw sums; backward per-sample inside
        # run_sample is not done there -- redo properly below
        return total_loss, total_tok, n_ok

    def run_valid(self, n, cursor, tok_cache):
        total, tok, n_ok = 0.0, 0, 0
        for i in range(n):
            ci = cursor[0] % len(self.valid_idx)
            cursor[0] = (cursor[0] + 1) % len(self.valid_idx)
            s_idx = self.valid_idx[ci]
            line = self.strata[s_idx[0]][1][s_idx[1]]
            sample = json.loads(line)
            try:
                ls, nt, ok = self.run_sample(sample, False, tok_cache)
            except Exception:
                continue
            if not ok or nt == 0:
                continue
            total += ls
            tok += nt
            n_ok += 1
        return total / max(tok, 1), tok, n_ok

    # ---- export ----
    def export(self, path):
        m = self.mem
        q_acts = torch.full((m.n_h,), METIS_ACT_SILU, dtype=torch.uint8)
        k_acts = torch.full((m.n_h,), METIS_ACT_SILU, dtype=torch.uint8)
        v_acts = torch.full((m.n_h,), METIS_ACT_SILU, dtype=torch.uint8)
        save_bnmem(path,
                   layer_ids=[self.args.layer],
                   d_model=m.d_model, dk=m.dk, dv=m.dv,
                   n_hq=m.n_h, n_hk=m.n_h, n_hv=m.n_h, n_hg=m.n_h,
                   gamma=m.gamma, tau=m.tau, rho=m.rho, k_min=m.k_min,
                   gdu_ab=m.gdu_ab, gdu_bb=m.gdu_bb,
                   q_acts=q_acts, k_acts=k_acts, v_acts=v_acts,
                   wq_gate=m.wq_gate.data, wq_mix=m.wq_mix.data,
                   wk_gate=m.wk_gate.data, wk_mix=m.wk_mix.data,
                   wv_gate=m.wv_gate.data, wv_mix=m.wv_mix.data,
                   w_agg=m.w_agg.data, gdu_aw=m.gdu_aw.data,
                   gdu_bw=m.gdu_bw.data, fuse_w=m.fuse_w.data,
                   qnorm_w=m.qnorm_w.data)
        print(f'{{"exported":"{path}"}}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("data")
    ap.add_argument("--output", default="metis_phaseA.bnmem")
    ap.add_argument("--lib", default=None)
    ap.add_argument("--tok-probe", default="tok_probe")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--dv", type=int, default=256)
    ap.add_argument("--n-h", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--valid", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--valid-every", type=int, default=0)
    ap.add_argument("--valid-subset", type=int, default=32)
    ap.add_argument("--export-only", action="store_true",
                    help="skip training (used by parity checks)")
    ap.add_argument("--bb-init", action="store_true",
                    help="seed memory projections from backbone block-31 k/v/q (v4)")
    args = ap.parse_args()

    tr = Trainer(args)
    tok_cache = {}
    cursor = [0]
    vcursor = [0]

    if args.export_only:
        tr.export(args.output)
        return

    t0 = time.time()
    eval_every = args.valid_every or max(1, args.steps // 20)
    for step in range(args.steps):
        # ---- grad accumulation with real backward ----
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
                # tokenize + forward + backward in one grad context
                with torch.enable_grad():
                    chunks = sample["messages"]
                    qturn = sample.get("query_turn_id", len(chunks) - 1)
                    tr.mem.reset_state()
                    prompts = []
                    for c in range(len(chunks)):
                        is_q = c == qturn
                        text, target = render_chunk(chunks[c], is_q)
                        key = ("b" if c == 0 else "n", text)
                        ids = tok_cache.get(key)
                        if ids is None:
                            ids = tr.tok.encode(text, add_bos=(c == 0))
                            tok_cache[key] = ids
                        prompts.append((ids, is_q, target))
                    tgt_ids = None
                    for ids, is_q, target in prompts:
                        if is_q:
                            tkey = ("t", target)
                            tgt_ids = tok_cache.get(tkey)
                            if tgt_ids is None:
                                tgt_ids = tr.tok.encode(target, add_bos=False)
                                tgt_ids = tgt_ids + [tr.eos]
                                tok_cache[tkey] = tgt_ids
                            break
                    loss_sum = torch.zeros((), device="cuda")
                    n_label = 0
                    # full-prefix replay (positions continue like the C
                    # runtime; fusion/capture gated to the current chunk)
                    prefix_ids = []
                    for c, (ids, is_q, target) in enumerate(prompts):
                        prefix_ids = prefix_ids + ids
                        start = len(prefix_ids) - len(ids)
                        toks = torch.tensor(prefix_ids, device="cuda")
                        _ = tr.backbone(toks, memory=tr.mem,
                                        fuse_start=start)
                        captured_all = tr.mem.take_captured()
                        tr.mem.discard_captured()
                        tr.mem.commit(captured_all[start:],
                                      tr.mem.attn_norm_w, tr.rms_eps)
                        if not is_q:
                            continue
                        full = prefix_ids + tgt_ids[:-1]
                        toks = torch.tensor(full, device="cuda")
                        logits_all = tr.backbone(toks, memory=tr.mem,
                                                 logits_all=True,
                                                 fuse_start=start)
                        lp = logits_all[len(full) - len(tgt_ids): len(full)]
                        tgt_t = torch.tensor(tgt_ids, device="cuda")
                        nll = F.cross_entropy(lp.float(), tgt_t,
                                              reduction="sum")
                        loss_sum = loss_sum + nll
                        n_label += len(tgt_ids)
                if n_label > 0:
                    (loss_sum / n_label).backward()
                    total_loss += loss_sum.item()
                    total_tok += n_label
                    n_ok += 1
            except Exception as e:
                print(f'{{"step":{step},"skip_sample":"{type(e).__name__}:"'
                      f'"{str(e)[:120]}"}}', flush=True)
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
