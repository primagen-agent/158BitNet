"""torch_backbone.py -- minimal llama decoder in torch bf16 matching the
158BitNet C runtime math exactly (Phase-A trainer, no NGMA).

Math contract (src/bitnet.c bitnet_eval):
- RMSNorm: y = x * inv_rms(x) * w with eps from
  llama.attention.layer_norm_rms_epsilon (C: bitnet_rms_norm_eps).
- GQA attention: n_heads x head_dim Q, n_kv_heads K/V, RoPE on first
  rope_dim dims (interleaved-pair rotation identical to HF rotate_half after
  the row deinterleave), scores * 1/sqrt(head_dim), causal.
- FFN: SiLU(gate)*up then down; residual scale 1.0 (llama arch).
- output: output_norm RMSNorm then separate Q6_K output projection (not tied).
- Metis hook at the LAST block: fusion replaces attn-branch BEFORE the
  residual add (matches metis_apply_layer being applied to `down`).

RoPE theta (bitnet.c): theta_j = 1/ base^(2j/head_dim), divided by
rope_factors[j] when present and > 0 (longrope; the runtime picks the LONG
table iff max_tokens > context_length -- the trainer's short contexts use
the SHORT table, matching minimal_generate/openai_server contexts).

Weight layout: GGUF dims [in,out] but flat storage is row-major over
dims[0]-contiguous data, so the dequantized buffer reshapes directly to
[out, in] = nn.Linear.weight layout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ggw import GGUFWeights
from model_identity import sha256_file


@dataclass
class BackboneConfig:
    n_layers: int
    hidden: int
    q_dim: int
    kv_dim: int
    ffn: int
    vocab: int
    rope_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rms_eps: float
    rope_freq_base: float
    rope_factors: torch.Tensor | None = None   # [rope_dim/2] fp32 cpu
    embedding_scale: float = 1.0
    residual_scale: float = 1.0
    logit_scale: float = 1.0


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # C: inv = 1/sqrt(sum(x^2)/n + eps); y = x*inv*w  (w applied AFTER scale)
    inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * inv * w.float()).to(x.dtype)


def rope_tables(cfg: BackboneConfig, positions: torch.Tensor,
                device, dtype=torch.float32):
    """cos/sin [T, rope_dim/2] matching the C cache build:
    theta_j = 1/(base^(2j/head_dim)) / factors[j]; angle = pos*theta."""
    # MPS does not implement float64.  CUDA/CPU retain the higher-precision
    # table construction used by the parity tests; Apple GPU evaluation uses
    # float32, matching the C runtime's stored rope cache precision.
    work_dtype = (
        torch.float32 if torch.device(device).type == "mps"
        else torch.float64)
    j = torch.arange(cfg.rope_dim // 2, dtype=work_dtype, device=device)
    theta = 1.0 / torch.pow(
        torch.tensor(
            float(cfg.rope_freq_base), dtype=work_dtype, device=device),
        (2.0 * j / cfg.head_dim))
    if cfg.rope_factors is not None:
        f = cfg.rope_factors.to(device=device, dtype=work_dtype)
        theta = torch.where(f > 0, theta / f, theta)
    ang = (
        positions.to(device=device, dtype=work_dtype).unsqueeze(1)
        * theta.unsqueeze(0))
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope_hf(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [H, T, D] (already transposed). HF rotate_half layout: cos/sin
    [T, D/2] duplicated. Algebraically identical to the C interleaved pair
    rotation once Q/K rows are deinterleaved at load."""
    cd = torch.cat([cos, cos], dim=-1).unsqueeze(0)   # [1, T, D]
    sd = torch.cat([sin, sin], dim=-1).unsqueeze(0)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cd + rot * sd


def _deinterleave_rope_rows(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """GGUF (llama.cpp) Q/K rows: consecutive row pairs (2j, 2j+1) form the
    rotary pair (interleaved rotation). HF pairs row j with j + head_dim/2
    within each head. Reorders per head: [0,2,...,d-2 | 1,3,...,d-1] ->
    [0..d/2-1 | d/2..d-1]."""
    out_rows = weight.shape[0]
    if out_rows % head_dim != 0:
        return weight
    n_heads = out_rows // head_dim
    view = weight.view(n_heads, head_dim, *weight.shape[1:])
    even, odd = view[:, 0::2], view[:, 1::2]
    return torch.cat((even, odd), dim=1).reshape(weight.shape)


class TorchBackbone(nn.Module):
    """Full 3B backbone in bf16 on GPU. Forward returns last-position logits
    (or all-position logits with logits_all=True) plus the hooks needed by
    the memory fusion."""

    def __init__(self, gw: GGUFWeights, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.gguf_path = gw.gguf_path
        self.cfg = BackboneConfig(
            n_layers=gw.n_layers, hidden=gw.hidden, q_dim=gw.q_dim,
            kv_dim=gw.kv_dim, ffn=gw.ffn, vocab=gw.vocab, rope_dim=gw.rope_dim,
            n_heads=gw.n_heads, n_kv_heads=gw.n_kv_heads,
            head_dim=gw.head_dim, rms_eps=gw.rms_eps,
            rope_freq_base=gw.rope_freq_base,
            rope_factors=None,
            embedding_scale=gw.embedding_scale,
            residual_scale=gw.residual_scale,
            logit_scale=gw.logit_scale)
        factors = gw.get_rope_factors("short")
        if factors is not None:
            self.cfg.rope_factors = torch.from_numpy(factors.copy())
        self.device = device
        self.dtype = dtype

        emb = torch.from_numpy(gw.get_f32("token_embd.weight",
                                          (gw.vocab, gw.hidden)).copy())
        self.token_embd = emb.to(device=device, dtype=dtype)

        self.layers = []
        for i in range(gw.n_layers):
            lw = gw.get_layer(i)
            layer = {
                "attn_norm": self._norm(gw, i, "attn_norm"),
                "ffn_norm": self._norm(gw, i, "ffn_norm"),
                "q": _deinterleave_rope_rows(lw.q, gw.head_dim).to(device=device, dtype=dtype),
                "k": _deinterleave_rope_rows(lw.k, gw.head_dim).to(device=device, dtype=dtype),
                "v": lw.v.to(device=device, dtype=dtype),
                "o": lw.o.to(device=device, dtype=dtype),
                "gate": lw.gate.to(device=device, dtype=dtype),
                "up": lw.up.to(device=device, dtype=dtype),
                "down": lw.down.to(device=device, dtype=dtype),
            }
            self.layers.append(layer)
        self.out_norm = self._norm_out(gw)
        try:
            output_array = gw.get_f32(
                "output.weight", (gw.vocab, gw.hidden))
        except RuntimeError:
            output_array = gw.get_f32(
                "token_embd.weight", (gw.vocab, gw.hidden))
        out = torch.from_numpy(output_array.copy())
        self.out_proj = out.to(device=device, dtype=dtype)
        self.backbone_lora = None
        self.output_lora = None
        self.answer_decoder = None

    def model_sha256(self) -> bytes:
        return sha256_file(self.gguf_path)

    def linear(self, hidden, weight, block_index, layer_id):
        output = F.linear(hidden, weight)
        if self.backbone_lora is not None:
            delta = self.backbone_lora.delta(
                block_index, layer_id, hidden)
            if delta is not None:
                output = output + delta.to(output.dtype)
        return output

    def _norm(self, gw, i, kind):
        arr = gw.get_f32(f"blk.{i}.{kind}.weight", (gw.hidden,))
        return torch.from_numpy(arr.copy()).to(device=self.device, dtype=torch.float32)

    def _norm_out(self, gw):
        arr = gw.get_f32("output_norm.weight", (gw.hidden,))
        return torch.from_numpy(arr.copy()).to(device=self.device, dtype=torch.float32)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0,
                memory=None, logits_all: bool = False,
                fuse_start: int = 0, memory_v6=None,
                return_hidden: bool = False):
        """Grad scope: ALL backbone weights are requires_grad=False, so
        autograd flows only through activations the memory module touches.
        Callers wrap in torch.no_grad() for pure eval; for training, the
        fusion (memory.fuse on the last block's attn branch) builds a graph
        into the memory params and the loss backprop reaches them through
        the last block's FFN + output projection (which are differentiable
        ops on the fused activations even with frozen weights).
        tokens: [T] int64 on device. start_pos: absolute position of
        tokens[0] (KV cache is rebuilt each call -- trainer evals whole
        sequences). memory: optional MetisMemoryTorch (v2 single-layer).
        memory_v6: optional MetisV6Torch — fuses at EVERY memory layer with
        pre-RoPE q reads + per-layer capture. fuse_start: fusion AND capture
        apply only to tokens with index >= fuse_start (full-prefix replay:
        earlier tokens must behave as they did when first processed, with
        memory inactive).
        Returns normalized hidden states [T, hidden] when
        return_hidden=True; otherwise logits [T, vocab] if logits_all else
        [vocab]."""
        cfg = self.cfg
        T = tokens.shape[0]
        device = self.device
        h = self.token_embd[tokens] * cfg.embedding_scale  # [T, D] bf16
        positions = torch.arange(start_pos, start_pos + T, device=device)
        cos, sin = rope_tables(cfg, positions, device, torch.bfloat16)
        cos = cos.to(self.dtype)
        sin = sin.to(self.dtype)

        v6 = memory_v6
        v6_caps = [] if v6 is not None else None
        for bi in range(cfg.n_layers):
            L = self.layers[bi]
            is_last = bi == cfg.n_layers - 1
            resid = h
            hn = rms_norm(h, L["attn_norm"], cfg.rms_eps)  # bf16

            q = self.linear(hn, L["q"], bi, 0).view(
                T, cfg.n_heads, cfg.head_dim).transpose(0, 1)
            k = self.linear(hn, L["k"], bi, 1).view(
                T, cfg.n_kv_heads, cfg.head_dim).transpose(0, 1)
            v = self.linear(hn, L["v"], bi, 2).view(
                T, cfg.n_kv_heads, cfg.head_dim).transpose(0, 1)
            q_pre = q                                        # pre-RoPE [H, T, hd]

            q = apply_rope_hf(q, cos, sin)
            k = apply_rope_hf(k, cos, sin)

            # GQA repeat + causal SDPA (repeat on the HEAD dim while k/v
            # are still [n_kv_heads, T, hd] -- .transpose(0,1) above)
            rep = cfg.n_heads // cfg.n_kv_heads
            kx = k.repeat_interleave(rep, dim=0) if rep > 1 else k
            vx = v.repeat_interleave(rep, dim=0) if rep > 1 else v
            att = torch.nn.functional.scaled_dot_product_attention(
                q, kx, vx, is_causal=True if start_pos == 0 else None,
                scale=1.0 / math.sqrt(cfg.head_dim))
            if start_pos != 0:
                # cross-chunk attention: [H, Tq, Tpast+Tk] -- caller passes
                # whole sequences in Phase A so this branch is unused.
                raise NotImplementedError("start_pos != 0 requires KV cache")
            attn = att.transpose(0, 1).reshape(T, cfg.q_dim)  # [T, q_dim]
            down = self.linear(attn, L["o"], bi, 3)          # [T, D] bf16

            if is_last and memory is not None:
                # capture attn-normed rows for the commit (C runtime captures
                # `hidden` == hn after the pointer swap); only the current
                # chunk's rows (index >= fuse_start) are captured/fused
                if fuse_start > 0:
                    hn_cur = hn[fuse_start:]
                else:
                    hn_cur = hn
                memory.capture(hn_cur)
                if memory.active:
                    if fuse_start > 0:
                        down_fused = memory.fuse(down[fuse_start:], hn_cur)
                        down = torch.cat([down[:fuse_start], down_fused],
                                         dim=0)
                    else:
                        down = memory.fuse(down, hn)

            if v6 is not None:
                slot = v6.slot_of(bi)
                if slot is not None:
                    # v4 read input: the layer's INPUT residual (raw, not
                    # normed) — the reference query_proj's input domain.
                    h_raw_cur = resid if fuse_start == 0 else resid[fuse_start:]
                    v6_caps.append((slot, h_raw_cur))
                    if v6.active:
                        if fuse_start > 0:
                            down_fused = v6.fuse(slot, down[fuse_start:], h_raw_cur)
                            down = torch.cat([down[:fuse_start], down_fused],
                                             dim=0)
                        else:
                            down = v6.fuse(slot, down, h_raw_cur)

            h = resid + down * cfg.residual_scale

            resid2 = h
            hn2 = rms_norm(h, L["ffn_norm"], cfg.rms_eps)
            g = self.linear(hn2, L["gate"], bi, 4)
            u = self.linear(hn2, L["up"], bi, 5)
            act = F.silu(g) * u
            dout = self.linear(act, L["down"], bi, 6)
            h = resid2 + dout * cfg.residual_scale

        if v6 is not None and v6_caps:
            # deliver per-layer captures in LAYER order (slot index), with
            # slots that saw no fusion this call contributing nothing
            per_layer = [None] * v6.n_layers
            for entry in v6_caps:
                # The hyper-memory applies the layer input norm itself.
                # Capturing the already-normalized rows here would apply
                # input_layernorm twice and diverge from the deployment
                # runtime as well as the paper's PreNorm(H) definition.
                per_layer[entry[0]] = entry[1]
            v6.capture(per_layer)

        h = rms_norm(h, self.out_norm, cfg.rms_eps)
        if return_hidden:
            return h
        if self.answer_decoder is not None:
            memory_summary = (
                (
                    v6.answer_memory_layers()
                    if self.answer_decoder.structured_memory
                    else v6.answer_memory_summary()
                )
                if (
                    v6 is not None
                    and self.answer_decoder.memory_aware
                ) else None)
            h = self.answer_decoder(h, memory_summary)
        logits = F.linear(h, self.out_proj)
        if self.output_lora is not None:
            logits = logits + self.output_lora(h)
        if cfg.logit_scale != 0.0:
            logits = logits / cfg.logit_scale
        return logits if logits_all else logits[-1]
