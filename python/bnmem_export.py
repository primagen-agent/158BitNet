"""bnmem_v3_export.py -- .bnmem v3 writer for the v6 full-path model.

File layout (src/metis/metis_v6.c, LAW):
  magic "BNMEM3\\0\\0" (8 bytes)
  u32 version (=3)
  u32 n_layers
  u32 layer_ids[32]  (unused = 0xFFFFFFFF)
  u32 d_model, kv_dim, q_dim, head_dim
  f32 gamma, tau, rho
  u32 k_min
  f32 gdu_ab, gdu_bb, beta_scale
  tensor payloads in fixed order (f32 row-major, layer-major):
    wk       [n_layers x kv_dim*d_model]
    wv       [n_layers x kv_dim*d_model]
    w_agg    [n_layers x d_model]
    gdu_aw   [n_layers x d_model]
    gdu_bw   [n_layers x d_model]
    mem_norm [n_layers x q_dim]
  manifest: u32 tensor_count (6), then per tensor
    { u32 name_len; name; u32 rows; u32 cols; u64 byte_offset; u32 crc32 }
All little-endian; crc32 = zlib-style reflected poly 0xEDB88320.
"""
from __future__ import annotations

import struct
import zlib

import torch

METIS_V6_MAX_LAYERS = 32
UNUSED_LAYER = 0xFFFFFFFF


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def save_bnmem_v3(path: str, *, layer_ids: list[int], d_model: int,
                  kv_dim: int, q_dim: int, head_dim: int,
                  gamma: float, tau: float, rho: float, k_min: int,
                  gdu_ab, gdu_bb, beta_scale: float,
                  wk, wv, w_agg, gdu_aw, gdu_bw, mem_norm,
                  query_a=None, query_b=None, query_norm=None,
                  query_proj=None) -> None:
    """Torch tensors in the module's own layouts:
      wk/wv [NL, kv_dim, d_model], w_agg/gdu_aw/gdu_bw [NL, d_model],
      mem_norm [NL, q_dim].
    v5: low-rank query factors query_a [NL, q_dim, r] / query_b [NL, r, d]
    written AS-IS (no fold) — the C runtime computes (h@B^T)@A^T on the fly.
    v4: a pre-folded full-rank query_proj [NL, q_dim, d] (legacy).
    """
    nl = len(layer_ids)
    assert nl <= METIS_V6_MAX_LAYERS
    hdr = bytearray()
    # v5 when low-rank query factors supplied (no fold); v4 when a folded
    # full-rank query_proj is supplied; v3 otherwise.
    if query_a is not None and query_b is not None and query_norm is not None:
        is_v4, is_v5 = False, True
    elif query_proj is not None and query_norm is not None:
        is_v4, is_v5 = True, False
    else:
        is_v4, is_v5 = False, False
    hdr += (b"BNMEM5\x00\x00" if is_v5 else
            b"BNMEM4\x00\x00" if is_v4 else b"BNMEM3\x00\x00")
    hdr += struct.pack("<I", 5 if is_v5 else 4 if is_v4 else 3)
    hdr += struct.pack("<I", nl)
    ids = list(layer_ids) + [UNUSED_LAYER] * (METIS_V6_MAX_LAYERS - nl)
    for i in ids:
        hdr += struct.pack("<I", i)
    hdr += struct.pack("<IIII", d_model, kv_dim, q_dim, head_dim)
    hdr += struct.pack("<fff", gamma, tau, rho)
    hdr += struct.pack("<I", k_min)
    # v8: per-layer trainable biases ([NL] tensors) — mean-fold into the
    # per-layer tensor payloads below; header keeps a representative
    # scalar (mean) for v3 backcompat readers.
    try:
        gdu_ab_mean = float(gdu_ab.detach().mean())
        gdu_bb_mean = float(gdu_bb.detach().mean())
    except AttributeError:
        gdu_ab_mean, gdu_bb_mean = float(gdu_ab), float(gdu_bb)
    hdr += struct.pack("<fff", gdu_ab_mean, gdu_bb_mean, beta_scale)
    if is_v4:
        gdu_ab_np = gdu_ab.detach().cpu().numpy().reshape(-1)
        gdu_bb_np = gdu_bb.detach().cpu().numpy().reshape(-1)
        hdr += struct.pack("<I", 1)          # denom_plus_one
        if is_v5:
            r = query_a.shape[-1]
            hdr += struct.pack("<I", r)   # query_rank (v5)
        for v in gdu_ab_np:
            hdr += struct.pack("<f", float(v))
        for v in gdu_bb_np:
            hdr += struct.pack("<f", float(v))

    def f32(t):
        return t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()

    tensors = [
        ("wk", nl, kv_dim * d_model, f32(wk)),
        ("wv", nl, kv_dim * d_model, f32(wv)),
        ("w_agg", nl, d_model, f32(w_agg)),
        ("gdu_aw", nl, d_model, f32(gdu_aw)),
        ("gdu_bw", nl, d_model, f32(gdu_bw)),
        ("mem_norm", nl, q_dim, f32(mem_norm)),
    ]
    if is_v5:
        tensors.append(("query_norm", nl, head_dim, f32(query_norm)))
        tensors.append(("query_a", nl, q_dim * query_a.shape[-1], f32(query_a)))
        tensors.append(("query_b", nl, query_a.shape[-1] * d_model, f32(query_b)))
    elif is_v4:
        tensors.append(("query_proj", nl, q_dim * d_model, f32(query_proj)))
        tensors.append(("query_norm", nl, head_dim, f32(query_norm)))

    out = [bytes(hdr)]
    off = len(hdr)
    manifest = []
    for name, rows, cols, payload in tensors:
        manifest.append((name, rows, cols, off, _crc32(payload)))
        out.append(payload)
        off += len(payload)
    out.append(struct.pack("<I", len(manifest)))
    for name, rows, cols, o, crc in manifest:
        nb = name.encode()
        out.append(struct.pack("<I", len(nb)))
        out.append(nb)
        out.append(struct.pack("<IIQI", rows, cols, o, crc))
    with open(path, "wb") as f:
        f.write(b"".join(out))
