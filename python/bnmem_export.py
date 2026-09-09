"""BNMEM1/2/6 reader/writer for full- or low-rank memory models.

File layout (src/metis/metis_file.c):
  magic "BNMEM1\\0\\0" (legacy unbound) or "BNMEM2\\0\\0" (bound)
  u32 version (=1 or 2)
  u32 n_layers
  u32 layer_ids[32]  (unused = 0xFFFFFFFF)
  u32 d_model, kv_dim, q_dim, head_dim
  f32 gamma, tau, rho
  u32 k_min
  BNMEM6 only: u32 alpha_max_tokens; f32 alpha_max_fraction
  f32 gdu_ab, gdu_bb, beta_scale
  BNMEM2 only: u8 backbone_sha256[32]
  tensor payloads in fixed order (f32 row-major, layer-major). K/V are
  either legacy full matrices or SVD factors:
    wk_a/wv_a [n_layers x kv_dim*kv_rank]
    wk_b/wv_b [n_layers x kv_rank*d_model]
    w_agg    [n_layers x d_model]
    gdu_aw   [n_layers x d_model]
    gdu_bw   [n_layers x d_model]
    mem_norm [n_layers x q_dim]
  manifest: u32 tensor_count, then per tensor
    { u32 name_len; name; u32 rows; u32 cols; u64 byte_offset; u32 crc32 }
All little-endian; crc32 = zlib-style reflected poly 0xEDB88320.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np
import torch

METIS_V6_MAX_LAYERS = 32
UNUSED_LAYER = 0xFFFFFFFF
QUERY_ADD_BACKBONE = 0x80000000
QUERY_RANK_MASK = 0x0000FFFF
KV_RANK_SHIFT = 16
KV_RANK_MASK = 0x7FFF0000


def load_bnmem_v1(path: str) -> dict:
    """Load a BNMEM1/2 memory-model checkpoint."""

    def read_exact(handle, size):
        data = handle.read(size)
        if len(data) != size:
            raise ValueError("truncated BNMEM1 file")
        return data

    def read_u32(handle):
        return struct.unpack("<I", read_exact(handle, 4))[0]

    def read_f32(handle):
        return struct.unpack("<f", read_exact(handle, 4))[0]

    def read_array(handle, shape):
        count = int(np.prod(shape))
        data = np.fromfile(handle, dtype="<f4", count=count)
        if data.size != count:
            raise ValueError("truncated BNMEM1 tensor payload")
        return torch.from_numpy(data.reshape(shape).copy())

    with open(path, "rb") as handle:
        magic = read_exact(handle, 8)
        version = read_u32(handle)
        if not (
            (magic == b"BNMEM1\x00\x00" and version == 1)
            or (magic == b"BNMEM2\x00\x00" and version == 2)
            or (magic == b"BNMEM5\x00\x00" and version == 5)
            or (magic == b"BNMEM6\x00\x00" and version == 6)
            or (magic == b"BNMEM7\x00\x00" and version == 7)
        ):
            raise ValueError(
                f"unsupported BNMEM magic/version {magic!r}/{version}")
        n_layers = read_u32(handle)
        raw_ids = [read_u32(handle) for _ in range(METIS_V6_MAX_LAYERS)]
        layer_ids = raw_ids[:n_layers]
        d_model = read_u32(handle)
        kv_dim = read_u32(handle)
        q_dim = read_u32(handle)
        head_dim = read_u32(handle)
        gamma = read_f32(handle)
        tau = read_f32(handle)
        rho = read_f32(handle)
        k_min = read_u32(handle)
        alpha_max_tokens = read_u32(handle) if version in (6, 7) else 0
        alpha_max_fraction = (
            read_f32(handle) if version in (6, 7) else 0.0)
        _mean_ab = read_f32(handle)
        _mean_bb = read_f32(handle)
        beta_scale = read_f32(handle)
        denom_mode = read_u32(handle)
        encoded_rank = read_u32(handle)
        query_add_backbone = bool(encoded_rank & QUERY_ADD_BACKBONE)
        query_rank = encoded_rank & QUERY_RANK_MASK
        kv_rank = (encoded_rank & KV_RANK_MASK) >> KV_RANK_SHIFT
        backbone_sha256 = (
            read_exact(handle, 32) if version in (2, 6, 7) else None)
        tensors = {
            "gdu_ab": read_array(handle, (n_layers,)),
            "gdu_bb": read_array(handle, (n_layers,)),
        }
        if kv_rank > 0:
            tensors.update({
                "wk_a": read_array(
                    handle, (n_layers, kv_dim, kv_rank)),
                "wk_b": read_array(
                    handle, (n_layers, kv_rank, d_model)),
                "wv_a": read_array(
                    handle, (n_layers, kv_dim, kv_rank)),
                "wv_b": read_array(
                    handle, (n_layers, kv_rank, d_model)),
            })
        else:
            tensors.update({
                "wk": read_array(handle, (n_layers, kv_dim, d_model)),
                "wv": read_array(handle, (n_layers, kv_dim, d_model)),
            })
        tensors.update({
            "w_agg": read_array(handle, (n_layers, d_model)),
            "gdu_aw": read_array(handle, (n_layers, d_model)),
            "gdu_bw": read_array(handle, (n_layers, d_model)),
            "mem_norm": read_array(handle, (n_layers, q_dim)),
        })
        if version == 7:
            tensors.update({
                "fusion_gate_w": read_array(
                    handle, (n_layers, d_model)),
                "fusion_gate_b": read_array(handle, (n_layers,)),
            })
        if query_rank > 0:
            tensors.update({
                "query_norm": read_array(handle, (n_layers, head_dim)),
                "query_a": read_array(
                    handle, (n_layers, q_dim, query_rank)),
                "query_b": read_array(
                    handle, (n_layers, query_rank, d_model)),
            })
        else:
            tensors.update({
                "query_norm": read_array(handle, (n_layers, head_dim)),
                "query_proj": read_array(
                    handle, (n_layers, q_dim, d_model)),
            })

        manifest_count = read_u32(handle)
        manifest_names = []
        for _ in range(manifest_count):
            name = read_exact(handle, read_u32(handle)).decode("utf-8")
            _rows = read_u32(handle)
            _cols = read_u32(handle)
            _offset = struct.unpack("<Q", read_exact(handle, 8))[0]
            _crc = read_u32(handle)
            manifest_names.append(name)
        if kv_rank > 0:
            expected_names = [
                "wk_a", "wk_b", "wv_a", "wv_b",
                "w_agg", "gdu_aw", "gdu_bw", "mem_norm",
            ]
        else:
            expected_names = [
                "wk", "wv", "w_agg", "gdu_aw", "gdu_bw", "mem_norm",
            ]
        if version == 7:
            expected_names.extend(["fusion_gate_w", "fusion_gate_b"])
        if query_rank > 0:
            expected_names.extend([
                "query_norm", "query_a", "query_b",
            ])
        else:
            expected_names.extend(["query_norm", "query_proj"])
        if manifest_names != expected_names or handle.read(1):
            raise ValueError(
                f"unexpected BNMEM1 manifest: {manifest_names}")

    return {
        "layer_ids": layer_ids,
        "d_model": d_model,
        "kv_dim": kv_dim,
        "q_dim": q_dim,
        "head_dim": head_dim,
        "gamma": gamma,
        "tau": tau,
        "rho": rho,
        "k_min": k_min,
        "alpha_max_tokens": alpha_max_tokens,
        "alpha_max_fraction": alpha_max_fraction,
        "beta_scale": beta_scale,
        "denom_mode": denom_mode,
        "query_rank": query_rank,
        "kv_rank": kv_rank,
        "query_add_backbone": query_add_backbone,
        "fusion_mode": "residual_gate" if version == 7 else "fixed",
        "backbone_sha256": backbone_sha256,
        "tensors": tensors,
    }


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def save_bnmem_v3(path: str, *, layer_ids: list[int], d_model: int,
                  kv_dim: int, q_dim: int, head_dim: int,
                  gamma: float, tau: float, rho: float, k_min: int,
                  gdu_ab, gdu_bb, beta_scale: float,
                  w_agg, gdu_aw, gdu_bw, mem_norm,
                  alpha_max_tokens: int = 0,
                  alpha_max_fraction: float = 0.0,
                  wk=None, wv=None,
                  wk_a=None, wk_b=None, wv_a=None, wv_b=None,
                  query_a=None, query_b=None, query_norm=None,
                  query_proj=None, denom_mode: int = 1,
                  query_add_backbone: bool = False,
                  fusion_gate_w=None, fusion_gate_b=None,
                  backbone_sha256: bytes | str | None = None) -> None:
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
    # Unified v1 format: query_rank > 0 stores factors, query_rank == 0
    # stores a folded full-rank query projection.
    if query_a is not None and query_b is not None and query_norm is not None:
        is_v4, is_v5 = False, True
    elif query_proj is not None and query_norm is not None:
        is_v4, is_v5 = True, False
    else:
        is_v4, is_v5 = False, False
    if not (is_v4 or is_v5):
        raise ValueError("BNMEM1 requires a full or low-rank query")
    if isinstance(backbone_sha256, str):
        backbone_sha256 = bytes.fromhex(backbone_sha256)
    if backbone_sha256 is not None and len(backbone_sha256) != 32:
        raise ValueError("backbone SHA-256 must contain exactly 32 bytes")
    if alpha_max_tokens < 0:
        raise ValueError("alpha_max_tokens must be non-negative")
    if not 0.0 <= alpha_max_fraction <= 1.0:
        raise ValueError("alpha_max_fraction must be in [0, 1]")
    gated_fusion = (
        fusion_gate_w is not None and fusion_gate_b is not None)
    if (fusion_gate_w is None) != (fusion_gate_b is None):
        raise ValueError("provide both fusion gate tensors or neither")
    bounded_selection = (
        alpha_max_tokens > 0 or alpha_max_fraction > 0.0)
    if (bounded_selection or gated_fusion) and backbone_sha256 is None:
        raise ValueError("BNMEM6/7 requires backbone binding")
    hdr += (
        b"BNMEM7\x00\x00" if gated_fusion else
        (b"BNMEM6\x00\x00" if bounded_selection
        else (
            b"BNMEM2\x00\x00"
            if backbone_sha256 is not None else b"BNMEM1\x00\x00"
        ))
    )
    hdr += struct.pack(
        "<I", 7 if gated_fusion else (6 if bounded_selection
        else (2 if backbone_sha256 is not None else 1))
        )
    hdr += struct.pack("<I", nl)
    ids = list(layer_ids) + [UNUSED_LAYER] * (METIS_V6_MAX_LAYERS - nl)
    for i in ids:
        hdr += struct.pack("<I", i)
    hdr += struct.pack("<IIII", d_model, kv_dim, q_dim, head_dim)
    hdr += struct.pack("<fff", gamma, tau, rho)
    hdr += struct.pack("<I", k_min)
    if bounded_selection or gated_fusion:
        hdr += struct.pack("<If", alpha_max_tokens, alpha_max_fraction)
    # v8: per-layer trainable biases ([NL] tensors) — mean-fold into the
    # per-layer tensor payloads below; header keeps a representative
    # scalar (mean) for v3 backcompat readers.
    try:
        gdu_ab_mean = float(gdu_ab.detach().mean())
        gdu_bb_mean = float(gdu_bb.detach().mean())
    except AttributeError:
        gdu_ab_mean, gdu_bb_mean = float(gdu_ab), float(gdu_bb)
    hdr += struct.pack("<fff", gdu_ab_mean, gdu_bb_mean, beta_scale)
    gdu_ab_np = gdu_ab.detach().cpu().numpy().reshape(-1)
    gdu_bb_np = gdu_bb.detach().cpu().numpy().reshape(-1)
    if denom_mode not in (1, 2):
        raise ValueError(f"unsupported denominator mode {denom_mode}")
    hdr += struct.pack("<I", denom_mode)
    r = query_a.shape[-1] if is_v5 else 0
    low_rank_kv = all(
        tensor is not None for tensor in (wk_a, wk_b, wv_a, wv_b))
    full_rank_kv = wk is not None and wv is not None
    if low_rank_kv == full_rank_kv:
        raise ValueError(
            "provide exactly one of full-rank or low-rank K/V tensors")
    kv_rank = wk_a.shape[-1] if low_rank_kv else 0
    if r > QUERY_RANK_MASK:
        raise ValueError("query rank does not fit BNMEM1 header")
    if kv_rank >= (1 << 15):
        raise ValueError("K/V rank does not fit BNMEM1 header")
    encoded_rank = (
        r | (kv_rank << KV_RANK_SHIFT)
        | (QUERY_ADD_BACKBONE if query_add_backbone else 0))
    hdr += struct.pack("<I", encoded_rank)
    if backbone_sha256 is not None:
        hdr += backbone_sha256
    for v in gdu_ab_np:
        hdr += struct.pack("<f", float(v))
    for v in gdu_bb_np:
        hdr += struct.pack("<f", float(v))

    def f32(t):
        return t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()

    tensors = []
    if low_rank_kv:
        tensors.extend([
            ("wk_a", nl, kv_dim * kv_rank, f32(wk_a)),
            ("wk_b", nl, kv_rank * d_model, f32(wk_b)),
            ("wv_a", nl, kv_dim * kv_rank, f32(wv_a)),
            ("wv_b", nl, kv_rank * d_model, f32(wv_b)),
        ])
    else:
        tensors.extend([
            ("wk", nl, kv_dim * d_model, f32(wk)),
            ("wv", nl, kv_dim * d_model, f32(wv)),
        ])
    tensors.extend([
        ("w_agg", nl, d_model, f32(w_agg)),
        ("gdu_aw", nl, d_model, f32(gdu_aw)),
        ("gdu_bw", nl, d_model, f32(gdu_bw)),
        ("mem_norm", nl, q_dim, f32(mem_norm)),
    ])
    if gated_fusion:
        tensors.extend([
            ("fusion_gate_w", nl, d_model, f32(fusion_gate_w)),
            ("fusion_gate_b", nl, 1, f32(fusion_gate_b)),
        ])
    if is_v5:
        tensors.append(("query_norm", nl, head_dim, f32(query_norm)))
        tensors.append(("query_a", nl, q_dim * query_a.shape[-1], f32(query_a)))
        tensors.append(("query_b", nl, query_a.shape[-1] * d_model, f32(query_b)))
    elif is_v4:
        tensors.append(("query_norm", nl, head_dim, f32(query_norm)))
        tensors.append(("query_proj", nl, q_dim * d_model, f32(query_proj)))

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
