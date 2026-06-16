"""
Step-by-step forward pass verification for BitNet 1.58B model.
Compares Python intermediate values with C code's debug output.

C code reference values:
  After embedding (BOS): norm=2.792910, hidden[0..4]=[-0.0121,-0.0121,0.0309,-0.0982,-0.0265]
  After block 0: norm=58.218143, hidden[0..4]=[-0.9022,-0.1161,0.8770,-1.2503,-2.3367]
  q norm=22.198961, q[0..4]=[0.2617,0.3134,0.1892,0.2192,0.6579]
  k norm=20.174088, k[0..4]=[0.0619,-0.0535,-0.0716,-0.1585,0.3245]

IMPORTANT: The C code's bitnet_rms_norm() modifies hidden IN-PLACE, so the
residual add after attention/FFN uses the NORMED hidden, not the original.
This is a bug in the C code - the standard LLaMA architecture uses:
  hidden = original + sublayer_output
But the C code does:
  hidden = RMSNorm(original) + sublayer_output
We implement BOTH paths to compare.
"""

import struct
import numpy as np
import sys
import os
import math

GGUF_FILE = "/Users/cuick/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf"

# Model parameters
EMB_DIM = 2048
BLOCK_COUNT = 28
N_HEADS = 16
N_KV_HEADS = 2
FFN_DIM = 6144
HEAD_DIM = EMB_DIM // N_HEADS  # 128
KV_DIM = N_KV_HEADS * HEAD_DIM  # 256
ROPE_DIM = 128
ROPE_FREQ_BASE = 10000.0
RMS_NORM_EPS = 1e-6
VOCAB_SIZE = 73448


def fp16_to_fp32(h):
    """Convert IEEE 754 half-precision (FP16) to single-precision (FP32)."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return 0.0 if sign == 0 else -0.0
        value = (mant / 1024.0) * (2.0 ** (-14))
        return -value if sign else value

    if exp == 31:
        if mant == 0:
            return float('inf') if sign == 0 else float('-inf')
        return float('nan')

    value = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -value if sign else value


# ============================================================
# Q4_K dequantization (for embeddings)
# ============================================================

def get_scale_min_k4(j, scales):
    if j < 4:
        sc = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, m


def dequantize_q4k_block(data, offset):
    """Dequantize one Q4_K block (144 bytes, 256 elements)."""
    d_raw = struct.unpack_from('<H', data, offset)[0]
    dmin_raw = struct.unpack_from('<H', data, offset + 2)[0]
    d = fp16_to_fp32(d_raw)
    dmin = fp16_to_fp32(dmin_raw)
    scales = list(data[offset + 4: offset + 16])
    qs = list(data[offset + 16: offset + 144])

    out = np.zeros(256, dtype=np.float32)
    is_idx = 0
    q_ptr = 0

    for j in range(0, 256, 64):
        sc0, m0 = get_scale_min_k4(is_idx + 0, scales)
        d1 = d * float(sc0)
        m1 = dmin * float(m0)

        sc1, m1_val = get_scale_min_k4(is_idx + 1, scales)
        d2 = d * float(sc1)
        m2 = dmin * float(m1_val)

        for l in range(32):
            out[j + l] = d1 * float(qs[q_ptr + l] & 0xF) - m1
        for l in range(32):
            out[j + 32 + l] = d2 * float(qs[q_ptr + l] >> 4) - m2

        q_ptr += 32
        is_idx += 2

    return out


def dequantize_embedding_q4k(emb_data, token_id, emb_dim):
    """Dequantize a single token's embedding from Q4_K data."""
    row = emb_data[token_id]
    n_blocks = emb_dim // 256
    block_size = 144

    emb_values = np.zeros(emb_dim, dtype=np.float32)
    row_bytes = bytes(row)
    for b in range(n_blocks):
        block_offset = b * block_size
        block_vals = dequantize_q4k_block(row_bytes, block_offset)
        emb_values[b * 256:(b + 1) * 256] = block_vals

    return emb_values


# ============================================================
# TQ2_0 dequantization and matmul (matching llama.cpp interleaved layout)
# ============================================================

def tq2_0_matmul(weight_data, out_dim, in_dim, vec):
    """TQ2_0 matmul matching C code's bitnet_tq2_0_matmul_vector exactly.

    weight_data: numpy array of shape (out_dim, bytes_per_row), dtype uint8
    out_dim: number of output elements
    in_dim: number of input elements (must match len(vec))
    vec: input vector of length in_dim
    """
    blocks_per_row = in_dim // 256
    block_size = 66
    output = np.zeros(out_dim, dtype=np.float32)

    for j in range(out_dim):
        row_bytes = bytes(weight_data[j])
        sum_val = 0.0

        for k in range(blocks_per_row):
            block_offset = k * block_size
            d_raw = struct.unpack_from('<H', row_bytes, block_offset + 64)[0]
            d = fp16_to_fp32(d_raw)
            block_sum = 0.0
            vec_base = k * 256

            # Interleaved layout matching llama.cpp
            for jj in range(0, 64, 32):
                for l in range(4):
                    for m in range(32):
                        q = (row_bytes[block_offset + jj + m] >> (l * 2)) & 3
                        vec_idx = vec_base + (jj // 32) * 128 + l * 32 + m
                        block_sum += (q - 1) * vec[vec_idx]

            sum_val += block_sum * d

        output[j] = sum_val

    return output


# ============================================================
# RMSNorm (in-place, matching C code)
# ============================================================

def rms_norm_inplace(x, weight, eps=RMS_NORM_EPS):
    """RMSNorm: x = x * inv_rms * weight (modifies x in-place, matching C code)"""
    n = len(x)
    sum_sq = float(np.sum(x * x))
    inv_rms = 1.0 / math.sqrt(sum_sq / n + eps)
    x *= inv_rms
    x *= weight
    return x


def rms_norm(x, weight, eps=RMS_NORM_EPS):
    """RMSNorm: returns new array (does not modify x)"""
    n = len(x)
    sum_sq = float(np.sum(x * x))
    inv_rms = 1.0 / math.sqrt(sum_sq / n + eps)
    return x * inv_rms * weight


# ============================================================
# RoPE
# ============================================================

def rope_apply(x, n_heads, head_dim, rope_dim, position, freq_base=ROPE_FREQ_BASE):
    """Apply RoPE. x has shape (n_heads * head_dim,)."""
    out = x.copy()
    for h in range(n_heads):
        for j in range(rope_dim // 2):
            idx0 = h * head_dim + 2 * j
            idx1 = h * head_dim + 2 * j + 1
            theta = 1.0 / (freq_base ** (2.0 * j / head_dim))
            cos_theta = math.cos(position * theta)
            sin_theta = math.sin(position * theta)
            x0 = out[idx0]
            x1 = out[idx1]
            out[idx0] = x0 * cos_theta - x1 * sin_theta
            out[idx1] = x0 * sin_theta + x1 * cos_theta
    return out


# ============================================================
# SiLU
# ============================================================

def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))


# ============================================================
# Softmax
# ============================================================

def softmax(x):
    max_val = np.max(x)
    e = np.exp(x - max_val)
    return e / np.sum(e)


def print_vals(name, arr, n=5):
    """Print first n elements and norm of an array."""
    vals = ', '.join(f'{v:.4f}' for v in arr[:n])
    norm = np.linalg.norm(arr)
    print(f"  {name}[0..{n-1}] = [{vals}]")
    print(f"  {name} norm = {norm:.6f}")


# ============================================================
# Main forward pass
# ============================================================

def main():
    print("=" * 70)
    print("BitNet 1.58B Forward Pass Verification (Block 0)")
    print("=" * 70)

    # Load GGUF model
    print("\n--- Loading GGUF model ---")
    import gguf
    reader = gguf.GGUFReader(GGUF_FILE)

    # Build tensor dict
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t

    print(f"Number of tensors: {len(tensors)}")

    # Verify key tensors exist
    required = [
        'token_embd.weight',
        'blk.0.attn_norm.weight', 'blk.0.attn_q.weight',
        'blk.0.attn_k.weight', 'blk.0.attn_v.weight',
        'blk.0.attn_output.weight',
        'blk.0.ffn_norm.weight', 'blk.0.ffn_gate.weight',
        'blk.0.ffn_up.weight', 'blk.0.ffn_down.weight',
    ]
    for name in required:
        if name not in tensors:
            print(f"ERROR: Required tensor '{name}' not found!")
            sys.exit(1)
        t = tensors[name]
        print(f"  {name}: shape={t.shape}, type={t.tensor_type}, data_shape={t.data.shape}")

    # Get tensor data references
    emb_data = tensors['token_embd.weight'].data
    attn_norm_w = np.array(tensors['blk.0.attn_norm.weight'].data, dtype=np.float32)
    ffn_norm_w = np.array(tensors['blk.0.ffn_norm.weight'].data, dtype=np.float32)

    attn_q_data = tensors['blk.0.attn_q.weight'].data
    attn_k_data = tensors['blk.0.attn_k.weight'].data
    attn_v_data = tensors['blk.0.attn_v.weight'].data
    attn_out_data = tensors['blk.0.attn_output.weight'].data
    ffn_gate_data = tensors['blk.0.ffn_gate.weight'].data
    ffn_up_data = tensors['blk.0.ffn_up.weight'].data
    ffn_down_data = tensors['blk.0.ffn_down.weight'].data

    # ============================================================
    # Step 1: Embedding lookup for token 1 (BOS)
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 1: Embedding lookup for token 1 (BOS)")
    print("=" * 70)

    token_id = 1  # BOS
    hidden = dequantize_embedding_q4k(emb_data, token_id, EMB_DIM)

    c_emb_norm = 2.792910
    c_emb_first5 = [-0.0121, -0.0121, 0.0309, -0.0982, -0.0265]
    print_vals("hidden", hidden)
    print(f"  C code:  norm={c_emb_norm:.6f}, hidden[0..4]={c_emb_first5}")
    print(f"  Match: {'OK' if abs(np.linalg.norm(hidden) - c_emb_norm) < 0.01 else 'MISMATCH'}")

    # ============================================================
    # Step 2: Block 0 forward pass - MATCHING C CODE BEHAVIOR
    # (C code's rms_norm modifies hidden in-place, so residual adds
    #  use the normed value instead of the original - this is a BUG)
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: Block 0 forward pass - MATCHING C CODE (with residual bug)")
    print("=" * 70)

    pos = 0

    # --- attn RMSNorm (modifies hidden in-place in C code) ---
    print("\n  --- attn RMSNorm (in-place, matching C code) ---")
    hidden_c = hidden.copy()
    rms_norm_inplace(hidden_c, attn_norm_w)
    print_vals("hidden (after norm)", hidden_c)

    # --- Q projection ---
    print("\n  --- Q projection ---")
    q = tq2_0_matmul(attn_q_data, EMB_DIM, EMB_DIM, hidden_c)
    print_vals("q", q)
    c_q_norm = 22.198961
    c_q_first5 = [0.2617, 0.3134, 0.1892, 0.2192, 0.6579]
    print(f"  C code:  q norm={c_q_norm:.6f}, q[0..4]={c_q_first5}")
    print(f"  Match: {'OK' if abs(np.linalg.norm(q) - c_q_norm) < 0.01 else 'MISMATCH'}")

    # --- K projection ---
    print("\n  --- K projection ---")
    k = tq2_0_matmul(attn_k_data, KV_DIM, EMB_DIM, hidden_c)
    print_vals("k", k)
    c_k_norm = 20.174088
    c_k_first5 = [0.0619, -0.0535, -0.0716, -0.1585, 0.3245]
    print(f"  C code:  k norm={c_k_norm:.6f}, k[0..4]={c_k_first5}")
    print(f"  Match: {'OK' if abs(np.linalg.norm(k) - c_k_norm) < 0.01 else 'MISMATCH'}")

    # --- V projection ---
    print("\n  --- V projection ---")
    v = tq2_0_matmul(attn_v_data, KV_DIM, EMB_DIM, hidden_c)
    print_vals("v", v)

    # --- RoPE on Q and K ---
    print("\n  --- RoPE on Q and K (position=0, identity) ---")
    q = rope_apply(q, N_HEADS, HEAD_DIM, ROPE_DIM, pos, ROPE_FREQ_BASE)
    k = rope_apply(k, N_KV_HEADS, HEAD_DIM, ROPE_DIM, pos, ROPE_FREQ_BASE)
    print_vals("q (after RoPE)", q)
    print_vals("k (after RoPE)", k)

    # --- KV cache store ---
    print("\n  --- KV cache store ---")
    key_cache = np.zeros((1, KV_DIM), dtype=np.float32)
    value_cache = np.zeros((1, KV_DIM), dtype=np.float32)
    key_cache[0, :] = k
    value_cache[0, :] = v

    # --- Attention computation (GQA) ---
    print("\n  --- Attention computation (GQA) ---")
    n_positions = pos + 1
    inv_sqrt_head_dim = 1.0 / math.sqrt(HEAD_DIM)
    attn_buffer = np.zeros(EMB_DIM, dtype=np.float32)
    scores_arr = np.zeros(n_positions, dtype=np.float32)

    for query_head in range(N_HEADS):
        kv_head = query_head // (N_HEADS // N_KV_HEADS)
        q_head = q[query_head * HEAD_DIM:(query_head + 1) * HEAD_DIM]

        for past_pos in range(n_positions):
            k_head = key_cache[past_pos, kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM]
            dot = float(np.dot(q_head, k_head)) * inv_sqrt_head_dim
            scores_arr[past_pos] = dot

        scores_softmax = softmax(scores_arr[:n_positions])

        attn_head = attn_buffer[query_head * HEAD_DIM:(query_head + 1) * HEAD_DIM]
        for past_pos in range(n_positions):
            w = scores_softmax[past_pos]
            v_head = value_cache[past_pos, kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM]
            attn_head += w * v_head

    print_vals("attn_buffer", attn_buffer)

    # --- O projection ---
    print("\n  --- O projection ---")
    tmp_out = tq2_0_matmul(attn_out_data, EMB_DIM, EMB_DIM, attn_buffer)
    print_vals("tmp_out (O proj output)", tmp_out)

    # --- Residual add (C code bug: adds to NORMED hidden) ---
    print("\n  --- Residual add (C code: hidden = normed + tmp_out) ---")
    hidden_c = hidden_c + tmp_out  # C code: hidden was already normed in-place
    print_vals("hidden (after attn residual, C code path)", hidden_c)

    # --- FFN RMSNorm (modifies hidden in-place in C code) ---
    print("\n  --- FFN RMSNorm (in-place, matching C code) ---")
    rms_norm_inplace(hidden_c, ffn_norm_w)
    print_vals("hidden (after FFN norm)", hidden_c)

    # --- Gate projection ---
    print("\n  --- Gate projection ---")
    gate = tq2_0_matmul(ffn_gate_data, FFN_DIM, EMB_DIM, hidden_c)
    print_vals("gate", gate)

    # --- Up projection ---
    print("\n  --- Up projection ---")
    up = tq2_0_matmul(ffn_up_data, FFN_DIM, EMB_DIM, hidden_c)
    print_vals("up", up)

    # --- SiLU(gate) * up ---
    print("\n  --- SiLU(gate) * up ---")
    gate_silu = silu(gate) * up
    print_vals("silu(gate)*up", gate_silu)

    # --- Down projection ---
    print("\n  --- Down projection ---")
    down = tq2_0_matmul(ffn_down_data, EMB_DIM, FFN_DIM, gate_silu)
    print_vals("down", down)

    # --- Residual add (C code bug: adds to NORMED hidden) ---
    print("\n  --- Residual add (C code: hidden = normed + down) ---")
    hidden_c = hidden_c + down
    print_vals("hidden (after FFN residual, C code path)", hidden_c)

    # ============================================================
    # Step 3: Block 0 forward pass - CORRECT BEHAVIOR
    # (Standard LLaMA: residual = original + sublayer_output)
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: Block 0 forward pass - CORRECT (standard LLaMA residual)")
    print("=" * 70)

    # --- attn RMSNorm ---
    print("\n  --- attn RMSNorm ---")
    hidden_correct = hidden.copy()  # original embedding
    residual_attn = hidden_correct.copy()  # save for residual
    hidden_normed = rms_norm(hidden_correct, attn_norm_w)
    print_vals("hidden_normed", hidden_normed)

    # --- Q/K/V projections ---
    print("\n  --- Q/K/V projections ---")
    q2 = tq2_0_matmul(attn_q_data, EMB_DIM, EMB_DIM, hidden_normed)
    k2 = tq2_0_matmul(attn_k_data, KV_DIM, EMB_DIM, hidden_normed)
    v2 = tq2_0_matmul(attn_v_data, KV_DIM, EMB_DIM, hidden_normed)
    print_vals("q", q2)
    print_vals("k", k2)
    print_vals("v", v2)

    # --- RoPE ---
    q2 = rope_apply(q2, N_HEADS, HEAD_DIM, ROPE_DIM, pos, ROPE_FREQ_BASE)
    k2 = rope_apply(k2, N_KV_HEADS, HEAD_DIM, ROPE_DIM, pos, ROPE_FREQ_BASE)

    # --- Attention ---
    key_cache2 = np.zeros((1, KV_DIM), dtype=np.float32)
    value_cache2 = np.zeros((1, KV_DIM), dtype=np.float32)
    key_cache2[0, :] = k2
    value_cache2[0, :] = v2

    attn_buffer2 = np.zeros(EMB_DIM, dtype=np.float32)
    scores2 = np.zeros(1, dtype=np.float32)
    for query_head in range(N_HEADS):
        kv_head = query_head // (N_HEADS // N_KV_HEADS)
        q_head = q2[query_head * HEAD_DIM:(query_head + 1) * HEAD_DIM]
        k_head = key_cache2[0, kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM]
        scores2[0] = float(np.dot(q_head, k_head)) * inv_sqrt_head_dim
        sw = softmax(scores2)
        v_head = value_cache2[0, kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM]
        attn_buffer2[query_head * HEAD_DIM:(query_head + 1) * HEAD_DIM] = sw[0] * v_head

    # --- O projection ---
    tmp_out2 = tq2_0_matmul(attn_out_data, EMB_DIM, EMB_DIM, attn_buffer2)
    print_vals("tmp_out (O proj)", tmp_out2)

    # --- Residual add (CORRECT: original + tmp_out) ---
    print("\n  --- Residual add (CORRECT: original + tmp_out) ---")
    hidden_correct = residual_attn + tmp_out2
    print_vals("hidden (after attn residual, correct)", hidden_correct)

    # --- FFN RMSNorm ---
    residual_ffn = hidden_correct.copy()  # save for residual
    hidden_normed2 = rms_norm(hidden_correct, ffn_norm_w)
    print_vals("hidden_normed (after FFN norm)", hidden_normed2)

    # --- Gate/Up projections ---
    gate2 = tq2_0_matmul(ffn_gate_data, FFN_DIM, EMB_DIM, hidden_normed2)
    up2 = tq2_0_matmul(ffn_up_data, FFN_DIM, EMB_DIM, hidden_normed2)
    gate_silu2 = silu(gate2) * up2

    # --- Down projection ---
    down2 = tq2_0_matmul(ffn_down_data, EMB_DIM, FFN_DIM, gate_silu2)
    print_vals("down", down2)

    # --- Residual add (CORRECT: pre-norm + down) ---
    print("\n  --- Residual add (CORRECT: pre-norm + down) ---")
    hidden_correct = residual_ffn + down2
    print_vals("hidden (after FFN residual, correct)", hidden_correct)

    # ============================================================
    # Final comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("Final Comparison")
    print("=" * 70)

    c_block0_norm = 58.218143
    c_block0_first5 = [-0.9022, -0.1161, 0.8770, -1.2503, -2.3367]

    print(f"\n  C code reference:")
    print(f"    norm={c_block0_norm:.6f}, hidden[0..4]={c_block0_first5}")

    print(f"\n  Python (matching C code behavior - normed residual):")
    h_c_norm = np.linalg.norm(hidden_c)
    print(f"    norm={h_c_norm:.6f}, hidden[0..4]=[{', '.join(f'{v:.4f}' for v in hidden_c[:5])}]")
    print(f"    Diff from C: {abs(h_c_norm - c_block0_norm):.6f} {'OK' if abs(h_c_norm - c_block0_norm) < 1.0 else 'MISMATCH'}")

    print(f"\n  Python (correct LLaMA behavior - original residual):")
    h_corr_norm = np.linalg.norm(hidden_correct)
    print(f"    norm={h_corr_norm:.6f}, hidden[0..4]=[{', '.join(f'{v:.4f}' for v in hidden_correct[:5])}]")
    print(f"    Diff from C: {abs(h_corr_norm - c_block0_norm):.6f}")

    # ============================================================
    # Diagnosis
    # ============================================================
    print("\n" + "=" * 70)
    print("Diagnosis")
    print("=" * 70)

    if abs(h_c_norm - c_block0_norm) < 1.0:
        print("\n  C code path matches! The C code has a residual connection bug:")
        print("  - bitnet_rms_norm() modifies hidden IN-PLACE")
        print("  - The residual add then uses the NORMED hidden instead of the original")
        print("  - Standard LLaMA: hidden = original + sublayer_output")
        print("  - C code bug:     hidden = RMSNorm(original) + sublayer_output")
    elif abs(h_corr_norm - c_block0_norm) < 1.0:
        print("\n  Correct LLaMA path matches C code output.")
        print("  The C code's residual connections are correct.")
    else:
        print("\n  Neither path matches the C code output exactly.")
        print("  There may be additional differences beyond the residual bug.")

        # Let's also try: what if C code saves residual BEFORE norm?
        # Actually let me check: maybe C code does something else
        # Let me trace more carefully what the C code does
        print("\n  Additional investigation needed...")

        # Try: what if the C code is actually correct and I'm wrong about the bug?
        # Let me re-examine: in C code, hidden is modified by rms_norm in-place
        # Then Q/K/V use the normed hidden
        # Then tmp_out = O_proj(attn_buffer)
        # Then hidden += tmp_out (hidden is normed at this point)
        # This means: hidden_after_attn = normed + tmp_out
        # Then hidden is normed again for FFN
        # Then gate/up use the double-normed hidden
        # Then down = down_proj(silu(gate)*up)
        # Then hidden += down (hidden is double-normed at this point)
        # This means: hidden_after_ffn = RMSNorm(normed + tmp_out) + down

        # The correct behavior should be:
        # hidden_after_attn = original + tmp_out
        # hidden_after_ffn = (original + tmp_out) + down

    # ============================================================
    # Detailed step-by-step comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("Detailed Step-by-Step Values (C code path)")
    print("=" * 70)

    # Re-run the C code path with detailed output
    h = dequantize_embedding_q4k(emb_data, 1, EMB_DIM)
    print(f"\n  [A] After embedding:")
    print_vals("  hidden", h)

    rms_norm_inplace(h, attn_norm_w)
    print(f"\n  [B] After attn RMSNorm (hidden modified in-place):")
    print_vals("  hidden", h)

    q = tq2_0_matmul(attn_q_data, EMB_DIM, EMB_DIM, h)
    k = tq2_0_matmul(attn_k_data, KV_DIM, EMB_DIM, h)
    v = tq2_0_matmul(attn_v_data, KV_DIM, EMB_DIM, h)
    print(f"\n  [C] After Q/K/V projections:")
    print_vals("  q", q)
    print_vals("  k", k)
    print_vals("  v", v)

    q = rope_apply(q, N_HEADS, HEAD_DIM, ROPE_DIM, pos, ROPE_FREQ_BASE)
    k = rope_apply(k, N_KV_HEADS, HEAD_DIM, ROPE_DIM, pos, ROPE_FREQ_BASE)
    print(f"\n  [D] After RoPE:")
    print_vals("  q", q)
    print_vals("  k", k)

    # Attention
    kc = np.zeros((1, KV_DIM), dtype=np.float32)
    vc = np.zeros((1, KV_DIM), dtype=np.float32)
    kc[0, :] = k
    vc[0, :] = v
    ab = np.zeros(EMB_DIM, dtype=np.float32)
    sc = np.zeros(1, dtype=np.float32)
    for qh in range(N_HEADS):
        kvh = qh // (N_HEADS // N_KV_HEADS)
        qh_vec = q[qh * HEAD_DIM:(qh + 1) * HEAD_DIM]
        kh_vec = kc[0, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]
        sc[0] = float(np.dot(qh_vec, kh_vec)) * inv_sqrt_head_dim
        sw = softmax(sc)
        vh_vec = vc[0, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]
        ab[qh * HEAD_DIM:(qh + 1) * HEAD_DIM] = sw[0] * vh_vec

    print(f"\n  [E] After attention:")
    print_vals("  attn_buffer", ab)

    tmp = tq2_0_matmul(attn_out_data, EMB_DIM, EMB_DIM, ab)
    print(f"\n  [F] After O projection:")
    print_vals("  tmp_out", tmp)

    h = h + tmp  # C code: hidden was normed, so this is normed + tmp_out
    print(f"\n  [G] After attn residual (normed + tmp_out):")
    print_vals("  hidden", h)

    rms_norm_inplace(h, ffn_norm_w)
    print(f"\n  [H] After FFN RMSNorm:")
    print_vals("  hidden", h)

    g = tq2_0_matmul(ffn_gate_data, FFN_DIM, EMB_DIM, h)
    u = tq2_0_matmul(ffn_up_data, FFN_DIM, EMB_DIM, h)
    print(f"\n  [I] After gate/up projections:")
    print_vals("  gate", g)
    print_vals("  up", u)

    gs = silu(g) * u
    print(f"\n  [J] After SiLU(gate)*up:")
    print_vals("  silu_gate_up", gs)

    d = tq2_0_matmul(ffn_down_data, EMB_DIM, FFN_DIM, gs)
    print(f"\n  [K] After down projection:")
    print_vals("  down", d)

    h = h + d  # C code: hidden was normed, so this is ffn_normed + down
    print(f"\n  [L] After FFN residual (ffn_normed + down):")
    print_vals("  hidden", h)

    print(f"\n  C code reference: norm={c_block0_norm:.6f}, hidden[0..4]={c_block0_first5}")
    print(f"  Python result:    norm={np.linalg.norm(h):.6f}, hidden[0..4]=[{', '.join(f'{v:.4f}' for v in h[:5])}]")

    print("\n" + "=" * 70)
    print("Verification complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
