"""
Verify Q4_K embedding dequantization and forward pass by comparing with C code output.

1. Read GGUF model file using the gguf Python package
2. Find token_embd.weight tensor
3. Dequantize embedding for token ID 1 (BOS) using Q4_K algorithm
4. Print first 10 values and L2 norm
5. Compare with C code's debug output: norm=2.792910, hidden[0..4]=[-0.0121,-0.0121,0.0309,-0.0982,-0.0265]
6. Use llama-cpp-python to generate reference logits for "The" (with BOS)
7. Compare with our C code's logits saved at build/our_logits.bin
"""

import struct
import numpy as np
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GGUF_FILE = os.path.join(ROOT, "models", "bitcpm4-1b-tq2_0.gguf")
OUR_LOGITS_PATH = os.path.join(ROOT, "build", "our_logits.bin")
REF_LOGITS_PATH = os.path.join(ROOT, "tests", "ref_logits.npy")


def fp16_to_fp32(h):
    """Convert IEEE 754 half-precision (FP16) to single-precision (FP32).
    Handles denormalized numbers properly, matching C code's implementation."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return 0.0 if sign == 0 else -0.0
        # Denormalized: value = (-1)^sign * 2^(-14) * (mant/1024)
        value = (mant / 1024.0) * (2.0 ** (-14))
        return -value if sign else value

    if exp == 31:
        if mant == 0:
            return float('inf') if sign == 0 else float('-inf')
        return float('nan')

    # Normalized: value = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    value = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -value if sign else value


def get_scale_min_k4(j, scales):
    """Extract scale and min from Q4_K scales array, matching llama.cpp's get_scale_min_k4.
    For j < 4: scale = scales[j] & 63, min = scales[j+4] & 63
    For j >= 4: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
                min   = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    """
    if j < 4:
        sc = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, m


def dequantize_q4k_block(data, offset):
    """Dequantize one Q4_K block (144 bytes, 256 elements).
    Block layout: d(2 bytes FP16) + dmin(2 bytes FP16) + scales[12] + qs[128]
    Each block encodes 256 elements (QK_K=256).
    """
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
        # Get scale and min for first sub-block (32 elements)
        sc0, m0 = get_scale_min_k4(is_idx + 0, scales)
        d1 = d * float(sc0)
        m1 = dmin * float(m0)

        # Get scale and min for second sub-block (32 elements)
        sc1, m1_val = get_scale_min_k4(is_idx + 1, scales)
        d2 = d * float(sc1)
        m2 = dmin * float(m1_val)

        # First 32 elements: low nibble
        for l in range(32):
            out[j + l] = d1 * float(qs[q_ptr + l] & 0xF) - m1
        # Next 32 elements: high nibble
        for l in range(32):
            out[j + 32 + l] = d2 * float(qs[q_ptr + l] >> 4) - m2

        q_ptr += 32
        is_idx += 2

    return out


def read_gguf_tensors(f):
    """Parse GGUF header, metadata, and tensor info. Return tensor dict and data_offset."""
    magic = struct.unpack('<I', f.read(4))[0]
    assert magic == 0x46554747, f"Not GGUF: 0x{magic:08x}"
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_metadata = struct.unpack('<Q', f.read(8))[0]

    print(f"GGUF version: {version}, n_tensors: {n_tensors}, n_metadata: {n_metadata}")

    # Read metadata
    for _ in range(n_metadata):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        vtype = struct.unpack('<I', f.read(4))[0]
        if vtype == 0:  # UINT8
            f.read(1)
        elif vtype == 1:  # INT8
            f.read(1)
        elif vtype == 2:  # UINT16
            f.read(2)
        elif vtype == 3:  # INT16
            f.read(2)
        elif vtype == 4:  # UINT32
            f.read(4)
        elif vtype == 5:  # INT32
            f.read(4)
        elif vtype == 6:  # FLOAT32
            f.read(4)
        elif vtype == 7:  # BOOL
            f.read(1)
        elif vtype == 8:  # STRING
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif vtype == 9:  # ARRAY
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 8:  # ARRAY of STRING - each element has its own length
                for _ in range(arr_len):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
            else:
                type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                f.read(arr_len * type_sizes.get(arr_type, 4))
        elif vtype == 10:  # UINT64
            f.read(8)
        elif vtype == 11:  # INT64
            f.read(8)
        elif vtype == 12:  # FLOAT64
            f.read(8)
        else:
            raise ValueError(f"Unknown metadata type: {vtype}")

    # Read tensor info
    tensors = {}
    alignment = 32

    for _ in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = {'dims': dims, 'type': ttype, 'offset': offset}

    # Data starts at aligned position after tensor info
    data_offset = f.tell()
    data_offset = (data_offset + alignment - 1) // alignment * alignment

    return tensors, data_offset


def dequantize_embedding(all_data, data_offset, emb_tensor, token_id):
    """Dequantize a single token's embedding from Q4_K data."""
    dims = emb_tensor['dims']
    ttype = emb_tensor['type']

    # GGUF dims: [emb_dim, vocab_size] for token_embd.weight
    emb_dim = int(dims[0])
    vocab_size = int(dims[1])

    print(f"  emb_dim={emb_dim}, vocab_size={vocab_size}, type={ttype}")

    if ttype != 12:
        print(f"  ERROR: Expected Q4_K (type 12), got type {ttype}")
        sys.exit(1)

    # Q4_K: 256 elements per block, 144 bytes per block
    block_size = 144
    elements_per_block = 256
    n_blocks = emb_dim // elements_per_block
    row_size = n_blocks * block_size

    print(f"  n_blocks per token: {n_blocks}, row_size: {row_size} bytes")

    # Calculate offset for the requested token
    row_offset = data_offset + int(emb_tensor['offset']) + token_id * row_size

    # Dequantize
    emb_values = np.zeros(emb_dim, dtype=np.float32)
    for b in range(n_blocks):
        block_offset = row_offset + b * block_size
        block_vals = dequantize_q4k_block(all_data, block_offset)
        emb_values[b * elements_per_block:(b + 1) * elements_per_block] = block_vals

    return emb_values


def compare_with_llama_cpp():
    """Use llama-cpp-python to generate reference logits and compare with our C code."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\nWARNING: llama-cpp-python not installed, skipping llama.cpp comparison")
        print("  Install with: pip install llama-cpp-python")
        return None

    print("\n" + "=" * 60)
    print("Generating reference logits with llama-cpp-python")
    print("=" * 60)

    try:
        llm = Llama(model_path=GGUF_FILE, logits_all=True, n_ctx=64, verbose=False)
    except Exception as e:
        print(f"ERROR loading model with llama-cpp-python: {e}")
        return None

    # Tokenize "The" with BOS prepended
    tokens = llm.tokenize(b"The", add_bos=True)
    print(f"Tokens for 'The' (with BOS): {tokens}")

    # Evaluate tokens
    llm.eval(tokens)

    # Get logits for the last token position
    n_tokens = len(tokens)
    ref_logits = llm.scores[n_tokens - 1, :].copy()

    print(f"Reference logits: shape={ref_logits.shape}, dtype={ref_logits.dtype}")
    print(f"  min={ref_logits.min():.4f}, max={ref_logits.max():.4f}")

    # Print top-10 tokens
    print("\nTop-10 tokens by logit value (llama.cpp):")
    top_indices = np.argsort(ref_logits)[::-1][:10]
    for idx in top_indices:
        token_bytes = llm.detokenize([int(idx)])
        try:
            token_text = token_bytes.decode("utf-8", errors="replace")
        except Exception:
            token_text = repr(token_bytes)
        print(f"  Token {idx}: logit={ref_logits[idx]:.4f} text={token_text!r}")

    # Save reference logits
    np.save(REF_LOGITS_PATH, ref_logits)
    print(f"\nSaved reference logits to {REF_LOGITS_PATH}")

    # Compare with our C code's logits
    if os.path.exists(OUR_LOGITS_PATH):
        our_logits = np.fromfile(OUR_LOGITS_PATH, dtype=np.float32)
        vocab_size = len(ref_logits)

        if len(our_logits) >= vocab_size:
            our_logits = our_logits[:vocab_size]
        else:
            print(f"WARNING: our_logits size ({len(our_logits)}) < vocab_size ({vocab_size})")

        print(f"\nOur C code logits: shape={our_logits.shape}")
        print(f"  min={our_logits.min():.4f}, max={our_logits.max():.4f}")

        # Print top-10 from our logits
        print("\nTop-10 tokens by logit value (our C code):")
        our_top_indices = np.argsort(our_logits)[::-1][:10]
        for idx in our_top_indices:
            token_bytes = llm.detokenize([int(idx)])
            try:
                token_text = token_bytes.decode("utf-8", errors="replace")
            except Exception:
                token_text = repr(token_bytes)
            print(f"  Token {idx}: logit={our_logits[idx]:.4f} text={token_text!r}")

        # Compute difference
        if len(our_logits) == len(ref_logits):
            diff = our_logits - ref_logits
            print(f"\nLogit comparison:")
            print(f"  MAE={np.abs(diff).mean():.4f}")
            print(f"  RMSE={np.sqrt((diff**2).mean()):.4f}")
            print(f"  Max abs diff={np.abs(diff).max():.4f}")
            corr = np.corrcoef(our_logits, ref_logits)[0, 1]
            print(f"  Correlation={corr:.6f}")

            # Top-10 overlap
            ref_top10 = set(np.argsort(ref_logits)[::-1][:10])
            our_top10 = set(np.argsort(our_logits)[::-1][:10])
            overlap = ref_top10 & our_top10
            print(f"  Top-10 overlap: {len(overlap)}/10")
        else:
            print(f"\nSize mismatch: ref={len(ref_logits)}, ours={len(our_logits)}")
    else:
        print(f"\nOur logits file not found at {OUR_LOGITS_PATH}")

    return ref_logits


def main():
    print("=" * 60)
    print("Q4_K Embedding Dequantization Verification")
    print("=" * 60)

    # Step 1: Read GGUF file and parse tensors
    print("\n--- Step 1: Read GGUF file ---")
    with open(GGUF_FILE, 'rb') as f:
        tensors, data_offset = read_gguf_tensors(f)
        f.seek(0)
        all_data = f.read()

    print(f"Data offset: {data_offset}")
    print(f"Number of tensors: {len(tensors)}")

    # Find token_embd.weight
    emb_tensor = tensors.get('token_embd.weight')
    if emb_tensor is None:
        print("ERROR: token_embd.weight not found")
        sys.exit(1)

    print(f"\ntoken_embd.weight: dims={emb_tensor['dims']}, type={emb_tensor['type']}, offset={emb_tensor['offset']}")

    # Step 2: Dequantize embedding for token 1 (BOS)
    print("\n--- Step 2: Dequantize BOS token embedding ---")
    token_id = 1
    emb_values = dequantize_embedding(all_data, data_offset, emb_tensor, token_id)

    print(f"\nEmbedding for token {token_id} (BOS):")
    print(f"  First 10 values: {emb_values[:10]}")
    print(f"  L2 norm: {np.linalg.norm(emb_values):.6f}")
    print(f"  Mean: {emb_values.mean():.6f}")
    print(f"  Std: {emb_values.std():.6f}")

    # Step 3: Compare with C code's debug output
    print("\n--- Step 3: Compare with C code debug output ---")
    c_norm = 2.792910
    c_hidden_first5 = [-0.0121, -0.0121, 0.0309, -0.0982, -0.0265]

    py_norm = np.linalg.norm(emb_values)
    py_first5 = emb_values[:5].tolist()

    print(f"  C code norm:   {c_norm:.6f}")
    print(f"  Python norm:   {py_norm:.6f}")
    print(f"  Norm diff:     {abs(py_norm - c_norm):.6f}")
    print(f"  Norm match:    {'YES' if abs(py_norm - c_norm) < 0.01 else 'NO'}")

    print(f"\n  C code first 5:   {c_hidden_first5}")
    print(f"  Python first 5:   {[round(v, 4) for v in py_first5]}")

    # Element-wise comparison for first 5
    print("\n  Element-wise comparison:")
    for i in range(5):
        diff = abs(py_first5[i] - c_hidden_first5[i])
        match = "OK" if diff < 0.005 else "MISMATCH"
        print(f"    [{i}] C={c_hidden_first5[i]:.4f}, Py={py_first5[i]:.4f}, diff={diff:.4f} {match}")

    # Step 4: Block-level detail
    print("\n--- Step 4: Block-level detail for token 1 ---")
    block_size = 144
    elements_per_block = 256
    emb_dim = int(emb_tensor['dims'][0])
    n_blocks = emb_dim // elements_per_block
    row_size = n_blocks * block_size
    row_offset = data_offset + int(emb_tensor['offset']) + token_id * row_size

    for b in range(n_blocks):
        block_offset = row_offset + b * block_size
        d_raw = struct.unpack_from('<H', all_data, block_offset)[0]
        dmin_raw = struct.unpack_from('<H', all_data, block_offset + 2)[0]
        d = fp16_to_fp32(d_raw)
        dmin = fp16_to_fp32(dmin_raw)
        scales = list(all_data[block_offset + 4: block_offset + 16])

        block_vals = emb_values[b * elements_per_block:(b + 1) * elements_per_block]
        block_norm = np.linalg.norm(block_vals)

        # Print scale/min pairs
        scale_min_pairs = []
        for j in range(8):
            sc, m = get_scale_min_k4(j, scales)
            scale_min_pairs.append(f"({sc},{m})")

        print(f"  Block {b}: d={d:.6f}, dmin={dmin:.6f}, norm={block_norm:.6f}, "
              f"scales={scale_min_pairs}, first4=[{', '.join(f'{v:.4f}' for v in block_vals[:4])}]")

    # Step 5: Also try using the gguf Python package for cross-validation
    print("\n--- Step 5: Cross-validate with gguf Python package ---")
    try:
        import gguf
        reader = gguf.GGUFReader(GGUF_FILE)

        emb_tensor_gguf = None
        for tensor in reader.tensors:
            if tensor.name == 'token_embd.weight':
                emb_tensor_gguf = tensor
                break

        if emb_tensor_gguf is not None:
            print(f"  gguf package: shape={emb_tensor_gguf.shape}, type={emb_tensor_gguf.tensor_type}")

            # The gguf package's .data property returns a numpy array view of the raw bytes
            # For Q4_K, we need to access the raw bytes directly
            raw_bytes = bytes(emb_tensor_gguf.data)
            print(f"  Raw bytes length: {len(raw_bytes)}")

            # The gguf package may return data in a different layout
            # Let's check if the first block matches our manual parsing
            if len(raw_bytes) >= 144:
                d_raw_gguf = struct.unpack_from('<H', raw_bytes, 0)[0]
                dmin_raw_gguf = struct.unpack_from('<H', raw_bytes, 2)[0]
                d_gguf = fp16_to_fp32(d_raw_gguf)
                dmin_gguf = fp16_to_fp32(dmin_raw_gguf)
                print(f"  gguf first block: d={d_gguf:.6f}, dmin={dmin_gguf:.6f}")

                # Compare with manual parsing
                manual_d_raw = struct.unpack_from('<H', all_data, row_offset)[0]
                manual_dmin_raw = struct.unpack_from('<H', all_data, row_offset + 2)[0]
                manual_d = fp16_to_fp32(manual_d_raw)
                manual_dmin = fp16_to_fp32(manual_dmin_raw)
                print(f"  manual first block: d={manual_d:.6f}, dmin={manual_dmin:.6f}")
                print(f"  Match: {'YES' if abs(d_gguf - manual_d) < 1e-6 else 'NO'}")
        else:
            print("  token_embd.weight not found via gguf package")
    except ImportError:
        print("  gguf package not installed, skipping cross-validation")
    except Exception as e:
        print(f"  gguf package error: {e}")

    # Step 6: Compare with llama-cpp-python
    print("\n--- Step 6: llama-cpp-python reference comparison ---")
    compare_with_llama_cpp()

    print("\n" + "=" * 60)
    print("Verification complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
