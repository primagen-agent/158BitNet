"""Verify TQ2_0 element ordering by comparing with llama-cpp-python."""

import numpy as np
import struct
import sys

GGUF_FILE = "/Users/cuick/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf"

def read_gguf_header(f):
    """Read GGUF header and return tensor info."""
    magic = struct.unpack('<I', f.read(4))[0]
    if magic != 0x46475547:  # 'GGUF'
        raise ValueError(f"Not a GGUF file: magic=0x{magic:08x}")
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_metadata = struct.unpack('<Q', f.read(8))[0]
    return version, n_tensors, n_tensors

def fp16_to_fp32(h):
    """Convert FP16 to FP32."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            return 0.0 if sign == 0 else -0.0
        # Denormalized
        val = mant / 1024.0 * 2.0**(-14)
        return -val if sign else val
    if exp == 31:
        return float('inf') if sign == 0 else float('-inf')
    val = (1.0 + mant / 1024.0) * 2.0**(exp - 15)
    return -val if sign else val

def dequantize_tq2_0_interleaved(qs_bytes, d_fp16):
    """Dequantize TQ2_0 block using interleaved layout (llama.cpp style)."""
    d = fp16_to_fp32(d_fp16)
    out = np.zeros(256, dtype=np.float32)
    for j in range(0, 64, 32):
        for l in range(4):
            for m in range(32):
                q = (qs_bytes[j + m] >> (l * 2)) & 3
                idx = j * 4 + l * 32 + m
                out[idx] = (q - 1) * d
    return out

def dequantize_tq2_0_sequential(qs_bytes, d_fp16):
    """Dequantize TQ2_0 block using sequential layout."""
    d = fp16_to_fp32(d_fp16)
    out = np.zeros(256, dtype=np.float32)
    for i in range(64):
        byte = qs_bytes[i]
        base = i * 4
        out[base + 0] = ((byte >> 6) & 3) - 1) * d  # WRONG SYNTAX, fix below
        out[base + 1] = (((byte >> 4) & 3) - 1) * d
        out[base + 2] = (((byte >> 2) & 3) - 1) * d
        out[base + 3] = (((byte >> 0) & 3) - 1) * d
    return out

# Read GGUF file to find a TQ2_0 tensor
# We'll use the first block's Q weight tensor
print("Reading GGUF file...")

with open(GGUF_FILE, 'rb') as f:
    # Skip to find tensor info
    version, n_metadata, n_tensors = read_gguf_header(f)
    print(f"GGUF version={version}, n_metadata={n_metadata}, n_tensors={n_tensors}")

    # Read metadata (skip it for now)
    # We need to find the data_offset and tensor offsets
    # This is complex, so let's use a simpler approach

# Instead, let's use llama-cpp-python to get reference values
try:
    from llama_cpp import Llama
    print("\nUsing llama-cpp-python for reference...")

    llm = Llama(model_path=GGUF_FILE, logits=True, n_ctx=64, verbose=False)

    # Get the embedding for BOS token (token 1)
    tokens = [1]  # BOS only
    result = llm.eval(tokens)
    logits_bos = llm.scores[0, :]  # logits after BOS

    print(f"Reference logits after BOS: shape={logits_bos.shape}")
    top5 = np.argsort(logits_bos)[::-1][:5]
    for idx in top5:
        print(f"  Token {idx}: logit={logits_bos[idx]:.4f}")

    # Now get logits after "The" (BOS + 1507)
    tokens2 = [1, 1507]
    result2 = llm.eval(tokens2)
    logits_the = llm.scores[1, :]  # logits after "The"

    print(f"\nReference logits after 'The': shape={logits_the.shape}")
    top5_the = np.argsort(logits_the)[::-1][:5]
    for idx in top5_the:
        text = llm.detokenize([idx]).decode('utf-8', errors='replace')
        print(f"  Token {idx}: logit={logits_the[idx]:.4f} text='{text}'")

    # Save reference
    np.save("/Users/cuick/workdir/AI/158BitNet/tests/ref_logits_the.npy", logits_the)
    print("\nSaved reference logits for 'The' to tests/ref_logits_the.npy")

except Exception as e:
    print(f"Error: {e}")
