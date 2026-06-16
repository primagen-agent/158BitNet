"""Verify embedding dequantization by reading GGUF file directly."""

import numpy as np
import struct

GGUF_FILE = "/Users/cuick/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf"

def fp16_to_fp32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            return 0.0 if sign == 0 else -0.0
        val = mant / 1024.0 * 2.0**(-14)
        return -val if sign else val
    if exp == 31:
        return float('inf') if sign == 0 else float('-inf')
    val = (1.0 + mant / 1024.0) * 2.0**(exp - 15)
    return -val if sign else val

def read_gguf_tensors(f):
    """Parse GGUF header, metadata, and tensor info. Return tensor dict and data_offset."""
    magic = struct.unpack('<I', f.read(4))[0]
    assert magic == 0x46554747, f"Not GGUF: 0x{magic:08x}"
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_metadata = struct.unpack('<Q', f.read(8))[0]

    # Read metadata
    for _ in range(n_metadata):
        # key string
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        # value type
        vtype = struct.unpack('<I', f.read(4))[0]
        # read value based on type
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
            type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1}
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
    alignment = 32  # default GGUF alignment

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

def dequantize_q4k_block(data, offset):
    """Dequantize one Q4_K block (144 bytes, 256 elements)."""
    # block_q4_K: d(2) + dmin(2) + scales[12] + qs[128] = 144 bytes
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
        # Get scale and min for first sub-block
        if is_idx < 4:
            sc = scales[is_idx] & 63
            m = scales[is_idx + 4] & 63
        else:
            sc = (scales[is_idx + 4] & 0xF) | ((scales[is_idx - 4] >> 6) << 4)
            m = (scales[is_idx + 4] >> 4) | ((scales[is_idx] >> 6) << 4)

        d1 = d * sc
        m1 = dmin * m

        # Get scale and min for second sub-block
        is1 = is_idx + 1
        if is1 < 4:
            sc1 = scales[is1] & 63
            m1_ = scales[is1 + 4] & 63
        else:
            sc1 = (scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4)
            m1_ = (scales[is1 + 4] >> 4) | ((scales[is1] >> 6) << 4)

        d2 = d * sc1
        m2 = dmin * m1_

        for l in range(32):
            out[j + l] = d1 * (qs[q_ptr + l] & 0xF) - m1
        for l in range(32):
            out[j + 32 + l] = d2 * (qs[q_ptr + l] >> 4) - m2

        q_ptr += 32
        is_idx += 2

    return out

def dequantize_q6k_block(data, offset):
    """Dequantize one Q6_K block (210 bytes, 256 elements)."""
    # block_q6_K: ql[128] + qh[64] + scales[16] + d(2) = 210 bytes
    ql = list(data[offset: offset + 128])
    qh = list(data[offset + 128: offset + 192])
    sc = [int(x) if x < 128 else x - 256 for x in data[offset + 192: offset + 208]]  # int8_t
    d_raw = struct.unpack_from('<H', data, offset + 208)[0]
    d = fp16_to_fp32(d_raw)

    out = np.zeros(256, dtype=np.float32)
    ql_ptr = 0
    qh_ptr = 0
    sc_ptr = 0

    for n in range(0, 256, 128):
        for l in range(32):
            is_idx = l // 16
            q1 = ((ql[ql_ptr + l] & 0xF) | (((qh[qh_ptr + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[ql_ptr + l + 32] & 0xF) | (((qh[qh_ptr + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[ql_ptr + l] >> 4) | (((qh[qh_ptr + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[ql_ptr + l + 32] >> 4) | (((qh[qh_ptr + l] >> 6) & 3) << 4)) - 32
            out[n + l] = d * sc[sc_ptr + is_idx + 0] * q1
            out[n + l + 32] = d * sc[sc_ptr + is_idx + 2] * q2
            out[n + l + 64] = d * sc[sc_ptr + is_idx + 4] * q3
            out[n + l + 96] = d * sc[sc_ptr + is_idx + 6] * q4
        ql_ptr += 64
        qh_ptr += 32
        sc_ptr += 8

    return out

def dequantize_tq2_0_block_interleaved(data, offset):
    """Dequantize one TQ2_0 block (66 bytes, 256 elements) using interleaved layout."""
    # block_tq2_0: qs[64] + d(2) = 66 bytes
    qs = list(data[offset: offset + 64])
    d_raw = struct.unpack_from('<H', data, offset + 64)[0]
    d = fp16_to_fp32(d_raw)

    out = np.zeros(256, dtype=np.float32)
    for j in range(0, 64, 32):
        for l in range(4):
            for m in range(32):
                q = (qs[j + m] >> (l * 2)) & 3
                idx = j * 4 + l * 32 + m
                out[idx] = (q - 1) * d
    return out

def dequantize_tq2_0_block_sequential(data, offset):
    """Dequantize one TQ2_0 block (66 bytes, 256 elements) using sequential layout."""
    qs = list(data[offset: offset + 64])
    d_raw = struct.unpack_from('<H', data, offset + 64)[0]
    d = fp16_to_fp32(d_raw)

    out = np.zeros(256, dtype=np.float32)
    for i in range(64):
        byte = qs[i]
        base = i * 4
        out[base + 0] = (((byte >> 6) & 3) - 1) * d
        out[base + 1] = (((byte >> 4) & 3) - 1) * d
        out[base + 2] = (((byte >> 2) & 3) - 1) * d
        out[base + 3] = (((byte >> 0) & 3) - 1) * d
    return out

# Main
with open(GGUF_FILE, 'rb') as f:
    tensors, data_offset = read_gguf_tensors(f)
    f.seek(0)
    all_data = f.read()

print(f"Data offset: {data_offset}")
print(f"Number of tensors: {len(tensors)}")

# Find token embedding tensor
emb_tensor = tensors.get('token_embd.weight')
if emb_tensor is None:
    print("ERROR: token_embd.weight not found")
    sys.exit(1)

print(f"\ntoken_embd.weight: dims={emb_tensor['dims']}, type={emb_tensor['type']}, offset={emb_tensor['offset']}")

# Dequantize embedding for token 1 (BOS)
emb_dims = emb_tensor['dims']  # [vocab_size, embedding_length]
vocab_size = emb_dims[0]
emb_dim = emb_dims[1]
print(f"  vocab_size={vocab_size}, emb_dim={emb_dim}")

token_id = 1  # BOS
emb_type = emb_tensor['type']

if emb_type == 12:  # Q4_K
    block_size = 144
    elements_per_block = 256
    n_blocks = emb_dim // elements_per_block
    row_size = n_blocks * block_size
    row_offset = data_offset + emb_tensor['offset'] + token_id * row_size

    emb_values = np.zeros(emb_dim, dtype=np.float32)
    for b in range(n_blocks):
        block_offset = row_offset + b * block_size
        block_vals = dequantize_q4k_block(all_data, block_offset)
        emb_values[b * elements_per_block:(b + 1) * elements_per_block] = block_vals

    print(f"\nEmbedding for token {token_id} (BOS):")
    print(f"  First 10 values: {emb_values[:10]}")
    print(f"  Norm: {np.linalg.norm(emb_values):.6f}")
    print(f"  Mean: {emb_values.mean():.6f}")

    # Save for comparison
    np.save("/Users/cuick/workdir/AI/158BitNet/tests/ref_emb_bos.npy", emb_values)
    print("  Saved to tests/ref_emb_bos.npy")
else:
    print(f"  Unsupported embedding type: {emb_type}")

# Also check a TQ2_0 tensor (e.g., blk.0.attn_q.weight)
for name, info in tensors.items():
    if 'blk.0.attn_q.weight' in name:
        print(f"\n{name}: dims={info['dims']}, type={info['type']}, offset={info['offset']}")
        if info['type'] == 35:  # TQ2_0
            block_size = 66
            elements_per_block = 256
            n_blocks_per_row = info['dims'][0] // elements_per_block
            row_size = n_blocks_per_row * block_size

            # Dequantize first row using both methods
            row_offset = data_offset + info['offset']

            interleaved = np.zeros(info['dims'][0], dtype=np.float32)
            sequential = np.zeros(info['dims'][0], dtype=np.float32)

            for b in range(n_blocks_per_row):
                block_offset = row_offset + b * block_size
                interleaved[b*256:(b+1)*256] = dequantize_tq2_0_block_interleaved(all_data, block_offset)
                sequential[b*256:(b+1)*256] = dequantize_tq2_0_block_sequential(all_data, block_offset)

            print(f"  Interleaved first 10: {interleaved[:10]}")
            print(f"  Sequential first 10: {sequential[:10]}")
            print(f"  Interleaved norm: {np.linalg.norm(interleaved):.6f}")
            print(f"  Sequential norm: {np.linalg.norm(sequential):.6f}")

            # Check how many non-zero values
            print(f"  Interleaved non-zero: {np.count_nonzero(interleaved)}/{len(interleaved)}")
            print(f"  Sequential non-zero: {np.count_nonzero(sequential)}/{len(sequential)}")
        break
