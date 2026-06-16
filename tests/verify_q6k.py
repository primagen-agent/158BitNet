import struct
import numpy as np

def fp16_to_fp32(h):
    return float(np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0])

DATA_OFFSET = 1692224
# output.weight: type=Q6_K dims=2048x73448 offset=0
# So output.weight starts at the very beginning of the data section

with open('models/bitcpm4-1b-tq2_0.gguf', 'rb') as f:
    f.seek(DATA_OFFSET + 0)  # output.weight offset = 0
    # Read first Q6_K block (210 bytes)
    data = f.read(210)

# Q6_K block layout: d(2) + ql(128) + qh(64) + scales(16) = 210
d_raw = struct.unpack('<H', data[0:2])[0]
d = fp16_to_fp32(d_raw)
ql = data[2:130]
qh = data[130:194]
scales = data[194:210]

print(f'Block 0: d={d:.6f}')
print(f'scales: {list(scales)}')
print(f'scales as int8: {[b - 256 if b > 127 else b for b in scales]}')

# Dequantize first 128 values
values = []
for l in range(32):
    is_idx = l // 16
    q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
    q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
    q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
    q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
    
    sc0 = scales[is_idx + 0] if scales[is_idx + 0] < 128 else scales[is_idx + 0] - 256
    sc2 = scales[is_idx + 2] if scales[is_idx + 2] < 128 else scales[is_idx + 2] - 256
    sc4 = scales[is_idx + 4] if scales[is_idx + 4] < 128 else scales[is_idx + 4] - 256
    sc6 = scales[is_idx + 6] if scales[is_idx + 6] < 128 else scales[is_idx + 6] - 256
    
    values.append(d * sc0 * q1)
    values.append(d * sc2 * q2)
    values.append(d * sc4 * q3)
    values.append(d * sc6 * q4)

arr = np.array(values[:128])
print(f'\nFirst 128 dequantized values:')
print(f'  First 10: {arr[:10]}')
print(f'  Max: {arr.max():.6f}, Min: {arr.min():.6f}')
print(f'  Mean abs: {np.abs(arr).mean():.6f}')

# Now check: what if the output.weight data is actually at a different offset?
# The GGUF tensor offset is relative to data_offset
# output.weight offset = 0, so it starts at DATA_OFFSET + 0 = 1692224
# Let's verify this is correct by checking the first few bytes
print(f'\nFirst 20 bytes of output.weight data: {[hex(b) for b in data[:20]]}')
