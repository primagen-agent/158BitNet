from llama_cpp import Llama
import numpy as np
import ctypes, struct

# Load model
llm = Llama(model_path='models/bitcpm4-1b-tq2_0.gguf', n_ctx=256, n_gpu_layers=0, verbose=False)

# Get the dequantized attn_q.weight for block 0
# We can't directly access dequantized weights from llama-cpp-python
# But we can compare the final output

# Instead, let's verify our TQ2_0 dequantization by comparing
# the first few values of the first row of blk.0.attn_q.weight

# Read the raw TQ2_0 data from the GGUF file
DATA_OFFSET = 1692224
# blk.0.attn_q.weight: type=TQ2_0 dims=2048x2048 offset=209237632
# Each row: 2048/256 = 8 blocks * 66 bytes = 528 bytes

with open('models/bitcpm4-1b-tq2_0.gguf', 'rb') as f:
    f.seek(DATA_OFFSET + 209237632)
    # Read first row (8 blocks = 528 bytes)
    data = f.read(528)

def fp16_to_fp32(h):
    return float(np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0])

# Dequantize first row using our interpretation
# TQ2_0: qs[64] + d(2) = 66 bytes per block
# 2-bit encoding: 00=-1, 01=0, 10=+1, 11=0
values = []
for block_idx in range(8):
    block = data[block_idx*66:(block_idx+1)*66]
    qs = block[:64]
    d_raw = struct.unpack('<H', block[64:66])[0]
    d = fp16_to_fp32(d_raw)
    
    for byte_idx in range(64):
        byte = qs[byte_idx]
        for shift in [6, 4, 2, 0]:
            code = (byte >> shift) & 3
            if code == 0:
                values.append(-d)
            elif code == 2:
                values.append(d)
            else:
                values.append(0.0)

arr = np.array(values[:2048])
print(f'Our TQ2_0 dequantization (first row of blk.0.attn_q.weight):')
print(f'  First 10: {arr[:10]}')
print(f'  Max: {arr.max():.6f}, Min: {arr.min():.6f}')
print(f'  Mean abs: {np.abs(arr).mean():.6f}')
print(f'  Non-zero: {np.count_nonzero(arr)}/{len(arr)}')

# Now let's try different encoding: maybe 00=+1, 10=-1?
# Or: 00=0, 01=+1, 10=-1, 11=0?
# Let's try all 4 possible mappings and see which gives reasonable weight values
print('\nTrying different 2-bit encodings:')
for mapping_name, mapping in [
    ("00=-1,01=0,10=+1,11=0", {0: -1, 1: 0, 2: 1, 3: 0}),
    ("00=+1,01=0,10=-1,11=0", {0: 1, 1: 0, 2: -1, 3: 0}),
    ("00=0,01=+1,10=-1,11=0", {0: 0, 1: 1, 2: -1, 3: 0}),
    ("00=0,01=-1,10=+1,11=0", {0: 0, 1: -1, 2: 1, 3: 0}),
]:
    vals = []
    for block_idx in range(8):
        block = data[block_idx*66:(block_idx+1)*66]
        qs = block[:64]
        d_raw = struct.unpack('<H', block[64:66])[0]
        d = fp16_to_fp32(d_raw)
        
        for byte_idx in range(64):
            byte = qs[byte_idx]
            for shift in [6, 4, 2, 0]:
                code = (byte >> shift) & 3
                vals.append(d * mapping[code])
    
    arr = np.array(vals[:2048])
    print(f'  {mapping_name}: mean_abs={np.abs(arr).mean():.6f}, max={arr.max():.6f}, min={arr.min():.6f}')
