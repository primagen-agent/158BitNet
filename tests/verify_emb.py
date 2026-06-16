import struct
import numpy as np

# Correct: data_offset + tensor_offset
DATA_OFFSET = 1692224
TOKEN_EMBD_OFFSET = 123400832
FILE_OFFSET = DATA_OFFSET + TOKEN_EMBD_OFFSET  # 125093056

with open('models/bitcpm4-1b-tq2_0.gguf', 'rb') as f:
    # Read BOS token (id=1) embedding
    f.seek(FILE_OFFSET + 1152)  # skip token 0
    data = f.read(1152)

block = data[:144]
d_raw = struct.unpack('<H', block[0:2])[0]
dmin_raw = struct.unpack('<H', block[2:4])[0]

def fp16_to_fp32(h):
    return float(np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0])

d = fp16_to_fp32(d_raw)
dmin = fp16_to_fp32(dmin_raw)
print(f'Block 0: d={d:.6f}, dmin={dmin:.6f}')

# Dequantize all 8 blocks
def get_scale_min_k4(j, q):
    if j < 4:
        sc = q[j] & 63
        m = q[j+4] & 63
    else:
        sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
        m = (q[j+0] >> 4) | ((q[j+4] >> 6) << 4)
    return sc, m

all_values = []
for block_idx in range(8):
    block = data[block_idx*144:(block_idx+1)*144]
    d = fp16_to_fp32(struct.unpack('<H', block[0:2])[0])
    dmin = fp16_to_fp32(struct.unpack('<H', block[2:4])[0])
    scales = list(block[4:16])
    qs = block[16:144]
    
    is_idx = 0
    qs_offset = 0
    for j_group in range(4):
        sc1, m1 = get_scale_min_k4(is_idx + 0, scales)
        sc2, m2 = get_scale_min_k4(is_idx + 1, scales)
        d1 = d * sc1
        min1 = dmin * m1
        d2 = d * sc2
        min2 = dmin * m2
        
        for l in range(32):
            all_values.append(d1 * (qs[qs_offset + l] & 0xF) - min1)
        for l in range(32):
            all_values.append(d2 * (qs[qs_offset + l] >> 4) - min2)
        
        qs_offset += 32
        is_idx += 2

arr = np.array(all_values)
print(f'\nBOS embedding (correct offset):')
print(f'  First 10: {arr[:10]}')
print(f'  Max: {arr.max():.6f}, Min: {arr.min():.6f}')
print(f'  Mean abs: {np.abs(arr).mean():.6f}, Std: {arr.std():.6f}')

# Compare with llama.cpp reference
from llama_cpp import Llama
llm = Llama(model_path='models/bitcpm4-1b-tq2_0.gguf', n_ctx=256, n_gpu_layers=0, verbose=False, embedding=True)
result = llm.create_embedding('')
ref = np.array(result['data'][0]['embedding'])
print(f'\nllama.cpp reference:')
print(f'  First 10: {ref[:10]}')
print(f'  Max: {ref.max():.6f}, Min: {ref.min():.6f}')
print(f'  Mean abs: {np.abs(ref).mean():.6f}, Std: {ref.std():.6f}')
