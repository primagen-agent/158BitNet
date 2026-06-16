from llama_cpp import Llama
import numpy as np
import ctypes, struct

# We need to get the dequantized weights from llama.cpp
# llama-cpp-python doesn't expose this directly, but we can
# use the low-level API to compute a simple matmul and compare

# Alternative approach: compute the expected Q projection output
# by feeding a known input through llama.cpp and our system

# Let's try: evaluate just the BOS token and compare the hidden state
# after the first transformer block

llm = Llama(model_path='models/bitcpm4-1b-tq2_0.gguf', n_ctx=256, n_gpu_layers=0, verbose=False, logits_all=True)

# Evaluate BOS
llm.eval([1])

# Get logits
scores = np.array(llm.scores[0])
print('llama.cpp after BOS:')
print(f'  Logit range: [{scores.min():.4f}, {scores.max():.4f}]')
top_idx = np.argsort(scores)[-5:][::-1]
for idx in top_idx:
    tok_str = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f'  token {idx} [{tok_str}]: {scores[idx]:.4f}')

# Now let's check: what if the TQ2_0 encoding is 00=+1, 10=-1 (swapped)?
# This would negate all weights, which would negate the output of matmul
# Let's check if negating our logits gives the right answer

# Our system's top tokens after BOS (from test_emb_debug output):
# Top 1: token 10719 [aters] logit=10.59
# Top 2: token 5648 [是] logit=10.51

# llama.cpp's top tokens after BOS:
# Top 1: token 59320 [ ]: 9.88
# Top 2: token 30368 [Understanding]: 9.25

# The logit values are similar in magnitude but completely different tokens
# This suggests the forward pass is computing something different

# Key insight: if we swap the TQ2_0 encoding (00=+1 instead of -1),
# the matmul output would be negated for all -1 weights
# But since the distribution is roughly 1/3 pos, 1/3 neg, 1/3 zero,
# this would significantly change the output

# Let's check: what encoding does the GGUF spec use?
# From ggml-common.h: "2 bits per element" - no specific mapping given
# The mapping is defined in the dequantization code

# Let's search for the TQ2_0 dequantization in the llamafile headers
print('\nSearching for TQ2_0 dequantization in llama.cpp source...')
print('The encoding mapping is the key question.')
print('Our current mapping: 00=-1, 01=0, 10=+1, 11=0')
print('Alternative:          00=+1, 01=0, 10=-1, 11=0')
