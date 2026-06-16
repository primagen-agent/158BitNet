"""
Compare logits: load GGUF model via llama-cpp-python, get logits for "The" (with BOS),
print top-10 tokens and specific token IDs, save full logit array for comparison.
"""

import sys
import numpy as np

model_path = "/Users/cuick/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf"
output_path = "/Users/cuick/workdir/AI/158BitNet/tests/ref_logits.npy"

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python is not installed. Run: pip install llama-cpp-python")
    sys.exit(1)

# Load model with logits_all=True so eval() stores logits in llm.scores
try:
    llm = Llama(model_path=model_path, logits_all=True, n_ctx=64, verbose=False)
except Exception as e:
    error_msg = str(e)
    if "TQ2_0" in error_msg or "tq2_0" in error_msg or "unknown" in error_msg.lower():
        print(f"ERROR: This build of llama-cpp-python does not support TQ2_0 quantization.")
        print(f"Detail: {e}")
        sys.exit(1)
    else:
        raise

# Tokenize "The" with BOS prepended
tokens = llm.tokenize(b"The", add_bos=True)
print(f"Tokens: {tokens}")

# Evaluate tokens - logits are stored in llm.scores after eval()
llm.eval(tokens)

# Get logits for the last token position
# llm.scores shape: (n_ctx, n_vocab), we want the row at index len(tokens)-1
n_tokens = len(tokens)
logits = llm.scores[n_tokens - 1, :].copy()

print(f"\nLogits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")

# Print top-10 tokens
print("\nTop-10 tokens by logit value:")
top_indices = np.argsort(logits)[::-1][:10]
for idx in top_indices:
    token_bytes = llm.detokenize([int(idx)])
    try:
        token_text = token_bytes.decode("utf-8", errors="replace")
    except Exception:
        token_text = repr(token_bytes)
    print(f"  Token {idx}: logit={logits[idx]:.4f} text={token_text!r}")

# Print specific token IDs
specific_ids = [0, 1, 1507, 8107]
print("\nSpecific token logits:")
for tid in specific_ids:
    if tid < len(logits):
        token_bytes = llm.detokenize([tid])
        try:
            token_text = token_bytes.decode("utf-8", errors="replace")
        except Exception:
            token_text = repr(token_bytes)
        print(f"  Token {tid}: logit={logits[tid]:.4f} text={token_text!r}")
    else:
        print(f"  Token {tid}: out of vocab range (vocab_size={len(logits)})")

# Save full logit array
np.save(output_path, logits)
print(f"\nSaved logits to {output_path}")
print(f"Saved array shape: {logits.shape}, dtype: {logits.dtype}")
