"""
Step-by-step token generation comparison using llama-cpp-python (reference).
Generates tokens for "The capital of France is" with greedy sampling,
printing top-5 logits at each generation step.

Uses internal _ctx API to get raw logits directly from llama.cpp.
"""

from llama_cpp import Llama
import numpy as np

MODEL_PATH = "/Users/cuick/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf"
VOCAB_SIZE = 73448

llm = Llama(model_path=MODEL_PATH, verbose=False, n_ctx=256, logits=True)

prompt = "The capital of France is"
tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
print(f"Prompt tokens ({len(tokens)}): {list(tokens)}")
print(f"Prompt token texts:")
for t in tokens:
    text = llm.detokenize([t]).decode("utf-8", errors="replace")
    print(f"  Token {t}: {repr(text)}")

# Process prompt tokens
llm.eval(tokens)

# Generate 10 tokens step by step
generated_tokens = []
for step in range(10):
    # Use internal API to get raw logits
    logits_ptr = llm._ctx.get_logits()
    logits = np.ctypeslib.as_array(logits_ptr, shape=(VOCAB_SIZE,)).copy()

    # Get top-5 by logit value
    top_indices = np.argsort(logits)[::-1][:5]
    next_token = int(top_indices[0])

    print(f"\n=== Step {step} ===")
    for idx in top_indices:
        text = llm.detokenize([int(idx)]).decode("utf-8", errors="replace")
        print(f"  Token {idx}: logit={logits[idx]:.6f} text={repr(text)}")

    selected_text = llm.detokenize([next_token]).decode("utf-8", errors="replace")
    print(f"  >> Selected: Token {next_token} = {repr(selected_text)}")

    generated_tokens.append(next_token)
    llm.eval([next_token])

print(f"\n\n=== Summary ===")
print(f"Prompt: {repr(prompt)}")
print(f"Generated tokens: {generated_tokens}")
full_text = llm.detokenize(generated_tokens).decode("utf-8", errors="replace")
print(f"Generated text: {repr(full_text)}")
