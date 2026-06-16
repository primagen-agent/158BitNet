"""Compare our logits with llama.cpp reference logits."""

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref_path = ROOT / "tests" / "ref_logits.npy"
our_path = ROOT / "build" / "our_logits.bin"
vocab_size = 73448

# Load reference logits
ref_logits = np.load(ref_path)
print(f"Reference logits: shape={ref_logits.shape}, dtype={ref_logits.dtype}")
print(f"  min={ref_logits.min():.4f}, max={ref_logits.max():.4f}, mean={ref_logits.mean():.4f}")

# Load our logits
our_logits = np.fromfile(our_path, dtype=np.float32, count=vocab_size)
print(f"\nOur logits: shape={our_logits.shape}, dtype={our_logits.dtype}")
print(f"  min={our_logits.min():.4f}, max={our_logits.max():.4f}, mean={our_logits.mean():.4f}")

# Compare
if len(our_logits) == len(ref_logits):
    diff = our_logits - ref_logits
    print(f"\nDifference: min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")
    print(f"  MAE={np.abs(diff).mean():.4f}")
    print(f"  RMSE={np.sqrt((diff**2).mean()):.4f}")

    # Correlation
    corr = np.corrcoef(our_logits, ref_logits)[0, 1]
    print(f"  Correlation={corr:.6f}")

    # Top-10 comparison
    ref_top10 = np.argsort(ref_logits)[::-1][:10]
    our_top10 = np.argsort(our_logits)[::-1][:10]

    print(f"\nReference top-10: {ref_top10}")
    print(f"Our top-10:       {our_top10}")
    print(f"Overlap: {len(set(ref_top10) & set(our_top10))}/10")

    # Specific tokens
    for tid in [0, 1, 1507, 8107]:
        print(f"  Token {tid}: ref={ref_logits[tid]:.4f}, ours={our_logits[tid]:.4f}, diff={our_logits[tid]-ref_logits[tid]:.4f}")
else:
    print(f"\nSize mismatch: ref={len(ref_logits)}, ours={len(our_logits)}")
