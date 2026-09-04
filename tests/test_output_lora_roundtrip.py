#!/usr/bin/env python3
"""Output LoRA BNLORA1 serialization round-trip."""

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from output_lora import OutputLoRA, load_output_lora, save_output_lora  # noqa: E402


def main():
    module = OutputLoRA(12, 19, rank=4, alpha=8.0, device="cpu")
    with torch.no_grad():
        module.a.copy_(torch.arange(48).reshape(4, 12) / 100)
        module.b.copy_(torch.arange(76).reshape(19, 4) / 200)
    sample = torch.arange(24).reshape(2, 12) / 10
    expected = module(sample)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = str(Path(temp_dir) / "output.bnlora")
        save_output_lora(path, module)
        loaded = load_output_lora(path, 12, 19, device="cpu")
    torch.testing.assert_close(loaded(sample), expected)
    assert loaded.rank == module.rank
    assert loaded.scale == module.scale
    print("output LoRA roundtrip: PASS")


if __name__ == "__main__":
    main()
