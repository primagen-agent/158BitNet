#!/usr/bin/env python3
"""Multi-projection BNLORA1 serialization round-trip."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from backbone_lora import (  # noqa: E402
    BackboneLoRA,
    load_lora_bundle,
    save_lora_bundle,
)
from output_lora import OutputLoRA  # noqa: E402


def main():
    cfg = SimpleNamespace(
        n_layers=3, hidden=12, q_dim=16, kv_dim=4, ffn=24)
    bundle = BackboneLoRA(
        cfg, blocks=[1, 2], targets=("q", "v", "o"),
        rank=4, alpha=8.0, device="cpu")
    with torch.no_grad():
        for index, tensor in enumerate(bundle.tensors()):
            tensor.a.copy_(
                torch.arange(tensor.a.numel()).reshape_as(tensor.a) / 100
                + index)
            tensor.b.copy_(
                torch.arange(tensor.b.numel()).reshape_as(tensor.b) / 200
                - index)

    samples = {}
    for tensor in bundle.tensors():
        key = bundle.key(tensor.block_index, tensor.layer_id)
        hidden = torch.arange(tensor.in_dim * 2).reshape(
            2, tensor.in_dim) / 10
        samples[key] = (hidden, tensor(hidden))
    output_module = OutputLoRA(
        cfg.hidden, 19, rank=4, alpha=8.0, device="cpu")
    with torch.no_grad():
        output_module.a.copy_(
            torch.arange(output_module.a.numel()).reshape_as(
                output_module.a) / 100)
        output_module.b.copy_(
            torch.arange(output_module.b.numel()).reshape_as(
                output_module.b) / 200)
    output_hidden = torch.arange(cfg.hidden * 2).reshape(
        2, cfg.hidden) / 10
    expected_output = output_module(output_hidden)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = str(Path(temp_dir) / "backbone.bnlora")
        save_lora_bundle(path, bundle.tensors() + [output_module])
        loaded, output = load_lora_bundle(path, cfg, device="cpu")

    assert output is not None
    assert len(loaded.tensors()) == len(bundle.tensors())
    for key, (hidden, expected) in samples.items():
        actual = loaded.projections[key](hidden)
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(output(output_hidden), expected_output)
    print("backbone LoRA roundtrip: PASS")


if __name__ == "__main__":
    main()
