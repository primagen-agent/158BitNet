#!/usr/bin/env python3
"""Memory answer decoder serialization and forward round-trip."""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from answer_decoder import MemoryAnswerDecoder  # noqa: E402


def main():
    torch.manual_seed(7)
    module = MemoryAnswerDecoder(12, 5, device="cpu")
    with torch.no_grad():
        module.scale.fill_(0.7)
    hidden = torch.randn(3, 12)
    expected = module(hidden)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "decoder.bnanswer")
        module.save(path)
        loaded = MemoryAnswerDecoder.load(path, 12, device="cpu")
        actual = loaded(hidden)
        memory_module = MemoryAnswerDecoder(
            12, 5, device="cpu", memory_aware=True)
        memory_path = os.path.join(directory, "memory-aware.bnanswer")
        memory_module.save(memory_path)
        memory_loaded = MemoryAnswerDecoder.load(
            memory_path, 12, device="cpu")
        memory = torch.randn(3, 12)
        memory_expected = memory_module(hidden, memory)
        memory_actual = memory_loaded(hidden, memory)
        structured_module = MemoryAnswerDecoder(
            12, 5, device="cpu", structured_memory=True)
        structured_path = os.path.join(
            directory, "structured-memory.bnanswer")
        structured_module.save(structured_path)
        structured_loaded = MemoryAnswerDecoder.load(
            structured_path, 12, device="cpu")
        memory_layers = torch.randn(4, 3, 12)
        structured_expected = structured_module(
            hidden, memory_layers)
        structured_actual = structured_loaded(
            hidden, memory_layers)
    torch.testing.assert_close(actual, expected)
    assert memory_loaded.memory_aware
    torch.testing.assert_close(memory_actual, memory_expected)
    assert structured_loaded.structured_memory
    torch.testing.assert_close(
        structured_actual, structured_expected)
    print("memory answer decoder roundtrip: PASS")


if __name__ == "__main__":
    main()
