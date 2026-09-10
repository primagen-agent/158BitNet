#!/usr/bin/env python3
"""Unit tests for exact token-span localization."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_memory_v89_copy_pointer import (  # noqa: E402
    BytePointer,
    locate_byte_span,
    reconstruction_paths,
    select_best_byte_span,
)
import torch  # noqa: E402


class FakeDecoder:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode_bytes(self, token_ids):
        return b"".join(self.pieces[token] for token in token_ids)


def test_locates_and_recovers_payload_inside_merged_boundaries():
    decoder = FakeDecoder({
        1: b"prefix:blue", 2: b" sky!", 3: b" suffix",
    })
    span = locate_byte_span(
        [1, 2, 3], "blue sky", decoder, max_span=3)
    assert span is not None
    assert (
        span["start"], span["end"],
        span["start_offset"], span["end_offset"]
    ) == (0, 1, 7, 4)
    assert span["record_bytes"][
        span["start_byte"]:span["end_byte"] + 1
    ] == b"blue sky"


def test_pointer_emits_dense_inside_scores():
    model = BytePointer(hidden=8, rank=4, max_token_bytes=8)
    hidden = torch.zeros(2, 3, 8)
    values = torch.zeros(2, 3, dtype=torch.long)
    offsets = torch.zeros(2, 3, dtype=torch.long)
    mask = torch.tensor([
        [True, True, True],
        [True, True, False],
    ])
    start, end, inside = model(
        hidden, values, values, values, offsets, mask)
    assert start.shape == end.shape == inside.shape == (2, 3)
    assert inside[1, 2].item() < -1e8


def test_reconstruction_paths_support_unified_length_roots():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for length in (8, 1, 4):
            target = root / f"len{length}"
            target.mkdir()
            (target / "reconstruction.jsonl").write_text("{}\n")
        assert [
            path.parent.name for path in reconstruction_paths(root)
        ] == ["len1", "len4", "len8"]


def test_dense_span_selection_penalizes_long_outside_regions():
    start = torch.tensor([3.0, 2.0, 0.0, 0.0])
    end = torch.tensor([0.0, 0.0, 3.0, 2.0])
    inside = torch.tensor([-4.0, 3.0, 3.0, -4.0])
    assert select_best_byte_span(
        start, end, inside, count=4, max_copy_bytes=4
    ) == (1, 2)


def test_normalized_inside_scores_reduce_length_bias():
    start = torch.tensor([2.0, 0.0, 0.0])
    end = torch.tensor([2.0, 0.0, 2.1])
    inside = torch.tensor([1.0, 0.4, 0.4])
    assert select_best_byte_span(
        start, end, inside, count=3, max_copy_bytes=3,
        inside_mode="sum",
    ) == (0, 2)
    assert select_best_byte_span(
        start, end, inside, count=3, max_copy_bytes=3,
        inside_mode="mean",
    ) == (0, 0)


if __name__ == "__main__":
    test_locates_and_recovers_payload_inside_merged_boundaries()
    test_pointer_emits_dense_inside_scores()
    test_reconstruction_paths_support_unified_length_roots()
    test_dense_span_selection_penalizes_long_outside_regions()
    test_normalized_inside_scores_reduce_length_bias()
    print("memory V89 copy-pointer tests: PASS")
