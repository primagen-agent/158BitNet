#!/usr/bin/env python3
import pathlib
import struct
import sys
import tempfile
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_episode_pointer import (  # noqa: E402
    EpisodeSpanPointer,
    best_spans,
    export_episode_pointer_binary,
    find_unique_token_span,
)


class EpisodePointerTest(unittest.TestCase):
    def test_unique_span(self):
        self.assertEqual(
            find_unique_token_span(
                [1, 2, 3, 4, 5], [3, 4]),
            (2, 3))
        self.assertIsNone(
            find_unique_token_span(
                [1, 2, 1, 2], [1, 2]))

    def test_joint_span_respects_order_and_limit(self):
        start = torch.tensor([[0.0, 5.0, 1.0, 0.0]])
        end = torch.tensor([[0.0, 1.0, 2.0, 6.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        left, right = best_spans(
            start, end, mask, max_span=3)
        self.assertEqual(
            (int(left[0]), int(right[0])), (1, 3))

    def test_pointer_shapes(self):
        pointer = EpisodeSpanPointer(
            hidden=16, rank=8, max_span=6,
            use_length_head=True)
        source = torch.randn(2, 7, 16)
        query = torch.randn(2, 5, 16)
        source_mask = torch.ones(2, 7, dtype=torch.bool)
        query_mask = torch.ones(2, 5, dtype=torch.bool)
        start, end, length = pointer(
            source, source_mask, query, query_mask)
        self.assertEqual(tuple(start.shape), (2, 7))
        self.assertEqual(tuple(end.shape), (2, 7))
        self.assertEqual(tuple(length.shape), (2, 6))

    def test_length_logits_can_disambiguate_endpoint(self):
        start = torch.tensor([
            [0.0, 5.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ])
        end = torch.tensor([
            [0.0, 4.0, 4.0, 4.0],
            [0.0, 3.0, 5.0, 0.0],
        ])
        mask = torch.ones(2, 4, dtype=torch.bool)
        length = torch.tensor([
            [0.0, 0.0, 6.0],
            [0.0, 6.0, 0.0],
        ])
        left, right = best_spans(
            start, end, mask, max_span=3,
            length_logits=length)
        self.assertEqual(
            (int(left[0]), int(right[0])), (1, 3))
        self.assertEqual(
            (int(left[1]), int(right[1])), (0, 1))
        self.assertTrue(bool((left < start.shape[1]).all()))
        self.assertTrue(bool((right < start.shape[1]).all()))

    def test_binary_export_header(self):
        pointer = EpisodeSpanPointer(
            hidden=16, rank=8, max_span=6)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "pointer.pt"
            output = pathlib.Path(directory) / "pointer.bneptr"
            torch.save({
                "format": "EPISODE_POINTER_V2",
                "backbone_sha256": "ab" * 32,
                "hidden": 16,
                "rank": 8,
                "max_span": 6,
                "state_dict": pointer.state_dict(),
            }, checkpoint)
            export_episode_pointer_binary(checkpoint, output)
            payload = output.read_bytes()
        self.assertEqual(payload[:8], b"BNEPTR1\x00")
        self.assertEqual(
            struct.unpack("<IIII", payload[8:24]),
            (1, 16, 8, 6))
        self.assertEqual(payload[24:56], bytes.fromhex("ab" * 32))
        self.assertEqual(struct.unpack("<I", payload[56:60]), (9,))


if __name__ == "__main__":
    unittest.main()
