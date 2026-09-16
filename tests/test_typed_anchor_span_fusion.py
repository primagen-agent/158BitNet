#!/usr/bin/env python3
"""Tests for adapted-anchor span fusion."""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tune_typed_anchor_span_fusion import predict  # noqa: E402


class TypedAnchorSpanFusionTests(unittest.TestCase):
    def test_hard_anchor_rejects_higher_unanchored_span(self):
        batch = {
            "base": torch.tensor([[[9.0], [2.0], [1.0]]]),
            "contains": torch.tensor([[
                [False], [True], [False],
            ]]),
        }
        self.assertEqual(int(predict(batch, 0.0)[0]), 0)
        self.assertEqual(int(predict(batch, "hard")[0]), 1)

    def test_soft_anchor_bonus_changes_ranking(self):
        batch = {
            "base": torch.tensor([[[3.0], [2.0]]]),
            "contains": torch.tensor([[[False], [True]]]),
        }
        self.assertEqual(int(predict(batch, 0.5)[0]), 0)
        self.assertEqual(int(predict(batch, 2.0)[0]), 1)


if __name__ == "__main__":
    unittest.main()
