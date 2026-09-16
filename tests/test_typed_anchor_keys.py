#!/usr/bin/env python3
"""Tests for field activation-key adaptation."""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_typed_anchor_keys import (  # noqa: E402
    TypedAnchorKeys,
    anchor_loss,
)
from train_typed_memory_writer import SLOT_NAMES  # noqa: E402


class TypedAnchorKeysTests(unittest.TestCase):
    def test_keys_copy_writer_initialization(self):
        initial = torch.randn(len(SLOT_NAMES), 4)
        keys = TypedAnchorKeys(initial)
        self.assertTrue(torch.equal(
            keys.keys["predicate"],
            initial[SLOT_NAMES.index("predicate")],
        ))
        self.assertTrue(torch.equal(
            keys.keys["value"],
            initial[SLOT_NAMES.index("value")],
        ))

    def test_anchor_loss_rewards_target_mass(self):
        target = torch.tensor([[False, True, True, False]])
        mask = torch.ones_like(target)
        good = anchor_loss(
            torch.tensor([[-3.0, 2.0, 2.0, -3.0]]),
            target, mask,
        )
        bad = anchor_loss(
            torch.tensor([[2.0, -3.0, -3.0, 2.0]]),
            target, mask,
        )
        self.assertLess(float(good), float(bad))


if __name__ == "__main__":
    unittest.main()
