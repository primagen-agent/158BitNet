#!/usr/bin/env python3
"""Tests for the frozen-writer contiguous span tagger."""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_typed_span_tagger import (  # noqa: E402
    FieldSpanTagger,
    balanced_token_loss,
    field_loss,
    stratified_sample_indices,
)


class TypedSpanTaggerTests(unittest.TestCase):
    def test_initial_decoder_uses_writer_boundaries(self):
        tagger = FieldSpanTagger(
            2, 4,
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        )
        tokens = torch.tensor([[
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        start, end = tagger.best_span(tokens, mask)
        self.assertEqual(int(start[0]), 1)
        self.assertEqual(int(end[0]), 2)

    def test_decoder_respects_max_span_and_mask(self):
        tagger = FieldSpanTagger(
            2, 2,
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        )
        tokens = torch.tensor([[
            [5.0, 0.0],
            [0.0, 0.0],
            [0.0, 9.0],
            [0.0, 20.0],
        ]])
        mask = torch.tensor([[True, True, True, False]])
        start, end = tagger.best_span(tokens, mask)
        self.assertLessEqual(int(end[0]) - int(start[0]) + 1, 2)
        self.assertLess(int(end[0]), 3)

    def test_losses_are_finite(self):
        logits = torch.tensor([
            [-1.0, 2.0, 1.0, -2.0],
            [1.0, -1.0, -2.0, -3.0],
        ])
        targets = torch.tensor([
            [False, True, True, False],
            [True, False, False, False],
        ])
        mask = torch.ones_like(targets)
        self.assertTrue(torch.isfinite(
            balanced_token_loss(logits, targets, mask)
        ))

        scores = torch.randn(2, 4, 3)
        emissions = torch.randn(2, 4)
        batch = {
            "entity_span_start": torch.tensor([1, 0]),
            "entity_span_end": torch.tensor([2, 0]),
            "entity_token_target": targets,
            "mask": mask,
        }
        self.assertTrue(torch.isfinite(
            field_loss(
                scores, emissions, batch, "entity", 3
            )
        ))

    def test_stratified_sampling_preserves_create_examples(self):
        import random
        examples = (
            [{"operation": 0}] * 2
            + [{"operation": 1}] * 98
        )
        selected = stratified_sample_indices(
            examples, 20, 0.25, random.Random(7)
        )
        self.assertEqual(
            sum(
                examples[index]["operation"] == 0
                for index in selected
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()
