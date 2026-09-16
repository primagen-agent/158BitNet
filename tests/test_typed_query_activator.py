#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_typed_query_activator import (  # noqa: E402
    SET_FEATURE_COUNT,
    TypedQueryActivator,
    evaluate,
    export_typed_query_activator_binary,
    row_loss,
    set_features,
)


def checkpoint():
    rank = 4
    width = 7
    hidden = 8
    return {
        "rank": rank,
        "verifier_state_dict": {
            "joint_head.0.weight":
                torch.randn(hidden, width),
            "joint_head.0.bias": torch.randn(hidden),
            "joint_head.2.weight": torch.randn(1, hidden),
            "joint_head.2.bias": torch.randn(1),
        },
    }


def row(null_target=False, target=1):
    return {
        "pair_features": torch.randn(3, 7),
        "entity_logits": torch.randn(3),
        "predicate_logits": torch.randn(3),
        "target": -1 if null_target else target,
        "null_target": null_target,
    }


class TypedQueryActivatorTest(unittest.TestCase):
    def test_set_features_are_finite(self):
        features = set_features(
            torch.tensor([1.0, 0.5, -0.5]),
            torch.tensor([0.9, 0.4, -0.2]),
            torch.tensor([0.8, 0.3, -0.3]),
        )
        self.assertEqual(
            tuple(features.shape),
            (SET_FEATURE_COUNT,),
        )
        self.assertTrue(torch.isfinite(features).all())

    def test_positive_and_null_losses_backpropagate(self):
        model = TypedQueryActivator(checkpoint())
        loss = (
            row_loss(model, row(), "cpu", 0.5)
            + row_loss(
                model, row(null_target=True),
                "cpu", 0.5,
            )
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(
            model.candidate_head[0].weight.grad
        )
        self.assertIsNotNone(
            model.null_head[0].weight.grad
        )

    def test_evaluation_reports_both_gates(self):
        model = TypedQueryActivator(checkpoint())
        metrics = evaluate(
            model,
            [row(), row(null_target=True)],
            "cpu",
        )
        self.assertEqual(metrics["examples"], 2)
        self.assertEqual(metrics["positive_examples"], 1)
        self.assertEqual(metrics["null_examples"], 1)

    def test_binary_export_binds_pair_checkpoint(self):
        model = TypedQueryActivator(checkpoint())
        with tempfile.TemporaryDirectory() as directory:
            pair = pathlib.Path(directory) / "pair.pt"
            pair_binary = (
                pathlib.Path(directory) / "pair.bntpair"
            )
            source = pathlib.Path(directory) / "query.pt"
            output = pathlib.Path(directory) / "query.bntqact"
            pair.write_bytes(b"pair-model")
            pair_binary.write_bytes(b"pair-binary")
            import hashlib
            pair_sha = hashlib.sha256(
                pair.read_bytes()
            ).hexdigest()
            torch.save({
                "format": "TYPED_QUERY_ACTIVATOR_V1",
                "backbone_sha256": "ab" * 32,
                "pair_checkpoint_fingerprint": pair_sha,
                "rank": model.rank,
                "input_width": model.input_width,
                "set_feature_count": SET_FEATURE_COUNT,
                "state_dict": model.state_dict(),
            }, source)
            export_typed_query_activator_binary(
                source, pair, pair_binary, output
            )
            payload = output.read_bytes()
        self.assertEqual(payload[:8], b"BNTQACT1")
        self.assertEqual(
            int.from_bytes(payload[8:12], "little"), 1
        )


if __name__ == "__main__":
    unittest.main()
