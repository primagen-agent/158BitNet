#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_typed_memory_writer import (  # noqa: E402
    TypedMemoryWriter,
    attention_span_loss,
    balanced_pair_loss,
    compiled_attention_version_accuracy,
    compiled_attention_topk_version_accuracy,
    compiled_token_version_accuracy,
    contiguous_span_distribution,
    evaluate_with_train_calibration,
    hard_negative_pair_loss,
    segment_span_loss,
    span_boundary_loss,
    version_link_accuracy,
    writer_loss,
)


class TypedMemoryWriterTest(unittest.TestCase):
    @patch("train_typed_memory_writer.evaluate")
    @patch(
        "train_typed_memory_writer.calibrate_version_threshold",
        return_value=(1.5, 0.92))
    def test_checkpoint_metrics_use_train_calibration(
        self, calibrate, evaluate,
    ):
        evaluate.return_value = {
            "version_link_accuracy": 0.81,
        }
        metrics = evaluate_with_train_calibration(
            ANY, ["train"], ["valid"], "cpu", 16)
        calibrate.assert_called_once_with(
            ANY, ["train"], "cpu", 16)
        evaluate.assert_called_once_with(
            ANY, ["valid"], "cpu", 16, 1.5)
        self.assertEqual(
            metrics[
                "train_calibrated_version_link_accuracy"],
            0.92)

    def test_forward_and_loss_have_gradients(self):
        model = TypedMemoryWriter(
            hidden=8, rank=4, bands=2)
        hidden = torch.randn(8, 6, 2, 8)
        mask = torch.ones(8, 6, dtype=torch.bool)
        rows = [
            {
                "entity": f"person-{index % 2}",
                "predicate": f"field-{index % 2}",
            }
            for index in range(8)
        ]
        batch = {
            "rows": rows,
            "operation": torch.tensor(
                [0, 1, 1, 1, 0, 1, 1, 1]),
            "position": torch.linspace(-1.0, 1.0, 8),
        }
        output = model(hidden, mask)
        loss, parts = writer_loss(
            model, output, batch,
            torch.ones(2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {"entity", "predicate", "joint",
             "entity_hard", "predicate_hard",
             "joint_hard", "operation", "time",
             "entity_attention",
             "predicate_attention",
             "value_attention",
             "time_attention",
             "entity_boundary",
             "predicate_boundary",
             "value_boundary",
             "time_boundary",
             "entity_segment",
             "predicate_segment",
             "value_segment",
             "time_segment"})
        self.assertTrue(all(
            state.shape == (8, 4)
            for name, state in output.items()
            if name in ("entity", "predicate",
                        "value", "time")))

    def test_balanced_pair_loss_rewards_separation(self):
        targets = torch.tensor([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        good = torch.tensor([
            [3.0, 2.0, -2.0],
            [2.0, 3.0, -2.0],
            [-2.0, -2.0, 3.0],
        ])
        bad = -good
        self.assertLess(
            float(balanced_pair_loss(good, targets)),
            float(balanced_pair_loss(bad, targets)))
        self.assertLess(
            float(hard_negative_pair_loss(
                good, targets, count=1)),
            float(hard_negative_pair_loss(
                bad, targets, count=1)))

    def test_attention_span_loss_rewards_gold_mass(self):
        target = torch.tensor([
            [False, True, True, False],
            [True, False, False, False],
        ])
        good = torch.tensor([
            [0.05, 0.45, 0.45, 0.05],
            [0.90, 0.05, 0.03, 0.02],
        ])
        bad = torch.tensor([
            [0.45, 0.05, 0.05, 0.45],
            [0.02, 0.40, 0.30, 0.28],
        ])
        self.assertLess(
            float(attention_span_loss(good, target)),
            float(attention_span_loss(bad, target)))

    def test_separate_localizer_decouples_gradients(self):
        model = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True)
        hidden = torch.randn(4, 6, 2, 8)
        mask = torch.ones(4, 6, dtype=torch.bool)
        output = model(hidden, mask)
        output["entity"][:, 0].sum().backward()
        self.assertIsNone(
            model.localizer_projections[
                "entity"].weight.grad)
        self.assertIsNotNone(
            model.projections["entity"].weight.grad)
        model.zero_grad(set_to_none=True)
        output = model(hidden, mask)
        target = torch.zeros(4, 6, dtype=torch.bool)
        target[:, 2] = True
        attention_span_loss(
            output["attentions"]["entity"],
            target).backward()
        self.assertIsNotNone(
            model.localizer_projections[
                "entity"].weight.grad)
        self.assertIsNone(
            model.projections["entity"].weight.grad)

    def test_token_embedding_address_ignores_context(self):
        model = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True)
        mask = torch.ones(2, 6, dtype=torch.bool)
        identity = torch.randn(2, 6, 8)
        identity[1] = identity[0]
        hidden = torch.randn(2, 6, 2, 8)
        hidden[1] = hidden[0]
        output = model(hidden, mask, identity)
        self.assertTrue(torch.allclose(
            output["entity"][0], output["entity"][1],
            atol=1e-6))

    def test_hard_address_pooling_selects_one_token(self):
        model = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True,
            hard_address_pooling=True)
        hidden = torch.randn(2, 6, 2, 8)
        identity = torch.randn(2, 6, 8)
        output = model(
            hidden,
            torch.ones(2, 6, dtype=torch.bool),
            identity)
        peak = output["attentions"][
            "entity"].argmax(dim=-1)
        expected = F.normalize(
            model.projections["entity"](
                identity[
                    torch.arange(2), peak]),
            dim=-1, eps=1e-12)
        self.assertTrue(torch.allclose(
            output["entity"], expected,
            atol=1e-6))

    def test_contiguous_span_pooling_is_normalized(self):
        score = torch.tensor([
            [0.0, 3.0, 3.0, 0.0],
        ])
        mask = torch.ones(1, 4, dtype=torch.bool)
        logits, inclusion = (
            contiguous_span_distribution(
                score, mask,
                torch.zeros(3), 3))
        self.assertEqual(logits.shape, (1, 4, 3))
        self.assertAlmostEqual(
            float(inclusion.sum()), 1.0,
            places=6)
        self.assertGreater(
            float(inclusion[0, 1:3].sum()),
            float(inclusion[0, [0, 3]].sum()))

    def test_segment_span_loss_rewards_exact_segment(self):
        output = {
            "segment_logits": {
                name: torch.full((2, 5, 4), -3.0)
                for name in (
                    "entity", "predicate", "value", "time"
                )
            },
        }
        batch = {
            "entity_span_start": torch.tensor([1, 2]),
            "entity_span_end": torch.tensor([1, 3]),
            "predicate_span_start": torch.tensor([2, 1]),
            "predicate_span_end": torch.tensor([3, 3]),
            "value_span_start": torch.tensor([0, 1]),
            "value_span_end": torch.tensor([2, 2]),
            "time_span_start": torch.tensor([1, 0]),
            "time_span_end": torch.tensor([3, 2]),
        }
        for name in (
            "entity", "predicate", "value", "time"
        ):
            for row in range(2):
                start = int(batch[
                    f"{name}_span_start"][row])
                end = int(batch[
                    f"{name}_span_end"][row])
                output["segment_logits"][
                    name][row, start, end - start] = 3.0
        loss, parts = segment_span_loss(
            output, batch, max_span=4)
        self.assertLess(float(loss), 0.2)
        self.assertEqual(
            set(parts),
            {"entity", "predicate", "value", "time"})

    def test_segment_head_receives_span_gradient(self):
        model = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True,
            contiguous_span_pooling=True,
            max_field_span=4)
        output = model(
            torch.randn(4, 6, 2, 8),
            torch.ones(4, 6, dtype=torch.bool),
            torch.randn(4, 6, 8))
        batch = {
            "entity_span_start":
                torch.tensor([1, 1, 2, 2]),
            "entity_span_end":
                torch.tensor([1, 2, 2, 3]),
            "predicate_span_start":
                torch.tensor([2, 2, 3, 3]),
            "predicate_span_end":
                torch.tensor([3, 4, 4, 5]),
            "value_span_start":
                torch.tensor([1, 1, 2, 2]),
            "value_span_end":
                torch.tensor([2, 3, 3, 4]),
            "time_span_start":
                torch.tensor([0, 0, 0, 0]),
            "time_span_end":
                torch.tensor([2, 2, 2, 2]),
        }
        loss, _ = segment_span_loss(
            output, batch, max_span=4)
        loss.backward()
        self.assertIsNotNone(
            model.segment_heads[
                "predicate"][-1].weight.grad)

    def test_span_boundary_heads_learn_exact_endpoints(self):
        model = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            predict_span_boundaries=True)
        hidden = torch.randn(4, 6, 2, 8)
        mask = torch.ones(4, 6, dtype=torch.bool)
        output = model(hidden, mask)
        batch = {
            "entity_span_start":
                torch.tensor([1, 1, 2, 2]),
            "entity_span_end":
                torch.tensor([1, 2, 2, 3]),
            "predicate_span_start":
                torch.tensor([3, 3, 4, 4]),
            "predicate_span_end":
                torch.tensor([4, 4, 5, 5]),
            "value_span_start":
                torch.tensor([1, 2, 1, 2]),
            "value_span_end":
                torch.tensor([2, 3, 2, 3]),
            "time_span_start":
                torch.tensor([0, 0, 0, 0]),
            "time_span_end":
                torch.tensor([1, 1, 1, 1]),
        }
        loss, parts = span_boundary_loss(
            output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {"entity", "predicate", "value", "time"})
        self.assertIsNotNone(
            model.span_boundary_keys.grad)

    def test_compiled_token_key_links_latest_version(self):
        examples = [
            {
                "world_id": "w",
                "episode": 0,
                "operation": 0,
                "previous_episode": None,
                "token_ids": [0, 10, 20],
            },
            {
                "world_id": "w",
                "episode": 1,
                "operation": 0,
                "previous_episode": None,
                "token_ids": [0, 10, 21],
            },
            {
                "world_id": "w",
                "episode": 2,
                "operation": 0,
                "previous_episode": None,
                "token_ids": [0, 11, 20],
            },
            {
                "world_id": "w",
                "episode": 3,
                "operation": 1,
                "previous_episode": 0,
                "token_ids": [0, 10, 20],
            },
        ]
        operation = torch.tensor([
            [2.0, -2.0],
            [2.0, -2.0],
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        output = {
            "entity_span_start":
                torch.tensor([1, 1, 1, 1]),
            "entity_span_end":
                torch.tensor([1, 1, 1, 1]),
            "predicate_span_start":
                torch.tensor([2, 2, 2, 2]),
            "predicate_span_end":
                torch.tensor([2, 2, 2, 2]),
        }
        accuracy, valid = (
            compiled_token_version_accuracy(
                examples, operation, output))
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(valid, 1.0)
        output["entity_span_start"][-1] = 0
        output["entity_span_end"][-1] = 0
        accuracy, _ = compiled_token_version_accuracy(
            examples, operation, output)
        self.assertEqual(accuracy, 0.75)

    def test_attention_peak_compiler_uses_token_identity(self):
        examples = [
            {
                "world_id": "w",
                "episode": 0,
                "operation": 0,
                "previous_episode": None,
                "token_ids": [0, 10, 20],
            },
            {
                "world_id": "w",
                "episode": 1,
                "operation": 1,
                "previous_episode": 0,
                "token_ids": [0, 10, 20],
            },
        ]
        operation = torch.tensor([
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        output = {
            "entity_attention_peak":
                torch.tensor([1, 1]),
            "predicate_attention_peak":
                torch.tensor([2, 2]),
        }
        accuracy, valid = (
            compiled_attention_version_accuracy(
                examples, operation, output))
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(valid, 1.0)

    def test_attention_topk_compiler_ignores_weight_order(self):
        examples = [
            {
                "world_id": "w",
                "episode": 0,
                "operation": 0,
                "previous_episode": None,
                "token_ids": [0, 10, 11, 20, 21],
            },
            {
                "world_id": "w",
                "episode": 1,
                "operation": 1,
                "previous_episode": 0,
                "token_ids": [0, 10, 11, 20, 21],
            },
        ]
        operation = torch.tensor([
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        output = {
            "entity_attention_top4": torch.tensor([
                [1, 2, 0, 3],
                [2, 1, 3, 0],
            ]),
            "predicate_attention_top4": torch.tensor([
                [3, 4, 2, 1],
                [4, 3, 1, 2],
            ]),
        }
        accuracy = (
            compiled_attention_topk_version_accuracy(
                examples, operation, output,
                entity_k=2, predicate_k=2))
        self.assertEqual(accuracy, 1.0)

    def test_version_compiler_uses_latest_matching_predecessor(self):
        examples = [
            {
                "world_id": "w",
                "episode": 0,
                "operation": 0,
                "previous_episode": None,
            },
            {
                "world_id": "w",
                "episode": 1,
                "operation": 0,
                "previous_episode": None,
            },
            {
                "world_id": "w",
                "episode": 2,
                "operation": 1,
                "previous_episode": 0,
            },
            {
                "world_id": "w",
                "episode": 3,
                "operation": 1,
                "previous_episode": 2,
            },
        ]
        operation = torch.tensor([
            [3.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
        ])
        joint = torch.full((4, 4), -2.0)
        joint[2, 0] = 2.0
        joint[3, 0] = 2.0
        joint[3, 2] = 2.0
        self.assertEqual(
            version_link_accuracy(
                examples, operation, joint),
            1.0)

    def test_version_threshold_rejects_tail_false_match(self):
        examples = [
            {
                "world_id": "w",
                "episode": 0,
                "operation": 0,
                "previous_episode": None,
            },
            {
                "world_id": "w",
                "episode": 1,
                "operation": 0,
                "previous_episode": None,
            },
            {
                "world_id": "w",
                "episode": 2,
                "operation": 1,
                "previous_episode": 0,
            },
        ]
        operation = torch.tensor([
            [3.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
        ])
        joint = torch.full((3, 3), -2.0)
        joint[2, 0] = 2.0
        joint[2, 1] = 0.25
        self.assertLess(
            version_link_accuracy(
                examples, operation, joint, 0.0),
            version_link_accuracy(
                examples, operation, joint, 0.5))

if __name__ == "__main__":
    unittest.main()
