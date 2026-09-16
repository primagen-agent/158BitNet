#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_typed_memory_writer import TypedMemoryWriter  # noqa: E402
from export_typed_link_parity_sample import (  # noqa: E402
    export_typed_link_parity_sample,
)
from export_typed_pair_encoder import (  # noqa: E402
    export_typed_pair_encoder_binary,
)
from export_typed_pair_parity_sample import (  # noqa: E402
    export_typed_pair_parity_sample,
)
from train_typed_pair_verifier import (  # noqa: E402
    TypedPairVerifier,
    balanced_binary_loss,
    build_active_pairs,
    build_layout_consistency_groups,
    build_predecessor_groups,
    calibrate_binary_threshold,
    evaluate_with_train_calibration,
    export_typed_link_binary,
    grouped_ranking_loss,
    layout_consistency_loss,
    predecessor_set_losses,
    select_calibration_pairs,
    verifier_loss,
    version_link_diagnostics,
)


def example(
    episode, entity, predicate,
    operation, previous=None,
):
    return {
        "world_id": "world",
        "episode": episode,
        "hidden": torch.randn(5, 2, 8),
        "entity": entity,
        "predicate": predicate,
        "operation": operation,
        "position": episode / 4,
        "previous_episode": previous,
    }


class TypedPairVerifierTest(unittest.TestCase):
    def test_typed_pair_encoder_and_sample_export(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True)
        model = TypedPairVerifier(
            writer, rank=4, dual_path=True,
            set_link_head=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            binary = Path(directory) / "model.bntpair"
            sample = Path(directory) / "sample.bin"
            torch.save({
                "format": "TYPED_PAIR_VERIFIER_V1",
                "backbone_sha256": "ab" * 32,
                "hidden": 8,
                "rank": 4,
                "hidden_layer_bands":
                    [[0, 1], [2, 3]],
                "separate_address_localizer": True,
                "predict_span_boundaries": False,
                "use_token_embeddings": True,
                "hard_address_pooling": False,
                "contiguous_span_pooling": False,
                "max_field_span": 8,
                "attention_weighted_alignment": False,
                "dual_path": True,
                "trainable_context_localizer": False,
                "set_link_head": True,
                "set_link_head_version": 2,
                "writer_state_dict": writer.state_dict(),
                "verifier_state_dict": {
                    name: value
                    for name, value
                    in model.state_dict().items()
                    if not name.startswith("writer.")
                },
            }, checkpoint)
            export_typed_pair_encoder_binary(
                checkpoint, binary)
            export_typed_pair_parity_sample(
                checkpoint, sample,
                left_count=3, right_count=4,
                seed=13)
            binary_payload = binary.read_bytes()
            sample_payload = sample.read_bytes()
        self.assertEqual(
            binary_payload[:8], b"BNTPAIR1")
        self.assertEqual(
            int.from_bytes(
                binary_payload[8:12], "little"), 1)
        self.assertEqual(
            int.from_bytes(
                binary_payload[12:16], "little"), 8)
        self.assertEqual(
            int.from_bytes(
                binary_payload[32:36], "little"), 13)
        self.assertEqual(
            int.from_bytes(
                binary_payload[36:40], "little"), 25)
        self.assertEqual(
            sample_payload[:8], b"BNTPARF1")

    def test_typed_link_parity_sample_export(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2)
        model = TypedPairVerifier(
            writer, rank=4,
            set_link_head=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            sample = Path(directory) / "sample.bin"
            torch.save({
                "format": "TYPED_PAIR_VERIFIER_V1",
                "backbone_sha256": "ab" * 32,
                "rank": 4,
                "set_link_head": True,
                "set_link_head_version": 2,
                "verifier_state_dict": {
                    name: value
                    for name, value
                    in model.state_dict().items()
                    if not name.startswith("writer.")
                },
            }, checkpoint)
            export_typed_link_parity_sample(
                checkpoint, sample,
                pair_count=5, seed=11)
            payload = sample.read_bytes()
        self.assertEqual(payload[:8], b"BNTLPAR1")
        self.assertEqual(
            int.from_bytes(payload[8:12], "little"), 1)
        self.assertEqual(
            int.from_bytes(payload[12:16], "little"), 12)
        self.assertEqual(
            int.from_bytes(payload[16:20], "little"), 5)

    def test_typed_link_binary_export(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2)
        model = TypedPairVerifier(
            writer, rank=4,
            set_link_head=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            binary = Path(directory) / "model.bntlink"
            torch.save({
                "format": "TYPED_PAIR_VERIFIER_V1",
                "backbone_sha256": "ab" * 32,
                "rank": 4,
                "set_link_head": True,
                "set_link_head_version": 2,
                "verifier_state_dict": {
                    name: value
                    for name, value
                    in model.state_dict().items()
                    if not name.startswith("writer.")
                },
            }, checkpoint)
            export_typed_link_binary(
                checkpoint, binary)
            payload = binary.read_bytes()
        self.assertEqual(payload[:8], b"BNTLINK1")
        self.assertEqual(
            int.from_bytes(payload[8:12], "little"), 1)
        self.assertEqual(
            int.from_bytes(payload[12:16], "little"), 2)
        self.assertEqual(
            int.from_bytes(payload[16:20], "little"), 4)
        self.assertEqual(
            int.from_bytes(payload[20:24], "little"), 12)
        self.assertEqual(
            int.from_bytes(payload[24:28], "little"), 8)
        self.assertEqual(
            int.from_bytes(payload[28:32], "little"), 5)
        self.assertEqual(
            int.from_bytes(payload[32:36], "little"), 8)

    def test_set_link_head_trains_on_missing_predecessor(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2)
        model = TypedPairVerifier(
            writer, rank=4,
            set_link_head=True)
        examples = [
            example(0, "a", "x", 0),
            example(1, "a", "y", 0),
            example(2, "b", "x", 0),
            example(3, "a", "x", 1, 0),
        ]
        pairs = build_active_pairs(examples)
        groups = build_predecessor_groups(
            examples, pairs)
        ranking, existence = predecessor_set_losses(
            model, examples, pairs,
            groups, [0], "cpu")
        (ranking + existence).backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter
            in model.joint_head.parameters()))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter
            in model.predecessor_exists_head.parameters()))

    def test_set_link_diagnostics_use_exist_head(self):
        examples = [
            example(0, "a", "x", 0),
            example(1, "a", "y", 0),
            example(2, "b", "x", 0),
            example(3, "a", "x", 1, 0),
        ]
        pairs = build_active_pairs(examples)
        entity = torch.zeros(len(pairs))
        predicate = torch.zeros(len(pairs))
        joint = torch.full((len(pairs),), -2.0)
        for pair_index, pair in enumerate(pairs):
            if pair["left"] == 3 and pair["joint_same"]:
                joint[pair_index] = 2.0
        operation = torch.tensor([
            [2.0, -2.0],
            [2.0, -2.0],
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        accepted = version_link_diagnostics(
            examples, pairs, entity,
            predicate, operation,
            joint_logits=joint,
            predecessor_exists={3: 2.0},
            missing_predecessor_exists={3: -2.0})
        self.assertEqual(
            accepted["version_link_accuracy"], 1.0)
        self.assertEqual(
            accepted[
                "predecessor_exists_balanced_accuracy"],
            1.0)
        rejected = version_link_diagnostics(
            examples, pairs, entity,
            predicate, operation,
            joint_logits=joint,
            predecessor_exists={3: -2.0},
            missing_predecessor_exists={3: -2.0})
        self.assertEqual(
            rejected["update_no_candidate_rate"], 1.0)

    def test_calibration_pairs_are_stratified(self):
        pairs = [
            {
                "index": index,
                "entity_same": entity_same,
                "predicate_same": predicate_same,
            }
            for entity_same in (False, True)
            for predicate_same in (False, True)
            for index in range(10)
        ]
        selected = select_calibration_pairs(
            pairs, 12, seed=19)
        counts = {}
        for pair in selected:
            key = (
                pair["entity_same"],
                pair["predicate_same"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(selected), 12)
        self.assertEqual(
            set(counts.values()), {3})
        self.assertEqual(
            select_calibration_pairs(
                pairs, 0, seed=19),
            pairs)

    @patch("train_typed_pair_verifier.evaluate")
    @patch(
        "train_typed_pair_verifier.calibrate_pair_thresholds",
        return_value=(1.25, -0.75, 0.91, 0.87))
    def test_checkpoint_metrics_use_train_calibration(
        self, calibrate, evaluate,
    ):
        evaluate.return_value = {
            "version_link_accuracy": 0.8,
        }
        metrics = evaluate_with_train_calibration(
            object(), object(),
            ["train"], ["train-pair"],
            ["valid"], ["valid-pair"],
            "cpu", 16)
        calibrate.assert_called_once()
        evaluate.assert_called_once_with(
            ANY, ANY,
            ["valid"], ["valid-pair"],
            "cpu", 16, 1.25, -0.75)
        self.assertEqual(
            metrics["train_calibrated_entity_accuracy"],
            0.91)
        self.assertEqual(
            metrics["train_calibrated_predicate_accuracy"],
            0.87)

    def test_active_pairs_include_joint_and_hard_negatives(self):
        examples = [
            example(0, "a", "x", 0),
            example(1, "a", "y", 0),
            example(2, "b", "x", 0),
            example(3, "a", "x", 1, 0),
        ]
        pairs = build_active_pairs(examples)
        final = [
            pair for pair in pairs if pair["left"] == 3]
        self.assertTrue(any(
            pair["joint_same"] for pair in final))
        self.assertTrue(any(
            pair["entity_same"]
            and not pair["predicate_same"]
            for pair in final))
        self.assertTrue(any(
            pair["predicate_same"]
            and not pair["entity_same"]
            for pair in final))

    def test_forward_and_loss_have_gradients(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2)
        model = TypedPairVerifier(writer, rank=4)
        left = torch.randn(8, 6, 2, 8)
        right = torch.randn(8, 5, 2, 8)
        output = model(
            left, right,
            torch.ones(8, 6, dtype=torch.bool),
            torch.ones(8, 5, dtype=torch.bool))
        batch = {
            "entity_target": torch.tensor(
                [1, 1, 0, 0, 1, 0, 1, 0],
                dtype=torch.float32),
            "predicate_target": torch.tensor(
                [1, 0, 1, 0, 1, 0, 0, 1],
                dtype=torch.float32),
        }
        batch["joint_target"] = (
            batch["entity_target"]
            * batch["predicate_target"])
        loss, _ = verifier_loss(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if not name.startswith("writer.")))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in writer.parameters()))

    def test_pair_verifier_uses_frozen_span_localizer(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True)
        model = TypedPairVerifier(writer, rank=4)
        output = model(
            torch.randn(4, 6, 2, 8),
            torch.randn(4, 5, 2, 8),
            torch.ones(4, 6, dtype=torch.bool),
            torch.ones(4, 5, dtype=torch.bool))
        self.assertEqual(
            output["entity_logit"].shape, (4,))
        self.assertIsNone(model.token_keys)

    def test_pair_verifier_accepts_tied_token_embeddings(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True)
        model = TypedPairVerifier(writer, rank=4)
        output = model(
            torch.randn(4, 6, 2, 8),
            torch.randn(4, 5, 2, 8),
            torch.ones(4, 6, dtype=torch.bool),
            torch.ones(4, 5, dtype=torch.bool),
            torch.randn(4, 6, 8),
            torch.randn(4, 5, 8))
        self.assertEqual(
            output["predicate_logit"].shape, (4,))

    def test_dual_path_fuses_context_and_identity(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True)
        model = TypedPairVerifier(
            writer, rank=4, dual_path=True)
        output = model(
            torch.randn(4, 6, 2, 8),
            torch.randn(4, 5, 2, 8),
            torch.ones(4, 6, dtype=torch.bool),
            torch.ones(4, 5, dtype=torch.bool),
            torch.randn(4, 6, 8),
            torch.randn(4, 5, 8))
        output["entity_logit"].sum().backward()
        self.assertIsNotNone(
            model.projections[
                "entity"].weight.grad)
        self.assertIsNotNone(
            model.identity_projections[
                "entity"].weight.grad)
        self.assertIsNotNone(
            model.fusion_gates[
                "entity"][-1].weight.grad)

    def test_dual_path_can_retrain_context_localizer(self):
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True)
        model = TypedPairVerifier(
            writer, rank=4, dual_path=True,
            trainable_context_localizer=True)
        output = model(
            torch.randn(4, 6, 2, 8),
            torch.randn(4, 5, 2, 8),
            torch.ones(4, 6, dtype=torch.bool),
            torch.ones(4, 5, dtype=torch.bool),
            torch.randn(4, 6, 8),
            torch.randn(4, 5, 8))
        output["predicate_logit"].sum().backward()
        self.assertIsNotNone(
            model.context_localizer_projections[
                "predicate"].weight.grad)
        self.assertIsNotNone(
            model.context_token_keys.grad)

    def test_layout_consistency_pairs_semantic_views(self):
        examples = []
        for view in (0, 1):
            for item in (
                example(0, "a", "x", 0),
                example(1, "a", "y", 0),
                example(2, "b", "x", 0),
                example(3, "a", "x", 1, 0),
            ):
                item["world_id"] = f"world-view{view}"
                item["semantic_world_id"] = "world"
                item["layout_view"] = view
                item["identity_hidden"] = torch.randn(5, 8)
                examples.append(item)
        pairs = build_active_pairs(examples)
        groups = build_layout_consistency_groups(
            examples)
        self.assertEqual(len(groups), 4)
        writer = TypedMemoryWriter(
            hidden=8, rank=4, bands=2,
            separate_address_localizer=True,
            use_token_embeddings=True)
        model = TypedPairVerifier(
            writer, rank=4, dual_path=True)
        loss = layout_consistency_loss(
            model, examples, groups,
            [0, 1], "cpu")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(
            model.projections[
                "entity"].weight.grad)

    def test_predecessor_groups_and_ranking_loss(self):
        examples = [
            example(0, "a", "x", 0),
            example(1, "a", "y", 0),
            example(2, "b", "x", 0),
            example(3, "a", "x", 1, 0),
        ]
        pairs = build_active_pairs(examples)
        groups = build_predecessor_groups(
            examples, pairs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            examples[pairs[
                groups[0]["pairs"][
                    groups[0]["target"]]
            ]["right"]]["episode"],
            0)
        target = groups[0]["target"]
        good = torch.full((3,), -2.0)
        good[target] = 3.0
        bad = torch.full((3,), -2.0)
        bad[(target + 1) % 3] = 3.0
        self.assertLess(
            float(grouped_ranking_loss(
                good, [3], [target])),
            float(grouped_ranking_loss(
                bad, [3], [target])))

    def test_binary_loss_rewards_separation(self):
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        good = torch.tensor([2.0, 1.0, -1.0, -2.0])
        bad = -good
        self.assertLess(
            float(balanced_binary_loss(good, targets)),
            float(balanced_binary_loss(bad, targets)))
        threshold, accuracy = calibrate_binary_threshold(
            good, targets)
        self.assertEqual(accuracy, 1.0)
        self.assertGreater(threshold, -1.0)
        self.assertLess(threshold, 1.0)

    def test_version_diagnostics_separate_ranking_and_rejection(self):
        examples = [
            example(0, "a", "x", 0),
            example(1, "a", "y", 0),
            example(2, "b", "x", 0),
            example(3, "a", "x", 1, 0),
        ]
        pairs = build_active_pairs(examples)
        entity = torch.full((len(pairs),), -2.0)
        predicate = torch.full((len(pairs),), -2.0)
        for pair_index, pair in enumerate(pairs):
            if pair["left"] == 3 and pair["joint_same"]:
                entity[pair_index] = 2.0
                predicate[pair_index] = 2.0
        operation = torch.tensor([
            [2.0, -2.0],
            [2.0, -2.0],
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        good = version_link_diagnostics(
            examples, pairs, entity,
            predicate, operation)
        self.assertEqual(
            good["version_link_accuracy"], 1.0)
        self.assertEqual(
            good["predecessor_rank_accuracy"], 1.0)
        rejected = version_link_diagnostics(
            examples, pairs, entity - 4.0,
            predicate - 4.0, operation)
        self.assertEqual(
            rejected["predecessor_rank_accuracy"], 1.0)
        self.assertEqual(
            rejected["predecessor_accept_accuracy"], 0.0)
        self.assertEqual(
            rejected["update_no_candidate_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
