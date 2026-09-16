import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_typed_memory_writer import writer_backbone_config, load_raw_worlds
from typed_memory_training import reject_evaluation_row
from eval_memory_chat_holdout import load_holdout, judge


class MemoryReviewTests(unittest.TestCase):
    @patch("train_typed_memory_writer.file_fingerprint", return_value="ab" * 32)
    @patch("train_typed_memory_writer.GGUFWeights")
    def test_fresh_writer_reads_actual_backbone(self, weights, fingerprint):
        weights.return_value = MagicMock(hidden=896, n_layers=24)
        config = writer_backbone_config("model.gguf", "shim.so", "0-5,6-11,12-17,18-23")
        self.assertEqual(config["hidden"], 896)
        self.assertEqual(config["hidden_layer_bands"][3], list(range(18, 24)))
        weights.return_value.close.assert_called_once()

    @patch("train_typed_memory_writer.file_fingerprint", return_value="ab" * 32)
    @patch("train_typed_memory_writer.GGUFWeights")
    def test_legacy_checkpoint_cannot_change_backbone(self, weights, fingerprint):
        weights.return_value = MagicMock(hidden=896, n_layers=24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            torch.save({"backbone_sha256": "cd" * 32}, path)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                writer_backbone_config("model.gguf", "shim.so", "0-5,6-11,12-17,18-23", path)

    def test_evaluation_data_is_rejected(self):
        for row in ({"evaluation_only": True}, {"metadata": {"evaluation_only": True}},
                    {"metadata": {"locomo_used": True}}):
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                reject_evaluation_row(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.jsonl"
            path.write_text(json.dumps({"evaluation_only": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                load_raw_worlds(path)

    def test_plain_generation_is_not_memory_success(self):
        query = {"kind": "current", "answer": "Tokyo"}
        response = {"choices": [{"message": {"content": "Tokyo"}}]}
        self.assertFalse(judge(query, response))
        response["memory_copy"] = {"mode": "neural_typed_query_activation_then_compiled_pointer"}
        self.assertTrue(judge(query, response))

    def test_holdout_contains_raw_messages_not_gold_events(self):
        worlds = load_holdout(Path(__file__).parent / "fixtures/memory_chat_holdout_v1.json")
        self.assertEqual(len(worlds), 6)
        for world in worlds:
            self.assertTrue(world["id"].startswith("sealed-chat-v1-"))
            self.assertTrue(all(isinstance(text, str) for text in world["writes"]))
            self.assertNotIn("typed_events", world)


if __name__ == "__main__":
    unittest.main()
