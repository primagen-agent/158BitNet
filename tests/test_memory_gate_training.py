import ast
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
import zlib

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_memory_gate import PREFIX, SUFFIX, cases, disjoint_cases, pool_features, selection_key, export_controller
from eval_memory_chat_holdout import assert_query_did_not_write


class MemoryGateTrainingTest(unittest.TestCase):
    def test_training_prompt_matches_server(self):
        text = (Path(__file__).resolve().parents[1] / "examples/openai_server.c").read_text()
        for name, expected in (("action_prefix", PREFIX), ("action_suffix", SUFFIX)):
            expression = re.search(r"static const char " + name + r"\[\] =(.+?);", text, re.S).group(1)
            actual = "".join(ast.literal_eval(part) for part in re.findall(r'"(?:\\.|[^"\\])*"', expression))
            self.assertEqual(actual, expected)

    def test_pool_includes_bos_and_is_mean_plus_last(self):
        x = torch.tensor([[10., 20.], [2., 4.], [3., 6.]])
        self.assertTrue(torch.equal(pool_features(x), x.mean(0) + x[-1]))
        self.assertTrue(torch.equal(pool_features(x, "last"), x[-1]))

    def test_export_has_crc_binding_and_disables_update_delete(self):
        head = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.SiLU(), torch.nn.Linear(4, 2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gate.bnctrl"
            export_controller(head, 8, 4, "ab" * 32, path)
            data = path.read_bytes()
        self.assertEqual(data[:8], b"BNCTRL4\0")
        self.assertEqual(data[36:68].hex(), "ab" * 32)
        crcs = struct.unpack("<6I", data[68:92]);offset = 92
        for length, crc in zip((32, 32, 128, 16, 64, 16), crcs):
            self.assertEqual(zlib.crc32(data[offset:offset + length]) & 0xffffffff, crc)
            offset += length
        self.assertEqual(offset, len(data))
        self.assertEqual(struct.unpack("<4f", data[-16:])[2:], (-1e9, -1e9))

    def test_false_write_safety_precedes_acceptance(self):
        safe = {"false_writes": 0, "worst_positive_recall": .8, "write_correct": 8}
        unsafe = {"false_writes": 1, "worst_positive_recall": 1., "write_correct": 10}
        self.assertGreater(selection_key(safe), selection_key(unsafe))
        with self.assertRaises(AssertionError):
            assert_query_did_not_write({"memory_auto": {"stored": 1}})

    def test_no_answer_labels_in_gate_and_no_cross_split_duplicates(self):
        row = {"question": "Where does Avery live?", "answer": "GOLD_NOT_TO_COPY",
               "metadata": {"world_id": "dialogue-train-0", "training_domain": "dialogue",
                            "raw_episodes": ["Avery: I live in Bergen. How are you?"], "typed_events": [{}]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.jsonl";path.write_text(json.dumps(row))
            result = cases(path, "train")
            self.assertNotIn("GOLD_NOT_TO_COPY", json.dumps(result))
            self.assertTrue(any(r["label"] == 1 and r["text"].endswith("?") for r in result))
            self.assertTrue(any(r["label"] == 0 and "Avery:" in r["text"] for r in result))
            self.assertNotIn(result[0], disjoint_cases(result, [result[0]]))
            row["evaluation_only"] = True;path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                cases(path, "train")

    def test_rich_intents_change_only_training_and_include_opposite_intents(self):
        row = {"question": "Which instrument does Morgan play?", "answer": "GOLD_NOT_TO_COPY",
               "metadata": {"world_id": "dialogue-train-0", "training_domain": "dialogue",
                            "raw_episodes": ["Morgan plays the cello."], "typed_events": [{}]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.jsonl"
            path.write_text(json.dumps(row))
            basic = cases(path, "train")
            rich = cases(path, "train", curriculum="rich_intents")
            self.assertTrue(all(r in rich for r in basic))
            self.assertNotIn("GOLD_NOT_TO_COPY", json.dumps(rich))
            self.assertTrue(any(r["label"] and "retain" in r["text"] or
                                r["label"] and "future conversations" in r["text"] for r in rich))
            self.assertTrue(any(r["label"] and "proofreading" in r["text"] for r in rich))
            self.assertTrue(any(not r["label"] and "translator" in r["text"] for r in rich))
            row["metadata"]["world_id"] = "dialogue-valid-0"
            path.write_text(json.dumps(row))
            self.assertEqual(cases(path, "valid"), cases(path, "valid", curriculum="rich_intents"))


if __name__ == "__main__":
    unittest.main()
