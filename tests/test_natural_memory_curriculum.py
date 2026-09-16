import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import prepare_repeated_dialogue_curriculum as catalog
from prepare_natural_memory_curriculum import world
from prepare_typed_memory_features import compile_episode_native_row
from prepare_typed_activation_curriculum import compile_row
from train_typed_memory_writer import load_writer_examples
from train_typed_pair_verifier import TypedPairVerifier


class Tokenizer:
    def __init__(self):
        self.ids = {}

    def encode(self, text, add_bos=False):
        result = [0] if add_bos else []
        for token in re.findall(r"\w+|[^\w\s]", text):
            result.append(self.ids.setdefault(token, len(self.ids) + 1))
        return result


class NaturalMemoryCurriculumTest(unittest.TestCase):
    def test_absolute_features_distinguish_small_candidate_sets(self):
        for positive, negative in (([10.0], [-10.0]), ([10.0, 8.0], [-10.0, -12.0])):
            left, right = torch.tensor(positive), torch.tensor(negative)
            self.assertTrue(torch.allclose(TypedPairVerifier.set_link_features(left),
                                          TypedPairVerifier.set_link_features(right)))
            self.assertFalse(torch.allclose(TypedPairVerifier.set_link_features(left, 7),
                                           TypedPairVerifier.set_link_features(right, 7)))
    def test_splits_do_not_share_relations(self):
        sets = [{item["name"] for item in relations} for relations in
                (catalog.LARGE_PREDICATES, catalog.VALIDATION_PREDICATES, catalog.TEST_PREDICATES)]
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))

    def test_raw_messages_keep_gold_offsets_and_active_targets(self):
        rows = world(2, 2710916, catalog.SPEAKERS, catalog.LARGE_PREDICATES, "train")
        first = rows[0]
        metadata = first["metadata"]
        events = metadata["typed_events"]
        self.assertEqual(sum(e["operation"] == "create" for e in events), len(events) // 2)
        self.assertTrue(any(not e["time"] for e in events))
        for row in rows:
            compiled = compile_episode_native_row(row)
            self.assertEqual(compiled["episodes"], metadata["raw_episodes"])
            active = compile_row(row)
            self.assertEqual(len(active["metadata"]["raw_episodes"]), len(events) // 2)
            for span in row["answer_spans"]:
                self.assertEqual(row["evidence"][span["start"]:span["end"]], row["answer"])

    def test_missing_time_uses_bos_without_inventing_a_date(self):
        rows = world(2, 2710916, catalog.SPEAKERS, catalog.LARGE_PREDICATES, "train")
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            features = Path(directory) / "features.pt"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            compiled = compile_episode_native_row(rows[0])
            ids = [tokenizer.encode(text, True) for text in compiled["episodes"]]
            torch.save({"backbone_sha256": "a" * 64,
                        "source_fingerprint": hashlib.sha256(raw.read_bytes()).hexdigest(),
                        "rows": [{**compiled, "episode_ids": ids,
                                  "episode_hidden": [torch.zeros(len(tokens), 2, 8) for tokens in ids]}]}, features)
            examples = load_writer_examples(features, raw, "a" * 64, tokenizer)
        for example in examples:
            if not example["time"]:
                self.assertEqual(example["time_token_target"].nonzero().flatten().tolist(), [0])


if __name__ == "__main__":
    unittest.main()
