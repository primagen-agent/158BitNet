import json
import re
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from prepare_dialogue_memory_curriculum import world, NAMES, SCHEMAS, TEMPLATES
from prepare_typed_memory_features import compile_episode_native_row
from train_typed_memory_writer import load_raw_worlds
from typed_memory_training import token_span_variants


class DialogueCurriculumTest(unittest.TestCase):
    def test_contextual_token_boundaries_are_aligned_without_retokenizing(self):
        class Tokenizer:
            def encode(self, text, add_bos=False):
                return [999]  # Standalone ids deliberately do not occur in source.
            def bos(self):
                return 0
            def decode_pieces(self, ids):
                bank = [b"<s>", b" [", b"2024", b"-", b"03", b"01", b"]", b" busy", b" bus", b"bus."]
                return [bank[i] for i in ids]
        tokenizer = Tokenizer()
        self.assertEqual(token_span_variants([0, 1, 2, 3, 4, 3, 5, 6], "2024-03-01", tokenizer), (2, 6))
        self.assertEqual(token_span_variants([0, 7, 8], "bus", tokenizer), (2, 2))
        self.assertIsNone(token_span_variants([0, 8, 8], "bus", tokenizer))
        self.assertIsNone(token_span_variants([0, 9], "bus", tokenizer))

    def test_split_templates_names_and_values_do_not_overlap(self):
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            self.assertTrue(set(TEMPLATES[left]).isdisjoint(TEMPLATES[right]))
            self.assertTrue(set(NAMES[left]).isdisjoint(NAMES[right]))
        for schema in SCHEMAS:
            self.assertEqual(len(set(schema[1])), 12)
            for operation in (2, 3):
                self.assertEqual(len(set(t[0] for t in schema[operation])), 4)

    def test_gold_spans_match_raw_dialogue(self):
        for split in NAMES:
            for index in range(24):
                rows = world(index, 2740916, split)
                meta = rows[0]["metadata"]
                self.assertEqual(len(meta["typed_events"]), 8)
                for message, event in zip(meta["raw_episodes"], meta["typed_events"]):
                    for field in ("entity", "predicate", "value", "time"):
                        surface = event[field + "_surface"]
                        if surface:
                            self.assertEqual(len(re.findall(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", message)),
                                             1, (message, field, surface))
                for row in rows:
                    self.assertEqual(compile_episode_native_row(row)["episodes"], meta["raw_episodes"])
                    for span in row["answer_spans"]:
                        self.assertEqual(row["evidence"][span["start"]:span["end"]], row["answer"])

    def test_sealed_test_cannot_be_loaded_by_writer_trainer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sealed.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in world(0, 2740916, "test")))
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                load_raw_worlds(path)


if __name__ == "__main__":
    unittest.main()
