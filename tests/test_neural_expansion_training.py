"""Authored material worlds expose specific, complete and absent supports."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import prepare_neural_reader_expansion as expansion
import train_neural_dialogue_reader as reader
from neural_reader_evidence_curriculum import augment_evidence
from neural_fit_support import semantic_relation_masks


class ExpansionTrainingContract(unittest.TestCase):
    def test_existing_evidence_samples_are_exact_and_new_questions_remain_raw(self):
        group = expansion.reader_expansion()
        original = dict(facts=group["facts"], queries=[q for q in group["queries"]
                        if not q.get("expansion_query_class")])
        expected = augment_evidence(original, 5322, 400)
        before = deepcopy(group)
        actual = reader.expansion_evidence(group)
        self.assertEqual(actual["facts"], expected["facts"])
        self.assertEqual(actual["queries"][:-16], expected["queries"])
        self.assertEqual(actual["queries"][-16:], [q for q in group["queries"]
                         if q.get("expansion_query_class")])
        self.assertFalse(any(q.get("augmented") for q in actual["queries"][-16:]))
        self.assertEqual(group, before)

    def test_every_batch_exposes_all_broad_and_unsupported_with_real_visit_counts(self):
        group = reader.expansion_evidence(expansion.reader_expansion())
        bank, rows, counts = reader.sample_expansion_training(
            group, random.Random(5301), random.Random(5303))
        self.assertEqual(len(rows), 52)
        self.assertEqual(len({q["text"] for q in rows}), 52)
        self.assertEqual(sum(q.get("expansion_query_class") == "broad" for q in rows), 8)
        self.assertEqual(sum(q.get("expansion_query_class") == "unsupported" for q in rows), 8)
        actual = Counter(q["expansion_world_class"] for q in rows)
        for key, value in actual.items():
            self.assertEqual(counts[key], value)
        self.assertEqual(sum(value for key, value in counts.items()
                             if key.startswith("broad_supports_")), 8)
        present = {f["text"] for f in bank}
        for row in rows:
            if row.get("expansion_query_class") == "broad":
                self.assertLessEqual(set(row["support_texts"]), present)
                self.assertEqual(len(row["support_texts"]), sum(f["entity"] in row["entities"]
                                 and f["prop"] == expansion.COEXISTENCE_PROP for f in bank))
            if row.get("augmented"):
                self.assertTrue(set(row["evidence_sources"]) <= present)

    def test_control_preserves_other_properties_and_world_rng_for_multiple_steps(self):
        group = expansion.reader_expansion()
        query_rngs = [random.Random(5301), random.Random(5301)]
        world_rngs = [random.Random(5303), random.Random(5303)]
        for _ in range(6):
            results = [reader.sample_expansion_training(group, query_rngs[i], world_rngs[i],
                       material_max_sources=limit) for i, limit in enumerate((1, 2))]
            banks = [bank for bank, _, _ in results]
            # Adding records changes physical insertion metadata, while every
            # nonmaterial source and its relative order remain the same.
            sources = [[{key: value for key, value in f.items()
                         if key not in ("insert_order", "bank_totalcount")}
                        for f in bank if f["prop"] != expansion.COEXISTENCE_PROP]
                       for bank in banks]
            self.assertEqual(sources[0], sources[1])
            for bank in banks:
                self.assertEqual([f["insert_order"] for f in bank], list(range(len(bank))))
                self.assertTrue(all(f["bank_totalcount"] == len(bank) for f in bank))
            self.assertEqual(world_rngs[0].getstate(), world_rngs[1].getstate())
            self.assertEqual(sum(f["prop"] == expansion.COEXISTENCE_PROP for f in banks[0]), 8)

    def test_semantic_teachers_keep_broad_empty_and_unsupported_distinct(self):
        group = expansion.reader_expansion()
        source_counts = {tuple(f["bank_group"]): 2 for f in group["facts"]
                         if f["prop"] == expansion.COEXISTENCE_PROP}
        world = expansion.sample_expansion_world(group, random.Random(5304), source_counts=source_counts)
        actor = expansion.FIT_ACTORS[0]
        broad = next(q for q in world["queries"] if q.get("expansion_query_class") == "broad"
                     and q["entities"] == [actor])
        unsupported = next(q for q in world["queries"] if q.get("expansion_query_class") == "unsupported"
                           and q["entities"] == [actor])
        empty = deepcopy(broad)
        empty["support_texts"] = []
        facts = world["facts"]
        positive, negative = semantic_relation_masks([broad, empty, unsupported], facts,
                                                   torch.zeros(3, len(facts)))
        material = torch.tensor([f["entity"] == actor and f["prop"] == expansion.COEXISTENCE_PROP
                                 for f in facts])
        self.assertEqual(int((positive[0] & material).sum()), 2)
        self.assertFalse(positive[1:].any())
        self.assertFalse((negative[0] & material).any())
        self.assertFalse((negative[1] & material).any())
        self.assertTrue((negative[2] & material).any())

    def test_diagnostics_use_actual_zero_one_two_world_labels_without_selection(self):
        group = expansion.reader_expansion()
        captured = []
        def evaluate(model, case, cached):
            captured.append(case)
            return dict(queries=len(case["queries"]))
        with patch.object(reader, "evaluate", side_effect=evaluate):
            result = reader.expansion_world_diagnostics(None, group, None)
        self.assertEqual(set(result), {"material_sources_0", "material_sources_1", "material_sources_2"})
        self.assertEqual(result["material_sources_0"]["broad_empty"]["queries"], 8)
        for case in captured:
            self.assertFalse(any(q.get("augmented") for q in case["queries"]))
            self.assertTrue(all(q["prop"] == expansion.COEXISTENCE_PROP for q in case["queries"]))
            present = {f["text"] for f in case["facts"]}
            for row in case["queries"]:
                if row.get("expansion_query_class") == "broad":
                    self.assertTrue(set(row["support_texts"]) <= present)

    def test_parser_exposes_only_supported_world_limits(self):
        parser = reader.reader_argument_parser()
        base = ["--gguf", "model", "--probe", "probe", "--output", "output"]
        self.assertEqual(parser.parse_args(base).expansion_material_max_sources, 2)
        self.assertEqual(parser.parse_args(base + ["--expansion-material-max-sources", "1"]).expansion_material_max_sources, 1)
        with self.assertRaises(ValueError):
            reader.sample_expansion_training(expansion.reader_expansion(), random.Random(0),
                                             random.Random(0), material_max_sources=3)


if __name__ == "__main__":
    unittest.main()
