"""Authored multi-source worlds, complete broad labels and immutable evidence."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from prepare_neural_reader_expansion import (
    FIT_ACTORS, PATTERNS, NULL_QUESTIONS, COEXISTENCE_PROP,
    BROAD_MATERIAL_QUESTION, UNSUPPORTED_MATERIAL_QUESTION,
    reader_expansion, sample_expansion_world, validate,
)


class ExpansionWorldContract(unittest.TestCase):
    def setUp(self):
        self.group = reader_expansion()

    def fixed_counts(self, count):
        return {(actor, COEXISTENCE_PROP): count for actor in FIT_ACTORS}

    def test_original_sources_specific_questions_and_nulls_are_unchanged(self):
        expected_sources, expected_queries = [], []
        for name in FIT_ACTORS:
            for i, (prop, content, questions, kind) in enumerate(PATTERNS):
                text = name + ". " + content
                expected_sources.append(dict(text=text, entity=name, reference=name,
                    prop=prop, kind=kind, episode=True, expansion=True,
                    bank_group=[name, prop], pattern_id="reader-expansion-" + str(i)))
                expected_queries.extend(dict(text=q.format(name=name), entities=[name],
                    prop=prop, kind=kind, support_texts=[text], semantic_specific=True,
                    bank_group=[name, prop], pattern_id="reader-expansion-" + str(i), expansion=True)
                    for q in questions)
            expected_queries.extend(dict(text=q.format(name=name), entities=[name],
                prop="", kind=0, support_texts=[], expansion=True) for q in NULL_QUESTIONS)
        self.assertEqual(self.group["facts"], expected_sources)
        old = [row for row in self.group["queries"] if "expansion_query_class" not in row]
        self.assertEqual(old, expected_queries)
        additions = [row for row in self.group["queries"] if "expansion_query_class" in row]
        self.assertEqual(len(additions), 16)
        self.assertEqual(len({row["text"] for row in self.group["queries"]}), 1520)
        self.assertTrue(all("reference" not in row and "observation" not in row
                            for row in self.group["queries"]))

    def test_material_scope_has_explicit_objects_and_homogeneous_original_kind(self):
        by_text = {fact["text"]: fact for fact in self.group["facts"]}
        for row in self.group["queries"]:
            if row.get("expansion_query_class") != "broad": continue
            self.assertEqual(row["text"], BROAD_MATERIAL_QUESTION.format(name=row["entities"][0]))
            self.assertEqual(row["support_texts"], row["broad_scope_source_texts"])
            self.assertEqual(len(row["support_texts"]), 5)
            self.assertIs(row["semantic_specific"], False)
            for text in row["support_texts"]:
                self.assertEqual(by_text[text]["kind"], 0)
                self.assertEqual(by_text[text]["prop"], COEXISTENCE_PROP)
                self.assertEqual(by_text[text]["entity"], row["entities"][0])
            for object_name in ("toy bridges", "painted signs", "notebook binding",
                                "model railway scenery", "shopping bags"):
                self.assertIn(object_name, row["text"])
        self.assertFalse(any("garden hammock" in fact["text"] for fact in self.group["facts"]))

    def test_zero_one_two_sources_bind_complete_broad_targets_and_counts(self):
        for count in (0, 1, 2):
            world = sample_expansion_world(self.group, random.Random(7313), source_counts=self.fixed_counts(count))
            present = {fact["text"] for fact in world["facts"]}
            self.assertEqual(len(present), 88 + 8 * count)
            classes = Counter(row["expansion_world_class"] for row in world["queries"])
            self.assertEqual(classes["source_present"], (88 + 8 * count) * 3)
            self.assertEqual(classes["source_absent"], 1440 - (88 + 8 * count) * 3)
            self.assertEqual(classes["unsupported"], 72)
            self.assertEqual(classes["broad_present"], 8 if count else 0)
            self.assertEqual(classes["broad_empty"], 0 if count else 8)
            for row in world["queries"]:
                if row.get("expansion_query_class") == "broad":
                    expected = [text for text in row["broad_scope_source_texts"] if text in present]
                    self.assertEqual(row["support_texts"], expected)
                    self.assertEqual(len(expected), count)
                    self.assertTrue(row["prop"])
                elif row.get("expansion_query_class") == "unsupported":
                    self.assertEqual(row["support_texts"], [])
                    self.assertEqual(row["prop"], COEXISTENCE_PROP)
                    self.assertIs(row["semantic_specific"], True)

    def test_specific_a_b_broad_and_unsupported_share_the_same_actual_bank(self):
        world = sample_expansion_world(self.group, random.Random(9009), source_counts=self.fixed_counts(2))
        sources = [fact for fact in world["facts"] if fact["entity"] == "Cyra" and fact["prop"] == COEXISTENCE_PROP]
        rows = [row for row in world["queries"] if row["entities"] == ["Cyra"] and row["prop"] == COEXISTENCE_PROP]
        for source in sources:
            specific = [row for row in rows if row.get("semantic_specific") and row["support_texts"] == [source["text"]]]
            self.assertEqual(len(specific), 3)
            self.assertTrue(all(row["expansion_world_class"] == "source_present" for row in specific))
            other = next(fact for fact in sources if fact is not source)
            self.assertTrue(all(other["text"] not in row["support_texts"] for row in specific))
        broad = next(row for row in rows if row.get("expansion_query_class") == "broad")
        self.assertEqual(set(broad["support_texts"]), {fact["text"] for fact in sources})
        unsupported = next(row for row in rows if row.get("expansion_query_class") == "unsupported")
        self.assertEqual(unsupported["text"], UNSUPPORTED_MATERIAL_QUESTION.format(name="Cyra"))
        self.assertEqual(unsupported["support_texts"], [])
        absent = [row for row in rows if row["expansion_world_class"] == "source_absent"]
        self.assertEqual(len(absent), 9)
        self.assertTrue(all(len(row["support_texts"]) == 1 for row in absent))

    def test_single_source_control_consumes_identical_rng_and_other_worlds(self):
        primary_rng, control_rng = random.Random(5303), random.Random(5303)
        counts = set()
        for _ in range(12):
            primary = sample_expansion_world(self.group, primary_rng)
            control = sample_expansion_world(self.group, control_rng, material_counts=(1,))
            self.assertEqual(primary_rng.getstate(), control_rng.getstate())
            self.assertEqual([f for f in primary["facts"] if f["prop"] != COEXISTENCE_PROP],
                             [f for f in control["facts"] if f["prop"] != COEXISTENCE_PROP])
            for name in FIT_ACTORS:
                a = [f for f in primary["facts"] if f["entity"] == name and f["prop"] == COEXISTENCE_PROP]
                b = [f for f in control["facts"] if f["entity"] == name and f["prop"] == COEXISTENCE_PROP]
                counts.add(len(a)); self.assertEqual(len(b), 1)
                if a: self.assertEqual(a[0], b[0])
        self.assertEqual(counts, {0, 1, 2})

    def test_no_mutation_reproducibility_and_nonmaterial_groups_never_coexist(self):
        before = deepcopy(self.group)
        first = sample_expansion_world(self.group, random.Random(8731))
        self.assertEqual(first, sample_expansion_world(self.group, random.Random(8731)))
        self.assertEqual(self.group, before)
        counts = Counter(tuple(fact["bank_group"]) for fact in first["facts"])
        for key, count in counts.items():
            self.assertEqual(count, 1 if key[1] != COEXISTENCE_PROP else count)
        first["facts"][0]["text"] = "Changed output"
        first["queries"][0]["support_texts"].clear()
        self.assertEqual(self.group, before)

    def test_augmented_broad_keeps_its_real_cues_and_binds_all_present_sources(self):
        seed = 6219
        counts = self.fixed_counts(2)
        original = sample_expansion_world(self.group, random.Random(seed), source_counts=counts)
        source = next(f for f in original["facts"] if f["entity"] == "Cyra" and f["prop"] == COEXISTENCE_PROP)
        noise = next(f for f in original["facts"] if f["entity"] == "Cyra" and f["prop"] != COEXISTENCE_PROP)
        broad = next(row for row in self.group["queries"] if row["entities"] == ["Cyra"] and row.get("expansion_query_class") == "broad")
        augmented = deepcopy(broad)
        cues = [noise["text"], source["text"]]
        augmented.update(initial_text=broad["text"], augmented=True,
                         evidence_sources=cues, evidence_count=2,
                         text=broad["text"] + "\nActivated evidence:\n" + "\n".join(cues) + "\n")
        group = deepcopy(self.group); group["queries"].append(augmented)
        before = deepcopy(group)
        world = sample_expansion_world(group, random.Random(seed), source_counts=counts)
        retained = next(row for row in world["queries"] if row["text"] == augmented["text"])
        raw = next(row for row in world["queries"] if row["text"] == broad["text"])
        self.assertEqual(retained["support_texts"], raw["support_texts"])
        self.assertEqual(len(retained["support_texts"]), 2)
        for field in ("text", "initial_text", "evidence_sources", "evidence_count"):
            self.assertEqual(retained[field], augmented[field])
        self.assertTrue(set(retained["evidence_sources"]) <= {f["text"] for f in world["facts"]})
        empty = sample_expansion_world(group, random.Random(seed), source_counts=self.fixed_counts(0))
        self.assertFalse(any(row["text"] == augmented["text"] for row in empty["queries"]))
        self.assertEqual(group, before)

    def test_reordered_raw_queries_accept_old_cues_then_new_raw_questions(self):
        group = deepcopy(self.group)
        group["queries"] = [q for q in group["queries"] if "expansion_query_class" not in q] + [
            q for q in group["queries"] if "expansion_query_class" in q]
        world = sample_expansion_world(group, random.Random(982))
        self.assertEqual(len(world["queries"]), 1520)

    def test_foreign_scope_kind_counts_and_phantom_evidence_are_rejected(self):
        for changes in ({"source_counts": {("Cyra", "work_change"): 2}},
                        {"source_counts": {("Cyra", COEXISTENCE_PROP): True}},
                        {"source_counts": {("Cyra", COEXISTENCE_PROP): 3}},
                        {"material_counts": (0, 2)}, {"material_counts": [1]}):
            with self.assertRaises(ValueError): sample_expansion_world(self.group, random.Random(3), **changes)
        for field, value in (("kind", 1), ("support_texts", []), ("semantic_specific", True),
                             ("broad_scope_source_texts", [self.group["facts"][0]["text"]])):
            group = deepcopy(self.group)
            broad = next(row for row in group["queries"] if row.get("expansion_query_class") == "broad")
            broad[field] = value
            with self.assertRaises(ValueError): validate(group)
        group = deepcopy(self.group)
        broad = deepcopy(next(row for row in group["queries"] if row.get("expansion_query_class") == "broad"))
        broad.update(augmented=True, initial_text=broad["text"], evidence_sources=["Unknown. A phantom cue."],
                     evidence_count=1, text=broad["text"] + "\nActivated evidence:\nUnknown. A phantom cue.\n")
        group["queries"].append(broad)
        with self.assertRaisesRegex(ValueError, "outside the authored expansion"):
            sample_expansion_world(group, random.Random(5))


if __name__ == "__main__":
    unittest.main()
