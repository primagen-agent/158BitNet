"""Structural checks for authored query binding; no model or feature extraction."""

from collections import Counter, defaultdict
from copy import deepcopy
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_query_binding_curriculum import (
    DIALOGUE_FIT_WORLDS,
    QUERY_BINDING_WORLDS,
    query_binding_curriculum,
    query_binding_dialogue_fit_augmentation,
)


class QueryBindingCurriculum(unittest.TestCase):
    def test_sizes_classes_and_write_abstention(self):
        for split, size in (("fit", 120), ("development", 60), ("sealed", 60)):
            rows = query_binding_curriculum(split)
            objects = len(QUERY_BINDING_WORLDS[split]["objects"])
            self.assertEqual(len(rows), size)
            self.assertEqual(len({r["source_id"] for r in rows}), size)
            self.assertEqual(Counter(r["query_binding_class"] for r in rows), {
                "literal-name": 2 * objects, "literal-possessive": 2 * objects,
                "subject-free-topic": 2 * objects, "first-person-header": 4 * objects,
                "first-person-anonymous": 4 * objects,
                "third-person-history": 2 * objects, "third-person-ambiguous": objects,
                "third-person-empty": objects, "third-person-dialogue": 2 * objects,
            })
            self.assertTrue(all(r["spans"] == [] for r in rows))
            self.assertEqual(sum(not r["whole_need"] for r in rows), 6 * objects)
            self.assertEqual(sum(bool(r["whole_refs"]) for r in rows), 12 * objects)
            self.assertEqual(sum(r["whole_need"] and not r["whole_refs"] for r in rows), 2 * objects)
            self.assertTrue(all(r["evaluation_only"] == (split != "fit") for r in rows))
            self.assertTrue(all(r["split"] == r["query_binding_split"] == split for r in rows))

    def test_split_identities_objects_dates_and_wording_are_disjoint(self):
        worlds = QUERY_BINDING_WORLDS
        generated = {split: query_binding_curriculum(split) for split in worlds}
        for a, b in (("fit", "development"), ("fit", "sealed"), ("development", "sealed")):
            names_a = {name for pair in worlds[a]["actors"] for name in pair}
            names_b = {name for pair in worlds[b]["actors"] for name in pair}
            self.assertFalse(names_a & names_b)
            self.assertFalse(set(worlds[a]["objects"]) & set(worlds[b]["objects"]))
            self.assertNotEqual(worlds[a]["timestamp"], worlds[b]["timestamp"])
            self.assertFalse(set(worlds[a]["questions"].values()) & set(worlds[b]["questions"].values()))
            self.assertFalse({r["body"] for r in generated[a]} & {r["body"] for r in generated[b]})
            self.assertFalse({r["source_id"] for r in generated[a]} & {r["source_id"] for r in generated[b]})
            for row in generated[a]:
                for name in names_b:
                    self.assertNotIn(name, row["body"] + row["context"])
                for obj in worlds[b]["objects"]:
                    self.assertNotIn(obj, row["body"] + row["context"])

    def test_same_query_responds_to_subject_changes_and_anonymous_context(self):
        for split in QUERY_BINDING_WORLDS:
            groups = defaultdict(list)
            for row in query_binding_curriculum(split):
                if row["query_binding_family"] in (
                    "first-person-storage", "first-person-care", "third-person",
                ):
                    groups[(row["query_binding_object"], row["body"])].append(row)
            self.assertEqual(len(groups), 3 * len(QUERY_BINDING_WORLDS[split]["objects"]))
            for group in groups.values():
                cases = {r["query_binding_case"]: r for r in group}
                self.assertEqual(set(cases), {"a", "b", "ambiguous", "empty"})
                actors = group[0]["query_binding_actors"]
                self.assertEqual(cases["a"]["whole_refs"], [actors[0]])
                self.assertEqual(cases["b"]["whole_refs"], [actors[1]])
                self.assertNotEqual(cases["a"]["context"], cases["b"]["context"])
                for case in ("ambiguous", "empty"):
                    self.assertTrue(cases[case]["whole_need"])
                    if cases[case]["query_binding_family"] == "third-person":
                        self.assertEqual(cases[case]["whole_refs"], [])
                        self.assertEqual(cases[case]["whole_ref_ranges"], [])
                    else:
                        self.assertEqual(cases[case]["whole_refs"], ["User"])
                        self.assertEqual(cases[case]["whole_ref_ranges"], [
                            dict(document="context", start=9, end=13)])
                self.assertEqual(cases["empty"]["past"], "")
                self.assertTrue(all(name in cases["ambiguous"]["past"] for name in actors))

    def test_literal_subjects_and_topic_controls_ignore_wrong_context(self):
        for split in QUERY_BINDING_WORLDS:
            for row in query_binding_curriculum(split):
                family = row["query_binding_family"]
                if family not in ("literal-name", "literal-possessive", "topic"):
                    continue
                self.assertFalse(row["whole_need"])
                self.assertEqual(row["whole_refs"], [])
                self.assertEqual(row["whole_ref_ranges"], [])
                actor_index = 0 if row["query_binding_case"] == "a" else 1
                actor, other = row["query_binding_actors"][actor_index], row["query_binding_actors"][1 - actor_index]
                self.assertEqual(row["speaker"], other)
                self.assertIn(other, row["past"])
                if family != "topic":
                    self.assertIn(actor, row["body"])
                    self.assertNotIn(actor, row["context"])
                else:
                    self.assertTrue(all(name not in row["body"] for name in row["query_binding_actors"]))

    def test_reference_ranges_are_exact_utf8_whole_names(self):
        unicode_offset_seen = False
        for split in QUERY_BINDING_WORLDS:
            for row in query_binding_curriculum(split):
                expected = f"Speaker: {row['speaker']}\nObserved at: {row['timestamp']}\n{row['past']}"
                self.assertEqual(row["context"], expected)
                self.assertEqual(len(row["whole_refs"]), len(row["whole_ref_ranges"]))
                for name, ref in zip(row["whole_refs"], row["whole_ref_ranges"]):
                    self.assertTrue(row["whole_need"])
                    self.assertEqual(ref["document"], "context")
                    encoded = row["context"].encode("utf-8")
                    self.assertEqual(encoded[ref["start"]:ref["end"]].decode("utf-8"), name)
                    before = encoded[:ref["start"]].decode("utf-8")
                    after = encoded[ref["end"]:].decode("utf-8")
                    self.assertTrue(not before or not before[-1].isalnum())
                    self.assertTrue(not after or not after[0].isalnum())
                    unicode_offset_seen |= len(before.encode()) != len(before)
                    if row["query_binding_class"] in ("first-person-header", "first-person-anonymous"):
                        self.assertEqual(name, row["speaker"])
                        self.assertEqual(ref["start"], 9)
                    else:
                        self.assertNotEqual(name, row["speaker"])
                        self.assertGreater(ref["start"], len(expected[:-len(row["past"])].encode()))
        self.assertTrue(unicode_offset_seen)

    def test_anonymous_first_person_matches_existing_writer_labels(self):
        # Read the existing authored label, without importing its Torch-backed
        # feature module or extracting any model features.
        path = Path(__file__).resolve().parents[1] / "python" / "prepare_neural_dialogue.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "add" and node.args
                 and isinstance(node.args[-1], ast.Constant)
                 and node.args[-1].value == "anonymous-self"]
        self.assertEqual(len(calls), 1)
        existing_need = ast.literal_eval(calls[0].args[-3])
        existing_refs = ast.literal_eval(calls[0].args[-2])
        self.assertTrue(existing_need)
        self.assertEqual(existing_refs, ["User"])
        for split in QUERY_BINDING_WORLDS:
            anonymous = [row for row in query_binding_curriculum(split)
                         if row["query_binding_class"] == "first-person-anonymous"]
            self.assertEqual(len(anonymous), 4 * len(QUERY_BINDING_WORLDS[split]["objects"]))
            for row in anonymous:
                self.assertEqual(row["whole_need"], existing_need)
                self.assertEqual(row["whole_refs"], existing_refs)
                self.assertEqual(row["whole_ref_ranges"], [dict(document="context", start=9, end=13)])
                self.assertTrue(all(name not in row["whole_refs"] for name in row["query_binding_actors"]))

    def test_actor_and_object_swaps_preserve_the_same_binding_task(self):
        for split in QUERY_BINDING_WORLDS:
            grouped = defaultdict(dict)
            for row in query_binding_curriculum(split):
                grouped[(row["query_binding_object"], row["query_binding_family"])][row["query_binding_case"]] = row
            for (obj, family), cases in grouped.items():
                a, b = cases["a"], cases["b"]
                self.assertEqual(a["whole_need"], b["whole_need"])
                self.assertEqual(a["query_binding_class"], b["query_binding_class"])
                if a["whole_need"]:
                    self.assertEqual(a["body"], b["body"])
                    self.assertNotEqual(a["whole_refs"], b["whole_refs"])
                elif family != "topic":
                    self.assertEqual(a["body"].replace(a["query_binding_actors"][0], "ACTOR"),
                                     b["body"].replace(b["query_binding_actors"][1], "ACTOR"))
                other_objects = [o for o in QUERY_BINDING_WORLDS[split]["objects"] if o != obj]
                for other_obj in other_objects:
                    other = grouped[(other_obj, family)]
                    for case, row in cases.items():
                        alternate = other[case]
                        self.assertEqual(row["whole_need"], alternate["whole_need"])
                        self.assertEqual(row["query_binding_class"], alternate["query_binding_class"])
                        expected = row["body"].replace(obj, "OBJECT")
                        actual = alternate["body"].replace(other_obj, "OBJECT")
                        for name in row["query_binding_actors"]:
                            expected = expected.replace(name, "ACTOR")
                        for name in alternate["query_binding_actors"]:
                            actual = actual.replace(name, "ACTOR")
                        # Pronoun gender changes with actor pairs, never labels.
                        expected = expected.replace("does he ", "does PERSON ").replace("does she ", "does PERSON ")
                        actual = actual.replace("does he ", "does PERSON ").replace("does she ", "does PERSON ")
                        self.assertEqual(expected, actual)

    def test_generation_is_deterministic_fresh_and_reads_no_external_data(self):
        original = {s: query_binding_curriculum(s) for s in QUERY_BINDING_WORLDS}
        before = deepcopy(original)
        with patch("builtins.open", side_effect=AssertionError("external data read")), \
                patch.object(Path, "read_text", side_effect=AssertionError("external text read")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("external bytes read")):
            fresh = {s: query_binding_curriculum(s) for s in QUERY_BINDING_WORLDS}
        self.assertEqual(fresh, original)
        fresh["fit"][0]["query_binding_actors"].append("Changed")
        row = next(r for r in fresh["fit"] if r["whole_refs"])
        row["whole_refs"].append("Changed")
        row["whole_ref_ranges"][0]["end"] += 1
        self.assertEqual(original, before)
        self.assertEqual(query_binding_curriculum("fit"), original["fit"])

    def test_dialogue_fit_augmentation_binds_the_addressed_name(self):
        rows = query_binding_dialogue_fit_augmentation()
        objects = sum(len(world["objects"]) for world in DIALOGUE_FIT_WORLDS)
        self.assertEqual(len(rows), 3 * objects)
        self.assertEqual(len({r["source_id"] for r in rows}), len(rows))
        worlds = QUERY_BINDING_WORLDS
        dialogue_names = [name for world in DIALOGUE_FIT_WORLDS for name in world["actors"]]
        dialogue_objects = [obj for world in DIALOGUE_FIT_WORLDS for obj in world["objects"]]
        dialogue_dates = [world["timestamp"] for world in DIALOGUE_FIT_WORLDS]
        self.assertEqual(len(set(dialogue_names)), len(dialogue_names))
        self.assertEqual(len(set(dialogue_objects)), len(dialogue_objects))
        self.assertEqual(len(set(dialogue_dates)), len(dialogue_dates))
        for split in worlds:
            names = {name for pair in worlds[split]["actors"] for name in pair}
            self.assertFalse(names & set(dialogue_names))
            self.assertFalse(set(worlds[split]["objects"]) & set(dialogue_objects))
            self.assertNotIn(worlds[split]["timestamp"], dialogue_dates)
        for row in rows:
            self.assertEqual(row["split"], row["query_binding_split"])
            self.assertEqual(row["split"], "fit")
            self.assertFalse(row["evaluation_only"])
            self.assertEqual(row["evaluation_role"], "fit-teacher")
            self.assertEqual(row["query_binding_family"], "third-person-dialogue")
            self.assertEqual(row["query_binding_class"], "third-person-dialogue")
            self.assertTrue(row["whole_need"])
            self.assertEqual(len(row["whole_refs"]), 1)
            reference = row["whole_refs"][0]
            self.assertIn(reference, row["query_binding_actors"])
            other = next(name for name in row["query_binding_actors"] if name != reference)
            self.assertIn(row["speaker"], ("User", other))
            self.assertEqual(row["context"],
                             f"Speaker: {row['speaker']}\nObserved at: {row['timestamp']}\n{row['past']}")
            ref = row["whole_ref_ranges"][0]
            self.assertEqual(ref["document"], "context")
            encoded = row["context"].encode("utf-8")
            self.assertEqual(encoded[ref["start"]:ref["end"]].decode("utf-8"), reference)
            assistant_start = len(row["context"][:row["context"].index("assistant:")].encode())
            self.assertGreater(ref["start"], assistant_start)
            self.assertNotIn(reference, row["body"])
            for name in row["query_binding_actors"]:
                if name != reference:
                    self.assertNotIn(name, row["body"])
            for split in ("development", "sealed"):
                for name in {n for pair in worlds[split]["actors"] for n in pair}:
                    self.assertNotIn(name, row["body"] + row["context"])
                for obj in worlds[split]["objects"]:
                    self.assertNotIn(obj, row["body"] + row["context"])
        self.assertEqual(rows, query_binding_dialogue_fit_augmentation())

    def test_invalid_split_rejects(self):
        for split in (None, 0, [], "test", "FIT", ""):
            with self.assertRaisesRegex(ValueError, "fit, development or sealed"):
                query_binding_curriculum(split)


if __name__ == "__main__":
    unittest.main()
