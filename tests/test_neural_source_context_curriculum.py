"""Fit-only history counterfactuals preserve exact source and reference labels."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import observation_context
from neural_writer_curriculum import (
    boundary_generalization_curriculum,
    narrative_curriculum,
    with_source_context_generalization,
)


def source_rows():
    base = deepcopy(boundary_generalization_curriculum("fit")[0])
    base.pop("boundary_world")
    base["source_id"] = "unmarked-base-fit"
    base["body"] = "Unmarked base message; do not generate context views for it."
    base.update(spans=[], whole_need=False, whole_refs=[], whole_ref_ranges=[])
    return dict(
        fit=[base] + boundary_generalization_curriculum("fit") + narrative_curriculum("fit"),
        development=boundary_generalization_curriculum("development")[:1],
        boundary_development=boundary_generalization_curriculum("development")[1:],
        narrative_development=narrative_curriculum("development"),
    )


class SourceContextCurriculum(unittest.TestCase):
    def test_only_authored_fit_worlds_add_views_and_development_bytes_stay_equal(self):
        rows = source_rows()
        before = {key: json.dumps(value, ensure_ascii=False).encode()
                  for key, value in rows.items()}
        result = with_source_context_generalization(rows)
        added = result["fit"][len(rows["fit"]):]
        self.assertEqual(len(added), 216)
        self.assertTrue(all(row.get("boundary_world") == "fit"
                            or row.get("narrative_world") == "fit" for row in added))
        self.assertFalse(any(row["source_id"].startswith("unmarked-base-fit::") for row in added))
        self.assertFalse(any(row["body"] == rows["fit"][0]["body"] for row in added))
        for key in rows:
            self.assertEqual(json.dumps(rows[key], ensure_ascii=False).encode(), before[key])
            if key != "fit":
                self.assertEqual(json.dumps(result[key], ensure_ascii=False).encode(), before[key])
        self.assertEqual(result, with_source_context_generalization(rows))

    def test_exact_source_actions_and_header_reference_labels_survive(self):
        rows = source_rows()
        parents = {row["source_id"]: row for row in rows["fit"]}
        result = with_source_context_generalization(rows)
        added = result["fit"][len(rows["fit"]):]
        for row in added:
            source_id, view = row["source_id"].rsplit("::source-context-", 1)
            parent = parents[source_id]
            for key in parent.keys() - {"past", "context", "source_id"}:
                self.assertEqual(row[key], parent[key])
            self.assertEqual(row.keys(), parent.keys())
            self.assertIn(view, ("empty", "neutral"))
            self.assertEqual(row["context"], observation_context(
                row["speaker"], row["timestamp"], row["past"]))
            for span in row["spans"]:
                a, b = span["start"], span["end"]
                self.assertEqual(row["body"].encode()[a:b], parent["body"].encode()[a:b])
                for ref in span.get("ref_ranges", []):
                    if ref["document"] == "context":
                        self.assertEqual(row["context"].encode()[ref["start"]:ref["end"]],
                                         row["speaker"].encode())
        self.assertTrue(any(row["tag"] == "writer-narrative-questions" for row in added))
        self.assertTrue(any(row["tag"] == "writer-narrative-fiction" for row in added))
        self.assertTrue(all(not row["spans"] for row in added
                            if row["tag"] in ("writer-narrative-questions", "writer-narrative-fiction")))

    def test_augmented_rows_and_nested_labels_do_not_alias_inputs_or_other_views(self):
        rows = source_rows()
        before = deepcopy(rows)
        result = with_source_context_generalization(rows)
        added = result["fit"][len(rows["fit"]):]
        first = added[0]
        parent_id = first["source_id"].rsplit("::", 1)[0]
        sibling = next(row for row in added
                       if row["source_id"].startswith(parent_id + "::") and row is not first)
        self.assertIsNot(first["spans"], sibling["spans"])
        first["spans"][0]["ref_ranges"][0]["end"] += 1
        first["whole_refs"].append("changed")
        self.assertEqual(rows, before)
        self.assertEqual(sibling["spans"], next(row["spans"] for row in rows["fit"]
                                              if row["source_id"] == parent_id))
        result["development"][0]["spans"][0]["start"] += 1
        self.assertEqual(rows, before)

    def test_existing_empty_or_neutral_context_is_skipped(self):
        row = deepcopy(boundary_generalization_curriculum("fit")[0])
        for past, expected_suffix in (("", "neutral"),
                                     ("user: Dorian: Hello there.\nassistant: Please continue.\n", "empty")):
            row.update(past=past, context=observation_context(row["speaker"], row["timestamp"], past))
            result = with_source_context_generalization(dict(fit=[row], development=[]))
            self.assertEqual(len(result["fit"]), 2)
            self.assertTrue(result["fit"][1]["source_id"].endswith("::source-context-" + expected_suffix))

    def test_unicode_speaker_uses_exact_header_byte_range(self):
        row = deepcopy(boundary_generalization_curriculum("fit")[0])
        row["speaker"] = "Éda"
        row["context"] = observation_context(row["speaker"], row["timestamp"], row["past"])
        ref = dict(document="context", start=9, end=9 + len(row["speaker"].encode()))
        row["spans"][0].update(references=[row["speaker"]], ref_ranges=[dict(ref)])
        row.update(whole_refs=[row["speaker"]], whole_ref_ranges=[dict(ref)])
        result = with_source_context_generalization(dict(fit=[row], development=[]))
        for clone in result["fit"][1:]:
            self.assertEqual(clone["context"].encode()[9:ref["end"]], "Éda".encode())
            self.assertEqual(clone["spans"][0]["ref_ranges"], [ref])

    def test_body_references_remain_literal_and_unchanged(self):
        row = deepcopy(boundary_generalization_curriculum("fit")[0])
        row["spans"][0]["ref_ranges"] = [dict(document="body", start=0, end=1)]
        result = with_source_context_generalization(dict(fit=[row], development=[]))
        for clone in result["fit"]:
            self.assertEqual(clone["spans"][0]["ref_ranges"], row["spans"][0]["ref_ranges"])

    def test_past_or_partial_header_references_are_rejected_before_transform(self):
        original = boundary_generalization_curriculum("fit")[0]
        for location in ("spans", "whole"):
            for ref in (dict(document="context", start=10, end=12),
                        dict(document="context", start=len(original["context"].encode()) - 5,
                             end=len(original["context"].encode())),
                        dict(document="past", start=0, end=1)):
                row = deepcopy(original)
                if location == "spans":
                    row["spans"][0]["ref_ranges"] = [ref]
                else:
                    row["whole_ref_ranges"] = [ref]
                before = deepcopy(row)
                with self.assertRaises(ValueError):
                    with_source_context_generalization(dict(fit=[row], development=[]))
                self.assertEqual(row, before)

    def test_repeated_attach_or_duplicate_source_ids_are_rejected(self):
        rows = source_rows()
        result = with_source_context_generalization(rows)
        ids = [row["source_id"] for group in result.values() for row in group]
        self.assertEqual(len(ids), len(set(ids)))
        with self.assertRaisesRegex(ValueError, "already attached"):
            with_source_context_generalization(result)
        row = boundary_generalization_curriculum("fit")[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            with_source_context_generalization(dict(fit=[row, deepcopy(row)], development=[]))

    def test_no_legal_fit_world_or_inconsistent_provenance_is_rejected(self):
        row = deepcopy(boundary_generalization_curriculum("fit")[0])
        row["boundary_world"] = "development"
        with self.assertRaisesRegex(ValueError, "authored fit world"):
            with_source_context_generalization(dict(fit=[row], development=[]))
        row["boundary_world"] = "fit"
        row["context"] += "Undeclared history.\n"
        with self.assertRaisesRegex(ValueError, "literal provenance"):
            with_source_context_generalization(dict(fit=[row], development=[]))

    def test_new_context_count_is_bounded_and_cli_flag_is_exposed(self):
        rows = source_rows()
        old = {row["context"] for group in rows.values() for row in group}
        result = with_source_context_generalization(rows)
        self.assertEqual(len({row["context"] for row in result["fit"]} - old), 12)
        command = [sys.executable, str(Path(__file__).resolve().parents[1]
                                     / "python/train_neural_dialogue.py"), "--help"]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertIn("--source-context-generalization", completed.stdout)


if __name__ == "__main__":
    unittest.main()
