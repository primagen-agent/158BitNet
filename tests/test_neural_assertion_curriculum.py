"""Independent assertion teachers preserve fit isolation and literal spans."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import observation_context, word_features, decode_plan
from train_neural_dialogue import plan_targets
from neural_writer_curriculum import (
    assertion_generalization_fit_rows,
    boundary_generalization_curriculum,
    narrative_curriculum,
    with_assertion_generalization,
)


class AssertionCurriculum(unittest.TestCase):
    def test_independent_facts_have_asserted_and_nonasserted_contexts(self):
        rows = assertion_generalization_fit_rows()
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({row["source_id"] for row in rows}), 64)
        groups = defaultdict(list)
        for row in rows:
            groups[row["assertion_fact_index"]].append(row)
            self.assertTrue(row["assertion_generalization"])
            self.assertEqual(row["assertion_world"], "fit")
            self.assertEqual(row["evaluation_role"], "fit-teacher")
        self.assertEqual(len(groups), 16)
        kinds = Counter(row["assertion_kind"] for row in rows)
        self.assertEqual(kinds, {
            "real-quote": 16, "passive-report": 16, "question-premise": 16,
            "fiction": 4, "conditional": 4, "pure-question": 4, "courtesy": 4,
        })
        for group in groups.values():
            self.assertEqual(sum(bool(row["spans"]) for row in group), 3)
            sources = {}
            for row in group:
                if row["spans"]:
                    span = row["spans"][0]
                    sources[row["assertion_kind"]] = row["body"].encode()[
                        span["start"]:span["end"]].decode()
            self.assertEqual(sources["real-quote"], sources["passive-report"])
            self.assertEqual(sources["question-premise"],
                             sources["real-quote"][0].lower() + sources["real-quote"][1:-1] + ",")
            control = next(row for row in group if not row["spans"])
            self.assertIn(sources["question-premise"][:-1].casefold(), control["body"].casefold())
        self.assertTrue(any('"' in row["body"] and row["spans"] for row in rows))
        self.assertTrue(any('"' in row["body"] and not row["spans"] for row in rows))
        self.assertTrue(any("?" in row["body"] and row["spans"] for row in rows))
        self.assertTrue(any("?" in row["body"] and not row["spans"] for row in rows))

    def test_speaker_and_history_do_not_predict_assertion_or_control_kind(self):
        rows = assertion_generalization_fit_rows()
        by_speaker = defaultdict(Counter)
        by_kind = defaultdict(Counter)
        for row in rows:
            by_speaker[row["speaker"]][row["assertion_kind"]] += 1
            by_kind[row["assertion_kind"]][bool(row["past"])] += 1
        for counts in by_speaker.values():
            self.assertEqual(counts["real-quote"], 4)
            self.assertEqual(counts["passive-report"], 4)
            self.assertEqual(counts["question-premise"], 4)
            for kind in ("fiction", "conditional", "pure-question", "courtesy"):
                self.assertEqual(counts[kind], 1)
        for counts in by_kind.values():
            self.assertEqual(counts[True], counts[False])

    def test_source_and_subject_ranges_are_exact_original_utf8_bytes(self):
        rows = assertion_generalization_fit_rows()
        unicode_prefix_seen = False
        for row in rows:
            self.assertEqual(row["context"], observation_context(
                row["speaker"], row["timestamp"], row["past"]))
            for span in row["spans"]:
                self.assertEqual(span["action"], 1)
                self.assertTrue(span["need"])
                self.assertEqual(span["references"], [row["speaker"]])
                raw = row["body"].encode()
                self.assertLess(span["start"], span["end"])
                self.assertLessEqual(span["end"], len(raw))
                source = raw[span["start"]:span["end"]].decode()
                self.assertLessEqual(len(source.encode()), 160)
                self.assertFalse(source.startswith(("I ", "My ")))
                before = raw[:span["start"]].decode()
                unicode_prefix_seen |= len(before) != len(before.encode())
                self.assertEqual(source, row["body"][len(before):len(before) + len(source)])
                for ref in span["ref_ranges"]:
                    self.assertEqual(ref["document"], "context")
                    self.assertEqual(row["context"].encode()[ref["start"]:ref["end"]],
                                     row["speaker"].encode())
            for ref in row["whole_ref_ranges"]:
                self.assertEqual(row["context"].encode()[ref["start"]:ref["end"]],
                                 row["speaker"].encode())
        self.assertTrue(unicode_prefix_seen)

    def test_gold_spans_are_representable_by_current_structural_source_units(self):
        for row in assertion_generalization_fit_rows():
            # A constant synthetic feature row tests only UTF-8 source units,
            # never tokenizer/backbone extraction or model predictions.
            item = dict(x=torch.zeros((1, 1)), bounds=[(0, len(row["body"].encode()))])
            _, units = word_features(row["body"], item)
            target = plan_targets(row, units, "cpu")
            spans = [(a, b, action) for a, b, action, _, _ in decode_plan(target, units)]
            self.assertEqual(spans, [(span["start"], span["end"], span["action"])
                                    for span in row["spans"]])

    def test_queries_can_require_a_subject_without_authorizing_a_write(self):
        for row in assertion_generalization_fit_rows():
            if row["assertion_kind"] in ("conditional", "pure-question"):
                self.assertEqual(row["spans"], [])
                self.assertTrue(row["whole_need"])
                self.assertEqual(row["whole_refs"], [row["speaker"]])
            elif not row["spans"]:
                self.assertFalse(row["whole_need"])
                self.assertEqual(row["whole_refs"], [])
                self.assertEqual(row["whole_ref_ranges"], [])

    def test_attach_preserves_existing_fit_and_every_development_group(self):
        rows = dict(
            fit=boundary_generalization_curriculum("fit")[:2],
            development=boundary_generalization_curriculum("development")[:1],
            boundary_development=boundary_generalization_curriculum("development")[1:],
            narrative_development=narrative_curriculum("development"),
        )
        before = {key: json.dumps(group, ensure_ascii=False).encode()
                  for key, group in rows.items()}
        result = with_assertion_generalization(rows)
        self.assertEqual(len(result["fit"]), len(rows["fit"]) + 64)
        self.assertEqual(result["fit"][:len(rows["fit"])], rows["fit"])
        for key in rows:
            self.assertEqual(json.dumps(rows[key], ensure_ascii=False).encode(), before[key])
            if key != "fit":
                self.assertEqual(json.dumps(result[key], ensure_ascii=False).encode(), before[key])
        result["fit"][0]["spans"][0]["references"].append("changed")
        result["development"][0]["spans"][0]["start"] += 1
        self.assertEqual(json.dumps(rows["fit"], ensure_ascii=False).encode(), before["fit"])
        self.assertEqual(json.dumps(rows["development"], ensure_ascii=False).encode(),
                         before["development"])

    def test_generated_labels_and_calls_do_not_alias_each_other(self):
        rows = assertion_generalization_fit_rows()
        again = assertion_generalization_fit_rows()
        before = deepcopy(again)
        rows[0]["spans"][0]["ref_ranges"][0]["start"] += 1
        rows[0]["whole_refs"].append("changed")
        rows[0]["whole_ref_ranges"][0]["end"] += 1
        self.assertEqual(again, before)
        self.assertEqual(rows[1], before[1])
        self.assertEqual(rows[2], before[2])
        self.assertEqual(rows[0]["spans"][0]["ref_ranges"][0]["end"],
                         before[0]["spans"][0]["ref_ranges"][0]["end"])

    def test_duplicate_attachment_or_ids_reject_without_mutating_input(self):
        base = dict(fit=boundary_generalization_curriculum("fit")[:1], development=[])
        attached = with_assertion_generalization(base)
        before = deepcopy(attached)
        with self.assertRaisesRegex(ValueError, "already attached"):
            with_assertion_generalization(attached)
        self.assertEqual(attached, before)
        duplicate = dict(fit=[deepcopy(base["fit"][0])] * 2, development=[])
        with self.assertRaisesRegex(ValueError, "unique"):
            with_assertion_generalization(duplicate)
        collision = deepcopy(base)
        collision["development"] = [deepcopy(assertion_generalization_fit_rows()[0])]
        before = deepcopy(collision)
        with self.assertRaisesRegex(ValueError, "already exists"):
            with_assertion_generalization(collision)
        self.assertEqual(collision, before)

    def test_authoring_does_not_read_training_or_evaluation_files(self):
        with patch("builtins.open", side_effect=AssertionError("external data read")), \
             patch.object(Path, "read_text", side_effect=AssertionError("external text read")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("external bytes read")):
            rows = assertion_generalization_fit_rows()
            attached = with_assertion_generalization(dict(fit=[], development=[]))
        self.assertEqual(rows, attached["fit"])


if __name__ == "__main__":
    unittest.main()
