"""Native scoring separates strict write availability from read component errors."""

from pathlib import Path
from copy import deepcopy
import json
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_neural_episodes_native import (
    curriculum_specification, pack, score, write_observations,
)


FACT = dict(
    content="I repair clocks.",
    reference="Cyra",
    text="Cyra. I repair clocks.",
    entity="Cyra",
    prop="job",
    kind=0,
)
SECOND_FACT = dict(
    content="I restore radios.",
    reference="Cyra",
    text="Cyra. I restore radios.",
    entity="Cyra",
    prop="job",
    kind=0,
)
SLOT = dict(id=9, content="I repair clocks.", reference="Cyra")
SECOND_SLOT = dict(id=12, content="I restore radios.", reference="Cyra")
ROW = dict(body="What work does Cyra do?", source_id="question", query_class="named")
LABEL = dict(entities=["Cyra"], prop="job", kind=0)
NULL_LABEL = dict(entities=[], prop="job", kind=0)


class NativeEpisodeSchema(unittest.TestCase):
    def curriculum(self, *, need=False, references=None, entity="Cyra"):
        span = dict(start=0, end=20, action=1, need=need,
                    references=[] if references is None else references,
                    entity=entity, prop="job", kind=0)
        fact = dict(body="Cyra repairs clocks.", past="", speaker="Observer",
                    timestamp="2031-05-20", source_id="fact", tag="episode-fact",
                    spans=[span], gold_only="must stay outside native transport")
        query = dict(body="What work does Cyra do?", past="", speaker="Observer",
                     timestamp="2031-05-21", source_id="query", tag="episode-query",
                     entities=[entity], prop="job", kind=0,
                     support_keys=[dict(entity=entity, prop="job", kind=0)])
        return dict(blind=[fact, query])

    def specification(self, curriculum):
        root = Path(__file__).resolve().parents[1] / "build"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "curriculum.json"
            path.write_text(json.dumps(curriculum))
            return curriculum_specification("blind", path)

    def test_explicit_name_fact_keeps_entity_separate_from_empty_external_reference(self):
        facts, queries, negatives = self.specification(self.curriculum())
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["content"], "Cyra repairs clocks.")
        self.assertEqual(facts[0]["reference"], "")
        self.assertEqual(facts[0]["entity"], "Cyra")
        self.assertEqual(facts[0]["text"], "Cyra. Cyra repairs clocks.")
        self.assertEqual(queries[0][1]["support_texts"], ["Cyra. Cyra repairs clocks."])
        self.assertEqual(negatives, [])

    def test_explicit_name_fact_requires_nonempty_canonical_entity(self):
        for entity in (None, "", " ", 4):
            with self.subTest(entity=entity):
                with self.assertRaisesRegex(ValueError, "entity"):
                    self.specification(self.curriculum(entity=entity))

    def test_no_external_reference_need_rejects_nonempty_references(self):
        with self.assertRaisesRegex(ValueError, "references"):
            self.specification(self.curriculum(references=["Cyra"]))

    def test_empty_expected_reference_still_requires_full_source_and_exact_reference(self):
        facts, queries, _ = self.specification(self.curriculum())
        row, label = queries[0]
        for content, reference, correct in (
            ("Cyra repairs clocks.", "", 1),
            ("Cyra repairs clocks.", "Cyra", 0),
            ("Cyra repairs", "", 0),
        ):
            with self.subTest(content=content, reference=reference):
                report, _ = score(
                    [dict(id=9, content=content, reference=reference)], facts,
                    [(row, label, dict(activated_ids=[9], complete=True))],
                )
                self.assertEqual(report["correct_writes"], correct)
                self.assertEqual(report["end_to_end_positive_exact"], correct)

    def test_native_writer_transport_excludes_all_scorer_labels(self):
        facts, _, negatives = self.specification(self.curriculum())
        writes, positive_count = write_observations(facts, negatives)
        self.assertEqual(positive_count, 1)
        root = Path(__file__).resolve().parents[1] / "build"
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "writes.in"
            pack(path, writes, True)
            expected = b"BNSOBS01" + struct.pack("<I", 1)
            for text in ("Cyra repairs clocks.", "", "Observer", "2031-05-20", "fact"):
                raw = text.encode()
                expected += struct.pack("<I", len(raw)) + raw
            self.assertEqual(path.read_bytes(), expected)

    def external_curriculum(self):
        curriculum = self.curriculum(need=True, references=["Cyra"])
        row = curriculum["blind"][0]
        row["body"] = "Cyra said: Cyra repairs clocks. Cyra"
        row["spans"][0].update(start=11, end=31)
        row["spans"][0].pop("entity")
        return curriculum

    def test_existing_external_reference_fact_without_explicit_ranges_is_preserved(self):
        facts, queries, _ = self.specification(self.external_curriculum())
        self.assertEqual(facts[0]["reference"], "Cyra")
        self.assertEqual(facts[0]["entity"], "Cyra")
        self.assertEqual(queries[0][1]["support_texts"], ["Cyra. Cyra repairs clocks."])

    def test_explicit_external_body_reference_must_precede_source_span(self):
        for start, end in ((11, 15), (32, 36), (0, 15)):
            curriculum = self.external_curriculum()
            curriculum["blind"][0]["spans"][0]["ref_ranges"] = [
                dict(document="body", start=start, end=end)
            ]
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, "causal"):
                    self.specification(curriculum)

    def test_explicit_external_body_prefix_and_context_addresses_are_accepted(self):
        for document, start, end in (("body", 0, 4), ("context", 9, 13)):
            curriculum = self.external_curriculum()
            row = curriculum["blind"][0]
            row["speaker"] = "Cyra"
            row["spans"][0]["ref_ranges"] = [dict(document=document, start=start, end=end)]
            with self.subTest(document=document):
                facts, _, _ = self.specification(curriculum)
                self.assertEqual(facts[0]["reference"], "Cyra")

    def test_explicit_external_address_uses_native_boundary_punctuation_trim(self):
        for document in ("body", "context"):
            for text in ("Cyra.", "'Cyra'", ".,!?;:'\"Cyra.,!?;:'\""):
                curriculum = self.external_curriculum()
                row = curriculum["blind"][0]
                span = row["spans"][0]
                if document == "body":
                    prefix = text + " said: "
                    row["body"] = prefix + "Cyra repairs clocks."
                    span.update(start=len(prefix.encode()), end=len(row["body"].encode()))
                    start = 0
                else:
                    row["speaker"] = text
                    start = 9
                span["ref_ranges"] = [dict(document=document, start=start,
                                           end=start + len(text.encode()))]
                with self.subTest(document=document, text=text):
                    facts, _, _ = self.specification(curriculum)
                    self.assertEqual(facts[0]["reference"], "Cyra")

    def test_explicit_external_reference_address_validates_original_utf8_bytes(self):
        base = self.external_curriculum()
        base["blind"][0]["past"] = "Zoë"
        for ref in (
            dict(document="other", start=0, end=4),
            dict(document="body", start=-1, end=4),
            dict(document="body", start=1, end=4),
            dict(document="body", start=0, end=0),
            dict(document="context", start=0, end=1000),
            dict(document="context", start=44, end=45),
        ):
            curriculum = deepcopy(base)
            curriculum["blind"][0]["spans"][0]["ref_ranges"] = [ref]
            with self.subTest(ref=ref):
                with self.assertRaises(ValueError):
                    self.specification(curriculum)


class NativeStageMetrics(unittest.TestCase):
    def score_fixture(self, slots, facts, outputs):
        report, records = score(slots, facts, outputs)
        self.assertIn("stage_metrics", report)
        for record in records:
            self.assertIn("first_failed_stage", record)
        return report, records

    def test_missing_or_partial_source_is_write_blocked_even_if_reference_rejects(self):
        # A useful prefix remains an incomplete strict source match. Neither a
        # prefix nor an unresolved read should become an independent head error.
        for slots in (
            [],
            [dict(id=9, content="I repair", reference="Cyra")],
            [dict(id=9, content="I repair clocks.", reference="Other person")],
        ):
            with self.subTest(slots=slots):
                report, records = self.score_fixture(
                    slots, [FACT],
                    [(ROW, LABEL, dict(activated_ids=[], unresolved=True, complete=True))],
                )
                stage = report["stage_metrics"]
                self.assertEqual(stage["positive_queries"], 1)
                self.assertEqual(stage["fully_written_positive_queries"], 0)
                self.assertEqual(stage["write_blocked_positive_queries"], 1)
                self.assertEqual(stage["reference_rejected_positive_queries"], 0)
                self.assertEqual(stage["activation_eligible_positive_queries"], 0)
                self.assertEqual(stage["activation_exact_positive_queries"], 0)
                self.assertIsNone(stage["conditional_activation_exact_accuracy"])
                self.assertEqual(records[0]["first_failed_stage"], "write")
                self.assertEqual(report["fn"], 1)
                self.assertEqual(report["end_to_end_positive_exact"], 0)

    def test_fully_written_unresolved_query_is_reference_failure(self):
        report, records = self.score_fixture(
            [SLOT], [FACT],
            [(ROW, LABEL, dict(activated_ids=[], unresolved=True, complete=True))],
        )
        self.assertEqual(report["stage_metrics"], dict(
            positive_queries=1, fully_written_positive_queries=1,
            write_blocked_positive_queries=0, reference_rejected_positive_queries=1,
            activation_eligible_positive_queries=0, activation_exact_positive_queries=0,
            conditional_activation_exact_accuracy=None,
        ))
        self.assertEqual(records[0]["first_failed_stage"], "reference")
        self.assertEqual(report["unresolved_queries"], 1)

    def test_exact_activation_keeps_every_existing_report_metric(self):
        report, records = self.score_fixture(
            [SLOT], [FACT],
            [(ROW, LABEL, dict(activated_ids=[9], unresolved=False, complete=True))],
        )
        legacy = {key: value for key, value in report.items() if key != "stage_metrics"}
        self.assertEqual(legacy, dict(
            correct_writes=1, expected_writes=1, write_recall=1.0, write_f1=1.0,
            positive_queries=1, positive_exact=1, end_to_end_positive_exact=1.0,
            incomplete_queries=0, unresolved_queries=0, resolved_converged_queries=1,
            converged_positive_exact=1.0, support_set_exact=1.0,
            activation_f1=1.0, tp=1, fp=0, fn=0, null_queries=0,
            null_false_activation=None,
        ))
        self.assertEqual(report["stage_metrics"], dict(
            positive_queries=1, fully_written_positive_queries=1,
            write_blocked_positive_queries=0, reference_rejected_positive_queries=0,
            activation_eligible_positive_queries=1, activation_exact_positive_queries=1,
            conditional_activation_exact_accuracy=1.0,
        ))
        self.assertEqual(records[0]["first_failed_stage"], "none")

    def test_missing_one_of_two_written_supports_is_activation_failure(self):
        report, records = self.score_fixture(
            [SLOT, SECOND_SLOT], [FACT, SECOND_FACT],
            [(ROW, LABEL, dict(activated_ids=[9], unresolved=False, complete=True))],
        )
        self.assertEqual(report["stage_metrics"]["fully_written_positive_queries"], 1)
        self.assertEqual(report["stage_metrics"]["activation_eligible_positive_queries"], 1)
        self.assertEqual(report["stage_metrics"]["activation_exact_positive_queries"], 0)
        self.assertEqual(report["stage_metrics"]["conditional_activation_exact_accuracy"], 0)
        self.assertEqual(records[0]["first_failed_stage"], "activation")
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (1, 0, 1))

    def test_extra_full_bank_record_prevents_conditional_exact_support(self):
        distractor = dict(id=41, content="The parcel is by the gate.", reference="Mira")
        report, records = self.score_fixture(
            [SLOT, distractor], [FACT],
            [(ROW, LABEL, dict(activated_ids=[9, 41], unresolved=False, complete=True))],
        )
        self.assertEqual(report["stage_metrics"]["activation_eligible_positive_queries"], 1)
        self.assertEqual(report["stage_metrics"]["activation_exact_positive_queries"], 0)
        self.assertEqual(report["stage_metrics"]["conditional_activation_exact_accuracy"], 0)
        self.assertEqual(records[0]["first_failed_stage"], "activation")
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (1, 1, 0))

    def test_ambiguous_null_reference_rejection_does_not_enter_positive_denominator(self):
        report, records = self.score_fixture(
            [SLOT], [FACT],
            [(ROW, NULL_LABEL, dict(activated_ids=[], unresolved=True, complete=True))],
        )
        self.assertEqual(report["stage_metrics"], dict(
            positive_queries=0, fully_written_positive_queries=0,
            write_blocked_positive_queries=0, reference_rejected_positive_queries=0,
            activation_eligible_positive_queries=0, activation_exact_positive_queries=0,
            conditional_activation_exact_accuracy=None,
        ))
        self.assertEqual(records[0]["first_failed_stage"], "none")
        self.assertEqual(report["support_set_exact"], 1)
        self.assertEqual(report["null_false_activation"], 0)

    def test_null_extra_activation_has_its_own_failure_stage(self):
        report, records = self.score_fixture(
            [SLOT], [FACT],
            [(ROW, NULL_LABEL, dict(activated_ids=[9], unresolved=False, complete=True))],
        )
        self.assertEqual(records[0]["first_failed_stage"], "null_false_activation")
        self.assertEqual(report["stage_metrics"]["activation_eligible_positive_queries"], 0)
        self.assertIsNone(report["stage_metrics"]["conditional_activation_exact_accuracy"])
        self.assertEqual(report["fp"], 1)
        self.assertEqual(report["null_false_activation"], 1)

    def test_mixed_stages_keep_convergence_separate_from_conditional_coverage(self):
        missing = dict(content="I tune pianos.", reference="Cyra",
                       text="Cyra. I tune pianos.", entity="Cyra", prop="job", kind=0)
        written_label = dict(support_texts=["Cyra. I repair clocks."])
        report, records = self.score_fixture(
            [SLOT], [FACT, missing],
            [
                (ROW, LABEL, dict(activated_ids=[], unresolved=True, complete=True)),
                (ROW, written_label, dict(activated_ids=[], unresolved=True, complete=True)),
                (ROW, written_label, dict(activated_ids=[], unresolved=False, complete=True)),
                (ROW, written_label, dict(activated_ids=[9], unresolved=False, complete=False)),
                (ROW, NULL_LABEL, dict(activated_ids=[], unresolved=True, complete=True)),
            ],
        )
        self.assertEqual([r["first_failed_stage"] for r in records],
                         ["write", "reference", "activation", "none", "none"])
        self.assertEqual(report["stage_metrics"], dict(
            positive_queries=4, fully_written_positive_queries=3,
            write_blocked_positive_queries=1, reference_rejected_positive_queries=1,
            activation_eligible_positive_queries=2, activation_exact_positive_queries=1,
            conditional_activation_exact_accuracy=0.5,
        ))
        self.assertEqual(report["support_set_exact"], 0.4)
        self.assertEqual(report["incomplete_queries"], 1)
        self.assertEqual(report["converged_positive_exact"], 0)


if __name__ == "__main__":
    unittest.main()
