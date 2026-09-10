#!/usr/bin/env python3
"""Unit tests for LoCoMo source-evidence accounting."""
import importlib.util
import json
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("eval_locomo.py")
SPEC = importlib.util.spec_from_file_location("eval_locomo", MODULE_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class LoCoMoEvidenceTest(unittest.TestCase):
    def test_turn_index_and_gold_context(self):
        conversation = {
            "session_1_date_time": "8 May 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "A", "text": "hello"},
                {"dia_id": "D1:2", "speaker": "B", "text": "yesterday"},
            ],
        }
        index = EVAL.build_turn_index(conversation)
        self.assertEqual(index["D1:2"]["date"], "8 May 2023")
        self.assertEqual(
            EVAL.source_context(["D1:2"], index),
            "[source D1:2 | 8 May 2023] B: yesterday")

    def test_limited_sessions_filter_unanswerable_questions(self):
        qas = [
            {"evidence": ["D1:1"]},
            {"evidence": ["D2:1"]},
            {"evidence": []},
        ]
        self.assertEqual(
            EVAL.eligible_qas(qas, {"D1:1"}, limited_sessions=True),
            [qas[0]])
        self.assertEqual(
            EVAL.eligible_qas(qas, {"D1:1"}, limited_sessions=False),
            qas)

    def test_ranked_evidence_and_metrics(self):
        context = (
            "[memory 1] chunk\n[source D2:1] B: distractor\n"
            "[memory 2] chunk\n[source D1:3] A: first\n"
            "[source D1:9] A: second\n")
        ranked = EVAL.ranked_evidence_ids(context)
        self.assertEqual(ranked, [{"D2:1"}, {"D1:3", "D1:9"}])
        metrics = EVAL.evidence_retrieval_metrics(
            ranked, ["D1:3", "D1:9"])
        self.assertEqual(metrics["evidence_recall_at_1"], 0.0)
        self.assertEqual(metrics["evidence_recall_at_5"], 1.0)
        self.assertEqual(metrics["evidence_complete_at_5"], 1.0)
        self.assertEqual(metrics["evidence_mrr"], 0.5)

    def test_partial_multi_evidence_recall(self):
        metrics = EVAL.evidence_retrieval_metrics(
            [{"D1:3"}], ["D1:3", "D1:9"])
        self.assertEqual(metrics["evidence_recall_at_1"], 0.5)
        self.assertEqual(metrics["evidence_complete_at_1"], 0.0)

    def test_packed_evidence_ids_are_normalized(self):
        self.assertEqual(
            EVAL.qa_evidence_ids({
                "evidence": ["D8:6; D9:17", "D10:2"]}),
            ["D8:6", "D9:17", "D10:2"])

    def test_load_write_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "plan.json"
            path.write_text(json.dumps([
                {
                    "conversation": 0,
                    "source_id": "D1:1",
                    "selected": True,
                },
                {
                    "conversation": 0,
                    "source_id": "D1:2",
                    "selected": False,
                },
            ]), encoding="utf-8")
            self.assertEqual(
                EVAL.load_write_plan(path),
                {(0, "D1:1"): True, (0, "D1:2"): False})

    def test_load_priority_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "priorities.json"
            path.write_text(json.dumps([
                {
                    "conversation": 2,
                    "source_id": "D1:1",
                    "write_probability": 0.75,
                },
            ]), encoding="utf-8")
            self.assertEqual(
                EVAL.load_priority_plan(path),
                {(2, "D1:1"): 0.75})


if __name__ == "__main__":
    unittest.main()
