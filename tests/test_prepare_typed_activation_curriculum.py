#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_typed_activation_curriculum import (  # noqa: E402
    compile_file,
    compile_row,
)


def fixture(intent):
    events = [
        {
            "episode": 0,
            "entity": "Mira",
            "entity_surface": "Mira",
            "predicate": "backup code",
            "predicate_surface": "backup code",
            "value": "violet-7319",
            "active": False,
        },
        {
            "episode": 1,
            "entity": "Jon",
            "entity_surface": "Jon",
            "predicate": "storage location",
            "predicate_surface": "storage location",
            "value": "blue cabinet",
            "active": True,
        },
        {
            "episode": 2,
            "entity": "Mira",
            "entity_surface": "Mira",
            "predicate": "backup code",
            "predicate_surface": "backup code",
            "value": "amber-2046",
            "active": True,
        },
    ]
    return {
        "sample_id": f"sample-{intent}",
        "question": "What is Mira's current backup code?",
        "answer": (
            "amber-2046"
            if intent == "current"
            else "No information available"
        ),
        "answer_spans": [],
        "answer_type": "span",
        "evidence": "\n".join([
            "[source s0 | 2026-01-01] "
            "Mira's backup code is violet-7319.",
            "[source s1 | 2026-01-02] "
            "Jon's storage location is blue cabinet.",
            "[source s2 | 2026-01-03] "
            "Mira's backup code is amber-2046.",
        ]),
        "metadata": {
            "world_id": "world-1",
            "locomo_used": False,
            "counterfactual_no_info": intent == "null",
            "typed_events": events,
            "query_plan": {
                "intent": intent,
                "targets": (
                    [{
                        "entity": "Mira",
                        "predicate": "backup code",
                        "version": "active",
                    }]
                    if intent == "current"
                    else []
                ),
            },
        },
    }


class TypedActivationCurriculumTest(unittest.TestCase):
    def test_current_keeps_only_active_evidence(self):
        row = compile_row(fixture("current"))
        self.assertNotIn("violet-7319", row["evidence"])
        self.assertIn("blue cabinet", row["evidence"])
        self.assertIn("amber-2046", row["evidence"])
        self.assertEqual(row["answer"], "amber-2046")
        span = row["answer_spans"][0]
        self.assertEqual(
            row["evidence"][span["start"]:span["end"]],
            "amber-2046",
        )
        self.assertEqual(
            row["metadata"][
                "typed_activation_candidate_count"
            ],
            2,
        )

    def test_null_has_explicit_null_target(self):
        row = compile_row(fixture("null"))
        self.assertEqual(
            row["answer"], "No information available"
        )
        self.assertEqual(row["answer_spans"], [])
        self.assertTrue(
            row["metadata"]["counterfactual_no_info"]
        )

    def test_file_skips_unsupported_intents(self):
        current = fixture("current")
        skipped = fixture("previous")
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source.jsonl"
            output = pathlib.Path(directory) / "output.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (current, skipped)
                )
                + "\n",
                encoding="utf-8",
            )
            counters = compile_file(source, output)
            self.assertEqual(counters["output"], 1)
            self.assertEqual(counters["skipped_intent"], 1)
            self.assertEqual(
                len(output.read_text(
                    encoding="utf-8"
                ).splitlines()),
                1,
            )


if __name__ == "__main__":
    unittest.main()
