#!/usr/bin/env python3
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_repeated_dialogue_curriculum import build_world  # noqa: E402
from validate_typed_version_memory import (  # noqa: E402
    resolve_query,
    validate_rows,
)


class TypedVersionMemoryTest(unittest.TestCase):
    def test_all_query_families_resolve_exactly(self):
        rows = build_world(
            5, 128, 12, random.Random(31))
        metrics = validate_rows(rows)
        self.assertEqual(metrics["rows"], 12)
        self.assertEqual(metrics["exact"], 12)
        self.assertEqual(metrics["worlds"], 1)
        self.assertEqual(metrics["accuracy_ppm"], 1_000_000)
        for family in (
            "direct", "paraphrase", "temporal",
            "previous", "multi", "null",
        ):
            self.assertEqual(
                metrics[f"family_{family}"], 2)

    def test_previous_uses_version_link(self):
        rows = build_world(
            2, 128, 6, random.Random(37))
        previous = next(
            row for row in rows
            if row["metadata"]["family"] == "previous")
        plan = previous["metadata"]["query_plan"]
        target = plan["targets"][0]
        events = {
            event["episode"]: event
            for event in previous["metadata"]["typed_events"]}
        anchor = events[target["anchor_episode"]]
        self.assertEqual(
            resolve_query(previous),
            [anchor["previous_episode"]])

    def test_broken_active_version_is_rejected(self):
        rows = build_world(
            1, 128, 6, random.Random(41))
        events = rows[0]["metadata"]["typed_events"]
        active = next(
            event for event in events if event["active"])
        active["active"] = False
        with self.assertRaisesRegex(
                ValueError, "latest active"):
            validate_rows(rows[:1])


if __name__ == "__main__":
    unittest.main()
