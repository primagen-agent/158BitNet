"""Balanced time batches preserve existing subjects and full physical banks."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neural_activation_curriculum import lifecycle_fit_group
from neural_temporal_curriculum import (
    TEMPORAL_MAIN_SEED, TEMPORAL_KIND_SEED, prepare_temporal_fit,
    sample_temporal_main, sample_temporal_kind,
)


def small_fit():
    facts, queries = [], []
    for kind in (0, 1, 2):
        for index in range(12):
            text = f"Mara. Kind {kind} fact {index}."
            facts.append(dict(text=text, reference="Mara", entity="Mara", prop="detail", kind=kind))
            if index < 8:
                body = f"Which kind {kind} detail number {index}?"
                queries.append(dict(text="Mara's. " + body, reference="Mara's",
                                    initial_text="Mara's. " + body, initial_reference="Mara's",
                                    entities=["Mara"], prop="detail", kind=kind,
                                    support_texts=[text], binding_predicted=True,
                                    observation_id=f"fit::{kind}::{index}",
                                    observation=dict(body=body, context="Earlier ordinary subject mention",
                                                     source_id=f"fit::{kind}::{index}")))
    for index in range(8):
        queries.append(dict(text=f"What missing value {index}?", entities=[], prop="", kind=0,
                            support_texts=[], observation_id=f"fit::null::{index}"))
    return dict(facts=facts, queries=queries)


class TemporalCurriculumTests(unittest.TestCase):
    def test_pools_preserve_all_original_queries_and_source_kind_not_status(self):
        fit, life = small_fit(), lifecycle_fit_group()
        originals = deepcopy((fit, life))
        prepared, manifest = prepare_temporal_fit(fit, life)
        self.assertEqual(manifest["fit_original_inputs"], 32)
        self.assertEqual(manifest["fit_positive_counts"], {0: 8, 1: 8, 2: 8})
        self.assertEqual(manifest["fit_null_queries"], 8)
        self.assertEqual(manifest["lifecycle_view_counts"], dict(current=12, history=12, tentative=12, all_versions=12))
        self.assertEqual(manifest["source_kind_counts"], {0: 36, 1: 12, 2: 24})
        self.assertEqual((fit, life), originals)
        historical = [row for row in prepared["source_kind_facts"][0] if row.get("status") == 1]
        self.assertEqual(len(historical), 12)
        self.assertTrue(all(row["kind"] == 0 for row in historical))
        self.assertTrue(manifest["no_texts_created"])
        self.assertEqual(manifest["main_seed"], 5362)
        self.assertEqual(manifest["kind_seed"], 5363)

    def test_each_step_uses_full_bank_and_preserves_predicted_original_references(self):
        prepared, _ = prepare_temporal_fit(small_fit(), lifecycle_fit_group())
        for step in (1, 2, 3, 4, 5):
            name, rows, facts = sample_temporal_main(prepared, step, random.Random(TEMPORAL_MAIN_SEED))
            self.assertEqual(len(rows), 32)
            self.assertFalse(any(row.get("augmented") for row in rows))
            self.assertFalse(any("Activated evidence:" in row["text"] for row in rows))
            if step % 5:
                self.assertEqual(name, "lifecycle_zero")
                self.assertEqual(facts, prepared["lifecycle_facts"])
                self.assertEqual(Counter(row["lifecycle_view"] for row in rows),
                                 dict(current=8, history=8, tentative=8, all_versions=8))
                self.assertTrue(all("reference" not in row for row in rows))
            else:
                self.assertEqual(name, "old_fit_zero")
                self.assertEqual(facts, prepared["fit_facts"])
                self.assertEqual(Counter(row["kind"] for row in rows if row["support_texts"]), {0: 8, 1: 8, 2: 8})
                for row in rows:
                    if row["support_texts"]:
                        self.assertEqual(row["reference"], "Mara's")
                        self.assertEqual(row["initial_reference"], "Mara's")
            facts[0]["text"] = "External mutation"
            rows[0]["text"] = "External mutation"
        self.assertFalse(any(f["text"] == "External mutation" for f in prepared["fit_facts"]))

    def test_six_hundred_counts_and_separate_random_streams(self):
        prepared, _ = prepare_temporal_fit(small_fit(), lifecycle_fit_group())
        main_rng, kind_rng = random.Random(TEMPORAL_MAIN_SEED), random.Random(TEMPORAL_KIND_SEED)
        steps, views, intent, kinds = Counter(), Counter(), Counter(), Counter()
        for step in range(1, 601):
            name, rows, facts = sample_temporal_main(prepared, step, main_rng)
            steps[name] += 1
            for row in rows:
                views[row.get("lifecycle_view", "old_null" if not row["support_texts"] else "old_positive")] += 1
                for kind in row.get("allowed_kinds", [row["kind"]]):
                    intent[kind] += 1
            for row in sample_temporal_kind(prepared, kind_rng):
                kinds[row.get("source_kind", row["kind"])] += 1
        self.assertEqual(steps, dict(lifecycle_zero=480, old_fit_zero=120))
        self.assertEqual(views, dict(current=3840, history=3840, tentative=3840,
                                    all_versions=3840, old_positive=2880, old_null=960))
        self.assertEqual(intent, {0: 9600, 1: 8640, 2: 8640})
        self.assertEqual(kinds, {0: 7200, 1: 7200, 2: 7200})
        self.assertEqual(sample_temporal_main(prepared, 5, random.Random(5362)),
                         sample_temporal_main(prepared, 5, random.Random(5362)))
        first = sample_temporal_main(prepared, 1, random.Random(5362))
        sample_temporal_kind(prepared, random.Random(5363))
        self.assertEqual(first, sample_temporal_main(prepared, 1, random.Random(5362)))

    def test_augmented_rows_filtered_and_equivalent_originals_deduplicated(self):
        fit = small_fit()
        duplicate = deepcopy(fit["queries"][0])
        duplicate["observation_id"] = "fit::alias"
        duplicate["observation"].update(source_id="fit::alias", context="Another original context")
        fit["queries"].append(duplicate)
        cue = deepcopy(fit["queries"][0]);cue.update(text=cue["text"] + "\nActivated evidence:\nOther facts\n", augmented=True, evidence_count=1)
        fit["queries"].append(cue)
        prepared, manifest = prepare_temporal_fit(fit, lifecycle_fit_group())
        self.assertEqual(manifest["fit_original_rows"], 33)
        self.assertEqual(manifest["fit_original_inputs"], 32)
        matched = [r for r in prepared["fit_positive"][0] if r["replay_equivalent_rows"] == 2]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["replay_equivalent_observations"], ["fit::0::0", "fit::alias"])
        conflict = deepcopy(fit["queries"][0]);conflict["support_texts"] = []
        fit["queries"].append(conflict)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            prepare_temporal_fit(fit, lifecycle_fit_group())

    def test_deleted_slots_leave_recall_holes_but_source_fact_classification_remains(self):
        fit = small_fit()
        deleted = deepcopy(fit["facts"][0]);deleted.update(status=2, insert_order=40, bank_totalcount=42)
        fit["facts"].append(deleted)
        prepared, manifest = prepare_temporal_fit(fit, lifecycle_fit_group())
        self.assertEqual(len(prepared["fit_facts"]), 36)
        self.assertTrue(all(row["bank_totalcount"] >= 41 for row in prepared["fit_facts"]))
        self.assertEqual([f["insert_order"] for f in prepared["fit_facts"]], list(range(36)))
        self.assertEqual(manifest["source_kind_facts"], 72)
        self.assertEqual(manifest["source_kind_counts"], {0: 36, 1: 12, 2: 24})
        deleted["source_kind"] = 2
        fit["facts"][-1] = deleted
        with self.assertRaisesRegex(ValueError, "conflicting original source-kind"):
            prepare_temporal_fit(fit, lifecycle_fit_group())

    def test_evaluation_new_binding_qualification_and_unbound_inputs_fail_closed(self):
        for split in ("development", "evaluation", "blind"):
            with self.assertRaisesRegex(ValueError, "fit-only"):
                prepare_temporal_fit(small_fit(), lifecycle_fit_group(), split=split)
        for change in (dict(evaluation_only=True), dict(split="development"),
                       dict(evidence_challenge_view=0), dict(binding_fit=True)):
            fit = small_fit();fit["queries"][0].update(change)
            with self.assertRaisesRegex(ValueError, "cannot enter"):
                prepare_temporal_fit(fit, lifecycle_fit_group())
        with self.assertRaisesRegex(ValueError, "qualification"):
            prepare_temporal_fit(small_fit(), lifecycle_fit_group(), excluded_observation_ids={"fit::0::0"})
        for change in (dict(binding_predicted=False), dict(unresolved=True)):
            fit = small_fit();fit["queries"][0].update(change)
            with self.assertRaisesRegex(ValueError, "binding"):
                prepare_temporal_fit(fit, lifecycle_fit_group())
        fit = small_fit();fit["facts"][0]["source_kind"] = 3
        with self.assertRaisesRegex(ValueError, "original text kind"):
            prepare_temporal_fit(fit, lifecycle_fit_group())
        for step in (0, -1, True, 1.5):
            prepared, _ = prepare_temporal_fit(small_fit(), lifecycle_fit_group())
            with self.assertRaisesRegex(ValueError, "positive integer"):
                sample_temporal_main(prepared, step, random.Random(5362))

    def test_current_operational_json_has_exact_pools_without_loading_features(self):
        source = ROOT / "build/reader-retention-1b/activation/curriculum.json"
        if not source.exists():
            self.skipTest("current operational curriculum is not generated")
        fit = json.loads(source.read_text())["fit"]
        before = deepcopy(fit)
        prepared, manifest = prepare_temporal_fit(fit, lifecycle_fit_group())
        self.assertEqual(manifest["fit_original_rows"], 1148)
        self.assertEqual(manifest["fit_original_inputs"], 1052)
        self.assertEqual(manifest["fit_positive_counts"], {0: 848, 1: 128, 2: 32})
        self.assertEqual(manifest["fit_null_queries"], 44)
        self.assertEqual(manifest["source_kind_counts"], {0: 536, 1: 64, 2: 28})
        self.assertEqual((manifest["fit_recall_bank"], manifest["lifecycle_recall_bank"]), (592, 36))
        self.assertEqual(fit, before)
        self.assertEqual(len(sample_temporal_kind(prepared, random.Random(5363))), 36)


if __name__ == "__main__":
    unittest.main()
