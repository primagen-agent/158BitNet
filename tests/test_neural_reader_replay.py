"""Fit replay preserves complete banks and learned original query subjects."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_activation_curriculum import lifecycle_fit_group, relation_variation_fit_group
from neural_reader_evidence_curriculum import augment_evidence
from neural_reader_replay import (
    build_replay_groups, replay_feature_texts, replay_manifest,
    sample_replay_group, sample_replay_queries,
)


def fit_bank():
    facts = [dict(text=f"{name}. Stored {prop} {i}.", entity=name,
                  reference=name, prop=prop, kind=0, status=0, insert_order=i,
                  bank_totalcount=12)
             for i, (name, prop) in enumerate(
                 [(name, prop) for name in ("Mara", "Nils")
                  for prop in ("color", "location", "material", "routine")])]
    facts.append(dict(text="Mara. Deleted old color.", entity="Mara", reference="Mara",
                      prop="color", kind=0, status=2, insert_order=10, bank_totalcount=12))
    queries = []
    for i in range(30):
        supported = [] if i % 3 == 0 else [facts[i % 4]["text"]]
        if i % 3 == 2:
            supported.append(facts[(i + 1) % 4]["text"])
        queries.append(dict(
            text=f"Mara's. Where is item {i}?", reference="Mara's",
            entities=["Mara"], prop="color", kind=0, support_texts=supported,
            observation_id=f"fit::{i}", binding_predicted=True,
            observation=dict(body=f"Where is item {i}?", context="Ordinary context",
                             source_id=f"fit::{i}"),
        ))
    return dict(facts=facts, queries=queries)


class NeuralReaderReplayTests(unittest.TestCase):
    def test_three_views_keep_full_physical_bank_and_original_nn_subject(self):
        group = fit_bank()
        before = deepcopy(group)
        replay = build_replay_groups(dict(fit=group), query_limit=24)
        self.assertEqual(set(replay), {"fit_cue_0", "fit_cue_1", "fit_cue_4"})
        selected_texts = None
        original = {row["text"]: row for row in group["queries"]}
        for count in (0, 1, 4):
            view = replay[f"fit_cue_{count}"]
            self.assertEqual(view["facts"], group["facts"])
            self.assertEqual(len(view["queries"]), 24)
            texts = {row["initial_text"] for row in view["queries"]}
            self.assertEqual(texts, selected_texts or texts)
            selected_texts = texts
            for row in view["queries"]:
                source = original[row["initial_text"]]
                self.assertEqual(row["initial_reference"], "Mara's")
                self.assertEqual(row["support_texts"], source["support_texts"])
                self.assertEqual(row["observation"], source["observation"])
                self.assertEqual(row["evidence_count"], count)
                self.assertNotIn("evaluation_only", row)
                self.assertEqual(len(row["evidence_sources"]), count)
                self.assertFalse(any("Deleted" in cue for cue in row["evidence_sources"]))
                if count:
                    self.assertNotIn("reference", row)
                    self.assertEqual(row["text"], row["initial_text"] + "\nActivated evidence:\n"
                                     + "\n".join(row["evidence_sources"]) + "\n")
                else:
                    self.assertEqual(row["reference"], "Mara's")
                    self.assertEqual(row["text"], source["text"])
        replay["fit_cue_0"]["facts"][0]["text"] = "Modified"
        replay["fit_cue_1"]["queries"][0]["observation"]["body"] = "Modified"
        self.assertEqual(group, before)
        self.assertEqual(replay["fit_cue_4"]["facts"], before["facts"])

    def test_deterministic_under_input_order_and_caller_rng_independent(self):
        group = fit_bank()
        first = build_replay_groups(dict(fit=group))
        random.seed(1234)
        shuffled = deepcopy(group)
        random.shuffle(shuffled["queries"])
        self.assertEqual(first, build_replay_groups(dict(fit=shuffled)))
        changed = build_replay_groups(dict(fit=group), seeds=dict(fit=5327))
        self.assertNotEqual(first, changed)
        self.assertEqual(group, fit_bank())

    def test_existing_fit_cues_reused_without_changing_reference_or_targets(self):
        group = fit_bank()
        live = [fact for fact in group["facts"] if fact["status"] != 2]
        augmented = augment_evidence(dict(facts=live, queries=group["queries"]), 5321, 600)
        augmented["facts"] = group["facts"]
        # Current cue references are removed for native semantics; original
        # frozen NN bindings remain exact, including possessive characters.
        for row in augmented["queries"]:
            if row.get("augmented"):
                row["reference"] = "wrong distractor"
        texts = {row["text"] for row in augmented["queries"]}
        replay = build_replay_groups(dict(fit=augmented))
        for view in replay.values():
            for row in view["queries"]:
                self.assertIn(row["text"], texts)
                self.assertEqual(row["initial_reference"], "Mara's")
                if row["evidence_count"]:
                    self.assertNotIn("reference", row)

    def test_equivalent_observations_are_collapsed_with_stable_provenance(self):
        group = fit_bank()
        duplicate = deepcopy(group["queries"][1])
        duplicate.update(observation_id="fit::alias")
        duplicate["observation"].update(source_id="fit::alias", context="Different original context")
        # Support order has no teacher meaning.
        duplicate["support_texts"] = list(reversed(duplicate["support_texts"]))
        group["queries"].append(duplicate)
        replay = build_replay_groups(dict(fit=group), query_limit=32)
        self.assertTrue(all(len(view["queries"]) == 30 for view in replay.values()))
        for view in replay.values():
            row = next(row for row in view["queries"] if row["initial_text"] == duplicate["text"])
            self.assertEqual(row["replay_equivalent_rows"], 2)
            self.assertEqual(row["replay_equivalent_observations"], ["fit::1", "fit::alias"])
        group["queries"].reverse()
        self.assertEqual(replay, build_replay_groups(dict(fit=group), query_limit=32))

    def test_conflicting_original_or_evidence_targets_fail_closed(self):
        for change in (dict(support_texts=[]), dict(kind=1), dict(prop="material"),
                       dict(allowed_kinds=[1]), dict(source_kinds=[2])):
            group = fit_bank()
            duplicate = deepcopy(group["queries"][1])
            duplicate.update(change, observation_id="fit::conflict")
            group["queries"].append(duplicate)
            with self.assertRaisesRegex(ValueError, "conflicting fit replay targets"):
                build_replay_groups(dict(fit=group))
        group = fit_bank()
        live = [fact for fact in group["facts"] if fact["status"] != 2]
        group["queries"] = augment_evidence(dict(facts=live, queries=group["queries"]),
                                             5321, 600)["queries"]
        original = next(row for row in group["queries"] if row.get("augmented"))
        duplicate = deepcopy(original)
        duplicate["kind"] = 2
        group["queries"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "conflicting fit replay targets"):
            build_replay_groups(dict(fit=group))

    def test_same_text_different_nn_reference_or_original_input_stay_distinct(self):
        group = fit_bank()
        first = deepcopy(group["queries"][1])
        second = deepcopy(first)
        second.update(reference="Nils", observation_id="fit::other-subject",
                      entities=["Nils"], support_texts=[group["facts"][5]["text"]])
        third = deepcopy(first)
        third.update(initial_text="Earlier original question", initial_reference="Earlier's",
                     observation_id="fit::earlier-original")
        group["queries"] = [first, second, third]
        replay = build_replay_groups(dict(fit=group), query_limit=32)
        for count in (0, 1, 4):
            rows = replay[f"fit_cue_{count}"]["queries"]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["initial_reference"] for row in rows},
                             {"Mara's", "Nils", "Earlier's"})
            earlier = next(row for row in rows if row["initial_reference"] == "Earlier's")
            self.assertEqual(earlier["initial_text"], "Earlier original question")
            for row in rows:
                if count:
                    self.assertNotIn("reference", row)
                if row["initial_reference"] == "Nils":
                    self.assertEqual(row["support_texts"], [group["facts"][5]["text"]])

    def test_selection_and_batch_balance_null_single_and_complete_support_sets(self):
        replay = build_replay_groups(dict(fit=fit_bank()), query_limit=24)
        group = replay["fit_cue_4"]
        support = Counter(len(row["support_texts"]) for row in group["queries"])
        self.assertEqual(support, {0: 8, 1: 8, 2: 8})
        batch = sample_replay_queries(group, random.Random(42), 12)
        self.assertEqual(Counter(len(row["support_texts"]) for row in batch), {0: 4, 1: 4, 2: 4})
        self.assertEqual(batch, sample_replay_queries(group, random.Random(42), 12))
        self.assertEqual(sample_replay_queries(group, random.Random(42), 0), [])
        self.assertEqual(len(sample_replay_queries(group, random.Random(42), 100)), 24)
        self.assertEqual(sample_replay_group(replay, random.Random(13)),
                         sample_replay_group(replay, random.Random(13)))

    def test_existing_lifecycle_and_variation_use_original_raw_questions(self):
        original = dict(lifecycle_fit=lifecycle_fit_group(),
                        relation_variation_fit=relation_variation_fit_group())
        before = deepcopy(original)
        replay = build_replay_groups(original)
        self.assertEqual(len(replay), 6)
        for name, group in original.items():
            texts = {row["text"] for row in group["queries"]}
            for count in (0, 1, 4):
                view = replay[f"{name}_cue_{count}"]
                self.assertEqual(view["facts"], group["facts"])
                self.assertEqual(len(view["queries"]), 24)
                for row in view["queries"]:
                    self.assertIn(row["initial_text"], texts)
                    self.assertNotIn("reference", row)
                    self.assertNotIn("initial_reference", row)
        self.assertEqual(original, before)

    def test_feature_manifest_exact_and_existing_cache_requirements_visible(self):
        replay = build_replay_groups(dict(fit=fit_bank()))
        texts = replay_feature_texts(replay)
        expected = {row["text"] for group in replay.values()
                    for rows in group.values() for row in rows}
        expected.add("Mara's")
        expected.update(fact["reference"] for fact in fit_bank()["facts"])
        self.assertEqual(texts, sorted(expected))
        existing = texts[::2]
        manifest = replay_manifest(replay, existing)
        self.assertEqual(manifest["feature_texts"], len(texts))
        self.assertEqual(manifest["additional_feature_texts"], sorted(set(texts) - set(existing)))
        self.assertEqual(manifest, replay_manifest(replay, existing))
        self.assertFalse(manifest["selection_data"])
        self.assertTrue(all(row["facts"] == 9 for row in manifest["groups"].values()))
        self.assertNotIn("Ordinary context", texts)

    def test_development_evaluation_binding_and_qualification_fail_closed(self):
        for split in ("development", "evaluation", "blind"):
            with self.assertRaisesRegex(ValueError, "fit-only"):
                build_replay_groups(dict(fit=fit_bank()), split=split)
        for name in ("development", "episode_development", "binding_fit", "reader_binding_fit"):
            with self.assertRaisesRegex(ValueError, "registered fit"):
                build_replay_groups({name: fit_bank()})
        for change in (dict(evaluation_only=True), dict(split="development"),
                       dict(evidence_challenge_view=0), dict(binding_fit=True)):
            damaged = fit_bank()
            damaged["queries"][0].update(change)
            with self.assertRaises(ValueError):
                build_replay_groups(dict(fit=damaged))
        with self.assertRaisesRegex(ValueError, "qualification"):
            build_replay_groups(dict(fit=fit_bank()), excluded_observation_ids={"fit::1"})
        damaged = fit_bank()
        damaged["queries"][0].pop("binding_predicted")
        with self.assertRaisesRegex(ValueError, "learned writer binding"):
            build_replay_groups(dict(fit=damaged))

    def test_invalid_budget_seed_and_bank_are_rejected(self):
        for count in (0, 33, True, 2.5):
            with self.assertRaisesRegex(ValueError, "limit"):
                build_replay_groups(dict(fit=fit_bank()), query_limit=count)
        for seeds in (dict(fit=True), dict(other=4), [4]):
            with self.assertRaisesRegex(ValueError, "seeds"):
                build_replay_groups(dict(fit=fit_bank()), seeds=seeds)
        damaged = fit_bank()
        damaged["facts"] = damaged["facts"][:3]
        with self.assertRaisesRegex(ValueError, "four-cue"):
            build_replay_groups(dict(fit=damaged))
        with self.assertRaisesRegex(ValueError, "complete banks"):
            sample_replay_group({}, random.Random(1))
        for count in (-1, True):
            with self.assertRaisesRegex(ValueError, "batch size"):
                sample_replay_queries(fit_bank(), random.Random(1), count)


if __name__ == "__main__":
    unittest.main()
