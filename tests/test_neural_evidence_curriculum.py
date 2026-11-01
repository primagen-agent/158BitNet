"""Bounded fit evidence follows native text format and preserves query targets."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_reader_evidence_curriculum import (
    augment_evidence, augment_relation_paraphrases, filter_bank_evidence_queries, split_evidence_views,
)


class EvidenceCurriculumContract(unittest.TestCase):
    def group(self):
        facts = [
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="collection", kind=0),
            dict(text="Ada. I collect glass beads.", entity="Ada", prop="collection", kind=0),
            dict(text="Ada. I grow sage.", entity="Ada", prop="gardening", kind=0),
            dict(text="Ada. I repair bicycles.", entity="Ada", prop="job", kind=0),
            dict(text="Ben. I collect railway badges.", entity="Ben", prop="collection", kind=0),
            dict(text="Ben. I live in York.", entity="Ben", prop="city", kind=0),
        ]
        positive = dict(text="What does Ada collect?", entities=["Ada"],
                        prop="collection", kind=0,
                        support_texts=[facts[0]["text"], facts[1]["text"]])
        negative = dict(text="Which instrument does Ada play?", entities=["Ada"],
                        prop="", kind=0, support_texts=[])
        return dict(facts=facts, queries=[positive, negative])

    def test_one_to_four_mixed_cues_keep_original_anchor_and_exact_targets(self):
        group = self.group()
        original = deepcopy(group)
        result = augment_evidence(group, 2107)
        self.assertEqual(group, original)
        originals = {q["text"]: q for q in original["queries"]}
        added = [q for q in result["queries"] if q.get("augmented")]
        self.assertEqual({q["evidence_count"] for q in added}, {1, 2, 3, 4})
        self.assertEqual(
            {q["evidence_mix"] for q in added},
            {"single", "support_and_noise", "noise_only", "support_only"},
        )
        known = {f["text"] for f in group["facts"]}
        for query in added:
            initial = originals[query["initial_text"]]
            cues = query["evidence_sources"]
            self.assertEqual(query["evidence_count"], len(cues))
            self.assertEqual(len(cues), len(set(cues)))
            self.assertTrue(set(cues) <= known)
            self.assertEqual(
                query["text"],
                initial["text"] + "\nActivated evidence:\n" + "\n".join(cues) + "\n",
            )
            self.assertEqual(query["support_texts"], initial["support_texts"])
            self.assertEqual(query["entities"], initial["entities"])
            self.assertEqual(query["prop"], initial["prop"])
            supports = set(initial["support_texts"])
            if query["evidence_mix"] == "support_and_noise":
                self.assertTrue(supports & set(cues))
                self.assertTrue(set(cues) - supports)
            elif query["evidence_mix"] == "noise_only":
                self.assertFalse(supports & set(cues))
            elif query["evidence_mix"] == "support_only":
                self.assertTrue(set(cues) <= supports)
            self.assertNotIn("reference", query)

    def test_budget_seed_and_nonrecursive_expansion_are_stable(self):
        group = self.group()
        first = augment_evidence(group, 2107)
        self.assertEqual(first, augment_evidence(group, 2107))
        self.assertEqual(first, augment_evidence(first, 2107))
        for limit in (0, 1, 4, 17, 601):
            bounded = augment_evidence(group, 2107, max_queries=limit)
            self.assertLessEqual(len(bounded["queries"]) - 2, min(limit, 600))
            self.assertEqual(bounded["queries"][:2], group["queries"])
        views = split_evidence_views(first)
        self.assertEqual(views["base"], group)
        self.assertTrue(all(q["augmented"] for q in views["augmented"]["queries"]))

    def test_negative_question_never_receives_supporting_cue_labels(self):
        result = augment_evidence(self.group(), 2107)
        negatives = [q for q in result["queries"] if q.get("augmented") and not q["prop"]]
        self.assertTrue(negatives)
        self.assertEqual({q["evidence_count"] for q in negatives}, {1, 2, 3, 4})
        self.assertTrue(all(q["support_texts"] == [] for q in negatives))
        self.assertTrue(all(q["evidence_mix"] in {"single", "noise_only"} for q in negatives))

    def test_held_out_records_are_opaque_and_unchanged(self):
        locked = dict(facts=[dict(opaque="locked")], queries=[dict(opaque="locked")])
        for split in ("development", "blind"):
            self.assertEqual(augment_evidence(locked, 2107, split=split), locked)

    def test_frozen_planner_reference_survives_cues_without_gold_replacement(self):
        group = self.group()
        # Deliberately distinct from the author entity label: the helper must
        # preserve the planner output even when it disagrees with the teacher.
        group["queries"][0]["reference"] = "NN predicted Dana"
        original = deepcopy(group)
        result = augment_evidence(group, 2107)
        self.assertEqual(group, original)
        self.assertEqual(result["queries"][:2], original["queries"])
        self.assertNotIn("initial_reference", group["queries"][0])
        referenced = [q for q in result["queries"] if q.get("augmented") and q["prop"]]
        self.assertTrue(referenced)
        self.assertTrue(any(q["evidence_type"] == "other_actor" for q in referenced))
        for query in referenced:
            self.assertEqual(query["initial_reference"], "NN predicted Dana")
            query.pop("reference", None)  # Match trainer transport cleanup.
            self.assertEqual(query["initial_reference"], "NN predicted Dana")
            self.assertNotIn("reference", query)
        unreferenced = [q for q in result["queries"] if q.get("augmented") and not q["prop"]]
        self.assertTrue(unreferenced)
        self.assertTrue(all("initial_reference" not in q for q in unreferenced))
        for split in ("development", "blind"):
            self.assertEqual(augment_evidence(group, 2107, split=split), original)

    def test_current_world_rejects_phantom_positive_and_absent_source_cues(self):
        from prepare_neural_reader_expansion import reader_expansion
        group = augment_evidence(reader_expansion(), 5322, 400)
        bank = [fact for fact in group["facts"] if fact["pattern_id"] in
                {"reader-expansion-1", "reader-expansion-6", "reader-expansion-10",
                 "reader-expansion-15", "reader-expansion-20", "reader-expansion-25",
                 "reader-expansion-30", "reader-expansion-35", "reader-expansion-40",
                 "reader-expansion-45", "reader-expansion-50", "reader-expansion-55"}]
        present = {fact["text"] for fact in bank}
        augmented = [row for row in group["queries"] if row.get("augmented")]
        phantom_positive = next(row for row in augmented
                                if present.intersection(row["support_texts"])
                                and not set(row["evidence_sources"]) <= present)
        phantom_absent = next(row for row in augmented
                              if row["prop"] and not present.intersection(row["support_texts"])
                              and set(row["evidence_sources"]).intersection(row["support_texts"]))
        before = deepcopy(group)
        retained = filter_bank_evidence_queries(group["queries"], bank)
        self.assertNotIn(phantom_positive, retained)
        self.assertNotIn(phantom_absent, retained)
        self.assertEqual(group, before)
        raw = [row for row in group["queries"] if not row.get("augmented")]
        self.assertEqual(retained[:len(raw)], raw)
        self.assertTrue(all(set(row.get("evidence_sources", [])) <= present for row in retained))

    def test_valid_multi_cues_and_raw_empty_cues_preserve_complete_targets(self):
        group = self.group()
        group["queries"][0]["reference"] = "NN predicted Dana"
        result = augment_evidence(group, 2107)
        multi = next(row for row in result["queries"]
                     if row.get("evidence_mix") == "support_and_noise" and row["evidence_count"] == 4)
        raw = dict(group["queries"][1], evidence_sources=[], evidence_count=0)
        rows = [multi, raw]
        before = deepcopy(rows)
        retained = filter_bank_evidence_queries(rows, group["facts"])
        self.assertEqual(retained, before)
        self.assertIs(retained[0], multi)
        self.assertIs(retained[1], raw)
        self.assertEqual(retained[0]["support_texts"], group["queries"][0]["support_texts"])
        self.assertEqual(retained[0]["initial_reference"], "NN predicted Dana")
        self.assertEqual(filter_bank_evidence_queries([raw], []), [raw])

    def test_evidence_provenance_rejects_missing_or_inconsistent_metadata(self):
        group = self.group()
        multi = next(row for row in augment_evidence(group, 2107)["queries"]
                     if row.get("evidence_count") == 4)
        invalid = []
        for field in ("initial_text", "evidence_sources", "evidence_count"):
            row = deepcopy(multi)
            row.pop(field)
            invalid.append(row)
        invalid.extend((dict(multi, evidence_count=3), dict(multi, text=multi["text"] + "tampered"),
                        dict(multi, evidence_sources=[multi["evidence_sources"][0]] * 4),
                        dict(multi, augmented=False), dict(multi, augmented=1)))
        for row in invalid:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "evidence|provenance|augmented"):
                    filter_bank_evidence_queries([row], group["facts"])


class RelationParaphraseContract(unittest.TestCase):
    def group(self):
        from neural_activation_curriculum import relation_variation_fit_group
        return relation_variation_fit_group()

    def test_two_supported_families_keep_every_original_source_and_question(self):
        group = self.group()
        before = deepcopy(group)
        result = augment_relation_paraphrases(group)
        self.assertEqual(group, before)
        self.assertEqual(result["facts"], before["facts"])
        self.assertEqual(result["queries"][:48], before["queries"])
        self.assertEqual(len(result["queries"]), 144)
        self.assertEqual(len({r["text"] for r in result["queries"]}), 144)
        originals = {row["support_texts"][0]: row for row in before["queries"]}
        by_source = {}
        identifiers = set()
        for row in result["queries"][48:]:
            self.assertEqual(len(row["support_texts"]), 1)
            original = originals[row["support_texts"][0]]
            for key in ("entities", "prop", "kind", "allowed_kinds",
                        "semantic_specific", "relation_variation", "pattern_id"):
                self.assertEqual(row[key], original[key])
            self.assertTrue(row["relation_paraphrase"])
            self.assertEqual(row["query_class"], "explicit")
            self.assertEqual(row["fit_paraphrase_id"],
                             row["pattern_id"] + "::" + row["question_family"])
            self.assertIn(row["entities"][0], row["text"])
            self.assertFalse(row["text"].startswith(row["entities"][0] + ". "))
            self.assertNotIn("reference", row)
            self.assertNotIn("initial_reference", row)
            self.assertNotIn("Activated evidence:", row["text"])
            by_source.setdefault(row["support_texts"][0], set()).add(row["question_family"])
            identifiers.add(row["observation_id"])
        self.assertEqual(len(identifiers), 96)
        self.assertEqual(len(by_source), 48)
        self.assertEqual(set(map(len, by_source.values())), {2})
        result["facts"][0]["text"] = "changed result"
        result["queries"][48]["support_texts"].clear()
        self.assertEqual(group, before)

    def test_supported_frequency_condition_and_causes_are_authored_aspects(self):
        result = augment_relation_paraphrases(self.group())
        rows = [row for row in result["queries"] if row.get("relation_paraphrase")]
        facts = {fact["text"]: fact for fact in result["facts"]}
        for row in rows:
            index = int(row["pattern_id"].rsplit("-", 1)[1])
            content = facts[row["support_texts"][0]]["content"]
            if index == 4 and row["question_family"] == "frequency":
                self.assertIn("twice a week", content)
            if index == 4 and row["question_family"] == "condition":
                self.assertIn("top layer feels dry", content)
            if index == 5 and row["question_family"] == "cause":
                self.assertIn("so the soil drains", content)
            if index == 12 and row["question_family"] == "cause":
                self.assertIn("freezes the motion", content)
            if index == 13 and row["question_family"] == "duration":
                self.assertIn("thirty seconds", content)
            if index == 15 and row["question_family"] == "cause":
                self.assertIn("a local failure will not lose both copies", content)
        self.assertTrue({"frequency", "condition", "cause", "duration"}
                        <= {row["question_family"] for row in rows})

    def test_specific_questions_keep_same_person_same_property_alternative_negative(self):
        import torch
        from neural_fit_support import semantic_relation_masks
        from train_neural_dialogue_reader import supported
        result = augment_relation_paraphrases(self.group())
        facts = result["facts"]
        for row in result["queries"][48:]:
            target = row["support_texts"][0]
            alternatives = [i for i, fact in enumerate(facts)
                            if fact["entity"] == row["entities"][0]
                            and fact["prop"] == row["prop"] and fact["text"] != target]
            self.assertEqual(len(alternatives), 1)
            hardness = torch.zeros(1, len(facts))
            hardness[0, alternatives[0]] = 100
            positive, negative = semantic_relation_masks([row], facts, hardness)
            self.assertEqual(int(positive.sum()), 3)
            self.assertTrue(negative[0, alternatives[0]])
            self.assertEqual(int(negative.sum()), 4)
            self.assertEqual(sum(supported(row, fact) for fact in facts), 1)
            for _, index in negative.nonzero():
                self.assertEqual(facts[index]["entity"], row["entities"][0])

    def test_balanced_sampler_visits_original_and_both_paraphrase_families(self):
        import random
        from neural_fit_support import sample_relation_queries
        result = augment_relation_paraphrases(self.group())
        rows = result["queries"]
        expected = {(r["pattern_id"], r.get("fit_paraphrase_id")) for r in rows}
        selected = sample_relation_queries(rows, random.Random(4261), 48)
        self.assertEqual(len(expected), 48)
        self.assertEqual({(r["pattern_id"], r.get("fit_paraphrase_id")) for r in selected}, expected)
        self.assertEqual(selected, sample_relation_queries(rows, random.Random(4261), 48))

    def test_held_out_and_unrelated_rows_are_copied_unchanged(self):
        opaque = dict(facts=[dict(opaque="locked")], queries=[dict(opaque="locked")])
        for split in ("development", "blind", "sealed"):
            result = augment_relation_paraphrases(opaque, split=split)
            self.assertEqual(result, opaque)
            self.assertIsNot(result, opaque)
        group = self.group()
        group["facts"].append(dict(text="Other source", unrelated=True))
        group["queries"].append(dict(text="Other question", unrelated=True))
        before = deepcopy(group)
        result = augment_relation_paraphrases(group)
        self.assertEqual(result["facts"], before["facts"])
        self.assertEqual(result["queries"][:49], before["queries"])
        self.assertEqual(group, before)

    def test_duplicates_rebinding_and_mismatched_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            augment_relation_paraphrases(augment_relation_paraphrases(self.group()))
        group = self.group()
        group["queries"].append(deepcopy(group["queries"][0]))
        with self.assertRaises(ValueError):
            augment_relation_paraphrases(group)
        group = self.group()
        group["facts"][0]["content"] += " Unsupported additional claim."
        with self.assertRaises(ValueError):
            augment_relation_paraphrases(group)
        group = self.group()
        group["queries"][0]["support_texts"] = [group["facts"][1]["text"]]
        with self.assertRaises(ValueError):
            augment_relation_paraphrases(group)
        group = self.group()
        group["queries"][0]["reference"] = "Injected subject"
        with self.assertRaises(ValueError):
            augment_relation_paraphrases(group)
        group = self.group()
        group["facts"].pop()
        with self.assertRaises(ValueError):
            augment_relation_paraphrases(group)


if __name__ == "__main__":
    unittest.main()
