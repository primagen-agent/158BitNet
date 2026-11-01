"""Fit teacher coherence without changing frozen development observations."""

import hashlib
import json
from pathlib import Path
import sys
import unittest
import random

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_fit_support import (
    add_fit_relation_queries,
    repair_fit_episode_support,
    semantic_relation_masks,
    sample_relation_queries,
    episode_relation_variation_fit_group,
    episode_aspect_variation_fit_group,
)
from prepare_neural_colloquial import reader_data
from prepare_neural_episodes import NAMES, PATTERNS, episode_curriculum
from prepare_neural_reader_expansion import (FIT_ACTORS, NULL_QUESTIONS, reader_expansion,
                                           sample_expansion_world, COEXISTENCE_PROP,
                                           validate as validate_expansion)
from train_neural_dialogue_reader import (
    add_episodes,
    episode_observation_group,
    supported,
)


def raw_episode_group(split):
    group = add_episodes(dict(facts=[], queries=[]), split)
    observations = {
        row["body"]: row
        for row in episode_curriculum(split)
        if row["tag"] == "episode-query"
    }
    for query in group["queries"]:
        row = observations[query["text"]]
        query.update(
            observation=dict(row),
            observation_id=row["source_id"],
            query_class="prior_episode",
        )
    return group


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class FitSupportContract(unittest.TestCase):
    def test_sampler_covers_rare_relation_before_many_common_families(self):
        common = [dict(text=str(i), prop="common", fit_paraphrase_id=str(i)) for i in range(80)]
        rare = dict(text="rare", prop="rare", query_class="self")
        batch = sample_relation_queries(common + [rare], random.Random(314), 12)
        self.assertIn(rare, batch)
        self.assertEqual(len({id(row) for row in batch}), 12)
        families = [dict(text=f"{i}-{j}", prop="p", query_class=str(i))
                    for i in range(4) for j in range(15)]
        selected = sample_relation_queries(families, random.Random(317), 4)
        self.assertEqual({r["query_class"] for r in selected}, {str(i) for i in range(4)})

    def test_episode_content_variants_keep_broad_and_specific_teachers_consistent(self):
        group = episode_relation_variation_fit_group()
        self.assertEqual(len(group["facts"]), 33)
        self.assertEqual(len(group["queries"]), 39)
        for query in group["queries"]:
            expected = set(query["support_texts"])
            self.assertEqual({f["text"] for f in group["facts"] if supported(query, f)}, expected)
            self.assertEqual(len(expected), 1 if query.get("semantic_specific") else 4)
            positive, negative = semantic_relation_masks([query], group["facts"],
                                                         torch.zeros(1, len(group["facts"])))
            # Three actors share each authored content variant; a broad query
            # covers all four variants of its relation.
            self.assertEqual(int(positive.sum()), 3 if query.get("semantic_specific") else 12)
            for index in negative[0].nonzero().flatten().tolist():
                fact = group["facts"][index]
                self.assertIn(fact["entity"], query["entities"])
                self.assertNotIn(fact["text"], expected)

    def test_episode_aspects_cover_every_fit_relation_with_one_source(self):
        group = episode_aspect_variation_fit_group()
        names = tuple(NAMES["fit"][:4])
        props = {entry[0] for entry in PATTERNS["fit"]}
        self.assertEqual(len(props), 12)
        self.assertEqual(len(group["facts"]), len(names) * len(props))
        self.assertEqual(len(group["queries"]), len(names) * len(props) * 4)
        self.assertEqual(len({f["text"] for f in group["facts"]}), len(group["facts"]))
        self.assertEqual(len({q["text"] for q in group["queries"]}), len(group["queries"]))
        names_only = set(NAMES["development"]) | set(NAMES["sealed"])
        evaluation_text = {
            entry[index]
            for split in ("development", "sealed")
            for entry in PATTERNS[split]
            for index in (1, 2, 3)
        }
        for fact in group["facts"]:
            self.assertIn(fact["entity"], names)
            self.assertTrue(fact["text"].startswith(fact["entity"] + ". "))
            self.assertTrue(all(name not in fact["text"] for name in names_only))
        for query in group["queries"]:
            self.assertEqual(len(query["support_texts"]), 1)
            supported_facts = [f for f in group["facts"] if supported(query, f)]
            self.assertEqual([f["text"] for f in supported_facts], query["support_texts"])
            self.assertTrue(all(name not in query["text"] for name in names_only))
            shape = query["text"].replace(query["entities"][0], "{name}")
            self.assertNotIn(shape, evaluation_text)
        self.assertEqual(group, episode_aspect_variation_fit_group())

    def test_broad_questions_support_every_related_fact_and_no_other_actor(self):
        group = episode_observation_group("fit")
        queries = {row["text"]: row for row in group["queries"]}
        for question, prop in (
            ("What does Cyra collect?", "collection"),
            ("What is Cyra's collecting hobby?", "collection"),
            ("What helps Cyra relax?", "relaxation"),
            ("What is Cyra's usual way of resting?", "relaxation"),
        ):
            with self.subTest(question=question):
                query = queries[question]
                predicted_labels = {
                    fact["text"] for fact in group["facts"] if supported(query, fact)
                }
                expected = {
                    fact["text"]
                    for fact in group["facts"]
                    if fact["entity"] == "Cyra"
                    and fact["prop"] == prop
                    and fact["kind"] == 0
                }
                self.assertEqual(len(expected), 2)
                self.assertEqual(predicted_labels, expected)
                historical = dict(
                    text="historical alternative", entity="Cyra", prop=prop, kind=1
                )
                self.assertFalse(supported(query, historical))

    def test_specific_questions_keep_exact_single_support(self):
        group = episode_observation_group("fit")
        queries = {row["text"]: row for row in group["queries"]}
        for question, required_fragment in (
            ("Where does Cyra keep the postcard collection?", "green album"),
            ("How does Cyra sort the stamps?", "country they came from"),
            ("What does Cyra drink while doing crosswords?", "mint tea"),
            ("How does Cyra unwind after a demanding day?", "chamber music"),
            ("Why did Cyra organize a repair club?", "share tools"),
        ):
            with self.subTest(question=question):
                query = queries[question]
                self.assertEqual(len(query["support_texts"]), 1)
                self.assertIn(required_fragment, query["support_texts"][0])

    def test_rewritten_questions_remain_grounded_raw_named_observations(self):
        old = raw_episode_group("fit")
        old_queries = {row["text"]: row for row in old["queries"]}
        new = episode_observation_group("fit")
        new_queries = {row["text"]: row for row in new["queries"]}
        for original, rewritten in (
            (
                "Why did Cyra change occupations?",
                "Why did Cyra switch from repairing radios to restoring clocks?",
            ),
            (
                "What work did Cyra switch to?",
                "What work did Cyra switch to after repairing radios?",
            ),
            (
                "How does Cyra earn a living now?",
                "What work pays Cyra's bills after leaving warehouse packing?",
            ),
            (
                "What motivated Cyra's neighborhood project?",
                "What motivated Cyra to organize the neighborhood repair club?",
            ),
        ):
            with self.subTest(question=rewritten):
                self.assertNotIn(original, new_queries)
                query = new_queries[rewritten]
                previous = old_queries[original]
                self.assertEqual(query["support_texts"], previous["support_texts"])
                self.assertEqual(len(query["support_texts"]), 1)
                self.assertEqual(query["observation"]["body"], rewritten)
                self.assertEqual(query["observation"]["context"], previous["observation"]["context"])
                self.assertFalse(query["observation"]["whole_need"])
                self.assertEqual(query["observation"]["whole_refs"], [])
                self.assertNotIn("reference", query)

    def test_facts_and_unmodified_queries_are_preserved_without_mutation(self):
        original = raw_episode_group("fit")
        before = digest(original)
        repaired = repair_fit_episode_support(original, "fit")
        self.assertEqual(digest(original), before)
        self.assertEqual(repaired["facts"], original["facts"])
        previous = {q["text"]: q for q in original["queries"]}
        for query in repaired["queries"]:
            if "fit_support_revision" not in query:
                original_fields = {k: v for k, v in query.items() if k != "semantic_specific"}
                self.assertEqual(original_fields, previous[query["text"]])

    def test_semantic_targets_transfer_content_without_changing_actor_gold(self):
        facts = [
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="collection", kind=0),
            dict(text="Ben. I collect copper pins.", entity="Ben", prop="collection", kind=0),
            dict(text="Ada. I collect glass beads.", entity="Ada", prop="collection", kind=0),
            dict(text="Ada. I grow sage.", entity="Ada", prop="gardening", kind=0),
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="collection", kind=1),
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="possession", kind=0),
        ]
        row = dict(entities=["Ada"], prop="collection", kind=0,
                   support_texts=[facts[0]["text"]], semantic_specific=True)
        # Identical text alone is insufficient: property and temporal kind
        # remain part of the complete semantic equivalence key.
        before = digest(row)
        positive, negative = semantic_relation_masks([row], facts, torch.zeros(1, 6))
        self.assertEqual(positive.nonzero().tolist(), [[0, 0], [0, 1]])
        self.assertEqual(negative.nonzero().tolist(), [[0, 2], [0, 3], [0, 5]])
        self.assertTrue(supported(row, facts[0]))
        self.assertFalse(supported(row, facts[1]))
        self.assertEqual(digest(row), before)

    def test_broad_and_unannotated_alternatives_are_not_semantic_negatives(self):
        facts = [
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="collection", kind=0),
            dict(text="Ada. I collect glass beads.", entity="Ada", prop="collection", kind=0),
            dict(text="Ben. I collect glass beads.", entity="Ben", prop="collection", kind=0),
            dict(text="Ada. I grow sage.", entity="Ada", prop="gardening", kind=0),
        ]
        broad = dict(entities=["Ada"], prop="collection", kind=0,
                     support_texts=[facts[0]["text"], facts[1]["text"]])
        positive, negative = semantic_relation_masks([broad], facts, torch.ones(1, 4))
        self.assertEqual(positive.nonzero().tolist(), [[0, 0], [0, 1], [0, 2]])
        self.assertEqual(negative.nonzero().tolist(), [[0, 3]])
        broad["support_texts"] = [facts[0]["text"]]
        positive, negative = semantic_relation_masks([broad], facts, torch.ones(1, 4))
        self.assertEqual(positive.nonzero().tolist(), [[0, 0]])
        self.assertEqual(negative.nonzero().tolist(), [[0, 3]])

    def test_specific_history_uses_source_kind_for_object_negatives(self):
        facts = [
            dict(text="Ada. I keep chisels in a chest.", entity="Ada", prop="storage", kind=0, status=1),
            dict(text="Ben. I keep chisels in a chest.", entity="Ben", prop="storage", kind=0),
            dict(text="Ada. I keep chisels in a crate.", entity="Ada", prop="storage", kind=0),
            dict(text="Ada. I kept chisels in a cabinet.", entity="Ada", prop="storage", kind=1),
            dict(text="Ben. I keep chisels in a crate.", entity="Ben", prop="storage", kind=0),
        ]
        row = dict(entities=["Ada"], prop="storage", kind=1, source_kinds=[0],
                   support_texts=[facts[0]["text"]], semantic_specific=True)
        before = digest((row, facts))
        # A high-scoring past-tense source is outside the original-text kind
        # teacher, although the question requests a historical physical slot.
        positive, negative = semantic_relation_masks([row], facts,
                                                     torch.tensor([[0., 0., 1., 100., 100.]]))
        self.assertEqual(positive.nonzero().tolist(), [[0, 0], [0, 1]])
        self.assertEqual(negative.nonzero().tolist(), [[0, 2]])
        self.assertEqual(digest((row, facts)), before)

    def test_explicit_fact_source_kind_drives_exact_transfer_and_broad_support(self):
        facts = [
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="collection",
                 kind=1, source_kind=0, status=1),
            dict(text="Ben. I collect copper pins.", entity="Ben", prop="collection",
                 kind=2, source_kind=0, status=3),
            dict(text="Ada. I collect glass beads.", entity="Ada", prop="collection",
                 kind=1, source_kind=0, status=1),
            dict(text="Ada. I collect glass beads.", entity="Ada", prop="collection",
                 kind=0, source_kind=1, status=0),
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="collection",
                 kind=0, source_kind=1, status=0),
            dict(text="Ada. I grow sage.", entity="Ada", prop="gardening", kind=0),
        ]
        exact = dict(entities=["Ada"], prop="collection", kind=1, source_kinds=[0],
                     support_texts=[facts[0]["text"]], semantic_specific=True)
        broad = dict(entities=["Ada"], prop="collection", kind=1, source_kinds=[0])
        before = digest((facts, exact, broad))
        positive, negative = semantic_relation_masks([exact, broad], facts, torch.zeros(2, 6))
        self.assertEqual(positive.nonzero().tolist(), [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])
        self.assertEqual(negative.nonzero().tolist(), [[0, 2], [0, 5], [1, 5]])
        self.assertEqual(digest((facts, exact, broad)), before)

    def test_specific_absent_source_kind_does_not_label_other_kinds_negative(self):
        facts = [
            dict(text="Ada. I keep chisels in a crate.", entity="Ada", prop="storage", kind=0),
            dict(text="Ada. I kept chisels in a cabinet.", entity="Ada", prop="storage", kind=1),
        ]
        row = dict(entities=["Ada"], prop="storage", kind=1, source_kinds=[2],
                   support_texts=["Ada. I plan to keep chisels in a chest."], semantic_specific=True)
        positive, negative = semantic_relation_masks([row], facts, torch.tensor([[10., 20.]]))
        self.assertFalse(positive.any())
        self.assertFalse(negative.any())

    def test_semantic_source_kind_teacher_rejects_invalid_overrides(self):
        row = dict(entities=["Ada"], prop="storage", kind=0)
        for source_kind in (True, -1, 3, "0", None):
            with self.subTest(source_kind=source_kind):
                fact = dict(text="Ada. I keep chisels in a chest.", entity="Ada", prop="storage",
                            kind=0, source_kind=source_kind)
                with self.assertRaisesRegex(ValueError, "original source kinds"):
                    semantic_relation_masks([row], [fact], torch.zeros(1, 1))

    def test_absent_paired_source_can_transfer_only_its_complete_content(self):
        facts = [
            dict(text="Ada. I collect glass beads.", entity="Ada", prop="collection", kind=0),
            dict(text="Ben. I collect copper pins.", entity="Ben", prop="collection", kind=0),
        ]
        row = dict(entities=["Ada"], prop="collection", kind=0,
                   support_texts=["Ada. I collect copper pins."])
        positive, negative = semantic_relation_masks([row], facts, torch.zeros(1, 2))
        self.assertEqual(positive.nonzero().tolist(), [[0, 1]])
        self.assertFalse(negative.any())
        self.assertFalse(any(supported(row, fact) for fact in facts))

    def test_semantic_negatives_are_bounded_and_do_not_select_another_actor(self):
        facts = [dict(text="Ada. Source.", entity="Ada", prop="a", kind=0)]
        facts += [dict(text=f"Ada. Other {i}.", entity="Ada", prop="b", kind=0) for i in range(6)]
        facts += [dict(text="Ben. Distractor.", entity="Ben", prop="b", kind=0)]
        row = dict(entities=["Ada"], prop="a", kind=0, support_texts=[facts[0]["text"]])
        hardness = torch.tensor([[0., 1., 2., 3., 4., 5., 6., 100.]])
        _, negative = semantic_relation_masks([row], facts, hardness)
        self.assertEqual(negative.nonzero().tolist(), [[0, 3], [0, 4], [0, 5], [0, 6]])
        with self.assertRaises(ValueError):
            semantic_relation_masks([row], facts, hardness, max_negatives=5)
        with self.assertRaises(ValueError):
            semantic_relation_masks([row], facts, hardness[:, :-1])

    def test_missing_support_key_covers_authored_multi_person_broad_queries(self):
        group = episode_observation_group("fit")
        rows = [q for q in group["queries"] if q["prop"] and "support_texts" not in q]
        self.assertEqual(len(rows), 16)
        before = digest(rows)
        facts = group["facts"]
        gold = torch.tensor([[supported(q, fact) for fact in facts] for q in rows])
        positive, _ = semantic_relation_masks(rows, facts, torch.zeros(len(rows), len(facts)))
        self.assertEqual(gold.sum(-1).tolist(), [4] * 16)
        self.assertEqual(positive.sum(-1).tolist(), [32] * 16)
        self.assertTrue((positive & gold).sum() == gold.sum())
        for i, j in (positive & ~gold).nonzero():
            self.assertNotIn(facts[int(j)]["entity"], rows[int(i)]["entities"])
        self.assertEqual(digest(rows), before)
        self.assertEqual(
            gold.tolist(),
            [[supported(q, fact) for fact in facts] for q in rows],
        )

    def test_explicit_empty_support_and_empty_property_do_not_take_fallback(self):
        facts = [
            dict(text="Ada. I collect copper pins.", entity="Ada", prop="collection", kind=0),
            dict(text="Ben. I collect copper pins.", entity="Ben", prop="collection", kind=0),
        ]
        rows = [
            dict(entities=["Ada"], prop="collection", kind=0, support_texts=[]),
            dict(entities=["Ada"], prop="", kind=0),
        ]
        positive, _ = semantic_relation_masks(rows, facts, torch.zeros(2, 2))
        self.assertFalse(positive.any())
        self.assertFalse(any(supported(row, fact) for row in rows for fact in facts))

    def test_complete_episode_development_hash_is_unchanged(self):
        # Locked before the fit repair: all 48 facts and 108 labeled observations.
        group = episode_observation_group("development")
        self.assertEqual(
            digest(group),
            "f31dcbecf14891861ec2eb73b3e9a8b6c5ed7a43195b9a1b450e1182c90820a3",
        )
        self.assertEqual(group, raw_episode_group("development"))
        self.assertEqual(repair_fit_episode_support(group, "development"), group)
        self.assertEqual(add_fit_relation_queries(group, "development"), group)
        opaque_blind = dict(facts=[dict(label="locked")], queries=[dict(label="locked")])
        self.assertEqual(repair_fit_episode_support(opaque_blind, "blind"), opaque_blind)
        self.assertEqual(add_fit_relation_queries(opaque_blind, "blind"), opaque_blind)

    def test_paraphrases_are_bounded_explicit_and_have_complete_fit_support(self):
        original = reader_data("fit")
        before = digest(original)
        expanded = add_fit_relation_queries(original, "fit")
        self.assertEqual(digest(original), before)
        self.assertEqual(expanded["facts"], original["facts"])
        extra = [q for q in expanded["queries"] if "fit_paraphrase_id" in q]
        self.assertEqual(len(extra), 288)
        by_relation = {}
        for query in extra:
            by_relation.setdefault(query["prop"], set()).add(query["fit_paraphrase_id"])
            observation = query["observation"]
            self.assertEqual(observation["body"], query["text"])
            self.assertIn(query["entities"][0], query["text"])
            self.assertEqual(observation["past"], "")
            self.assertFalse(observation["whole_need"])
            self.assertEqual(observation["whole_refs"], [])
            self.assertEqual(observation["whole_ref_ranges"], [])
            self.assertNotIn("reference", query)
            expected = {
                fact["text"]
                for fact in original["facts"]
                if fact["entity"] in query["entities"]
                and fact["prop"] == query["prop"]
                and fact["kind"] == query["kind"]
            }
            self.assertEqual(set(query["support_texts"]), expected)
        self.assertEqual(len(by_relation), 8)
        self.assertEqual({len(patterns) for patterns in by_relation.values()}, {6})
        self.assertEqual(add_fit_relation_queries(expanded, "fit"), expanded)
        capped = add_fit_relation_queries(original, "fit", max_queries=11)
        self.assertEqual(len(capped["queries"]) - len(original["queries"]), 11)
        with self.assertRaises(ValueError):
            add_fit_relation_queries(original, "fit", max_queries=401)

    def test_colloquial_development_is_unchanged(self):
        original = reader_data("development")
        before = digest(original)
        self.assertEqual(digest(add_fit_relation_queries(original, "development")), before)
        self.assertEqual(digest(original), before)

    def test_expansion_alternative_worlds_keep_specific_fact_labels(self):
        group = reader_expansion()
        before = digest(group)
        # The original source-specific expansion questions retain their labels.
        # Material worlds may now contain zero/one/two compatible sources. The
        # object-specific questions are outside the episode repair fixtures.
        self.assertEqual(repair_fit_episode_support(group, "fit"), group)
        self.assertEqual(add_fit_relation_queries(group, "fit"), group)
        self.assertEqual(digest(group), before)

    def test_expansion_source_specific_labels_preserve_raw_and_null_questions(self):
        group = reader_expansion()
        positive = [row for row in group["queries"] if row["support_texts"]
                    and "expansion_query_class" not in row]
        null = [row for row in group["queries"] if not row["prop"]]
        self.assertEqual(len(positive), 1440)
        self.assertTrue(all(row.get("semantic_specific") is True for row in positive))
        self.assertTrue(all("reference" not in row and "observation" not in row
                            for row in group["queries"]))
        self.assertEqual(null, [dict(text=question.format(name=name), entities=[name],
                                    prop="", kind=0, support_texts=[], expansion=True)
                                for name in FIT_ACTORS for question in NULL_QUESTIONS])
        self.assertEqual(len(null), 64)
        cork_questions = [row for row in positive if row["text"].startswith("What benefits does cork provide")]
        self.assertEqual([row["text"] for row in cork_questions],
                         [f"What benefits does cork provide in {name}'s model railway?" for name in FIT_ACTORS])
        self.assertTrue(all("easy to shape and absorbs the vibration" in row["support_texts"][0]
                            for row in cork_questions))
        self.assertFalse(any("avoid a rigid base" in row["text"] for row in group["queries"]))

    def test_expansion_absent_world_reserves_same_topic_semantic_negative(self):
        group = reader_expansion()
        worlds = {}
        for fact in group["facts"]:
            worlds.setdefault(tuple(fact["bank_group"]), []).append(fact)
        bank = [alternatives[1] for alternatives in worlds.values()]
        row = next(query for query in group["queries"]
                   if query["text"] == "What occupation did Cyra take up after hotel reception?")
        courier = next(index for index, fact in enumerate(bank)
                       if fact["entity"] == "Cyra" and fact["prop"] == "work_change")
        self.assertIn("parcel courier", bank[courier]["text"])
        self.assertFalse(any(supported(row, fact) for fact in bank))
        hardness = torch.full((1, len(bank)), 100.0)
        hardness[0, courier] = -10.0
        positive, negative = semantic_relation_masks([row], bank, hardness)
        self.assertFalse(positive.any())
        self.assertTrue(negative[0, courier])
        self.assertEqual(int(negative.sum()), 4)
        self.assertTrue(all(bank[int(index)]["entity"] == "Cyra" for _, index in negative.nonzero()))
        unlabeled = {key: value for key, value in row.items() if key != "semantic_specific"}
        self.assertFalse(semantic_relation_masks([unlabeled], bank, hardness)[1][0, courier])

    def test_material_world_source_broad_and_empty_teachers_keep_complete_semantics(self):
        group = reader_expansion()
        counts = {(name, COEXISTENCE_PROP): 2 for name in FIT_ACTORS}
        world = sample_expansion_world(group, random.Random(719), source_counts=counts)
        bank = world["facts"]
        selected = [(index, fact) for index, fact in enumerate(bank)
                    if fact["entity"] == "Cyra" and fact["prop"] == COEXISTENCE_PROP]
        first, second = selected
        specific = next(row for row in world["queries"] if row["entities"] == ["Cyra"]
                        and row["support_texts"] == [first[1]["text"]])
        broad = next(row for row in world["queries"] if row["entities"] == ["Cyra"]
                     and row.get("expansion_query_class") == "broad")
        unsupported = next(row for row in world["queries"] if row["entities"] == ["Cyra"]
                           and row.get("expansion_query_class") == "unsupported")
        hardness = torch.full((3, len(bank)), 100.)
        hardness[:, second[0]] = -100.
        positive, negative = semantic_relation_masks([specific, broad, unsupported], bank, hardness)
        self.assertTrue(positive[0, first[0]])
        self.assertFalse(positive[0, second[0]])
        self.assertTrue(negative[0, second[0]])
        self.assertTrue(positive[1, [first[0], second[0]]].all())
        self.assertFalse(negative[1, [first[0], second[0]]].any())
        self.assertFalse(positive[2].any())
        self.assertTrue(negative[2, [first[0], second[0]]].any())
        self.assertEqual([fact["text"] for fact in bank if supported(broad, fact)],
                         [fact["text"] for _, fact in selected])
        empty = sample_expansion_world(group, random.Random(719), source_counts={key: 0 for key in counts})
        empty_broad = next(row for row in empty["queries"] if row["entities"] == ["Cyra"]
                           and row.get("expansion_query_class") == "broad")
        self.assertEqual(empty_broad["support_texts"], [])
        self.assertEqual(empty_broad["expansion_world_class"], "broad_empty")
        self.assertTrue(empty_broad["prop"])
        positive, negative = semantic_relation_masks([empty_broad], empty["facts"],
                                                    torch.ones(1, len(empty["facts"])))
        self.assertFalse(positive.any())
        self.assertEqual(int(negative.sum()), 4)
        self.assertTrue(all(empty["facts"][int(index)]["entity"] == "Cyra"
                            and empty["facts"][int(index)]["prop"] != COEXISTENCE_PROP
                            for _, index in negative.nonzero()))

    def test_expansion_validation_rejects_missing_or_nontrue_specific_labels(self):
        group = reader_expansion()
        for value in (None, False, 1):
            with self.subTest(value=value):
                row = dict(group["queries"][0])
                if value is None:
                    row.pop("semantic_specific")
                else:
                    row["semantic_specific"] = value
                invalid = dict(facts=group["facts"], queries=[row] + group["queries"][1:])
                with self.assertRaisesRegex(ValueError, "source-specific semantic label"):
                    validate_expansion(invalid)


if __name__ == "__main__":
    unittest.main()
