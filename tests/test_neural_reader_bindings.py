"""Independent fit worlds separate an object's neighboring properties."""

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_fit_support import sample_relation_queries, semantic_relation_masks
from neural_reader_evidence_curriculum import (
    READER_BINDING_PROPERTIES, reader_binding_feature_texts, reader_binding_fit_group,
)
from train_neural_dialogue_reader import supported
import train_neural_dialogue_reader as reader


class ReaderBindingCurriculumContract(unittest.TestCase):
    def test_every_actor_object_has_all_properties_and_complete_support_sets(self):
        group = reader_binding_fit_group()
        self.assertEqual(len(group["facts"]), 36)
        self.assertEqual(len(group["queries"]), 96)
        self.assertEqual(Counter(row["query_class"] for row in group["queries"]),
                         dict(explicit=48, context=36, null=12))
        objects = defaultdict(list)
        for fact in group["facts"]:
            objects[fact["entity"], fact["binding_object"]].append(fact)
            self.assertLessEqual(len(fact["text"].encode()), 80)
            self.assertEqual(fact["status"], 0)
        self.assertEqual(len(objects), 12)
        for facts in objects.values():
            self.assertEqual({fact["prop"] for fact in facts}, set(READER_BINDING_PROPERTIES))
        support_counts = Counter()
        for row in group["queries"]:
            gold = [fact for fact in group["facts"] if supported(row, fact)]
            self.assertEqual({fact["text"] for fact in gold}, set(row["support_texts"]))
            support_counts[len(gold)] += 1
            for fact in gold:
                self.assertEqual(fact["entity"], row["entities"][0])
                self.assertEqual(fact["binding_object"], row["binding_object"])
                self.assertIn(fact["prop"], row["prop"] if isinstance(row["prop"], list) else [row["prop"]])
            self.assertEqual(set(row["binding_contrast_texts"]),
                             {fact["text"] for fact in objects[row["entities"][0], row["binding_object"]]} - set(row["support_texts"]))
            if len(gold) == 2:
                self.assertEqual(len({fact["prop"] for fact in gold}), 2)
        self.assertEqual(support_counts, {0: 12, 1: 72, 2: 12})

    def test_context_reference_ranges_are_original_sources_not_query_prefixes(self):
        group = reader_binding_fit_group()
        contents = {fact["content"] for fact in group["facts"]}
        observation_keys = set()
        identifiers = set()
        for row in group["queries"]:
            obs = row["observation"]
            self.assertEqual(row["text"], obs["body"])
            self.assertEqual(obs["source_id"], row["observation_id"])
            self.assertNotIn(obs["source_id"], identifiers)
            identifiers.add(obs["source_id"])
            key = (obs["body"], obs["context"])
            self.assertNotIn(key, observation_keys)
            observation_keys.add(key)
            self.assertNotIn("reference", row)
            self.assertNotIn("initial_reference", row)
            self.assertNotIn("Activated evidence:", row["text"])
            self.assertFalse(row["text"].startswith(row["entities"][0] + ". "))
            self.assertFalse(any(content in obs["context"] for content in contents))
            if row["query_class"] == "context":
                name = row["entities"][0]
                self.assertNotIn(name, obs["body"])
                self.assertTrue(obs["whole_need"])
                self.assertEqual(obs["whole_refs"], [name])
                source = obs["whole_ref_ranges"][0]
                self.assertEqual(source["document"], "context")
                raw = obs["context"].encode()
                self.assertEqual(raw[source["start"]:source["end"]].decode(), name)
                self.assertEqual(source["start"], raw.rfind(name.encode()))
            else:
                self.assertFalse(obs["whole_need"])
                self.assertEqual(obs["whole_refs"], [])

    def test_writer_observations_keep_complete_bounded_spans_and_literal_speaker(self):
        for fact in reader_binding_fit_group()["facts"]:
            obs = fact["observation"]
            span = obs["spans"][0]
            self.assertEqual(obs["body"], fact["content"])
            self.assertEqual(span["start"], 0)
            self.assertEqual(span["end"], len(fact["content"].encode()))
            self.assertEqual(obs["body"].encode()[span["start"]:span["end"]].decode(), fact["content"])
            self.assertEqual(span["action"], 1)
            self.assertEqual(span["references"], [fact["entity"]])
            source = span["ref_ranges"][0]
            self.assertEqual(obs[source["document"]].encode()[source["start"]:source["end"]].decode(), fact["entity"])
            self.assertEqual(obs["whole_ref_ranges"], span["ref_ranges"])

    def test_same_object_wrong_properties_are_eligible_semantic_negatives(self):
        group = reader_binding_fit_group()
        for row in group["queries"]:
            hardness = torch.zeros(1, 36)
            contrast = set(row["binding_contrast_texts"])
            for index, fact in enumerate(group["facts"]):
                if fact["text"] in contrast:
                    hardness[0, index] = 100
            positive, negative = semantic_relation_masks([row], group["facts"], hardness)
            self.assertEqual(int(positive.sum()), 3 * len(row["support_texts"]))
            selected = {fact["text"] for index, fact in enumerate(group["facts"]) if negative[0, index]}
            self.assertTrue(contrast <= selected)
            self.assertFalse(contrast & set(row["support_texts"]))
            for index in negative[0].nonzero().flatten():
                self.assertEqual(group["facts"][index]["entity"], row["entities"][0])

    def test_existing_family_sampler_balances_objects_and_question_forms(self):
        queries = [row for row in reader_binding_fit_group()["queries"] if row["prop"]]
        expected = {(tuple(row["prop"]) if isinstance(row["prop"], list) else (row["prop"],),
                     row["pattern_id"], row["fit_paraphrase_id"]) for row in queries}
        self.assertEqual(len(expected), 28)
        sampled = sample_relation_queries(queries, random.Random(1057), 28)
        actual = {(tuple(row["prop"]) if isinstance(row["prop"], list) else (row["prop"],),
                   row["pattern_id"], row["fit_paraphrase_id"]) for row in sampled}
        self.assertEqual(actual, expected)
        self.assertEqual(sampled, sample_relation_queries(queries, random.Random(1057), 28))

    def test_raw_feature_manifest_is_exact_and_held_out_input_is_rejected(self):
        group = reader_binding_fit_group()
        before = deepcopy(group)
        expected = set()
        for rows in group.values():
            for row in rows:
                expected.update((row["text"], row["observation"]["body"], row["observation"]["context"]))
                if row.get("reference"):
                    expected.add(row["reference"])
        self.assertEqual(reader_binding_feature_texts(group), sorted(expected))
        self.assertEqual(group, before)
        for split in ("development", "blind", "evaluation"):
            with self.assertRaisesRegex(ValueError, "fit-only"):
                reader_binding_fit_group(split=split)
        for change in (dict(reference="Aileen"), dict(initial_reference="Aileen"),
                       dict(evaluation_only=True), dict(text="Aileen. Injected question")):
            damaged = deepcopy(group)
            damaged["queries"][0].update(change)
            with self.assertRaises(ValueError):
                reader_binding_feature_texts(damaged)

    def test_existing_fit_actors_are_variable_and_returned_data_is_independent(self):
        first = reader_binding_fit_group()
        before = deepcopy(first)
        self.assertEqual(first, reader_binding_fit_group())
        other = reader_binding_fit_group(actors=("Desmond", "Etta", "Felix"))
        self.assertEqual({fact["entity"] for fact in other["facts"]}, {"Desmond", "Etta", "Felix"})
        other["facts"][0]["observation"]["spans"][0]["ref_ranges"].clear()
        self.assertEqual(first, before)
        for actors in (("Aileen", "Aileen", "Celia"), ("Aster", "Bram", "Celia"),
                       ("Aileen", "Bram"), ("Aileen", "Bram", "Celia", "Desmond"), "Aileen"):
            with self.assertRaisesRegex(ValueError, "existing fit actors"):
                reader_binding_fit_group(actors=actors)

    def test_trainer_option_rejects_wrong_head_and_topic_before_creating_output(self):
        settings = dict(steps=1, train_heads="matcher", semantic_topic_weight=0,
                        resume="parent.pt", relation_bindings=True,
                        output="uncreated-output", relation_paraphrases=False)
        for change in (dict(train_heads="all"), dict(train_heads="read"),
                       dict(train_heads="identity"), dict(train_heads="scope"),
                       dict(train_heads="temporal"),
                       dict(train_heads="semantic", semantic_topic_weight=0.05)):
            with patch.object(Path, "mkdir", side_effect=AssertionError("output was created")):
                with self.assertRaisesRegex(ValueError, "relation bindings"):
                    reader.train(SimpleNamespace(**dict(settings, **change)))
        for mode in ("semantic", "matcher"):
            with patch.object(Path, "mkdir", side_effect=RuntimeError("validated option")):
                with self.assertRaisesRegex(RuntimeError, "validated option"):
                    reader.train(SimpleNamespace(**dict(settings, train_heads=mode)))

    def test_opt_in_group_keeps_original_fit_and_development_inputs_unchanged(self):
        from prepare_neural_colloquial import reader_data
        originals = []

        def capture(split, observations):
            group = reader_data(split, observations)
            originals.append((group, deepcopy(group)))
            return group

        settings = dict(steps=1, train_heads="matcher", semantic_topic_weight=0,
                        resume="parent.pt", relation_paraphrases=False,
                        curriculum=None, fit_paraphrases=False, replay_episodes=False,
                        gguf="unused-backbone")
        with tempfile.TemporaryDirectory() as temporary:
            for enabled in (False, True):
                with patch("prepare_neural_colloquial.reader_data", side_effect=capture), \
                     patch("neural_reader_evidence_curriculum.reader_binding_fit_group", wraps=reader_binding_fit_group) as builder, \
                     patch.object(reader, "sha", side_effect=RuntimeError("stop before extraction")):
                    with self.assertRaisesRegex(RuntimeError, "stop before extraction"):
                        reader.train(SimpleNamespace(**dict(settings, relation_bindings=enabled,
                                     output=str(Path(temporary) / str(enabled)))))
                self.assertEqual(builder.call_count, int(enabled))
        self.assertEqual(len(originals), 4)
        for group, before in originals:
            self.assertEqual(group, before)
        self.assertEqual(originals[0][0], originals[2][0])
        self.assertEqual(originals[1][0], originals[3][0])


if __name__ == "__main__":
    unittest.main()
