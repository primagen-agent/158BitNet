"""Scope, stored-version and unbiased held-out feedback teacher contracts."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_activation_curriculum import (
    lifecycle_fit_group,
    scope_fit_group,
    scope_relation_masks,
    evidence_challenge,
    identity_multisubject_fit_group,
    scope_development_group,
    SCOPE_DEVELOPMENT_ACTORS,
    _registered_actor_names,
    relation_variation_fit_group,
)
from neural_fit_support import semantic_relation_masks
from train_neural_dialogue_reader import supported


class ActivationCurriculumContract(unittest.TestCase):
    def test_history_preserves_present_text_kind_and_c_physical_status(self):
        group = lifecycle_fit_group()
        facts = group["facts"]
        self.assertEqual(len(facts), 36)
        self.assertEqual([f["insert_order"] for f in facts], list(range(36)))
        self.assertEqual([f["status"] for f in facts], [1, 0, 3] * 12)
        self.assertEqual([f["kind"] for f in facts], [0, 0, 2] * 12)
        self.assertTrue(all(f["content"].startswith("I ") for f in facts))
        for query in group["queries"]:
            sources = [f for f in facts if supported(query, f)]
            if query["lifecycle_view"] == "history":
                self.assertEqual(query["kind"], 1)
                self.assertEqual(query["allowed_kinds"], [1])
                self.assertEqual(query["source_kinds"], [0])
                self.assertEqual(len(sources), 1)
                self.assertEqual(sources[0]["status"], 1)
                self.assertEqual(sources[0]["kind"], 0)
            elif query["lifecycle_view"] == "current":
                self.assertEqual([f["status"] for f in sources], [0])
                self.assertEqual(query["allowed_kinds"], [0])
            elif query["lifecycle_view"] == "tentative":
                self.assertEqual([f["status"] for f in sources], [3])
                self.assertEqual(query["allowed_kinds"], [2])
            else:
                self.assertEqual([f["status"] for f in sources], [1, 0, 3])
                self.assertEqual(query["allowed_kinds"], [0, 1, 2])
            self.assertNotIn("reference", query)

    def test_recall_exact_content_and_scope_all_versions_are_independent(self):
        group = lifecycle_fit_group()
        scope = scope_fit_group()
        self.assertEqual(scope["facts"], group["facts"])
        self.assertNotIn("queries", scope)
        self.assertEqual(len(scope["scope_queries"]), 54)
        current = next(q for q in group["queries"] if q["prop"] == "residence" and q["lifecycle_view"] == "current")
        declaration = next(q for q in scope["scope_queries"] if q["entities"] == current["entities"] and q["prop"] == "residence" and "Sion" in q["text"])
        self.assertEqual(len(current["support_texts"]), 1)
        self.assertEqual(len(declaration["support_texts"]), 3)
        self.assertTrue(any("Tartu" in text for text in declaration["support_texts"]))
        self.assertTrue(any("Dinan" in text for text in declaration["support_texts"]))
        for query in scope["scope_queries"]:
            sources = [f for f in scope["facts"] if supported(query, f)]
            all_subject = query["scope_operation"] == "delete_all_subject"
            self.assertEqual(len(sources), 12 if all_subject else 3)
            self.assertEqual({f["entity"] for f in sources}, set(query["entities"]))
            self.assertEqual({f["prop"] for f in sources}, set(query["prop"]) if all_subject else {query["prop"]})
            self.assertEqual({f["status"] for f in sources}, {0, 1, 3})
            self.assertNotIn("reference", query)
            self.assertFalse(query["text"].startswith(query["entities"][0] + ". "))

    def test_scope_semantics_use_property_not_old_values_or_kind(self):
        group = scope_fit_group()
        row = next(q for q in group["scope_queries"] if q["prop"] == "residence" and "Sion" in q["text"])
        facts = group["facts"]
        positive, negative = scope_relation_masks([row], facts, torch.arange(len(facts)).float()[None])
        self.assertEqual(int(positive.sum()), 9)
        self.assertEqual(int(negative.sum()), 4)
        self.assertEqual({facts[i]["prop"] for _, i in positive.nonzero()}, {"residence"})
        self.assertEqual({facts[i]["status"] for _, i in positive.nonzero()}, {0, 1, 3})
        self.assertTrue(all(facts[i]["entity"] in row["entities"] and facts[i]["prop"] != "residence" for _, i in negative.nonzero()))
        # Deleted records remain excluded even if a caller accidentally passes
        # them; the tentative physical enum is 3, not 2.
        deleted = dict(facts[0], status=2)
        p, n = scope_relation_masks([row], [deleted], torch.zeros(1, 1))
        self.assertFalse(p.any())
        self.assertFalse(n.any())

    def test_source_kind_auxiliary_does_not_confuse_history_with_text_tense(self):
        group = lifecycle_fit_group()
        rows = group["queries"][:4]
        before = deepcopy(rows)
        positive, _ = semantic_relation_masks(rows, group["facts"], torch.zeros(4, 36))
        self.assertEqual(positive.sum(-1).tolist(), [3, 3, 3, 9])
        self.assertEqual(rows, before)
        self.assertEqual([sum(supported(q, f) for f in group["facts"]) for q in rows], [1, 1, 1, 3])

    def challenge_group(self):
        facts = [dict(text=f"Actor{i}. Complete fact {i}.", entity=f"Actor{i}", prop="topic", kind=0) for i in range(6)]
        queries = [
            dict(text="Which item was chosen?", observation_id="heldout-a", entities=["Actor0"], prop="topic", kind=0, support_texts=[facts[0]["text"]], reference="NN original actor"),
            dict(text="What happened next?", observation_id="native-pilot-excluded", entities=["Actor1"], prop="topic", kind=0, support_texts=[facts[1]["text"]]),
        ]
        return dict(facts=facts, queries=queries)

    def test_evidence_views_exclude_pilot_and_preserve_nn_anchor(self):
        group = self.challenge_group()
        before = deepcopy(group)
        result = evidence_challenge(group, 4217, excluded_observation_ids={"native-pilot-excluded"})
        self.assertEqual(group, before)
        self.assertEqual(len(result["queries"]), 3)
        for row in result["queries"]:
            count = row["evidence_challenge_view"]
            self.assertEqual(row["original_observation_id"], "heldout-a")
            self.assertEqual(row["initial_text"], group["queries"][0]["text"])
            self.assertEqual(row["initial_reference"], "NN original actor")
            self.assertEqual(row["support_texts"], group["queries"][0]["support_texts"])
            self.assertTrue(row["evaluation_only"])
            if count:
                self.assertEqual(len(row["evidence_sources"]), count)
                self.assertNotIn("reference", row)
                self.assertEqual(row["text"], row["initial_text"] + "\nActivated evidence:\n" + "\n".join(row["evidence_sources"]) + "\n")
            else:
                self.assertEqual(row["reference"], "NN original actor")
                self.assertEqual(row["text"], row["initial_text"])

    def test_evidence_cues_are_independent_of_all_gold_support_labels(self):
        group = self.challenge_group()
        changed = deepcopy(group)
        for row in changed["queries"]:
            row.update(entities=["Unrelated"], prop="other", kind=2, support_texts=[])
        first = evidence_challenge(group, 4217)
        second = evidence_challenge(changed, 4217)
        self.assertEqual(
            [(q["original_observation_id"], q["text"], q.get("evidence_sources")) for q in first["queries"]],
            [(q["original_observation_id"], q["text"], q.get("evidence_sources")) for q in second["queries"]],
        )
        self.assertEqual(first, evidence_challenge(group, 4217))
        with self.assertRaises(ValueError):
            evidence_challenge(group, 4217, limit=25)

    def test_evidence_questions_cover_properties_and_available_strata(self):
        group = self.challenge_group()
        group["queries"] = []
        for prop in ("preference", "activity", "collection", "project"):
            for kind in (0, 1):
                for query_class in ("explicit", "context"):
                    # One populous stratum must not displace rare relations or
                    # temporal/contextual views from this sixteen-question budget.
                    copies = 40 if (prop, kind, query_class) == ("preference", 0, "explicit") else 1
                    for index in range(copies):
                        identifier = f"{prop}-{kind}-{query_class}-{index}"
                        group["queries"].append(dict(
                            text=identifier, observation_id=identifier,
                            prop=prop, kind=kind, query_class=query_class,
                            support_texts=[group["facts"][0]["text"]],
                        ))
        original = deepcopy(group)
        result = evidence_challenge(group, 3219, limit=16)
        chosen = [q for q in result["queries"] if q["evidence_challenge_view"] == 0]
        self.assertEqual(len(chosen), 16)
        self.assertEqual(
            {(q["prop"], q["kind"], q["query_class"]) for q in chosen},
            {(q["prop"], q["kind"], q["query_class"]) for q in group["queries"]},
        )
        self.assertEqual(group, original)
        self.assertEqual(result, evidence_challenge(group, 3219, limit=16))
        self.assertEqual(result, evidence_challenge(dict(group, queries=list(reversed(group["queries"]))), 3219, limit=16))
        with self.assertRaisesRegex(ValueError, "cover every nonempty property"):
            evidence_challenge(group, 3219, limit=3)

    def test_cues_for_retained_question_ignore_selection_and_stratum_changes(self):
        group = self.challenge_group()
        first = evidence_challenge(group, 4317)
        changed = deepcopy(group)
        changed["queries"][0].update(prop="new-topic", kind=2, query_class="self", support_texts=[])
        second = evidence_challenge(changed, 4317, excluded_observation_ids={"native-pilot-excluded"})
        first = [q for q in first["queries"] if q["original_observation_id"] == "heldout-a"]
        self.assertEqual(
            [(q["text"], q.get("evidence_sources")) for q in first],
            [(q["text"], q.get("evidence_sources")) for q in second["queries"]],
        )
        duplicate = dict(group, queries=group["queries"] + [group["queries"][0]])
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            evidence_challenge(duplicate, 4317)

    def test_multisubject_identity_sources_stay_fit_only_and_budgeted(self):
        from prepare_neural_identity import ACTORS, _existing_actor_names
        identity = identity_multisubject_fit_group()
        self.assertEqual(len(identity["queries"]), 18)
        self.assertEqual({len(q["entities"]) for q in identity["queries"]}, {2, 3})
        self.assertTrue(all(set(q["entities"]) <= set(ACTORS) for q in identity["queries"]))
        self.assertTrue(all(not (set(q["entities"]) & _existing_actor_names()) for q in identity["queries"]))
        self.assertTrue(all("reference" not in q and "support_texts" not in q for q in identity["queries"]))
        life = lifecycle_fit_group()
        texts = {r["text"] for rows in life.values() for r in rows}
        texts.update(q["text"] for q in identity["queries"])
        self.assertEqual(len(texts), 150)

    def test_scope_development_is_independent_bounded_and_selection_only(self):
        development = scope_development_group()
        fit = scope_fit_group()
        self.assertEqual(len(development["facts"]), 32)
        self.assertEqual(len(development["scope_queries"]), 20)
        self.assertFalse(set(SCOPE_DEVELOPMENT_ACTORS) & _registered_actor_names())
        fit_texts = {r["text"] for rows in fit.values() for r in rows}
        dev_texts = {r["text"] for rows in development.values() for r in rows}
        self.assertFalse(fit_texts & dev_texts)
        self.assertEqual(len(dev_texts), 52)
        self.assertFalse({f["content"] for f in fit["facts"]} & {f["content"] for f in development["facts"]})
        self.assertEqual([f["status"] for f in development["facts"]], [1, 1, 0, 3] * 8)
        self.assertEqual([f["kind"] for f in development["facts"]], [0, 0, 0, 2] * 8)
        self.assertEqual([f["insert_order"] for f in development["facts"]], list(range(32)))
        for q in development["scope_queries"]:
            self.assertTrue(q["evaluation_only"])
            self.assertNotIn("reference", q)
            self.assertFalse(q["text"].startswith(q["entities"][0] + ". "))
            support = [f for f in development["facts"] if supported(q, f)]
            all_subject = q["scope_operation"] == "delete_all_subject"
            self.assertEqual(len(support), 16 if all_subject else 4)
            self.assertEqual({f["entity"] for f in support}, set(q["entities"]))
            self.assertEqual({f["status"] for f in support}, {0, 1, 3})
            if all_subject:
                self.assertEqual(len({f["prop"] for f in support}), 4)
        self.assertEqual(development, scope_development_group())

    def test_all_subject_fit_requests_preserve_sources_and_reject_other_actor(self):
        original = lifecycle_fit_group()
        expanded = scope_fit_group()
        self.assertEqual(expanded["facts"], original["facts"])
        all_subject = [q for q in expanded["scope_queries"] if q["scope_operation"] == "delete_all_subject"]
        self.assertEqual(len(all_subject), 6)
        for q in all_subject:
            self.assertEqual(sum(supported(q, f) for f in expanded["facts"]), 12)
            self.assertEqual(len(q["prop"]), 4)
            self.assertFalse(any(supported(q, f) for f in expanded["facts"] if f["entity"] not in q["entities"]))

    def test_relation_variation_is_bounded_named_and_exactly_source_supported(self):
        from collections import Counter
        from neural_fit_support import FIT_RELATION_QUESTIONS
        from prepare_neural_identity import ACTORS
        fit_before = deepcopy(scope_fit_group())
        development_before = deepcopy(scope_development_group())
        group = relation_variation_fit_group()
        self.assertEqual((len(group["facts"]), len(group["queries"])), (48, 48))
        self.assertEqual(len({r["text"] for rows in group.values() for r in rows}), 96)
        self.assertEqual({f["prop"] for f in group["facts"]}, set(FIT_RELATION_QUESTIONS))
        self.assertEqual(set(Counter((f["entity"], f["prop"]) for f in group["facts"]).values()), {2})
        self.assertTrue({f["entity"] for f in group["facts"]} <= set(ACTORS))
        self.assertFalse({f["entity"] for f in group["facts"]} & set(SCOPE_DEVELOPMENT_ACTORS))
        facts = {f["text"]: f for f in group["facts"]}
        for query in group["queries"]:
            self.assertNotIn("reference", query)
            self.assertIn(query["entities"][0], query["text"])
            self.assertFalse(query["text"].startswith(query["entities"][0] + ". "))
            self.assertEqual(len(query["support_texts"]), 1)
            fact = facts[query["support_texts"][0]]
            self.assertEqual(fact["text"], fact["reference"] + ". " + fact["content"])
            self.assertEqual((fact["kind"], fact["status"]), (0, 0))
            self.assertEqual(fact["entity"], query["entities"][0])
            self.assertEqual(fact["prop"], query["prop"])
            self.assertEqual(sum(supported(query, f) for f in group["facts"]), 1)
        self.assertEqual(scope_fit_group(), fit_before)
        self.assertEqual(scope_development_group(), development_before)
        self.assertEqual(group, relation_variation_fit_group())

    def test_variation_specific_objects_train_same_property_negative_without_actor_leak(self):
        group = relation_variation_fit_group()
        row, facts = group["queries"][0], group["facts"]
        hardness = torch.zeros(1, len(facts))
        hardness[0, 1] = 100  # Same actor/property, different actual object.
        positive, negative = semantic_relation_masks([row], facts, hardness)
        self.assertEqual(int(positive.sum()), 3)
        self.assertTrue(negative[0, 1])
        self.assertEqual(int(negative.sum()), 4)
        for _, index in negative.nonzero():
            self.assertEqual(facts[index]["entity"], row["entities"][0])
            self.assertNotIn(facts[index]["text"], row["support_texts"])
        self.assertEqual(sum(supported(row, fact) for fact in facts), 1)


if __name__ == "__main__":
    unittest.main()
