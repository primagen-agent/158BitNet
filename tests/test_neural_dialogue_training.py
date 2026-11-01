"""Current training invariants and held-out synthetic supervision contracts."""

from pathlib import Path
from copy import deepcopy
import json
import sys
import unittest
import torch
import itertools
import math
import tempfile
import random
import struct
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_slot_model import (
    NeuralSlots,
    INPUT,
    WIDTH,
    SEMANTIC,
    QUERY,
    LEXICAL,
    FORMAT,
    word_features,
    identity_word_features,
    observation_context,
)
from train_neural_dialogue import (
    batch_loss,
    prepare,
    observation_loss,
    source_words,
    source_types,
    decode_references,
    reference_targets,
    bounded_dialogue_replay,
    sample_action_balanced_batch,
    writer_development_safe,
)
from train_neural_dialogue_reader import (
    DynamicFeatures,
    make_bank,
    query,
    raw_query,
    add_episodes,
    supported,
    semantic_relation_masks,
    sample_relation_queries,
    semantic_relation_auxiliary,
    scope_relation_auxiliary,
    scope_training_loss,
    semantic_training_loss,
    scope_evaluate,
    scope_selection_score,
    kind_bank_diagnostics,
    attention_bank_diagnostics,
    reader_optimizer,
    configure_reader_heads,
    bind_curriculum_queries,
    sync_query_planner,
    identity_calibration,
    active_bank_facts,
    intent_targets,
    evaluate as reader_evaluate,
    reader_checkpoint_payload,
    reader_initial_selection,
    reader_planner_provenance,
    load_model as load_reader_model,
    balanced,
    scope_refinement_safe,
    LOSS_REDUCTION,
    train as train_reader,
)
from compose_neural_dialogue import compose
from prepare_neural_episodes import episode_curriculum, NAMES as EPISODE_NAMES


class TrainingContract(unittest.TestCase):
    def reader_baseline_views(self):
        component = {
            "authored": dict(activation_f1=.75, positive_support_set_exact=.75,
                             null_false_activation=0., confusion=[[1, 2]]),
            "episode": dict(activation_f1=.5, positive_support_set_exact=.5,
                            null_false_activation=0., queries=16),
        }
        feedback = {
            "0": dict(positive_queries=8, positive_support_set_exact=.5,
                      null_false_activation=0., activation_f1=.625),
            "1": dict(positive_queries=8, positive_support_set_exact=.25,
                      null_false_activation=0., activation_f1=.5),
            "4": dict(positive_queries=8, positive_support_set_exact=.25,
                      null_false_activation=.125, activation_f1=.375,
                      source_aliases=[dict(names=["Ada"])]),
        }
        return component, feedback

    def test_reader_initial_selection_owns_independent_baseline_views_for_each_run(self):
        component, feedback = self.reader_baseline_views()
        expected_component, expected_feedback = deepcopy(component), deepcopy(feedback)
        first = reader_initial_selection(component, feedback)
        second = reader_initial_selection(component, feedback)
        self.assertEqual(set(first), {"initial_development_views", "initial_evidence_development_views", "initial_selection_score"})
        self.assertEqual(first["initial_selection_score"], (.125, 7 / 24, .55))
        component["authored"]["activation_f1"] = .9
        component["authored"]["confusion"][0][0] = 999
        feedback["4"]["source_aliases"][0]["names"].append("Ben")
        self.assertEqual(first["initial_development_views"], expected_component)
        self.assertEqual(first["initial_evidence_development_views"], expected_feedback)
        first["initial_development_views"]["episode"]["queries"] = 999
        first["initial_evidence_development_views"]["4"]["source_aliases"][0]["names"].append("Cara")
        self.assertEqual(second["initial_development_views"], expected_component)
        self.assertEqual(second["initial_evidence_development_views"], expected_feedback)
        self.assertEqual(component["episode"]["queries"], 16)
        self.assertEqual(feedback["4"]["source_aliases"][0]["names"], ["Ada", "Ben"])

    def test_selected_reader_checkpoint_and_report_keep_new_run_baseline_after_nonzero_step(self):
        component, feedback = self.reader_baseline_views()
        initial = reader_initial_selection(component, feedback)
        expected = deepcopy(initial)
        resume = dict(format=FORMAT, backbone_sha256="b" * 64, reader_step=20,
                      initial_development_views=dict(stale=dict(activation_f1=0.)),
                      initial_evidence_development_views=dict(stale=dict(positive_queries=1)),
                      initial_selection_score=(-1., -1., -1.),
                      writer_training=dict(mode="writer-membership"))
        previous = deepcopy(resume)
        # A later selected point replaces live metrics, but must not redefine
        # this run's initial point or inherit the resumed run's old baseline.
        component["authored"]["activation_f1"] = .875
        feedback["4"]["positive_support_set_exact"] = .75
        selected_score = (.5, .625, .875)
        payload = reader_checkpoint_payload(resume, reader_step=80,
            state_dict=dict(weight=torch.tensor([2.])), selection_metrics=component,
            feedback_metrics=feedback, selection_score=selected_score, **initial)
        report = dict(selected_step=80, selection_score=selected_score, **initial)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.pt"
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False, map_location="cpu")
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps(report))
            saved_report = json.loads(report_path.read_text())
        self.assertEqual(loaded["reader_step"], 80)
        self.assertEqual(loaded["selection_metrics"], component)
        self.assertEqual(loaded["feedback_metrics"], feedback)
        self.assertEqual(loaded["selection_score"], selected_score)
        self.assertTrue(torch.equal(loaded["state_dict"]["weight"], torch.tensor([2.])))
        for key, value in expected.items():
            self.assertEqual(loaded[key], value)
            self.assertEqual(saved_report[key], list(value) if isinstance(value, tuple) else value)
        self.assertEqual(resume, previous)
        self.assertEqual(loaded["writer_training"], previous["writer_training"])

    def test_reader_initial_scope_selection_preserves_strict_set_tuple_and_rejects_no_positives(self):
        component, feedback = self.reader_baseline_views()
        scope = dict(positive_support_set_exact=.75, fp=4, queries=16,
                     activation_f1=.625, null_false_activation=.125)
        initial = reader_initial_selection(component, feedback, scope=scope)
        self.assertEqual(initial["initial_selection_score"], (.75, -.25, 1.25))
        self.assertEqual(initial["initial_development_views"], component)
        self.assertEqual(initial["initial_evidence_development_views"], feedback)
        scope["positive_support_set_exact"] = None
        self.assertEqual(initial["initial_selection_score"], (.75, -.25, 1.25))
        with self.assertRaisesRegex(ValueError, "positive development queries"):
            reader_initial_selection(component, feedback, scope=scope)

    def test_reader_checkpoint_roundtrip_keeps_writer_provenance_and_current_selection(self):
        previous = dict(
            format=FORMAT,
            backbone_sha256="b" * 64,
            selected_step=640,
            writer_training=dict(mode="writer-raw", trainable_tensors=["plan_boundary_raw.weight"]),
            writer_source_sha256="c" * 64,
            reader_step=20,
            selection_metrics=dict(activation_f1=.2),
            state_dict=dict(weight=torch.tensor([1.])),
        )
        payload = reader_checkpoint_payload(
            {key: value for key, value in previous.items() if key != "state_dict"},
            format=FORMAT, backbone_sha256="b" * 64,
            state_dict=dict(weight=torch.tensor([2.])), reader_step=80,
            selection_metrics=dict(activation_f1=.4),
            loss_reduction=LOSS_REDUCTION,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.pt"
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False, map_location="cpu")
        self.assertEqual(loaded["selected_step"], 640)
        self.assertEqual(loaded["writer_training"], previous["writer_training"])
        self.assertEqual(loaded["writer_source_sha256"], previous["writer_source_sha256"])
        self.assertEqual(loaded["reader_step"], 80)
        self.assertEqual(loaded["selection_metrics"], dict(activation_f1=.4))
        self.assertEqual(loaded["loss_reduction"], "query-balanced")
        self.assertTrue(torch.equal(loaded["state_dict"]["weight"], torch.tensor([2.])))
        self.assertEqual(previous["reader_step"], 20)
        self.assertTrue(torch.equal(previous["state_dict"]["weight"], torch.tensor([1.])))

    def test_resumed_reader_exports_exact_binding_planner_and_preserves_other_parameters(self):
        from neural_slot_model import PLAN_KEYS, QUERY_KEYS
        reader, planner = NeuralSlots(), NeuralSlots()
        before = {key: value.clone() for key, value in reader.state_dict().items()}
        source_keys = set(PLAN_KEYS + QUERY_KEYS)
        self.assertEqual(len(source_keys), 46)
        self.assertEqual(sum(key.startswith("plan_boundary.") for key in source_keys), 4)
        self.assertIn("plan_boundary_raw.weight", source_keys)
        self.assertTrue(any(not torch.equal(before[key], planner.state_dict()[key]) for key in source_keys))
        sync_query_planner(reader, planner)
        after = reader.state_dict()
        for key in after:
            if key in source_keys:
                torch.testing.assert_close(after[key], planner.state_dict()[key], atol=0, rtol=0)
                self.assertFalse(dict(reader.named_parameters())[key].requires_grad)
            else:
                torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
        configure_reader_heads(reader, "semantic")
        self.assertTrue(all(not dict(reader.named_parameters())[key].requires_grad for key in source_keys))
        # Sync owns copies; subsequently changing a planner cannot mutate the reader.
        key = QUERY_KEYS[0]
        with torch.no_grad():
            planner.state_dict()[key].add_(1)
        self.assertFalse(torch.equal(reader.state_dict()[key], planner.state_dict()[key]))

    def test_distinct_resume_and_planner_bind_source_parameters_and_metadata(self):
        from neural_slot_model import PLAN_KEYS, QUERY_KEYS
        from neural_training_common import sha

        source_keys = set(PLAN_KEYS + QUERY_KEYS)
        reader, writer = NeuralSlots(), NeuralSlots()
        with torch.no_grad():
            for index, (key, parameter) in enumerate(writer.named_parameters()):
                if key in source_keys:
                    parameter.fill_((index + 1) / 100)
        previous = dict(
            format=FORMAT, backbone_sha256="a" * 64,
            state_dict=reader.state_dict(), reader_step=20,
            reader_training_mode="semantic",
            selection_metrics=dict(activation_f1=.4, views=[dict(positive_exact=3)]),
            writer_training=dict(mode="writer-raw", trainable_tensors=["plan_boundary_raw.weight"]),
            writer_checkpoint_sha256="0" * 64, frozen_query_planner_sha256="0" * 64,
        )
        planner_checkpoint = dict(
            format=FORMAT, backbone_sha256="a" * 64,
            state_dict=writer.state_dict(), reader_step=999,
            selection_metrics=dict(activation_f1=.9),
            writer_training=dict(mode="writer-membership", trainable_tensors=["plan_membership_raw.weight"]),
            writer_checkpoint_sha256="1" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resume_path, planner_path = root / "reader.pt", root / "writer.pt"
            torch.save(previous, resume_path)
            torch.save(planner_checkpoint, planner_path)
            resume = torch.load(resume_path, weights_only=False, map_location="cpu")
            planner = torch.load(planner_path, weights_only=False, map_location="cpu")
            resumed, actual_planner = load_reader_model(resume), load_reader_model(planner)
            sync_query_planner(resumed, actual_planner)
            actual_sha = sha(planner_path)
            provenance = reader_planner_provenance(planner, actual_sha)
            metadata = reader_checkpoint_payload(
                {key: value for key, value in resume.items() if key != "state_dict"},
                **provenance,
            )
            payload = reader_checkpoint_payload(metadata, state_dict=resumed.state_dict())
            torch.save(payload, root / "selected.pt")
            saved = torch.load(root / "selected.pt", weights_only=False, map_location="cpu")
        self.assertEqual(len(source_keys), 46)
        for key, value in saved["state_dict"].items():
            expected = planner["state_dict"][key] if key in source_keys else resume["state_dict"][key]
            self.assertTrue(torch.equal(value, expected), key)
            if key in source_keys:
                self.assertFalse(dict(resumed.named_parameters())[key].requires_grad)
        self.assertEqual(saved["writer_checkpoint_sha256"], actual_sha)
        self.assertEqual(saved["frozen_query_planner_sha256"], actual_sha)
        self.assertEqual(saved["writer_training"], planner["writer_training"])
        self.assertEqual(saved["reader_step"], resume["reader_step"])
        self.assertEqual(saved["selection_metrics"], resume["selection_metrics"])
        self.assertEqual(saved["reader_training_mode"], resume["reader_training_mode"])
        payload["writer_training"]["trainable_tensors"].append("changed")
        payload["selection_metrics"]["views"][0]["positive_exact"] = 99
        self.assertEqual(planner["writer_training"]["trainable_tensors"], ["plan_membership_raw.weight"])
        self.assertEqual(resume["selection_metrics"]["views"][0]["positive_exact"], 3)
        self.assertEqual(metadata["selection_metrics"]["views"][0]["positive_exact"], 3)
        self.assertEqual(provenance["writer_training"]["trainable_tensors"], ["plan_membership_raw.weight"])
        self.assertEqual(planner["writer_checkpoint_sha256"], "1" * 64)
        self.assertEqual(resume["writer_checkpoint_sha256"], "0" * 64)

    def test_missing_planner_training_metadata_does_not_inherit_resume_writer(self):
        resume = dict(writer_training=dict(mode="wrong-resume-writer"), reader_step=7)
        provenance = reader_planner_provenance(dict(reader_step=999), "a" * 64)
        saved = reader_checkpoint_payload(resume, **provenance)
        self.assertIsNone(saved["writer_training"])
        self.assertEqual(saved["writer_checkpoint_sha256"], "a" * 64)
        self.assertEqual(saved["reader_step"], 7)
        self.assertEqual(resume["writer_training"], dict(mode="wrong-resume-writer"))

    def test_writer_boundary_learning_preserves_encoded_reference_inputs(self):
        torch.manual_seed(9937)
        model = NeuralSlots()
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(("plan_boundary.", "plan_boundary_raw.")))
        self.assertEqual(tuple(model.plan_boundary_raw.weight.shape), (2, 3 * INPUT))
        self.assertEqual(float(model.plan_boundary[2].weight.detach().abs().sum()), 0)
        current, context, words = torch.randn(8, INPUT), torch.randn(6, INPUT), torch.randn(5, INPUT)
        original_words = words.clone()
        source = torch.randn(7, INPUT)
        kinds = F.one_hot(torch.tensor([0, 1, 2, 2, 3, 3, 3]), 4).float()
        raw_context = model.query_binding_context(current, context)
        before, before_logits, before_ends = model.plan(current, context, words)
        before_whole = model.query_reference(before.mean(0), source, kinds, raw_context=raw_context)
        before_span = model.plan_reference(before[1:4].mean(0), source, kinds)
        frozen = {name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        target = torch.tensor([0, 1, 4, 4, 0])
        for _ in range(2):
            _, logits, ends = model.plan(current, context, words)
            optimizer.zero_grad()
            (F.cross_entropy(logits, target) + model.sequence_loss(logits, target, ends)).backward()
            self.assertGreater(float(model.plan_boundary[2].weight.grad.abs().sum()), 0)
            raw_gradient = model.plan_boundary_raw.weight.grad
            self.assertTrue(torch.isfinite(raw_gradient).all())
            # The direct logits residual learns immediately even while the
            # original hidden-MLP output matrix starts at zero.
            self.assertGreater(float(raw_gradient.abs().sum()), 0)
            optimizer.step()
        after, after_logits, after_ends = model.plan(current, context, words)
        torch.testing.assert_close(words, original_words, atol=0, rtol=0)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assertGreater(float((after_logits-before_logits).detach().abs().sum() +
                                 (after_ends-before_ends).detach().abs().sum()), 0)
        for before_value, after_value in zip(before_whole, model.query_reference(
                after.mean(0), source, kinds, raw_context=raw_context)):
            torch.testing.assert_close(before_value, after_value, atol=0, rtol=0)
        for before_value, after_value in zip(before_span, model.plan_reference(after[1:4].mean(0), source, kinds)):
            torch.testing.assert_close(before_value, after_value, atol=0, rtol=0)
        for name,p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad, name)
                torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
        # All reader phases copy this writer branch but cannot train it.
        for mode in ("all", "read", "semantic", "matcher", "scope", "identity"):
            configure_reader_heads(model, mode)
            self.assertTrue(all(not p.requires_grad for name,p in model.named_parameters()
                                if name.startswith(("plan_boundary.", "plan_boundary_raw."))))

    def test_reference_only_binding_preserves_predictions_and_skips_writer_boundary_computation(self):
        torch.manual_seed(9011)
        model = NeuralSlots()
        body, speaker, timestamp = "Where did I leave the sketchpad?", "Ada", "2042-04-03"
        context = observation_context(speaker, timestamp, "")
        row = dict(body=body, context=context, speaker=speaker, timestamp=timestamp,
                   past="", source_id="reference-only-contract", tag="episode-query",
                   whole_need=True, whole_refs=["GoldReferenceMustNotBeUsed"], spans=[])
        group = dict(facts=[], queries=[dict(text=body, reference="GoldReferenceMustNotBeUsed",
                                            entities=["GoldReferenceMustNotBeUsed"], prop="location",
                                            kind=0, support_texts=[], observation=row)])
        cache = {text:dict(x=torch.randn(len(text), INPUT),
                           bounds=[(i,i+1) for i in range(len(text))]) for text in (body, context)}
        baseline = bind_curriculum_queries(group, model, cache)
        item = prepare(dict(test=[row]), cache, "cpu")[body, context]
        before, before_logits, before_ends = model.plan(item["body"], item["context"], item["body_words"])
        with torch.no_grad():
            model.plan_boundary[2].weight.normal_()
            model.plan_boundary[2].bias.fill_(0.25)
            model.plan_boundary_raw.weight.normal_(std=0.01)
        after, after_logits, after_ends = model.plan(item["body"], item["context"], item["body_words"])
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assertGreater(float((after_logits-before_logits).detach().abs().sum() +
                                 (after_ends-before_ends).detach().abs().sum()), 0)
        with patch.object(model, "plan", side_effect=AssertionError("writer plan must remain unused")), \
             patch.object(model, "plan_boundary_scores", side_effect=AssertionError("boundary GEMV must remain unused")):
            actual = bind_curriculum_queries(group, model, cache)
        self.assertEqual(actual, baseline)
        self.assertNotIn("GoldReferenceMustNotBeUsed", actual[0]["queries"][0]["text"])

    def test_query_identity_anchor_preserves_predicted_reference_and_raw_update_order(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.identity_update.weight[0, QUERY] = 1.0
        original = "Predicted. Where is the lamp?"
        reference = "Predicted"
        augmented = original + "\nActivated evidence:\nDistractor. A spare lamp stands nearby.\n"
        row = dict(text=augmented, initial_text=original, initial_reference=reference,
                   entities=["GoldLabelMustNotBeUsed"])
        cache = {text:dict(x=torch.randn(len(text), INPUT),
                           bounds=[(i,i+1) for i in range(len(text))])
                 for text in (original, reference, augmented)}
        cached = DynamicFeatures(model, dict(fit=dict(facts=[], queries=[row])), cache, "cpu")
        self.assertIn(reference, cached.index)
        self.assertIn(original, cached.index)
        oq, oid = raw_query(model, [dict(text=original, reference=reference)], cached)
        cq, cid = raw_query(model, [row], cached)
        q, identity = query(model, [row], cached)
        expected = model.advance_identity(oq, cq, oid, cid)
        torch.testing.assert_close(identity, expected)
        torch.testing.assert_close(q, model.advance_query(oq, cq))
        self.assertGreater(float((identity-cid).detach().abs().sum()), 0.1)
        wrong_order = model.advance_identity(oq, q, oid, cid)
        self.assertGreater(float((identity-wrong_order).detach().abs().sum()), 1e-6)
        identity[:, :LEXICAL].square().sum().add(identity[:, LEXICAL:].sum()).backward()
        self.assertIsNotNone(model.identity_update.weight.grad)
        # Plain contextual/self rows also retain the NN reference override;
        # otherwise their first-pass identity would be re-pooled from the body.
        normal = dict(text=original, reference=reference, entities=row["entities"])
        _, normal_identity = query(model, [normal], cached)
        torch.testing.assert_close(normal_identity, cached.reference_ids([normal]), atol=1e-6, rtol=1e-6)

    def test_relation_auxiliary_selects_bounded_same_actor_wrong_relation(self):
        rows = [dict(entities=["Ada"], prop="light", kind=0, support_texts=["Ada. support"])]
        facts = [dict(text="Ada. support", entity="Ada", prop="light", kind=0),
                 dict(text="Ada. alternative", entity="Ada", prop="light", kind=0),
                 dict(text="Ben. foreign", entity="Ben", prop="soil", kind=0)]
        facts += [dict(text="Ada. "+str(i), entity="Ada", prop="soil", kind=0) for i in range(6)]
        hardness = torch.tensor([[100.0, 1000.0, 900.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        positive, negative = semantic_relation_masks(rows, facts, hardness)
        self.assertEqual(positive.nonzero().tolist(), [[0, 0]])
        self.assertEqual(negative.nonzero().tolist(), [[0, 5], [0, 6], [0, 7], [0, 8]])
        # A query covering several properties excludes every overlap, even
        # if the alternative is absent from its exact support supervision.
        rows[0]["prop"] = ["light", "soil"]
        self.assertFalse(semantic_relation_masks(rows, facts, hardness)[1].any())
        with self.assertRaises(ValueError):
            semantic_relation_masks(rows, facts, hardness, max_negatives=5)

    def test_specific_relation_alternative_is_reserved_under_other_hard_negatives(self):
        rows = [dict(entities=["Ada"], prop="light", kind=0, semantic_specific=True,
                     support_texts=["Ada. white lamp"])]
        facts = [dict(text="Ada. white lamp", entity="Ada", prop="light", kind=0),
                 dict(text="Ada. red lamp", entity="Ada", prop="light", kind=0)]
        facts += [dict(text="Ada. soil " + str(i), entity="Ada", prop="soil", kind=0) for i in range(6)]
        hardness = torch.tensor([[0., -10., 10., 11., 12., 13., 14., 15.]])
        positive, negative = semantic_relation_masks(rows, facts, hardness)
        self.assertEqual(positive.nonzero().tolist(), [[0, 0]])
        self.assertEqual(negative.nonzero().tolist(), [[0, 1], [0, 5], [0, 6], [0, 7]])
        rows[0]["semantic_specific"] = False
        self.assertFalse(semantic_relation_masks(rows, facts, hardness)[1][0, 1])
        self.assertFalse(semantic_relation_masks(rows, facts, hardness, max_negatives=0)[1].any())

    def test_relation_margin_uses_cross_actor_support_and_no_foreign_negative(self):
        model = NeuralSlots()
        rows = [dict(entities=["Ada"], prop="light", kind=0, semantic_specific=True,
                     support_texts=["Ada. support"])]
        facts = [dict(text="Ada. support", entity="Ada", prop="light", kind=0),
                 dict(text="Ada. alternative", entity="Ada", prop="light", kind=0),
                 dict(text="Ada. wrong", entity="Ada", prop="soil", kind=0),
                 dict(text="Ben. support", entity="Ben", prop="light", kind=0),
                 dict(text="Ben. foreign", entity="Ben", prop="soil", kind=0)]
        logits = torch.tensor([[2., 1.6, 2.1, 0.8, 9.]], requires_grad=True)
        with patch.object(model, "semantic_bank_logits", return_value=logits):
            loss = semantic_relation_auxiliary(model, rows, facts, torch.zeros(1, QUERY),
                                              torch.zeros(5, 1, WIDTH), torch.ones(5, 1, dtype=torch.bool), memory_context_latents=(torch.zeros(5, 1, WIDTH)).new_zeros((len(torch.zeros(5, 1, WIDTH)), 16)))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertLess(float(logits.grad[0, 3]), float(logits.grad[0, 0]))
        self.assertGreater(float(logits.grad[0, 1]), 0)
        self.assertGreater(float(logits.grad[0, 2]), 0)
        self.assertEqual(float(logits.grad[0, 4]), 0)

    def test_relation_batch_covers_rare_relations_and_author_patterns(self):
        rows = [dict(text=str(i), prop="common") for i in range(100)]
        rows += [dict(text="rare", prop="rare")]
        batch = sample_relation_queries(rows, random.Random(671), 24)
        self.assertEqual(len(batch), 24)
        self.assertEqual(len({id(row) for row in batch}), 24)
        self.assertIn(rows[-1], batch)
        patterned = [dict(text=f"{pattern}-{actor}", prop="one-property", pattern_id=str(pattern))
                     for pattern in range(16) for actor in range(3)]
        batch = sample_relation_queries(patterned, random.Random(617), 24)
        self.assertEqual({r["pattern_id"] for r in batch}, {str(i) for i in range(16)})
        self.assertEqual(sample_relation_queries([], random.Random(617), 24), [])
        with self.assertRaisesRegex(ValueError, "evaluation"):
            sample_relation_queries([dict(text="heldout", prop="p", evaluation_only=True)], random.Random(671), 1)

    def test_relation_auxiliary_trains_semantics_without_false_negative_gradients(self):
        torch.manual_seed(4391)
        model = NeuralSlots()
        rows = [dict(entities=["Ada"], prop="light", kind=0, support_texts=["Ada. support"])]
        facts = [dict(text="Ada. support", entity="Ada", prop="light", kind=0),
                 dict(text="Ada. alternative", entity="Ada", prop="light", kind=0),
                 dict(text="Ben. foreign", entity="Ben", prop="soil", kind=0),
                 dict(text="Ada. wrong-relation", entity="Ada", prop="soil", kind=0),
                 dict(text="Ben. support", entity="Ben", prop="light", kind=0)]
        queries = torch.randn(1, QUERY, requires_grad=True)
        bank = torch.randn(len(facts), 3, WIDTH, requires_grad=True)
        mask = torch.ones(len(facts), 3, dtype=torch.bool)
        loss = semantic_relation_auxiliary(model, rows, facts, queries, bank, mask, memory_context_latents=(bank).new_zeros((len(bank), 16)))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.read[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.scope[0].weight.grad)
        self.assertGreater(float(queries.grad[:, :SEMANTIC].abs().sum()), 0)
        self.assertEqual(float(queries.grad[:, SEMANTIC:].abs().sum()), 0)
        self.assertGreater(float(bank.grad[0].abs().sum()), 0)
        self.assertGreater(float(bank.grad[3].abs().sum()), 0)
        self.assertGreater(float(bank.grad[4].abs().sum()), 0)
        self.assertFalse(supported(rows[0], facts[4]))
        self.assertEqual(float(bank.grad[1:3].abs().sum()), 0)
        for name, parameter in model.named_parameters():
            if name.startswith(("identity", "intent.", "version.", "plan_", "query_")):
                self.assertIsNone(parameter.grad, name)

    def test_relation_auxiliary_empty_support_and_negative_sets_are_finite(self):
        model = NeuralSlots()
        rows = [dict(entities=["Ada"], prop="light", kind=0, support_texts=[])]
        facts = [dict(text="Ada. alternative", entity="Ada", prop="light", kind=0)]
        q, bank = torch.randn(1, QUERY), torch.randn(1, 2, WIDTH)
        loss = semantic_relation_auxiliary(model, rows, facts, q, bank,
                                          torch.ones(1, 2, dtype=torch.bool), memory_context_latents=(bank).new_zeros((len(bank), 16)))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(model.read[0].weight.grad.abs().sum()), 0)

    def test_matcher_refinement_updates_only_read_and_preserves_scope_scores(self):
        torch.manual_seed(99351)
        model = NeuralSlots()
        for frozen_memory in (False, True):
            self.assertEqual(configure_reader_heads(model, "matcher", freeze_memory_encoder=frozen_memory), ("read.",))
            self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                             {"read.0.weight", "read.0.bias", "read.2.weight", "read.2.bias"})
        facts = [dict(text="Ada. I keep a notebook upstairs.", entity="Ada", reference="Ada",
                      prop="storage", kind=0, status=0),
                 dict(text="Ada. I use charcoal for murals.", entity="Ada", reference="Ada",
                      prop="art", kind=0, status=0),
                 dict(text="Ben. I keep a notebook upstairs.", entity="Ben", reference="Ben",
                      prop="storage", kind=0, status=0)]
        rows = [dict(text="Where does Ada store the notebook?", entities=["Ada"], reference="Ada",
                     prop="storage", kind=0, support_texts=[facts[0]["text"]])]
        texts = {f["text"] for f in facts} | {r["text"] for r in rows} | {"Ada", "Ben"}
        cache = {t:dict(x=torch.randn(len(t), INPUT) * .1,
                       bounds=[(i, i+1) for i in range(len(t))]) for t in texts}
        cached = DynamicFeatures(model, dict(fit=dict(facts=facts, queries=rows)), cache,
                                 "cpu", freeze_memory_encoder=True)
        topic = torch.nn.Linear(WIDTH, 2)
        optimizer = reader_optimizer(model, topic, "matcher", .01, .01)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in model.read.parameters()})
        self.assertFalse(optimized & {id(p) for p in topic.parameters()})
        self.assertEqual([g["name"] for g in optimizer.param_groups], ["reader"])
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        topic_before = {name:p.detach().clone() for name,p in topic.named_parameters()}
        model.eval()
        with torch.no_grad():
            bank, ids, mask, metadata = make_bank(facts, cached)
            q, identity = query(model, rows, cached, scope=True)
            scope_before = model.score_bank(q, identity, bank, ids, metadata, mask, scope=True, memory_context_latents=(q).new_zeros((len(mask), 16))).clone()
        model.train()
        optimizer.zero_grad()
        loss = semantic_training_loss(model, rows, facts, cached, topic, ["art", "storage"], topic_weight=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("read."):
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(all(p.grad is None for p in topic.parameters()))
        optimizer.step()
        changed = {name for name,p in model.named_parameters() if not torch.equal(p, before[name])}
        self.assertEqual(changed, {"read.0.weight", "read.0.bias", "read.2.weight", "read.2.bias"})
        for name,p in topic.named_parameters():
            torch.testing.assert_close(p, topic_before[name], atol=0, rtol=0)
        model.eval()
        with torch.no_grad():
            bank, ids, mask, metadata = make_bank(facts, cached)
            q_after, identity_after = query(model, rows, cached, scope=True)
            scope_after = model.score_bank(q_after, identity_after, bank, ids, metadata, mask, scope=True, memory_context_latents=(q_after).new_zeros((len(mask), 16)))
        torch.testing.assert_close(q_after, q, atol=0, rtol=0)
        torch.testing.assert_close(identity_after, identity, atol=0, rtol=0)
        torch.testing.assert_close(scope_after, scope_before, atol=0, rtol=0)

    def test_matcher_requires_zero_topic_weight_before_creating_training_output(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "must-not-be-created"
            args = SimpleNamespace(steps=1, train_heads="matcher", semantic_topic_weight=.05,
                                   output=str(destination))
            with self.assertRaisesRegex(ValueError, "matcher training requires --semantic-topic-weight 0"):
                train_reader(args)
            self.assertFalse(destination.exists())

    def test_semantic_learning_changes_read_queries_and_feedback_without_changing_scope(self):
        torch.manual_seed(84165)
        model = NeuralSlots()
        configure_reader_heads(model, "semantic", freeze_memory_encoder=True)
        facts = [dict(text="Ada. I moved to a coastal studio.", entity="Ada", reference="Ada",
                      prop="home", kind=0, status=0),
                 dict(text="Ada. I changed my daily shift.", entity="Ada", reference="Ada",
                      prop="work", kind=0, status=0)]
        initial = "Ada. Where is my studio?"
        rows = [dict(text=initial + "\nActivated evidence:\n" + facts[0]["text"] + "\n",
                     initial_text=initial, initial_reference="Ada", entities=["Ada"],
                     prop="home", kind=0, support_texts=[facts[0]["text"]])]
        scope_rows = [dict(text="Forget Ada's homes.", entities=["Ada"], prop="home", kind=0,
                           support_texts=[facts[0]["text"]])]
        groups = dict(fit=dict(facts=facts, queries=rows), scope_dev=dict(facts=facts, scope_queries=scope_rows))
        texts = {f["text"] for f in facts} | {rows[0]["text"], scope_rows[0]["text"], initial, "Ada"}
        cache = {t:dict(x=torch.randn(len(t), INPUT) * .1,
                       bounds=[(i, i+1) for i in range(len(t))]) for t in texts}
        cached = DynamicFeatures(model, groups, cache, "cpu", freeze_memory_encoder=True)
        frozen = {name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        original_feedback = model.read_update.weight.detach().clone()
        model.eval()
        with torch.no_grad():
            bank, ids, mask, metadata = make_bank(facts, cached)
            read_before = query(model, rows, cached)[0].clone()
            scope_q, scope_id = query(model, scope_rows, cached, scope=True)
            scales = model.attention_scales(scope=True).clone()
            score = model.score_bank(scope_q, scope_id, bank, ids, metadata, mask, scope=True, memory_context_latents=(scope_q).new_zeros((len(mask), 16))).clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        model.train()
        loss = semantic_training_loss(model, rows, facts, cached, torch.nn.Linear(WIDTH, 2),
                                      ["home", "work"], topic_weight=0)
        optimizer.zero_grad()
        loss.backward()
        self.assertGreater(float(model.query_encode.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.query_pool.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.read_update.weight.grad.abs().sum()), 0)
        for name,p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad, name)
        optimizer.step()
        self.assertFalse(torch.equal(original_feedback, model.read_update.weight))
        for name,p in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
        model.eval()
        with torch.no_grad():
            self.assertFalse(torch.equal(query(model, rows, cached)[0], read_before))
            with patch.object(model, "advance_query", side_effect=AssertionError("scope is a fresh source query")), \
                 patch.object(model, "advance_identity", side_effect=AssertionError("scope must not update identity")):
                after_q, after_id = query(model, scope_rows, cached, scope=True)
                scope_evaluate(model, groups["scope_dev"], cached)
            torch.testing.assert_close(after_q, scope_q, atol=0, rtol=0)
            torch.testing.assert_close(after_id, scope_id, atol=0, rtol=0)
            torch.testing.assert_close(model.attention_scales(scope=True), scales, atol=0, rtol=0)
            torch.testing.assert_close(model.score_bank(after_q, after_id, bank, ids, metadata, mask, scope=True, memory_context_latents=(after_q).new_zeros((len(mask), 16))),
                                       score, atol=0, rtol=0)

    def test_scope_teacher_trains_all_versions_without_recall_gradients(self):
        model = NeuralSlots()
        rows = [dict(text="Ada now has a different job.", entities=["Ada"], prop="job", support_texts=["Ada. old job", "Ada. new job"])]
        facts = [dict(text="Ada. old job", entity="Ada", prop="job", kind=0, status=1),
                 dict(text="Ada. new job", entity="Ada", prop="job", kind=0, status=0),
                 dict(text="Ada. old home", entity="Ada", prop="home", kind=0, status=1)]
        q, bank = torch.randn(1, QUERY), torch.randn(3, 2, WIDTH, requires_grad=True)
        loss = scope_relation_auxiliary(model, rows, facts, q, bank, torch.ones(3, 2, dtype=torch.bool), memory_context_latents=(bank).new_zeros((len(bank), 16)))
        loss.backward()
        self.assertGreater(float(bank.grad[0].abs().sum()), 0)
        self.assertGreater(float(bank.grad[1].abs().sum()), 0)
        self.assertGreater(float(model.scope[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.read[0].weight.grad)

    def test_identity_phase_only_match_role_update_train_and_read_phase_freezes_them(self):
        model = NeuralSlots()
        configure_reader_heads(model, "identity")
        trainable = {name for name,p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all(name.startswith(("identity_match.", "identity_update.", "role.")) for name in trainable))
        self.assertIn("identity_update.weight", trainable)
        self.assertNotIn("identity.weight", trainable)
        qid, mids = torch.randn(2, LEXICAL + WIDTH, requires_grad=True), torch.randn(3, LEXICAL + WIDTH, requires_grad=True)
        targets = torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        identity_calibration(model, qid, mids, targets).backward()
        self.assertGreater(float(model.identity_match[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(qid.grad)
        self.assertIsNone(mids.grad)
        self.assertIsNone(model.read[0].weight.grad)
        self.assertIsNone(model.scope[0].weight.grad)
        self.assertIsNone(model.identity.weight.grad)
        configure_reader_heads(model, "read")
        self.assertFalse(any(p.requires_grad for name,p in model.named_parameters() if name.startswith(("identity", "role."))))

    def test_scope_phase_trains_only_scope_and_ignores_recall_specific_negatives(self):
        torch.manual_seed(3187)
        model = NeuralSlots()
        self.assertEqual(configure_reader_heads(model, "scope"),
                         ("scope.", "scope_query_encode.", "scope_query_pool.", "scope_memory_attention_logits"))
        expected = {"scope.0.weight", "scope.0.bias", "scope.2.weight", "scope.2.bias",
                    "scope_query_encode.weight", "scope_query_encode.bias", "scope_query_pool.weight",
                    "scope_memory_attention_logits"}
        self.assertEqual({n for n, p in model.named_parameters() if p.requires_grad}, expected)
        facts = [dict(text="Ada. old job", entity="Ada", reference="Ada", prop="job", kind=0, status=1),
                 dict(text="Ada. new job", entity="Ada", reference="Ada", prop="job", kind=0, status=0),
                 dict(text="Ada. proposed job", entity="Ada", reference="Ada", prop="job", kind=2, status=3),
                 dict(text="Ada. old home", entity="Ada", reference="Ada", prop="home", kind=0, status=1),
                 dict(text="Ben. other job", entity="Ben", reference="Ben", prop="job", kind=0, status=0)]
        row = dict(text="Forget Ada's jobs.", entities=["Ada"], prop="job", kind=0,
                   semantic_specific=True, support_texts=[f["text"] for f in facts[:3]])
        group = dict(facts=facts, scope_queries=[row])
        texts = [f["text"] for f in facts] + ["Ada", "Ben", row["text"]]
        cache = {t: dict(x=torch.randn(2, INPUT), bounds=[(0, 1), (1, len(t))]) for t in texts}
        cached = DynamicFeatures(model, dict(scope_fit=group), cache, "cpu")
        with patch("train_neural_dialogue_reader.semantic_relation_masks", side_effect=AssertionError("recall teacher must be unused")), \
             patch.object(model, "advance_query", side_effect=AssertionError("scope must not use read feedback")), \
             patch.object(model, "advance_identity", side_effect=AssertionError("scope must not update identity")):
            loss = scope_training_loss(model, group, cached, random.Random(7))
            row["semantic_specific"] = False
            ordinary = scope_training_loss(model, group, cached, random.Random(7))
        torch.testing.assert_close(loss, ordinary, atol=0, rtol=0)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if n not in expected}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in expected:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in before:
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "development"):
            scope_training_loss(model, dict(group, scope_queries=[dict(row, evaluation_only=True)]), cached, random.Random(7))

    def test_scope_evaluation_uses_all_live_versions_and_exact_scope_support(self):
        model = NeuralSlots()
        facts = [dict(text="Ada. old job", entity="Ada", reference="Ada", prop="job", kind=0, status=1),
                 dict(text="deleted", entity="Ada", prop="job", kind=0, status=2),
                 dict(text="Ada. new job", entity="Ada", reference="Ada", prop="job", kind=0, status=0),
                 dict(text="Ada. planned job", entity="Ada", reference="Ada", prop="job", kind=2, status=3),
                 dict(text="Ben. other job", entity="Ben", reference="Ben", prop="job", kind=0, status=0)]
        rows = [dict(text="Forget Ada's jobs.", entities=["Ada"], prop="job", kind=0,
                     support_texts=[facts[i]["text"] for i in (0, 2, 3)]),
                dict(text="Unknown request", entities=[], prop="", support_texts=[])]
        live = active_bank_facts(facts)
        texts = [f["text"] for f in live] + ["Ada", "Ben"] + [r["text"] for r in rows]
        cache = {t: dict(x=torch.randn(1, INPUT), bounds=[(0, len(t))]) for t in texts}
        cached = DynamicFeatures(model, dict(scope_dev=dict(facts=live, scope_queries=rows)), cache, "cpu")
        def scores(q, identity, bank, ids, meta, mask, *, scope, memory_context_latents):
            self.assertTrue(scope)
            self.assertEqual(meta[:, :3].argmax(-1).tolist(), [1, 0, 2, 0])
            return torch.tensor([[1., 1., 1., -1.], [-1., -1., -1., 1.]])
        with patch.object(model, "score_bank", side_effect=scores):
            metrics = scope_evaluate(model, dict(facts=facts, scope_queries=rows), cached)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (3, 1, 0))
        self.assertEqual(metrics["bank_size"], 4)
        self.assertEqual(metrics["positive_support_set_exact"], 1.0)
        self.assertEqual(metrics["null_false_activation"], 1.0)
        self.assertEqual(metrics["support_set_exact"], 0.5)

    def test_read_evaluation_skips_deleted_slots_and_keeps_physical_age(self):
        model = NeuralSlots()
        facts = [dict(text="Ada. old job", entity="Ada", reference="Ada", prop="job", kind=0, status=1),
                 dict(text="deleted", entity="Ada", prop="job", kind=0, status=2),
                 dict(text="Ada. new job", entity="Ada", reference="Ada", prop="job", kind=0, status=0),
                 dict(text="Ada. planned job", entity="Ada", reference="Ada", prop="job", kind=2, status=3),
                 dict(text="Ben. other job", entity="Ben", reference="Ben", prop="job", kind=0, status=0)]
        rows = [dict(text="What is Ada's job?", entities=["Ada"], prop="job", kind=0,
                     support_texts=[facts[2]["text"]])]
        live = active_bank_facts(facts)
        texts = [f["text"] for f in live] + ["Ada", "Ben", rows[0]["text"]]
        cache = {t: dict(x=torch.randn(1, INPUT), bounds=[(0, len(t))]) for t in texts}
        cached = DynamicFeatures(model, dict(development=dict(facts=live, queries=rows)), cache, "cpu")
        self.assertNotIn("deleted", cached.index)
        def scores(q, identity, bank, ids, meta, mask, *, scope, memory_context_latents):
            self.assertFalse(scope)
            self.assertEqual(meta[:, :3].argmax(-1).tolist(), [1, 0, 2, 0])
            torch.testing.assert_close(meta[:, 3], torch.tensor([1., .5, .25, 0.]), atol=0, rtol=0)
            return torch.tensor([[-1., 1., -1., -1.]])
        with patch.object(model, "score_bank", side_effect=scores):
            metrics = reader_evaluate(model, dict(facts=facts, queries=rows), cached)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))
        self.assertEqual(metrics["bank_size"], 4)
        self.assertEqual(metrics["positive_support_set_exact"], 1.)

    def test_scope_selection_keeps_initial_eligible_and_prioritizes_wrong_matches(self):
        initial = dict(positive_support_set_exact=0.5, fp=4, queries=20,
                       activation_f1=0.8, null_false_activation=0)
        score = scope_selection_score(initial)
        self.assertEqual(score, scope_selection_score(dict(initial)))
        self.assertLess(scope_selection_score(dict(initial, positive_support_set_exact=0.4, fp=0)), score)
        self.assertGreater(scope_selection_score(dict(initial, fp=3, activation_f1=0.7)), score)
        self.assertGreater(scope_selection_score(dict(initial, activation_f1=0.9)), score)
        with self.assertRaisesRegex(ValueError, "positive"):
            scope_selection_score(dict(initial, positive_support_set_exact=None))

    def test_semantic_phase_learns_both_encoders_with_identity_and_other_heads_frozen(self):
        torch.manual_seed(6441)
        model = NeuralSlots()
        prefixes = ("read.", "encode.", "query_encode.", "query_pool.", "read_update.", "memory_attention_logits", "read_context_memory.", "read_context_query.", "read_context_delta.")
        self.assertEqual(configure_reader_heads(model, "semantic"), prefixes)
        expected = {name for name, _ in model.named_parameters() if name.startswith(prefixes)}
        self.assertEqual(len(expected), 15)
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad}, expected)
        facts = [dict(text="Ada. I mend clocks.", entity="Ada", reference="Ada", prop="job", kind=0, status=0),
                 dict(text="Ada. I mend radios.", entity="Ada", reference="Ada", prop="job", kind=0, status=0),
                 dict(text="Ada. I live in a flat.", entity="Ada", reference="Ada", prop="home", kind=0, status=0),
                 dict(text="Ben. I mend clocks.", entity="Ben", reference="Ben", prop="job", kind=0, status=0)]
        initial = "What does Ada repair instead of radios?"
        row = dict(text=initial + "\nActivated evidence:\nBen. I live nearby.\n", initial_text=initial,
                   initial_reference="Ada", entities=["Ada"], prop="job", kind=0,
                   semantic_specific=True, support_texts=[facts[0]["text"]])
        group = dict(facts=facts, queries=[row])
        texts = [f["text"] for f in facts] + ["Ada", "Ben", initial, row["text"]]
        cache = {t: dict(x=torch.randn(3, INPUT), bounds=[(0, 1), (1, 2), (2, len(t))]) for t in texts}
        cached = DynamicFeatures(model, dict(fit=group), cache, "cpu")
        topic_head = torch.nn.Linear(WIDTH, 3)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if name not in expected}
        identity_logits = model.identity_logits
        def checked_identity(qid, ids):
            self.assertFalse(qid.requires_grad)
            self.assertFalse(ids.requires_grad)
            return identity_logits(qid, ids)
        original_query, original_bank = query, make_bank
        captured = {}
        def capture_query(*args):
            q, identity = original_query(*args)
            q.retain_grad()
            captured["query"] = q
            return q, identity
        def capture_bank(*args):
            result = original_bank(*args)
            result[-1].retain_grad()
            captured["metadata"] = result[-1]
            return result
        with patch.object(model, "identity_logits", side_effect=checked_identity), \
             patch("train_neural_dialogue_reader.query", side_effect=capture_query), \
             patch("train_neural_dialogue_reader.make_bank", side_effect=capture_bank), \
             patch("train_neural_dialogue_reader.scope_relation_auxiliary", side_effect=AssertionError("scope teacher must be unused")), \
             patch("train_neural_dialogue_reader.identity_calibration", side_effect=AssertionError("identity calibration must be unused")), \
             patch("train_neural_dialogue_reader.role_supervision", side_effect=AssertionError("role supervision must be unused")):
            loss = semantic_training_loss(model, [row], facts, cached, topic_head, ["job", "home", ""])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(captured["metadata"].grad)
        self.assertGreater(float(captured["query"].grad[:, :SEMANTIC].abs().sum()), 0)
        self.assertEqual(float(captured["query"].grad[:, SEMANTIC:].abs().sum()), 0)
        for name, parameter in model.named_parameters():
            if name in expected:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if name in ("read_context_memory.weight", "read_context_query.weight"):
                    self.assertEqual(float(parameter.grad.abs().sum()), 0., name)  # Zero output learns first.
                else:
                    self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(float(topic_head.weight.grad.abs().sum()), 0)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad] + list(topic_head.parameters()), lr=0.01)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in before:
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)

    def test_query_only_semantic_phase_keeps_memory_and_kind_stable_with_dropout(self):
        torch.manual_seed(3771)
        model = NeuralSlots()
        heads = configure_reader_heads(model, "semantic", freeze_memory_encoder=True)
        self.assertEqual(heads, ("read.", "query_encode.", "query_pool.", "read_update.", "memory_attention_logits", "read_context_memory.", "read_context_query.", "read_context_delta."))
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 13)
        with self.assertRaisesRegex(ValueError, "semantic"):
            configure_reader_heads(model, "scope", freeze_memory_encoder=True)
        facts = [dict(text="Ada. I mend clocks.", entity="Ada", reference="Ada", prop="job", kind=0),
                 dict(text="Ada. I live nearby.", entity="Ada", reference="Ada", prop="home", kind=0)]
        initial = "What does Ada repair?"
        row = dict(text=initial + "\nActivated evidence:\nA neighbor lives nearby.\n",
                   initial_text=initial, entities=["Ada"], prop="job", kind=0,
                   support_texts=[facts[0]["text"]])
        group = dict(facts=facts, queries=[row])
        texts = [f["text"] for f in facts] + ["Ada", initial, row["text"]]
        cache = {text:dict(x=torch.randn(3, INPUT), bounds=[(0, 1), (1, 2), (2, len(text))]) for text in texts}
        cached = DynamicFeatures(model, dict(fit=group), cache, "cpu", 0.5, freeze_memory_encoder=True)
        memory, mask, _ = cached.encode([f["text"] for f in facts])
        self.assertFalse(memory.requires_grad)
        torch.testing.assert_close(memory, cached.encode([f["text"] for f in facts])[0], atol=0, rtol=0)
        first_query = cached.encode([row["text"]], query=True)[0]
        self.assertFalse(torch.equal(first_query, cached.encode([row["text"]], query=True)[0]))
        before_kind = kind_bank_diagnostics(model, group, cached)
        frozen = {name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        topic_head = torch.nn.Linear(WIDTH, 3)
        loss = semantic_training_loss(model, [row], facts, cached, topic_head, ["job", "home", ""])
        loss.backward()
        self.assertIsNone(model.encode.weight.grad)
        self.assertIsNone(model.kind.weight.grad)
        self.assertGreater(float(model.query_encode.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.read[0].weight.grad.abs().sum()), 0)
        torch.optim.SGD(trainable + list(topic_head.parameters()), lr=0.01).step()
        torch.testing.assert_close(memory, cached.encode([f["text"] for f in facts])[0], atol=0, rtol=0)
        self.assertEqual(before_kind, kind_bank_diagnostics(model, group, cached))
        for name,p in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)

    def test_kind_diagnostics_compare_source_text_kind_separately_from_slot_status(self):
        model = NeuralSlots()
        facts = [dict(text="Ada. old", entity="Ada", reference="Ada", prop="job", kind=0, status=1),
                 dict(text="deleted", entity="Ada", prop="job", kind=0, status=2),
                 dict(text="Ada. planned", entity="Ada", reference="Ada", prop="job", kind=2, status=3),
                 dict(text="Ada. current", entity="Ada", reference="Ada", prop="job", kind=0, status=0)]
        active = active_bank_facts(facts)
        cache = {text:dict(x=torch.randn(1, INPUT), bounds=[(0, len(text))]) for text in [f["text"] for f in active] + ["Ada"]}
        cached = DynamicFeatures(model, dict(dev=dict(facts=active)), cache, "cpu")
        with patch.object(model.kind, "forward", return_value=torch.tensor([[3., 1., 0.], [0., 1., 4.], [0., 1., 2.]])):
            result = kind_bank_diagnostics(model, dict(facts=facts), cached)
        self.assertEqual(result["facts"], 3)
        self.assertAlmostEqual(result["source_kind_accuracy"], 2/3, places=6)
        self.assertEqual(result["prediction_counts"], [1, 0, 2])
        self.assertEqual(result["teacher_counts"], [2, 0, 1])
        self.assertEqual(result["bank_status_counts"], [1, 1, 1])
        self.assertEqual(result["confusion"], [[1, 0, 1], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(result["minimum_top2_margin"], 1.0)
        self.assertIsNone(kind_bank_diagnostics(model, dict(facts=[]), cached)["source_kind_accuracy"])

    def test_attention_temperature_has_its_own_optimizer_group_without_duplicate_parameters(self):
        model = NeuralSlots()
        topic = torch.nn.Linear(WIDTH, 3)
        for mode in ("all", "read", "semantic"):
            configure_reader_heads(model, mode)
            self.assertTrue(model.memory_attention_logits.requires_grad)
        configure_reader_heads(model, "semantic", freeze_memory_encoder=True)
        optimizer = reader_optimizer(model, topic, "semantic", 3e-5, 0.01)
        groups = {group["name"]:group for group in optimizer.param_groups}
        self.assertEqual(groups["memory_attention"]["lr"], 0.01)
        self.assertEqual(groups["reader"]["lr"], 3e-5)
        self.assertEqual(groups["memory_attention"]["weight_decay"], 0)
        self.assertEqual([id(p) for p in groups["memory_attention"]["params"]], [id(model.memory_attention_logits)])
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 15)  # 9 query/read + 3 context tensors, 1 temperature, 2 topic tensors.
        # Defend against an accidentally aliased auxiliary module/parameter list.
        alias = reader_optimizer(model, model, "semantic", 3e-5, 0.01)
        aliased = [id(p) for group in alias.param_groups for p in group["params"]]
        self.assertEqual(len(aliased), 13)
        self.assertEqual(len(aliased), len(set(aliased)))
        for mode in ("scope", "identity"):
            configure_reader_heads(model, mode)
            self.assertFalse(model.memory_attention_logits.requires_grad)
            frozen = reader_optimizer(model, topic, mode, 3e-5, 0.01)
            self.assertEqual(len(frozen.param_groups), 2 if mode == "scope" else 1)
            self.assertFalse(any(p is model.memory_attention_logits for group in frozen.param_groups for p in group["params"]))
            if mode == "scope":
                group = frozen.param_groups[1]
                self.assertEqual(group["name"], "scope_memory_attention")
                self.assertEqual(group["lr"], .01)
                self.assertEqual([id(p) for p in group["params"]], [id(model.scope_memory_attention_logits)])
            optimized = [id(p) for group in frozen.param_groups for p in group["params"]]
            self.assertEqual(len(optimized), len(set(optimized)))
        with self.assertRaisesRegex(ValueError, "finite"):
            reader_optimizer(model, topic, "identity", 3e-5, float("nan"))

    def test_attention_diagnostics_measure_scaled_memory_tokens_and_ignore_padding(self):
        model = NeuralSlots()
        bank = torch.zeros(2, 3, WIDTH)
        bank[0, :2, 0] = torch.tensor([1., -1.])
        bank[:, 2, :] = 1e6  # Padded tokens must not affect entropy or peak mass.
        mask = torch.tensor([[True, True, False], [True, False, False]])
        q = torch.zeros(1, QUERY)
        q[0, 0], q[0, WIDTH] = 1, -1
        cached = SimpleNamespace(encode=lambda texts:(bank, mask, None))
        group = dict(facts=[dict(text="a", kind=0), dict(text="b", kind=0)], queries=[dict(text="q")])
        with patch("train_neural_dialogue_reader.query", return_value=(q, None)):
            baseline = attention_bank_diagnostics(model, group, cached)
            with torch.no_grad():
                model.memory_attention_logits[:2].fill_(4)
            sharp = attention_bank_diagnostics(model, group, cached)
        self.assertEqual(baseline["scales"], [1.] * 4)
        self.assertEqual(baseline["single_token_facts"], 1)
        self.assertLess(sharp["head_normalized_entropy"][0], baseline["head_normalized_entropy"][0])
        self.assertGreater(sharp["head_maximum_weight"][0], baseline["head_maximum_weight"][0])
        self.assertAlmostEqual(sharp["head_normalized_entropy"][2], 1.0, places=6)
        self.assertTrue(all(0 <= value <= 1 for value in sharp["head_normalized_entropy"]))
        self.assertIsNone(attention_bank_diagnostics(model, dict(facts=[], queries=[]), cached)["normalized_entropy"])

    def test_scope_attention_diagnostics_use_scope_rows_queries_and_temperature(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.scope_memory_attention_logits.copy_(torch.tensor([1., -.4, .5, .8]))
        bank = torch.zeros(1, 3, WIDTH)
        bank[0, :2, 0] = torch.tensor([1., -1.])
        bank[0, 2] = 1e6
        mask = torch.tensor([[True, True, False]])
        read_q, scope_q = torch.zeros(1, QUERY), torch.zeros(1, QUERY)
        scope_q[:, :SEMANTIC].reshape(-1, WIDTH)[:, 0] = 4
        group = dict(facts=[dict(text="fact", kind=0)], queries=[dict(text="read")],
                     scope_queries=[dict(text="scope")])
        cached = SimpleNamespace(encode=lambda texts:(bank, mask, None))
        def routed_query(_model, rows, _cached, *, scope=False):
            self.assertEqual(rows, group["scope_queries" if scope else "queries"])
            return (scope_q if scope else read_q), None
        with patch("train_neural_dialogue_reader.query", side_effect=routed_query):
            read = attention_bank_diagnostics(model, group, cached)
            scope = attention_bank_diagnostics(model, group, cached, scope=True)
        scales = model.attention_scales(scope=True)
        expected = torch.stack((.5 * scales, -.5 * scales), -1).softmax(-1)
        entropy = -(expected * expected.log()).sum(-1) / math.log(2)
        self.assertEqual(read["channel"], "read")
        self.assertEqual(scope["channel"], "scope")
        self.assertEqual(read["scales"], [1.] * 4)
        self.assertEqual(scope["scales"], scales.tolist())
        torch.testing.assert_close(torch.tensor(scope["head_normalized_entropy"]), entropy.detach())
        torch.testing.assert_close(torch.tensor(scope["head_maximum_weight"]), expected.max(-1).values.detach())
        self.assertAlmostEqual(read["normalized_entropy"], 1., places=6)
        self.assertLess(scope["normalized_entropy"], read["normalized_entropy"])

    def test_physical_states_preserve_insert_age_and_separate_text_kind(self):
        model = NeuralSlots()
        facts = [dict(text="Ada old", entity="Ada", reference="Ada", prop="job", kind=0, status=1),
                 dict(text="deleted", entity="Ada", prop="job", kind=0, status=2),
                 dict(text="Ada new", entity="Ada", reference="Ada", prop="job", kind=0, status=0),
                 dict(text="Ada planned", entity="Ada", reference="Ada", prop="job", kind=2, status=3)]
        active = active_bank_facts(facts)
        cache={text:dict(x=torch.randn(1,INPUT),bounds=[(0,len(text))]) for text in [f['text'] for f in active]+['Ada']}
        cached=DynamicFeatures(model,dict(fit=dict(facts=active,queries=[])),cache,"cpu")
        _,_,_,meta=make_bank(active,cached)
        self.assertEqual(meta[:,:3].argmax(-1).tolist(),[1,0,2])
        torch.testing.assert_close(meta[:,3],torch.tensor([1.,1/3,0.]))
        self.assertEqual([f['kind'] for f in active],[0,0,2])
        with self.assertRaises(ValueError):
            make_bank([facts[1]],cached)
        targets=intent_targets([dict(kind=1),dict(kind=2),dict(),dict(kind=0,allowed_kinds=[0,1,2])],"cpu")
        self.assertEqual(targets.tolist(),[[0.,1.,0.],[0.,0.,1.],[1.,1.,1.],[1.,1.,1.]])

    def test_feature_extraction_repeated_phases_use_exclusive_temporary_output(self):
        from neural_training_common import extract, load_features
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe, gguf = root / "probe", root / "backbone"
            probe.write_bytes(b"fake feature transport probe")
            gguf.write_bytes(b"fake identity only")
            args = SimpleNamespace(output=str(root), probe=str(probe), gguf=str(gguf))
            destinations = []
            def fake_probe(command, check):
                source, destination = Path(command[2]), Path(command[3])
                destinations.append(destination)
                with source.open("rb") as f, destination.open("xb") as out:
                    count = struct.unpack("<I", f.read(4))[0]
                    out.write(b"BNSFEAT1" + struct.pack("<I", count))
                    for _ in range(count):
                        length = struct.unpack("<I", f.read(4))[0]
                        f.read(length)
                        out.write(struct.pack("<III", 1, 0, length))
                        out.write(np.ones((1, INPUT), dtype="<f4").tobytes())
            with patch("neural_training_common.subprocess.run", side_effect=fake_probe):
                first, _ = extract(args, dict(fit=[dict(body="one", context="speaker one")]))
                second, _ = extract(args, dict(fit=[dict(body="two", context="speaker two")]))
            self.assertNotEqual(destinations[0].parent, destinations[1].parent)
            self.assertTrue(all(not p.exists() for p in destinations))
            self.assertTrue(all(x["x"].shape == (1, INPUT) for x in [*first.values(), *second.values()]))
            self.assertEqual(set(load_features(root / "features.pt")["features"]), set(second))
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["backbone", "features.pt", "probe"])

    def test_fit_replay_preserves_all_lifecycle_and_rare_action_cases(self):
        from prepare_neural_dialogue import dialogue_curriculum
        source = dialogue_curriculum("fit")
        replay = bounded_dialogue_replay(source)
        self.assertEqual(len(replay), 512)
        self.assertEqual(replay, bounded_dialogue_replay(source))
        ids = {r["source_id"] for r in replay}
        required = [r for r in source if r["tag"] in ("lifecycle", "reference-update")
                    or any(s["action"] in (2, 3) for s in r["spans"])]
        self.assertTrue(all(r["source_id"] in ids for r in required))
        for action in (2, 3):
            self.assertEqual(sum(s["action"] == action for r in replay for s in r["spans"]),
                             sum(s["action"] == action for r in source for s in r["spans"]))

    def test_positive_batch_reserves_rare_action_coverage_without_reweighting_loss(self):
        rows = [dict(source_id=f"{action}-{i}", spans=[dict(action=action)])
                for action, count in ((1, 1000), (2, 6), (3, 6)) for i in range(count)]
        negatives = [dict(source_id=f"negative-{i}", spans=[]) for i in range(20)]
        batch = sample_action_balanced_batch(rows, negatives, random.Random(72))
        self.assertEqual(len(batch), 20)
        self.assertEqual(len({r["source_id"] for r in batch}), 20)
        self.assertEqual(sum(not r["spans"] for r in batch), 8)
        for action, expected in ((1, 8), (2, 2), (3, 2)):
            self.assertEqual(sum(bool(r["spans"]) and r["spans"][0]["action"] == action for r in batch), expected)
        tiny = sample_action_balanced_batch(rows[:2], negatives[:1], random.Random(72))
        self.assertEqual(len(tiny), 3)

    def query_sample(self, need):
        body = "Where does she work?" if need else "Where does Cyra work?"
        context = observation_context()
        row = dict(
            body=body,
            context=context,
            speaker=None,
            timestamp=None,
            spans=[],
            whole_need=need,
            whole_refs=[],
        )
        cache = {
            text: dict(
                x=torch.randn(len(text), INPUT) * 0.1,
                bounds=[(i, i + 1) for i in range(len(text))],
            )
            for text in (body, context)
        }
        item = prepare(dict(test=[row]), cache, "cpu")[body, context]
        model = NeuralSlots()
        return model, row, item

    def test_self_contained_query_loss_leaves_write_binding_untouched(self):
        model, row, item = self.query_sample(False)
        observation_loss(model, row, item).backward()
        self.assertGreater(float(model.query_need.bias.grad), 0)
        self.assertGreater(float(model.plan_encode.weight.grad.abs().sum()), 0)
        for prefix in (
            "plan_need.",
            "plan_ref_query.",
            "plan_origin.",
            "query_ref_query.",
            "query_origin.",
            "plan_ref_encode.",
            "plan_ref_pair.",
        ):
            for name, parameter in model.named_parameters():
                if name.startswith(prefix):
                    self.assertIsNone(parameter.grad, name)

    def test_ambiguous_query_retains_need_and_empty_source_supervision(self):
        model, row, item = self.query_sample(True)
        observation_loss(model, row, item).backward()
        # A truly unresolved query must learn that a referent is required,
        # while every candidate source stays outside the predicted BIO span.
        self.assertLess(float(model.query_need.bias.grad), 0)
        self.assertLess(float(model.query_ref_pair[2].bias.grad[0]), 0)
        self.assertGreater(float(model.query_origin.bias.grad.sum()), 0)
        self.assertGreater(float(model.query_ref_query.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.query_ref_encode.weight.grad.abs().sum()), 0)
        for prefix in ("plan_need.", "plan_ref_query.", "plan_origin."):
            for name, parameter in model.named_parameters():
                if name.startswith(prefix):
                    self.assertIsNone(parameter.grad, name)

        # Need=True and no selected original source remains unresolved;
        # adding the query head must not turn this into unrestricted recall.
        with torch.no_grad():
            model.query_need.weight.zero_()
            model.query_need.bias.fill_(4)
            model.query_origin.weight.zero_()
            model.query_origin.bias.zero_()
            model.query_ref_pair[2].weight.zero_()
            model.query_ref_pair[2].bias.copy_(torch.tensor([4.0, -4.0, -4.0]))
            encoded, _, _ = model.plan(item["body"], item["context"], item["body_words"])
            need, tags = model.query_reference(
                encoded.mean(0),
                source_words(item, len(encoded)),
                source_types(row, item, len(encoded)),
                raw_context=model.query_binding_context(item["body"], item["context"]),
            )
            refs = decode_references(row, tags.argmax(-1), item, len(encoded))
        self.assertGreater(float(need), 0)
        self.assertEqual(refs, [])
        self.assertTrue(bool(need > 0) and not refs)

    def test_evidence_curriculum_preserves_target_with_wrong_cue(self):
        from neural_reader_evidence_curriculum import augment_evidence
        from copy import deepcopy

        facts = [
            dict(text="Cyra. I live in Bergen.", entity="Cyra", prop="city", kind=0),
            dict(text="Cyra. I enjoy music.", entity="Cyra", prop="hobby", kind=0),
            dict(text="Mavis. I live in Rome.", entity="Mavis", prop="city", kind=0),
        ]
        question = dict(
            text="Where does Cyra live?",
            entities=["Cyra"],
            prop="city",
            kind=0,
            support_texts=[facts[0]["text"]],
        )
        original = dict(facts=facts, queries=[question])
        before = deepcopy(original)
        augmented = augment_evidence(original, 83, 600)
        self.assertEqual(original, before)
        self.assertEqual(augmented, augment_evidence(original, 83, 600))
        self.assertEqual(
            {q.get("evidence_type") for q in augmented["queries"][1:]},
            {"supporting", "same_actor_wrong_relation", "other_actor"},
        )
        for row in augmented["queries"][1:]:
            self.assertEqual(row["initial_text"], question["text"])
            self.assertEqual(row["support_texts"], question["support_texts"])
            self.assertTrue(supported(row, facts[0]))
            self.assertFalse(supported(row, facts[1]))
            self.assertTrue(
                row["text"].startswith(question["text"] + "\nActivated evidence:\n")
            )

    def test_predicted_kind_metadata_matches_native_argmax(self):
        model = NeuralSlots()
        text = "Cyra. I may travel."
        cache = {
            text: dict(
                x=torch.randn(len(text), INPUT),
                bounds=[(i, i + 1) for i in range(len(text))],
            )
        }
        fact = dict(text=text, kind=0)
        dynamic = DynamicFeatures(model, dict(test=dict(facts=[fact])), cache, "cpu")
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([0.0, -1.0, 2.0]))
        _, _, _, meta = make_bank([fact], dynamic)
        torch.testing.assert_close(meta[0, :3], torch.tensor([0.0, 0.0, 1.0]))
        meta[0, 0].backward()
        self.assertGreater(float(model.kind.bias.grad.abs().sum()), 0)

    def test_legal_sequence_partition_and_decoder_match_valid_paths(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_transition.copy_(torch.randn(5, 5))
            model.plan_transition[0, 4] = 100
        logits = torch.randn(3, 5, requires_grad=True)
        ends = torch.tensor([0.2, -0.9, 1.7], requires_grad=True)
        paths = [
            p
            for p in itertools.product(range(5), repeat=3)
            if p[0] != 4 and not any(a == 0 and b == 4 for a, b in zip(p, p[1:]))
        ]
        scores = torch.stack(
            [
                model.plan_start[p[0]]
                + model.plan_end[p[-1]]
                + sum(logits[i, t] for i, t in enumerate(p))
                + sum(model.plan_transition[a, b] for a, b in zip(p, p[1:]))
                + sum((F.logsigmoid(-ends[i] if b == 4 else ends[i]) + math.log(2))
                      for i, (a, b) in enumerate(zip(p, p[1:])) if a > 0)
                + ((F.logsigmoid(ends[-1]) + math.log(2)) if p[-1] > 0 else 0)
                for p in paths
            ]
        )
        self.assertEqual(
            model.plan_labels(logits, ends).tolist(), list(paths[int(scores.argmax())])
        )
        target = torch.tensor(paths[17])
        torch.testing.assert_close(
            model.sequence_loss(logits, target, ends), (scores.logsumexp(0) - scores[17]) / 3
        )
        expected_grad = torch.autograd.grad((scores.logsumexp(0) - scores[17]) / 3,
                                            (logits, ends), retain_graph=True)
        model.sequence_loss(logits, target, ends).backward()
        torch.testing.assert_close(logits.grad, expected_grad[0])
        torch.testing.assert_close(ends.grad, expected_grad[1])
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(ends.grad).all())
        self.assertGreater(float(ends.grad.abs().sum()), 0)
        self.assertEqual(float(model.plan_transition.grad[0, 4]), 0)

    def test_source_origin_is_a_soft_learned_multilabel_bias(self):
        model = NeuralSlots()
        candidate = torch.randn(64)
        words = torch.randn(4, INPUT)
        kinds = torch.eye(4)
        _, before = model.plan_reference(candidate, words, kinds)
        with torch.no_grad():
            model.plan_origin.bias.copy_(torch.tensor([2.0, -3.0, 1.0, -4.0]))
        _, after = model.plan_reference(candidate, words, kinds)
        torch.testing.assert_close(after[:, 0], before[:, 0])
        torch.testing.assert_close(
            after[:, 1:] - before[:, 1:], model.plan_origin.bias[:, None].expand(4, 2)
        )
        torch.nn.functional.cross_entropy(after, torch.tensor([1, 0, 2, 0])).backward()
        self.assertGreater(float(model.plan_origin.weight.grad.abs().sum()), 0)

    def test_feature_cache_rejects_nonfinite_values_and_wrong_geometry(self):
        from neural_training_common import load_features, feature_texts_sha256

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            sample = dict(
                x=torch.ones(2, INPUT),
                bounds=torch.tensor([[0, 1], [1, 2]], dtype=torch.uint32),
            )
            saved = dict(features={"hi": sample}, identity=dict(backbone="a" * 64, probe="b" * 64,
                         texts_sha256=feature_texts_sha256(["hi"])))
            torch.save(saved, path)
            self.assertEqual(set(load_features(path)["features"]), {"hi"})
            sample["x"][0, 0] = float("nan")
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "invalid frozen C feature"):
                load_features(path)
            sample["x"] = torch.zeros(2, INPUT - 1)
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "invalid frozen C feature"):
                load_features(path)

    def test_missing_native_write_counts_as_missing_activation(self):
        from eval_neural_episodes_native import score

        facts = [
            dict(
                content="I repair clocks.",
                reference="Cyra",
                text="Cyra. I repair clocks.",
                entity="Cyra",
                prop="job",
                kind=0,
            )
        ]
        query = dict(entities=["Cyra"], prop="job", kind=0)
        report, records = score(
            [],
            facts,
            [(dict(body="What does Cyra do?"), query, dict(activated_ids=[]))],
        )
        self.assertEqual(report["fn"], 1)
        self.assertEqual(report["activation_f1"], 0)
        self.assertEqual(report["end_to_end_positive_exact"], 0)
        self.assertFalse(records[0]["exact"])

    def test_native_support_accuracy_reports_unconverged_activation_separately(self):
        from eval_neural_episodes_native import score

        fact = dict(content="I repair clocks.", reference="Cyra",
                    text="Cyra. I repair clocks.", entity="Cyra", prop="job", kind=0)
        slot = dict(id=9, content=fact["content"], reference=fact["reference"])
        label = dict(entities=["Cyra"], prop="job", kind=0)
        outputs = [
            (dict(body="What does Cyra do?"), label,
             dict(activated_ids=[9], complete=complete))
            for complete in (True, False)
        ]
        report, records = score([slot], [fact], outputs)
        self.assertEqual(report["end_to_end_positive_exact"], 1)
        self.assertEqual(report["incomplete_queries"], 1)
        self.assertEqual(report["converged_positive_exact"], 0.5)
        self.assertTrue(all(record["exact"] for record in records))

    def test_specific_episode_question_excludes_other_same_topic_fact(self):
        group = add_episodes(dict(facts=[], queries=[]), "fit")
        q = next(q for q in group["queries"] if "warehouse packing" in q["text"])
        candidates = [
            f
            for f in group["facts"]
            if f["entity"] in q["entities"] and f["prop"] == q["prop"]
        ]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(sum(supported(q, f) for f in candidates), 1)
        self.assertTrue(
            all("warehouse packing" in f["text"] for f in candidates if supported(q, f))
        )

    def test_live_identity_matches_inference_and_receives_gradients(self):
        model = NeuralSlots()
        texts = ["Mavis", "Where does Mavis live?"]
        cache = {
            t: dict(
                x=torch.randn(len(t), INPUT) * 0.1,
                bounds=[(i, i + 1) for i in range(len(t))],
            )
            for t in texts
        }
        rows = [dict(text=t) for t in texts]
        dynamic = DynamicFeatures(model, dict(test=dict(queries=rows)), cache, "cpu")
        actual = dynamic.reference_ids(rows)
        from neural_slot_model import identity_word_features

        expected = torch.stack(
            [
                model.prepare((cache[t]["x"], identity_word_features(t, cache[t])[0]))[
                    1
                ]
                for t in texts
            ]
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        (actual[0] * actual[1]).sum().backward()
        self.assertGreater(float(model.role[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.identity.weight.grad.abs().sum()), 0)

    def test_episode_subject_ranges_do_not_label_every_repeated_name(self):
        rows = episode_curriculum("fit")
        row = next(
            r for r in rows if r["tag"] == "episode-fact" and r["spans"][0]["start"] > 0
        )
        self.assertGreater(row["context"].count(row["speaker"]), 1)
        units = [(i, i + 1) for i in range(len(row["context"].encode()))]
        gold = reference_targets(row, row["spans"][0], units, [], 0, "cpu")
        self.assertEqual(int((gold > 0).sum()), len(row["speaker"].encode()))
        self.assertTrue(torch.all(gold[9 : 9 + len(row["speaker"].encode())] > 0))
        self.assertEqual(int((gold[9 + len(row["speaker"].encode()) :] > 0).sum()), 0)
        for a, b in itertools.combinations(EPISODE_NAMES, 2):
            self.assertFalse(set(EPISODE_NAMES[a]) & set(EPISODE_NAMES[b]))

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5308)

    def test_read_update_uses_original_and_current_query(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.read_update.bias.zero_()
            model.read_update.weight.zero_()
            model.read_update.weight[0, QUERY] = -0.25
        original = torch.zeros(QUERY)
        current = torch.ones(QUERY) * 2
        expected = torch.ones(QUERY)
        expected[0] = 2 / (1 + torch.exp(torch.tensor(0.5)))
        torch.testing.assert_close(model.advance_query(original, current), expected)
        model.advance_query(original, current).sum().backward()
        self.assertGreater(float(model.read_update.weight.grad.abs().sum()), 0)

    def test_batch_loss_gradient_is_invariant_to_order_and_repetition(self):
        model = NeuralSlots()
        context = observation_context("Cyra")
        rows = [
            dict(
                body="I live.",
                context=context,
                speaker="Cyra",
                timestamp=None,
                spans=[dict(start=0, end=7, action=1, need=True, references=["Cyra"])],
                whole_need=True,
                whole_refs=["Cyra"],
            ),
            dict(
                body="Hello.",
                context=context,
                speaker="Cyra",
                timestamp=None,
                spans=[],
                whole_need=False,
                whole_refs=[],
            ),
        ]
        cache = {
            t: dict(
                x=torch.randn(len(t), INPUT) * 0.1,
                bounds=[(i, i + 1) for i in range(len(t))],
            )
            for t in ("I live.", "Hello.", context)
        }
        values = prepare(dict(test=rows), cache, "cpu")

        def gradient(order):
            model.zero_grad()
            loss = batch_loss(model, order, values)
            loss.backward()
            return float(loss.detach()), torch.cat(
                [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            )

        loss, derivative = gradient(rows)
        for order in (rows[::-1], rows + rows):
            actual, gradients = gradient(order)
            self.assertAlmostEqual(actual, loss, places=6)
            torch.testing.assert_close(gradients, derivative, atol=1e-7, rtol=1e-5)

    def test_dynamic_padding_matches_individual_inference_with_reference(self):
        model = NeuralSlots()
        texts = ["Cyra", "Cyra. I live.", "Cyra. Where do I live?"]
        cache = {
            t: dict(
                x=torch.randn(len(t), INPUT) * 0.1,
                bounds=[(i, i + 1) for i in range(len(t))],
            )
            for t in texts
        }
        facts = [dict(text=texts[1], reference="Cyra", kind=0)]
        questions = [dict(text=texts[2], reference="Cyra")]
        dynamic = DynamicFeatures(
            model, dict(test=dict(facts=facts, queries=questions)), cache, "cpu"
        )
        bank, ids, mask, meta = make_bank(facts, dynamic)
        q, identity = query(model, questions, dynamic)

        def value(text):
            words, _ = identity_word_features(text, cache[text])
            return cache[text]["x"], words

        enc, _ = model.prepare(value(texts[1]))
        individual, _ = model.query(value(texts[2]))
        _, expectedid = model.prepare(value("Cyra"))
        torch.testing.assert_close(bank[0][mask[0]], enc, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(q[0], individual, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(ids[0], expectedid, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(identity[0], expectedid, atol=2e-6, rtol=1e-5)

    def test_component_composition_preserves_writer_and_rejects_untrained_heads(self):
        from neural_slot_model import PLAN_KEYS, QUERY_KEYS
        model = NeuralSlots()
        writer = dict(
            backbone_sha256="00" * 32,
            state_dict=model.state_dict(),
            format=FORMAT,
        )
        allowed = configure_reader_heads(model, "semantic", freeze_memory_encoder=True)
        reader = dict(
            writer,
            state_dict={k: v.clone() for k, v in model.state_dict().items()},
            reader_heads=allowed,
            reader_step=1,
        )
        sources = set(PLAN_KEYS + QUERY_KEYS)
        self.assertEqual(len(sources), 46)
        for key, value in reader["state_dict"].items():
            if key in sources:
                value.fill_(55)
            elif key.startswith(allowed):
                value.fill_(33)
        result = compose(writer, reader)
        for key in sources:
            torch.testing.assert_close(result["state_dict"][key], writer["state_dict"][key], atol=0, rtol=0)
        for key in result["state_dict"]:
            if key.startswith(allowed):
                torch.testing.assert_close(result["state_dict"][key], reader["state_dict"][key], atol=0, rtol=0)
            else:
                torch.testing.assert_close(result["state_dict"][key], writer["state_dict"][key], atol=0, rtol=0)
        self.assertEqual(float(result["state_dict"]["read.2.bias"][0]), 33.0)
        self.assertTrue(all(float(result["state_dict"]["memory_attention_logits"][i]) == 33 for i in range(4)))
        for invalid in ("plan_tags.", "plan_boundary.", "plan_boundary_raw.", "query_need.", "pool.",
                        "identity_scale", "identity_bias", "identity_mix", "read"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "selection"):
                compose(writer, dict(reader, reader_heads=(invalid,)))
        with self.assertRaisesRegex(ValueError, "selection"):
            compose(writer, dict(reader, reader_heads=("read.", "read.")))
        with self.assertRaisesRegex(ValueError, "backbone"):
            compose(writer, dict(reader, backbone_sha256="11" * 32))


class QueryBalancedLossContract(unittest.TestCase):
    def test_equivalent_support_and_negative_copies_keep_each_query_weight(self):
        small = torch.tensor([[-2., .4], [2., -.4]], dtype=torch.float64, requires_grad=True)
        small_gold = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)
        # One query has seven equivalent supports; the other has one. Both
        # questions retain their original positive/negative difficulty.
        large = torch.tensor([[-2.] * 7 + [.4], [2.] + [-.4] * 7],
                             dtype=torch.float64, requires_grad=True)
        large_gold = torch.tensor([[1.] * 7 + [0.], [1.] + [0.] * 7], dtype=torch.float64)
        before, after = balanced(small, small_gold), balanced(large, large_gold)
        expected = (F.softplus(torch.tensor(2., dtype=torch.float64))
                    + F.softplus(torch.tensor(-2., dtype=torch.float64))) / 2
        expected += 2.5 * (F.softplus(torch.tensor(.4, dtype=torch.float64))
                           + F.softplus(torch.tensor(-.4, dtype=torch.float64))) / 2
        torch.testing.assert_close(before, expected)
        torch.testing.assert_close(after, expected)
        before.backward()
        after.backward()
        for row in range(2):
            for positive in (True, False):
                old = small.grad[row][small_gold[row].bool() == positive].sum()
                new = large.grad[row][large_gold[row].bool() == positive].sum()
                torch.testing.assert_close(old, new)

    def test_rare_single_support_has_equal_total_positive_gradient(self):
        logits = torch.zeros(2, 8, dtype=torch.float64, requires_grad=True)
        gold = torch.tensor([[1.] * 7 + [0.], [1.] + [0.] * 7], dtype=torch.float64)
        balanced(logits, gold).backward()
        positive = (logits.grad * gold).sum(-1)
        negative = (logits.grad * (1 - gold)).sum(-1)
        torch.testing.assert_close(positive, torch.tensor([-.25, -.25], dtype=torch.float64))
        torch.testing.assert_close(negative, torch.tensor([.625, .625], dtype=torch.float64))

    def test_query_order_preserves_loss_and_reorders_gradients(self):
        original = torch.tensor([[-1., .4, -.7], [.8, -2., -.6], [.3, .6, .1]],
                                dtype=torch.float64, requires_grad=True)
        gold = torch.tensor([[1., 1., 0.], [0., 1., 0.], [0., 0., 0.]], dtype=torch.float64)
        order = torch.tensor([2, 0, 1])
        reordered = original.detach()[order].clone().requires_grad_()
        first, second = balanced(original, gold), balanced(reordered, gold[order])
        torch.testing.assert_close(first, second)
        first.backward()
        second.backward()
        torch.testing.assert_close(original.grad[order], reordered.grad)

    def test_queries_without_a_class_do_not_dilute_that_class_gradient(self):
        original = torch.tensor([[-.3, .8]], dtype=torch.float64, requires_grad=True)
        gold = torch.tensor([[1., 0.]], dtype=torch.float64)
        balanced(original, gold).backward()
        for extra_gold, tested_column in ((0., 0), (1., 1)):
            with self.subTest(extra_gold=extra_gold):
                combined = torch.cat((original.detach(), torch.full((9, 2), .2, dtype=torch.float64))).requires_grad_()
                labels = torch.cat((gold, torch.full((9, 2), extra_gold, dtype=torch.float64)))
                balanced(combined, labels).backward()
                torch.testing.assert_close(combined.grad[0, tested_column], original.grad[0, tested_column])

    def test_null_positive_and_empty_batches_have_finite_gradients(self):
        for shape in ((3, 4), (0, 4), (3, 0), (0, 0)):
            for positive in (False, True):
                with self.subTest(shape=shape, positive=positive):
                    logits = torch.zeros(shape, requires_grad=True)
                    loss = balanced(logits, torch.full(shape, float(positive)))
                    self.assertTrue(torch.isfinite(loss))
                    if not logits.numel():
                        self.assertEqual(float(loss.detach()), 0.)
                    loss.backward()
                    self.assertIsNotNone(logits.grad)
                    self.assertTrue(torch.isfinite(logits.grad).all())

    @staticmethod
    def auxiliary(logits, positive, negative):
        model = SimpleNamespace(semantic_bank_logits=lambda *args, **kwargs: logits)
        with patch("train_neural_dialogue_reader.semantic_relation_masks", return_value=(positive, negative)):
            return semantic_relation_auxiliary(model, [{}] * len(logits), [{}] * logits.shape[1], None, None, None, memory_context_latents=None)

    def test_semantic_absolute_and_worst_support_losses_have_query_equal_gradients(self):
        logits = torch.tensor([[-.4, -.4, .3, 12.], [-.4, .3, .3, .3]],
                              dtype=torch.float64, requires_grad=True)
        positive = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        negative = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
        loss = self.auxiliary(logits, positive, negative)
        expected = .25 * (F.softplus(torch.tensor(.4, dtype=torch.float64))
                          + F.softplus(torch.tensor(.3, dtype=torch.float64))
                          + F.softplus(torch.tensor(1.7, dtype=torch.float64)))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for mask in (positive, negative):
            gradient = logits.grad.masked_fill(~mask, 0).sum(-1)
            torch.testing.assert_close(gradient[0], gradient[1])
        self.assertEqual(float(logits.grad[0, 3]), 0.)

    def test_semantic_copies_and_query_permutations_preserve_weight(self):
        original = torch.tensor([[-.4, -.4, .3, 12.], [-.4, .3, .3, .3]], dtype=torch.float64)
        positive = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        negative = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
        baseline = original.clone().requires_grad_()
        expected = self.auxiliary(baseline, positive, negative)
        expected.backward()
        for copy_column in (1, 2):
            copied = torch.cat((original, original[:, copy_column:copy_column+1]), -1).requires_grad_()
            pos = torch.cat((positive, positive[:, copy_column:copy_column+1]), -1)
            neg = torch.cat((negative, negative[:, copy_column:copy_column+1]), -1)
            loss = self.auxiliary(copied, pos, neg)
            torch.testing.assert_close(loss, expected)
            loss.backward()
            for old_mask, new_mask in ((positive, pos), (negative, neg)):
                torch.testing.assert_close(baseline.grad.masked_fill(~old_mask, 0).sum(-1),
                                           copied.grad.masked_fill(~new_mask, 0).sum(-1))
        shuffled = original.flip(0).clone().requires_grad_()
        loss = self.auxiliary(shuffled, positive.flip(0), negative.flip(0))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(shuffled.grad, baseline.grad.flip(0))

    def test_semantic_null_positive_and_empty_batches_have_finite_gradients(self):
        for shape in ((2, 4), (0, 4), (2, 0), (0, 0)):
            for supports in (False, True):
                with self.subTest(shape=shape, supports=supports):
                    logits = torch.zeros(shape, requires_grad=True)
                    positive = torch.full(shape, supports, dtype=torch.bool)
                    loss = self.auxiliary(logits, positive, ~positive)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertIsNotNone(logits.grad)
                    self.assertTrue(torch.isfinite(logits.grad).all())

    def test_scope_guard_keeps_baseline_and_rejects_each_regression(self):
        initial = dict(positive_support_set_exact=.75, fp=4, null_false_activation=.2)
        self.assertTrue(scope_refinement_safe(dict(initial), initial))
        self.assertTrue(scope_refinement_safe(dict(initial, positive_support_set_exact=.8, fp=3,
                                                   null_false_activation=.1), initial))
        for changes in (dict(positive_support_set_exact=.7), dict(fp=5), dict(null_false_activation=.3),
                        dict(positive_support_set_exact=None), dict(null_false_activation=None)):
            self.assertFalse(scope_refinement_safe(dict(initial, **changes), initial))
        no_null = dict(initial, null_false_activation=None)
        self.assertTrue(scope_refinement_safe(dict(no_null), no_null))
        self.assertFalse(scope_refinement_safe(dict(no_null, null_false_activation=0.), no_null))


class WriterDevelopmentGuardContract(unittest.TestCase):
    @staticmethod
    def baselines():
        fixed = dict(rows=180, committed_fact_f1=5 / 6, plan_exact=174 / 180,
                     source_exact=174 / 180, action_exact=174 / 180,
                     reference_exact=1., whole_reference_exact=1., ambiguous_unsafe_writes=0)
        boundary = dict(fixed, rows=72, committed_fact_f1=16 / 17,
                        plan_exact=66 / 72, source_exact=66 / 72, action_exact=66 / 72,
                        reference_exact=66 / 72)
        narrative = dict(fixed, rows=12, committed_fact_f1=12 / 17,
                         plan_exact=7 / 12, source_exact=7 / 12, action_exact=7 / 12,
                         reference_exact=.75)
        return fixed, boundary, narrative

    def test_equal_fact_f1_cannot_promote_fewer_complete_plans_in_any_group(self):
        baselines = self.baselines()
        for group in range(3):
            for key in ("plan_exact", "source_exact", "action_exact"):
                with self.subTest(group=group, metric=key):
                    candidate = [dict(metrics) for metrics in baselines]
                    candidate[1]["committed_fact_f1"] = 96 / 101
                    candidate[group][key] -= 1 / candidate[group]["rows"]
                    self.assertFalse(writer_development_safe(
                        candidate[0], baselines[0], candidate[1], baselines[1],
                        candidate[2], baselines[2]))
        # The observed candidate had the same narrative fact F1 but changed
        # complete source/action plans from seven to six of twelve turns.
        fixed, boundary, narrative = baselines
        improved_boundary = dict(boundary, committed_fact_f1=96 / 101,
                                 plan_exact=67 / 72, source_exact=67 / 72, action_exact=67 / 72)
        regressed_narrative = dict(narrative, plan_exact=.5, source_exact=.5, action_exact=.5)
        self.assertFalse(writer_development_safe(fixed, fixed, improved_boundary, boundary,
                                                 regressed_narrative, narrative))
        self.assertFalse(writer_development_safe(dict(fixed, source_exact=.9), fixed))

    def test_reference_and_unsafe_write_regressions_are_rejected_for_each_group(self):
        baselines = self.baselines()
        for group in range(3):
            for key, change in (("reference_exact", -.01), ("whole_reference_exact", -.01),
                                ("ambiguous_unsafe_writes", 1)):
                with self.subTest(group=group, metric=key):
                    candidate = [dict(metrics) for metrics in baselines]
                    candidate[group]["committed_fact_f1"] = 1.
                    candidate[group][key] += change
                    self.assertFalse(writer_development_safe(
                        candidate[0], baselines[0], candidate[1], baselines[1],
                        candidate[2], baselines[2]))

    def test_valid_boundary_improvement_and_equal_baseline_remain_eligible(self):
        fixed, boundary, narrative = self.baselines()
        improved_boundary = dict(boundary, committed_fact_f1=96 / 101,
                                 plan_exact=67 / 72, source_exact=67 / 72,
                                 action_exact=67 / 72, reference_exact=67 / 72)
        self.assertTrue(writer_development_safe(fixed, fixed, improved_boundary, boundary,
                                                narrative, narrative))
        self.assertTrue(writer_development_safe(fixed, fixed, boundary, boundary, narrative, narrative))
        self.assertTrue(writer_development_safe(fixed, fixed, improved_boundary, boundary))
        self.assertTrue(writer_development_safe(fixed, fixed))

    def test_every_attached_group_requires_its_initial_baseline(self):
        fixed, boundary, narrative = self.baselines()
        with self.assertRaisesRegex(ValueError, "initial metrics"):
            writer_development_safe(fixed, fixed, boundary)
        with self.assertRaisesRegex(ValueError, "initial metrics"):
            writer_development_safe(fixed, fixed, narrative=narrative)


class ReaderTerminalCheckpointLifecycle(unittest.TestCase):
    def test_diagnostic_resume_keeps_terminal_weights_separate_from_eligible_selected_baseline(self):
        import train_neural_dialogue_reader as reader
        from test_neural_temporal_curriculum import small_fit

        fit = small_fit()
        for row in fit["queries"]:
            row.setdefault("observation", dict(body=row["text"], context="Original fit context",
                                               source_id=row["observation_id"]))
            row.update(binding_predicted=True, unresolved=False)

        class TinyCache:
            def __init__(self, owner, *args, **kwargs):
                self.model = owner
                self.freeze_memory_encoder = True
                self.feature_dropout = 0.
            def for_model(self, owner):
                return TinyCache(owner)

        metrics = dict(activation_f1=.8, support_set_exact=.75, positive_queries=4,
                       positive_exact=3, positive_support_set_exact=.75, null_false_activation=0.,
                       tp=3, fp=1, fn=1, bank_size=36, queries=4)
        identity = "a" * 64
        model = NeuralSlots()
        before = deepcopy(model.state_dict())
        stale = dict(initial_development_views={"stale": -1},
                     initial_evidence_development_views={"stale": -2},
                     initial_selection_score=(-3, -3, -3))
        def extracted(_args, groups):
            texts = {r[k] for rows in groups.values() for r in rows for k in ("body", "context")}
            return {text: dict(x=torch.zeros(1, INPUT), bounds=[(0, len(text))]) for text in texts}, {}
        def correction(student, *args, **kwargs):
            return (student.read[2].bias - (before["read.2.bias"] + .5)).square().sum()
        def replay(student, _parent, rows, *args, **kwargs):
            zero = student.read[2].bias.sum() * 0
            return zero, dict(supervised=zero.detach(), margin=zero.detach(), distillation=zero.detach(),
                              protected=zero.detach(), intent_supervision=zero.detach(), queries=len(rows))
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "build") as directory:
            directory = Path(directory)
            checkpoint = directory / "diagnostic.pt"
            torch.save(dict(format=FORMAT, backbone_sha256=identity, state_dict=before,
                            reader_step=19, diagnostic_only=True, eligible_for_selection=False,
                            **stale), checkpoint)
            output = directory / "run"
            args = SimpleNamespace(steps=1, eval_every=1, train_heads="matcher", resume=str(checkpoint),
                planner_checkpoint=str(checkpoint), output=str(output), kind_context_prefit=False,
                matcher_corrections=True, reader_functional_weight=1., reader_functional_margin=1.,
                reader_anchor_weight=0., reader_anchor_margin=1., freeze_memory_encoder=True,
                feature_dropout=0., semantic_topic_weight=0., curriculum=None, fit_paraphrases=False,
                replay_episodes=False, relation_paraphrases=False, episode_relations=False,
                relation_bindings=False, gguf="mock-backbone", probe="mock-probe", features="mock-features",
                extra_features=[], evidence_cues=False, expansion=False, identity_diversity=False,
                qualification_curriculum=None, device="cpu", learning_rate=.001,
                attention_learning_rate=.01, activation_positive_weight=1.)
            with patch("prepare_neural_colloquial.reader_data", side_effect=lambda *_: deepcopy(fit)), \
                 patch.object(reader, "sha", return_value=identity), \
                 patch.object(reader, "load_features", return_value=dict(identity=dict(backbone=identity, probe=identity), features={})), \
                 patch.object(reader, "extract", side_effect=extracted), \
                 patch.object(reader, "bind_curriculum_queries", side_effect=lambda group, *_: (deepcopy(group), [])), \
                 patch.object(reader, "DynamicFeatures", TinyCache), \
                 patch.object(reader, "evaluate", side_effect=lambda *_: deepcopy(metrics)), \
                 patch.object(reader, "scope_evaluate", side_effect=lambda *_: deepcopy(metrics)), \
                 patch.object(reader, "kind_bank_diagnostics", return_value={}), \
                 patch.object(reader, "attention_bank_diagnostics", return_value={}), \
                 patch.object(reader, "matcher_correction_training_loss", side_effect=correction), \
                 patch.object(reader, "functional_replay_loss", side_effect=replay), \
                 patch.object(reader, "export", return_value="test-artifact-identity"), \
                 patch("builtins.print"):
                reader.train(args)
            selected = torch.load(output / "selected.pt", weights_only=False, map_location="cpu")
            terminal = torch.load(output / "last-step.pt", weights_only=False, map_location="cpu")
            report = json.loads((output / "report.json").read_text())
            self.assertEqual((selected["reader_step"], terminal["reader_step"], report["selected_step"]), (0, 1, 0))
            self.assertEqual((selected["diagnostic_only"], selected["eligible_for_selection"]), (False, True))
            self.assertEqual((terminal["diagnostic_only"], terminal["eligible_for_selection"]), (True, False))
            self.assertFalse(torch.equal(terminal["state_dict"]["read.2.bias"], before["read.2.bias"]))
            for name, value in before.items():
                torch.testing.assert_close(selected["state_dict"][name], value, atol=0, rtol=0)
                if name != "read.2.bias":
                    torch.testing.assert_close(terminal["state_dict"][name], value, atol=0, rtol=0)
            for key in stale:
                self.assertEqual(selected[key], terminal[key])
                self.assertNotEqual(selected[key], stale[key])
                self.assertEqual(report[key], json.loads(json.dumps(selected[key])))
            self.assertEqual(selected["initial_selection_score"], (.75, .75, .8))
            self.assertTrue(report["reader_functional_teacher_all_parameters_unchanged"])
            self.assertEqual(report["training_steps_executed"], 1)


if __name__ == "__main__":
    unittest.main()
