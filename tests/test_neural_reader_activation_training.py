"""Tiny operator regressions for learned relation/time activation, not qualification."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, PLAN_KEYS, QUERY_KEYS, SEMANTIC, WIDTH, NeuralSlots
from neural_feedback_evaluation import component_nonregression_safe, feedback_selection_score
import train_neural_dialogue_reader as reader


EXPECTED_TRAINABLE = {
    "query_encode.weight", "query_encode.bias", "query_pool.weight",
    "read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight",
    "read.0.weight", "read.0.bias", "read.2.weight", "read.2.bias",
    "read_update.weight", "read_update.bias", "memory_attention_logits",
    "intent.weight", "intent.bias", "version.0.weight", "version.0.bias",
    "version.2.weight", "version.2.bias",
}


class ReaderActivationTrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, mode="activation", device="cpu"):
        torch.manual_seed(73127)
        model = NeuralSlots().eval()
        facts = [
            dict(text="Ada. I keep chisels in a chest.", entity="Ada", reference="Ada",
                 prop="storage", kind=0, status=1),
            dict(text="Ada. I keep chisels in a cabinet.", entity="Ada", reference="Ada",
                 prop="storage", kind=0, status=0),
            dict(text="Ada. I intend to keep chisels in a crate.", entity="Ada", reference="Ada",
                 prop="storage", kind=2, status=3),
        ]
        original = "List Ada's current and former chisel locations."
        rows = [
            dict(text="List Ada's past, current and planned chisel locations.", reference="Ada",
                 entities=["Ada"], prop="storage", kind=0, allowed_kinds=[0, 1, 2],
                 source_kinds=[0, 2], support_texts=[f["text"] for f in facts]),
            dict(text=original + "\nActivated evidence:\n" + facts[-1]["text"] + "\n",
                 initial_text=original, initial_reference="Ada", augmented=True,
                 entities=["Ada"], prop="storage", kind=0, allowed_kinds=[0, 1],
                 source_kinds=[0], support_texts=[f["text"] for f in facts[:2]]),
        ]
        texts = sorted({r["text"] for r in facts + rows} | {"Ada", original})
        cache = {}
        for index, text in enumerate(texts):
            length = 3 + index % 3
            cache[text] = dict(
                x=torch.randn(length, INPUT),
                bounds=[(i * len(text) // length, (i + 1) * len(text) // length)
                        for i in range(length)],
            )
        # The source/identity/semantic channels are deliberately sufficient;
        # only the requested planned-state probability blocks the gold slot.
        with torch.no_grad():
            model.intent.weight.zero_()
            model.intent.bias.zero_()
            model.intent.bias[:2].fill_(5.)
            model.intent.bias[2] = math.log(.15 / .85)
            model.identity_match[2].weight.zero_()
            model.identity_match[2].bias.fill_(20.)
            model.version[2].weight.fill_(.01)
            model.version[2].bias.fill_(11.)
            model.read[2].weight.mul_(.1)
            model.read[2].bias.fill_(8.)
        reader.configure_reader_heads(model, mode, freeze_memory_encoder=True)
        cached = reader.DynamicFeatures(model, {"fit": dict(facts=facts, queries=rows)},
                                        cache, device, freeze_memory_encoder=True)
        model.to(device)
        return model, facts, rows, cached

    def metrics(self, model, facts, rows, cached):
        return reader.evaluate(model, dict(facts=facts, queries=rows), cached)

    def scope_scores(self, model, facts, rows, cached):
        bank, ids, mask, metadata = reader.make_bank(facts, cached)
        query, identity = reader.query(model, rows, cached, scope=True)
        return model.score_bank(query, identity, bank, ids, metadata, mask, scope=True, memory_context_latents=(query).new_zeros((len(mask), 16)))

    def test_semantic10_cannot_correct_a_fixed_planned_state_ceiling(self):
        model, facts, rows, cached = self.fixture("semantic")
        rows = rows[:1]
        initial = self.metrics(model, facts, rows, cached)
        self.assertEqual(initial["fixed_input_gate_blocked_positives"], 1)
        self.assertEqual(initial["fixed_input_gate_reachable_positive_queries"], 0)
        before = reader.functional_reader_trace(model, rows, facts, cached)["temporal_gate"].detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
        for _ in range(3):
            optimizer.zero_grad()
            reader.semantic_training_loss(model, rows, facts, cached, torch.nn.Linear(WIDTH, 1),
                                          ["storage"], topic_weight=0.).backward()
            self.assertIsNone(model.intent.weight.grad)
            self.assertIsNone(model.intent.bias.grad)
            self.assertTrue(all(p.grad is None for p in model.version.parameters()))
            optimizer.step()
        after = reader.functional_reader_trace(model, rows, facts, cached)["temporal_gate"].detach()
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        terminal = self.metrics(model, facts, rows, cached)
        self.assertEqual(terminal["fixed_input_gate_blocked_positives"], 1)
        self.assertEqual(terminal["positive_exact"], 0)

    def test_activation16_connects_time_gradients_and_unlocks_the_same_positive(self):
        model, facts, rows, cached = self.fixture()
        rows = rows[:1]
        initial = self.metrics(model, facts, rows, cached)
        self.assertEqual(initial["fixed_input_gate_blocked_positives"], 1)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.)
        # Fixed-length operator control; diagnostics never select parameters.
        for _ in range(8):
            optimizer.zero_grad()
            loss = reader.activation_training_loss(model, rows, facts, cached)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith(("intent.", "version.")):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            optimizer.step()
        terminal = self.metrics(model, facts, rows, cached)
        self.assertEqual(terminal["fixed_input_gate_blocked_positives"], 0)
        self.assertEqual(terminal["fixed_input_gate_reachable_positive_queries"], 1)
        self.assertEqual(terminal["positive_exact"], 1)
        trace = reader.functional_reader_trace(model, rows, facts, cached)
        self.assertGreater(float(trace["temporal_gate"][0, 2].detach()), .5)
        self.assertTrue((trace["activation"] > 0).all())

    def test_only19_parameters_can_change_and_source_scope_and72_others_stay_exact(self):
        model, facts, rows, cached = self.fixture()
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, EXPECTED_TRAINABLE)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        self.assertEqual(len(frozen), 72)
        self.assertTrue(set(PLAN_KEYS + QUERY_KEYS) <= frozen.keys())
        with torch.no_grad():
            scope_before = self.scope_scores(model, facts, rows, cached).clone()
        optimizer = reader.reader_optimizer(model, torch.nn.Linear(WIDTH, 1), "activation", .001, .01)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]},
                         {id(p) for p in model.parameters() if p.requires_grad})
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        loss = reader.activation_training_loss(model, rows, facts, cached)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
        self.assertTrue(changed and changed <= EXPECTED_TRAINABLE)
        self.assertTrue({"intent.weight", "intent.bias", "version.2.bias"} <= changed)
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(self.scope_scores(model, facts, rows, cached), scope_before,
                                       atol=0, rtol=0)

    def test_full_teacher_protects_old_correct_sets_when_time_parameters_change(self):
        model, facts, rows, cached = self.fixture()
        old = rows[1:]
        self.assertEqual(self.metrics(model, facts, old, cached)["positive_exact"], 1)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        parent_before = {name: p.detach().clone() for name, p in parent.named_parameters()}
        initial_loss, initial_stats = reader.functional_replay_loss(model, parent, old, facts, cached, parent_cached)
        self.assertEqual(int(initial_stats["protected"]), 1)
        self.assertEqual(float(initial_stats["margin"]), 0.)
        self.assertEqual(float(initial_stats["distillation"]), 0.)
        del initial_loss
        with torch.no_grad():
            model.intent.bias[2].add_(4.)
        loss, stats = reader.functional_replay_loss(model, parent, old, facts, cached, parent_cached)
        self.assertGreater(float(stats["margin"]), 0.)
        self.assertGreater(float(stats["distillation"]), 0.)
        loss.backward()
        self.assertGreater(float(model.intent.bias.grad[2]), 0.)
        for name, parameter in parent.named_parameters():
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, parent_before[name], atol=0, rtol=0)

    def test_main_activation_and_multi_state_intent_targets_have_live_time_inputs(self):
        model, facts, rows, cached = self.fixture()
        with patch.object(F, "binary_cross_entropy_with_logits", wraps=F.binary_cross_entropy_with_logits) as binary, \
             patch.object(model, "score_bank", wraps=model.score_bank) as scoring, \
             patch.object(F, "cross_entropy", side_effect=AssertionError("frozen source-kind head was trained")):
            loss = reader.activation_training_loss(model, rows, facts, cached)
        intent_calls = [call for call in binary.call_args_list if "reduction" not in call.kwargs]
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(intent_calls[0].args[1].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertTrue(intent_calls[0].args[0].requires_grad)
        self.assertTrue(scoring.call_args.args[0][:, SEMANTIC:].requires_grad)
        for index in (1, 3, 4):
            self.assertFalse(scoring.call_args.args[index].requires_grad)
        self.assertTrue(torch.isfinite(loss))

    def test_old_replay_intent_supervision_is_explicit_live_and_never_changes_teacher_eligibility(self):
        model, facts, rows, cached = self.fixture()
        rows = rows[:1]
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        default_loss, default_stats = reader.functional_replay_loss(
            model, parent, rows, facts, cached, parent_cached)
        self.assertEqual(float(default_stats["intent_supervision"]), 0.)
        default_gradient = torch.autograd.grad(default_loss, model.intent.bias)[0]
        with patch.object(F, "binary_cross_entropy_with_logits", wraps=F.binary_cross_entropy_with_logits) as binary:
            weighted_loss, weighted_stats = reader.functional_replay_loss(
                model, parent, rows, facts, cached, parent_cached, intent_weight=.25)
        intent_calls = [call for call in binary.call_args_list if "reduction" not in call.kwargs]
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(intent_calls[0].args[1].tolist(), [[1, 1, 1]])
        self.assertTrue(intent_calls[0].args[0].requires_grad)
        self.assertGreater(float(weighted_stats["intent_supervision"]), 0.)
        self.assertEqual(int(default_stats["protected"]), 0)
        self.assertEqual(int(weighted_stats["protected"]), 0)
        torch.testing.assert_close(weighted_loss.detach() - default_loss.detach(),
                                   weighted_stats["intent_supervision"], atol=1e-6, rtol=1e-5)
        weighted_gradient = torch.autograd.grad(weighted_loss, model.intent.bias)[0]
        expected = torch.zeros_like(model.intent.bias)
        expected[:3] = .25 * (model.intent.bias[:3].detach().sigmoid() - 1) / 3
        torch.testing.assert_close(weighted_gradient - default_gradient, expected, atol=1e-6, rtol=1e-5)
        self.assertTrue(all(parameter.grad is None for parameter in parent.parameters()))
        for weight in (-1., float("nan"), float("inf")):
            with self.subTest(weight=weight), \
                 patch.object(reader, "functional_reader_trace", side_effect=AssertionError("invalid weight entered forward")):
                with self.assertRaisesRegex(ValueError, "intent weight"):
                    reader.functional_replay_loss(model, parent, rows, facts, cached,
                                                  parent_cached, intent_weight=weight)

    def test_evaluation_rows_and_empty_groups_fail_before_forward(self):
        model, facts, rows, cached = self.fixture()
        for changes in (dict(evaluation_only=True), dict(split="development"),
                        dict(evidence_challenge_view=4)):
            for bad_rows, bad_facts in (([dict(rows[0], **changes)], facts),
                                        (rows, [dict(facts[0], **changes)])):
                with self.subTest(changes=changes), \
                     patch.object(reader, "make_bank", side_effect=AssertionError("evaluation entered forward")):
                    with self.assertRaises(ValueError):
                        reader.activation_training_loss(model, bad_rows, bad_facts, cached)
        for bad_rows, bad_facts in (([], facts), (rows, [])):
            with self.assertRaisesRegex(ValueError, "fit facts and queries"):
                reader.activation_training_loss(model, bad_rows, bad_facts, cached)

    def settings(self, **changes):
        values = dict(steps=1, eval_every=40, train_heads="activation", semantic_topic_weight=0.,
                      resume="parent.pt", output="never-created", reader_anchor_weight=0.,
                      reader_anchor_margin=1., reader_functional_weight=1., reader_functional_margin=1.,
                      freeze_memory_encoder=True, feature_dropout=0., relation_paraphrases=False,
                      episode_relations=False, relation_bindings=False)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_required_parent_retention_and_frozen_inputs_are_checked_before_output_creation(self):
        for changes in (dict(resume=None), dict(reader_functional_weight=0.),
                        dict(freeze_memory_encoder=False), dict(feature_dropout=.1),
                        dict(semantic_topic_weight=.05), dict(reader_anchor_weight=1.)):
            with self.subTest(changes=changes), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("output was created")):
                with self.assertRaises(ValueError):
                    reader.train(self.settings(**changes))
        with patch.object(Path, "mkdir", side_effect=RuntimeError("valid output stage")):
            with self.assertRaisesRegex(RuntimeError, "valid output stage"):
                reader.train(self.settings())

    def test_fixed_gate_diagnostics_never_change_selection_or_nonregression_guards(self):
        metrics = dict(activation_f1=.8, positive_queries=3, positive_support_set_exact=2/3,
                       fp=2, null_false_activation=0., fixed_input_gate_blocked_positives=1,
                       fixed_input_gate_reachable_positive_queries=2)
        changed = dict(metrics, fixed_input_gate_blocked_positives=1000,
                       fixed_input_gate_reachable_positive_queries=0)
        self.assertEqual(feedback_selection_score(dict(view=metrics), dict(view=metrics)),
                         feedback_selection_score(dict(view=changed), dict(view=changed)))
        self.assertTrue(component_nonregression_safe(dict(view=changed), dict(view=metrics)))
        self.assertTrue(reader.scope_refinement_safe(changed, metrics))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_joint_activation_gradients_and_gate_unlock(self):
        model, facts, rows, cached = self.fixture(device="mps")
        old = rows[1:]
        rows = rows[:1]
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        self.assertIs(parent_cached.inputs, cached.inputs)
        old_loss, old_stats = reader.functional_replay_loss(
            model, parent, old, facts, cached, parent_cached, intent_weight=.25)
        self.assertEqual(int(old_stats["protected"]), 1)
        self.assertEqual(float(old_stats["margin"]), 0.)
        self.assertEqual(float(old_stats["distillation"]), 0.)
        self.assertGreater(float(old_stats["intent_supervision"]), 0.)
        self.assertTrue(torch.isfinite(old_loss))
        old_loss.backward()
        self.assertTrue(torch.isfinite(model.intent.bias.grad).all())
        model.zero_grad(set_to_none=True)
        before = self.metrics(model, facts, rows, cached)
        self.assertEqual(before["fixed_input_gate_blocked_positives"], 1)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.)
        for _ in range(8):
            optimizer.zero_grad()
            loss = reader.activation_training_loss(model, rows, facts, cached)
            self.assertEqual(loss.device.type, "mps")
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if name.startswith(("intent.", "version.")):
                        self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        after = self.metrics(model, facts, rows, cached)
        self.assertEqual(after["fixed_input_gate_blocked_positives"], 0)
        self.assertEqual(after["positive_exact"], 1)
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in parent.parameters()))
        self.assertEqual(self.metrics(parent, facts, rows, parent_cached)["fixed_input_gate_blocked_positives"], 1)


if __name__ == "__main__":
    unittest.main()
