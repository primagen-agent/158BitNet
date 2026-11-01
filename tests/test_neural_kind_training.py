"""Capped source-kind correction through the complete temporal training path."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_kind_correction import snapshot_kind_teacher, kind_correction_loss
from neural_slot_model import PLAN_KEYS, QUERY_KEYS, WIDTH
import train_neural_dialogue_reader as reader
import test_neural_temporal_retention as temporal_training


class NeuralKindTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device="cpu"):
        model, facts, rows, kinds, cached = temporal_training.TemporalRetentionContract().fixture(device)
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([3., 0., -2.], device=device))
            model.kind_context.weight.normal_(0, .05)
            model.kind_context.bias.copy_(torch.tensor([.1, -.1, 0.], device=device))
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        return model, parent, facts, rows, kinds, cached, parent_cached

    @staticmethod
    def means(cached, facts):
        encoded, mask, _ = cached.encode([fact["text"] for fact in facts])
        return (encoded * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]

    def expected_loss(self, model, parent, facts, rows, kinds, cached, parent_cached):
        with torch.no_grad():
            old = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            labels = reader.source_kind_targets(kinds, cached.device)
            teacher = snapshot_kind_teacher(parent_cached.kind_logits([fact["text"] for fact in kinds], self.means(parent_cached, kinds)), labels)
        live = reader.functional_reader_trace(model, rows, facts, cached)
        gold = torch.tensor([[reader.supported(row, fact) for fact in facts] for row in rows],
                            dtype=live["activation"].dtype, device=cached.device)
        activation, _ = reader.temporal_activation_correction(
            live["activation"], gold, rows, facts, old["semantic"], old["identity"])
        categorical, stats = kind_correction_loss(cached.kind_logits([fact["text"] for fact in kinds], self.means(cached, kinds)), teacher)
        intent = .25 * F.binary_cross_entropy_with_logits(
            live["intent"][:, :3], reader.intent_targets(rows, cached.device))
        return activation + categorical + intent, stats, teacher

    def test_focused_loss_matches_complete_class_capped_objective_without_kind_ce(self):
        model, parent, facts, rows, kinds, cached, parent_cached = self.fixture()
        expected, stats, teacher = self.expected_loss(
            model, parent, facts, rows, kinds, cached, parent_cached)
        self.assertEqual(teacher.labels.tolist(), [0] * 12 + [1] * 12 + [2] * 12)
        self.assertEqual(int(stats["parent_correct_samples"]), 12)
        self.assertEqual(int(stats["parent_wrong_samples"]), 24)
        self.assertEqual(float(stats["correct_preservation"]), 0.)
        self.assertGreater(float(stats["wrong_correction"]), 0.)
        with patch.object(F, "cross_entropy", side_effect=AssertionError("focused kind CE used")), \
             patch.object(model, "score_bank", wraps=model.score_bank) as scoring:
            actual = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                                  parent=parent, parent_cached=parent_cached)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(scoring.call_args.args[2].shape[0], len(facts))
        self.assertEqual(len(kinds), 36)

    def test_parent_kind_teacher_uses_own_frozen_encoder_after_student_input_drift(self):
        model, parent, facts, rows, kinds, cached, parent_cached = self.fixture()
        parent_before = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
        # A diagnostic input drift makes sharing student learned encodings
        # observable even though normal Time10 optimization freezes this encoder.
        with torch.no_grad():
            model.encode.bias.add_(.75)
            model.kind.bias.copy_(torch.tensor([-2., 0., 3.]))
            expected_means = self.means(parent_cached, kinds)
            student_means = self.means(cached, kinds)
        self.assertGreater(float((expected_means - student_means).abs().max()), .05)
        parent_inputs = []
        hook = parent.kind.register_forward_pre_hook(
            lambda _module, args: parent_inputs.append(args[0].detach().clone()))
        try:
            loss = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                                parent=parent, parent_cached=parent_cached)
        finally:
            hook.remove()
        classification_inputs = [value for value in parent_inputs if value.shape[0] == 36]
        self.assertEqual(len(classification_inputs), 1)
        torch.testing.assert_close(classification_inputs[0], expected_means, atol=0, rtol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.kind.weight.grad).all())
        self.assertGreater(float(model.kind.weight.grad.abs().sum()), 0.)
        for name, parameter in parent.named_parameters():
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, parent_before[name], atol=0, rtol=0)

    def test_nonretention_temporal_path_keeps_original_source_kind_ce(self):
        model, _, facts, rows, kinds, cached, _ = self.fixture()
        with patch.object(F, "cross_entropy", wraps=F.cross_entropy) as ce:
            loss = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(ce.call_count, 1)
        self.assertEqual(ce.call_args.args[1].tolist(), [0] * 12 + [1] * 12 + [2] * 12)
        self.assertEqual(ce.call_args.args[0].shape, (36, 3))

    def test_physical_history_preserves_original_current_label_and_zero_kind_deficit(self):
        model, parent, facts, rows, _, cached, parent_cached = self.fixture()
        kinds = [deepcopy(facts[0])]
        self.assertEqual(kinds[0]["status"], 1)
        self.assertEqual(kinds[0]["source_kind"], 0)
        expected, stats, teacher = self.expected_loss(
            model, parent, facts, rows, kinds, cached, parent_cached)
        self.assertEqual(teacher.labels.tolist(), [0])
        self.assertEqual(teacher.correct.tolist(), [True])
        self.assertEqual(teacher.required_gap.tolist(), [1.])
        self.assertEqual(float(stats["correct_preservation"]), 0.)
        self.assertEqual(float(stats["wrong_correction"]), 0.)
        categorical, _ = kind_correction_loss(cached.kind_logits([fact["text"] for fact in kinds], self.means(cached, kinds)), teacher)
        gradients = torch.autograd.grad(categorical, (model.kind.weight, model.kind.bias))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
        with patch.object(F, "cross_entropy", side_effect=AssertionError("physical history CE used")):
            actual = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                                  parent=parent, parent_cached=parent_cached)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def exercise_trainable_and_teacher_contract(self, device):
        model, parent, facts, rows, kinds, cached, parent_cached = self.fixture(device)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        teacher_before = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
        self.assertEqual(len(frozen), 81)
        self.assertTrue(set(PLAN_KEYS + QUERY_KEYS) <= frozen.keys())
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(len(trainable), 10)
        self.assertTrue(all(name.startswith(("kind.", "kind_context.", "intent.", "version.")) for name in trainable))

        def scope_scores():
            bank, ids, mask, metadata = reader.make_bank(facts, cached)
            query, identity = reader.query(model, rows, cached, scope=True)
            return model.score_bank(query, identity, bank, ids, metadata, mask, scope=True, memory_context_latents=(query).new_zeros((len(mask), 16)))

        with torch.no_grad():
            before_scope = scope_scores().clone()
        main = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                            parent=parent, parent_cached=parent_cached)
        replay, stats = reader.functional_replay_loss(model, parent, rows, facts, cached,
                                                     parent_cached, intent_weight=.25,
                                                     temporal_only=True)
        self.assertEqual(float(stats["margin"]), 0.)
        self.assertEqual(float(stats["distillation"]), 0.)
        self.assertTrue(torch.isfinite(main + replay))
        (main + replay).backward()
        optimizer = reader.reader_optimizer(model, torch.nn.Linear(WIDTH, 3), "temporal", .001, .01)
        for name, parameter in model.named_parameters():
            if name in trainable:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        for name, parameter in parent.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, teacher_before[name], atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(scope_scores(), before_scope, atol=0, rtol=0)

    def test_capped_main_and_full_reader_replay_change_only_time10(self):
        self.exercise_trainable_and_teacher_contract("cpu")

    def test_evaluation_kind_corpus_and_shared_parent_context_fail_closed(self):
        model, parent, facts, rows, kinds, cached, parent_cached = self.fixture()
        for fields in (dict(evaluation_only=True), dict(split="development"),
                       dict(evidence_challenge_view="cue_0")):
            changed = deepcopy(kinds)
            changed[0].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                reader.temporal_training_loss(model, rows, facts, cached, kind_facts=changed,
                                              parent=parent, parent_cached=parent_cached)
        with self.assertRaises(ValueError):
            reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                          parent=parent, parent_cached=cached)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_capped_kind_and_frozen_complete_reader(self):
        self.exercise_trainable_and_teacher_contract("mps")


if __name__ == "__main__":
    unittest.main()
