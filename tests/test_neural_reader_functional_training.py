"""Full-parent replay uses independent learned paths over shared raw C inputs."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, KEYS
import train_neural_dialogue_reader as reader
import test_neural_reader_temporal as temporal_tests


class ReaderFunctionalTrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device="cpu"):
        model, facts, originals, _ = temporal_tests.TemporalReaderContract().fixture()
        rows = deepcopy(originals[:2])
        rows[0].update(
            text=originals[0]["text"] + "\nActivated evidence:\n" + facts[-1]["text"] + "\n",
            initial_text=originals[0]["text"],
            initial_reference="Ada",
            augmented=True,
        )
        with torch.no_grad():
            model.identity_update.weight.normal_(0, .02)
            model.read_update.weight.normal_(0, .03)
        texts = sorted({row["text"] for row in facts + rows}
                       | {row.get("initial_text", row["text"]) for row in rows}
                       | {"Ada", "Ben"})
        cache = {}
        for index, text in enumerate(texts):
            length = 3 + index % 3
            cache[text] = dict(
                x=torch.randn(length, INPUT),
                bounds=[(i * len(text) // length, (i + 1) * len(text) // length)
                        for i in range(length)],
            )
        reader.configure_reader_heads(model, "semantic", freeze_memory_encoder=True)
        cached = reader.DynamicFeatures(model, {"fit": dict(facts=facts, queries=rows)},
                                        cache, device, freeze_memory_encoder=True)
        model.to(device)
        return model, facts, rows, cached

    def correct_parent_rows(self, model, facts, rows, cached):
        with torch.no_grad():
            trace = reader.functional_reader_trace(model, rows, facts, cached)
        rows = deepcopy(rows)
        for index, row in enumerate(rows):
            row["support_texts"] = [fact["text"] for column, fact in enumerate(facts)
                                    if trace["activation"][index, column] > 0]
        return rows

    def test_raw_features_share_storage_but_all_parent_parameters_are_independent(self):
        model, facts, rows, cached = self.fixture()
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        self.assertIsNot(parent, model)
        self.assertIsNot(parent_cached, cached)
        self.assertIs(parent_cached.model, parent)
        self.assertIs(cached.model, model)
        self.assertFalse(parent.training)
        self.assertEqual(set(parent.state_dict()), set(KEYS))
        self.assertEqual(len(list(parent.parameters())), 91)
        for name in ("inputs", "summaries", "word_inputs", "word_lex", "lengths",
                     "offsets", "word_lengths", "word_offsets"):
            self.assertIs(getattr(parent_cached, name), getattr(cached, name), name)
        for (name, student), (parent_name, teacher) in zip(
                model.named_parameters(), parent.named_parameters()):
            self.assertEqual(name, parent_name)
            self.assertNotEqual(student.data_ptr(), teacher.data_ptr(), name)
            self.assertFalse(teacher.requires_grad, name)
            torch.testing.assert_close(student, teacher, atol=0, rtol=0)
        with torch.no_grad():
            old_bank = parent_cached.encode([fact["text"] for fact in facts])[0].clone()
            old_identity = parent_cached.reference_ids(facts).clone()
            model.encode.bias.add_(.2)
            model.identity.weight[0].add_(.2)
            model.role[0].weight[0].add_(.1)
            live_bank = cached.encode([fact["text"] for fact in facts])[0]
            live_identity = cached.reference_ids(facts)
            torch.testing.assert_close(parent_cached.encode([fact["text"] for fact in facts])[0],
                                       old_bank, atol=0, rtol=0)
            torch.testing.assert_close(parent_cached.reference_ids(facts), old_identity,
                                       atol=0, rtol=0)
        self.assertFalse(torch.equal(live_bank, old_bank))
        self.assertFalse(torch.equal(live_identity, old_identity))

    def test_teacher_trace_is_invariant_after_joint_query_and_feedback_changes(self):
        model, facts, rows, cached = self.fixture()
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        with torch.no_grad():
            original = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            model.query_encode.bias.add_(.2)
            model.query_pool.weight.mul_(1.5)
            model.read_update.bias.add_(1.)
            model.memory_attention_logits.add_(.3)
            after = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            student = reader.functional_reader_trace(model, rows, facts, cached)
        self.assertTrue(all(torch.equal(original[key], after[key]) for key in original))
        self.assertFalse(torch.equal(student["semantic"], original["semantic"]))
        self.assertFalse(torch.equal(student["identity"], original["identity"]))
        self.assertFalse(torch.equal(student["temporal_gate"], original["temporal_gate"]))
        with self.assertRaisesRegex(ValueError, "own"):
            reader.functional_reader_trace(parent, rows, facts, cached)

    def test_complete_trace_matches_ordinary_score_and_exact_gate_product(self):
        model, facts, rows, cached = self.fixture()
        trace = reader.functional_reader_trace(model, rows, facts, cached)
        bank, identities, mask, metadata = reader.make_bank(facts, cached)
        query, identity = reader.query(model, rows, cached)
        ordinary = model.score_bank(query, identity, bank, identities, metadata, mask, memory_context_latents=(query).new_zeros((len(mask), 16)))
        torch.testing.assert_close(trace["activation"], ordinary, atol=0, rtol=0)
        probability = (F.logsigmoid(trace["semantic"])
                       + F.logsigmoid(trace["identity"])
                       + trace["temporal_gate"].log()).clamp_max(-1e-6)
        decomposed = probability - torch.log(-torch.expm1(probability))
        torch.testing.assert_close(trace["activation"], decomposed, atol=3e-6, rtol=3e-6)

    def test_initial_preservation_is_zero_while_old_gold_replay_remains_supervised(self):
        model, facts, rows, cached = self.fixture()
        rows = self.correct_parent_rows(model, facts, rows, cached)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        loss, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        self.assertEqual(float(stats["margin"]), 0.)
        self.assertEqual(float(stats["distillation"]), 0.)
        self.assertEqual(int(stats["protected"]), len(rows))
        self.assertGreater(float(stats["supervised"]), 0.)
        torch.testing.assert_close(loss.detach(), stats["supervised"], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)

    def test_wrong_parent_complete_set_receives_gold_supervision_without_preservation(self):
        model, facts, rows, cached = self.fixture()
        rows = self.correct_parent_rows(model, facts, rows[:1], cached)
        # Flip one old fit label so the parent makes an actual full-set error.
        first = facts[0]["text"]
        if first in rows[0]["support_texts"]:
            rows[0]["support_texts"].remove(first)
        else:
            rows[0]["support_texts"].append(first)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        loss, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        self.assertEqual(int(stats["protected"]), 0)
        self.assertEqual(float(stats["margin"]), 0.)
        self.assertEqual(float(stats["distillation"]), 0.)
        self.assertGreater(float(stats["supervised"]), 0.)
        torch.testing.assert_close(loss.detach(), stats["supervised"], atol=0, rtol=0)
        loss.backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum())
                               for name, parameter in model.named_parameters()
                               if name.startswith("read.")), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in parent.parameters()))

    def test_replay_gradients_reach_joint_reader_but_no_frozen_gates_or_parent(self):
        model, facts, rows, cached = self.fixture()
        rows = self.correct_parent_rows(model, facts, rows, cached)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        parent_before = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
        with torch.no_grad():
            model.query_encode.bias.add_(.3)
        loss, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        self.assertGreater(float(stats["distillation"]), 0.)
        loss.backward()
        for group in ("query_encode.", "query_pool.", "read.", "read_update.", "memory_attention_logits"):
            gradients = [parameter.grad for name, parameter in model.named_parameters()
                         if name.startswith(group)]
            self.assertTrue(gradients and all(gradient is not None for gradient in gradients), group)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), group)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0., group)
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        for name, parameter in parent.named_parameters():
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, parent_before[name], atol=0, rtol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_complete_teacher_replay_and_joint_reader_gradients(self):
        model, facts, rows, cached = self.fixture("mps")
        rows = self.correct_parent_rows(model, facts, rows, cached)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        self.assertEqual(len(list(parent.parameters())), 91)
        self.assertIs(parent_cached.inputs, cached.inputs)
        self.assertIs(parent_cached.word_inputs, cached.word_inputs)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        self.assertEqual(len(frozen), 78)
        parent_before = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
        for (name, student), (parent_name, teacher) in zip(
                model.named_parameters(), parent.named_parameters()):
            self.assertEqual(name, parent_name)
            self.assertEqual(student.device.type, "mps")
            self.assertEqual(teacher.device.type, "mps")
            self.assertNotEqual(student.data_ptr(), teacher.data_ptr(), name)
            self.assertFalse(teacher.requires_grad, name)
        with torch.no_grad():
            parent_trace = reader.functional_reader_trace(parent, rows, facts, parent_cached)
        initial_loss, initial_stats = reader.functional_replay_loss(
            model, parent, rows, facts, cached, parent_cached)
        self.assertEqual(initial_loss.device.type, "mps")
        self.assertTrue(torch.isfinite(initial_loss))
        self.assertEqual(float(initial_stats["margin"]), 0.)
        self.assertEqual(float(initial_stats["distillation"]), 0.)
        self.assertEqual(int(initial_stats["protected"]), len(rows))
        del initial_loss
        with torch.no_grad():
            model.query_encode.bias.add_(.3)
            model.read_update.bias.add_(.25)
        loss, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(stats["distillation"]), 0.)
        loss.backward()
        for group in ("query_encode.", "query_pool.", "read.", "read_update.", "memory_attention_logits"):
            gradients = [parameter.grad for name, parameter in model.named_parameters()
                         if name.startswith(group)]
            self.assertTrue(gradients and all(gradient is not None for gradient in gradients), group)
            self.assertTrue(all(gradient.device.type == "mps" and torch.isfinite(gradient).all()
                                for gradient in gradients), group)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0., group)
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        for name, parameter in parent.named_parameters():
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, parent_before[name], atol=0, rtol=0)
        with torch.no_grad():
            after = reader.functional_reader_trace(parent, rows, facts, parent_cached)
        for name, expected in parent_trace.items():
            torch.testing.assert_close(after[name], expected, atol=0, rtol=0)

    def test_teacher_feature_binding_rejects_trainable_training_and_mismatched_precision(self):
        model, _, _, cached = self.fixture()
        with self.assertRaisesRegex(ValueError, "frozen evaluation"):
            cached.for_model(deepcopy(model))
        candidate = deepcopy(model).eval().requires_grad_(False)
        candidate.train()
        with self.assertRaisesRegex(ValueError, "frozen evaluation"):
            cached.for_model(candidate)
        candidate.eval().double()
        with self.assertRaisesRegex(ValueError, "precision"):
            cached.for_model(candidate)

    def test_replay_rejects_nonindependent_parents_and_nonfit_or_new_binding_rows(self):
        model, facts, rows, cached = self.fixture()
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        with self.assertRaisesRegex(ValueError, "independent"):
            reader.functional_replay_loss(parent, parent, rows, facts, parent_cached, parent_cached)
        parent.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        parent.eval()
        next(parent.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        parent.requires_grad_(False)
        for changes in (dict(evaluation_only=True), dict(split="development"), dict(binding_fit=True)):
            for damaged_rows, damaged_facts in (([dict(rows[0], **changes)], facts),
                                                (rows, [dict(facts[0], **changes)])):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    reader.functional_replay_loss(model, parent, damaged_rows, damaged_facts,
                                                  cached, parent_cached)

    def settings(self, **changes):
        values = dict(steps=1, eval_every=40, train_heads="semantic", semantic_topic_weight=0.,
                      resume="initial.pt", output="never-created", reader_anchor_weight=0.,
                      reader_anchor_margin=1., reader_functional_weight=1., reader_functional_margin=1.,
                      freeze_memory_encoder=True, feature_dropout=0., relation_paraphrases=False,
                      episode_relations=False, relation_bindings=False)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_invalid_modes_and_controls_fail_before_output_creation(self):
        cases = [dict(train_heads=mode) for mode in ("all", "read", "identity", "scope")]
        cases += [dict(resume=None), dict(freeze_memory_encoder=False), dict(feature_dropout=.1),
                  dict(train_heads="matcher", reader_anchor_weight=1.)]
        cases += [dict(reader_functional_weight=value) for value in (-1., float("nan"), float("inf"))]
        cases += [dict(reader_functional_margin=value) for value in (0., -1., float("nan"), float("inf"))]
        for changes in cases:
            with self.subTest(changes=changes), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("output was created")):
                with self.assertRaises(ValueError):
                    reader.train(self.settings(**changes))
        for mode in ("semantic", "matcher", "temporal"):
            with self.subTest(valid_mode=mode), \
                 patch.object(Path, "mkdir", side_effect=RuntimeError("valid output stage")):
                with self.assertRaisesRegex(RuntimeError, "valid output stage"):
                    reader.train(self.settings(train_heads=mode))


if __name__ == "__main__":
    unittest.main()
