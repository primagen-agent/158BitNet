"""Time-channel retention and source-kind training over complete tiny banks."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, PLAN_KEYS, QUERY_KEYS, WIDTH
import train_neural_dialogue_reader as reader
import test_neural_reader_temporal as temporal_tests


READ_CONTEXT_KEYS = {"read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"}


class TemporalRetentionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device="cpu"):
        model, facts, rows, _ = temporal_tests.TemporalReaderContract().fixture()
        kinds = []
        for kind in (0, 1, 2):
            for index in range(12):
                kinds.append(dict(text=f"Ada. Source kind {kind} example {index}.",
                                  entity="Ada", reference="Ada", prop="source_kind",
                                  kind=kind, source_kind=kind))
        # A historical physical version still has an original current-tense
        # source. Its old logical kind must not become the classifier target.
        kinds[0] = deepcopy(facts[0])
        texts = sorted({row["text"] for row in facts + rows + kinds} | {"Ada", "Ben"})
        cache = {}
        for index, text in enumerate(texts):
            count = 3 + index % 3
            cache[text] = dict(x=torch.randn(count, INPUT),
                               bounds=[(i * len(text) // count, (i + 1) * len(text) // count)
                                       for i in range(count)])
        reader.configure_reader_heads(model, "temporal", freeze_memory_encoder=True)
        cached = reader.DynamicFeatures(model, {"fit": dict(facts=facts + kinds, queries=rows)},
                                        cache, device, freeze_memory_encoder=True)
        model.to(device)
        return model, facts, rows, kinds, cached

    def test_only_time10_updates_and_source_scope_and81_other_parameters_are_exact(self):
        model, facts, rows, kinds, cached = self.fixture()
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, temporal_tests.TEMPORAL_PARAMETERS)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        self.assertEqual(len(frozen), 81)
        self.assertTrue(READ_CONTEXT_KEYS <= frozen.keys())
        self.assertTrue(set(PLAN_KEYS + QUERY_KEYS) <= frozen.keys())

        def scope_scores():
            bank, ids, mask, metadata = reader.make_bank(facts, cached)
            query, identity = reader.query(model, rows, cached, scope=True)
            return model.score_bank(query, identity, bank, ids, metadata, mask, scope=True, memory_context_latents=(query).new_zeros((len(mask), 16)))

        with torch.no_grad():
            before = scope_scores().clone()
        optimizer = reader.reader_optimizer(model, torch.nn.Linear(WIDTH, 3), "temporal", .001, .01)
        self.assertEqual({id(parameter) for group in optimizer.param_groups for parameter in group["params"]},
                         {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        with patch.object(reader, "semantic_relation_auxiliary", side_effect=AssertionError("semantic correction used")):
            loss = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
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
                if name in READ_CONTEXT_KEYS:
                    self.assertFalse(parameter.requires_grad, name)
                    self.assertIsNone(parameter.grad, name)
                    self.assertEqual(parameter.detach().cpu().numpy().tobytes(),
                                     frozen[name].cpu().numpy().tobytes(), name)
        with torch.no_grad():
            torch.testing.assert_close(scope_scores(), before, atol=0, rtol=0)

    def test_source_kind36_is_separate_from_full_recall_bank_and_physical_history(self):
        model, facts, rows, kinds, cached = self.fixture()
        with patch.object(F, "cross_entropy", wraps=F.cross_entropy) as source_ce, \
             patch.object(model, "score_bank", wraps=model.score_bank) as scoring:
            loss = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(source_ce.call_count, 1)
        self.assertEqual(source_ce.call_args.args[1].tolist(), [0] * 12 + [1] * 12 + [2] * 12)
        self.assertEqual(scoring.call_args.args[2].shape[0], len(facts))
        self.assertEqual(source_ce.call_args.args[0].shape[0], 36)
        metadata = scoring.call_args.args[4]
        self.assertEqual(metadata[0, :3].tolist(), [0, 1, 0])
        self.assertEqual(facts[0]["status"], 1)
        self.assertEqual(facts[0]["source_kind"], 0)

    def test_hard_kind_straight_through_changes_only_unspecified_physical_metadata(self):
        model, facts, _, _, cached = self.fixture()
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([-2., -3., 2.]))
        _, _, _, metadata = reader.make_bank(facts, cached)
        self.assertEqual(metadata[:, :3].tolist(), [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
        fixed_gradient = torch.autograd.grad(metadata[:3, 0].sum(), model.kind.bias,
                                             retain_graph=True)[0]
        torch.testing.assert_close(fixed_gradient, torch.zeros_like(fixed_gradient), atol=0, rtol=0)
        live_gradient = torch.autograd.grad(metadata[3, 0], model.kind.bias)[0]
        self.assertTrue(torch.isfinite(live_gradient).all())
        self.assertGreater(float(live_gradient.abs().sum()), 0.)

    def test_parent_owns_kind_and_metadata_after_student_kind_flips(self):
        model, facts, rows, _, cached = self.fixture()
        facts, rows = [deepcopy(facts[2])], [deepcopy(rows[2])]
        facts[0].pop("status")
        rows[0]["support_texts"] = [facts[0]["text"]]
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([2., -2., -3.]))
            model.identity_match[2].weight.zero_()
            model.identity_match[2].bias.fill_(20.)
            model.version[2].weight.zero_()
            model.version[2].bias.fill_(20.)
            model.intent.weight.zero_()
            model.intent.bias[:3].copy_(torch.tensor([5., -3., -3.]))
            model.read[2].weight.zero_()
            model.read[2].bias.fill_(8.)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        parent_before = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
        initial, stats = reader.functional_replay_loss(model, parent, rows, facts, cached,
                                                       parent_cached, intent_weight=.25, temporal_only=True)
        self.assertEqual(int(stats["protected"]), 1)
        self.assertEqual(float(stats["margin"]), 0.)
        del initial
        with torch.no_grad():
            model.kind.bias.copy_(torch.tensor([-2., -3., 2.]))
        _, _, _, teacher_metadata = reader.make_bank(facts, parent_cached)
        _, _, _, student_metadata = reader.make_bank(facts, cached)
        self.assertEqual(teacher_metadata[0, :3].tolist(), [1, 0, 0])
        self.assertEqual(student_metadata[0, :3].tolist(), [0, 0, 1])
        loss, stats = reader.functional_replay_loss(model, parent, rows, facts, cached,
                                                   parent_cached, intent_weight=.25, temporal_only=True)
        self.assertEqual(int(stats["protected"]), 1)
        self.assertGreater(float(stats["margin"]), 0.)
        self.assertGreater(float(stats["distillation"]), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(model.kind.bias.grad).all())
        self.assertGreater(float(model.kind.bias.grad.abs().sum()), 0.)
        for name, parameter in parent.named_parameters():
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, parent_before[name], atol=0, rtol=0)

    def test_time_correction_ignores_impossible_semantics_and_unrelated_negatives(self):
        rows = [dict(text="Current Ada storage", entities=["Ada"], prop="storage",
                     kind=0, allowed_kinds=[0])]
        facts = [dict(text=str(index), entity=entity, prop=prop, kind=kind, status=status)
                 for index, (entity, prop, kind, status) in enumerate([
                     ("Ada", "storage", 0, 0), ("Ada", "storage", 0, 0),
                     ("Ada", "storage", 0, 1), ("Ada", "color", 0, 0),
                     ("Ben", "storage", 0, 1)])]
        logits = torch.tensor([[-2., -2., 2., 2., 2.]], requires_grad=True)
        gold = torch.tensor([[1., 1., 0., 0., 0.]])
        semantic = torch.tensor([[5., -5., 5., 5., 5.]])
        identity = torch.full_like(semantic, 5.)
        loss, stats = reader.temporal_activation_correction(logits, gold, rows, facts, semantic, identity)
        self.assertEqual(int(stats["eligible_positive_pairs"]), 1)
        self.assertEqual(int(stats["frozen_gate_blocked_gold_pairs"]), 1)
        self.assertEqual(int(stats["temporal_negative_pairs"]), 1)
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertLess(float(gradient[0, 0]), 0.)
        self.assertGreater(float(gradient[0, 2]), 0.)
        torch.testing.assert_close(gradient[0, [1, 3, 4]], torch.zeros(3), atol=0, rtol=0)
        self.assertFalse(semantic.requires_grad)

    def test_parent_time_correction_requires_own_frozen_pipeline(self):
        model, facts, rows, kinds, cached = self.fixture()
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        with patch.object(reader, "functional_reader_trace", wraps=reader.functional_reader_trace) as trace, \
             patch.object(reader, "temporal_activation_correction", wraps=reader.temporal_activation_correction) as correction:
            loss = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                                 parent=parent, parent_cached=parent_cached)
        self.assertTrue(torch.isfinite(loss))
        self.assertIs(trace.call_args.args[0], parent)
        self.assertIs(trace.call_args.args[3], parent_cached)
        self.assertFalse(correction.call_args.args[4].requires_grad)
        self.assertFalse(correction.call_args.args[5].requires_grad)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in parent.parameters()))

    def test_correction_rejects_bad_states_and_never_turns_a_gold_positive_into_negative(self):
        rows = [dict(text="Ada current location", entities=["Ada"], prop="storage",
                     kind=0, allowed_kinds=[0])]
        fact = dict(text="Ada original current source", entity="Ada", prop="storage",
                    kind=0, source_kind=0, status=1)
        logits = torch.tensor([[-2.]], requires_grad=True)
        loss, stats = reader.temporal_activation_correction(
            logits, torch.ones_like(logits), rows, [fact],
            torch.full_like(logits, 5.), torch.full_like(logits, 5.))
        # Authored gold wins if a fit label is inconsistent with allowed states;
        # correction must not silently rewrite it as a temporal negative.
        self.assertEqual(int(stats["temporal_negative_pairs"]), 0)
        self.assertLess(float(torch.autograd.grad(loss, logits)[0]), 0.)
        for changes in (dict(status=True), dict(status=2), dict(status=4)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                reader.temporal_activation_correction(logits, torch.zeros_like(logits), rows,
                                                      [dict(fact, **changes)],
                                                      torch.full_like(logits, 5.), torch.full_like(logits, 5.))
        without_status = dict(fact, kind=1)
        without_status.pop("status")
        with self.assertRaisesRegex(ValueError, "explicit status"):
            reader.temporal_activation_correction(logits, torch.zeros_like(logits), rows,
                                                  [without_status], torch.full_like(logits, 5.),
                                                  torch.full_like(logits, 5.))

    def test_mismatched_parent_context_is_rejected(self):
        model, facts, rows, kinds, cached = self.fixture()
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        for parent_arg, cached_arg in ((parent, None), (None, parent_cached), (parent, cached)):
            with self.subTest(parent=parent_arg is not None, features=cached_arg is not None):
                with self.assertRaises(ValueError):
                    reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                                  parent=parent_arg, parent_cached=cached_arg)
        parent.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                          parent=parent, parent_cached=parent_cached)

    def test_evaluation_rows_facts_and_kind_supervision_are_rejected(self):
        model, facts, rows, kinds, cached = self.fixture()
        for change in (dict(evaluation_only=True), dict(split="development"), dict(evidence_challenge_view=1)):
            for key in ("rows", "facts", "kinds"):
                values = dict(rows=rows, facts=facts, kinds=kinds)
                values[key] = [dict(values[key][0], **change)]
                with self.subTest(change=change, key=key), \
                     patch.object(reader, "make_bank", side_effect=AssertionError("evaluation entered forward")):
                    with self.assertRaises(ValueError):
                        reader.temporal_training_loss(model, values["rows"], values["facts"], cached,
                                                      kind_facts=values["kinds"])
        with self.assertRaisesRegex(ValueError, "source-kind"):
            reader.temporal_training_loss(model, rows, facts, cached, kind_facts=[])

    def settings(self, **changes):
        values = dict(steps=1, eval_every=40, train_heads="temporal", semantic_topic_weight=0.,
                      resume="parent.pt", output="never-created", reader_anchor_weight=0.,
                      reader_anchor_margin=1., reader_functional_weight=1., reader_functional_margin=1.,
                      freeze_memory_encoder=True, feature_dropout=0., relation_paraphrases=False,
                      episode_relations=False, relation_bindings=False)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_temporal_full_teacher_config_checks_are_before_output_creation(self):
        for change in (dict(resume=None), dict(freeze_memory_encoder=False), dict(feature_dropout=.1),
                       dict(semantic_topic_weight=.05), dict(reader_anchor_weight=1.)):
            with self.subTest(change=change), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("output was created")):
                with self.assertRaises(ValueError):
                    reader.train(self.settings(**change))
        with patch.object(Path, "mkdir", side_effect=RuntimeError("valid output stage")):
            with self.assertRaisesRegex(RuntimeError, "valid output stage"):
                reader.train(self.settings())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_time10_kind36_and_independent_full_teacher(self):
        model, facts, rows, kinds, cached = self.fixture("mps")
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        main = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                            parent=parent, parent_cached=parent_cached)
        replay, _ = reader.functional_replay_loss(model, parent, rows, facts, cached,
                                                   parent_cached, intent_weight=.25, temporal_only=True)
        loss = main + replay
        self.assertEqual(loss.device.type, "mps")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.kind.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in parent.parameters()))


if __name__ == "__main__":
    unittest.main()
