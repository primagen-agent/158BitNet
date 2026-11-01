"""Query-only refinement protects full-parent behavior with live feedback gates."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import KEYS, PLAN_KEYS, QUERY_KEYS, QUERY, SEMANTIC
import train_neural_dialogue_reader as reader
import test_neural_reader_correction_training as correction_fixtures
import test_neural_reader_functional_training as functional_fixtures


QUERY3 = {"query_encode.weight", "query_encode.bias", "query_pool.weight"}


class QueryTrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device="cpu"):
        model, _, facts, rows, cached, _ = correction_fixtures.ReaderCorrectionTrainingContract().fixture(device)
        reader.configure_reader_heads(model, "query", freeze_memory_encoder=True)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        return model, parent, facts, rows, cached, parent_cached

    def scope_scores(self, model, facts, rows, cached):
        with torch.no_grad():
            bank, ids, mask, metadata = reader.make_bank(facts, cached)
            q, identity = reader.query(model, rows, cached, scope=True)
            return model.score_bank(q, identity, bank, ids, metadata, mask, scope=True, memory_context_latents=(q).new_zeros((len(mask), 16)))

    def exercise_gradients(self, device):
        model, parent, facts, rows, cached, parent_cached = self.fixture(device)
        parameters = dict(model.named_parameters())
        trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
        self.assertEqual(trainable, QUERY3)
        self.assertEqual(set(model.state_dict()), set(KEYS))
        self.assertEqual(len(parameters), 91)
        frozen = {name: parameter.detach().clone() for name, parameter in parameters.items()
                  if not parameter.requires_grad}
        self.assertEqual(len(frozen), 88)
        self.assertEqual(len(set(PLAN_KEYS + QUERY_KEYS)), 46)
        self.assertTrue(set(PLAN_KEYS + QUERY_KEYS) <= frozen.keys())
        before_parent = {name: value.detach().clone() for name, value in parent.state_dict().items()}
        before_scope = self.scope_scores(model, facts, rows, cached)
        topic = torch.nn.Linear(64, 3).to(device)
        main = reader.semantic_training_loss(model, rows, facts, cached, topic, [], topic_weight=0.)
        replay, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        self.assertTrue(torch.isfinite(main + replay))
        self.assertEqual(float(stats["margin"]), 0.)
        self.assertEqual(float(stats["distillation"]), 0.)
        for label, loss in (("main", main), ("replay", replay)):
            gradients = torch.autograd.grad(loss, [parameters[name] for name in sorted(QUERY3)],
                                            retain_graph=True)
            for name, gradient in zip(sorted(QUERY3), gradients):
                self.assertEqual(gradient.device.type, device, (label, name))
                self.assertTrue(torch.isfinite(gradient).all(), (label, name))
                self.assertGreater(float(gradient.abs().sum()), 0., (label, name))
        (main + replay).backward()
        optimizer = reader.reader_optimizer(model, topic, "query", .002, .01)
        self.assertEqual({id(parameter) for group in optimizer.param_groups for parameter in group["params"]},
                         {id(parameters[name]) for name in QUERY3})
        before_query = {name: parameters[name].detach().clone() for name in QUERY3}
        optimizer.step()
        for name, parameter in parameters.items():
            if name in QUERY3:
                self.assertFalse(torch.equal(parameter, before_query[name]), name)
            else:
                self.assertIsNone(parameter.grad, name)
                self.assertEqual(parameter.detach().cpu().numpy().tobytes(),
                                 frozen[name].cpu().numpy().tobytes(), name)
        self.assertTrue(all(parameter.grad is None for parameter in topic.parameters()))
        torch.testing.assert_close(self.scope_scores(model, facts, rows, cached), before_scope, atol=0, rtol=0)
        for name, parameter in parent.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, before_parent[name], atol=0, rtol=0)

    def test_cpu_main_and_replay_train_query3_preserving_other88_source46_and_scope(self):
        self.exercise_gradients("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_main_and_replay_train_query3_preserving_other88_and_parent91(self):
        self.exercise_gradients("mps")

    def test_independent_complete_parent_owns_query_encoder_over_shared_raw_inputs(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        self.assertIs(parent_cached.model, parent)
        self.assertIs(cached.model, model)
        for name in ("inputs", "summaries", "kind_contexts", "word_inputs", "word_lex",
                     "lengths", "offsets", "word_lengths", "word_offsets"):
            self.assertIs(getattr(parent_cached, name), getattr(cached, name), name)
        for name, parameter in model.named_parameters():
            self.assertNotEqual(parameter.data_ptr(), dict(parent.named_parameters())[name].data_ptr(), name)
        with torch.no_grad():
            old_query, _ = reader.query(parent, rows, parent_cached)
            old_trace = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            old_tokens = parent_cached.encode([rows[0]["text"]], query=True)[0]
            model.query_encode.bias.add_(.3)
            model.query_pool.weight.mul_(1.5)
            live_query, _ = reader.query(model, rows, cached)
            live_tokens = cached.encode([rows[0]["text"]], query=True)[0]
        self.assertFalse(torch.equal(live_query, old_query))
        self.assertFalse(torch.equal(live_tokens, old_tokens))
        with patch.object(reader, "build_reader_functional_teacher", wraps=reader.build_reader_functional_teacher) as snapshot:
            reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        for index, key in enumerate(("activation", "semantic", "identity", "intent", "temporal_gate")):
            torch.testing.assert_close(snapshot.call_args.args[index], old_trace[key], atol=0, rtol=0)
        with torch.no_grad():
            after = reader.functional_reader_trace(parent, rows, facts, parent_cached)
        for key in old_trace:
            torch.testing.assert_close(after[key], old_trace[key], atol=0, rtol=0)

    def test_original_predicted_subject_reaches_identity_update_across_evidence(self):
        model, _, facts, rows, cached, _ = self.fixture()
        before = deepcopy(rows)
        initial = [dict(text=row.get("initial_text", row["text"]),
                        reference=row.get("initial_reference", row["reference"])) for row in rows]
        original_q, original_identity = reader.raw_query(model, initial, cached)
        current_q, current_identity = reader.raw_query(model, rows, cached)
        with patch.object(model, "advance_identity", wraps=model.advance_identity) as advance:
            _, identity = reader.query(model, rows, cached)
        self.assertEqual(advance.call_count, 1)
        for actual, expected in zip(advance.call_args.args,
                                    (original_q, current_q, original_identity, current_identity)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(original_identity[0], current_identity[0]))
        torch.testing.assert_close(original_identity[:1], cached.reference_ids([dict(text="Ada", reference="Ada")]), atol=0, rtol=0)
        torch.testing.assert_close(current_identity[:1], cached.reference_ids([dict(text="Ben", reference="Ben")]), atol=0, rtol=0)
        bank_ids = reader.make_bank(facts, cached)[1]
        scores = model.identity_logits(identity, bank_ids)
        self.assertGreater(float(scores[0, 0].detach()), float(scores[0, 3].detach()))
        self.assertEqual(rows, before)

    def test_frozen_feedback_parameters_allow_gate_drift_and_full_parent_loss_repairs_it(self):
        model, _, facts, rows, cached, _ = self.fixture()
        # Deliberately sensitive frozen gates expose input drift from query-only learning.
        with torch.no_grad():
            model.identity_update.weight.zero_()
            model.identity_update.bias.zero_()
            model.identity_update.weight[0, QUERY] = 2.
            model.read_update.weight.zero_()
            model.read_update.bias.zero_()
            model.read_update.weight[SEMANTIC, QUERY] = 2.
            model.intent.weight.zero_()
            model.intent.weight[0, 0] = 1.
            model.intent.bias[0] = .2
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        with torch.no_grad():
            old = reader.functional_reader_trace(parent, rows, facts, parent_cached)
        protected_rows = deepcopy(rows)
        for index, row in enumerate(protected_rows):
            row["support_texts"] = [fact["text"] for column, fact in enumerate(facts)
                                    if old["activation"][index, column] > 0]
        gold = torch.tensor([[reader.supported(row, fact) for fact in facts] for row in protected_rows],
                            dtype=old["activation"].dtype)
        teacher = reader.build_reader_functional_teacher(old["activation"], old["semantic"], old["identity"],
            old["intent"], old["temporal_gate"], gold, split="fit")
        self.assertTrue(teacher.protected_queries.all())
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        with torch.no_grad():
            model.query_encode.bias[0].add_(.4)
        live = reader.functional_reader_trace(model, protected_rows, facts, cached)
        for key in ("identity", "intent", "temporal_gate", "fixed_log_gate"):
            self.assertGreater(float((live[key] - old[key]).abs().max().detach()), 0., key)
        def preservation(trace):
            return reader.reader_functional_distillation_loss(trace["activation"], trace["semantic"],
                trace["identity"], trace["intent"], trace["temporal_gate"], teacher)
        loss = preservation(live)
        self.assertGreater(float(loss.detach()), 0.)
        gradient = torch.autograd.grad(loss, model.query_encode.bias)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[0]), 0.)
        _, stats = reader.functional_replay_loss(model, parent, protected_rows, facts, cached, parent_cached)
        torch.testing.assert_close(stats["distillation"], loss.detach(), atol=0, rtol=0)
        with torch.no_grad():
            model.query_encode.bias[0].sub_(.05)
        repaired = preservation(reader.functional_reader_trace(model, protected_rows, facts, cached))
        self.assertLess(float(repaired.detach()), float(loss.detach()))
        for name, expected in frozen.items():
            torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)

    def test_fixed_gate_matcher_correction_cannot_be_applied_to_trainable_query_inputs(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        with self.assertRaises(ValueError):
            reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
        with self.assertRaises(ValueError):
            reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached,
                                          semantic_corrections=True)

    def test_query_controls_require_independent_full_parent_before_creating_outputs(self):
        settings = functional_fixtures.ReaderFunctionalTrainingContract().settings
        model, _, _, _, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            reader.configure_reader_heads(model, "query", freeze_memory_encoder=False)
        valid = dict(resumed=True, frozen=True, feature_dropout=0., functional_weight=1.,
                     topic_weight=0., anchor_weight=0., matcher_corrections=False)
        self.assertIs(reader.validate_query_refinement(True, **valid), True)
        for flag in (0, 1, None, "query"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                reader.validate_query_refinement(flag, **valid)
        changes = (dict(resumed=False), dict(frozen=False), dict(feature_dropout=.1),
                   dict(functional_weight=0.), dict(functional_weight=float("nan")),
                   dict(functional_weight=float("inf")), dict(topic_weight=.1),
                   dict(anchor_weight=.1), dict(matcher_corrections=True), dict(matcher_corrections=0))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.validate_query_refinement(True, **(valid | change))
        train_changes = (dict(resume=None), dict(freeze_memory_encoder=False), dict(feature_dropout=.1),
                         dict(reader_functional_weight=0.), dict(semantic_topic_weight=.1),
                         dict(reader_anchor_weight=.1), dict(matcher_corrections=True))
        for change in train_changes:
            with self.subTest(train_change=change), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("output created before validation")), \
                 self.assertRaises(ValueError):
                reader.train(settings(train_heads="query", **change))
        with patch.object(Path, "mkdir", side_effect=RuntimeError("validated output stage")), \
             self.assertRaisesRegex(RuntimeError, "validated output stage"):
            reader.train(settings(train_heads="query"))

    def test_existing_minimal_namespace_does_not_require_new_query_controls(self):
        minimal = SimpleNamespace(steps=1, train_heads="matcher", semantic_topic_weight=.05)
        with self.assertRaisesRegex(ValueError, "semantic-topic-weight"):
            reader.train(minimal)
        minimal.semantic_topic_weight = 0.
        minimal.output = "never-created"
        with patch.object(Path, "mkdir", side_effect=RuntimeError("existing mode output stage")), \
             self.assertRaisesRegex(RuntimeError, "existing mode output stage"):
            reader.train(minimal)


if __name__ == "__main__":
    unittest.main()
