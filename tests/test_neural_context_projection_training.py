"""Projection-only refinement uses live raw summaries and an independent parent."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_slot_model import INPUT, WIDTH, PLAN_KEYS, QUERY_KEYS, NeuralSlots
import train_neural_dialogue_reader as reader

CONTEXT3 = {"read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"}
CONTEXT_QUERY1 = {"read_context_query.weight"}


def parsed(mode="context", **changes):
    arguments = reader.reader_argument_parser().parse_args([
        "--gguf", "unused.gguf", "--probe", "unused-probe", "--output", "unused-output",
        "--resume", "unused-resume.pt", "--planner-checkpoint", "unused-writer.pt", "--steps", "1",
        "--train-heads", mode, "--freeze-memory-encoder", "--semantic-topic-weight", "0",
        "--reader-functional-weight", "1",
    ])
    for key, value in changes.items(): setattr(arguments, key, value)
    return arguments


class ContextProjectionTrainingContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, mode, device="cpu"):
        torch.manual_seed(83217)
        model = NeuralSlots().eval()
        facts = [
            dict(text="Ada. I paint the bird feeder glossy green.", entity="Ada", reference="Ada", prop="appearance", kind=0, status=0),
            dict(text="Ada. I coat the bird feeder to protect its wood from rain.", entity="Ada", reference="Ada", prop="cause", kind=0, status=0),
            dict(text="Ada. I mix gravel into the rosemary potting soil.", entity="Ada", reference="Ada", prop="material", kind=0, status=0),
            dict(text="Ben. I paint the bird feeder glossy green.", entity="Ben", reference="Ben", prop="appearance", kind=0, status=0),
        ]
        original = "What color and finish does Ada choose for the bird feeder?"
        rows = [
            dict(text=original + "\nActivated evidence:\n" + facts[1]["text"] + "\n",
                 initial_text=original, initial_reference="Ada", entities=["Ada"], prop="appearance",
                 kind=0, semantic_specific=True, support_texts=[facts[0]["text"]]),
            dict(text="Why does Ada coat the bird feeder?", entities=["Ada"], prop="cause",
                 kind=0, semantic_specific=True, support_texts=[facts[1]["text"]]),
            dict(text="What screws does Ada use on the bird feeder?", entities=["Ada"], prop="component",
                 kind=0, semantic_specific=True, support_texts=[]),
        ]
        texts = sorted({row["text"] for row in facts + rows} | {"Ada", "Ben", original})
        raw = {}
        for index, text in enumerate(texts):
            count = 3 + index % 3
            raw[text] = dict(x=torch.randn(count, INPUT) * .2,
                bounds=[(i * len(text) // count, (i + 1) * len(text) // count) for i in range(count)])
        with torch.no_grad():
            model.read_context_delta.weight.normal_(0, .05)  # Current trained B is nonzero.
            model.identity_match[2].weight.zero_(); model.identity_match[2].bias.fill_(3.)
            model.version[2].weight.zero_(); model.version[2].bias.fill_(3.)
            model.intent.weight.zero_(); model.intent.bias[:3].fill_(3.)
        reader.configure_reader_heads(model, mode, freeze_memory_encoder=True)
        cached = reader.DynamicFeatures(model, {"fit": dict(facts=facts, queries=rows)}, raw,
                                        device, freeze_memory_encoder=True)
        model.to(device)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        return model, parent, facts, rows, cached, parent_cached

    def exercise_gradients(self, mode, device):
        model, parent, facts, rows, cached, parent_cached = self.fixture(mode, device)
        selected = CONTEXT_QUERY1 if mode == "context-query" else CONTEXT3
        parameters = dict(model.named_parameters())
        self.assertEqual({name for name, parameter in parameters.items() if parameter.requires_grad}, selected)
        self.assertEqual(sum(parameters[name].numel() for name in selected), 4096 if mode == "context-query" else 70656)
        frozen = {name: parameter.detach().clone() for name, parameter in parameters.items() if name not in selected}
        self.assertEqual(len(frozen), 90 if mode == "context-query" else 88)
        self.assertTrue(set(PLAN_KEYS + QUERY_KEYS) <= frozen.keys())
        parent_before = {name: value.detach().clone() for name, value in parent.state_dict().items()}
        topic = torch.nn.Linear(WIDTH, 3).to(device)
        main = reader.semantic_training_loss(model, rows, facts, cached, topic, [], topic_weight=0.)
        replay, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        self.assertEqual(float(stats["margin"]), 0.)
        self.assertEqual(float(stats["distillation"]), 0.)
        for label, loss in (("main", main), ("replay", replay)):
            gradients = torch.autograd.grad(loss, [parameters[name] for name in sorted(selected)], retain_graph=True)
            for name, gradient in zip(sorted(selected), gradients):
                self.assertEqual(gradient.device.type, device, (label, name))
                self.assertTrue(torch.isfinite(gradient).all(), (label, name))
                self.assertGreater(float(gradient.abs().sum()), 0., (label, name))
        before = {name: parameters[name].detach().clone() for name in selected}
        (main + replay).backward()
        optimizer = reader.reader_optimizer(model, topic, mode, .002, .01)
        self.assertEqual({id(parameter) for group in optimizer.param_groups for parameter in group["params"]},
                         {id(parameters[name]) for name in selected})
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertTrue(all(parameter.grad is None for parameter in topic.parameters()))
        optimizer.step()
        for name, parameter in parameters.items():
            if name in selected:
                self.assertFalse(torch.equal(parameter, before[name]), name)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
                self.assertEqual(parameter.detach().cpu().numpy().tobytes(), frozen[name].cpu().numpy().tobytes(), name)
        for name, parameter in parent.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            self.assertEqual(parameter.detach().cpu().numpy().tobytes(), parent_before[name].cpu().numpy().tobytes(), name)

    def test_cpu_context_query_one_gradient_and_other89_exact(self):
        self.exercise_gradients("context-query", "cpu")

    def test_cpu_context_three_gradients_and_original88_source46_exact(self):
        self.exercise_gradients("context", "cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_context_query_and_context_preserve_frozen_parameters(self):
        for mode in reader.CONTEXT_MODES:
            with self.subTest(mode=mode): self.exercise_gradients(mode, "mps")

    def exercise_live_sidecars(self, device):
        model, parent, facts, rows, cached, parent_cached = self.fixture("context", device)
        texts = [fact["text"] for fact in facts]
        raw_before = cached.kind_contexts.clone()
        with torch.no_grad():
            parent_initial = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            initial = reader.functional_reader_trace(model, rows, facts, cached)
            old_sidecars = cached.memory_context_latents(texts).clone()
            parent_sidecars = parent_cached.memory_context_latents(texts).clone()
        self.assertEqual(parent_cached.kind_contexts.data_ptr(), cached.kind_contexts.data_ptr())
        self.assertNotEqual(parent.read_context_memory.weight.data_ptr(), model.read_context_memory.weight.data_ptr())
        topic = torch.nn.Linear(WIDTH, 3).to(device)
        optimizer = reader.reader_optimizer(model, topic, "context", .002, .01)
        previous = old_sidecars
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            main = reader.semantic_training_loss(model, rows, facts, cached, topic, [], topic_weight=0.)
            replay, _ = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
            (main + replay).backward(); optimizer.step()
            with torch.no_grad():
                live = cached.memory_context_latents(texts).clone()
                expected = model.memory_context_latents(cached.kind_contexts[cached.indices(texts)])
                after_parent = reader.functional_reader_trace(parent, rows, facts, parent_cached)
                current = reader.functional_reader_trace(model, rows, facts, cached)
            self.assertFalse(torch.equal(live, previous))
            torch.testing.assert_close(live, expected, atol=0, rtol=0)
            torch.testing.assert_close(parent_cached.memory_context_latents(texts), parent_sidecars, atol=0, rtol=0)
            self.assertTrue(all(torch.equal(parent_initial[key], after_parent[key]) for key in parent_initial))
            for key in ("identity", "intent", "temporal_gate", "fixed_log_gate"):
                torch.testing.assert_close(current[key], initial[key], atol=0, rtol=0)
            previous = live
        torch.testing.assert_close(cached.kind_contexts, raw_before, atol=0, rtol=0)

    def test_two_steps_recompute_updated_memory_sidecars_and_leave_complete_parent_unchanged(self):
        self.exercise_live_sidecars("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_two_steps_recompute_student_sidecars_and_preserve_parent(self):
        self.exercise_live_sidecars("mps")

    def test_context_configuration_and_schema_contract_are_explicit(self):
        for mode in reader.CONTEXT_MODES:
            with self.assertRaisesRegex(ValueError, "frozen memory encoder"):
                reader.configure_reader_heads(NeuralSlots(), mode, freeze_memory_encoder=False)
            args = parsed(mode)
            self.assertEqual(args.train_heads, mode)
            self.assertTrue(args.freeze_memory_encoder)
            self.assertIn(mode, reader.RELATION_MODES)
            self.assertIn(mode, reader.FUNCTIONAL_MODES)
            contract = reader.context_refinement_contract(mode)
            self.assertEqual(contract["trainable_parameters"], 4096 if mode == "context-query" else 70656)
            self.assertEqual(contract["frozen_tensors"], 90 if mode == "context-query" else 88)
            self.assertIn("U recomputed live", contract["memory_context_inputs"])
            self.assertIn("independent frozen complete91", contract["teacher"])
        self.assertIsNone(reader.context_refinement_contract("semantic"))

    def test_unsafe_context_parser_settings_reject_before_file_or_output_io(self):
        changes = (dict(resume=None), dict(freeze_memory_encoder=False), dict(feature_dropout=.1),
                   dict(reader_functional_weight=0.), dict(reader_functional_weight=float("nan")),
                   dict(reader_functional_weight=float("inf")), dict(semantic_topic_weight=.1),
                   dict(reader_anchor_weight=.1), dict(matcher_corrections=True))
        for mode in reader.CONTEXT_MODES:
            for change in changes:
                with self.subTest(mode=mode, change=change), \
                     patch.object(Path, "mkdir", side_effect=AssertionError("premature output")), \
                     patch.object(Path, "read_text", side_effect=AssertionError("premature data read")), \
                     patch.object(torch, "load", side_effect=AssertionError("premature model read")), \
                     self.assertRaises(ValueError):
                    reader.train(parsed(mode, **change))
            with patch.object(Path, "mkdir", side_effect=RuntimeError("validated output stage")), \
                 self.assertRaisesRegex(RuntimeError, "validated output stage"):
                reader.train(parsed(mode))

    def test_context_guard_rejects_invalid_flags_modes_and_read4_corrections(self):
        valid = dict(mode="context", resumed=True, frozen=True, feature_dropout=0.,
                     functional_weight=1., topic_weight=0., anchor_weight=0., matcher_corrections=False)
        self.assertIs(reader.validate_context_refinement(True, **valid), True)
        for flag in (0, 1, None, "context"):
            with self.assertRaises(ValueError): reader.validate_context_refinement(flag, **valid)
        for change in (dict(mode="semantic"), dict(matcher_corrections=0), dict(topic_weight=float("nan"))):
            with self.assertRaises(ValueError): reader.validate_context_refinement(True, **(valid | change))
        model, parent, facts, rows, cached, parent_cached = self.fixture("context")
        with self.assertRaisesRegex(ValueError, "read4"):
            reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
        with self.assertRaisesRegex(ValueError, "read4"):
            reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached, semantic_corrections=True)

    def test_evaluation_rows_reject_before_live_feature_encoding(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture("context")
        for field, value in (("evaluation_only", True), ("split", "development"), ("evidence_challenge_view", 0)):
            invalid_rows = [dict(rows[0], **{field: value})]
            with patch.object(cached, "encode", side_effect=AssertionError("evaluation inputs encoded")), \
                 self.assertRaisesRegex(ValueError, "development or evaluation"):
                reader.semantic_training_loss(model, invalid_rows, facts, cached, torch.nn.Linear(WIDTH, 3), [], topic_weight=0.)
            with self.assertRaisesRegex(ValueError, "excluding evaluation"):
                reader.functional_replay_loss(model, parent, invalid_rows, facts, cached, parent_cached)


if __name__ == "__main__":
    unittest.main()
