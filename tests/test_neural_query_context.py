"""Original-feature query binding residual and frozen writer contracts."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from neural_slot_model import (NeuralSlots, INPUT, WIDTH, TOKEN_LIMIT, KEYS,
                               PLAN_KEYS, QUERY_KEYS, TENSOR_SHAPES,
                               require_parameters, export)


class QueryContextContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19031)

    def test_zero_projection_preserves_parent_query_binding_exactly(self):
        model = NeuralSlots()
        candidate, source, kinds = torch.randn(WIDTH), torch.randn(4, INPUT), torch.eye(4)
        raw = model.query_binding_context(torch.randn(7, INPUT), torch.randn(3, INPUT))
        expected = model._reference(candidate, source, kinds, True)
        actual = model.query_reference(candidate, source, kinds, raw_context=raw)
        for old, new in zip(expected, actual):
            torch.testing.assert_close(new, old, atol=0, rtol=0)
        self.assertIsNone(model.query_context.bias)
        self.assertEqual(float(model.query_context.weight.detach().abs().sum()), 0.)
        torch.testing.assert_close(model.query_binding_candidate(candidate, raw), candidate, atol=0, rtol=0)

    def test_raw_context_keeps_last_mean_normalization_separate_for_each_stream(self):
        current, context = torch.zeros(2, INPUT, dtype=torch.float64), torch.zeros(3, INPUT, dtype=torch.float64)
        current[:, 0] = torch.tensor([0., 6.])
        context[:, INPUT - 1] = torch.tensor([3., 0., 9.])
        expected = torch.zeros(4 * INPUT, dtype=torch.float64)
        expected[0], expected[INPUT] = 6 / (45 ** .5), 3 / (45 ** .5)
        expected[3 * INPUT - 1], expected[4 * INPUT - 1] = 9 / (97 ** .5), 4 / (97 ** .5)
        actual = NeuralSlots.query_binding_context(current, context)
        torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-15)
        torch.testing.assert_close(actual[:2 * INPUT].norm(), actual.new_tensor(1.), atol=1e-15, rtol=0)
        torch.testing.assert_close(actual[2 * INPUT:].norm(), actual.new_tensor(1.), atol=1e-15, rtol=0)
        self.assertEqual(actual.dtype, current.dtype)
        self.assertEqual(actual.device, current.device)
        zeros = NeuralSlots.query_binding_context(torch.zeros(1, INPUT), torch.zeros(1, INPUT))
        self.assertTrue(torch.equal(zeros, torch.zeros(4 * INPUT)))

    def test_original_hidden_and_lexical_channels_receive_summary_gradients(self):
        current = torch.randn(3, INPUT, requires_grad=True)
        context = torch.randn(4, INPUT, requires_grad=True)
        raw = NeuralSlots.query_binding_context(current, context)
        (raw * torch.randn_like(raw)).sum().backward()
        for features in (current, context):
            self.assertTrue(torch.isfinite(features.grad).all())
            for block in features.grad.split(INPUT // 2, dim=-1):
                self.assertGreater(float(block.abs().sum()), 0.)

    def test_nonzero_projection_changes_need_bio_and_origin_using_raw_context(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.query_context.weight.zero_()
            model.query_context.weight[0, 2 * INPUT] = 2.
            model.query_need.weight.zero_()
            model.query_need.weight[0, 0] = 3.
            model.query_need.bias.zero_()
            model.query_ref_query.weight.zero_()
            model.query_ref_query.weight[0, 0] = 1.
            model.query_ref_query.bias.zero_()
            for layer in (model.query_ref_pair[0], model.query_ref_pair[2]):
                layer.weight.zero_()
                layer.bias.zero_()
            model.query_ref_pair[0].weight[0, 0] = 1.
            model.query_ref_pair[2].weight[:, 0] = torch.tensor([1., 2., 3.])
            model.query_origin.weight.zero_()
            model.query_origin.weight[:, 0] = torch.tensor([0., 1., 2., 3.])
            model.query_origin.bias.zero_()
        candidate = torch.zeros(WIDTH, dtype=torch.float64)
        source, kinds = torch.randn(4, INPUT, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
        raw = torch.zeros(4 * INPUT, dtype=torch.float64)
        raw[2 * INPUT] = .5
        corrected = torch.zeros_like(candidate)
        corrected[0] = torch.tanh(candidate.new_tensor(1.))
        need, tags = model.query_reference(candidate, source, kinds, raw_context=raw)
        torch.testing.assert_close(need, 3 * corrected[0], atol=0, rtol=0)
        pair_signal = corrected[0].tanh().tanh()
        bio = pair_signal * candidate.new_tensor([1., 2., 3.])
        origins = corrected[0] * torch.arange(4, dtype=torch.float64)
        expected_tags = bio[None].expand(4, 3) + torch.stack((torch.zeros_like(origins), origins, origins), -1)
        torch.testing.assert_close(tags, expected_tags, atol=1e-15, rtol=1e-15)
        zero_need, zero_tags = model.query_reference(candidate, source, kinds, raw_context=torch.zeros_like(raw))
        self.assertEqual(float(zero_need.detach()), 0.)
        self.assertTrue(torch.equal(zero_tags, torch.zeros_like(zero_tags)))
        self.assertGreater(float(tags[:, 0].detach().abs().sum()), 0.)
        self.assertGreater(float((tags[3, 1:] - tags[0, 1:]).detach().abs().sum()), 0.)

    def test_nonzero_query_context_leaves_writer_encoding_and_reference_exact(self):
        model = NeuralSlots()
        current, context, source = torch.randn(5, INPUT), torch.randn(3, INPUT), torch.randn(4, INPUT)
        kinds = torch.eye(4)
        before = model.plan_with_encodings(current, context, current)
        candidate = before[0].mean(0)
        reference = model.plan_reference(candidate, source, kinds)
        raw = model.query_binding_context(current, context)
        query_before = model.query_reference(candidate, source, kinds, raw_context=raw)
        with torch.no_grad():
            model.query_context.weight.normal_(0, .2)
        after = model.plan_with_encodings(current, context, current)
        for old, new in zip(before, after):
            torch.testing.assert_close(new, old, atol=0, rtol=0)
        for old, new in zip(reference, model.plan_reference(candidate, source, kinds)):
            torch.testing.assert_close(new, old, atol=0, rtol=0)
        query_after = model.query_reference(candidate, source, kinds, raw_context=raw)
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(query_before, query_after)))
        torch.testing.assert_close(model.query_binding_candidate(candidate, torch.zeros_like(raw)), candidate, atol=0, rtol=0)

    def test_constructor_preserves_source_rng_and_all_current_tensor_keys(self):
        original, captured = torch.nn.Linear, []
        def linear(*args, **kwargs):
            if args == (4 * INPUT, WIDTH) and kwargs.get("bias") is False:
                captured.append(torch.get_rng_state().clone())
            return original(*args, **kwargs)
        with patch("neural_slot_model.nn.Linear", side_effect=linear):
            model = NeuralSlots()
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), captured[0]))
        self.assertEqual(KEYS[-1], "query_context.weight")
        self.assertEqual(TENSOR_SHAPES[KEYS[-1]], (WIDTH, 4 * INPUT))
        self.assertEqual(len(KEYS), 91)
        self.assertEqual(set(model.state_dict()), set(KEYS))
        self.assertEqual(len(PLAN_KEYS), 33)
        self.assertEqual(len(QUERY_KEYS), 13)
        self.assertIn("query_context.weight", QUERY_KEYS)

    def exercise_frozen_parent(self, device):
        model = NeuralSlots().to(device)
        parent = deepcopy(model).requires_grad_(False)
        model.requires_grad_(False)
        model.query_context.weight.requires_grad_(True)
        before = {name: value.clone() for name, value in parent.state_dict().items()}
        candidate = torch.randn(WIDTH, device=device)
        words, kinds = torch.randn(4, INPUT, device=device), torch.eye(4, device=device)
        raw = model.query_binding_context(torch.randn(5, INPUT, device=device), torch.randn(3, INPUT, device=device))
        need, tags = model.query_reference(candidate, words, kinds, raw_context=raw)
        loss = (need - 2).square() + F.cross_entropy(tags, torch.tensor([0, 1, 2, 0], device=device))
        loss.backward()
        gradient = model.query_context.weight.grad
        self.assertEqual(gradient.device.type, device)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        torch.optim.SGD([model.query_context.weight], lr=.1).step()
        self.assertFalse(torch.equal(model.query_context.weight, before["query_context.weight"]))
        for name, parameter in model.named_parameters():
            if name != "query_context.weight":
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        for name, parameter in parent.named_parameters():
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        if device == "mps":
            cpu = deepcopy(model).cpu()
            actual = model.query_reference(candidate, words, kinds, raw_context=raw)
            expected = cpu.query_reference(candidate.cpu(), words.cpu(), kinds.cpu(), raw_context=raw.cpu())
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value.cpu(), reference, atol=3e-5, rtol=3e-5)

    def test_cpu_projection_gradient_preserves_all_frozen_parent_tensors(self):
        self.exercise_frozen_parent("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_projection_gradient_frozen_parent_and_cpu_parity(self):
        self.exercise_frozen_parent("mps")

    def test_query_reference_requires_explicit_raw_context(self):
        model = NeuralSlots()
        with self.assertRaises(TypeError):
            model.query_reference(torch.zeros(WIDTH), torch.zeros(1, INPUT), torch.eye(4)[:1])

    def test_raw_context_rejects_malformed_nonfinite_and_misaligned_token_rows(self):
        good = torch.zeros(1, INPUT)
        bad = [None, [], torch.zeros(INPUT), torch.zeros(0, INPUT), torch.zeros(2, INPUT - 1),
               torch.zeros(1, INPUT, dtype=torch.int64), torch.zeros(1, INPUT, dtype=torch.float64),
               torch.zeros(1, INPUT, device="meta"), good.expand(TOKEN_LIMIT, -1),
               torch.full_like(good, float("nan")), torch.full_like(good, float("inf"))]
        for value in bad:
            for current, context in ((value, good), (good, value)):
                with self.subTest(shape=getattr(value, "shape", None)):
                    with self.assertRaises(ValueError):
                        NeuralSlots.query_binding_context(current, context)

    def test_candidate_projection_rejects_malformed_nonfinite_and_misaligned_inputs(self):
        model = NeuralSlots()
        candidate, raw = torch.zeros(WIDTH), torch.zeros(4 * INPUT)
        bad_candidates = [None, [], torch.zeros(WIDTH - 1), torch.zeros(WIDTH, dtype=torch.int64),
                          candidate.double(), torch.zeros(WIDTH, device="meta"),
                          torch.full_like(candidate, float("nan"))]
        bad_contexts = [None, [], torch.zeros(4 * INPUT - 1), raw[None], raw.long(), raw.double(),
                        torch.zeros(4 * INPUT, device="meta"), torch.full_like(raw, float("inf"))]
        for item, context in [(value, raw) for value in bad_candidates] + [(candidate, value) for value in bad_contexts]:
            with self.assertRaises(ValueError):
                model.query_binding_candidate(item, context)

    def test_current_checkpoint_and_export_reject_missing_or_malformed_query_projection(self):
        model = NeuralSlots()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        old = {name: value for name, value in before.items() if name != "query_context.weight"}
        malformed = dict(before, **{"query_context.weight": torch.zeros(WIDTH, 2 * INPUT)})
        nonfinite = dict(before, **{"query_context.weight": torch.full((WIDTH, 4 * INPUT), float("nan"))})
        for state in (old, malformed, nonfinite):
            with self.assertRaises(ValueError):
                require_parameters(model, state)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            artifact = Path(directory) / "invalid.bnsmodel"
            model.query_context = torch.nn.Linear(2 * INPUT, WIDTH, bias=False)
            with self.assertRaisesRegex(ValueError, "geometry"):
                export(model, "ab" * 32, artifact)
            self.assertFalse(artifact.exists())


if __name__ == "__main__":
    unittest.main()
