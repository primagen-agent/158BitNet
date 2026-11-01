"""Complete fit support sets, frozen matcher inputs and capped decision margins."""

from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import NeuralSlots, QUERY, SEMANTIC, WIDTH
from neural_reader_anchor import (
    MATCHER_KEYS,
    build_reader_anchor_teacher,
    matcher_logits_from_parameters,
    matcher_pair_features,
    reader_support_anchor_loss,
    snapshot_reader_matcher,
)


def activation_logits(semantic, gate):
    gate = torch.as_tensor(gate, dtype=semantic.dtype, device=semantic.device).expand_as(semantic)
    probability = (F.logsigmoid(semantic) + gate.log()).clamp_max(-1e-6)
    return probability - torch.log(-torch.expm1(probability))


def scalar(value):
    return float(value.detach())


class ReaderMatcherAnchorContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(59371)
        torch.set_num_threads(1)

    def test_snapshot_contains_only_four_independent_detached_matcher_tensors(self):
        model = NeuralSlots()
        parent = snapshot_reader_matcher(model)
        self.assertEqual(tuple(parent), MATCHER_KEYS)
        self.assertEqual(len(parent), 4)
        before = {name: value.clone() for name, value in parent.items()}
        for name, value in parent.items():
            self.assertFalse(value.requires_grad)
            self.assertNotEqual(value.data_ptr(), model.state_dict()[name].data_ptr())
        with torch.no_grad():
            model.read[0].weight.add_(.25)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in parent.items()))

    def test_frozen_pair_and_static_parent_forward_match_the_model_in_both_precisions(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = NeuralSlots().to(dtype=dtype)
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                queries = torch.randn(3, QUERY, dtype=dtype)
                memories = torch.randn(4, 5, WIDTH, dtype=dtype)
                mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
                pairs = matcher_pair_features(queries, memories, mask, model.attention_scales())
                parent = snapshot_reader_matcher(model)
                expected = model.semantic_bank_logits(queries, memories, mask, memory_context_latents=(queries).new_zeros((len(mask), 16)))
                self.assertTrue(torch.equal(matcher_logits_from_parameters(pairs, parent), expected))
                with torch.no_grad():
                    model.read[2].bias.add_(.5)
                self.assertTrue(torch.equal(matcher_logits_from_parameters(pairs, parent), expected))
                self.assertFalse(torch.equal(model.semantic_bank_logits(queries, memories, mask, memory_context_latents=(queries).new_zeros((len(mask), 16))), expected))

    def test_initial_nonzero_parent_anchor_is_exact_zero_and_complete_nulls_are_protected(self):
        for dtype in (torch.float32, torch.float64):
            semantic = torch.tensor([[5., -5., 2.], [-4., -2., -7.], [-4., -2., -7.]], dtype=dtype)
            gold = torch.tensor([[1., 0., 1.], [0., 0., 0.], [1., 0., 0.]], dtype=dtype)
            teacher = build_reader_anchor_teacher(semantic, activation_logits(semantic, .9), gold,
                                                  fixed_log_gate=torch.full_like(semantic, .9).log(), split="fit", margin=1.)
            self.assertEqual(teacher.correct_queries.tolist(), [True, True, False])
            self.assertEqual(teacher.protected_queries.tolist(), [True, True, False])
            current = semantic.clone().requires_grad_()
            loss = reader_support_anchor_loss(current, teacher)
            self.assertEqual(scalar(loss), 0.)
            self.assertTrue(torch.equal(torch.autograd.grad(loss, current)[0], torch.zeros_like(current)))
            # Labels and parent values cannot mutate the cached teacher later.
            gold.fill_(1.)
            semantic.fill_(40.)
            self.assertEqual(teacher.correct_queries.tolist(), [True, True, False])
            self.assertEqual(scalar(reader_support_anchor_loss(current, teacher)), 0.)

    def test_capped_margin_allows_saturated_confidence_changes_but_penalizes_decision_drift(self):
        parent = torch.tensor([[10., -10.]], dtype=torch.float64)
        gold = torch.tensor([[1., 0.]], dtype=torch.float64)
        teacher = build_reader_anchor_teacher(parent, activation_logits(parent, 1.), gold,
                                              fixed_log_gate=torch.zeros_like(parent), split="fit", margin=1.)
        lower_confidence = torch.tensor([[2., -2.]], dtype=torch.float64, requires_grad=True)
        self.assertEqual(scalar(reader_support_anchor_loss(lower_confidence, teacher)), 0.)
        wrong = torch.tensor([[-.5, -2.]], dtype=torch.float64, requires_grad=True)
        loss = reader_support_anchor_loss(wrong, teacher)
        self.assertAlmostEqual(scalar(loss), 2.25, places=10)
        gradient = torch.autograd.grad(loss, wrong)[0]
        self.assertAlmostEqual(float(gradient[0, 0]), -3., places=10)
        self.assertEqual(float(gradient[0, 1]), 0.)

    def test_complete_query_maximum_does_not_dilute_one_lost_margin_with_unrelated_slots(self):
        results = []
        for negatives in (1, 99):
            parent = torch.tensor([[2.] + [-2.] * negatives], dtype=torch.float64)
            gold = torch.tensor([[1.] + [0.] * negatives], dtype=torch.float64)
            teacher = build_reader_anchor_teacher(parent, activation_logits(parent, 1.), gold,
                                                  fixed_log_gate=torch.zeros_like(parent), split="fit", margin=1.)
            current = torch.tensor([[.5] + [-2.] * negatives], dtype=torch.float64, requires_grad=True)
            loss = reader_support_anchor_loss(current, teacher)
            results.append((scalar(loss), float(torch.autograd.grad(loss, current)[0][0, 0])))
        self.assertEqual(results, [(.25, -1.), (.25, -1.)])

    def test_student_drift_cannot_remove_parent_masks_and_has_a_restoring_matcher_gradient(self):
        model = NeuralSlots().double()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name in MATCHER_KEYS:
                parameter.requires_grad_(True)
        with torch.no_grad():
            model.read[2].weight.zero_()
            model.read[2].bias.fill_(-.2)
        pairs = torch.randn(2, 3, 4 * SEMANTIC, dtype=torch.float64)
        parent = snapshot_reader_matcher(model)
        semantic = matcher_logits_from_parameters(pairs, parent)
        teacher = build_reader_anchor_teacher(semantic, activation_logits(semantic, .9),
                                              torch.zeros_like(semantic), fixed_log_gate=torch.full_like(semantic, .9).log(), split="fit")
        current = model.read(pairs).squeeze(-1)
        self.assertEqual(scalar(reader_support_anchor_loss(current, teacher)), 0.)
        with torch.no_grad():
            model.read[2].bias.add_(.5)
        current = model.read(pairs).squeeze(-1)
        self.assertTrue((activation_logits(current, .9) > 0).all())
        loss = reader_support_anchor_loss(current, teacher)
        self.assertGreater(scalar(loss), 0.)
        loss.backward()
        self.assertGreater(float(model.read[2].bias.grad), 0.)
        self.assertEqual(teacher.correct_queries.tolist(), [True, True])
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if name not in MATCHER_KEYS))

    def test_near_half_gates_unreachable_slots_and_empty_groups_stay_finite(self):
        for dtype in (torch.float32, torch.float64):
            parent = torch.tensor([[100., 100., 25.]], dtype=dtype)
            gate = torch.tensor([[.5, .49, .500001]], dtype=dtype)
            teacher = build_reader_anchor_teacher(parent, activation_logits(parent, gate),
                                                  torch.tensor([[0., 0., 1.]], dtype=dtype), fixed_log_gate=gate.log(), split="fit")
            self.assertEqual(teacher.eligible_pairs.tolist(), [[False, False, True]])
            current = parent.clone().requires_grad_()
            self.assertEqual(scalar(reader_support_anchor_loss(current, teacher)), 0.)
            shifted = current + torch.tensor([[1000., 1000., 0.]], dtype=dtype)
            self.assertEqual(scalar(reader_support_anchor_loss(shifted, teacher)), 0.)
            self.assertTrue(torch.isfinite(teacher.semantic_thresholds).all())
            for shape in ((0, 3), (2, 0)):
                empty = torch.empty(shape, dtype=dtype, requires_grad=True)
                teacher = build_reader_anchor_teacher(empty, empty, torch.zeros(shape, dtype=dtype), fixed_log_gate=torch.zeros_like(empty), split="fit")
                loss = reader_support_anchor_loss(empty, teacher)
                self.assertEqual(scalar(loss), 0.)
                self.assertEqual(torch.autograd.grad(loss, empty)[0].shape, empty.shape)
        parent = torch.tensor([[-2.]], dtype=torch.float64)
        teacher = build_reader_anchor_teacher(parent, activation_logits(parent, 1.), torch.ones_like(parent), fixed_log_gate=torch.zeros_like(parent), split="fit")
        corrected = torch.tensor([[4.]], dtype=torch.float64, requires_grad=True)
        self.assertEqual(scalar(reader_support_anchor_loss(corrected, teacher)), 0.)
        self.assertEqual(float(torch.autograd.grad(reader_support_anchor_loss(corrected, teacher), corrected)[0]), 0.)

    def test_rejects_nonfit_source_labels_invalid_geometry_and_nonfinite_settings(self):
        parent = torch.tensor([[1., -1.]], dtype=torch.float64)
        activation = activation_logits(parent, .9)
        gold = torch.tensor([[1., 0.]], dtype=torch.float64)
        for split in ("development", "qualification", "runtime", "source"):
            with self.assertRaisesRegex(ValueError, "must come from fit"):
                build_reader_anchor_teacher(parent, activation, gold, fixed_log_gate=torch.full_like(parent, .9).log(), split=split)
        with self.assertRaisesRegex(ValueError, "must come from fit"):
            build_reader_anchor_teacher(parent, activation, gold, fixed_log_gate=torch.full_like(parent, .9).log(), split="fit", evaluation_only=True)
        for invalid in (torch.tensor([[2., 0.]]), gold[:, :1], torch.tensor([[float("nan"), 0.]])):
            with self.assertRaises(ValueError):
                build_reader_anchor_teacher(parent, activation, invalid, fixed_log_gate=torch.full_like(parent, .9).log(), split="fit")
        for margin in (-1., 0., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "margin must be finite and positive"):
                build_reader_anchor_teacher(parent, activation, gold, fixed_log_gate=torch.full_like(parent, .9).log(), split="fit", margin=margin)
        teacher = build_reader_anchor_teacher(parent, activation, gold, fixed_log_gate=torch.full_like(parent, .9).log(), split="fit")
        for weight in (-1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "weight must be finite and nonnegative"):
                reader_support_anchor_loss(parent, teacher, weight)
        with self.assertRaises(ValueError):
            build_reader_anchor_teacher(parent, parent + 1., gold, fixed_log_gate=torch.full_like(parent, .9).log(), split="fit")
        with self.assertRaises(ValueError):
            reader_support_anchor_loss(parent.float(), teacher)
        with self.assertRaises(ValueError):
            matcher_pair_features(torch.randn(1, QUERY, requires_grad=True),
                                  torch.randn(1, 2, WIDTH), torch.ones(1, 2, dtype=torch.bool), torch.ones(4))
        with self.assertRaises(ValueError):
            matcher_pair_features(torch.randn(1, QUERY), torch.randn(1, 2, WIDTH),
                                  torch.zeros(1, 2, dtype=torch.bool), torch.ones(4))
        model = NeuralSlots()
        parameters = snapshot_reader_matcher(model)
        parameters.pop(MATCHER_KEYS[-1])
        with self.assertRaises(ValueError):
            matcher_logits_from_parameters(torch.randn(1, 2, 1024), parameters)

    def test_exact_half_actual_gate_is_unreachable_across_801_semantic_logits(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                semantic = torch.linspace(-20, 20, 801, dtype=dtype)[None]
                log_gate = torch.full_like(semantic, .5).log()
                teacher = build_reader_anchor_teacher(
                    semantic, activation_logits(semantic, .5), torch.zeros_like(semantic),
                    fixed_log_gate=log_gate, split="fit")
                self.assertEqual(teacher.correct_queries.tolist(), [True])
                self.assertFalse(teacher.eligible_pairs.any())
                self.assertFalse(teacher.protected_queries.any())
                current = torch.full_like(semantic, 1e30).requires_grad_()
                loss = reader_support_anchor_loss(current, teacher)
                self.assertEqual(scalar(loss), 0.)
                torch.testing.assert_close(torch.autograd.grad(loss, current)[0],
                                           torch.zeros_like(current), atol=0, rtol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "actual MPS required")
    def test_exact_half_actual_gate_has_zero_loss_and_gradient_on_mps(self):
        semantic = torch.linspace(-20, 20, 801, device="mps")[None]
        teacher = build_reader_anchor_teacher(
            semantic, activation_logits(semantic, .5), torch.zeros_like(semantic),
            fixed_log_gate=torch.full_like(semantic, .5).log(), split="fit")
        self.assertFalse(teacher.eligible_pairs.any())
        current = torch.full_like(semantic, 1e30).requires_grad_()
        loss = reader_support_anchor_loss(current, teacher)
        self.assertEqual(scalar(loss), 0.)
        torch.testing.assert_close(torch.autograd.grad(loss, current)[0],
                                   torch.zeros_like(current), atol=0, rtol=0)

    def test_one_log_step_around_half_has_exact_reachability_and_initial_zero_gradient(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                half_log = torch.tensor(.5, dtype=dtype).log()
                below = torch.nextafter(half_log, torch.tensor(-float("inf"), dtype=dtype))
                above = torch.nextafter(half_log, torch.tensor(float("inf"), dtype=dtype))
                log_gate = torch.stack((below, half_log, above))[None]
                parent = torch.zeros((1, 3), dtype=dtype)
                teacher = build_reader_anchor_teacher(
                    parent, activation_logits(parent, log_gate.exp()), torch.zeros_like(parent),
                    fixed_log_gate=log_gate, split="fit")
                self.assertEqual(teacher.eligible_pairs.tolist(), [[False, False, True]])
                expected_threshold = -torch.expm1(above - half_log).log()
                torch.testing.assert_close(teacher.semantic_thresholds[0, 2], expected_threshold, atol=0, rtol=0)
                current = parent.clone().requires_grad_()
                loss = reader_support_anchor_loss(current, teacher)
                self.assertEqual(scalar(loss), 0.)
                torch.testing.assert_close(torch.autograd.grad(loss, current)[0],
                                           torch.zeros_like(current), atol=0, rtol=0)
                shifted = torch.tensor([[1e30, 1e30, float(expected_threshold) + 2.]],
                                       dtype=dtype, requires_grad=True)
                restored = reader_support_anchor_loss(shifted, teacher)
                self.assertGreater(scalar(restored), 0.)
                gradient = torch.autograd.grad(restored, shifted)[0]
                torch.testing.assert_close(gradient[0, :2], torch.zeros(2, dtype=dtype), atol=0, rtol=0)
                self.assertGreater(float(gradient[0, 2]), 0.)

    def test_actual_gate_is_an_independent_frozen_teacher_snapshot(self):
        parent = torch.tensor([[3., -3.]], dtype=torch.float64)
        log_gate = torch.full_like(parent, .9).log().requires_grad_()
        teacher = build_reader_anchor_teacher(parent, activation_logits(parent, .9),
                                              torch.tensor([[1., 0.]], dtype=parent.dtype),
                                              fixed_log_gate=log_gate, split="fit")
        expected = teacher.fixed_log_gate.clone()
        self.assertFalse(teacher.fixed_log_gate.requires_grad)
        self.assertNotEqual(teacher.fixed_log_gate.data_ptr(), log_gate.data_ptr())
        with torch.no_grad():
            log_gate.fill_(-10.)
        torch.testing.assert_close(teacher.fixed_log_gate, expected, atol=0, rtol=0)
        current = parent.clone().requires_grad_()
        loss = reader_support_anchor_loss(current, teacher)
        self.assertEqual(scalar(loss), 0.)
        loss.backward()
        self.assertIsNone(log_gate.grad)
        torch.testing.assert_close(current.grad, torch.zeros_like(current), atol=0, rtol=0)

    def test_gate_is_required_aligned_finite_and_cannot_understate_activation_probability(self):
        parent = torch.tensor([[1., -1.]], dtype=torch.float64)
        activation = activation_logits(parent, .9)
        gold = torch.tensor([[1., 0.]], dtype=parent.dtype)
        with self.assertRaises(TypeError):
            build_reader_anchor_teacher(parent, activation, gold, split="fit")
        for gate in (None, torch.zeros(1, 1, dtype=parent.dtype), torch.zeros_like(parent).float(),
                     torch.zeros_like(parent, device="meta"), torch.full_like(parent, float("nan")),
                     torch.full_like(parent, float("inf")), torch.ones_like(parent),
                     torch.full_like(parent, -5.)):
            with self.subTest(gate=getattr(gate, "shape", None)), self.assertRaises(ValueError):
                build_reader_anchor_teacher(parent, activation, gold, fixed_log_gate=gate, split="fit")
        with self.assertRaisesRegex(ValueError, "semantic or fixed gate probability"):
            build_reader_anchor_teacher(parent, parent + 1., gold,
                                        fixed_log_gate=torch.zeros_like(parent), split="fit")

    def test_unreachable_pairs_are_masked_before_squaring_huge_finite_student_values(self):
        parent = torch.tensor([[-2., -2.]], dtype=torch.float64)
        gates = torch.tensor([[1., .5]], dtype=parent.dtype)
        teacher = build_reader_anchor_teacher(parent, activation_logits(parent, gates),
                                              torch.zeros_like(parent), fixed_log_gate=gates.log(), split="fit")
        current = torch.tensor([[-2., 1e200]], dtype=parent.dtype, requires_grad=True)
        loss = reader_support_anchor_loss(current, teacher)
        self.assertEqual(scalar(loss), 0.)
        torch.testing.assert_close(torch.autograd.grad(loss, current)[0], torch.zeros_like(current), atol=0, rtol=0)

    def test_finite_loss_overflow_and_non_numeric_settings_fail_closed(self):
        parent = torch.tensor([[-2.]], dtype=torch.float64)
        teacher = build_reader_anchor_teacher(parent, parent, torch.zeros_like(parent),
                                              fixed_log_gate=torch.zeros_like(parent), split="fit")
        for current, weight in ((torch.tensor([[1e200]], dtype=parent.dtype), 1.),
                                (torch.tensor([[2.]], dtype=parent.dtype), 1e308)):
            with self.assertRaisesRegex(ValueError, "arithmetic|weighted"):
                reader_support_anchor_loss(current, teacher, weight)
        for setting in (None, "1"):
            with self.assertRaises(ValueError):
                build_reader_anchor_teacher(parent, parent, torch.zeros_like(parent),
                                            fixed_log_gate=torch.zeros_like(parent), split="fit", margin=setting)
            with self.assertRaises(ValueError):
                reader_support_anchor_loss(parent, teacher, setting)


if __name__ == "__main__":
    unittest.main()
