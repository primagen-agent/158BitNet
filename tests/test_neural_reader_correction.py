"""Frozen fit correction: complete sets, reachability and finite margins."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_reader_correction import snapshot_reader_corrections, reader_correction_loss


class ReaderCorrectionTests(unittest.TestCase):
    @staticmethod
    def activation(semantic, gate=.8):
        probability = semantic.sigmoid() * torch.as_tensor(gate, dtype=semantic.dtype, device=semantic.device)
        return probability.log() - torch.log1p(-probability)

    def teacher(self, semantic, gold, gate=.8, **options):
        fixed_log_gate = torch.as_tensor(gate, dtype=semantic.dtype, device=semantic.device).expand_as(semantic).log()
        return snapshot_reader_corrections(semantic, self.activation(semantic, gate), gold,
                                           fixed_log_gate=fixed_log_gate, split="fit", **options)

    def partial(self):
        semantic = torch.tensor([[2., -2., 2., -2.]], dtype=torch.float64)
        gold = torch.tensor([[1., 1., 0., 0.]], dtype=torch.float64)
        return semantic, self.teacher(semantic, gold)

    def test_partial_queries_preserve_their_correct_pairs(self):
        semantic, teacher = self.partial()
        self.assertEqual(teacher.correct_pairs.tolist(), [[True, False, False, True]])
        self.assertEqual(teacher.wrong_positive_pairs.tolist(), [[False, True, False, False]])
        self.assertEqual(teacher.wrong_negative_pairs.tolist(), [[False, False, True, False]])
        live = semantic.clone().requires_grad_()
        loss, stats = reader_correction_loss(live, teacher)
        self.assertGreater(float(loss.detach()), 0)
        self.assertEqual(float(stats['preservation']), 0)
        loss.backward()
        self.assertLess(float(live.grad[0, 1]), 0)
        self.assertGreater(float(live.grad[0, 2]), 0)
        self.assertEqual(float(live.grad[0, 0]), 0)
        self.assertEqual(float(live.grad[0, 3]), 0)

    def test_partial_correct_pair_regression_is_protected(self):
        semantic, teacher = self.partial()
        semantic[0, 0] = -2.
        live = semantic.requires_grad_()
        _, stats = reader_correction_loss(live, teacher)
        self.assertGreater(float(stats['preservation']), 0)
        stats_loss, _ = reader_correction_loss(live, teacher)
        stats_loss.backward()
        self.assertLess(float(live.grad[0, 0]), 0)

    def test_complete_correct_parent_has_exact_zero_initial_loss_and_gradient(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                semantic = torch.tensor([[2., -2.], [-3., -4.]], dtype=dtype, requires_grad=True)
                teacher = self.teacher(semantic, torch.tensor([[True, False], [False, False]]))
                loss, stats = reader_correction_loss(semantic, teacher)
                self.assertEqual(float(loss.detach()), 0)
                self.assertEqual(int(stats['wrong_positive_pairs']), 0)
                self.assertEqual(int(stats['wrong_negative_pairs']), 0)
                loss.backward()
                torch.testing.assert_close(semantic.grad, torch.zeros_like(semantic), atol=0, rtol=0)

    def test_correct_tiny_margin_is_retained_without_roundoff_clamping(self):
        semantic = torch.tensor([[0.]], dtype=torch.float64, requires_grad=True)
        activation = torch.zeros_like(semantic)
        teacher = snapshot_reader_corrections(semantic, activation, torch.zeros_like(semantic),
                                              fixed_log_gate=torch.zeros_like(semantic), split='fit')
        expected = teacher.signed_targets * (teacher.parent_semantic - teacher.semantic_thresholds)
        torch.testing.assert_close(teacher.required_margins, expected.clamp_max(1), atol=0, rtol=0)
        loss, _ = reader_correction_loss(semantic, teacher)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(semantic.grad), 0)

    def test_repaired_errors_stop_at_finite_margin(self):
        _, teacher = self.partial()
        live = (teacher.semantic_thresholds + teacher.signed_targets * 1.5).requires_grad_()
        loss, _ = reader_correction_loss(live, teacher)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        torch.testing.assert_close(live.grad, torch.zeros_like(live), atol=0, rtol=0)
        self.assertEqual(float(teacher.required_margins.max()), 1)

    def test_threshold_matches_fixed_gate_decision(self):
        semantic = torch.tensor([[2., -2.]], dtype=torch.float64)
        teacher = self.teacher(semantic, torch.tensor([[1, 0]]), gate=.8)
        expected = -torch.log(torch.tensor(.6, dtype=torch.float64))
        torch.testing.assert_close(teacher.semantic_thresholds, expected.expand_as(semantic), atol=1e-14, rtol=1e-14)
        boundary_probability = teacher.semantic_thresholds.sigmoid() * .8
        torch.testing.assert_close(boundary_probability, torch.full_like(semantic, .5), atol=1e-14, rtol=1e-14)

    def test_unreachable_positive_and_negative_have_no_loss_or_gradient(self):
        semantic = torch.tensor([[0., 2., -2.]], dtype=torch.float64)
        teacher = self.teacher(semantic, torch.tensor([[1, 0, 1]]), gate=.25)
        self.assertFalse(teacher.reachable_pairs.any())
        live = torch.full_like(semantic, 1e100, requires_grad=True)
        loss, stats = reader_correction_loss(live, teacher)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(int(stats['unreachable_positive_pairs']), 2)
        loss.backward()
        torch.testing.assert_close(live.grad, torch.zeros_like(live), atol=0, rtol=0)

    def test_half_gate_is_unreachable_and_slightly_larger_gate_has_finite_threshold(self):
        semantic = torch.zeros((1, 2), dtype=torch.float64)
        teacher = self.teacher(semantic, torch.ones_like(semantic),
                               gate=torch.tensor([[.5, .50000001]], dtype=semantic.dtype))
        self.assertEqual(teacher.reachable_pairs.tolist(), [[False, True]])
        self.assertTrue(torch.isfinite(teacher.semantic_thresholds).all())
        self.assertGreater(float(teacher.semantic_thresholds[0, 1]), 17)

    def test_errors_use_separate_query_means_and_preservation_uses_global_worst(self):
        semantic = torch.tensor([[-2., -3.], [2., 3.], [2., -2.]], dtype=torch.float64)
        gold = torch.tensor([[1, 1], [0, 0], [1, 0]])
        teacher = self.teacher(semantic, gold)
        live = semantic.clone()
        live[2, 0] = -2
        loss, stats = reader_correction_loss(live, teacher)
        signed = teacher.signed_targets * (live - teacher.semantic_thresholds)
        violations = (teacher.required_margins - signed).clamp_min(0).square()
        expected = violations[0].max() + violations[1].max() + violations[2].max()
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
        self.assertEqual(int(stats['positive_queries']), 1)
        self.assertEqual(int(stats['negative_queries']), 1)
        self.assertEqual(int(stats['protected_queries']), 1)

    def preservation_padding(self, device='cpu'):
        results = []
        for extra in (0, 23):
            parent = torch.full((1 + extra, 2), -2., dtype=torch.float32, device=device)
            teacher = self.teacher(parent, torch.zeros_like(parent), gate=1.)
            live = parent.clone()
            live[0, 0] = .5
            live.requires_grad_()
            loss, stats = reader_correction_loss(live, teacher)
            gradient = torch.autograd.grad(loss, live)[0]
            results.append((float(loss.detach()), float(gradient[0, 0])))
            self.assertEqual(int(stats['protected_queries']), 1 + extra)
            self.assertEqual(float(gradient[1:].abs().sum()), 0.)
        self.assertEqual(results, [(2.25, 3.), (2.25, 3.)])

    def test_23_unchanged_perfect_queries_cannot_dilute_one_lost_correct_negative(self):
        self.preservation_padding()

    @unittest.skipUnless(torch.backends.mps.is_available(), 'actual MPS required')
    def test_actual_mps_perfect_queries_do_not_weaken_correct_negative_protection(self):
        self.preservation_padding('mps')

    def test_partial_query_correct_negative_is_protected_after_other_errors_are_repaired(self):
        parent, teacher = self.partial()
        live = teacher.semantic_thresholds + teacher.signed_targets * 1.5
        live[0, 3] = teacher.semantic_thresholds[0, 3] + .5
        live.requires_grad_()
        loss, stats = reader_correction_loss(live, teacher)
        self.assertEqual(float(stats['correction_positive']), 0.)
        self.assertEqual(float(stats['correction_negative']), 0.)
        self.assertEqual(float(stats['preservation']), 2.25)
        gradient = torch.autograd.grad(loss, live)[0]
        self.assertEqual(gradient[0].tolist(), [0., 0., 0., 3.])
        self.assertEqual(teacher.correct_pairs.tolist(), [[True, False, False, True]])

    def test_global_preservation_gradient_reaches_only_the_worst_correct_query(self):
        parent = torch.full((3, 2), -2., dtype=torch.float64)
        teacher = self.teacher(parent, torch.zeros_like(parent), gate=1.)
        live = torch.tensor([[.5, -2.], [0., -2.], [-2., -2.]], dtype=parent.dtype, requires_grad=True)
        loss, stats = reader_correction_loss(live, teacher)
        self.assertEqual(float(stats['preservation']), 2.25)
        gradient = torch.autograd.grad(loss, live)[0]
        torch.testing.assert_close(gradient, torch.tensor([[3., 0.], [0., 0.], [0., 0.]], dtype=parent.dtype), atol=0, rtol=0)

    def test_tied_worst_correct_pairs_split_the_preservation_gradient(self):
        parent = torch.full((2, 2), -2., dtype=torch.float64)
        teacher = self.teacher(parent, torch.zeros_like(parent), gate=1.)
        live = torch.tensor([[.5, -2.], [.5, -2.]], dtype=parent.dtype, requires_grad=True)
        loss, _ = reader_correction_loss(live, teacher)
        self.assertEqual(float(loss.detach()), 2.25)
        torch.testing.assert_close(torch.autograd.grad(loss, live)[0],
                                   torch.tensor([[1.5, 0.], [1.5, 0.]], dtype=parent.dtype), atol=0, rtol=0)

    def test_multiple_error_queries_retain_class_mean_with_global_correct_pair_max(self):
        parent = torch.tensor([[-2., -3.], [-4., -1.], [2., 3.], [4., 1.], [-2., -2.], [-2., -2.]], dtype=torch.float64)
        gold = torch.tensor([[1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
        teacher = self.teacher(parent, gold, gate=1.)
        live = parent.clone()
        live[4, 0], live[5, 0] = .5, 0.
        loss, stats = reader_correction_loss(live, teacher)
        signed = teacher.signed_targets * (live - teacher.semantic_thresholds)
        violations = (teacher.required_margins - signed).clamp_min(0).square()
        expected_positive = violations[:2].amax(-1).mean()
        expected_negative = violations[2:4].amax(-1).mean()
        expected_preservation = violations[4:].amax()
        torch.testing.assert_close(stats['correction_positive'], expected_positive, atol=0, rtol=0)
        torch.testing.assert_close(stats['correction_negative'], expected_negative, atol=0, rtol=0)
        torch.testing.assert_close(stats['preservation'], expected_preservation, atol=0, rtol=0)
        torch.testing.assert_close(loss, expected_positive + expected_negative + expected_preservation, atol=0, rtol=0)

    def test_empty_error_queries_do_not_dilute_active_corrections(self):
        semantic, teacher = self.partial()
        original, _ = reader_correction_loss(semantic, teacher)
        padded = torch.cat((semantic, torch.tensor([[-2., -3., -4., -5.]], dtype=semantic.dtype)))
        gold = torch.cat((teacher.gold, torch.zeros((1, 4), dtype=torch.bool)))
        expanded = self.teacher(padded, gold)
        actual, _ = reader_correction_loss(padded, expanded)
        torch.testing.assert_close(actual, original, atol=0, rtol=0)

    def test_worst_pair_prevents_many_correct_slots_from_diluting_preservation(self):
        semantic = torch.full((1, 100), -2., dtype=torch.float64)
        teacher = self.teacher(semantic, torch.zeros_like(semantic))
        live = semantic.clone()
        live[0, 50] = 2
        loss, stats = reader_correction_loss(live, teacher)
        signed = teacher.signed_targets * (live - teacher.semantic_thresholds)
        expected = (teacher.required_margins - signed).clamp_min(0).square().max()
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
        self.assertEqual(int(stats['correct_pairs']), 100)

    def test_teacher_clones_labels_and_parent_outputs(self):
        semantic = torch.tensor([[2., -2.]], requires_grad=True)
        activation = self.activation(semantic)
        gold = torch.tensor([[True, False]])
        log_gate = torch.full_like(semantic, .8).log()
        teacher = snapshot_reader_corrections(semantic, activation, gold,
                                              fixed_log_gate=log_gate, split='fit')
        before = teacher.parent_semantic.clone()
        with torch.no_grad():
            semantic.add_(5)
            activation.add_(5)
            log_gate.add_(5)
            gold.logical_not_()
        torch.testing.assert_close(teacher.parent_semantic, before, atol=0, rtol=0)
        self.assertEqual(teacher.gold.tolist(), [[True, False]])
        self.assertFalse(teacher.parent_semantic.requires_grad)
        self.assertFalse(teacher.semantic_thresholds.requires_grad)
        torch.testing.assert_close(teacher.fixed_log_gate, torch.full_like(semantic, .8).log(), atol=0, rtol=0)
        with self.assertRaises(FrozenInstanceError):
            teacher.margin = 2

    def test_empty_queries_and_banks_are_safe(self):
        for shape in ((0, 3), (2, 0), (0, 0)):
            with self.subTest(shape=shape):
                live = torch.empty(shape, requires_grad=True)
                teacher = self.teacher(live, torch.empty(shape))
                loss, stats = reader_correction_loss(live, teacher)
                self.assertEqual(float(loss.detach()), 0)
                self.assertEqual(int(stats['reachable_pairs']), 0)
                loss.backward()
                self.assertEqual(tuple(live.grad.shape), shape)

    def test_fit_only_geometry_dtype_device_and_labels_are_checked(self):
        semantic, teacher = self.partial()
        activation, gold = teacher.parent_activation, teacher.gold
        invalid = [dict(split='development'), dict(split='sealed'), dict(evaluation_only=True),
                   dict(margin=0), dict(margin=None), dict(margin=float('nan')), dict(parent_semantic=semantic[0]),
                   dict(parent_activation=activation.float()), dict(gold=gold[:, :3]),
                   dict(fixed_log_gate=None), dict(fixed_log_gate=teacher.fixed_log_gate[:, :3]),
                   dict(fixed_log_gate=teacher.fixed_log_gate.float()),
                   dict(fixed_log_gate=teacher.fixed_log_gate.to('meta')),
                   dict(fixed_log_gate=torch.full_like(semantic, float('inf'))),
                   dict(fixed_log_gate=torch.ones_like(semantic)),
                   dict(fixed_log_gate=torch.full_like(semantic, -5.)),
                   dict(gold=torch.tensor([[1, 2, 0, 0]])), dict(gold=gold.to('meta')),
                   dict(parent_semantic=torch.full_like(semantic, float('inf'))),
                   dict(parent_activation=semantic + 1)]
        for options in invalid:
            arguments = dict(parent_semantic=semantic, parent_activation=activation, gold=gold,
                             fixed_log_gate=teacher.fixed_log_gate, split='fit')
            arguments.update(options)
            with self.subTest(options=list(options)), self.assertRaises(ValueError):
                snapshot_reader_corrections(**arguments)

    def test_student_validation_weight_zero_and_detached_statistics(self):
        semantic, teacher = self.partial()
        for live, weight in [(semantic.float(), 1), (semantic[:, :3], 1),
                             (semantic.to('meta'), 1), (torch.full_like(semantic, float('nan')), 1),
                             (semantic, -1), (semantic, None), (semantic, float('inf'))]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                reader_correction_loss(live, teacher, weight)
        live = semantic.clone().requires_grad_()
        loss, stats = reader_correction_loss(live, teacher, weight=0)
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(all(not value.requires_grad for value in stats.values() if isinstance(value, torch.Tensor)))
        loss.backward()
        torch.testing.assert_close(live.grad, torch.zeros_like(live), atol=0, rtol=0)

    def test_outer_weight_scales_loss_and_gradients_once(self):
        semantic, teacher = self.partial()
        a, b = semantic.clone().requires_grad_(), semantic.clone().requires_grad_()
        first, _ = reader_correction_loss(a, teacher)
        second, _ = reader_correction_loss(b, teacher, weight=3)
        first.backward(); second.backward()
        torch.testing.assert_close(second, first * 3, atol=0, rtol=0)
        torch.testing.assert_close(b.grad, a.grad * 3, atol=0, rtol=0)

    def test_overflow_from_finite_student_values_fails_closed(self):
        semantic, teacher = self.partial()
        semantic[0, 1] = -1e200
        with self.assertRaisesRegex(ValueError, 'arithmetic'):
            reader_correction_loss(semantic, teacher)

    def test_exact_half_gate_stays_unreachable_across_801_semantic_logits(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                semantic = torch.linspace(-20, 20, 801, dtype=dtype)[None]
                log_gate = torch.full_like(semantic, .5).log()
                teacher = snapshot_reader_corrections(semantic, self.activation(semantic, .5),
                    torch.ones_like(semantic), fixed_log_gate=log_gate, split='fit')
                self.assertFalse(teacher.reachable_pairs.any())
                self.assertEqual(int(teacher.wrong_positive_pairs.sum()), 0)
                loss, stats = reader_correction_loss(semantic.clone().requires_grad_(), teacher)
                self.assertEqual(float(loss.detach()), 0)
                self.assertEqual(int(stats['unreachable_positive_pairs']), 801)

    def test_one_representable_log_step_above_half_is_reachable_for_each_dtype(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                semantic = torch.zeros((1, 2), dtype=dtype)
                half_log = torch.tensor(.5, dtype=dtype).log()
                above = torch.nextafter(half_log, torch.tensor(float('inf'), dtype=dtype))
                log_gate = torch.stack((half_log, above))[None]
                activation = self.activation(semantic, log_gate.exp())
                teacher = snapshot_reader_corrections(semantic, activation, torch.ones_like(semantic),
                                                       fixed_log_gate=log_gate, split='fit')
                self.assertEqual(teacher.reachable_pairs.tolist(), [[False, True]])
                expected = -torch.expm1(above - half_log).log()
                torch.testing.assert_close(teacher.semantic_thresholds[0, 1], expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
