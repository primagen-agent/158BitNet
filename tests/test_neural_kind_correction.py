"""Capped source-kind margins correct mistakes without confidence inflation."""

from dataclasses import fields
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_kind_correction import snapshot_kind_teacher, kind_correction_loss


def number(value):
    return float(value.detach())


class KindCorrectionContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71101)

    def test_teacher_clones_detaches_labels_logits_and_frozen_eligibility(self):
        logits = torch.tensor([[3., 1., 0.], [3., 0., 1.]], requires_grad=True)
        labels = torch.tensor([0, 2], dtype=torch.int32)
        teacher = snapshot_kind_teacher(logits, labels)
        self.assertEqual(teacher.correct.tolist(), [True, False])
        self.assertEqual(teacher.required_gap.tolist(), [1., 1.])
        self.assertEqual(teacher.class_counts.tolist(), [1, 0, 1])
        self.assertEqual(teacher.labels.dtype, torch.long)
        self.assertNotEqual(teacher.parent_logits.data_ptr(), logits.data_ptr())
        self.assertNotEqual(teacher.labels.data_ptr(), labels.data_ptr())
        snapshots = {field.name: getattr(teacher, field.name).clone()
                     for field in fields(teacher) if isinstance(getattr(teacher, field.name), torch.Tensor)}
        self.assertTrue(all(not value.requires_grad for value in snapshots.values()))
        with torch.no_grad():
            logits.fill_(100)
            labels.fill_(1)
        for name, value in snapshots.items():
            self.assertTrue(torch.equal(getattr(teacher, name), value), name)
        with self.assertRaises(AttributeError):
            teacher.margin = 4.

    def test_correct_parent_initial_preservation_and_gradient_are_exact_zero(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                parent = torch.tensor([[2., 1., 0.], [0., 2., 1.7], [-1., 0., .2]], dtype=dtype)
                teacher = snapshot_kind_teacher(parent, torch.tensor([0, 1, 2]))
                live = parent.clone().requires_grad_()
                loss, stats = kind_correction_loss(live, teacher)
                self.assertEqual(number(loss), 0.)
                self.assertEqual(number(stats["correct_preservation"]), 0.)
                self.assertEqual(number(stats["wrong_correction"]), 0.)
                self.assertEqual(int(stats["current_actual_correct"]), 3)
                self.assertTrue(torch.equal(torch.autograd.grad(loss, live)[0], torch.zeros_like(live)))

    def test_saturated_parent_keeps_only_capped_margin_not_original_confidence(self):
        parent = torch.tensor([[100., 0., -10.]], dtype=torch.float64)
        teacher = snapshot_kind_teacher(parent, torch.tensor([0]))
        live = torch.tensor([[2., 0., -.5]], dtype=torch.float64, requires_grad=True)
        loss, stats = kind_correction_loss(live, teacher)
        self.assertEqual(number(loss), 0.)
        self.assertEqual(number(stats["correct_preservation"]), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, live)[0], torch.zeros_like(live)))

    def test_wrong_parent_corrects_until_margin_without_student_removing_eligibility(self):
        parent = torch.tensor([[-2., 0., 1.]], dtype=torch.float64)
        teacher = snapshot_kind_teacher(parent, torch.tensor([0]))
        live = parent.clone().requires_grad_()
        loss, stats = kind_correction_loss(live, teacher)
        self.assertEqual(number(stats["correct_preservation"]), 0.)
        self.assertEqual(number(stats["wrong_correction"]), 16.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, live)[0],
                                    torch.tensor([[-8., 0., 8.]], dtype=torch.float64)))
        # Argmax is now right, but the immutable required margin is not met.
        weak = torch.tensor([[1.5, 1., 0.]], dtype=torch.float64, requires_grad=True)
        loss, stats = kind_correction_loss(weak, teacher)
        self.assertEqual(number(loss), .25)
        self.assertEqual(int(stats["current_actual_correct"]), 1)
        self.assertEqual(int(stats["parent_wrong_samples"]), 1)
        strong = torch.tensor([[2., 1., 0.]], dtype=torch.float64, requires_grad=True)
        loss, _ = kind_correction_loss(strong, teacher)
        self.assertEqual(number(loss), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, strong)[0], torch.zeros_like(strong)))

    def test_gold_vs_hardest_other_class_tracks_changed_rival_and_common_offset(self):
        parent = torch.tensor([[5., 2., 0.]], dtype=torch.float64)
        teacher = snapshot_kind_teacher(parent, torch.tensor([0]))
        student = torch.tensor([[5., 4., 4.75]], dtype=torch.float64, requires_grad=True)
        loss, _ = kind_correction_loss(student, teacher)
        self.assertEqual(number(loss), .5625)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, student)[0],
                                    torch.tensor([[-1.5, 0., 1.5]], dtype=torch.float64)))
        shifted = (student.detach() + 40).requires_grad_()
        shifted_loss, _ = kind_correction_loss(shifted, teacher)
        self.assertEqual(number(shifted_loss), number(loss))
        self.assertTrue(torch.equal(torch.autograd.grad(shifted_loss, shifted)[0],
                                    torch.tensor([[-1.5, 0., 1.5]], dtype=torch.float64)))

    def test_complete_class_balance_does_not_renormalize_rare_parent_error(self):
        parent = torch.tensor([[2., 0., -1.]] * 100 + [[0., 2., -1.], [0., -1., 2.]], dtype=torch.float64)
        parent[99] = torch.tensor([0., 1., -1.], dtype=torch.float64)
        labels = torch.tensor([0] * 100 + [1, 2])
        teacher = snapshot_kind_teacher(parent, labels)
        live = parent.clone().requires_grad_()
        loss, stats = kind_correction_loss(live, teacher)
        self.assertAlmostEqual(number(loss), 4 / 100 / 3, places=15)
        self.assertAlmostEqual(number(stats["wrong_correction"]), 4 / 100 / 3, places=15)
        self.assertEqual(number(stats["correct_preservation"]), 0.)
        gradient = torch.autograd.grad(loss, live)[0]
        self.assertAlmostEqual(number(gradient[99, 0]), -4 / 100 / 3, places=15)
        self.assertTrue(torch.equal(gradient[:99], torch.zeros_like(gradient[:99])))

    def test_absent_classes_skip_once_and_stats_cannot_alias_teacher_counts(self):
        parent = torch.tensor([[2., 0., -1.], [2., 0., -1.], [0., -1., 2.]], dtype=torch.float64)
        teacher = snapshot_kind_teacher(parent, torch.tensor([0, 0, 2]))
        live = torch.tensor([[-1., 0., -2.], [2., 0., -1.], [0., -1., 0.]], dtype=torch.float64, requires_grad=True)
        loss, stats = kind_correction_loss(live, teacher)
        self.assertEqual(number(loss), 1.5)
        self.assertEqual(number(stats["correct_preservation"]), 1.5)
        self.assertEqual(int(stats["samples"]), 3)
        self.assertEqual(stats["class_sample_counts"].tolist(), [2, 0, 1])
        stats["class_sample_counts"].fill_(100)
        self.assertEqual(teacher.class_counts.tolist(), [2, 0, 1])

    def test_original_source_labels_remain_independent_of_physical_history_status(self):
        facts = [dict(kind=0, source_kind=0, status=1), dict(kind=2, source_kind=2, status=3)]
        labels = torch.tensor([fact["source_kind"] for fact in facts])
        logits = torch.tensor([[3., 1., -1.], [0., -1., 3.]])
        teacher = snapshot_kind_teacher(logits, labels)
        self.assertEqual(teacher.labels.tolist(), [0, 2])
        loss, _ = kind_correction_loss(logits.requires_grad_(), teacher)
        self.assertEqual(number(loss), 0.)
        for fact in facts:
            fact["status"] = 0
        self.assertEqual(teacher.labels.tolist(), [0, 2])

    def test_parent_ties_and_smaller_declared_margin_stay_finite_and_start_at_zero(self):
        tied = torch.tensor([[1., 1., 0.]])
        teacher = snapshot_kind_teacher(tied, torch.tensor([0]))
        self.assertEqual(teacher.correct.tolist(), [True])
        self.assertEqual(teacher.required_gap.tolist(), [0.])
        live = tied.clone().requires_grad_()
        loss, _ = kind_correction_loss(live, teacher)
        self.assertEqual(number(loss), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, live)[0], torch.zeros_like(live)))
        parent = torch.tensor([[2., 0., 1.], [2., 0., 1.]])
        teacher = snapshot_kind_teacher(parent, torch.tensor([0, 2]), margin=.5)
        self.assertEqual(teacher.required_gap.tolist(), [.5, .5])

    def test_rejects_nonfit_bad_labels_geometry_precision_devices_and_nonfinite_values(self):
        parent = torch.tensor([[2., 0., 1.]], dtype=torch.float64)
        labels = torch.tensor([0])
        for split in ("development", "evaluation", "qualification", "runtime"):
            with self.assertRaisesRegex(ValueError, "must come from fit"):
                snapshot_kind_teacher(parent, labels, split=split)
        with self.assertRaisesRegex(ValueError, "must come from fit"):
            snapshot_kind_teacher(parent, labels, evaluation_only=True)
        for bad in (labels.float(), torch.tensor([True]), torch.tensor([-1]), torch.tensor([3]),
                    torch.tensor([[0]]), torch.tensor([0, 1]), torch.empty(1, dtype=torch.long, device="meta")):
            with self.assertRaises(ValueError):
                snapshot_kind_teacher(parent, bad)
        for bad in (torch.empty(0, 3), torch.zeros(1, 2), torch.zeros(1, 3, dtype=torch.long),
                    torch.full((1, 3), float("nan")), torch.full((1, 3), float("inf"))):
            with self.assertRaises(ValueError):
                snapshot_kind_teacher(bad, labels)
        for margin in (-1., 0., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "margin must be finite and positive"):
                snapshot_kind_teacher(parent, labels, margin=margin)
        teacher = snapshot_kind_teacher(parent, labels)
        for bad in (parent.float(), parent.expand(2, -1), torch.empty(1, 3, device="meta")):
            with self.assertRaises(ValueError):
                kind_correction_loss(bad, teacher)
        with self.assertRaisesRegex(ValueError, "frozen parent teacher"):
            kind_correction_loss(parent, None)

    def test_finite_but_overflowing_margin_arithmetic_fails_before_nonfinite_gradient(self):
        maximum = torch.finfo(torch.float32).max
        with self.assertRaisesRegex(ValueError, "margin arithmetic"):
            snapshot_kind_teacher(torch.tensor([[maximum, -maximum, -maximum]]), torch.tensor([0]))
        teacher = snapshot_kind_teacher(torch.tensor([[2., 0., 1.]]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "loss arithmetic"):
            kind_correction_loss(torch.tensor([[-1e20, 1e20, 0.]], requires_grad=True), teacher)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_snapshot_class_balance_and_finite_restoring_gradient(self):
        parent = torch.tensor([[4., 1., 0.], [3., 0., 1.], [0., 3., -1.]], device="mps")
        labels = torch.tensor([0, 2, 1], device="mps")
        teacher = snapshot_kind_teacher(parent, labels)
        self.assertEqual(teacher.class_counts.tolist(), [1, 1, 1])
        live = parent.clone().requires_grad_()
        loss, stats = kind_correction_loss(live, teacher)
        self.assertEqual(loss.device.type, "mps")
        self.assertEqual(number(stats["correct_preservation"]), 0.)
        self.assertGreater(number(stats["wrong_correction"]), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(live.grad).all())
        self.assertLess(number(live.grad[1, 2]), 0.)
        self.assertGreater(number(live.grad[1, 0]), 0.)
        self.assertTrue(torch.equal(live.grad[[0, 2]], torch.zeros_like(live.grad[[0, 2]])))
        self.assertFalse(teacher.parent_logits.requires_grad)
        self.assertFalse(teacher.required_gap.requires_grad)


if __name__ == "__main__":
    unittest.main()
