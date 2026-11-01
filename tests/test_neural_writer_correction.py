"""Full-lattice occupancy and conditional structure correction contracts."""

from dataclasses import FrozenInstanceError
from itertools import product
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, WIDTH, NeuralSlots
from neural_writer_correction import (
    build_writer_correction_teacher, conditional_structure_margin_loss,
    source_occupancy_margin_loss, strongest_wrong_occupancy_path,
    writer_correction_losses,
)
from train_neural_dialogue import plan_path_scores, writer_train_keys


def model(dtype=torch.float64):
    result = NeuralSlots().to(dtype)
    result.requires_grad_(False)
    with torch.no_grad():
        result.plan_start.zero_()
        result.plan_transition.zero_()
        result.plan_end.zero_()
    return result


def legal_paths(n):
    return torch.tensor([p for p in product(range(5), repeat=n)
                         if p[0] != 4 and all(b != 4 or a != 0 for a, b in zip(p, p[1:]))])


def fixture(dtype=torch.float64):
    m = model(dtype)
    base = torch.tensor([[0., 2., -4., -5., -10.],
                         [2., -2., -3., -4., 1.]], dtype=dtype)
    boundary = torch.zeros(2, 2, dtype=dtype)
    member = torch.tensor([0., -5.], dtype=dtype)
    gold = torch.tensor([1, 4])
    return m, base, boundary, member, gold


class WriterCorrections(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72041)

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is unavailable")
    def test_mps_cpu_viterbi_labels_keep_teacher_and_isolated_gradients_on_device(self):
        m = model(torch.float32).to("mps")
        base = torch.tensor([[0., 2., -8., -8., -9.],
                             [0., 2., -8., -8., 0.]], device="mps")
        boundary = torch.zeros(2, 2, device="mps")
        member = torch.full((2,), -10., device="mps")
        gold = torch.tensor([1, 4], device="mps")
        teacher = build_writer_correction_teacher(m, base, boundary, member, gold, split="fit")
        self.assertEqual(m.plan_labels(base, boundary[:, 1]).device.type, "cpu")
        for value in (teacher.target, teacher.occupancy, teacher.source_required_gap,
                      teacher.structure_required_gap):
            self.assertEqual(value.device.type, "mps")
        live_boundary = boundary.clone().requires_grad_()
        live_member = member.clone().requires_grad_()
        source, structure = writer_correction_losses(m, base, live_boundary, live_member, teacher)
        source_grads = torch.autograd.grad(source, (live_boundary, live_member),
                                           retain_graph=True, allow_unused=True)
        structure_grads = torch.autograd.grad(structure, (live_boundary, live_member),
                                              allow_unused=True)
        self.assertTrue(torch.isfinite(source).item() and torch.isfinite(structure).item())
        self.assertIsNone(source_grads[0])
        self.assertIsNone(structure_grads[1])
        self.assertTrue(torch.isfinite(source_grads[1]).all().item())
        self.assertTrue(torch.isfinite(structure_grads[0]).all().item())
        self.assertGreater(source_grads[1].abs().sum().item(), 0.)
        self.assertGreater(structure_grads[0].abs().sum().item(), 0.)

    def test_wrong_occupancy_max_marginal_values_and_gradients_match_exhaustive_lattices(self):
        for n in range(1, 5):
            for occupancy in (torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool),
                              torch.arange(n) % 2 == 0):
                with self.subTest(n=n, occupancy=occupancy.tolist()):
                    m = model()
                    with torch.no_grad():
                        m.plan_start.normal_(0, .2)
                        m.plan_transition.normal_(0, .2)
                        m.plan_end.normal_(0, .2)
                    logits = torch.randn(n, 5, dtype=torch.float64, requires_grad=True)
                    ends = torch.randn(n, dtype=torch.float64, requires_grad=True)
                    paths = legal_paths(n)
                    wrong = ((paths > 0) != occupancy[None]).any(-1)
                    expected = plan_path_scores(m, logits, paths[wrong], ends).amax()
                    actual = strongest_wrong_occupancy_path(m, logits, ends, occupancy)
                    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
                    expected_grad = torch.autograd.grad(expected, (logits, ends), retain_graph=True)
                    actual_grad = torch.autograd.grad(actual, (logits, ends))
                    for a, b in zip(actual_grad, expected_grad):
                        torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)

    def test_occupancy_positive_is_best_matching_path_and_does_not_require_frozen_action_gold(self):
        m = model()
        base = torch.tensor([[1., 3., -7., -8., -9.]], dtype=torch.float64)
        boundary = torch.zeros(1, 2, dtype=torch.float64)
        member = torch.tensor([-5.], dtype=torch.float64)
        # UPDATE loses permanently to WRITE on the same occupied word; the
        # membership head must still be able to learn that word's occupancy.
        t = build_writer_correction_teacher(m, base, boundary, member, torch.tensor([2]), split="fit")
        self.assertTrue(t.source_eligible)
        self.assertTrue(t.frozen_action_unreachable)
        self.assertFalse(t.structure_eligible)
        self.assertEqual(float(t.parent_source_gap), -3.)
        live = member.clone().requires_grad_()
        value = source_occupancy_margin_loss(m, base, boundary, live, t)
        self.assertEqual(float(value.detach()), 16.)
        value.backward()
        self.assertEqual(float(live.grad[0]), -8.)
        self.assertEqual(float(source_occupancy_margin_loss(m, base, boundary, torch.tensor([0.], dtype=live.dtype), t)), 0.)

    def test_source_margin_values_and_membership_gradients_match_exhaustive_short_lattices(self):
        for n in range(1, 5):
            for occupancy in (torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool),
                              torch.arange(n) % 2 == 0):
                m = model()
                base = torch.randn(n, 5, dtype=torch.float64)
                boundary = torch.randn(n, 2, dtype=torch.float64) / 10
                tags = [0 if not active else (4 if i and occupancy[i - 1] else 1)
                        for i, active in enumerate(occupancy)]
                parent_member = (torch.full((n,), -100., dtype=torch.float64) if occupancy.any()
                                 else torch.full((n,), 100., dtype=torch.float64))
                t = build_writer_correction_teacher(m, base, boundary, parent_member,
                                                    torch.tensor(tags), split="fit")
                self.assertTrue(t.source_eligible)
                live = torch.randn(n, dtype=torch.float64, requires_grad=True)
                logits = (base.detach() + F.pad(boundary[:, :1].detach().expand(-1, 3), (1, 1))
                          + F.pad(live[:, None].expand(-1, 4), (1, 0)))
                paths = legal_paths(n)
                matching = ((paths > 0) == occupancy[None]).all(-1)
                scores = plan_path_scores(m, logits, paths, boundary[:, 1].detach())
                gap = scores[matching].amax() - scores[~matching].amax()
                expected = F.relu(t.source_required_gap - gap).square()
                actual = source_occupancy_margin_loss(m, base, boundary, live, t)
                torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
                eg = torch.autograd.grad(expected, live, retain_graph=True)[0]
                ag = torch.autograd.grad(actual, live)[0]
                torch.testing.assert_close(ag, eg, atol=2e-13, rtol=2e-13)

    def test_conditional_structure_values_and_gradients_match_exhaustive_short_lattices(self):
        for n in range(2, 5):
            m = model()
            gold = torch.tensor([1] + [4] * (n - 1))
            base = torch.randn(n, 5, dtype=torch.float64) / 10
            base[:, 2:4] = -8.
            base[0, 1] = 3.
            base[1:, 1] = 4.
            base[1:, 4] = -2.
            parent_boundaries = torch.zeros(n, 2, dtype=torch.float64)
            member = torch.zeros(n, dtype=torch.float64)
            t = build_writer_correction_teacher(m, base, parent_boundaries, member, gold, split="fit")
            self.assertTrue(t.structure_eligible)
            live = torch.randn(n, 2, dtype=torch.float64, requires_grad=True) / 10
            paths = legal_paths(n)
            matching = ((paths > 0) == (gold > 0)[None]).all(-1)
            alternatives = paths[matching & (paths != gold[None]).any(-1)]
            logits = base.detach() + F.pad(live[:, :1].expand(-1, 3), (1, 1))
            gold_score = plan_path_scores(m, logits, gold[None], live[:, 1])[0]
            rival = plan_path_scores(m, logits, alternatives, live[:, 1]).amax()
            expected = F.relu(t.structure_required_gap - (gold_score - rival)).square()
            actual = conditional_structure_margin_loss(m, base, live, t)
            torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
            eg = torch.autograd.grad(expected, live, retain_graph=True)[0]
            ag = torch.autograd.grad(actual, live)[0]
            torch.testing.assert_close(ag, eg, atol=2e-13, rtol=2e-13)

    def test_real_source5_and_edge5_parameters_are_isolated_and_base_logits_detached(self):
        m = model()
        source = set(writer_train_keys("writer-context-residual")) | {"plan_membership_raw.weight"}
        edge = set(writer_train_keys("writer-context-content")) - source
        self.assertEqual((len(source), len(edge)), (5, 5))
        for name, p in m.named_parameters():
            p.requires_grad_(name in source | edge)
        with torch.no_grad():
            m.plan_source_context[2].weight.normal_(0, .03)
            m.plan_source_context[2].bias.fill_(-10.)
            m.plan_boundary[2].weight.normal_(0, .03)
        encoded = torch.randn(3, WIDTH, dtype=torch.float64) / 20
        body = torch.randn_like(encoded) / 20
        words = torch.randn(3, INPUT, dtype=torch.float64)
        base = torch.tensor([[0., 3., -6., -7., -10.],
                             [0., 5., -6., -7., -3.],
                             [0., 5., -6., -7., -3.]], dtype=torch.float64, requires_grad=True)
        boundaries = m.plan_boundary_scores(encoded, words)
        membership = m.plan_total_membership(encoded, body, words)
        t = build_writer_correction_teacher(m, base, boundaries.detach(), membership.detach(),
                                            torch.tensor([1, 4, 4]), split="fit")
        self.assertTrue(t.source_eligible)
        self.assertTrue(t.structure_eligible)
        source_loss, edge_loss = writer_correction_losses(m, base, boundaries, membership, t)
        source_loss.backward(retain_graph=True)
        self.assertIsNone(base.grad)
        for name, p in m.named_parameters():
            if name in source:
                self.assertIsNotNone(p.grad, name)
                self.assertGreater(float(p.grad.abs().sum()), 0., name)
            else:
                self.assertIsNone(p.grad, name)
        m.zero_grad(set_to_none=True)
        edge_loss.backward()
        self.assertIsNone(base.grad)
        for name, p in m.named_parameters():
            if name in edge:
                self.assertIsNotNone(p.grad, name)
                self.assertGreater(float(p.grad.abs().sum()), 0., name)
            else:
                self.assertIsNone(p.grad, name)

    def test_capped_source_correction_stops_after_margin_and_parent_mask_stays_fixed(self):
        for dtype in (torch.float32, torch.float64):
            m, base, boundary, member, gold = fixture(dtype)
            t = build_writer_correction_teacher(m, base, boundary, member, gold, split="fit")
            self.assertTrue(t.source_eligible)
            self.assertTrue(t.structure_eligible)
            self.assertEqual(float(conditional_structure_margin_loss(m, base, boundary, t)), 0.)
            improved = torch.tensor([0., 3.], dtype=dtype, requires_grad=True)
            value = source_occupancy_margin_loss(m, base, boundary, improved, t)
            self.assertEqual(float(value.detach()), 0.)
            value.backward()
            self.assertTrue(torch.equal(improved.grad, torch.zeros_like(improved)))
            self.assertTrue(t.source_eligible)

    def test_correct_conditional_structure_preserves_capped_parent_gap_and_has_zero_initial_gradient(self):
        m, base, boundary, member, gold = fixture()
        t = build_writer_correction_teacher(m, base, boundary, member, gold, split="fit")
        self.assertEqual(float(t.structure_required_gap), 1.)
        live = boundary.clone().requires_grad_()
        loss = conditional_structure_margin_loss(m, base, live, t)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(live.grad, torch.zeros_like(live)))
        # Changing a begin score enough to fragment the same gold occupancy
        # activates only the structural preservation objective.
        shifted = boundary.clone()
        shifted[1, 0] = 5.
        shifted.requires_grad_()
        self.assertGreater(float(conditional_structure_margin_loss(m, base, shifted, t).detach()), 0.)

    def test_structure_target_caps_at_attainable_frozen_action_gap(self):
        m = model()
        base = torch.tensor([[0., 4., 3.75, -9., -10.],
                             [0., 4., -9., -9., 1.]], dtype=torch.float64)
        boundary = torch.zeros(2, 2, dtype=torch.float64)
        member = torch.zeros(2, dtype=torch.float64)
        t = build_writer_correction_teacher(m, base, boundary, member, torch.tensor([1, 4]), split="fit")
        self.assertTrue(t.structure_eligible)
        self.assertEqual(float(t.frozen_action_gap), .25)
        self.assertEqual(float(t.structure_required_gap), .25)
        self.assertGreater(float(conditional_structure_margin_loss(m, base, boundary, t)), 0.)
        fixed = boundary.clone()
        fixed[1, 0] = -5.
        fixed.requires_grad_()
        loss = conditional_structure_margin_loss(m, base, fixed, t)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(fixed.grad, torch.zeros_like(fixed)))

    def test_correct_occupancy_on_wrong_structure_row_preserves_parent_capped_margin(self):
        m = model()
        base = torch.tensor([[0., 3., -8., -9., -10.],
                             [0., 4., -8., -9., -2.]], dtype=torch.float64)
        boundary = torch.zeros(2, 2, dtype=torch.float64)
        member = torch.zeros(2, dtype=torch.float64)
        t = build_writer_correction_teacher(m, base, boundary, member, torch.tensor([1, 4]), split="fit")
        self.assertFalse(t.source_eligible)
        self.assertTrue(t.source_preservation_eligible)
        self.assertTrue(t.structure_eligible)
        self.assertEqual(float(t.source_required_gap), 1.)
        unchanged = member.clone().requires_grad_()
        loss = source_occupancy_margin_loss(m, base, boundary, unchanged, t)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(unchanged.grad, torch.zeros_like(unchanged)))
        drifted = torch.full_like(member, -5., requires_grad=True)
        restoring = source_occupancy_margin_loss(m, base, boundary, drifted, t)
        self.assertGreater(float(restoring.detach()), 0.)
        restoring.backward()
        self.assertTrue((drifted.grad < 0).all())
        self.assertTrue(t.source_preservation_eligible)
        action_only = build_writer_correction_teacher(m, base[:1], boundary[:1], member[:1],
                                                     torch.tensor([2]), split="fit")
        self.assertTrue(action_only.frozen_action_unreachable)
        self.assertFalse(action_only.source_eligible)
        self.assertTrue(action_only.source_preservation_eligible)

    def test_null_and_single_word_spans_have_finite_differentiable_zero_structure_loss(self):
        for tags in ([0], [1], [2], [3], [0, 0, 0], [1, 0, 2]):
            with self.subTest(tags=tags):
                m = model()
                n = len(tags)
                base = torch.randn(n, 5, dtype=torch.float64)
                boundary = torch.zeros(n, 2, dtype=torch.float64, requires_grad=True)
                member = torch.randn(n, dtype=torch.float64, requires_grad=True)
                t = build_writer_correction_teacher(m, base, boundary.detach(), member.detach(),
                                                    torch.tensor(tags), split="fit")
                self.assertFalse(t.structure_eligible)
                loss = conditional_structure_margin_loss(m, base, boundary, t)
                self.assertEqual(float(loss.detach()), 0.)
                loss.backward()
                self.assertTrue(torch.equal(boundary.grad, torch.zeros_like(boundary)))
                source_loss = source_occupancy_margin_loss(m, base, boundary, member, t)
                self.assertTrue(bool(torch.isfinite(source_loss)))
                source_loss.backward()
                self.assertTrue(bool(torch.isfinite(member.grad).all()))

    def test_teachers_clone_parent_targets_are_detached_and_do_not_follow_student_masks(self):
        m, base, boundary, member, gold = fixture()
        t = build_writer_correction_teacher(m, base, boundary, member, gold, split="fit")
        original = (t.target.clone(), t.source_required_gap.clone(), t.structure_required_gap.clone())
        base.add_(40)
        boundary.add_(50)
        member.add_(100)
        gold.zero_()
        self.assertTrue(torch.equal(t.target, original[0]))
        self.assertTrue(torch.equal(t.source_required_gap, original[1]))
        self.assertTrue(torch.equal(t.structure_required_gap, original[2]))
        self.assertTrue(t.source_eligible)
        self.assertTrue(all(not value.requires_grad for value in vars(t).values() if isinstance(value, torch.Tensor)))
        with self.assertRaises(FrozenInstanceError):
            t.source_eligible = False

    def test_fit_only_geometry_valid_action_grammar_and_finite_configuration(self):
        m, base, boundary, member, gold = fixture()
        for split in ("development", "runtime", "locomo"):
            with self.assertRaisesRegex(ValueError, "fit-only"):
                build_writer_correction_teacher(m, base, boundary, member, gold, split=split)
        for margin in (0., -1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                build_writer_correction_teacher(m, base, boundary, member, gold, split="fit", margin=margin)
        for invalid in (torch.tensor([4, 4]), torch.tensor([0, 4]), torch.tensor([1, 5])):
            with self.assertRaisesRegex(ValueError, "grammar"):
                build_writer_correction_teacher(m, base, boundary, member, invalid, split="fit")
        for valid in (torch.tensor([2, 4]), torch.tensor([3, 4]), torch.tensor([1, 3])):
            build_writer_correction_teacher(m, base, boundary, member, valid, split="fit")
        with self.assertRaisesRegex(ValueError, "aligned"):
            build_writer_correction_teacher(m, base, boundary[:, :1], member, gold, split="fit")
        with self.assertRaisesRegex(ValueError, "finite"):
            build_writer_correction_teacher(m, base * float("nan"), boundary, member, gold, split="fit")
        t = build_writer_correction_teacher(m, base, boundary, member, gold, split="fit")
        for weight in (-1., float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                source_occupancy_margin_loss(m, base, boundary, member, t, weight=weight)
        m.plan_transition.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen source grammar"):
            writer_correction_losses(m, base, boundary, member, t)


if __name__ == "__main__":
    unittest.main()
