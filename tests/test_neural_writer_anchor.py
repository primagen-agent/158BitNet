"""Full source-lattice anchor values, derivatives and missing-competitor guards."""

import itertools
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_slot_model import NeuralSlots
from neural_writer_anchor import strongest_wrong_path, _ordered_scan
from train_neural_dialogue import (
    build_writer_teacher, path_anchor_required_gaps, plan_path_score,
    structured_path_anchor_loss,
)


def sequence_model(seed=7119, *, zero=False):
    model = NeuralSlots()
    generator = torch.Generator().manual_seed(seed)
    for name, shape in (("plan_start", (5,)), ("plan_end", (5,)), ("plan_transition", (5, 5))):
        value = torch.zeros(shape, dtype=torch.float64) if zero else torch.randn(
            shape, generator=generator, dtype=torch.float64
        ) * .7
        setattr(model, name, torch.nn.Parameter(value))
    return model


def exhaustive_wrong(model, logits, ends, target):
    scores = []
    gold = tuple(target.tolist())
    for path in itertools.product(range(5), repeat=len(target)):
        if path != gold:
            scores.append(plan_path_score(model, logits, target.new_tensor(path), ends))
    return torch.stack(scores).amax()


def gap(model, logits, ends, target):
    return plan_path_score(model, logits, target, ends) - strongest_wrong_path(model, logits, ends, target)


class FullWriterAnchor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_paths_length_one_to_four_match_values_and_gradients(self):
        targets = ([0], [1, 4], [1, 0, 1], [0, 1, 4, 0])
        for seed in (619, 1721, 3917):
            model = sequence_model(seed)
            generator = torch.Generator().manual_seed(seed + 1)
            for target_values in targets:
                with self.subTest(seed=seed, words=len(target_values)):
                    n = len(target_values)
                    logits = torch.randn(n, 5, generator=generator, dtype=torch.float64, requires_grad=True)
                    ends = torch.randn(n, generator=generator, dtype=torch.float64, requires_grad=True)
                    target = torch.tensor(target_values)
                    before_logits, before_ends, before_target = logits.detach().clone(), ends.detach().clone(), target.clone()
                    actual = strongest_wrong_path(model, logits, ends, target)
                    expected = exhaustive_wrong(model, logits, ends, target)
                    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
                    variables = (logits, ends, model.plan_start, model.plan_transition, model.plan_end)
                    gradients = torch.autograd.grad(actual, variables, allow_unused=True, retain_graph=True)
                    oracle = torch.autograd.grad(expected, variables, allow_unused=True)
                    for a, b in zip(gradients, oracle):
                        if a is None or b is None:
                            self.assertIs(a, b)
                        else:
                            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
                    self.assertTrue(torch.equal(logits.detach(), before_logits))
                    self.assertTrue(torch.equal(ends.detach(), before_ends))
                    self.assertTrue(torch.equal(target, before_target))

    def test_ordered_scans_cover_odd_lengths_without_commuting_edges(self):
        generator = torch.Generator().manual_seed(6619)
        matrices = torch.randn(9, 5, 5, generator=generator, dtype=torch.float64)
        prefix, suffix = _ordered_scan(matrices), _ordered_scan(matrices, suffix=True)
        for index in range(len(matrices)):
            left = matrices[0]
            for edge in matrices[1:index + 1]:
                left = (left[:, :, None] + edge[None]).amax(1)
            right = matrices[index]
            for edge in matrices[index + 1:]:
                right = (right[:, :, None] + edge[None]).amax(1)
            torch.testing.assert_close(prefix[index], left, rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(suffix[index], right, rtol=1e-11, atol=1e-11)

    def test_far_outside_new_source_is_penalized_in_128_word_plan(self):
        model = sequence_model(zero=True)
        target = torch.zeros(128, dtype=torch.long)
        target[4], target[5:8] = 1, 4
        parent = torch.full((128, 5), -4., dtype=torch.float64)
        parent[:, 0], parent[4, 1], parent[5:8, 4] = 0., 6., 6.
        ends = torch.zeros(128, dtype=torch.float64)
        begin = torch.zeros(128, dtype=torch.float64)
        begin[64] = 10.
        current = (parent + F.pad(begin[:, None].expand(-1, 3), (1, 1))).requires_grad_()
        self.assertTrue(torch.equal(model.plan_labels(parent, ends), target))
        self.assertFalse(torch.equal(model.plan_labels(current, ends), target))
        required = gap(model, parent, ends, target).detach().clamp(max=1000.)
        loss = F.relu(required - gap(model, current, ends, target)).square()
        self.assertGreater(float(loss.detach()), 0.)
        gradient = torch.autograd.grad(loss, current)[0]
        self.assertGreater(float(gradient[64, 1:4].sum()), 0.)

    def test_losing_one_of_several_facts_is_not_hidden_by_all_outside_anchor(self):
        model = sequence_model(zero=True)
        target = torch.tensor([1, 0, 1])
        parent = torch.full((3, 5), -4., dtype=torch.float64)
        parent[:, 0], parent[0, 1], parent[2, 1] = 0., 5., 5.
        ends = torch.zeros(3, dtype=torch.float64)
        begin = torch.tensor([-6., 0., 6.], dtype=torch.float64)
        current = parent + F.pad(begin[:, None].expand(-1, 3), (1, 1))
        self.assertEqual(model.plan_labels(current, ends).tolist(), [0, 0, 1])
        torch.testing.assert_close(strongest_wrong_path(model, current, ends, target), exhaustive_wrong(model, current, ends, target))
        required = gap(model, parent, ends, target).detach().clamp(max=1000.)
        self.assertGreater(float(F.relu(required - gap(model, current, ends, target)).square().detach()), 0.)

    def test_correct_parent_initial_anchor_has_zero_loss_and_zero_gradient(self):
        model = sequence_model(zero=True)
        for target in (torch.tensor([0]), torch.tensor([1, 4, 0, 1]), torch.zeros(11, dtype=torch.long)):
            with self.subTest(words=len(target)):
                parent = torch.full((len(target), 5), -4., dtype=torch.float64)
                parent.scatter_(1, target[:, None], 5.)
                ends = torch.zeros(len(target), dtype=torch.float64)
                self.assertTrue(torch.equal(model.plan_labels(parent, ends), target))
                required = gap(model, parent, ends, target).detach().clamp(max=1000.)
                live, live_ends = parent.clone().requires_grad_(), ends.clone().requires_grad_()
                loss = F.relu(required - gap(model, live, live_ends, target)).square()
                self.assertEqual(float(loss.detach()), 0.)
                for gradient in torch.autograd.grad(loss, (live, live_ends)):
                    self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_tied_zero_margin_has_finite_gradient_and_zero_initial_penalty(self):
        model = sequence_model(zero=True)
        logits = torch.zeros(4, 5, dtype=torch.float64, requires_grad=True)
        ends = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        target = model.plan_labels(logits, ends)
        wrong = strongest_wrong_path(model, logits, ends, target)
        torch.testing.assert_close(wrong, exhaustive_wrong(model, logits, ends, target))
        wrong_gradients = torch.autograd.grad(wrong, (logits, ends), retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() for g in wrong_gradients))
        self.assertTrue((wrong_gradients[0] >= 0).all())
        torch.testing.assert_close(wrong_gradients[0].sum(-1), torch.ones(4, dtype=torch.float64))
        self.assertEqual(float(wrong_gradients[0][0, 4]), 0.)  # Illegal initial inside tag.
        required = gap(model, logits, ends, target).detach().clamp(max=1000.)
        loss = F.relu(required - gap(model, logits, ends, target)).square()
        self.assertEqual(float(loss.detach()), 0.)
        gradients = torch.autograd.grad(loss, (logits, ends))
        self.assertTrue(all(torch.isfinite(g).all() and not g.any() for g in gradients))

    def test_float32_negative_rounding_gap_keeps_same_parent_loss_and_gradient_zero(self):
        model = sequence_model(zero=True).float()
        generator = torch.Generator().manual_seed(1)
        common = torch.randn(128, generator=generator) * 10
        parent = common[:, None].expand(-1, 5).clone()
        ends = torch.zeros(128)
        target = torch.zeros(128, dtype=torch.long)
        self.assertTrue(torch.equal(model.plan_labels(parent, ends), target))
        # All legal paths have the same real score. Different float32 reduction
        # orders make this parent advantage slightly negative; flooring it at
        # zero would inject a nonzero loss into the identical initial model.
        advantage = gap(model, parent, ends, target).detach()
        self.assertLess(float(advantage), 0.)
        teacher = build_writer_teacher(target)
        required = path_anchor_required_gaps(model, parent, ends, teacher, 1000.)
        self.assertTrue(torch.equal(required, advantage))
        live = parent.clone().requires_grad_()
        live_ends = ends.clone().requires_grad_()
        loss = structured_path_anchor_loss(
            model, None, live, target, live_ends, 1000.,
            teacher=teacher, required_gap=required,
        )
        self.assertEqual(float(loss.detach()), 0.)
        for gradient in torch.autograd.grad(loss, (live, live_ends)):
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_bad_geometry_is_rejected(self):
        model = sequence_model()
        logits, ends, target = torch.zeros(2, 5), torch.zeros(2), torch.zeros(2, dtype=torch.long)
        for x, e, t in ((logits[:0], ends[:0], target[:0]), (logits[:, :4], ends, target),
                        (logits, ends[:1], target), (logits, ends, target[:1]),
                        (logits, ends, target.float())):
            with self.assertRaises(ValueError):
                strongest_wrong_path(model, x, e, t)


if __name__ == "__main__":
    unittest.main()
