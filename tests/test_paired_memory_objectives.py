"""DG-009 measurement/loss safeguards; these are not recall accuracy tests."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from availability_supervision import ReplyTargets
from diagnose_paired_memory_objectives import (pair_metrics, objectives_and_metrics,
    finite_difference, analyze_pair)
from memory_role_separated import RoleOutputs, RoleSeparatedMemory, role_loss
from train_availability_memory import ForwardFeatures


class PairedObjectiveTests(unittest.TestCase):
    def fixtures(self, kind):
        torch.manual_seed(91)
        base = torch.randn(3, 7)
        outputs = [RoleOutputs(base, torch.randn(3, 7, requires_grad=True),
                   torch.randn(3, 7, requires_grad=True), torch.randn(1, 3, requires_grad=True)) for _ in range(2)]
        targets = [ReplyTargets((1,), (2, 3, 0), 1, 2, 'a'),
                   ReplyTargets((1,), (2, 4, 0), 1 if kind == 'value' else 2, 2 if kind == 'value' else 1, 'b')]
        return outputs, targets

    def test_difference_alone_does_not_establish_both_sides_or_full_vocab(self):
        metrics, _ = pair_metrics(torch.tensor([5., 2., 9.]), torch.tensor([4., 2., 9.]), 0, 1)
        self.assertGreater(float(metrics['difference']), 0)
        self.assertLess(float(metrics['right_correct_margin']), 0)
        self.assertLess(float(metrics['left_full_margin']), 0)
        self.assertLess(float(metrics['right_full_margin']), 0)

    def test_same_common_logit_shift_cancels_in_difference_not_center(self):
        a = torch.tensor([2., 3., 0.], requires_grad=True)
        b = torch.tensor([1., 4., 0.], requires_grad=True)
        shift = torch.tensor([7., -3., 0.])
        m, _ = pair_metrics(a, b, 0, 1)
        n, _ = pair_metrics(a + shift, b + shift, 0, 1)
        self.assertEqual(m['difference'], n['difference'])
        self.assertNotEqual(m['center'], n['center'])
        ga, gb = torch.autograd.grad(m['difference'], (a, b))
        torch.testing.assert_close(ga, -gb)

    def test_value_objectives_keep_full_vocab_and_initial_state_supervision(self):
        outputs, targets = self.fixtures('value')
        objectives, metrics, _ = objectives_and_metrics(outputs, targets, 'value', 1)
        old = sum(role_loss(o, t)['weighted_total'] for o, t in zip(outputs, targets)) / 4
        torch.testing.assert_close(objectives['old'], old)
        torch.testing.assert_close(objectives['paired'], objectives['focused'] + F.softplus(1 - metrics['difference']))
        gradients = torch.autograd.grad(objectives['focused'], (outputs[0].content, outputs[0].state_logits))
        self.assertEqual(float(gradients[0][0].norm()), 0)
        self.assertGreater(float(gradients[0][1, 6].abs()), 0)  # Non-pair vocabulary remains supervised.
        self.assertGreater(float(gradients[1].norm()), 0)

    def test_subject_balancing_removes_two_to_one_weight_not_branches(self):
        outputs, targets = self.fixtures('subject')
        objectives, _, _ = objectives_and_metrics(outputs, targets, 'subject')
        losses = [role_loss(o, t) for o, t in zip(outputs, targets)]
        torch.testing.assert_close(objectives['old'], sum(v['weighted_total'] for v in losses) / 3)
        torch.testing.assert_close(objectives['balanced'], sum(v['state'] + v['branch'] for v in losses) / 2)
        gradient = torch.autograd.grad(objectives['balanced'], outputs[1].uncertainty)[0]
        self.assertGreater(float(gradient.norm()), 0)

    def test_fixed_competitors_are_not_reselected_and_invalid_pairs_rejected(self):
        a = torch.tensor([2., 3., 9.]); b = torch.tensor([2., 4., 8.])
        _, competitors = pair_metrics(a, b, 0, 1)
        m, retained = pair_metrics(torch.tensor([2., 13., 9.]), b, 0, 1, competitors)
        self.assertEqual(retained, competitors)
        self.assertEqual(float(m['left_full_margin']), -7)
        for aa, bb, ia, ib in ((a, b, 0, 0), (a, b, 0, 3), (a * float('nan'), b, 0, 1)):
            with self.assertRaises(ValueError): pair_metrics(aa, bb, ia, ib)

    def test_symmetric_numeric_derivative_and_restore_even_on_failure(self):
        model = torch.nn.Linear(2, 1, bias=False).double()
        original = model.weight.detach().clone(); direction = torch.tensor([.6, .8], dtype=torch.float64)
        def evaluate(): return {'square': model.weight.square().sum()}
        expected = {'square': float(2 * (original.flatten() * direction).sum())}
        rows = finite_difference(model, direction, evaluate, expected, (.001, .003))
        self.assertTrue(all(r['metrics']['square']['passed'] for r in rows))
        self.assertTrue(torch.equal(original, model.weight))
        def fail(): raise RuntimeError('intentional')
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            finite_difference(model, direction, fail, expected, (.001,))
        self.assertTrue(torch.equal(original, model.weight))

    def test_small_real_module_analysis_never_updates_or_accumulates_gradients(self):
        torch.manual_seed(72); model = RoleSeparatedMemory(8, 2, 2)
        with torch.no_grad(): model.content.output.weight.normal_(std=.1)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        hidden = torch.randn(3, 8); base = torch.randn(3, 7); head = torch.randn(7, 8)
        _, targets = self.fixtures('value')
        samples = [(ForwardFeatures(hidden, base, torch.randn(5, 16)), t) for t in targets]
        result = analyze_pair(model, samples, head, 4., 'value', 1, True)
        self.assertEqual(set(result['objectives']), {'old', 'focused', 'paired'})
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(m['passed'] for o in result['objectives'].values() for r in o['finite_difference'] for m in r['metrics'].values()))


if __name__ == '__main__': unittest.main()
