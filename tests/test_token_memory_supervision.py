"""Zero-step supervision/composition safeguards, not semantic accuracy."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from availability_supervision import ReplyTargets
from memory_role_separated import RoleSeparatedMemory
from memory_token_read import TokenReadInput, TokenReadPrototype
from token_memory_composition import ComposedTokenMemory, apply_factor_support, mix_content_copy
from token_memory_supervision import TokenSupervision, make_position_targets, supervised_token_loss


class TokenSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1011)
        self.m = ComposedTokenMemory(TokenReadPrototype(8, 13, 'test'), RoleSeparatedMemory(8, 2, 2))
        self.inputs = TokenReadInput(torch.randn(4, 16), torch.randn(4, 16), torch.tensor([3, 3, 5, 6]),
                                     torch.tensor([True, True, True, False]), 'test')
        self.h = torch.randn(3, 8); self.base = torch.randn(3, 13); self.head = torch.randn(13, 8)

    def targets(self, state=1, positions=((1, 2), (0,), None)):
        return TokenSupervision(ReplyTargets((1,), (3, 4, 2), state, 1., 'reply'),
                               (True, False) if state != 0 else (None, None), positions)

    def test_uncopyable_tokens_remain_and_repeated_positions_marginalize(self):
        targets = make_position_targets((3, 4, 2), 1, (0, 1), (3, 3, 5, 6), (True, True, True, False), (0, 1, 2))
        self.assertEqual(targets, ((1, 2), (0,), None))
        with self.assertRaises(ValueError): TokenSupervision(self.targets().reply, (True, True), ((1,),))

    def test_factor_filter_preserves_normal_mass_and_transfers_rejected_support(self):
        raw = torch.tensor([.2, 2., -.3]); factors = torch.tensor([0., 0.])
        original = raw.softmax(-1); final = apply_factor_support(raw, factors).exp()
        torch.testing.assert_close(final[0], original[0])
        torch.testing.assert_close(final[1], original[1] * .25)
        torch.testing.assert_close(final[2], original[2] + original[1] * .75)
        self.assertAlmostEqual(float(final.sum()), 1., places=6)

    def test_full_loss_keeps_gradient_to_content_copy_and_factor_heads(self):
        state = self.m.prefill(self.inputs)
        out = self.m.branches(self.h, self.base, self.head, 4., state)
        loss = supervised_token_loss(out, self.targets())
        self.assertEqual(loss['reply_tokens'], 3); self.assertEqual(loss['position_targets'], 2)
        loss['weighted_total'].backward()
        for name in ('roles.content.output.weight', 'reader.position_query.weight', 'reader.copy_gate.weight',
                     'factor_heads.0.weight', 'factor_heads.1.weight'):
            g = dict(self.m.named_parameters())[name].grad
            self.assertIsNotNone(g); self.assertTrue(torch.isfinite(g).all()); self.assertGreater(float(g.norm()), 0)
        self.assertIsNone(self.head.grad); self.assertIsNone(self.base.grad)

    def test_factor_labels_applied_only_after_forward(self):
        out = self.m.branches(self.h, self.base, self.head, 4., self.m.prefill(self.inputs))
        before = out.supported.detach().clone()
        a = supervised_token_loss(out, self.targets())
        b = supervised_token_loss(out, replace(self.targets(), factors=(False, True)))
        torch.testing.assert_close(a['generation'], b['generation'])
        self.assertTrue(torch.equal(before, out.supported))

    def test_non_support_requires_copy_off_without_gold_positions(self):
        out = self.m.branches(self.h, self.base, self.head, 4., self.m.prefill(self.inputs))
        for index in (0, 2):
            loss = supervised_token_loss(out, self.targets(index, (None,) * 3))
            self.assertGreater(float(loss['copy_off'].detach()), 0)
            self.assertEqual(loss['position_targets'], 0)
            with self.assertRaises(ValueError): self.targets(index)

    def test_predicted_routes_no_gold_argument_and_uncertainty_never_calls_content(self):
        for route in (0, 1, 2):
            with torch.no_grad():
                self.m.reader.state_head.weight.zero_(); self.m.reader.state_head.bias.fill_(-30)
                self.m.reader.state_head.bias[route] = 30
                for head in self.m.factor_heads: head.weight.zero_(); head.bias.fill_(30)
            state = self.m.prefill(self.inputs)
            self.assertEqual(int(state.state_logits.argmax()), route)
            self.assertIs(self.m.read(self.h, self.base, self.head, 4., state, enabled=False), self.base)
            if route in (0, 2):
                with patch.object(self.m.reader, 'proposal', side_effect=AssertionError('copy on wrong route')):
                    output = self.m.read(self.h, self.base, self.head, 4., state)
                    if route == 0: self.assertIs(output, self.base)
            else:
                self.assertTrue(torch.allclose(self.m.read(self.h, self.base, self.head, 4., state).exp().sum(-1), torch.ones(3)))

    def test_specialist_mutation_invalidates_composed_state(self):
        state = self.m.prefill(self.inputs)
        with torch.no_grad(): self.m.roles.uncertainty_output.weight.add_(.1)
        with self.assertRaises(ValueError): self.m.read(self.h, self.base, self.head, 4., state)

    def test_empty_pointer_keeps_continuous_branch_not_detached(self):
        content = self.base.clone().requires_grad_()
        positions = torch.ones(3, 1)
        output = mix_content_copy(content, positions, torch.zeros(3, 1), torch.empty(0, dtype=torch.long))
        self.assertIs(output, content)
        output.sum().backward(); self.assertIsNotNone(content.grad)

    def test_copy_mass_requires_real_positions(self):
        with self.assertRaises(ValueError):
            mix_content_copy(self.base, torch.ones(3, 1), torch.ones(3, 1) * .1, torch.empty(0, dtype=torch.long))

    def test_null_target_cannot_be_mixed_with_positive_positions(self):
        with self.assertRaises(ValueError): self.targets(1, ((0, 1), None, None))


if __name__ == '__main__': unittest.main()
