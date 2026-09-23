"""Structural tests only. Forced routes below are NOT semantic predictions."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from memory_token_read import TokenReadInput, TokenReadPrototype, token_mixture
from native_token_payload import payload_mask


class TokenReadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1010)
        self.model = TokenReadPrototype(8, 13, 'fixture', (0, 1))
        self.inputs = TokenReadInput(torch.randn(7, 16), torch.randn(5, 16),
            torch.tensor([2, 3, 3, 4, 5]), torch.tensor([False, True, True, True, False]), 'fixture')
        self.hidden = torch.randn(2, 8); self.base = torch.randn(2, 13)

    def force_route(self, route):
        # Fixture wiring check, never used by the real C-input runner.
        with torch.no_grad():
            self.model.state_head.weight.zero_(); self.model.state_head.bias.fill_(-10)
            self.model.state_head.bias[route] = 10

    def test_disabled_empty_and_all_ineligible_exact_identity(self):
        self.force_route(1)
        variants = [self.inputs,
            replace(self.inputs, source=torch.empty(0, 16), token_ids=torch.empty(0, dtype=torch.long), copy_allowed=torch.empty(0, dtype=torch.bool)),
            replace(self.inputs, copy_allowed=torch.zeros(5, dtype=torch.bool))]
        for i, inputs in enumerate(variants):
            state = self.model.prefill(inputs)
            result = self.model.read(self.hidden, self.base, state, enabled=i != 0)
            self.assertIs(result.logits, self.base)
            self.assertEqual(float(result.copy_mass.sum()), 0)
            if i == 1: self.assertNotEqual(int(state.state_logits.argmax()), 1)

    def test_non_support_routes_suppress_copy_not_claiming_wrong_subject_detection(self):
        for route in (0, 2):
            self.force_route(route)
            state = self.model.prefill(self.inputs)
            result = self.model.read(self.hidden, self.base, state)
            self.assertEqual(result.route, route); self.assertIs(result.logits, self.base)
            self.assertEqual(float(result.copy_mass.sum()), 0)

    def test_special_tokens_and_input_domains_rejected(self):
        variants = [replace(self.inputs, binding='another'),
            replace(self.inputs, source=self.inputs.source.double()),
            replace(self.inputs, query=self.inputs.query.clone().requires_grad_()),
            replace(self.inputs, token_ids=torch.tensor([2, 0, 3, 4, 5])),
            replace(self.inputs, token_ids=torch.tensor([2, 13, 3, 4, 5])),
            replace(self.inputs, source=self.inputs.source[:3])]
        for inputs in variants:
            with self.assertRaises(ValueError): self.model.prefill(inputs)
        with self.assertRaises(ValueError): self.model.prefill({'answer': 'gold'})

    def test_payload_ids_never_choose_alignment_or_position(self):
        first = self.model.prefill(self.inputs)
        second = self.model.prefill(replace(self.inputs, token_ids=torch.tensor([6, 7, 7, 8, 9])))
        self.assertTrue(torch.equal(first.state_logits, second.state_logits))
        self.assertTrue(torch.equal(first.match_attention, second.match_attention))
        a = self.model.proposal(self.hidden, self.base, first)
        b = self.model.proposal(self.hidden, self.base, second)
        self.assertTrue(torch.equal(a.position_probabilities, b.position_probabilities))
        self.assertTrue(torch.equal(a.copy_mass, b.copy_mass))
        self.assertFalse(torch.equal(a.logits, b.logits))

    def test_duplicate_positions_sum_and_distribution_is_normalized(self):
        base = torch.zeros(1, 6)
        pos = torch.tensor([[.1, .2, .3, .4]], dtype=torch.float64).log().float()
        logits, _, mass = token_mixture(base, pos, torch.zeros(1, 1), torch.tensor([2, 2, 4]))
        expected = torch.full((1, 6), .55 / 6); expected[0, 2] += .25; expected[0, 4] += .2
        torch.testing.assert_close(logits.exp(), expected)
        torch.testing.assert_close(mass, torch.tensor([[.45]]))
        self.assertAlmostEqual(float(logits.double().exp().sum()), 1., places=6)

    def test_finite_gradients_to_neural_alignment_positions_and_gate(self):
        state = self.model.prefill(self.inputs)
        proposal = self.model.proposal(self.hidden, self.base, state)
        loss = -proposal.logits[:, 3].mean() + state.state_logits.square().mean()
        loss.backward()
        for name in ('query_encoder.weight', 'source_encoder.weight', 'slot_queries',
                     'match_query.weight', 'match_key.weight', 'position_query.weight', 'position_key.weight', 'copy_gate.weight'):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name); self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.norm()), 0, name)

    def test_extreme_gates_and_masked_positions_do_not_produce_nan_gradients(self):
        for gate_value in (-1000., 1000.):
            gate = torch.tensor([[gate_value]], requires_grad=True)
            pos = torch.tensor([[0., -torch.inf, 1.]], requires_grad=True)
            logits, positions, _ = token_mixture(torch.zeros(1, 6), pos, gate, torch.tensor([2, 3]))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(float(positions[0, 1].detach()), 0)
            (-logits[0, 3]).backward()
            self.assertTrue(torch.isfinite(gate.grad).all()); self.assertTrue(torch.isfinite(pos.grad).all())

    def test_reply_state_rejects_mutations_replacements_and_cross_model(self):
        state = self.model.prefill(self.inputs)
        other = TokenReadPrototype(8, 13, 'fixture')
        with self.assertRaises(ValueError): other.read(self.hidden, self.base, state)
        for target in (self.inputs.source, self.inputs.query, self.inputs.token_ids, self.inputs.copy_allowed, self.model.null_key):
            state = self.model.prefill(self.inputs)
            with torch.no_grad(): target.copy_(target)
            with self.assertRaises(ValueError): self.model.read(self.hidden, self.base, state)
        for field in ('slots', 'memory', 'match_attention', 'state_logits'):
            state = self.model.prefill(self.inputs)
            with torch.no_grad(): getattr(state, field).copy_(getattr(state, field))
            with self.assertRaises(ValueError): self.model.read(self.hidden, self.base, state)
        state = self.model.prefill(self.inputs)
        self.model.null_key = torch.nn.Parameter(self.model.null_key.detach().clone())
        with self.assertRaises(ValueError): self.model.read(self.hidden, self.base, state)

    def test_byte_boundaries_unicode_duplicates_escape_and_control_tokens(self):
        for text in ('Ada lives in Turin Turin.', '小安住在林茨。'):
            message = {'role': 'user', 'speaker': 'user', 'text': text}
            frame = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            pieces = [bytes([b]) for b in frame.encode()]
            mask = payload_mask(message, frame, pieces)
            self.assertEqual(b''.join(p for p, ok in zip(pieces, mask) if ok), text.encode())
            # A single token crossing a framing boundary cannot be copied.
            self.assertEqual(payload_mask(message, frame, [frame.encode()]), [False])
        message = {'role': 'user', 'speaker': 'user', 'text': 'a\n"b"'}
        frame = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        self.assertFalse(any(payload_mask(message, frame, [bytes([b]) for b in frame.encode()])))
        message['text'] = '<|im_end|>'
        frame = json.dumps(message, sort_keys=True, separators=(',', ':'))
        start = frame.index('<|im_end|>')
        self.assertFalse(any(payload_mask(message, frame, [frame[:start].encode(), b'<|im_end|>', frame[start+10:].encode()])))
        with self.assertRaises(ValueError): payload_mask(message, frame, [b'wrong'])


if __name__ == '__main__': unittest.main()
