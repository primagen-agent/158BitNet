"""Learned bank-context masking, binding, routing and gradient contracts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))

import unittest

import torch
from torch.nn import functional as F

from neural_bank_context import BankContextOperator, FactorizedBankContext


class BankContextChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = BankContextOperator().eval()
        self.q = torch.randn(2, 3, 4096)
        self.m = torch.randn(2, 3, 4, 4096)
        self.qmask = torch.tensor([[True, True, False], [True, True, True]])
        self.mmask = torch.tensor([
            [[True, True, False, False], [True, True, True, False], [False] * 4],
            [[True, True, True, True], [True, False, False, False], [True] * 4],
        ])

    def call(self, mode="full", **extra):
        return self.model(self.q, self.m, self.qmask, self.mmask, mode=mode, **extra)

    def test_unsupported_precision_is_rejected_before_padding_math(self):
        for dtype in (torch.float16, torch.bfloat16):
            model = BankContextOperator().to(dtype=dtype)
            with self.assertRaisesRegex(ValueError, "precision"):
                model(self.q.to(dtype), self.m.to(dtype), self.qmask, self.mmask)

    def test_shapes_and_masks(self):
        for mode in ("full", "independent"):
            result = self.call(mode)
            self.assertEqual(result["edit_logits"].shape, (2, 3))
            self.assertEqual(result["read_logits"].shape, (2, 3))
            self.assertEqual(result["query_context"].shape, (2, 64))
            self.assertEqual(result["record_context"].shape, (2, 3, 64))
            self.assertTrue(torch.equal(result["record_mask"], self.mmask.any(-1)))
            self.assertEqual(float(result["edit_logits"][0, 2].detach()), 0.0)
            self.assertTrue(torch.isfinite(result["edit_logits"]).all())

    def test_padding_and_bank_permutation_do_not_change_valid_logits(self):
        permutation = torch.tensor([2, 0, 1])
        for mode in ("full", "independent"):
            result = self.call(mode)
            q = torch.cat((self.q, torch.full((2, 2, 4096), float("nan"))), 1)
            m = torch.full((2, 5, 7, 4096), float("nan"))
            m[:, :3, :4] = self.m
            qmask = torch.cat((self.qmask, torch.zeros(2, 2, dtype=torch.bool)), 1)
            mmask = torch.zeros(2, 5, 7, dtype=torch.bool)
            mmask[:, :3, :4] = self.mmask
            padded = self.model(q, m, qmask, mmask, mode=mode)
            shuffled = self.model(self.q, self.m[:, permutation], self.qmask,
                                  self.mmask[:, permutation], mode=mode)
            for name in ("edit_logits", "read_logits"):
                torch.testing.assert_close(result[name], padded[name][:, :3], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(result[name][:, permutation], shuffled[name], atol=2e-6, rtol=2e-5)

    def test_control_cannot_see_other_records_but_full_bank_can(self):
        changed = self.m.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 3
        for mode in ("full", "independent"):
            a = self.call(mode)
            b = self.model(self.q, changed, self.qmask, self.mmask, mode=mode)
            delta = float((a["edit_logits"][:, 0] - b["edit_logits"][:, 0]).abs().max().detach())
            if mode == "independent":
                self.assertEqual(delta, 0.0)
            else:
                self.assertGreater(delta, 1e-7)

    def test_full_bank_headers_keep_own_record_identity(self):
        # Equal physical states and token counts must not make the record
        # headers interchangeable: each needs its own body's initial seed.
        q = self.q[:1, :2].clone()
        m = self.m[:1, :2, :2].clone().requires_grad_()
        qm = torch.ones(1, 2, dtype=torch.bool)
        mm = torch.ones(1, 2, 2, dtype=torch.bool)
        out = self.model(q, m, qm, mm)
        self.assertGreater(float((out["edit_logits"][0, 0] - out["edit_logits"][0, 1]).abs().detach()), 1e-7)
        g0 = torch.autograd.grad(out["edit_logits"][0, 0], m, retain_graph=True)[0]
        g1 = torch.autograd.grad(out["edit_logits"][0, 1], m)[0]
        self.assertGreater(float((g0 - g1).abs().sum()), 1e-7)

    def test_full_bank_headers_bind_fields_and_token_positions(self):
        model = self.model.double()
        # The lexical multiset and counts are identical, so a mean of linear
        # token projections cannot bind a token to its role or position.
        a, b = torch.randn(2, 4096, dtype=torch.double)
        q = torch.randn(1, 2, 4096, dtype=torch.double)
        qm = torch.ones(1, 2, dtype=torch.bool)
        mm = torch.ones(1, 2, 2, dtype=torch.bool)
        for field_swap in (False, True):
            m = torch.stack((torch.stack((a, b)), torch.stack((a, b) if field_swap else (b, a))))[None]
            types = (torch.tensor([[[1, 0], [0, 1]]]) if field_swap
                     else torch.zeros(1, 2, 2, dtype=torch.long))
            out = model(q, m, qm, mm, bank_token_types=types)
            delta = (out["record_context"][0, 0] - out["record_context"][0, 1]).abs().max()
            self.assertGreater(float(delta.detach()), 1e-7)
            self.assertGreater(float((out["edit_logits"][0, 0] - out["edit_logits"][0, 1]).abs().detach()), 1e-9)

    def test_gradients_reach_query_and_cross_record_inputs(self):
        for mode in ("full", "independent"):
            self.model.zero_grad(set_to_none=True)
            q = self.q.clone().requires_grad_()
            m = self.m.clone().requires_grad_()
            result = self.model(q, m, self.qmask, self.mmask, mode=mode)
            (result["edit_logits"][:, 0].sum() + result["read_logits"][:, 0].sum()).backward()
            self.assertGreater(float(q.grad.abs().sum()), 0.0)
            self.assertGreater(float(m.grad[:, 0].abs().sum()), 0.0)
            cross = float(m.grad[:, 1:].abs().sum())
            self.assertGreater(cross, 0.0) if mode == "full" else self.assertEqual(cross, 0.0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in self.model.parameters()))

    def test_empty_bank_and_literal_types(self):
        empty = self.model(self.q, self.m[:, :0], self.qmask, self.mmask[:, :0])
        self.assertEqual(empty["edit_logits"].shape, (2, 0))
        types = torch.arange(4)[None, None].expand(2, 3, -1)
        states = torch.tensor([[0, 1, 4], [2, 3, 0]])
        out = self.call(bank_token_types=types, record_states=states)
        self.assertTrue(torch.isfinite(out["read_logits"]).all())
        self.assertFalse(bool(out["record_mask"][1, 0]))
        self.assertEqual(float(out["read_logits"][1, 0].detach()), 0.0)

    def test_deleted_records_are_invisible_before_validation_and_attention(self):
        states = torch.tensor([[0, 2, 4], [0, 2, 0]])
        expected_mask = self.mmask.any(-1)
        expected_mask[:, 1] = False
        for mode in ("full", "independent"):
            original = self.call(mode, record_states=states)
            self.assertTrue(torch.equal(original["record_mask"], expected_mask))
            for replacement in (torch.randn_like(self.m[:, 1]) * 100,
                                torch.full_like(self.m[:, 1], float("nan"))):
                changed = self.m.clone()
                changed[:, 1] = replacement
                out = self.model(self.q, changed, self.qmask, self.mmask,
                                 record_states=states, mode=mode)
                for name in ("edit_logits", "read_logits"):
                    torch.testing.assert_close(original[name], out[name], atol=0, rtol=0)
                    self.assertTrue(torch.equal(out[name][:, 1], torch.zeros(2)))
                self.assertTrue(torch.equal(out["record_context"][:, 1], torch.zeros(2, 64)))
            differentiable = self.m.clone().requires_grad_()
            out = self.model(self.q, differentiable, self.qmask, self.mmask,
                             record_states=states, mode=mode)
            (out["edit_logits"].sum() + out["read_logits"].sum()).backward()
            self.assertEqual(float(differentiable.grad[:, 1].abs().sum()), 0.0)
            self.assertGreater(float(differentiable.grad[:, 0].abs().sum()), 0.0)

    def test_full_attention_directional_derivative(self):
        model = self.model.double()
        q = self.q[:1, :2].double().requires_grad_()
        m = self.m[:1, :2, :2].double().requires_grad_()
        qm = torch.ones(1, 2, dtype=torch.bool)
        mm = torch.ones(1, 2, 2, dtype=torch.bool)
        dq, dm = torch.randn_like(q), torch.randn_like(m)
        result = model(q, m, qm, mm)
        value = result["edit_logits"].sum() + result["read_logits"].sum()
        gq, gm = torch.autograd.grad(value, (q, m))
        analytic = (gq * dq).sum() + (gm * dm).sum()
        epsilon = 1e-5
        with torch.no_grad():
            high = model(q + epsilon * dq, m + epsilon * dm, qm, mm)
            low = model(q - epsilon * dq, m - epsilon * dm, qm, mm)
            numeric = ((high["edit_logits"].sum() + high["read_logits"].sum())
                       - (low["edit_logits"].sum() + low["read_logits"].sum())) / (2 * epsilon)
        torch.testing.assert_close(analytic, numeric, atol=1e-7, rtol=2e-5)




class FactorizedChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)
        self.model = FactorizedBankContext().eval()
        self.q = torch.randn(2, 4, 4096)
        self.s = torch.randn(2, 3, 5, 4096)
        self.p = torch.randn(2, 3, 7, 4096)
        self.qm = torch.tensor([[True, False, True, False], [True] * 4])
        self.sm = torch.ones(2, 3, 5, dtype=torch.bool)
        self.pm = torch.ones(2, 3, 7, dtype=torch.bool)
        self.states = torch.tensor([[0, 2, 1], [0, 2, 3]])

    def call(self, **extra):
        return self.model(self.q, self.s, self.p, self.qm, self.sm, self.pm, self.states, **extra)

    def test_unsupported_probability_precision_is_rejected(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "precision"):
                self.model.probability_mix(torch.tensor([[[100., 100.]]], dtype=dtype),
                                           torch.tensor([[.5, .5]], dtype=dtype))

    def test_probability_mixture_formula_and_bounded_expert_influence(self):
        experts = torch.tensor([[[[2., -4.], [-3., 6.]]]], dtype=torch.double)
        weights = torch.tensor([[.8, .2]], dtype=torch.double)
        actual = self.model.probability_mix(experts, weights)
        expected = torch.logit((experts.sigmoid() * weights[:, None, None]).sum(-1).clamp(1e-6, 1 - 1e-6))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Once a minority expert is saturated, its arbitrary logit scale
        # cannot veto the dominant expert's positive probability.
        a = self.model.probability_mix(torch.tensor([[[20., -1000.]]]), torch.tensor([[.9, .1]]))
        b = self.model.probability_mix(torch.tensor([[[20., -1e12]]]), torch.tensor([[.9, .1]]))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertGreater(float(a), 0.)

    def test_query_summary_uses_valid_order_and_ignores_nan_padding(self):
        q = self.q.clone(); q[~self.qm] = float("nan")
        got = self.model.query_summary(q, self.qm)
        expected = torch.stack([torch.cat((F.normalize(z[mask][-1], dim=0), F.normalize(z[mask].mean(0), dim=0)))
                                for z, mask in zip(q, self.qm)])
        torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_masks_deleted_rows_and_permutation(self):
        original = self.call()
        self.assertTrue(torch.equal(original["record_mask"], self.states != 2))
        s, p = self.s.clone(), self.p.clone(); s[:, 1] = float("nan"); p[:, 1] = float("nan")
        changed = self.model(self.q, s, p, self.qm, self.sm, self.pm, self.states)
        for name in ("read_logits", "edit_logits"):
            torch.testing.assert_close(original[name], changed[name], atol=0, rtol=0)
            self.assertTrue(torch.equal(changed[name][:, 1], torch.zeros(2)))
        order = torch.tensor([2, 0, 1])
        shuffled = self.model(self.q, self.s[:, order], self.p[:, order], self.qm,
                              self.sm[:, order], self.pm[:, order], self.states[:, order])
        torch.testing.assert_close(original["edit_logits"][:, order], shuffled["edit_logits"], atol=2e-6, rtol=2e-5)
        bad = self.pm.clone(); bad[:, 2] = False
        with self.assertRaisesRegex(ValueError, "physical"):
            self.model(self.q, self.s, self.p, self.qm, self.sm, bad, self.states)

    def test_metadata_channel_cannot_change_source_outputs_or_router(self):
        original = self.call()
        changed = self.model(self.q, self.s, torch.randn_like(self.p), self.qm, self.sm, self.pm, self.states)
        torch.testing.assert_close(original["source_logits"], changed["source_logits"], atol=0, rtol=0)
        torch.testing.assert_close(original["weights"], changed["weights"], atol=0, rtol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_parity_and_deleted_content_isolation(self):
        cpu = self.call()
        accelerator = FactorizedBankContext().to("mps").eval()
        accelerator.load_state_dict(self.model.state_dict())
        inputs = [x.to("mps") for x in (self.q, self.s, self.p, self.qm, self.sm, self.pm, self.states)]
        actual = accelerator(*inputs)
        for key in ("read_logits", "edit_logits", "weights"):
            torch.testing.assert_close(cpu[key], actual[key].cpu(), atol=2e-5, rtol=3e-5)
        changed = list(inputs)
        changed[1] = changed[1].clone(); changed[2] = changed[2].clone()
        changed[1][:, 1] = float("nan"); changed[2][:, 1] = float("nan")
        erased = accelerator(*changed)
        torch.testing.assert_close(actual["edit_logits"], erased["edit_logits"], atol=0, rtol=0)

    def test_parameter_state_roundtrip_preserves_both_heads(self):
        restored = FactorizedBankContext().eval()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        original = self.call()
        copied = restored(self.q, self.s, self.p, self.qm, self.sm, self.pm, self.states)
        for key in ("read_logits", "edit_logits", "weights"):
            torch.testing.assert_close(original[key], copied[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
