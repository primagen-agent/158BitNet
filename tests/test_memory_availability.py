"""Availability mechanics only; hand-set routing is not semantic accuracy."""
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from continuous_memory import continuous_logits
from memory_availability import AvailabilityMemoryFusion, STATES


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3180927)
        self.module = AvailabilityMemoryFusion(8, 3, 2)
        self.hidden = torch.randn(2, 8)
        self.features = torch.randn(4, 16)

    def test_initial_empty_and_nonempty_residuals_are_zero(self):
        for prepared in (None, self.module.prepare(self.features)):
            result = self.module.inspect(2, self.hidden, prepared)
            self.assertEqual(int(result.residual.count_nonzero()), 0)
            base = torch.tensor([[-0., 1.], [2., -0.]])
            logits, _ = continuous_logits(base, result.residual, torch.randn(2, 8), 1.)
            self.assertEqual(logits.detach().numpy().tobytes(), base.numpy().tobytes())

    def test_empty_content_zero_but_generation_gradient_reaches_uncertainty(self):
        result = self.module.inspect(2, self.hidden, None)
        logits, _ = continuous_logits(torch.randn(2, 24), result.residual, torch.randn(24, 8), 1.)
        F.cross_entropy(logits, torch.tensor([3, 4])).backward()
        self.assertGreater(float(self.module.uncertainty_output.weight.grad.norm()), 0)
        self.assertEqual(int(result.content_residual.count_nonzero()), 0)
        # Zero output initialization intentionally delays upstream generation gradients.
        self.assertEqual(int(self.module.state_encoder.weight.grad.count_nonzero()), 0)

    def test_auxiliary_state_loss_can_bootstrap_upstream_on_empty(self):
        result = self.module.inspect(2, self.hidden, None)
        F.cross_entropy(result.state_logits, torch.tensor([STATES.index("insufficient"), STATES.index("no_memory_needed")])).backward()
        self.assertGreater(float(self.module.state_head.weight.grad.norm()), 0)
        self.assertGreater(float(self.module.state_encoder.weight.grad.norm()), 0)
        self.assertTrue(torch.equal(result.state_probabilities[:, 1], torch.zeros(2)))
        self.assertTrue(torch.allclose(result.state_probabilities.sum(-1), torch.ones(2)))

    def test_empty_enabled_can_change_logits_but_disabled_cannot(self):
        with torch.no_grad(): self.module.uncertainty_output.weight.normal_(std=.1)
        enabled = self.module.inspect(2, self.hidden, None)
        disabled = self.module.inspect(2, self.hidden, {"unread_payload": "must_not_be_used"}, enabled=False)
        self.assertGreater(float(enabled.residual.detach().norm()), 0)
        self.assertEqual(int(enabled.content_residual.count_nonzero()), 0)
        self.assertEqual(int(disabled.residual.count_nonzero()), 0)
        self.assertIsNone(disabled.state_logits)

    def test_route_controls_do_not_force_all_queries_to_uncertainty(self):
        # Synthetic routing fixture; not a trained classification result.
        with torch.no_grad():
            self.module.uncertainty_output.weight.normal_(std=.1)
            self.module.state_head.weight.zero_()
            self.module.state_head.bias.copy_(torch.tensor([20., 0., -20.]))
        normal = self.module.inspect(2, self.hidden, None)
        with torch.no_grad(): self.module.state_head.bias.copy_(torch.tensor([-20., 0., 20.]))
        missing = self.module.inspect(2, self.hidden, None)
        self.assertLess(float(normal.residual.detach().norm()), 1e-12)
        self.assertGreater(float(missing.residual.detach().norm()), .01)

    def test_nonempty_content_and_state_both_reach_neural_features(self):
        with torch.no_grad():
            self.module.content.layer_gain[2] = .1
            self.module.uncertainty_output.weight.normal_(std=.01)
        result = self.module.inspect(2, self.hidden, self.module.prepare(self.features))
        self.assertTrue(torch.allclose(result.residual, result.content_residual + result.uncertainty_residual))
        result.residual.square().sum().backward()
        self.assertGreater(float(self.module.content.encoder.weight.grad.norm()), 0)
        self.assertGreater(float(self.module.state_encoder.weight.grad.norm()), 0)

    def test_post_forward_labels_do_not_enter_module(self):
        before = self.module.inspect(2, self.hidden, None).state_logits.detach().clone()
        labels = {"target": "insufficient"}; labels["target"] = "supported"
        after = self.module.inspect(2, self.hidden, None).state_logits.detach()
        self.assertTrue(torch.equal(before, after))
        with self.assertRaises(TypeError): self.module.inspect(2, self.hidden, None, labels=labels)

    def test_serialization_preserves_both_branches(self):
        with torch.no_grad(): self.module.uncertainty_output.weight.normal_(std=.1)
        buffer = io.BytesIO(); torch.save(self.module.state_dict(), buffer); buffer.seek(0)
        restored = AvailabilityMemoryFusion(8, 3, 2)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        for prepared_features in (None, self.features):
            one = self.module.inspect(2, self.hidden, None if prepared_features is None else self.module.prepare(prepared_features))
            two = restored.inspect(2, self.hidden, None if prepared_features is None else restored.prepare(prepared_features))
            self.assertTrue(torch.equal(one.residual, two.residual))

    def test_invalid_runtime_geometry_and_domain_fail_closed(self):
        for hidden in (self.hidden.half(), self.hidden + float("nan"), self.hidden[:, :3], self.hidden[0]):
            with self.assertRaises(ValueError): self.module.inspect(2, hidden, None)
        for prepared in ({"gold": "leak"}, (torch.zeros(2, 1, 4), torch.zeros(2, 1, 4)), (torch.zeros(1),)):
            with self.assertRaises(ValueError): self.module.inspect(2, self.hidden, prepared)
        with self.assertRaises(ValueError): self.module.inspect(2, self.hidden, None, enabled=1)


if __name__ == "__main__": unittest.main()
