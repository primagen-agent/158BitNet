"""Continuous branch mechanics, not memory learning or generalization."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from continuous_memory import continuous_logits
from memory_fusion import GatedMemoryFusion


class ContinuousMemoryTests(unittest.TestCase):
    def test_zero_preserves_signed_zero_and_gradient(self):
        base = torch.tensor([[-0., 1., 2.]], dtype=torch.float32)
        delta = torch.zeros(1, 2, requires_grad=True)
        weights = torch.tensor([[1., 2.], [2., 3.], [3., 4.]])
        result, _ = continuous_logits(base, delta, weights, 1.)
        self.assertEqual(result.detach().numpy().tobytes(), base.numpy().tobytes())
        result.sum().backward()
        self.assertTrue(torch.equal(delta.grad, weights.sum(0)[None, :]))

    def test_residual_matches_original_layer_addition(self):
        torch.manual_seed(8)
        module = GatedMemoryFusion(8, 3, 2)
        with torch.no_grad(): module.layer_gain[2] = .1
        hidden, memory = torch.randn(2, 8), torch.randn(4, 16)
        prepared = module.prepare(memory)
        self.assertTrue(torch.equal(module(2, hidden, prepared), hidden + module.residual(2, hidden, prepared)))
        self.assertTrue(torch.equal(module.residual(2, hidden, None), torch.zeros_like(hidden)))
        with self.assertRaises(ValueError): module.residual(3, hidden, None)

    def test_zero_gain_can_open_and_projection_stays_frozen(self):
        torch.manual_seed(18)
        module = GatedMemoryFusion(8, 3, 2)
        hidden, memory = torch.randn(1, 8), torch.randn(4, 16)
        weight = torch.randn(24, 8)
        result, _ = continuous_logits(torch.zeros(1, 24), module.residual(2, hidden, module.prepare(memory)), weight, 1.)
        torch.nn.functional.cross_entropy(result, torch.tensor([5])).backward()
        self.assertGreater(float(module.layer_gain.grad[2].abs()), 0)
        self.assertIsNone(weight.grad)

    def test_empty_content_bypass_has_no_trainable_abstention_path(self):
        # This is a diagnosed limitation, not the product's missing-evidence
        # behavior. A separate learned availability/task branch is still needed.
        module = GatedMemoryFusion(8, 3, 2)
        hidden = torch.randn(1, 8)
        prepared = module.prepare(torch.empty(0, 16))
        self.assertIsNone(prepared)
        delta = module.residual(2, hidden, prepared)
        result, _ = continuous_logits(torch.randn(1, 24), delta, torch.randn(24, 8), 1.)
        self.assertFalse(result.requires_grad)
        self.assertEqual(int(torch.count_nonzero(delta)), 0)

    def test_ordinary_gradient_matches_smooth_finite_difference(self):
        theta = torch.tensor(.2, requires_grad=True)
        base = torch.tensor([[1., .5, 0.]])
        weights = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])
        def objective(value):
            logits, _ = continuous_logits(base, value * torch.tensor([[1., -.2]]), weights, 1.)
            return torch.nn.functional.cross_entropy(logits.double(), torch.tensor([2]))
        gradient = torch.autograd.grad(objective(theta), theta)[0]
        difference = (objective(theta.detach() + .01) - objective(theta.detach() - .01)) / .02
        self.assertTrue(torch.allclose(gradient.double(), difference, atol=2e-4, rtol=1e-3))

    def test_continuous_increment_can_change_ranking_not_only_temperature(self):
        base = torch.tensor([[1., .5, 0.]])
        weights = torch.tensor([[0., 0.], [0., 0.], [1., 0.]])
        changed, _ = continuous_logits(base, torch.tensor([[2., 0.]]), weights, 1.)
        self.assertEqual(int(base.argmax()), 0)
        self.assertEqual(int(changed.argmax()), 2)

    def test_rejects_trainable_head_mixed_precision_and_nonfinite_values(self):
        base, delta, weights = torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(3, 2)
        with self.assertRaises(ValueError): continuous_logits(base, delta, weights.clone().requires_grad_(), 1.)
        with self.assertRaises(ValueError): continuous_logits(base, delta.half(), weights, 1.)
        with self.assertRaises(ValueError): continuous_logits(base, delta + float("nan"), weights, 1.)

    @unittest.skipUnless((ROOT / "build/memory_continuous_probe").exists(), "build continuous probe first")
    def test_probe_rejects_corruption_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input", Path(directory) / "output"
            source.write_bytes(b"invalid")
            command = [str(ROOT / "build/memory_continuous_probe"), "missing.gguf", str(source), str(output)]
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
            output.write_bytes(b"preserved")
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
            self.assertEqual(output.read_bytes(), b"preserved")


if __name__ == "__main__": unittest.main()
