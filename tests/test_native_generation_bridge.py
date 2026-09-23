"""Mechanical checks only; surrogate gradients require native direction tests."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from native_generation_bridge import native_value_with_surrogate_gradient, suffix_surrogate
from memory_fusion import GatedMemoryFusion
from test_memory_fusion import tiny_backbone


class GenerationBridgeTests(unittest.TestCase):
    def test_native_values_are_bit_exact_even_with_signed_zero(self):
        native = torch.tensor([1e20, -0., 1e-38], dtype=torch.float32)
        parameter = torch.tensor([2., 3., 4.], requires_grad=True)
        actual = native_value_with_surrogate_gradient(native, parameter.square())
        self.assertEqual(actual.detach().numpy().tobytes(), native.numpy().tobytes())
        actual.sum().backward()
        self.assertTrue(torch.equal(parameter.grad, 2 * parameter.detach()))

    def test_surrogate_is_not_misrepresented_as_true_quantized_derivative(self):
        parameter = torch.tensor(.24, requires_grad=True)
        actual = native_value_with_surrogate_gradient(parameter.detach().round(), parameter * 2)
        actual.backward()
        central_difference = (torch.tensor(.241).round() - torch.tensor(.239).round()) / .002
        self.assertEqual(float(parameter.grad), 2.)
        self.assertEqual(float(central_difference), 0.)

    def test_backward_contract_rejects_wrong_shape_dtype_nonfinite_or_native_grad(self):
        good = torch.ones(3)
        for native, proxy in ((good[:2], good), (good, good.double()), (good, good * float("nan")),
                              (good.clone().requires_grad_(), good)):
            with self.assertRaises(ValueError): native_value_with_surrogate_gradient(native, proxy)

    def test_both_suffix_variants_reach_fusion_without_changing_backbone(self):
        torch.manual_seed(7)
        backbone = tiny_backbone()
        fusion = GatedMemoryFusion(8, 3, 2)
        with torch.no_grad(): fusion.layer_gain[2] = .1
        memory, hidden = torch.randn(4, 16), torch.randn(1, 8)
        original = backbone.layers[-1]["down"].clone()
        for variant in ("dense_suffix", "q8_ste_suffix"):
            fusion.zero_grad(set_to_none=True)
            state = fusion(2, hidden, fusion.prepare(memory))
            proxy = suffix_surrogate(backbone, state, variant)
            bridged = native_value_with_surrogate_gradient(torch.zeros_like(proxy), proxy)
            torch.nn.functional.cross_entropy(bridged, torch.tensor([4])).backward()
            self.assertTrue(torch.isfinite(fusion.encoder.weight.grad).all())
            self.assertGreater(float(fusion.encoder.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(original, backbone.layers[-1]["down"]))
        self.assertIsNone(backbone.layers[-1]["down"].grad)

    def test_unregistered_variant_or_trainable_backbone_is_rejected(self):
        backbone = tiny_backbone()
        with self.assertRaises(ValueError): suffix_surrogate(backbone, torch.ones(1, 8), "implicit-ste")
        backbone.out_proj.requires_grad_()
        with self.assertRaises(ValueError): suffix_surrogate(backbone, torch.ones(1, 8), "dense_suffix")

    @unittest.skipUnless((ROOT / "build/memory_gradient_probe").exists(), "build native probe first")
    def test_native_probe_rejects_invalid_input_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "invalid", Path(directory) / "result"
            source.write_bytes(b"invalid")
            command = [str(ROOT / "build/memory_gradient_probe"), "missing.gguf", str(source), str(output)]
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
            output.write_bytes(b"preserve")
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
            self.assertEqual(output.read_bytes(), b"preserve")


if __name__ == "__main__": unittest.main()
