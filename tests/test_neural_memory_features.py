"""Feature diagnostic mechanics, not semantic capability or platform parity."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from diagnose_neural_memory_features import compare, read_native, c_q8_activation_reference
from torch_backbone import rope_tables
from test_memory_fusion import tiny_backbone


class FeatureTests(unittest.TestCase):
    def test_comparison_does_not_accept_high_cosine_with_wrong_scale(self):
        a = np.array([[1., 2., 3.]], dtype=np.float32)
        result = compare(a, a * 1.04, 1e-5, 1e-4)
        self.assertGreater(result["minimum_row_cosine"], .99999)
        self.assertFalse(result["allclose"])
        self.assertEqual(result["outside_tolerance"], 3)

    def test_nonfinite_and_geometry_errors_are_not_scores(self):
        a = np.ones((2, 4), dtype=np.float32)
        with self.assertRaises(ValueError): compare(a, a[:1], 1e-5, 1e-4)
        a[0, 0] = np.nan
        with self.assertRaises(ValueError): compare(a, a, 1e-5, 1e-4)

    def test_binary_capture_shape_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.bin"
            raw = b"BNFP0001" + struct.pack("<III", 1, 4, 2) + struct.pack("<Iii", 2, 1, 9)
            raw += np.arange(32, dtype="<f4").tobytes()
            path.write_bytes(raw)
            row = read_native(path, 1, 4, 2)[0]
            self.assertEqual(row[0].tolist(), [1, 9])
            self.assertEqual(row[3].shape, (2, 8))
            path.write_bytes(raw + b"x")
            with self.assertRaisesRegex(ValueError, "trailing"): read_native(path, 1, 4, 2)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError): read_native(path, 1, 4, 2)

    def test_q8_control_uses_per_token_scale_and_nearest_even(self):
        values = torch.tensor([[0., .5, 1.5, 127.], [0., 1., 3., 254.], [0., 0., 0., 0.]])
        self.assertTrue(torch.equal(c_q8_activation_reference(values), torch.tensor([[0., 0., 2., 127.], [0., 0., 4., 254.], [0., 0., 0., 0.]])))

    def test_fp32_forward_does_not_round_rope_to_bfloat16(self):
        model = tiny_backbone()
        with mock.patch("torch_backbone.rope_tables", wraps=rope_tables) as call:
            model(torch.tensor([1, 3, 4]), return_hidden=True)
        self.assertEqual(call.call_args.args[-1], torch.float32)
        model.dtype = torch.bfloat16
        with mock.patch("torch_backbone.rope_tables", wraps=rope_tables) as call:
            actual = model(torch.tensor([1, 3, 4]), return_hidden=True)
        self.assertEqual(call.call_args.args[-1], torch.bfloat16)
        with mock.patch("torch_backbone.rope_tables", side_effect=lambda cfg, positions, device, dtype: rope_tables(cfg, positions, device, torch.bfloat16)):
            legacy = model(torch.tensor([1, 3, 4]), return_hidden=True)
        self.assertTrue(torch.equal(actual, legacy))

    @unittest.skipUnless((ROOT / "build/memory_feature_probe").exists(), "build the diagnostic C probe first")
    def test_probe_fails_closed_and_never_overwrites_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "invalid.bin", Path(directory) / "output.bin"
            source.write_bytes(b"xx")
            command = [str(ROOT / "build/memory_feature_probe"), "missing.gguf", str(source), str(output), "24"]
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
            output.write_bytes(b"preserved")
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
            self.assertEqual(output.read_bytes(), b"preserved")


if __name__ == "__main__": unittest.main()
