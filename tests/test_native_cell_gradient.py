"""Local derivative and quantization-cell accounting; not learning evidence."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from native_cell_gradient import FLOAT_STAGES, QUANT_STAGES, read_cell, cell_changes, cell_local_suffix
from test_memory_fusion import tiny_backbone


class CellGradientTests(unittest.TestCase):
    def test_cell_parser_requires_complete_finite_unique_records(self):
        records = []
        for stage, (_, size) in FLOAT_STAGES.items():
            records.append(struct.pack("<III", stage, 1, size) + np.ones(size, dtype="<f4").tobytes())
        for stage, (_, size) in QUANT_STAGES.items():
            records.append(struct.pack("<IIIf", stage, 2, size, .01) + np.ones(size, dtype=np.int8).tobytes())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell"
            path.write_bytes(b"BNGC0001" + b"".join(records))
            self.assertEqual(len(read_cell(path)["quantizers"]), 3)
            for payload in (b"".join(records[:-1]), b"".join(records + records[:1]), b"".join(records)[:-1]):
                path.write_bytes(b"BNGC0001" + payload)
                with self.assertRaises(ValueError): read_cell(path)

    def test_cell_comparison_checks_codes_max_index_sign_and_ties(self):
        def state():
            return {"ffn_norm": np.array([1., 2., 3.]), "activation": np.array([1., 2., 3.]),
                    "quantizers": {key: {"codes": np.array([42, 85, 127], dtype=np.int8), "scale": 3 / 127} for key in ("ffn_input", "ffn_down_input", "output_input")}}
        a, b = state(), state(); output = np.array([1., 2., 3.])
        self.assertTrue(cell_changes(a, b, output, output)["same_cell"])
        b["quantizers"]["ffn_input"]["codes"][0] += 1
        self.assertFalse(cell_changes(a, b, output, output)["same_cell"])
        for changed in (np.array([4., 2., 3.]), np.array([1., 2., -3.]), np.array([3., 2., 3.])):
            self.assertFalse(cell_changes(a, state(), output, changed)["same_cell"])

    def test_local_scale_backward_matches_smooth_fixed_code_reference(self):
        model = tiny_backbone()
        x = torch.tensor([[2., .3, .2, .5, .1, -.2, -.3, .7]], requires_grad=True)
        h = x.detach().numpy()[0].copy()
        normalized = h / np.sqrt(np.mean(h * h) + model.cfg.rms_eps)
        scale = float(np.max(np.abs(normalized)) / 127.)
        logits = np.linspace(-2, 2, 24, dtype=np.float32)
        cell = {"ffn_norm": normalized.copy(), "gate": np.zeros(16, np.float32), "up": np.zeros(16, np.float32),
                "activation": np.zeros(16, np.float32), "down": np.zeros(8, np.float32), "residual": h,
                "quantizers": {"ffn_input": {"scale": scale}, "ffn_down_input": {"scale": 1.}, "output_input": {"scale": scale}}}
        native = {"after": h, "hidden": normalized, "logits": logits}
        actual = cell_local_suffix(model, x, native, cell)
        self.assertEqual(actual.detach().numpy().tobytes(), logits.tobytes())
        loss = torch.nn.functional.cross_entropy(actual.double()[None, :], torch.tensor([3]))
        grad = torch.autograd.grad(loss, x)[0]
        reference_x = x.detach().double().requires_grad_()
        norm = reference_x * torch.rsqrt(reference_x.square().mean(-1, keepdim=True) + model.cfg.rms_eps)
        ref_logits = torch.tensor(logits, dtype=torch.float64) * (norm.abs().amax() / 127. / scale)
        reference_loss = torch.nn.functional.cross_entropy(ref_logits[None, :], torch.tensor([3]))
        reference_grad = torch.autograd.grad(reference_loss, reference_x)[0]
        self.assertTrue(torch.allclose(grad.double(), reference_grad, atol=2e-6, rtol=2e-5))

    def test_loss_decrease_does_not_imply_rank_or_answer_improvement(self):
        logits = torch.tensor([[4., 2., 1.]])
        target = torch.tensor([2])
        softened = logits * .9
        self.assertLess(float(torch.nn.functional.cross_entropy(softened, target)), float(torch.nn.functional.cross_entropy(logits, target)))
        self.assertTrue(torch.equal(logits.argsort(-1), softened.argsort(-1)))
        self.assertEqual(int(logits.argmax()), int(softened.argmax()))


if __name__ == "__main__": unittest.main()
