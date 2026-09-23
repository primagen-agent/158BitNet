"""Operator observer protocol and layout regressions; no memory training."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from diagnose_neural_memory_operators import STAGES, read_trace, to_split, to_interleaved, quantization_decision_changes, first_layer_chain
from diagnose_neural_memory_features import c_q8_activation_reference
from test_memory_fusion import tiny_backbone


class OperatorTests(unittest.TestCase):
    def test_rope_layout_roundtrip_and_known_pairs(self):
        x = torch.arange(16).reshape(2, 8).float()
        self.assertEqual(to_split(x, 4)[0].tolist(), [0, 2, 1, 3, 4, 6, 5, 7])
        self.assertTrue(torch.equal(to_interleaved(to_split(x, 4), 4), x))

    def test_trace_rejects_missing_duplicate_and_truncated_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.bin"
            records = [struct.pack("<IIII", 0, stage, 0, 4) + np.ones(4, dtype="<f4").tobytes() for stage in STAGES]
            path.write_bytes(b"BNTR0001" + b"".join(records))
            self.assertEqual(set(read_trace(path, [1])[0]), set(STAGES.values()))
            for data in (b"BNTR0001" + b"".join(records[:-1]), b"BNTR0001" + b"".join(records + records[:1]),
                         b"BNTR0001" + b"".join(records)[:-1], b"BNTR0001" + b"x"):
                path.write_bytes(data)
                with self.assertRaises(ValueError): read_trace(path, [1])

    def test_tiny_rounding_difference_can_change_an_int8_code(self):
        x = torch.tensor([[.5, 127.]])
        changed = x.clone(); changed[0, 0] += 1e-6
        self.assertEqual(quantization_decision_changes(x, changed), {"changed_codes": 1, "total_codes": 2})

    def test_chain_matches_actual_backbone_without_observer_state(self):
        torch.manual_seed(41)
        model = tiny_backbone()
        model.linear = lambda h, w: torch.nn.functional.linear(c_q8_activation_reference(h), w)
        ids = torch.tensor([1, 4, 5, 8])
        chain = first_layer_chain(model, model.token_embd[ids] * model.cfg.embedding_scale)
        self.assertTrue(torch.equal(chain["output"], model(ids, return_hidden_layer=0)))


if __name__ == "__main__": unittest.main()
