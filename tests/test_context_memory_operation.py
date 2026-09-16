import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_context_memory_operation import operation_features


class ContextMemoryOperationTest(unittest.TestCase):
    def test_complete_message_and_not_bos_controls_features(self):
        hidden = torch.randn(6, 4, 8)
        baseline = operation_features(hidden)
        changed_bos = hidden.clone(); changed_bos[0] = 1000
        self.assertTrue(torch.allclose(baseline, operation_features(changed_bos)))
        changed_tail = hidden.clone(); changed_tail[-1] = -hidden[-1]
        self.assertFalse(torch.allclose(baseline, operation_features(changed_tail)))
        self.assertEqual(tuple(baseline.shape), (16,))

    def test_empty_signal_and_single_token_are_finite(self):
        for length in (1, 8):
            self.assertTrue(torch.isfinite(operation_features(torch.zeros(length, 4, 8))).all())


if __name__ == "__main__":
    unittest.main()
