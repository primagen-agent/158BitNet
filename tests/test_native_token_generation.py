"""Prefix/lifecycle and joint-binding counterexample fixtures, not accuracy."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from native_token_generation import check_extension, NativeTokenBackend
from diagnose_joint_native_generation import panel, joint_annotation
from prepare_neural_memory_protocol import fact


class NativeTokenTests(unittest.TestCase):
    def test_joint_presence_does_not_imply_binding(self):
        rows = panel()
        self.assertEqual(len(rows), 6)
        for a, b in ((rows[0], rows[1]), (rows[2], rows[3])):
            for key in ('subject_present', 'relation_present'):
                self.assertTrue(a['annotation'][key]); self.assertTrue(b['annotation'][key])
            self.assertTrue(a['annotation']['joint_current_actual'])
            self.assertFalse(b['annotation']['joint_current_actual'])
            self.assertEqual(set(a['runtime']), {'id', 'context', 'episodes'})

    def test_old_negated_and_quoted_not_supported_by_presence(self):
        for time, status in (('2020', 'actual'), ('current', 'negated'), ('current', 'quoted')):
            a = joint_annotation([fact('A', 'home_city', 'Graz', time, status)], 'A', 'home_city')
            self.assertTrue(a['subject_present'] and a['relation_present'])
            self.assertFalse(a['joint_current_actual'])

    def test_only_natural_initial_and_actual_prediction_prefixes(self):
        initial = (1, 2); source = ('natural source',)
        check_extension(initial, None, None, initial, source, source)
        check_extension(initial, initial, 3, (1, 2, 3), source, source)
        for prefix, sources in (((1, 2, 5), source), ((1, 2, 3), ('changed',)), ((1, 2, 3), list(source))):
            with self.assertRaises(ValueError): check_extension(initial, initial, 3, prefix, source, sources)
        with self.assertRaises(ValueError): check_extension(initial, None, None, (1, 2, 3), source, source)

    def test_numerical_parity_requires_top1_and_registered_tolerance(self):
        a = torch.tensor([[1., 2.]])
        self.assertEqual(NativeTokenBackend.compare(a, a), 0)
        with self.assertRaises(ValueError): NativeTokenBackend.compare(a, a + .1)
        a = torch.tensor([[1., 1.000001]])
        with self.assertRaises(ValueError): NativeTokenBackend.compare(a, a.flip(-1))


if __name__ == '__main__': unittest.main()
