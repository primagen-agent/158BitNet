"""Full-preflight panel and inactive-gradient contracts, not memory scores."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from preflight_joint_full import select_pairs, gradient_requirements, bit_exact
from joint_binding_schedule import KINDS
from prepare_neural_memory_protocol import digest

ROOT = Path(__file__).resolve().parents[1]


class FullPreflightTests(unittest.TestCase):
    def test_retained_manifest_numeric_types_preserve_package_digests(self):
        checks = ROOT / 'training/memory/neural-system/checks'
        report = json.loads((checks / 'CG-003-full-preflight.json').read_text())
        for filename, field in (('CG-003-full-features.json', 'package_digest'), ('CG-003-smoke-features.json', 'smoke_digest')):
            manifest = json.loads((checks / filename).read_text())
            self.assertEqual(digest(manifest), report[field])

    def test_bit_identity_does_not_ignore_signed_zero_or_dtype(self):
        value = np.array([0.], dtype=np.float32)
        self.assertTrue(bit_exact(value, value.copy()))
        self.assertFalse(bit_exact(value, -value))
        self.assertFalse(bit_exact(value, value.astype(np.float64)))

    def records(self):
        rows = []
        for split in ('train', 'dev'):
            for line in (ROOT / 'training/memory/neural-system/data/JB-001' / f'{split}.index.jsonl').read_text().splitlines():
                r = json.loads(line)
                rows.append({**r, 'relation': r['relation_family']})
        return rows

    def test_panel_covers_all_kinds_languages_relations_but_only_one_train_world(self):
        world, pairs = select_pairs(self.records())
        self.assertEqual(len(pairs), 40)
        self.assertEqual({k for _, _, k, _ in pairs}, set(KINDS))
        self.assertEqual({(l, r) for l, r, _, _ in pairs}, {('en', 'home_city'), ('en', 'work_city'), ('zh', 'home_city'), ('zh', 'work_city')})
        for _, _, kind, records in pairs:
            self.assertTrue(all(r['split'] == 'train' and r['world_id'] == world for r in records))
            self.assertEqual(records[0]['scenario'], 'ordinary_source' if kind == 'ordinary_empty' else 'correct')

    def test_missing_case_cannot_reduce_denominator(self):
        rows = self.records(); world, _ = select_pairs(rows)
        rows = [r for r in rows if not (r['world_id'] == world and r['scenario'] == 'empty')]
        with self.assertRaises(ValueError): select_pairs(rows)
        with self.assertRaises(ValueError): select_pairs([r for r in rows if r['split'] == 'dev'])

    def gradients(self):
        names = ['reader.query_encoder.weight', 'reader.source_encoder.weight', 'reader.state_head.weight',
            'reader.position_query.weight', 'reader.copy_gate.weight', 'roles.content.output.weight',
            'roles.uncertainty_output.weight', 'factor_heads.0.weight', 'factor_heads.1.weight']
        return names, [torch.ones(1) for _ in names]

    def targets(self, states):
        return [SimpleNamespace(reply=SimpleNamespace(state_index=s)) for s in states]

    def test_supported_only_does_not_require_unused_uncertainty(self):
        names, grads = self.gradients(); grads[names.index('roles.uncertainty_output.weight')] = None
        result = gradient_requirements(names, grads, self.targets([1, 1]), 'value_swap', 'joint_aux')
        self.assertIn('roles.uncertainty_output.weight', result['none'])
        with self.assertRaises(ValueError): gradient_requirements(names, grads, self.targets([1, 2]), 'role_swap', 'joint_aux')

    def test_aux_ordinary_factors_may_be_disconnected_not_active_copy(self):
        names, grads = self.gradients()
        grads[-2:] = [None, None]
        gradient_requirements(names, grads, self.targets([0, 0]), 'ordinary_empty', 'joint_aux')
        with self.assertRaises(ValueError): gradient_requirements(names, grads, self.targets([0, 0]), 'ordinary_empty', 'joint_product')
        grads[names.index('reader.copy_gate.weight')] = None
        with self.assertRaises(ValueError): gradient_requirements(names, grads, self.targets([0, 0]), 'ordinary_empty', 'joint_aux')

    def test_nonfinite_rejected_but_zero_upstream_gradient_allowed(self):
        names, grads = self.gradients(); grads[0].zero_()
        gradient_requirements(names, grads, self.targets([1, 2]), 'role_swap', 'joint_product')
        grads[0].fill_(float('nan'))
        with self.assertRaises(ValueError): gradient_requirements(names, grads, self.targets([1, 2]), 'role_swap', 'joint_product')


if __name__ == '__main__': unittest.main()
