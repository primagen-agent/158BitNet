"""Synthetic design guards, not trained memory accuracy."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from prepare_joint_binding_curriculum import worlds, make_cases, build, validate, materialize
from joint_binding_schedule import pair_schedule

CONFIG = Path(__file__).resolve().parents[1] / 'training/memory/neural-system/data/JB-001.json'


class JointCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text())
        cls.rows = [row for world, split in worlds(cls.config) for row in make_cases(world, split)]

    def test_deterministic_counts_and_all_split_isolation(self):
        files, report = build(self.config)
        self.assertEqual(files, build(self.config)[0])
        self.assertEqual(report['records'], 960); self.assertEqual(report['pairs'], 800)
        self.assertEqual(report['role_swap_groups_verified'], 80)
        self.assertEqual(sum(v for k, v in report['counts'].items() if k.startswith('train/')), 576)
        for split in ('dev', 'test'):
            self.assertEqual(sum(v for k, v in report['counts'].items() if k.startswith(split+'/')), 192)

    def test_presence_and_joint_binding_are_distinct(self):
        for runtime, label, meta in self.rows:
            if meta['scenario'] == 'role_swap':
                self.assertEqual(label['factor_targets'], [True, True])
                self.assertFalse(label['joint_current_actual']); self.assertEqual(label['state'], 'insufficient')
            self.assertEqual(set(runtime), {'id', 'context', 'episodes'})

    def test_identical_source_requires_query_relation_not_person_identity(self):
        groups = {}
        for row in self.rows:
            m = row[2]
            groups.setdefault((m['world_id'], m['language']), {})[(m['relation_family'], m['scenario'])] = row
        for group in groups.values():
            a = group[('home_city', 'correct')]; b = group[('work_city', 'role_swap')]
            reordered = group[('home_city', 'clause_reorder')]
            self.assertEqual(a[2]['query_subject'], b[2]['query_subject'])
            self.assertEqual(reordered[0]['episodes'], b[0]['episodes'])
            self.assertNotEqual(reordered[0]['context'], b[0]['context'])
            self.assertEqual(reordered[1]['state'], 'supported'); self.assertEqual(b[1]['state'], 'insufficient')

    def test_leakage_corruption_and_missing_pair_rejected(self):
        for mode in ('split', 'state', 'joint', 'source', 'gold'):
            rows = copy.deepcopy(self.rows)
            runtime, label, meta = rows[0]
            if mode == 'split': meta['split'] = 'test' if meta['split'] != 'test' else 'train'
            elif mode == 'state': label['state'] = 'insufficient'
            elif mode == 'joint': label['joint_current_actual'] = False
            elif mode == 'source': label['relevant_episode_indices'] = [10]
            else: runtime['gold_subject'] = 'someone'
            with self.subTest(mode=mode), self.assertRaises(ValueError): validate(rows)
        with self.assertRaises(ValueError): validate(self.rows[:-1])

    def test_time_and_negation_not_conflated_with_relation_absence(self):
        for _, label, meta in self.rows:
            if meta['scenario'] in ('negated', 'historical'):
                self.assertEqual(label['factor_targets'], [True, True]); self.assertFalse(label['joint_current_actual'])
            if meta['scenario'].startswith('ordinary_'):
                self.assertEqual(label['factor_targets'], [None, None]); self.assertIsNone(label['joint_current_actual'])

    def test_materialization_is_immutable_and_exactly_regenerable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            first = materialize(CONFIG, root)
            self.assertEqual(first, materialize(CONFIG, root, True))
            with self.assertRaises(FileExistsError): materialize(CONFIG, root)
            (root / 'test.labels.jsonl').write_text('tampered')
            with self.assertRaises(ValueError): materialize(CONFIG, root, True)

    def test_new_budget_is_not_launch_authority(self):
        plan = json.loads((CONFIG.parents[1] / 'experiments/CG-003.json').read_text())
        # The user approved the bounded budget; this does not satisfy launch gates.
        self.assertTrue(plan['training_approved'])
        from train_joint_memory import require_joint_launch
        with self.assertRaisesRegex(ValueError, 'Training blocked'):
            require_joint_launch(config=plan)
        self.assertEqual(plan['proposed_budget']['total_optimizer_steps'], 100)

    def test_pair_schedule_is_fixed_and_rejects_dev_test_or_extra_budget(self):
        pairs, _ = validate(self.rows); train = [p for p in pairs if p['split'] == 'train']
        first = pair_schedule(train)
        self.assertEqual(first, pair_schedule(train)); self.assertEqual(len(first), 50)
        self.assertEqual(sum(len(s['pairs']) for s in first), 200)
        with self.assertRaises(ValueError): pair_schedule(pairs)
        with self.assertRaises(ValueError): pair_schedule(train, steps=100)


if __name__ == '__main__': unittest.main()
