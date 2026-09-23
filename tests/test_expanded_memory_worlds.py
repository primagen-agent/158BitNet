from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from prepare_expanded_memory_worlds import (JB1_CONFIG, SCENARIOS, SUPPORTED, build, materialize,
                                            validate, worlds)
from prepare_neural_memory_protocol import digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'training/memory/neural-system/data/JB-002.json'
CORPUS = ROOT / 'training/memory/neural-system/data/JB-002'


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.report = build(json.loads(CONFIG.read_text()))

    def test_registered_inventory(self):
        self.assertEqual(self.report['records'], 768)
        self.assertEqual((self.report['worlds'], self.report['pairs']), (24, 480))
        self.assertEqual(self.report['counts'], {'dev/insufficient': 120, 'dev/supported': 72,
                                                 'test/insufficient': 120, 'test/supported': 72,
                                                 'train/insufficient': 240, 'train/supported': 144})

    def test_frozen_corpus_matches_rebuild(self):
        manifest = json.loads((CORPUS / 'manifest.json').read_text())
        for name, data in self.files.items():
            self.assertEqual((CORPUS / name).read_bytes(), data, name)
            self.assertEqual(manifest['file_sha256'][name],
                             __import__('hashlib').sha256(data).hexdigest(), name)

    def test_disjoint_from_jb001(self):
        jb1 = json.loads(JB1_CONFIG.read_text()); jb2 = json.loads(CONFIG.read_text())
        self.assertFalse(set(jb1['people']) & set(jb2['people']))
        self.assertFalse(set(jb1['cities']) & set(jb2['cities']))

    def test_label_and_index_are_physically_separate(self):
        inputs = {json.loads(l)['id']: json.loads(l) for l in self.files['train.inputs.jsonl'].decode().splitlines()}
        for line in self.files['train.labels.jsonl'].decode().splitlines():
            label = json.loads(line)
            runtime = inputs[label['id']]
            self.assertEqual(digest(runtime), label['input_sha256'])
            # Inputs are pure user messages; no supervision field is embedded.
            self.assertEqual(sorted(runtime), ['context', 'episodes', 'id'])
            self.assertTrue(all(sorted(m) == ['role', 'speaker', 'text']
                                for m in runtime['context'] + runtime['episodes']))
            self.assertNotIn(label['state'], json.dumps(runtime, ensure_ascii=False))

    def test_time_pair_changes_exactly_one_clause(self):
        index = {json.loads(l)['id']: json.loads(l) for l in self.files['train.index.jsonl'].decode().splitlines()}
        for meta in index.values():
            if meta['scenario'] == 'historical_stated':
                self.assertEqual(meta['query_time'], '2020')
        rows = [row for world, split in worlds(json.loads(CONFIG.read_text()))
                for row in [(world, split)]]
        self.assertEqual(len(rows), 24)


class GeneratorTests(unittest.TestCase):
    def test_scenario_state_table_frozen(self):
        built, _ = build(json.loads(CONFIG.read_text()))
        metas = [json.loads(l) for l in built['train.index.jsonl'].decode().splitlines()]
        labels = {json.loads(l)['id']: json.loads(l) for l in built['train.labels.jsonl'].decode().splitlines()}
        for meta in metas:
            expected = 'supported' if meta['scenario'] in SUPPORTED else 'insufficient'
            self.assertEqual(labels[meta['id']]['state'], expected)

    def test_deterministic_rebuild(self):
        first, _ = build(json.loads(CONFIG.read_text()))
        second, _ = build(json.loads(CONFIG.read_text()))
        self.assertEqual(first, second)

    def test_inventory_overlap_rejected(self):
        config = json.loads(CONFIG.read_text())
        config['people'][0] = json.loads(JB1_CONFIG.read_text())['people'][0]
        with self.assertRaises(ValueError):
            list(worlds(config))

    def test_verify_detects_tampering(self):
        import shutil, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'corpus'; shutil.copytree(CORPUS, copy)
            materialize(str(CONFIG), str(copy), verify=True)  # intact copy passes
            target = copy / 'train.labels.jsonl'; data = bytearray(target.read_bytes())
            data[data.index(b'"sample_weight"') + 2] = ord('2')
            target.write_bytes(bytes(data))
            with self.assertRaises(ValueError):
                materialize(str(CONFIG), str(copy), verify=True)
        manifest = json.loads((CORPUS / 'manifest.json').read_text())
        self.assertEqual(manifest['test_policy'], 'sealed_no_training_selection_or_model_forward')


if __name__ == '__main__':
    unittest.main()
