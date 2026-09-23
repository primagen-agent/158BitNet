from dataclasses import asdict
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from compile_time_scoped_teacher import compile_teacher_time, positive_bytes_time
from neural_memory_contract import ModelBinding
from native_memory_encoder import BACKBONE_SHA256
from value_teacher_trajectory import compile_teacher

ROOT = Path(__file__).resolve().parents[1]
JB1 = ROOT / 'training/memory/neural-system/data/JB-001'
JB2 = ROOT / 'training/memory/neural-system/data/JB-002'


def load(corpus, split):
    data = {}
    for name in ('inputs', 'labels', 'index'):
        data[name] = {r['id']: r for r in map(json.loads, (corpus / f'{split}.{name}.jsonl').read_text().splitlines())}
    return data


class ResolverTests(unittest.TestCase):
    def test_selects_2020_fact_not_current(self):
        idx = load(JB2, 'train')['index']
        meta = next(m for m in idx.values() if m['scenario'] == 'historical_stated')
        raw = load(JB2, 'train')['inputs'][meta['id']]['episodes'][0]['text'].encode()
        spans = positive_bytes_time(raw, meta)
        clause = raw[spans.fact_bytes[0]:spans.fact_bytes[1]].decode()
        self.assertIn('2020', clause)
        self.assertEqual(raw[spans.value_bytes[0]:spans.value_bytes[1]].decode(),
                         next(f['value'] for f in meta['source_facts']
                              if f['time'] == '2020' and f['status'] == 'actual'))

    def test_ambiguous_or_missing_time_fact_rejected(self):
        idx = load(JB2, 'train')['index']
        meta = next(m for m in idx.values() if m['scenario'] == 'historical_stated')
        inputs = load(JB2, 'train')['inputs']
        raw = inputs[meta['id']]['episodes'][0]['text'].encode()
        doubled = raw + b' ' + raw  # duplicates every clause
        with self.assertRaises(ValueError):
            positive_bytes_time(doubled, meta)
        stripped = [f for f in meta['source_facts'] if f['time'] == 'current']
        from copy import deepcopy
        broken = deepcopy(meta); broken['source_facts'] = stripped
        with self.assertRaises(ValueError):
            positive_bytes_time(raw, broken)


class CompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from append_value_transport import AppendValueCodec
        fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
        cls.codec = AppendValueCodec(ROOT / 'build/neural-memory-cg003-frozen-build/tok_probe',
                                     ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf', fm['tokenizer_sha256'])
        cls.binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'], '0' * 64, '0' * 64)
        cls.jb1 = load(JB1, 'train'); cls.jb2 = load(JB2, 'train')

    @classmethod
    def tearDownClass(cls):
        cls.codec.verify_identity(); cls.codec.close()

    def compile(self, corpus_data, rid):
        return compile_teacher_time(corpus_data['inputs'][rid], corpus_data['labels'][rid],
                                    corpus_data['index'][rid], self.codec, self.binding, '0' * 64,
                                    purpose='training_diagnostic')

    def test_current_time_delegates_byte_identically(self):
        jb1_ids = [m['id'] for m in self.jb1['index'].values() if m['scenario'] in ('correct', 'value_swap')][:4]
        jb2_ids = [m['id'] for m in self.jb2['index'].values() if m['scenario'] in ('correct', 'multi_fact')][:4]
        for rid in jb1_ids + jb2_ids:
            data = self.jb1 if rid.startswith('joint-') else self.jb2
            wrapped = self.compile(data, rid)
            direct = compile_teacher(data['inputs'][rid], data['labels'][rid], data['index'][rid],
                                     self.codec, self.binding, '0' * 64, purpose='training_diagnostic')
            self.assertEqual(asdict(wrapped), asdict(direct), rid)

    def test_historical_stated_compiles_with_2020_value(self):
        metas = [m for m in self.jb2['index'].values() if m['scenario'] == 'historical_stated']
        self.assertGreater(len(metas), 0)
        for meta in metas[:8]:
            t = self.compile(self.jb2, meta['id'])
            self.assertEqual(t.static_targets.route, 1)
            self.assertEqual(t.phases.count('START'), 1)
            self.assertEqual(t.phases.count('END'), 1)
            self.assertEqual(t.completion_ids[-1], self.codec.tokenizer.eos())
            expected = next(f['value'] for f in meta['source_facts']
                            if f['time'] == '2020' and f['status'] == 'actual')
            self.assertIn(expected, t.response)
            again = self.compile(self.jb2, meta['id'])
            self.assertEqual(asdict(t), asdict(again))

    def test_non_supported_time_target_rejected(self):
        wrong_time = next(m for m in self.jb2['index'].values() if m['scenario'] == 'wrong_time')
        # Time-qualified refusals need no span, so they delegate and compile.
        t = self.compile(self.jb2, wrong_time['id'])
        self.assertEqual(t.static_targets.route, 2)
        self.assertEqual(t.completion_ids[-1], self.codec.tokenizer.eos())
        direct = compile_teacher(self.jb2['inputs'][wrong_time['id']], self.jb2['labels'][wrong_time['id']],
                                 wrong_time, self.codec, self.binding, '0' * 64, purpose='training_diagnostic')
        self.assertEqual(asdict(t), asdict(direct))


if __name__ == '__main__':
    unittest.main()
