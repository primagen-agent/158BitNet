import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from diagnose_joint_value_path import annotate_source, mass_by_category


class ValuePathTests(unittest.TestCase):
    def fixture(self):
        fact = dict(subject='Seline', relation='home_city', value='Riga', time='current', status='actual')
        message = dict(role='user', speaker='user', text='Seline currently lives in Riga.')
        framed = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        meta = dict(source_facts=[fact], scenario='correct', language='en', query_subject='Seline', query_relation='home_city')
        return message, framed, meta

    def test_whitespace_partial_pieces_and_boundary(self):
        msg, frame, meta = self.fixture()
        before, after = frame.split('Riga')
        pieces = [before[:-1].encode(), b' R', b'iga', after.encode()]
        spans, rows = annotate_source(msg, frame, pieces, [False, True, True, False], meta)
        self.assertEqual([r['category'] for r in rows], ['structural', 'target_value', 'target_value', 'structural'])
        pieces = [before.encode(), b'Riga.', after[1:].encode()]
        _, rows = annotate_source(msg, frame, pieces, [False, True, False], meta)
        self.assertEqual(rows[1]['category'], 'ambiguous_value_boundary')

    def test_joint_mismatch_not_target(self):
        msg, frame, meta = self.fixture(); meta['query_relation'] = 'work_city'
        before, after = frame.split('Riga')
        spans, rows = annotate_source(msg, frame, [before.encode(), b'Riga', after.encode()], [False, True, False], meta)
        self.assertFalse(spans[0]['target']); self.assertEqual(rows[1]['category'], 'other_value')

    def test_bad_annotation_rejected(self):
        msg, frame, meta = self.fixture(); meta['source_facts'][0]['value'] = 'Rome'
        with self.assertRaises(ValueError): annotate_source(msg, frame, [frame.encode()], [False], meta)

    def test_partition_includes_null_and_structural(self):
        result = mass_by_category([.2, .3, .5], [{'category': 'structural'}, {'category': 'target_value'}])
        self.assertEqual(result, dict(null=.2, structural=.3, target_value=.5))
        with self.assertRaises(ValueError): mass_by_category([.2], [{'category': 'structural'}])
        with self.assertRaises(ValueError): mass_by_category([.1], [])

    def test_empty_source(self):
        self.assertEqual(annotate_source(None, None, [], [], {'source_facts': []}), ([], []))

    def test_utf8_byte_spans_with_tokenizer_padding(self):
        msg, _, meta = self.fixture(); msg['text'] = 'Seline现在住在Riga。'; meta['language'] = 'zh'
        frame = json.dumps(msg, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        before, after = frame.split('Riga')
        pieces = [b' ' + before.encode(), b'R', b'iga', after.encode()]
        spans, rows = annotate_source(msg, frame, pieces, [False, True, True, False], meta)
        self.assertEqual(spans[0]['value_start'], len(before.encode()) + 1)
        self.assertEqual(rows[1]['byte_start'], spans[0]['value_start'])
        self.assertEqual(rows[2]['byte_end'], spans[0]['value_end'])

    def test_missing_or_misaligned_payload_fails_closed(self):
        msg, frame, meta = self.fixture()
        with self.assertRaises(ValueError): annotate_source(msg, frame, [b'bad'], [False], meta)
        with self.assertRaises(ValueError): annotate_source(msg, frame, [frame.encode()], [], meta)


if __name__ == '__main__': unittest.main()
