from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from neural_memory_contract import ModelBinding
from value_transport import (ActivatedFact, FactPayload, Origin, PayloadSnapshot, ReplyBinding,
    ValueSpan, TransportLimits, TransportError, PrefixRetokenizationRequired, UnsafeToken,
    TransportBudgetExceeded, compile_value, select_token)


class ByteFixtureCodec:
    """Explicit artificial codec for software contracts, never model evaluation."""
    vocab = 258
    bos_id = 0
    backbone_sha256 = 'a' * 64
    tokenizer_sha256 = 'b' * 64

    def __init__(self): self.calls = []
    def verify_identity(self): pass
    def encode_with_bos(self, text):
        self.calls.append(text); return (0,) + tuple(2 + b for b in text.encode())
    def decode_pieces(self, ids): return tuple(bytes([i-2]) for i in ids)
    def forbidden(self, token): return token in (0, 1)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.codec = ByteFixtureCodec()
        self.binding = ModelBinding('a'*64, 'b'*64, 'c'*64, 'd'*64)
        self.snapshot = PayloadSnapshot(self.binding, 'e'*64, 'user-1', 'memory-1', 2,
            (FactPayload('fact-1', b'city=Quimper;'), FactPayload('fact-2', b'city=Riga;')))
        self.reply = ReplyBinding('user-1', 'request-1', 'f'*64)
        self.handle = ActivatedFact(self.snapshot.digest, self.reply, ValueSpan(0, 5, 12), .8, Origin.ORACLE_FIXTURE)
        self.prefix_text = 'City:'; self.prefix = self.codec.encode_with_bos(self.prefix_text)

    def compile(self, **kw):
        args = dict(handle=self.handle, snapshot=self.snapshot, reply=self.reply, prefix_ids=self.prefix,
                    prefix_text=self.prefix_text, codec=self.codec, allow_oracle=True)
        args.update(kw); return compile_value(**args)

    def select(self, cursor, base=1, **kw):
        args = dict(snapshot=self.snapshot, reply=self.reply, prefix=cursor.expected_prefix, allow_oracle=True)
        args.update(kw); return cursor.select(base, **args)

    def commit(self, cursor, token, **kw):
        args = dict(snapshot=self.snapshot, reply=self.reply, prefix=cursor.expected_prefix, allow_oracle=True)
        args.update(kw); return cursor.commit(token, **args)

    def test_exact_ordered_value_ignores_base_and_finishes(self):
        cursor = self.compile(); pieces = []
        while not cursor.done:
            token = self.select(cursor, base=1)
            pieces += self.codec.decode_pieces((token,)); cursor = self.commit(cursor, token)
        self.assertEqual(b''.join(pieces), b'Quimper'); self.assertEqual(cursor.offset, 7)
        self.assertEqual(self.select(cursor, base=123), 123)
        self.assertIs(cursor.handle.origin, Origin.ORACLE_FIXTURE)

    def test_null_and_normal_do_not_call_codec(self):
        self.codec.calls.clear()
        self.assertIsNone(self.compile(handle=replace(self.handle, span=None)))
        self.assertEqual(self.codec.calls, [])
        sentinel = object()
        self.assertIs(select_token(None, sentinel, None, None, None), sentinel)

    def test_oracle_requires_explicit_opt_in_at_each_operation(self):
        with self.assertRaises(TransportError): self.compile(allow_oracle=False)
        c = self.compile()
        with self.assertRaises(TransportError): self.select(c, allow_oracle=False)
        with self.assertRaises(TransportError): self.commit(c, c.value_tokens[0], allow_oracle=False)
        with self.assertRaises(TransportError): self.compile(handle=replace(self.handle, span=None), allow_oracle=False)

    def test_wrong_token_premature_stop_and_skip_rejected(self):
        c = self.compile()
        for token in (1, c.value_tokens[1], True):
            with self.subTest(token=token), self.assertRaises(TransportError): self.commit(c, token)
        self.assertEqual(c.offset, 0)

    def test_actual_prefix_required(self):
        c = self.compile(); c1 = self.commit(c, c.value_tokens[0])
        for prefix in (c.expected_prefix, c1.expected_prefix + (12,), list(c1.expected_prefix)):
            with self.assertRaises(TransportError): self.select(c1, prefix=prefix)
        self.assertEqual(c.offset, 0)

    def test_repeated_tokens_are_distinct_positions(self):
        snap = replace(self.snapshot, facts=(FactPayload('fact-1', b'aaa'),))
        h = replace(self.handle, snapshot_sha256=snap.digest, span=ValueSpan(0, 0, 3))
        c = self.compile(handle=h, snapshot=snap)
        self.assertEqual(len(set(c.value_tokens)), 1)
        for i in range(3):
            self.assertEqual(c.offset, i); c = self.commit(c, self.select(c, snapshot=snap), snapshot=snap)
        self.assertTrue(c.done)
        with self.assertRaises(TransportError): self.commit(c, c.value_tokens[0], snapshot=snap)

    def test_changed_snapshot_same_revision_rejected(self):
        bad = replace(self.snapshot, facts=(FactPayload('fact-1', b'city=Changed;'),))
        with self.assertRaises(TransportError): self.compile(snapshot=bad)
        with self.assertRaises(TransportError): self.select(self.compile(), snapshot=bad)

    def test_model_scope_revision_and_store_binding(self):
        changes = [dict(scope='user-2'), dict(revision=3), dict(memory_id='memory-2'),
                   dict(memory_model_sha256='9'*64)]
        changes += [dict(binding=replace(self.binding, **{k:'9'*64})) for k in vars(self.binding)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(TransportError): self.compile(snapshot=replace(self.snapshot, **change))

    def test_request_and_context_binding(self):
        for change in (dict(scope='user-2'), dict(request_id='request-2'), dict(context_sha256='9'*64)):
            with self.assertRaises(TransportError): self.compile(reply=replace(self.reply, **change))

    def test_span_out_of_bounds_and_invalid_numbers(self):
        for span in (ValueSpan(2, 0, 1), ValueSpan(0, 0, 999)):
            with self.assertRaises(TransportError): self.compile(handle=replace(self.handle, span=span))
        for args in ((0, 2, 2), (-1, 0, 1), (0, True, 2)):
            with self.assertRaises(TransportError): ValueSpan(*args)

    def test_utf8_partial_token_pieces_roundtrip(self):
        snap = replace(self.snapshot, facts=(FactPayload('fact-1', '里加🧠'.encode()),))
        h = replace(self.handle, snapshot_sha256=snap.digest, span=ValueSpan(0, 0, 10))
        c = self.compile(handle=h, snapshot=snap)
        self.assertEqual(b''.join(c.pieces).decode(), '里加🧠')
        for span in (ValueSpan(0, 1, 3), ValueSpan(0, 0, 2)):
            with self.assertRaises(TransportError): self.compile(handle=replace(h, span=span), snapshot=snap)

    def test_partial_utf8_not_exposed_as_replacement_characters(self):
        snap = replace(self.snapshot, facts=(FactPayload('fact-1', '里加'.encode()),))
        h = replace(self.handle, snapshot_sha256=snap.digest, span=ValueSpan(0, 0, 6))
        c = self.compile(handle=h, snapshot=snap); seen = [c.committed_text]
        while not c.done:
            c = self.commit(c, self.select(c, snapshot=snap), snapshot=snap); seen.append(c.committed_text)
        self.assertEqual(seen, ['', '', '', '里', '里', '里', '里加'])

    def test_cursor_corruption_is_rejected(self):
        c = self.compile()
        for change in (dict(offset=-1), dict(offset=len(c.value_tokens)+1), dict(value=b'bad'),
                       dict(pieces=(b'',)+c.pieces[1:]), dict(value_tokens=list(c.value_tokens))):
            with self.assertRaises(TransportError): replace(c, **change)

    def test_neural_provenance_is_only_a_declaration(self):
        # No neural producer exists here. This verifies propagation, not origin authenticity.
        c = self.compile(handle=replace(self.handle, origin=Origin.NEURAL_PREDICTION), allow_oracle=False)
        self.assertIs(c.handle.origin, Origin.NEURAL_PREDICTION)
        self.assertEqual(self.select(c, allow_oracle=False), c.value_tokens[0])

    def test_budget_rejected_before_any_emission(self):
        for limits in (TransportLimits(max_value_bytes=6), TransportLimits(max_value_tokens=6),
                       TransportLimits(max_total_tokens=len(self.prefix)+6), TransportLimits(max_total_tokens=1)):
            with self.assertRaises(TransportBudgetExceeded): self.compile(limits=limits)
        self.assertEqual(self.handle.span, ValueSpan(0, 5, 12))

    def test_codec_identity_must_match(self):
        self.codec.tokenizer_sha256 = '9'*64
        with self.assertRaises(TransportError): self.compile()

    def test_noncanonical_actual_prefix_rejected(self):
        with self.assertRaises(PrefixRetokenizationRequired): self.compile(prefix_ids=self.prefix + (40,))

    def test_retokenizing_append_rejected(self):
        old = self.codec.encode_with_bos
        self.codec.encode_with_bos = lambda text: old(text) if text == self.prefix_text else (0, 40, 41)
        with self.assertRaises(PrefixRetokenizationRequired): self.compile()

    def test_wrong_suffix_bytes_rejected(self):
        old = self.codec.encode_with_bos
        self.codec.encode_with_bos = lambda text: old(text) if text == self.prefix_text else self.prefix + (40,)
        with self.assertRaises(TransportError): self.compile()

    def test_control_token_rejected(self):
        self.codec.forbidden = lambda t: t == ord('Q') + 2
        with self.assertRaises(UnsafeToken): self.compile()

    def test_immutable_inventory_and_provenance(self):
        with self.assertRaises(FrozenInstanceError): self.snapshot.revision = 3
        with self.assertRaises(TransportError): replace(self.snapshot, facts=list(self.snapshot.facts))
        with self.assertRaises(TransportError): FactPayload('fact-1', bytearray(b'x'))
        with self.assertRaises(TransportError): FactPayload('fact-1', b'\xff')
        with self.assertRaises(TransportError): replace(self.handle, confidence=float('nan'))
        with self.assertRaises(TransportError): replace(self.handle, origin='neural_prediction')

    def test_no_whole_source_in_tokenizer_prompt(self):
        self.codec.calls.clear(); self.compile()
        self.assertEqual(self.codec.calls, ['City:', 'City:Quimper'])
        self.assertNotIn('city=', ''.join(self.codec.calls))
        self.assertNotIn('Riga', ''.join(self.codec.calls))

    def test_transport_does_not_repair_wrong_fact_or_incomplete_span(self):
        # These choices would be wrong for Quimper, but this operator has no
        # semantic query/answer oracle and must not silently choose a different fact.
        for span, value in ((ValueSpan(1, 5, 9), b'Riga'), (ValueSpan(0, 5, 7), b'Qu')):
            c = self.compile(handle=replace(self.handle, span=span))
            self.assertEqual(b''.join(c.pieces), value)


if __name__ == '__main__': unittest.main()
