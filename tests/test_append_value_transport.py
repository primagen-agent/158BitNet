from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from append_value_transport import AppendValueCodec, compile_append_value
from neural_memory_contract import ModelBinding
from value_transport import (ActivatedFact, FactPayload, Origin, PayloadSnapshot, ReplyBinding,
                             ValueSpan, TransportLimits, TransportError, TransportBudgetExceeded, UnsafeToken)


class FixtureCodec:
    vocab = 260; bos_id = 0
    backbone_sha256 = 'a'*64; tokenizer_sha256 = 'b'*64
    def verify_identity(self): pass
    def encode_value(self, value): return tuple(b+2 for b in value)
    def decode_pieces(self, ids): return tuple(b'X:' if i == 259 else bytes([i-2]) for i in ids)
    def forbidden(self, i): return i in (0, 1)
    def encode_with_bos(self, text): raise AssertionError('actual prefix must not be re-encoded')


class AppendTests(unittest.TestCase):
    def setUp(self):
        self.codec = FixtureCodec()
        self.snapshot = PayloadSnapshot(ModelBinding('a'*64,'b'*64,'c'*64,'d'*64), 'e'*64, 'u', 'm', 0,
                                        (FactPayload('f', b'value="Riga";'),))
        self.reply = ReplyBinding('u', 'r', 'f'*64)
        self.handle = ActivatedFact(self.snapshot.digest, self.reply, ValueSpan(0,6,12), .7, Origin.ORACLE_FIXTURE)
        self.prefix = (0,ord('X')+2,ord(':')+2)

    def compile(self, **kw):
        args = dict(handle=self.handle, snapshot=self.snapshot, reply=self.reply, prefix_ids=self.prefix,
                    prefix_text='X:', codec=self.codec, allow_oracle=True)
        args.update(kw); return compile_append_value(**args)

    def test_actual_prefix_not_reencoded_and_quotes_preserved(self):
        c = self.compile(); self.assertEqual(c.initial_prefix, self.prefix); self.assertEqual(c.value, b'"Riga"')
        while not c.done:
            token = c.select(1, self.snapshot, self.reply, c.expected_prefix, allow_oracle=True)
            c = c.commit(token, self.snapshot, self.reply, c.expected_prefix, allow_oracle=True)
        self.assertEqual(c.committed_text, '"Riga"')
        self.assertEqual(c.select(99,self.snapshot,self.reply,c.expected_prefix,allow_oracle=True),99)

    def test_alternate_valid_prefix_segmentation_preserved(self):
        c = self.compile(prefix_ids=(0,259)); self.assertEqual(c.initial_prefix,(0,259))

    def test_prefix_byte_mismatch_rejected(self):
        with self.assertRaises(TransportError): self.compile(prefix_text='Y:')

    def test_oracle_requires_opt_in(self):
        with self.assertRaises(TransportError): self.compile(allow_oracle=False)

    def test_stale_scope_request_context_and_payload(self):
        for snap in (replace(self.snapshot,revision=1), replace(self.snapshot,scope='v'),
                     replace(self.snapshot,facts=(FactPayload('f',b'wrong'),))):
            with self.assertRaises(TransportError): self.compile(snapshot=snap)
        for reply in (replace(self.reply,request_id='s'), replace(self.reply,context_sha256='1'*64)):
            with self.assertRaises(TransportError): self.compile(reply=reply)

    def test_null_never_calls_codec(self):
        self.codec.verify_identity = lambda: (_ for _ in ()).throw(AssertionError('codec called'))
        self.assertIsNone(self.compile(handle=replace(self.handle,span=None)))

    def test_budget_is_pre_emission(self):
        for limit in (TransportLimits(max_value_bytes=1),TransportLimits(max_value_tokens=1),TransportLimits(max_total_tokens=3)):
            with self.assertRaises(TransportBudgetExceeded): self.compile(limits=limit)

    def test_suffix_control_and_wrong_bytes_rejected(self):
        self.codec.forbidden = lambda t: t == ord('"')+2
        with self.assertRaises(UnsafeToken): self.compile()
        self.codec.encode_value = lambda v: (90,)
        with self.assertRaises(TransportError): self.compile()

    def test_identity_and_utf8_span_validation(self):
        self.codec.tokenizer_sha256 = '9'*64
        with self.assertRaises(TransportError): self.compile()
        self.codec.tokenizer_sha256 = 'b'*64
        s = replace(self.snapshot,facts=(FactPayload('f','里'.encode()),))
        h = replace(self.handle,snapshot_sha256=s.digest,span=ValueSpan(0,1,3))
        with self.assertRaises(TransportError): self.compile(handle=h,snapshot=s)


class AnchorTests(unittest.TestCase):
    def codec(self, combined):
        # No subprocess: fixture of exact token protocol failures.
        c = AppendValueCodec.__new__(AppendValueCodec)
        c.vocab=5; c.bos_id=0
        c.encode_with_bos=lambda text: (0,1,2) if text == c.anchor else combined
        c.decode_pieces=lambda ids: tuple({1:b' ',2:c.anchor.encode(),3:b'x',4:b' x'}[i] for i in ids)
        c.forbidden=lambda i:i in (0,2)
        return c

    def test_anchor_never_in_suffix(self):
        self.assertEqual(self.codec((0,1,2,3)).encode_value(b'x'),(3,))

    def test_anchor_merge_and_dummy_space_leak_rejected(self):
        for ids in ((0,1,3),(0,1,2,4),(0,1,2)):
            with self.assertRaises(TransportError): self.codec(ids).encode_value(b'x')


if __name__ == '__main__': unittest.main()
