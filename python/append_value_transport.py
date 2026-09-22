"""DG-016 append-only C-tokenized value transport; oracle component, not recall.

The suffix is byte-exact but prefix+suffix need not be canonical whole-text BPE.
No existing output token is ever changed. DG-015's contract stays untouched.
"""
from native_value_codec import NativeValueCodec
from value_transport import (ActivatedFact, TransportLimits, TransportBudgetExceeded, UnsafeToken,
                             ValueCursor, require, text_bytes, token_ids)


class AppendValueCodec(NativeValueCodec):
    # Tokenizer-only boundary. NEVER part of the backbone input or emitted value.
    anchor = '<|im_end|>'

    def encode_value(self, value):
        text = text_bytes(value)
        anchor_ids = self.encode_with_bos(self.anchor)
        anchor_pieces = self.decode_pieces(anchor_ids[1:])
        require(anchor_pieces in ((self.anchor.encode(),), (b' ', self.anchor.encode())),
                'unqualified C tokenizer anchor framing')
        require(self.forbidden(anchor_ids[-1]), 'anchor must be a recognized control boundary')
        combined = self.encode_with_bos(self.anchor + text)
        token_ids(combined, self)
        require(combined[:len(anchor_ids)] == anchor_ids, 'C tokenizer merged across anchor')
        suffix = combined[len(anchor_ids):]
        require(bool(suffix), 'empty encoded value')
        pieces = self.decode_pieces(suffix)
        require(b''.join(pieces) == value, 'anchor suffix changed value bytes')
        return suffix


def compile_append_value(handle, snapshot, reply, prefix_ids, prefix_text, codec, *, limits=TransportLimits(), allow_oracle=False):
    require(type(handle) is ActivatedFact and type(limits) is TransportLimits, 'typed handle/limits required')
    value = handle.validate(snapshot, reply, allow_oracle=allow_oracle)
    if value is None: return None
    if len(value) > limits.max_value_bytes: raise TransportBudgetExceeded('value byte budget exceeded')
    codec.verify_identity()
    require(codec.backbone_sha256 == snapshot.binding.backbone_sha256 and codec.tokenizer_sha256 == snapshot.binding.tokenizer_sha256,
            'codec identity differs from bound model')
    token_ids(prefix_ids, codec)
    require(prefix_ids[0] == codec.bos_id and type(prefix_text) is str and bool(prefix_text) and '\0' not in prefix_text,
            'complete actual prefix required')
    if len(prefix_ids) >= limits.max_total_tokens: raise TransportBudgetExceeded('no remaining context capacity')
    raw_prefix = prefix_text.encode('utf-8', errors='strict')
    pieces = codec.decode_pieces(prefix_ids[1:])
    require(len(pieces) == len(prefix_ids)-1 and all(type(p) is bytes and p for p in pieces), 'invalid prefix pieces')
    # Do not replace actual autoregressive segmentation with re-encoded text.
    require(b''.join(pieces) in (raw_prefix, b' '+raw_prefix), 'actual prefix bytes do not match caller text')
    suffix = codec.encode_value(value)
    token_ids(suffix, codec)
    suffix_pieces = tuple(codec.decode_pieces(suffix))
    require(len(suffix_pieces) == len(suffix) and all(type(p) is bytes and p for p in suffix_pieces), 'invalid suffix pieces')
    require(b''.join(suffix_pieces) == value, 'suffix does not preserve value bytes')
    for t, p in zip(suffix, suffix_pieces):
        if codec.forbidden(t) or b'\0' in p: raise UnsafeToken('control token in value suffix')
    if len(suffix) > limits.max_value_tokens or len(prefix_ids)+len(suffix) > limits.max_total_tokens:
        raise TransportBudgetExceeded('token budget exceeded before emission')
    codec.verify_identity()
    return ValueCursor(handle, prefix_ids, suffix, suffix_pieces, value, 0)
