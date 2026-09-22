"""Structural payload boundary checks only; no subject/value extraction."""
import json

import torch

from memory_token_read import TokenReadInput
from native_memory_encoder import HIDDEN, VOCAB, digest


def payload_mask(message, framed_text, pieces):
    expected = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if framed_text != expected or set(message) != {'role', 'speaker', 'text'}:
        raise ValueError('source framing mismatch')
    literal = json.dumps(message['text'], ensure_ascii=False).encode()
    raw = framed_text.encode(); decoded = b''.join(pieces)
    pad = 0 if decoded == raw else 1 if decoded == b' ' + raw else None
    if pad is None: raise ValueError('C token bytes do not reconstruct source')
    # Text is the last key in the canonical source frame. No semantic search.
    if not raw.endswith(literal + b'}'): raise ValueError('text field boundary mismatch')
    if literal[1:-1] != message['text'].encode():
        return [False] * len(pieces)  # Escapes need a future byte-level channel.
    start = len(raw) - len(literal) + pad
    end = len(raw) - 2 + pad
    result = []; offset = 0
    for piece in pieces:
        stop = offset + len(piece)
        control = piece.startswith(b'<|') and piece.endswith(b'|>')
        result.append(bool(piece) and offset >= start and stop <= end and not control)
        offset = stop
    return result


def input_binding(encoder_identity, tokenizer_sha256):
    return digest({'format': 'neural-token-read-interface-v1', 'encoder': encoder_identity,
                   'tokenizer_sha256': tokenizer_sha256, 'payload': 'whole-unescaped-json-text-tokens-v1'})


def native_token_input(query, source, message, framed_text, tokenizer, binding):
    query.validate()
    q = torch.from_numpy(query.features.copy())
    if source is None:
        if message is not None or framed_text is not None: raise ValueError('inconsistent empty source')
        return TokenReadInput(q, torch.empty(0, 2 * HIDDEN), torch.empty(0, dtype=torch.long),
                              torch.empty(0, dtype=torch.bool), binding)
    source.validate()
    ids = source.token_ids[1:].tolist()
    if tokenizer.encode(framed_text, True) != source.token_ids.tolist(): raise ValueError('C source token identity mismatch')
    allowed = payload_mask(message, framed_text, tokenizer.decode_pieces(ids))
    # BOS/EOS must not become payload even if a future piece renderer changes.
    blocked = (tokenizer.bos(), tokenizer.eos())
    for i, token in enumerate(ids):
        if token in blocked: allowed[i] = False
    if any(not 0 <= token < VOCAB for token in ids): raise ValueError('vocabulary binding mismatch')
    return TokenReadInput(q, torch.from_numpy(source.features.copy()), torch.tensor(ids, dtype=torch.long),
                          torch.tensor(allowed, dtype=torch.bool), binding)
