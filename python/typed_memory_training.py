"""Shared training utilities for the typed-event memory pipeline."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def reject_evaluation_row(row):
    metadata = row.get("metadata") or {}
    if (row.get("evaluation_only") or metadata.get("evaluation_only")
            or metadata.get("locomo_used")):
        raise ValueError("evaluation-only data cannot enter training")


def clone_state_dict(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_unique_token_span(source_ids, target_ids):
    source = [int(token) for token in source_ids]
    target = [int(token) for token in target_ids]
    if not target or len(target) > len(source):
        return None
    matches = [
        start
        for start in range(len(source) - len(target) + 1)
        if source[start:start + len(target)] == target
    ]
    if len(matches) != 1:
        return None
    return matches[0], matches[0] + len(target) - 1


def token_span_variants(source_ids, surface, tokenizer):
    variants = []
    stripped = surface.strip()
    for text in (surface, stripped, " " + stripped):
        if not text or text in variants:
            continue
        variants.append(text)
    matches = []
    for text in variants:
        target_ids = tokenizer.encode(text, add_bos=False)
        span = find_unique_token_span(source_ids, target_ids)
        if span is not None:
            matches.append(span)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    # Standalone encoding can prepend a dummy space absent after punctuation.
    # Align against the already encoded sentence, never approximate token ids.
    if not hasattr(tokenizer, "decode_pieces") or not stripped:
        return None
    pieces = tokenizer.decode_pieces(source_ids)
    if source_ids and int(source_ids[0]) == tokenizer.bos():
        pieces[0] = b""
    decoded = b"".join(pieces)
    target = stripped.encode("utf-8")
    spans = []
    offset = 0
    for index, piece in enumerate(pieces):
        spans.append((offset, offset + len(piece), index))
        offset += len(piece)
    found = []
    begin = decoded.find(target)
    def ascii_word(byte):
        return byte < 128 and (chr(byte).isalnum() or byte == ord("_"))
    while begin >= 0:
        end = begin + len(target)
        left_ok = not (begin and ascii_word(target[0]) and ascii_word(decoded[begin - 1]))
        right_ok = not (end < len(decoded) and ascii_word(target[-1]) and ascii_word(decoded[end]))
        tokens = [index for start, stop, index in spans if stop > start and start < end and stop > begin]
        if left_ok and right_ok and tokens:
            first, last = tokens[0], tokens[-1]
            # Do not supervise a span whose token boundaries include extra text.
            if b"".join(pieces[first:last + 1]).strip() == target:
                found.append((first, last))
        begin = decoded.find(target, begin + 1)
    return found[0] if len(found) == 1 else None


def parse_hidden_layer_bands(spec, n_layers):
    if not spec:
        return ()
    bands = []
    for item in str(spec).split(","):
        bounds = item.strip().split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 0 or end < start or end >= n_layers:
            raise ValueError(
                f"invalid hidden-layer band: {item}")
        bands.append(tuple(range(start, end + 1)))
    flattened = tuple(
        layer for band in bands for layer in band)
    if flattened != tuple(range(n_layers)):
        raise ValueError(
            "hidden-layer bands must cover every backbone layer "
            "exactly once and in order")
    if len(bands) < 2:
        raise ValueError(
            "hidden-layer band mode requires at least two bands")
    return tuple(bands)


def encode_backbone_features(backbone, token_ids, layer_bands):
    tokens = torch.tensor(
        token_ids, device=backbone.device, dtype=torch.long)
    if not layer_bands:
        return backbone(
            tokens, return_hidden=True
        ).detach().cpu().to(torch.float16)
    requested_layers = tuple(
        layer for band in layer_bands for layer in band)
    hidden = backbone(
        tokens,
        return_hidden_layers=requested_layers,
    ).reshape(
        len(token_ids), len(requested_layers),
        backbone.cfg.hidden,
    ).float()
    hidden = hidden * torch.rsqrt(
        hidden.square().mean(
            dim=-1, keepdim=True).clamp_min(1e-12))
    band_features = []
    offset = 0
    for band in layer_bands:
        count = len(band)
        band_features.append(
            hidden[:, offset:offset + count].mean(dim=1))
        offset += count
    return torch.stack(
        band_features, dim=1
    ).detach().cpu().to(torch.float16)
