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
    return unique[0] if len(unique) == 1 else None


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
