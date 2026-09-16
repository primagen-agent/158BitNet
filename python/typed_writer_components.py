"""Shared loading for frozen typed-writer Python components."""

from __future__ import annotations

import torch

from train_typed_memory_writer import TypedMemoryWriter
from train_typed_span_tagger import (
    FORMAT as TAGGER_FORMAT,
    TypedSpanTagger,
)


def load_writer_and_tagger(
    writer_path, tagger_path, device,
):
    writer_checkpoint = torch.load(
        writer_path, map_location="cpu", weights_only=True
    )
    if (
        writer_checkpoint.get("format")
        != "TYPED_MEMORY_WRITER_V1"
    ):
        raise ValueError("bad typed writer checkpoint")
    writer = TypedMemoryWriter(
        int(writer_checkpoint["hidden"]),
        int(writer_checkpoint["rank"]),
        len(writer_checkpoint["hidden_layer_bands"]),
        bool(writer_checkpoint.get(
            "separate_address_localizer", False)),
        bool(writer_checkpoint.get(
            "predict_span_boundaries", False)),
        bool(writer_checkpoint.get(
            "use_token_embeddings", False)),
        bool(writer_checkpoint.get(
            "hard_address_pooling", False)),
        bool(writer_checkpoint.get(
            "contiguous_span_pooling", False)),
        int(writer_checkpoint.get("max_field_span", 8)),
    )
    writer.load_state_dict(
        writer_checkpoint["state_dict"], strict=True
    )
    writer.requires_grad_(False)
    writer.eval().to(device)

    saved = torch.load(
        tagger_path, map_location="cpu", weights_only=True
    )
    if saved.get("format") != TAGGER_FORMAT:
        raise ValueError("bad typed span tagger checkpoint")
    if (
        saved.get("backbone_sha256")
        != writer_checkpoint["backbone_sha256"]
    ):
        raise ValueError("tagger backbone identity mismatch")
    boundary = writer_checkpoint[
        "state_dict"
    ]["span_boundary_keys"]
    tagger = TypedSpanTagger(
        int(saved["rank"]), int(saved["max_span"]), boundary
    )
    tagger.load_state_dict(saved["state_dict"], strict=True)
    tagger.requires_grad_(False)
    tagger.eval().to(device)
    return writer_checkpoint, writer, tagger
