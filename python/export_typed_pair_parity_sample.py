#!/usr/bin/env python3
"""Export deterministic token-level V251 inputs and Python pair outputs."""

import argparse
import struct
from pathlib import Path

import torch

from train_typed_pair_verifier import (
    TypedPairVerifier,
)
from train_typed_memory_writer import TypedMemoryWriter


def load_typed_pair_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu",
        weights_only=True)
    if (
        checkpoint.get("format")
        != "TYPED_PAIR_VERIFIER_V1"
        or not checkpoint.get("dual_path", False)
        or not checkpoint.get(
            "separate_address_localizer", False)
        or not checkpoint.get("use_token_embeddings", False)
    ):
        raise ValueError(
            "typed-pair parity requires V251 dual-path checkpoint")
    writer = TypedMemoryWriter(
        int(checkpoint["hidden"]),
        int(checkpoint["rank"]),
        len(checkpoint["hidden_layer_bands"]),
        bool(checkpoint.get(
            "separate_address_localizer", False)),
        bool(checkpoint.get(
            "predict_span_boundaries", False)),
        bool(checkpoint.get(
            "use_token_embeddings", False)),
        bool(checkpoint.get(
            "hard_address_pooling", False)),
        bool(checkpoint.get(
            "contiguous_span_pooling", False)),
        int(checkpoint.get("max_field_span", 8)))
    writer.load_state_dict(
        checkpoint["writer_state_dict"], strict=True)
    writer.eval()
    model = TypedPairVerifier(
        writer, int(checkpoint["rank"]),
        bool(checkpoint.get(
            "attention_weighted_alignment", False)),
        bool(checkpoint.get("dual_path", False)),
        bool(checkpoint.get(
            "trainable_context_localizer", False)),
        bool(checkpoint.get("set_link_head", False)))
    missing, unexpected = model.load_state_dict(
        checkpoint["verifier_state_dict"],
        strict=False)
    if (
        any(not name.startswith("writer.") for name in missing)
        or unexpected
    ):
        raise ValueError(
            f"typed-pair parity state mismatch: "
            f"{missing} {unexpected}")
    model.eval()
    return checkpoint, model


def export_typed_pair_parity_sample(
    checkpoint_path, output_path,
    left_count=7, right_count=9,
    seed=20260915,
):
    if min(left_count, right_count) < 1:
        raise ValueError("pair token counts must be positive")
    checkpoint, model = load_typed_pair_model(
        checkpoint_path)
    hidden = int(checkpoint["hidden"])
    bands = len(checkpoint["hidden_layer_bands"])
    generator = torch.Generator().manual_seed(seed)
    left_hidden = torch.randn(
        1, left_count, bands, hidden,
        generator=generator)
    right_hidden = torch.randn(
        1, right_count, bands, hidden,
        generator=generator)
    left_identity = torch.randn(
        1, left_count, hidden,
        generator=generator)
    right_identity = torch.randn(
        1, right_count, hidden,
        generator=generator)
    with torch.inference_mode():
        result = model(
            left_hidden, right_hidden,
            torch.ones(
                1, left_count, dtype=torch.bool),
            torch.ones(
                1, right_count, dtype=torch.bool),
            left_identity, right_identity,
            return_internal=True)
    entity_feature = result["internals"][
        "pair_features"]["entity"][0]
    predicate_feature = result["internals"][
        "pair_features"]["predicate"][0]
    entity_logit = result["entity_logit"].reshape(1)
    predicate_logit = result[
        "predicate_logit"].reshape(1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray(b"BNTPARF1")
    payload += struct.pack(
        "<IIIIII",
        1,
        hidden,
        bands,
        int(entity_feature.numel()),
        left_count,
        right_count,
    )
    for tensor in (
        left_hidden, left_identity,
        right_hidden, right_identity,
        entity_feature, predicate_feature,
        entity_logit, predicate_logit,
    ):
        payload += tensor.detach().contiguous().numpy().astype(
            "<f4", copy=False).tobytes()
    output.write_bytes(payload)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--left-tokens", type=int, default=7)
    parser.add_argument("--right-tokens", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    print(export_typed_pair_parity_sample(
        args.checkpoint, args.output,
        args.left_tokens, args.right_tokens,
        args.seed))


if __name__ == "__main__":
    main()
