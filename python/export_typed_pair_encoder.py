#!/usr/bin/env python3
"""Export the frozen V251 token-level pair encoder for the C runtime."""

import argparse
import struct
import zlib
from pathlib import Path

import torch


PAIR_ENCODER_TENSORS = (
    ("band_logits", "verifier"),
    ("projections.entity.weight", "verifier"),
    ("projections.predicate.weight", "verifier"),
    ("identity_projections.entity.weight", "verifier"),
    ("identity_projections.predicate.weight", "verifier"),
    ("localizer_band_logits", "writer"),
    ("localizer_projections.entity.weight", "writer"),
    ("localizer_projections.predicate.weight", "writer"),
    ("token_keys", "derived"),
    ("fusion_gates.entity.0.weight", "verifier"),
    ("fusion_gates.entity.0.bias", "verifier"),
    ("fusion_gates.entity.2.weight", "verifier"),
    ("fusion_gates.entity.2.bias", "verifier"),
    ("fusion_gates.predicate.0.weight", "verifier"),
    ("fusion_gates.predicate.0.bias", "verifier"),
    ("fusion_gates.predicate.2.weight", "verifier"),
    ("fusion_gates.predicate.2.bias", "verifier"),
    ("entity_head.0.weight", "verifier"),
    ("entity_head.0.bias", "verifier"),
    ("entity_head.2.weight", "verifier"),
    ("entity_head.2.bias", "verifier"),
    ("predicate_head.0.weight", "verifier"),
    ("predicate_head.0.bias", "verifier"),
    ("predicate_head.2.weight", "verifier"),
    ("predicate_head.2.bias", "verifier"),
)


def export_typed_pair_encoder_binary(
    checkpoint_path, output_path,
):
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
        or checkpoint.get(
            "attention_weighted_alignment", False)
        or checkpoint.get(
            "trainable_context_localizer", False)
    ):
        raise ValueError(
            "typed-pair C export requires frozen V251 dual-path geometry")
    hidden = int(checkpoint["hidden"])
    rank = int(checkpoint["rank"])
    bands = checkpoint.get("hidden_layer_bands") or []
    flattened = [
        int(layer) for band in bands for layer in band]
    if (
        len(bands) < 2
        or flattened != list(range(len(flattened)))
    ):
        raise ValueError(
            "typed-pair export requires contiguous full-layer bands")
    pair_width = rank * 2 + 4
    head_width = pair_width + 1
    verifier = checkpoint["verifier_state_dict"]
    writer = checkpoint["writer_state_dict"]
    derived = {
        "token_keys": writer["token_keys"][:2],
    }
    expected = {
        "band_logits": (2, len(bands)),
        "projections.entity.weight": (rank, hidden),
        "projections.predicate.weight": (rank, hidden),
        "identity_projections.entity.weight": (rank, hidden),
        "identity_projections.predicate.weight": (rank, hidden),
        "localizer_band_logits": (2, len(bands)),
        "localizer_projections.entity.weight": (rank, hidden),
        "localizer_projections.predicate.weight": (rank, hidden),
        "token_keys": (2, rank),
        "fusion_gates.entity.0.weight": (rank, pair_width * 2),
        "fusion_gates.entity.0.bias": (rank,),
        "fusion_gates.entity.2.weight": (1, rank),
        "fusion_gates.entity.2.bias": (1,),
        "fusion_gates.predicate.0.weight": (rank, pair_width * 2),
        "fusion_gates.predicate.0.bias": (rank,),
        "fusion_gates.predicate.2.weight": (1, rank),
        "fusion_gates.predicate.2.bias": (1,),
        "entity_head.0.weight": (rank * 2, head_width),
        "entity_head.0.bias": (rank * 2,),
        "entity_head.2.weight": (1, rank * 2),
        "entity_head.2.bias": (1,),
        "predicate_head.0.weight": (rank * 2, head_width),
        "predicate_head.0.bias": (rank * 2,),
        "predicate_head.2.weight": (1, rank * 2),
        "predicate_head.2.bias": (1,),
    }
    payloads = []
    for name, source in PAIR_ENCODER_TENSORS:
        table = (
            verifier if source == "verifier"
            else writer if source == "writer"
            else derived)
        if name not in table:
            raise ValueError(
                f"typed-pair tensor missing: {name}")
        tensor = table[name].detach().float().contiguous()
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"typed-pair tensor geometry mismatch: {name}")
        payloads.append(
            tensor.numpy().astype(
                "<f4", copy=False).tobytes())
    header = bytearray(b"BNTPAIR1")
    header += struct.pack(
        "<IIIIIIII",
        1,
        hidden,
        rank,
        len(bands),
        len(flattened),
        pair_width,
        head_width,
        len(payloads),
    )
    for band in bands:
        header += struct.pack(
            "<II", int(band[0]), int(band[-1]))
    header += bytes.fromhex(
        checkpoint["backbone_sha256"])
    for payload in payloads:
        header += struct.pack(
            "<II", len(payload),
            zlib.crc32(payload) & 0xFFFFFFFF)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + b"".join(payloads))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    print(export_typed_pair_encoder_binary(
        args.checkpoint, args.output))


if __name__ == "__main__":
    main()
