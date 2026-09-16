#!/usr/bin/env python3
"""Export the autonomous typed writer for the C runtime."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import torch

from typed_memory_training import file_fingerprint
from train_typed_anchor_keys import (
    FORMAT as ANCHOR_FORMAT,
)
from train_typed_memory_writer import SLOT_NAMES
from train_typed_span_tagger import (
    FORMAT as TAGGER_FORMAT,
)


MAGIC = b"BNTWRITE"
VERSION = 1


def tensor_payload(tensor, shape, name):
    value = tensor.detach().float().contiguous()
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"typed-writer tensor geometry mismatch: {name}"
        )
    return value.numpy().astype(
        "<f4", copy=False
    ).tobytes()


def export_typed_writer_binary(
    writer_path, tagger_path, anchor_path, output_path,
    predicate_anchor_weight=1.0,
    value_anchor_weight=0.25,
):
    writer = torch.load(
        writer_path, map_location="cpu", weights_only=True
    )
    tagger = torch.load(
        tagger_path, map_location="cpu", weights_only=True
    )
    anchors = torch.load(
        anchor_path, map_location="cpu", weights_only=True
    )
    if (
        writer.get("format") != "TYPED_MEMORY_WRITER_V1"
        or not writer.get("separate_address_localizer", False)
        or not writer.get("predict_span_boundaries", False)
        or not writer.get("use_token_embeddings", False)
        or writer.get("contiguous_span_pooling", False)
    ):
        raise ValueError(
            "typed-writer C export requires V257 writer geometry"
        )
    writer_fingerprint = file_fingerprint(writer_path)
    if (
        tagger.get("format") != TAGGER_FORMAT
        or anchors.get("format") != ANCHOR_FORMAT
        or tagger.get("backbone_sha256")
            != writer.get("backbone_sha256")
        or anchors.get("backbone_sha256")
            != writer.get("backbone_sha256")
        or tagger.get("writer_checkpoint_fingerprint")
            != writer_fingerprint
        or anchors.get("writer_checkpoint_fingerprint")
            != writer_fingerprint
    ):
        raise ValueError(
            "typed-writer component identity mismatch"
        )
    hidden = int(writer["hidden"])
    rank = int(writer["rank"])
    max_span = int(tagger["max_span"])
    bands = writer.get("hidden_layer_bands") or []
    flattened = [
        int(layer) for band in bands for layer in band
    ]
    if (
        len(bands) < 2
        or flattened != list(range(len(flattened)))
        or int(tagger["rank"]) != rank
        or max_span < 1
    ):
        raise ValueError(
            "typed-writer component geometry mismatch"
        )
    writer_state = writer["state_dict"]
    tagger_state = tagger["state_dict"]
    anchor_state = anchors["state_dict"]
    payloads = []
    payloads.append(tensor_payload(
        writer_state["band_logits"],
        (len(SLOT_NAMES), len(bands)),
        "band_logits",
    ))
    for name in SLOT_NAMES:
        payloads.append(tensor_payload(
            writer_state[f"projections.{name}.weight"],
            (rank, hidden),
            f"projections.{name}.weight",
        ))
    payloads.append(tensor_payload(
        writer_state["localizer_band_logits"],
        (2, len(bands)),
        "localizer_band_logits",
    ))
    for name in ("entity", "predicate"):
        payloads.append(tensor_payload(
            writer_state[
                f"localizer_projections.{name}.weight"
            ],
            (rank, hidden),
            f"localizer_projections.{name}.weight",
        ))
    payloads.append(tensor_payload(
        writer_state["token_keys"],
        (len(SLOT_NAMES), rank),
        "token_keys",
    ))
    payloads.append(tensor_payload(
        writer_state["span_boundary_keys"],
        (len(SLOT_NAMES), 2, rank),
        "span_boundary_keys",
    ))
    payloads.extend((
        tensor_payload(
            writer_state["operation_head.0.weight"],
            (rank, rank * len(SLOT_NAMES)),
            "operation_head.0.weight",
        ),
        tensor_payload(
            writer_state["operation_head.0.bias"],
            (rank,), "operation_head.0.bias",
        ),
        tensor_payload(
            writer_state["operation_head.2.weight"],
            (2, rank), "operation_head.2.weight",
        ),
        tensor_payload(
            writer_state["operation_head.2.bias"],
            (2,), "operation_head.2.bias",
        ),
    ))
    adapted = torch.stack((
        anchor_state["keys.predicate"],
        anchor_state["keys.value"],
    ))
    payloads.append(tensor_payload(
        adapted, (2, rank), "adapted_anchor_keys"
    ))
    for name in SLOT_NAMES:
        prefix = f"fields.{name}."
        payloads.extend((
            tensor_payload(
                tagger_state[prefix + "context.weight"],
                (rank, rank, 3),
                prefix + "context.weight",
            ),
            tensor_payload(
                tagger_state[prefix + "context.bias"],
                (rank,), prefix + "context.bias",
            ),
            tensor_payload(
                tagger_state[prefix + "output.weight"],
                (1, rank), prefix + "output.weight",
            ),
            tensor_payload(
                tagger_state[prefix + "output.bias"],
                (1,), prefix + "output.bias",
            ),
            tensor_payload(
                tagger_state[prefix + "length_logits"],
                (max_span,), prefix + "length_logits",
            ),
            tensor_payload(
                tagger_state[prefix + "residual_scale"],
                (), prefix + "residual_scale",
            ),
        ))
    header = bytearray(MAGIC)
    header += struct.pack(
        "<IIIIIIIff",
        VERSION,
        hidden,
        rank,
        len(bands),
        len(flattened),
        max_span,
        len(payloads),
        float(predicate_anchor_weight),
        float(value_anchor_weight),
    )
    for band in bands:
        header += struct.pack(
            "<II", int(band[0]), int(band[-1])
        )
    header += bytes.fromhex(writer["backbone_sha256"])
    for payload in payloads:
        header += struct.pack(
            "<II",
            len(payload),
            zlib.crc32(payload) & 0xFFFFFFFF,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + b"".join(payloads))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("writer_checkpoint")
    parser.add_argument("tagger_checkpoint")
    parser.add_argument("anchor_checkpoint")
    parser.add_argument("output")
    parser.add_argument(
        "--predicate-anchor-weight",
        type=float, default=1.0,
    )
    parser.add_argument(
        "--value-anchor-weight",
        type=float, default=0.25,
    )
    args = parser.parse_args()
    print(export_typed_writer_binary(
        args.writer_checkpoint,
        args.tagger_checkpoint,
        args.anchor_checkpoint,
        args.output,
        args.predicate_anchor_weight,
        args.value_anchor_weight,
    ))


if __name__ == "__main__":
    main()
