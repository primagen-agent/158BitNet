#!/usr/bin/env python3
"""Export one Python typed-writer inference for C parity testing."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from train_typed_anchor_keys import TypedAnchorKeys
from train_typed_memory_writer import (
    SLOT_NAMES,
    collate_examples,
    load_token_embedding_table,
    load_writer_examples,
)
from typed_writer_components import load_writer_and_tagger


def export_sample(
    writer_path, tagger_path, anchor_path,
    feature_path, jsonl_path, gguf, lib, tok_probe,
    example_index, output_path,
    predicate_anchor_weight=1.0,
    value_anchor_weight=0.25,
    device="cpu",
):
    checkpoint, writer, tagger = load_writer_and_tagger(
        writer_path, tagger_path, device
    )
    anchor_checkpoint = torch.load(
        anchor_path, map_location="cpu", weights_only=True
    )
    keys = TypedAnchorKeys(writer.token_keys).to(device)
    keys.load_state_dict(
        anchor_checkpoint["state_dict"], strict=True
    )
    keys.eval()
    tokenizer = CTokenizer(tok_probe, gguf)
    embeddings = load_token_embedding_table(
        gguf, lib, checkpoint["backbone_sha256"],
        int(checkpoint["hidden"]),
    )
    examples = load_writer_examples(
        feature_path, jsonl_path,
        checkpoint["backbone_sha256"],
        tokenizer, embeddings,
    )
    if not 0 <= example_index < len(examples):
        raise ValueError("parity example index is out of range")
    batch = collate_examples(
        examples, [example_index], device
    )
    with torch.inference_mode():
        writer_output = writer(
            batch["hidden"], batch["mask"],
            batch["identity_hidden"],
        )
        operation_logits = writer_output[
            "operation_logits"
        ][0].float().cpu()
        operation = int(operation_logits.argmax())
        spans = {}
        anchors = {}
        for name in SLOT_NAMES:
            scores, _ = tagger.field_output(
                writer_output, batch["mask"], name
            )
            if name in ("predicate", "value"):
                anchor = int(keys.logits(
                    writer_output, batch["mask"], name
                )[0].argmax())
                weight = (
                    predicate_anchor_weight
                    if name == "predicate"
                    else value_anchor_weight
                )
                token_count = scores.shape[1]
                starts = torch.arange(
                    token_count, device=device
                )[:, None]
                lengths = torch.arange(
                    1, tagger.max_span + 1,
                    device=device,
                )[None, :]
                ends = starts + lengths - 1
                contains = (
                    (anchor >= starts)
                    & (anchor <= ends)
                )
                scores = (
                    scores
                    + weight * contains[None].float()
                )
            else:
                anchor = int(writer_output[
                    "attentions"
                ][name][0].argmax())
            selected = int(scores[0].flatten().argmax())
            start = selected // tagger.max_span
            length = selected % tagger.max_span + 1
            spans[name] = (start, start + length - 1)
            anchors[name] = anchor
    hidden = batch["hidden"][0].float().cpu().contiguous()
    identity = batch[
        "identity_hidden"
    ][0].float().cpu().contiguous()
    token_count = int(batch["mask"][0].sum())
    hidden = hidden[:token_count]
    identity = identity[:token_count]
    payload = bytearray(b"BNTWPAR1")
    payload += struct.pack(
        "<IIIIIff",
        1,
        int(checkpoint["hidden"]),
        len(checkpoint["hidden_layer_bands"]),
        token_count,
        operation,
        float(operation_logits[0]),
        float(operation_logits[1]),
    )
    for name in SLOT_NAMES:
        payload += struct.pack(
            "<III",
            spans[name][0], spans[name][1], anchors[name],
        )
    payload += hidden.numpy().astype(
        "<f4", copy=False
    ).tobytes()
    payload += identity.numpy().astype(
        "<f4", copy=False
    ).tobytes()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("writer_checkpoint")
    parser.add_argument("tagger_checkpoint")
    parser.add_argument("anchor_checkpoint")
    parser.add_argument("features")
    parser.add_argument("jsonl")
    parser.add_argument("gguf")
    parser.add_argument("output")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--example-index", type=int, default=47)
    parser.add_argument(
        "--predicate-anchor-weight",
        type=float, default=1.0,
    )
    parser.add_argument(
        "--value-anchor-weight",
        type=float, default=0.25,
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(export_sample(
        args.writer_checkpoint,
        args.tagger_checkpoint,
        args.anchor_checkpoint,
        args.features,
        args.jsonl,
        args.gguf,
        args.lib,
        args.tok_probe,
        args.example_index,
        args.output,
        args.predicate_anchor_weight,
        args.value_anchor_weight,
        args.device,
    ))


if __name__ == "__main__":
    main()
