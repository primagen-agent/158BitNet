#!/usr/bin/env python3
"""Tune span-candidate fusion with adapted neural field anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from train_typed_anchor_keys import (
    FORMAT as ANCHOR_FORMAT,
    FIELDS,
    TypedAnchorKeys,
)
from train_typed_memory_writer import (
    collate_examples,
    load_token_embedding_table,
    load_writer_examples,
)
from typed_writer_components import load_writer_and_tagger


def load_anchor_keys(path, writer, device):
    saved = torch.load(
        path, map_location="cpu", weights_only=True
    )
    if saved.get("format") != ANCHOR_FORMAT:
        raise ValueError("bad typed anchor-key checkpoint")
    keys = TypedAnchorKeys(writer.token_keys).to(device)
    keys.load_state_dict(saved["state_dict"], strict=True)
    keys.requires_grad_(False)
    keys.eval()
    return keys


@torch.inference_mode()
def collect(
    writer, tagger, keys, examples, device, batch_size,
):
    cached = {name: [] for name in FIELDS}
    for offset in range(0, len(examples), batch_size):
        selected = list(range(
            offset, min(offset + batch_size, len(examples))
        ))
        batch = collate_examples(examples, selected, device)
        output = writer(
            batch["hidden"], batch["mask"],
            batch["identity_hidden"],
        )
        token_count = batch["mask"].shape[1]
        starts = torch.arange(
            token_count, device=device
        )[:, None]
        lengths = torch.arange(
            1, tagger.max_span + 1, device=device
        )[None, :]
        ends = starts + lengths - 1
        for name in FIELDS:
            base, _ = tagger.field_output(
                output, batch["mask"], name
            )
            anchor = keys.logits(
                output, batch["mask"], name
            ).argmax(dim=-1)
            contains = (
                (anchor[:, None, None] >= starts[None, :, :])
                & (anchor[:, None, None] <= ends[None, :, :])
            )
            target = (
                batch[f"{name}_span_start"]
                * tagger.max_span
                + batch[f"{name}_span_end"]
                - batch[f"{name}_span_start"]
            )
            cached[name].append({
                "base": base.cpu(),
                "contains": contains.cpu(),
                "target": target.cpu(),
                "operation": batch["operation"].cpu(),
            })
    return cached


def predict(batch, weight):
    score = batch["base"]
    if weight == "hard":
        score = score.masked_fill(
            ~batch["contains"], -1e9
        )
    else:
        score = (
            score + float(weight)
            * batch["contains"].float()
        )
    return score.flatten(1).argmax(dim=-1)


def accuracy(cached, field, weight):
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    for batch in cached[field]:
        exact = predict(batch, weight) == batch["target"]
        for operation in (0, 1):
            chosen = batch["operation"] == operation
            correct[operation] += int(
                (exact & chosen).sum()
            )
            total[operation] += int(chosen.sum())
    return {
        "create": correct[0] / max(total[0], 1),
        "update": correct[1] / max(total[1], 1),
        "all": sum(correct.values()) / max(sum(total.values()), 1),
    }


def robust_key(primary, extra):
    values = (
        primary["create"], primary["update"],
        extra["create"], extra["update"],
    )
    return (min(values), sum(values) / len(values))


def tune(primary, extra, field):
    best = None
    for weight in (
        0.0, 0.25, 0.5, 1.0, 2.0,
        4.0, 8.0, 16.0, "hard",
    ):
        primary_metrics = accuracy(primary, field, weight)
        extra_metrics = accuracy(extra, field, weight)
        key = robust_key(primary_metrics, extra_metrics)
        candidate = (
            key,
            primary_metrics["all"] + extra_metrics["all"],
            weight,
            primary_metrics,
            extra_metrics,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return {
        "anchor_weight": best[2],
        "primary": best[3],
        "extra": best[4],
        "robust_min": best[0][0],
        "robust_average": best[0][1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("writer_checkpoint")
    parser.add_argument("tagger_checkpoint")
    parser.add_argument("anchor_checkpoint")
    parser.add_argument("primary_features")
    parser.add_argument("primary_jsonl")
    parser.add_argument("extra_features")
    parser.add_argument("extra_jsonl")
    parser.add_argument("output")
    parser.add_argument("--test-features")
    parser.add_argument("--test-jsonl")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if bool(args.test_features) != bool(args.test_jsonl):
        parser.error(
            "test features and JSONL must be supplied together"
        )

    checkpoint, writer, tagger = load_writer_and_tagger(
        args.writer_checkpoint,
        args.tagger_checkpoint,
        args.device,
    )
    keys = load_anchor_keys(
        args.anchor_checkpoint, writer, args.device
    )
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    embeddings = load_token_embedding_table(
        args.gguf, args.lib,
        checkpoint["backbone_sha256"],
        int(checkpoint["hidden"]),
    )
    def load(features, jsonl):
        return load_writer_examples(
            features, jsonl,
            checkpoint["backbone_sha256"],
            tokenizer, embeddings,
        )
    primary_examples = load(
        args.primary_features, args.primary_jsonl
    )
    extra_examples = load(
        args.extra_features, args.extra_jsonl
    )
    test_examples = (
        load(args.test_features, args.test_jsonl)
        if args.test_features else None
    )
    del embeddings
    primary = collect(
        writer, tagger, keys, primary_examples,
        args.device, args.batch,
    )
    extra = collect(
        writer, tagger, keys, extra_examples,
        args.device, args.batch,
    )
    selected = {
        name: tune(primary, extra, name)
        for name in FIELDS
    }
    test = None
    if test_examples is not None:
        test_cache = collect(
            writer, tagger, keys, test_examples,
            args.device, args.batch,
        )
        test = {
            name: accuracy(
                test_cache, name,
                selected[name]["anchor_weight"],
            )
            for name in FIELDS
        }
    result = {
        "format": "TYPED_ANCHOR_SPAN_FUSION_V1",
        "fields": selected,
        "test": test,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
