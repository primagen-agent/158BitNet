#!/usr/bin/env python3
"""Train field activation keys while freezing all writer projections."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from c_tokenizer import CTokenizer
from typed_memory_training import (
    clone_state_dict,
    file_fingerprint,
)
from train_typed_memory_writer import (
    SLOT_NAMES,
    collate_examples,
    load_token_embedding_table,
    load_writer_examples,
)
from train_typed_span_tagger import (
    stratified_sample_indices,
)
from typed_writer_components import load_writer_and_tagger


FORMAT = "TYPED_ANCHOR_KEYS_V1"
FIELDS = ("predicate", "value")


class TypedAnchorKeys(nn.Module):
    def __init__(self, initial_keys):
        super().__init__()
        self.keys = nn.ParameterDict({
            name: nn.Parameter(
                initial_keys[
                    SLOT_NAMES.index(name)
                ].detach().float().clone()
            )
            for name in FIELDS
        })

    def logits(self, writer_output, mask, name):
        return torch.einsum(
            "btr,r->bt",
            writer_output["score_tokens"][name],
            self.keys[name],
        ).masked_fill(~mask, -1e9)


def anchor_loss(logits, targets, mask):
    attention = F.softmax(logits, dim=-1)
    target_mass = (
        attention * targets.float()
    ).sum(dim=-1)
    mass_loss = -target_mass.clamp_min(1e-9).log().mean()
    positive = targets & mask
    negative = ~targets & mask
    token_loss = 0.5 * (
        F.softplus(-logits[positive]).mean()
        + F.softplus(logits[negative]).mean()
    )
    return 0.8 * mass_loss + 0.2 * token_loss


@torch.inference_mode()
def evaluate(
    writer, keys, examples, device, batch_size,
):
    result = {}
    totals = {0: 0, 1: 0}
    correct = {
        name: {0: 0, 1: 0}
        for name in FIELDS
    }
    keys.eval()
    for offset in range(0, len(examples), batch_size):
        selected = list(range(
            offset, min(offset + batch_size, len(examples))
        ))
        batch = collate_examples(examples, selected, device)
        output = writer(
            batch["hidden"], batch["mask"],
            batch["identity_hidden"],
        )
        for operation in (0, 1):
            totals[operation] += int(
                (batch["operation"] == operation).sum()
            )
        for name in FIELDS:
            anchor = keys.logits(
                output, batch["mask"], name
            ).argmax(dim=-1)
            start = batch[f"{name}_span_start"]
            end = batch[f"{name}_span_end"]
            exact = (anchor >= start) & (anchor <= end)
            for operation in (0, 1):
                chosen = batch["operation"] == operation
                correct[name][operation] += int(
                    (exact & chosen).sum()
                )
    for name in FIELDS:
        result[name] = {
            "create": (
                correct[name][0] / max(totals[0], 1)
            ),
            "update": (
                correct[name][1] / max(totals[1], 1)
            ),
            "all": (
                sum(correct[name].values())
                / max(sum(totals.values()), 1)
            ),
        }
    return result


def field_key(primary, extra, name):
    values = (
        primary[name]["create"],
        primary[name]["update"],
        extra[name]["create"],
        extra[name]["update"],
    )
    return (min(values), sum(values) / len(values))


def train(
    writer, keys,
    train_examples, primary_examples, extra_examples,
    device, steps, batch_size, learning_rate,
    weight_decay, eval_every, patience,
    create_fraction, seed,
):
    optimizer = torch.optim.AdamW(
        keys.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = random.Random(seed)
    primary = evaluate(
        writer, keys, primary_examples, device, batch_size
    )
    extra = evaluate(
        writer, keys, extra_examples, device, batch_size
    )
    best_scores = {
        name: field_key(primary, extra, name)
        for name in FIELDS
    }
    best_steps = {name: 0 for name in FIELDS}
    best_keys = {
        name: keys.keys[name].detach().cpu().clone()
        for name in FIELDS
    }
    stale = 0
    print(json.dumps({
        "phase": "typed_anchor_keys_baseline",
        "primary": primary,
        "extra": extra,
    }, separators=(",", ":")), flush=True)
    for step in range(1, steps + 1):
        selected = stratified_sample_indices(
            train_examples, batch_size,
            create_fraction, rng,
        )
        rng.shuffle(selected)
        batch = collate_examples(
            train_examples, selected, device
        )
        with torch.no_grad():
            output = writer(
                batch["hidden"], batch["mask"],
                batch["identity_hidden"],
            )
        keys.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {
            name: anchor_loss(
                keys.logits(output, batch["mask"], name),
                batch[f"{name}_token_target"],
                batch["mask"],
            )
            for name in FIELDS
        }
        loss = sum(losses.values()) / len(losses)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            keys.parameters(), 1.0
        )
        optimizer.step()
        if step % eval_every and step != steps:
            continue
        primary = evaluate(
            writer, keys, primary_examples,
            device, batch_size,
        )
        extra = evaluate(
            writer, keys, extra_examples,
            device, batch_size,
        )
        print(json.dumps({
            "phase": "typed_anchor_keys_valid",
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "loss_parts": {
                name: float(value.detach())
                for name, value in losses.items()
            },
            "primary": primary,
            "extra": extra,
        }, separators=(",", ":")), flush=True)
        improved = False
        for name in FIELDS:
            score = field_key(primary, extra, name)
            if score > best_scores[name]:
                best_scores[name] = score
                best_steps[name] = step
                best_keys[name] = (
                    keys.keys[name].detach().cpu().clone()
                )
                improved = True
        stale = 0 if improved else stale + 1
        if stale >= patience:
            break
    with torch.no_grad():
        for name in FIELDS:
            keys.keys[name].copy_(
                best_keys[name].to(device)
            )
    return (
        evaluate(
            writer, keys, primary_examples,
            device, batch_size,
        ),
        evaluate(
            writer, keys, extra_examples,
            device, batch_size,
        ),
        best_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("writer_checkpoint")
    parser.add_argument("tagger_checkpoint")
    parser.add_argument("train_features")
    parser.add_argument("train_jsonl")
    parser.add_argument("primary_features")
    parser.add_argument("primary_jsonl")
    parser.add_argument("extra_features")
    parser.add_argument("extra_jsonl")
    parser.add_argument("output")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--test-features")
    parser.add_argument("--test-jsonl")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--create-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if bool(args.test_features) != bool(args.test_jsonl):
        parser.error(
            "test features and JSONL must be supplied together"
        )

    checkpoint, writer, _ = load_writer_and_tagger(
        args.writer_checkpoint,
        args.tagger_checkpoint,
        args.device,
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
    train_examples = load(
        args.train_features, args.train_jsonl
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
    keys = TypedAnchorKeys(
        writer.token_keys
    ).to(args.device)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if initial.get("format") != FORMAT:
            raise ValueError(
                "bad typed anchor-key init checkpoint"
            )
        if (
            initial.get("backbone_sha256")
            != checkpoint["backbone_sha256"]
        ):
            raise ValueError(
                "anchor-key backbone identity mismatch"
            )
        if (
            initial.get("writer_checkpoint_fingerprint")
            != file_fingerprint(args.writer_checkpoint)
        ):
            raise ValueError(
                "anchor-key writer identity mismatch"
            )
        keys.load_state_dict(
            initial["state_dict"], strict=True
        )
    primary, extra, best_steps = train(
        writer, keys,
        train_examples, primary_examples, extra_examples,
        args.device, args.steps, args.batch,
        args.learning_rate, args.weight_decay,
        args.eval_every, args.patience,
        args.create_fraction, args.seed,
    )
    test = (
        evaluate(
            writer, keys, test_examples,
            args.device, args.batch,
        )
        if test_examples is not None else None
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT,
        "backbone_sha256": checkpoint["backbone_sha256"],
        "writer_checkpoint_fingerprint":
            file_fingerprint(args.writer_checkpoint),
        "state_dict": clone_state_dict(keys),
        "selected_steps": best_steps,
        "primary_metrics": primary,
        "extra_metrics": extra,
        "test_metrics": test,
        "writer_frozen": True,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, output)
    print(json.dumps({
        "phase": "typed_anchor_keys_done",
        "output": str(output),
        "selected_steps": best_steps,
        "primary": primary,
        "extra": extra,
        "test": test,
        "writer_frozen": True,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
