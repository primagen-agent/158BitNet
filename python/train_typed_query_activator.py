#!/usr/bin/env python3
"""Train query-to-active-event ranking and explicit NULL on V251 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from export_typed_pair_parity_sample import (
    load_typed_pair_model,
)
from typed_memory_training import (
    clone_state_dict,
    file_fingerprint,
)
from train_typed_memory_writer import (
    load_token_embedding_table,
)


PAIR_FEATURE_CACHE = "TYPED_QUERY_PAIR_FEATURES_V1"
QUERY_ACTIVATOR = "TYPED_QUERY_ACTIVATOR_V1"
SET_FEATURE_COUNT = 10
QUERY_TENSORS = (
    "candidate_head.0.weight",
    "candidate_head.0.bias",
    "candidate_head.2.weight",
    "candidate_head.2.bias",
    "null_head.0.weight",
    "null_head.0.bias",
    "null_head.2.weight",
    "null_head.2.bias",
)


def export_typed_query_activator_binary(
    checkpoint_path, pair_checkpoint_path,
    pair_binary_path, output_path,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("format") != QUERY_ACTIVATOR:
        raise ValueError(
            "typed-query export requires V1 checkpoint"
        )
    pair_fingerprint = file_fingerprint(
        pair_checkpoint_path
    )
    if (
        checkpoint.get("pair_checkpoint_fingerprint")
        != pair_fingerprint
    ):
        raise ValueError(
            "typed-query pair checkpoint mismatch"
        )
    pair_binary_fingerprint = file_fingerprint(
        pair_binary_path
    )
    state = checkpoint["state_dict"]
    rank = int(checkpoint["rank"])
    input_width = int(checkpoint["input_width"])
    set_width = int(checkpoint["set_feature_count"])
    candidate_hidden = rank * 2
    expected = {
        "candidate_head.0.weight":
            (candidate_hidden, input_width),
        "candidate_head.0.bias": (candidate_hidden,),
        "candidate_head.2.weight":
            (1, candidate_hidden),
        "candidate_head.2.bias": (1,),
        "null_head.0.weight": (rank, set_width),
        "null_head.0.bias": (rank,),
        "null_head.2.weight": (1, rank),
        "null_head.2.bias": (1,),
    }
    payloads = []
    for name in QUERY_TENSORS:
        tensor = state[name].detach().float().contiguous()
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"typed-query tensor geometry mismatch: {name}"
            )
        payloads.append(
            tensor.numpy().astype(
                "<f4", copy=False
            ).tobytes()
        )
    header = bytearray(b"BNTQACT1")
    header += struct.pack(
        "<IIIIIII",
        1,
        rank,
        input_width,
        candidate_hidden,
        set_width,
        rank,
        len(payloads),
    )
    header += bytes.fromhex(
        checkpoint["backbone_sha256"]
    )
    header += bytes.fromhex(pair_binary_fingerprint)
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


def pad_hidden(values, device):
    maximum = max(value.shape[0] for value in values)
    shape = tuple(values[0].shape[1:])
    output = torch.zeros(
        len(values), maximum, *shape,
        device=device, dtype=torch.float32,
    )
    mask = torch.zeros(
        len(values), maximum,
        device=device, dtype=torch.bool,
    )
    for index, value in enumerate(values):
        count = value.shape[0]
        output[index, :count] = value.to(
            device=device, dtype=torch.float32
        )
        mask[index, :count] = True
    return output, mask


def pad_identity(token_rows, embedding_table, device):
    maximum = max(len(tokens) for tokens in token_rows)
    hidden = embedding_table.shape[1]
    output = torch.zeros(
        len(token_rows), maximum, hidden,
        device=device, dtype=torch.float32,
    )
    for index, tokens in enumerate(token_rows):
        value = embedding_table[
            torch.tensor(tokens, dtype=torch.long)
        ]
        output[index, :len(tokens)] = value.to(
            device=device, dtype=torch.float32
        )
    return output


@torch.inference_mode()
def compile_pair_row(
    pair_model, row, embedding_table,
    device, pair_batch,
):
    query_hidden = row["query_hidden"]
    query_ids = [int(token) for token in row["query_ids"]]
    feature_rows = []
    entity_rows = []
    predicate_rows = []
    for offset in range(
        0, len(row["episode_hidden"]), pair_batch
    ):
        right_values = row["episode_hidden"][
            offset:offset + pair_batch
        ]
        right_ids = row["episode_ids"][
            offset:offset + pair_batch
        ]
        count = len(right_values)
        left_values = [query_hidden] * count
        left_ids = [query_ids] * count
        left, left_mask = pad_hidden(
            left_values, device
        )
        right, right_mask = pad_hidden(
            right_values, device
        )
        left_identity = pad_identity(
            left_ids, embedding_table, device
        )
        right_identity = pad_identity(
            right_ids, embedding_table, device
        )
        output = pair_model(
            left, right, left_mask, right_mask,
            left_identity, right_identity,
            return_internal=True,
        )
        pair_features = output[
            "internals"
        ]["pair_features"]
        feature_rows.append(torch.cat(
            (
                pair_features["entity"],
                pair_features["predicate"],
            ),
            dim=-1,
        ).half().cpu())
        entity_rows.append(
            output["entity_logit"].half().cpu()
        )
        predicate_rows.append(
            output["predicate_logit"].half().cpu()
        )
    gold = row.get("gold_episode_indices") or []
    if not row["null_target"] and len(gold) != 1:
        raise ValueError(
            f"typed current row needs one target: "
            f"{row.get('sample_id', '')}"
        )
    return {
        "sample_id": row.get("sample_id", ""),
        "world_id": row.get("world_id", ""),
        "pair_features": torch.cat(feature_rows),
        "entity_logits": torch.cat(entity_rows),
        "predicate_logits": torch.cat(predicate_rows),
        "target": -1 if row["null_target"] else int(gold[0]),
        "null_target": bool(row["null_target"]),
    }


def load_or_compile(
    source_cache, output_cache, pair_checkpoint,
    pair_model, embedding_table, device, pair_batch,
    expected_backbone=None,
):
    source = torch.load(
        source_cache, map_location="cpu",
        weights_only=True,
    )
    if source.get("evaluation_only"):
        raise ValueError("evaluation-only features cannot enter training")
    if expected_backbone is not None and source.get("backbone_sha256") != expected_backbone:
        raise ValueError("query feature cache backbone mismatch")
    metadata = {
        "format": PAIR_FEATURE_CACHE,
        "source_fingerprint":
            file_fingerprint(source_cache),
        "pair_checkpoint_fingerprint":
            file_fingerprint(pair_checkpoint),
        "backbone_sha256":
            source["backbone_sha256"],
    }
    output = Path(output_cache)
    if output.exists():
        cached = torch.load(
            output, map_location="cpu",
            weights_only=True,
        )
        if all(
            cached.get(key) == value
            for key, value in metadata.items()
        ):
            print(json.dumps({
                "phase": "typed_query_pair_cache_hit",
                "path": str(output),
                "rows": len(cached["rows"]),
            }, separators=(",", ":")), flush=True)
            return cached["rows"]
    rows = []
    for index, row in enumerate(source["rows"], 1):
        rows.append(compile_pair_row(
            pair_model, row, embedding_table,
            device, pair_batch,
        ))
        if index % 16 == 0 or index == len(source["rows"]):
            print(json.dumps({
                "phase": "typed_query_pair_compile",
                "encoded": index,
                "total": len(source["rows"]),
            }, separators=(",", ":")), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "rows": rows}, output)
    return rows


def set_features(
    scores, entity_logits, predicate_logits,
):
    if scores.numel() == 0:
        raise ValueError("query activation needs candidates")
    top = torch.topk(
        scores, min(2, scores.numel())
    ).values
    top1 = top[0]
    top2 = top[1] if top.numel() > 1 else top1
    mean = scores.mean()
    std = scores.std(unbiased=False).clamp_min(1e-4)
    normalized = (scores - mean) / std
    probability = F.softmax(normalized, dim=0)
    entropy = -(
        probability
        * probability.clamp_min(1e-9).log()
    ).sum()
    entropy = (
        entropy / math.log(scores.numel())
        if scores.numel() > 1
        else entropy * 0.0
    )
    return torch.stack((
        top1,
        top2,
        top1 - top2,
        mean,
        std,
        scores.min(),
        entropy,
        scores.new_tensor(math.log1p(scores.numel())),
        entity_logits.max(),
        predicate_logits.max(),
    ))


class TypedQueryActivator(nn.Module):
    def __init__(self, pair_checkpoint):
        super().__init__()
        rank = int(pair_checkpoint["rank"])
        state = pair_checkpoint["verifier_state_dict"]
        input_width = int(
            state["joint_head.0.weight"].shape[1]
        )
        hidden_width = int(
            state["joint_head.0.weight"].shape[0]
        )
        self.rank = rank
        self.input_width = input_width
        self.candidate_head = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        with torch.no_grad():
            self.candidate_head[0].weight.copy_(
                state["joint_head.0.weight"]
            )
            self.candidate_head[0].bias.copy_(
                state["joint_head.0.bias"]
            )
            self.candidate_head[2].weight.copy_(
                state["joint_head.2.weight"]
            )
            self.candidate_head[2].bias.copy_(
                state["joint_head.2.bias"]
            )
        self.null_head = nn.Sequential(
            nn.Linear(SET_FEATURE_COUNT, rank),
            nn.GELU(),
            nn.Linear(rank, 1),
        )

    def logits(self, row, device):
        features = row["pair_features"].to(
            device=device, dtype=torch.float32
        )
        entity = row["entity_logits"].to(
            device=device, dtype=torch.float32
        )
        predicate = row["predicate_logits"].to(
            device=device, dtype=torch.float32
        )
        scores = (
            torch.minimum(entity, predicate)
            + self.candidate_head(
                features
            ).squeeze(-1)
        )
        null = self.null_head(
            set_features(scores, entity, predicate)
        ).squeeze(-1)
        return torch.cat((scores, null.unsqueeze(0)))


def row_loss(model, row, device, margin):
    logits = model.logits(row, device)
    null_index = logits.numel() - 1
    target = (
        null_index
        if row["null_target"]
        else int(row["target"])
    )
    classification = F.cross_entropy(
        logits.unsqueeze(0),
        torch.tensor([target], device=device),
    )
    positive = logits[target]
    mask = torch.ones(
        logits.numel(), device=device,
        dtype=torch.bool,
    )
    mask[target] = False
    separation = F.softplus(
        margin + logits[mask].max() - positive
    )
    return classification + 0.25 * separation


@torch.inference_mode()
def evaluate(model, rows, device):
    model.eval()
    totals = {
        "examples": len(rows),
        "positive_examples": 0,
        "null_examples": 0,
        "top1_correct": 0,
        "positive_top1": 0,
        "positive_rank": 0,
        "null_correct": 0,
    }
    for row in rows:
        logits = model.logits(row, device)
        predicted = int(logits.argmax())
        null_index = logits.numel() - 1
        if row["null_target"]:
            totals["null_examples"] += 1
            correct = predicted == null_index
            totals["null_correct"] += int(correct)
            totals["top1_correct"] += int(correct)
            continue
        target = int(row["target"])
        totals["positive_examples"] += 1
        totals["positive_rank"] += int(
            int(logits[:-1].argmax()) == target
        )
        correct = predicted == target
        totals["positive_top1"] += int(correct)
        totals["top1_correct"] += int(correct)
    positive = max(totals["positive_examples"], 1)
    null = max(totals["null_examples"], 1)
    return {
        "examples": totals["examples"],
        "top1_accuracy":
            totals["top1_correct"]
            / max(totals["examples"], 1),
        "positive_top1":
            totals["positive_top1"] / positive,
        "positive_rank":
            totals["positive_rank"] / positive,
        "null_accuracy":
            totals["null_correct"] / null,
        "positive_examples":
            totals["positive_examples"],
        "null_examples": totals["null_examples"],
    }


def metric_key(metrics):
    return (
        min(
            metrics["positive_top1"],
            metrics["null_accuracy"],
        ),
        metrics["top1_accuracy"],
        metrics["positive_rank"],
    )


def train(
    model, train_rows, valid_rows, device,
    steps, batch_size, learning_rate,
    weight_decay, eval_every, patience,
    margin, seed,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    positive = [
        row for row in train_rows
        if not row["null_target"]
    ]
    null = [
        row for row in train_rows
        if row["null_target"]
    ]
    if not positive or not null or batch_size < 2:
        raise ValueError(
            "typed query training needs positive and NULL rows"
        )
    rng = random.Random(seed)
    best_metrics = evaluate(model, valid_rows, device)
    best_state = clone_state_dict(model)
    stale = 0
    print(json.dumps({
        "phase": "typed_query_baseline",
        **best_metrics,
    }, separators=(",", ":")), flush=True)
    for step in range(1, steps + 1):
        selected = []
        null_count = batch_size // 2
        selected.extend(
            null[rng.randrange(len(null))]
            for _ in range(null_count)
        )
        selected.extend(
            positive[rng.randrange(len(positive))]
            for _ in range(batch_size - null_count)
        )
        rng.shuffle(selected)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = [
            row_loss(model, row, device, margin)
            for row in selected
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        if step % eval_every and step != steps:
            continue
        metrics = evaluate(model, valid_rows, device)
        print(json.dumps({
            "phase": "typed_query_valid",
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            **metrics,
        }, separators=(",", ":")), flush=True)
        if metric_key(metrics) > metric_key(best_metrics):
            best_metrics = metrics
            best_state = clone_state_dict(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_checkpoint")
    parser.add_argument("gguf")
    parser.add_argument("train_feature_cache")
    parser.add_argument("valid_feature_cache")
    parser.add_argument("output")
    parser.add_argument("--lib", required=True)
    parser.add_argument(
        "--train-pair-cache", required=True
    )
    parser.add_argument(
        "--valid-pair-cache", required=True
    )
    parser.add_argument("--pair-batch", type=int, default=16)
    parser.add_argument("--export-binary")
    parser.add_argument("--pair-binary")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--learning-rate", type=float, default=2.0e-4
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.02
    )
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint, pair_model = load_typed_pair_model(
        args.pair_checkpoint
    )
    backbone_sha = hashlib.sha256(
        Path(args.gguf).read_bytes()
    ).hexdigest()
    if checkpoint["backbone_sha256"] != backbone_sha:
        raise ValueError("typed pair backbone mismatch")
    pair_model.to(args.device)
    pair_model.requires_grad_(False)
    pair_model.eval()
    embedding_table = load_token_embedding_table(
        args.gguf, args.lib, backbone_sha,
        int(checkpoint["hidden"]),
    )
    train_rows = load_or_compile(
        args.train_feature_cache,
        args.train_pair_cache,
        args.pair_checkpoint,
        pair_model, embedding_table,
        args.device, args.pair_batch, backbone_sha,
    )
    valid_rows = load_or_compile(
        args.valid_feature_cache,
        args.valid_pair_cache,
        args.pair_checkpoint,
        pair_model, embedding_table,
        args.device, args.pair_batch, backbone_sha,
    )
    del pair_model
    del embedding_table
    torch.cuda.empty_cache()
    train_worlds = {row["world_id"] for row in train_rows}
    if train_worlds.intersection(row["world_id"] for row in valid_rows):
        raise ValueError("query training and validation worlds overlap")

    model = TypedQueryActivator(
        checkpoint
    ).to(args.device)
    metrics = train(
        model, train_rows, valid_rows,
        args.device, args.steps, args.batch,
        args.learning_rate, args.weight_decay,
        args.eval_every, args.patience,
        args.margin, args.seed + 1,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": QUERY_ACTIVATOR,
        "backbone_sha256": backbone_sha,
        "pair_checkpoint_fingerprint":
            file_fingerprint(args.pair_checkpoint),
        "rank": model.rank,
        "input_width": model.input_width,
        "set_feature_count": SET_FEATURE_COUNT,
        "state_dict": clone_state_dict(model),
        "valid_metrics": metrics,
        "validation_role": "checkpoint_selection_not_final_test",
        "training_config": vars(args),
        "train_sample_ids": [row["sample_id"] for row in train_rows],
        "valid_sample_ids": [row["sample_id"] for row in valid_rows],
        "objective":
            "candidate_cross_entropy_plus_hard_margin"
            "_with_candidate_set_null",
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, output)
    if args.export_binary:
        if not args.pair_binary:
            raise ValueError(
                "--export-binary requires --pair-binary"
            )
        export_typed_query_activator_binary(
            output, args.pair_checkpoint,
            args.pair_binary,
            args.export_binary,
        )
    print(json.dumps({
        "phase": "typed_query_done",
        "output": str(output),
        "binary_output": args.export_binary,
        **metrics,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
