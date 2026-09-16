#!/usr/bin/env python3
"""Train field-wise contiguous span taggers on a frozen typed writer."""

from __future__ import annotations

import argparse
import json
import math
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
    TypedMemoryWriter,
    collate_examples,
    load_token_embedding_table,
    load_writer_examples,
)


FORMAT = "TYPED_SPAN_TAGGER_V1"


class FieldSpanTagger(nn.Module):
    """Residual token tagger around frozen writer boundary logits."""

    def __init__(self, rank, max_span, start_key, end_key):
        super().__init__()
        self.rank = int(rank)
        self.max_span = int(max_span)
        self.register_buffer(
            "base_start_key",
            start_key.detach().float().clone(),
        )
        self.register_buffer(
            "base_end_key",
            end_key.detach().float().clone(),
        )
        self.context = nn.Conv1d(
            self.rank, self.rank, kernel_size=3, padding=1,
        )
        self.output = nn.Linear(self.rank, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.length_logits = nn.Parameter(
            torch.zeros(self.max_span)
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(0.0)
        )

    def token_logits(self, tokens):
        contextual = self.context(
            tokens.transpose(1, 2)
        ).transpose(1, 2)
        return self.output(F.gelu(contextual)).squeeze(-1)

    def segment_scores(self, tokens, mask):
        batch, token_count, _ = tokens.shape
        start_score = torch.einsum(
            "btr,r->bt", tokens, self.base_start_key,
        )
        end_score = torch.einsum(
            "btr,r->bt", tokens, self.base_end_key,
        )
        emissions = self.token_logits(tokens)
        prefix = F.pad(
            emissions.cumsum(dim=1), (1, 0)
        )
        starts = torch.arange(
            token_count, device=tokens.device
        )[:, None]
        lengths = torch.arange(
            1, self.max_span + 1, device=tokens.device
        )[None, :]
        ends = starts + lengths - 1
        valid = ends < token_count
        safe_ends = ends.clamp_max(token_count - 1)
        segment_emission = (
            prefix[:, safe_ends + 1]
            - prefix[:, starts]
        )
        score = (
            start_score[:, :, None]
            + end_score[:, safe_ends]
            + self.residual_scale.exp().clamp_max(20.0)
            * segment_emission
            + self.length_logits[None, None, :]
        )
        valid = (
            valid[None, :, :]
            & mask[:, :, None]
            & mask[:, safe_ends]
        )
        return score.masked_fill(~valid, -1e9), emissions

    def best_span(self, tokens, mask):
        scores, _ = self.segment_scores(tokens, mask)
        selected = scores.flatten(1).argmax(dim=-1)
        length_count = scores.shape[-1]
        start = selected // length_count
        length = selected % length_count + 1
        return start, start + length - 1


class TypedSpanTagger(nn.Module):
    def __init__(self, rank, max_span, boundary_keys):
        super().__init__()
        if tuple(boundary_keys.shape) != (
            len(SLOT_NAMES), 2, rank,
        ):
            raise ValueError(
                "writer boundary keys do not cover all fields"
            )
        self.rank = int(rank)
        self.max_span = int(max_span)
        self.fields = nn.ModuleDict({
            name: FieldSpanTagger(
                rank, max_span,
                boundary_keys[index, 0],
                boundary_keys[index, 1],
            )
            for index, name in enumerate(SLOT_NAMES)
        })

    def field_output(self, writer_output, mask, name):
        return self.fields[name].segment_scores(
            writer_output["score_tokens"][name], mask
        )

    def best_spans(self, writer_output, mask):
        return {
            name: self.fields[name].best_span(
                writer_output["score_tokens"][name], mask
            )
            for name in SLOT_NAMES
        }


def balanced_token_loss(logits, targets, mask):
    positive = targets & mask
    negative = ~targets & mask
    positive_loss = F.softplus(-logits[positive]).mean()
    negative_loss = F.softplus(logits[negative]).mean()
    probability = torch.sigmoid(logits) * mask
    target = targets.float()
    dice = 1.0 - (
        2.0 * (probability * target).sum(dim=1) + 1e-6
    ) / (
        probability.sum(dim=1) + target.sum(dim=1) + 1e-6
    )
    return 0.4 * (
        positive_loss + negative_loss
    ) + 0.2 * dice.mean()


def field_loss(scores, emissions, batch, name, max_span):
    target_start = batch[f"{name}_span_start"]
    target_end = batch[f"{name}_span_end"]
    target_length = target_end - target_start
    if bool((target_length >= max_span).any()):
        raise ValueError(
            f"{name} target exceeds maximum span"
        )
    target_segment = target_start * max_span + target_length
    segment_loss = F.cross_entropy(
        scores.flatten(1), target_segment
    )
    token_loss = balanced_token_loss(
        emissions,
        batch[f"{name}_token_target"],
        batch["mask"],
    )
    return 0.6 * segment_loss + 0.4 * token_loss


def load_writer(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("format") != "TYPED_MEMORY_WRITER_V1":
        raise ValueError("bad typed writer checkpoint")
    if not checkpoint.get("predict_span_boundaries", False):
        raise ValueError(
            "typed writer has no span-boundary initialization"
        )
    model = TypedMemoryWriter(
        int(checkpoint["hidden"]),
        int(checkpoint["rank"]),
        len(checkpoint["hidden_layer_bands"]),
        bool(checkpoint.get(
            "separate_address_localizer", False)),
        True,
        bool(checkpoint.get(
            "use_token_embeddings", False)),
        bool(checkpoint.get(
            "hard_address_pooling", False)),
        bool(checkpoint.get(
            "contiguous_span_pooling", False)),
        int(checkpoint.get("max_field_span", 8)),
    )
    model.load_state_dict(
        checkpoint["state_dict"], strict=True
    )
    model.requires_grad_(False)
    model.eval()
    return checkpoint, model.to(device)


@torch.inference_mode()
def evaluate(writer, tagger, examples, device, batch_size):
    field_correct = {
        name: 0 for name in SLOT_NAMES
    }
    all_correct = 0
    operation_total = {0: 0, 1: 0}
    operation_all_correct = {0: 0, 1: 0}
    operation_field_correct = {
        operation: {
            name: 0 for name in SLOT_NAMES
        }
        for operation in (0, 1)
    }
    tagger.eval()
    for offset in range(0, len(examples), batch_size):
        selected = list(range(
            offset,
            min(offset + batch_size, len(examples)),
        ))
        batch = collate_examples(examples, selected, device)
        writer_output = writer(
            batch["hidden"], batch["mask"],
            batch["identity_hidden"],
        )
        spans = tagger.best_spans(
            writer_output, batch["mask"]
        )
        joint = torch.ones(
            len(selected), device=device, dtype=torch.bool
        )
        operations = batch["operation"]
        for operation in (0, 1):
            operation_total[operation] += int(
                (operations == operation).sum()
            )
        for name in SLOT_NAMES:
            start, end = spans[name]
            exact = (
                start == batch[f"{name}_span_start"]
            ) & (
                end == batch[f"{name}_span_end"]
            )
            field_correct[name] += int(exact.sum())
            for operation in (0, 1):
                operation_field_correct[operation][name] += int(
                    (exact & (operations == operation)).sum()
                )
            joint &= exact
        all_correct += int(joint.sum())
        for operation in (0, 1):
            operation_all_correct[operation] += int(
                (joint & (operations == operation)).sum()
            )
    total = max(len(examples), 1)
    metrics = {
        "examples": len(examples),
        "all_field_span_exact": all_correct / total,
        **{
            f"{name}_span_exact":
                field_correct[name] / total
            for name in SLOT_NAMES
        },
    }
    for operation, label in ((0, "create"), (1, "update")):
        operation_count = max(operation_total[operation], 1)
        metrics[f"{label}_examples"] = (
            operation_total[operation]
        )
        metrics[f"{label}_all_field_span_exact"] = (
            operation_all_correct[operation]
            / operation_count
        )
        for name in SLOT_NAMES:
            metrics[
                f"{name}_{label}_span_exact"
            ] = (
                operation_field_correct[operation][name]
                / operation_count
            )
    return metrics


def stratified_sample_indices(
    examples, batch_size, create_fraction, rng,
):
    create = [
        index for index, example in enumerate(examples)
        if int(example["operation"]) == 0
    ]
    update = [
        index for index, example in enumerate(examples)
        if int(example["operation"]) == 1
    ]
    if not create or not update:
        raise ValueError(
            "training examples need create and update events"
        )
    create_count = round(batch_size * create_fraction)
    create_count = min(max(create_count, 1), batch_size - 1)
    return [
        rng.choice(create)
        for _ in range(create_count)
    ] + [
        rng.choice(update)
        for _ in range(batch_size - create_count)
    ]


def train(
    writer, tagger, train_examples, valid_examples,
    extra_valid_examples,
    device, steps, batch_size, learning_rate,
    weight_decay, eval_every, patience, seed,
    create_fraction,
):
    optimizer = torch.optim.AdamW(
        tagger.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = random.Random(seed)
    best_metrics = evaluate(
        writer, tagger, valid_examples, device, batch_size
    )
    best_extra_metrics = (
        evaluate(
            writer, tagger, extra_valid_examples,
            device, batch_size,
        )
        if extra_valid_examples is not None
        else None
    )
    def field_key(metrics, extra_metrics, name):
        values = [
            metrics[f"{name}_create_span_exact"],
            metrics[f"{name}_update_span_exact"],
        ]
        if extra_metrics is not None:
            values.extend((
                extra_metrics[
                    f"{name}_create_span_exact"
                ],
                extra_metrics[
                    f"{name}_update_span_exact"
                ],
            ))
        return (min(values), sum(values) / len(values))
    best_field_scores = {
        name: field_key(
            best_metrics, best_extra_metrics, name
        )
        for name in SLOT_NAMES
    }
    best_field_steps = {
        name: 0 for name in SLOT_NAMES
    }
    best_field_states = {
        name: clone_state_dict(tagger.fields[name])
        for name in SLOT_NAMES
    }
    stale = 0
    print(json.dumps({
        "phase": "typed_span_tagger_baseline",
        **best_metrics,
        "extra_valid": best_extra_metrics,
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
            writer_output = writer(
                batch["hidden"], batch["mask"],
                batch["identity_hidden"],
            )
        tagger.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name in SLOT_NAMES:
            scores, emissions = tagger.field_output(
                writer_output, batch["mask"], name
            )
            losses[name] = field_loss(
                scores, emissions, batch, name,
                tagger.max_span,
            )
        loss = sum(losses.values()) / len(SLOT_NAMES)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            tagger.parameters(), 1.0
        )
        optimizer.step()
        if step % eval_every and step != steps:
            continue
        metrics = evaluate(
            writer, tagger, valid_examples,
            device, batch_size,
        )
        extra_metrics = (
            evaluate(
                writer, tagger, extra_valid_examples,
                device, batch_size,
            )
            if extra_valid_examples is not None
            else None
        )
        print(json.dumps({
            "phase": "typed_span_tagger_valid",
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "loss_parts": {
                name: float(value.detach())
                for name, value in losses.items()
            },
            **metrics,
            "extra_valid": extra_metrics,
        }, separators=(",", ":")), flush=True)
        improved = False
        for name in SLOT_NAMES:
            score = field_key(
                metrics, extra_metrics, name
            )
            if score > best_field_scores[name]:
                best_field_scores[name] = score
                best_field_steps[name] = step
                best_field_states[name] = clone_state_dict(
                    tagger.fields[name]
                )
                improved = True
        stale = 0 if improved else stale + 1
        if stale >= patience:
            break
    for name in SLOT_NAMES:
        tagger.fields[name].load_state_dict(
            best_field_states[name]
        )
    final_metrics = evaluate(
        writer, tagger, valid_examples, device, batch_size
    )
    final_extra_metrics = (
        evaluate(
            writer, tagger, extra_valid_examples,
            device, batch_size,
        )
        if extra_valid_examples is not None
        else None
    )
    return (
        final_metrics,
        final_extra_metrics,
        best_field_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("writer_checkpoint")
    parser.add_argument("train_features")
    parser.add_argument("train_jsonl")
    parser.add_argument("valid_features")
    parser.add_argument("valid_jsonl")
    parser.add_argument("output")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--extra-valid-features")
    parser.add_argument("--extra-valid-jsonl")
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--create-fraction", type=float, default=0.25
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if bool(args.extra_valid_features) != bool(
        args.extra_valid_jsonl
    ):
        parser.error(
            "extra validation features and JSONL "
            "must be supplied together"
        )
    if not 0.0 < args.create_fraction < 1.0:
        parser.error(
            "--create-fraction must be between zero and one"
        )

    checkpoint, writer = load_writer(
        args.writer_checkpoint, args.device
    )
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    embeddings = load_token_embedding_table(
        args.gguf, args.lib,
        checkpoint["backbone_sha256"],
        int(checkpoint["hidden"]),
    )
    train_examples = load_writer_examples(
        args.train_features, args.train_jsonl,
        checkpoint["backbone_sha256"],
        tokenizer, embeddings,
    )
    valid_examples = load_writer_examples(
        args.valid_features, args.valid_jsonl,
        checkpoint["backbone_sha256"],
        tokenizer, embeddings,
    )
    extra_valid_examples = (
        load_writer_examples(
            args.extra_valid_features,
            args.extra_valid_jsonl,
            checkpoint["backbone_sha256"],
            tokenizer, embeddings,
        )
        if args.extra_valid_features
        else None
    )
    del embeddings
    boundary = checkpoint["state_dict"].get(
        "span_boundary_keys"
    )
    if boundary is None:
        raise ValueError("writer boundary keys are missing")
    tagger = TypedSpanTagger(
        int(checkpoint["rank"]),
        args.max_span,
        boundary,
    ).to(args.device)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if initial.get("format") != FORMAT:
            raise ValueError(
                "bad typed span tagger checkpoint"
            )
        if (
            initial.get("backbone_sha256")
            != checkpoint["backbone_sha256"]
        ):
            raise ValueError(
                "span tagger backbone identity mismatch"
            )
        if (
            initial.get("writer_checkpoint_fingerprint")
            != file_fingerprint(args.writer_checkpoint)
        ):
            raise ValueError(
                "span tagger writer identity mismatch"
            )
        if (
            int(initial["rank"]) != tagger.rank
            or int(initial["max_span"]) != tagger.max_span
        ):
            raise ValueError(
                "span tagger geometry mismatch"
            )
        tagger.load_state_dict(
            initial["state_dict"], strict=True
        )
    (
        metrics,
        extra_metrics,
        selected_steps,
    ) = train(
        writer, tagger,
        train_examples, valid_examples,
        extra_valid_examples,
        args.device, args.steps, args.batch,
        args.learning_rate, args.weight_decay,
        args.eval_every, args.patience, args.seed,
        args.create_fraction,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT,
        "backbone_sha256": checkpoint["backbone_sha256"],
        "writer_checkpoint_fingerprint":
            file_fingerprint(args.writer_checkpoint),
        "rank": tagger.rank,
        "max_span": tagger.max_span,
        "state_dict": clone_state_dict(tagger),
        "selected_steps": selected_steps,
        "create_fraction": args.create_fraction,
        "valid_metrics": metrics,
        "extra_valid_metrics": extra_metrics,
        "writer_frozen": True,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, output)
    print(json.dumps({
        "phase": "typed_span_tagger_done",
        "output": str(output),
        "selected_steps": selected_steps,
        "create_fraction": args.create_fraction,
        **metrics,
        "extra_valid": extra_metrics,
        "writer_frozen": True,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
