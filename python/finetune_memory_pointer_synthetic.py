#!/usr/bin/env python3
"""Fine-tune the extractive pointer on exact addressed-memory facts.

This closes the token-boundary gap left by the LoCoMo-only pointer (for
example ``matcha`` -> ``a``).  The backbone and null/no-answer head stay
frozen; only start/end span weights are updated.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_data import CTokenizer
from memory_pointer_training import (
    SpanPointer, collate, evaluate, export_pointer, find_subsequence,
)
from train_addressed_memory_controller import ATTRS


WRITE_TEMPLATES = [
    "Please remember that my {attr} is {value}.",
    "For future reference, my {attr} is {value}.",
    "My {attr} is {value}; remember it.",
]
UPDATE_TEMPLATES = [
    "Update: my {attr} is now {value}.",
    "My {attr} changed and is now {value}.",
    "Replace the old {attr}; the new value is {value}.",
    "I used to say my {attr} was {old}, but these days it is {value}.",
    "Funny thing: my {attr} is not {old} anymore; I moved on to {value}.",
    "My {attr} changed; I got tired of {old} and went with {value}.",
]
QUERY_TEMPLATES = [
    "What is my {attr}?",
    "What is the current value of my {attr}?",
    "Do you remember my {attr}?",
]


@torch.inference_mode()
def build_rows(backbone, tokenizer, seed):
    rng = random.Random(seed)
    rows = []
    for attr, values in ATTRS:
        for value in values:
            old = rng.choice([candidate for candidate in values
                              if candidate != value])
            answer_ids = tokenizer.encode(value, add_bos=False)
            for templates in (WRITE_TEMPLATES, UPDATE_TEMPLATES):
                for template in templates:
                    record = template.format(
                        attr=attr, value=value, old=old)
                    question = rng.choice(QUERY_TEMPLATES).format(attr=attr)
                    prefix = f"Question: {question}\nMemory record:\n"
                    prefix_ids = tokenizer.encode(prefix, add_bos=True)
                    record_ids = tokenizer.encode(record, add_bos=False)
                    span = find_subsequence(record_ids, answer_ids)
                    if span is None:
                        continue
                    hidden = backbone(
                        torch.tensor(
                            prefix_ids + record_ids,
                            device=backbone.device),
                        return_hidden=True)
                    rows.append({
                        "conversation": 0,
                        "qa_index": len(rows),
                        "record_ids": record_ids,
                        "hidden": hidden[len(prefix_ids):].to(
                            dtype=torch.float16, device="cpu"),
                        "start": span[0],
                        "end": span[1],
                        "answerable": True,
                    })
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument(
        "init_pointer",
        help="pointer.pt to fine-tune, or '-' to initialize from scratch")
    parser.add_argument("output")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--cache")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"),
        default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-span", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("mps" if args.device == "auto" and
              torch.backends.mps.is_available()
              else ("cpu" if args.device == "auto" else args.device)))
    torch.manual_seed(args.seed)
    started = time.time()
    initial = (
        None if args.init_pointer == "-" else
        torch.load(
            args.init_pointer, map_location="cpu", weights_only=False))

    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        rows = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        weights = GGUFWeights(args.gguf, args.lib)
        backbone = TorchBackbone(
            weights, device=device, dtype=torch.bfloat16)
        tokenizer = CTokenizer(args.tok_probe, args.gguf)
        rows = build_rows(backbone, tokenizer, args.seed)
        weights.close()
        del backbone
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rows, cache)
    split = max(1, len(rows) // 5)
    valid_rows = rows[:split]
    train_rows = rows[split:]
    hidden = int(
        initial["hidden"] if initial is not None
        else rows[0]["hidden"].shape[1])
    max_span = int(
        initial["max_span"] if initial is not None else args.max_span)
    model = SpanPointer(hidden).to(device)
    if initial is not None:
        model.load_state_dict(initial["state_dict"], strict=True)
    for parameter in model.null.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [*model.start.parameters(), *model.end.parameters()],
        lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    best_exact = -1.0
    best_state = None
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        losses = []
        model.train()
        for offset in range(0, len(train_rows), args.batch):
            batch = train_rows[offset:offset + args.batch]
            hidden, mask, starts, ends = collate(batch, device)
            start_scores, end_scores, _ = model(hidden, mask)
            loss = (
                F.cross_entropy(start_scores, starts) +
                F.cross_entropy(end_scores, ends))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.eval()
            report = evaluate(
                model, valid_rows, device, max_span)
            print(json.dumps({
                "epoch": epoch + 1,
                "loss": sum(losses) / max(len(losses), 1),
                **report,
            }), flush=True)
            if report["span_exact"] > best_exact:
                best_exact = report["span_exact"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
            if best_exact >= 1.0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, valid_rows, device, max_span)
    threshold = float(
        initial["metrics"]["accept_threshold"]
        if initial is not None else final_metrics["accept_threshold"])
    export_pointer(
        args.output, model, args.gguf, hidden, max_span, threshold)
    torch.save({
        **({} if initial is None else initial),
        "state_dict": model.state_dict(),
        "hidden": hidden,
        "max_span": max_span,
        "metrics": final_metrics,
        "synthetic_metrics": evaluate(
            model, valid_rows, device, max_span),
    }, str(args.output) + ".pt")
    print(json.dumps({
        "result": "done",
        "output": args.output,
        "examples": len(rows),
        "best_span_exact": best_exact,
        "seconds": time.time() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
