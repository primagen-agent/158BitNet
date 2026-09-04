#!/usr/bin/env python3
"""Measure answer coverage for every memory-attention layer."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from eval_bnmem_torch_locomo import (  # noqa: E402
    CTokenDecoder,
    CTokenizer,
    GGUFWeights,
    TorchBackbone,
    forward_chunk,
    load_memory,
)
from memory_retrieval import clean_excerpt  # noqa: E402
from train_data import render_chunk  # noqa: E402


def normalized_tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def excerpts_from_weights(weights, token_ids, decoder, count, size):
    if weights is None or weights.shape[-1] != token_ids.numel():
        return []
    scores = weights[-1].float().clone()
    scores[token_ids < 0] = -float("inf")
    radius = max(size // 2, 1)
    blocked = torch.zeros_like(scores, dtype=torch.bool)
    excerpts = []
    for _ in range(count):
        candidates = scores.masked_fill(blocked, -float("inf"))
        center = int(candidates.argmax().item())
        if not torch.isfinite(candidates[center]):
            break
        start = max(0, center - radius)
        end = min(token_ids.numel(), start + size)
        start = max(0, end - size)
        ids = token_ids[start:end]
        ids = ids[ids >= 0].tolist()
        if ids:
            text = clean_excerpt(decoder.decode(ids))
            if text and text not in excerpts:
                excerpts.append(text)
        blocked[max(0, start - radius):min(
            token_ids.numel(), end + radius)] = True
    return excerpts


def coverage(answer, excerpts):
    gold = normalized_tokens(answer)
    source = normalized_tokens(" ".join(excerpts))
    source_set = set(source)
    value = sum(token in source_set for token in gold) / max(len(gold), 1)
    exact = " ".join(gold) in " ".join(source)
    return value, exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("memory_model")
    parser.add_argument("jsonl")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--max-samples", type=int, default=390)
    parser.add_argument("--max-memory-slots", type=int, default=2048)
    parser.add_argument("--slot-temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-windows", type=int, default=4)
    parser.add_argument("--retrieval-window-size", type=int, default=32)
    args = parser.parse_args()

    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(weights, device="cuda", dtype=torch.bfloat16)
    memory = load_memory(
        args.memory_model, backbone, state_mode="slots",
        max_memory_slots=args.max_memory_slots,
        slot_temperature=args.slot_temperature)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)

    rows, seen = [], set()
    for line_index, line in enumerate(
        Path(args.jsonl).read_text(encoding="utf-8").splitlines()
    ):
        row = json.loads(line)
        locomo_id = str(row.get("locomo_id", ""))
        key = (
            ("locomo", ":".join(locomo_id.split(":")[:2]))
            if locomo_id else ("line", line_index))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= args.max_samples:
            break

    names = [f"layer_{index}" for index in range(memory.n_layers)]
    names += ["mean", "max", "rrf"]
    values = {name: [] for name in names}
    exacts = {name: 0 for name in names}
    categories = []
    with torch.inference_mode():
        for row_index, row in enumerate(rows):
            memory.reset_state()
            query_index = row.get(
                "query_turn_id", len(row["messages"]) - 1)
            for chunk_index, chunk in enumerate(row["messages"]):
                if chunk_index == query_index:
                    break
                text, _ = render_chunk(chunk, False)
                ids = tokenizer.encode(text, add_bos=True)
                forward_chunk(backbone, memory, ids)
                memory.commit_all()

            prompt, answer = render_chunk(
                row["messages"][query_index], True)
            categories.append(row.get("category"))
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            _ = backbone(
                torch.tensor(prompt_ids, device="cuda"),
                memory_v6=memory)
            memory.discard_captured()
            token_ids = memory.pointer_token_ids
            layer_weights = memory.last_pointer_weights_by_layer
            candidates = {
                f"layer_{index}": layer
                for index, layer in enumerate(layer_weights)}
            available = [layer for layer in layer_weights
                         if layer is not None]
            if available:
                stacked = torch.stack(available)
                candidates["mean"] = stacked.mean(dim=0)
                candidates["max"] = stacked.max(dim=0).values
                final_scores = stacked[:, -1, :]
                order = final_scores.argsort(dim=-1, descending=True)
                ranks = order.argsort(dim=-1).float()
                candidates["rrf"] = (
                    1.0 / (60.0 + ranks + 1.0)
                ).sum(dim=0, keepdim=True)
            for name, attention in candidates.items():
                excerpts = excerpts_from_weights(
                    attention, token_ids, decoder,
                    args.retrieval_windows,
                    args.retrieval_window_size)
                score, exact = coverage(answer, excerpts)
                values[name].append(score)
                exacts[name] += int(exact)
            if (row_index + 1) % 25 == 0:
                print(f"[{row_index + 1}/{len(rows)}]", flush=True)

    report = {}
    for name in names:
        scores = values[name]
        report[name] = {
            "samples": len(scores),
            "coverage_mean": sum(scores) / max(len(scores), 1),
            "coverage_ge80": sum(score >= 0.8 for score in scores)
                             / max(len(scores), 1),
            "exact": exacts[name] / max(len(scores), 1),
        }
    ranking = sorted(
        report, key=lambda name: report[name]["coverage_mean"],
        reverse=True)
    layer_names = [
        f"layer_{index}" for index in range(memory.n_layers)]
    oracle_scores = []
    oracle_exact = 0
    best_layer_counts = collections.Counter()
    category_oracle = collections.defaultdict(list)
    for sample_index in range(len(rows)):
        best_name = max(
            layer_names,
            key=lambda name: values[name][sample_index])
        best_score = values[best_name][sample_index]
        oracle_scores.append(best_score)
        oracle_exact += int(
            any(
                # Exact implies coverage 1, but the reverse may include
                # reordered answer tokens. Count exact from the chosen layer.
                values[name][sample_index] == 1.0
                for name in layer_names))
        best_layer_counts[best_name] += 1
        category_oracle[categories[sample_index]].append(best_score)
    print(json.dumps({
        "samples": len(rows),
        "retrieval_windows": args.retrieval_windows,
        "retrieval_window_size": args.retrieval_window_size,
        "ranking": ranking,
        "per_sample_layer_oracle": {
            "coverage_mean": sum(oracle_scores) / max(len(oracle_scores), 1),
            "coverage_ge80": sum(score >= 0.8 for score in oracle_scores)
                             / max(len(oracle_scores), 1),
            "coverage_one": sum(score == 1.0 for score in oracle_scores)
                            / max(len(oracle_scores), 1),
            "best_layer_counts": dict(best_layer_counts.most_common()),
            "categories": {
                str(category): {
                    "samples": len(scores),
                    "coverage_mean": sum(scores) / len(scores),
                    "coverage_ge80": sum(score >= 0.8 for score in scores)
                                     / len(scores),
                }
                for category, scores in sorted(category_oracle.items())
            },
        },
        "results": report,
    }, indent=2))


if __name__ == "__main__":
    main()
