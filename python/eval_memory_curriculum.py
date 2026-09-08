#!/usr/bin/env python3
"""Greedy held-out evaluation for staged memory-curriculum checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from eval_bnmem_torch_locomo import forward_chunk, generate, load_memory
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_data import CTokenDecoder, CTokenizer, render_chunk


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_samples(directory: str, limit: int) -> list[dict]:
    rows = []
    paths = sorted(Path(directory).glob("*.jsonl"))
    per_file = (
        max(1, (limit + len(paths) - 1) // len(paths))
        if limit and paths else 0)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if per_file and index >= per_file:
                    break
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def position_accuracy(prediction_ids: list[int], target_ids: list[int]) -> float:
    if not target_ids:
        return float(not prediction_ids)
    correct = sum(
        left == right
        for left, right in zip(prediction_ids, target_ids)
    )
    return correct / len(target_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("memory_model")
    parser.add_argument("data")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = resolve_device(args.device)
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(
        weights, device=device, dtype=torch.bfloat16)
    memory = load_memory(args.memory_model, backbone, device)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)
    eos = tokenizer.eos()
    stop_ids = {eos}
    im_end = tokenizer.encode("<|im_end|>", add_bos=False)
    if len(im_end) == 1:
        stop_ids.add(im_end[0])

    results = []
    totals = defaultdict(lambda: {
        "samples": 0,
        "memory_exact": 0,
        "no_memory_exact": 0,
        "memory_token_accuracy": 0.0,
        "no_memory_token_accuracy": 0.0,
    })
    for index, sample in enumerate(load_samples(args.data, args.samples)):
        chunks = sample["messages"]
        query_value = sample.get("query_turn_id", len(chunks) - 1)
        query_indices = set(
            int(value) for value in query_value
        ) if isinstance(query_value, list) else {int(query_value)}
        memory.reset_state()
        for chunk_index, chunk in enumerate(chunks):
            is_query = chunk_index in query_indices
            text, chunk_target = render_chunk(chunk, is_query)
            ids = tokenizer.encode(text, add_bos=True)
            if not is_query:
                forward_chunk(backbone, memory, ids)
                memory.commit_all()
                continue

            target = chunk_target.strip()
            saved_state = memory.clone_runtime_state()
            prediction = generate(
                backbone, memory, ids, decoder, stop_ids,
                args.max_answer_tokens)
            memory.restore_runtime_state(saved_state)
            memory.active = False
            no_memory_prediction = generate(
                backbone, memory, ids, decoder, stop_ids,
                args.max_answer_tokens)
            memory.active = True

            target_ids = tokenizer.encode(target, add_bos=False)
            prediction_ids = tokenizer.encode(
                prediction, add_bos=False)
            no_memory_ids = tokenizer.encode(
                no_memory_prediction, add_bos=False)
            kind = sample.get("metadata", {}).get("type", "unknown")
            row = {
                "sample_id": sample.get("sample_id"),
                "query_chunk_index": chunk_index,
                "type": kind,
                "target": target,
                "prediction": prediction,
                "no_memory_prediction": no_memory_prediction,
                "memory_exact": prediction == target,
                "no_memory_exact": no_memory_prediction == target,
                "memory_token_accuracy": position_accuracy(
                    prediction_ids, target_ids),
                "no_memory_token_accuracy": position_accuracy(
                    no_memory_ids, target_ids),
            }
            results.append(row)
            group = totals[kind]
            group["samples"] += 1
            group["memory_exact"] += int(row["memory_exact"])
            group["no_memory_exact"] += int(row["no_memory_exact"])
            group["memory_token_accuracy"] += row[
                "memory_token_accuracy"]
            group["no_memory_token_accuracy"] += row[
                "no_memory_token_accuracy"]
            print(json.dumps({
                "sample": index + 1,
                **row,
            }, ensure_ascii=False, separators=(",", ":")), flush=True)

            # Use the gold completed exchange as subsequent conversation
            # history so each query measures memory rather than accumulated
            # generation errors from earlier queries.
            complete_text, _ = render_chunk(chunk, False)
            complete_ids = tokenizer.encode(
                complete_text, add_bos=True)
            forward_chunk(backbone, memory, complete_ids)
            memory.commit_all()

    summary = {}
    for kind, values in sorted(totals.items()):
        count = values["samples"]
        summary[kind] = {
            "samples": count,
            "memory_exact": values["memory_exact"] / count,
            "no_memory_exact": values["no_memory_exact"] / count,
            "memory_token_accuracy": (
                values["memory_token_accuracy"] / count),
            "no_memory_token_accuracy": (
                values["no_memory_token_accuracy"] / count),
        }
    all_count = len(results)
    summary["all"] = {
        "samples": all_count,
        "memory_exact": sum(row["memory_exact"] for row in results)
        / max(all_count, 1),
        "no_memory_exact": sum(
            row["no_memory_exact"] for row in results)
        / max(all_count, 1),
        "memory_token_accuracy": sum(
            row["memory_token_accuracy"] for row in results)
        / max(all_count, 1),
        "no_memory_token_accuracy": sum(
            row["no_memory_token_accuracy"] for row in results)
        / max(all_count, 1),
        "device": device,
        "kv_cache": False,
    }
    payload = {"summary": summary, "results": results}
    print(json.dumps(
        {"summary": summary}, ensure_ascii=False,
        separators=(",", ":")), flush=True)
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")


if __name__ == "__main__":
    main()
