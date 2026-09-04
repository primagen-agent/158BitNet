#!/usr/bin/env python3
"""Retrieve token-preserving LoCoMo memory entries with backbone embeddings."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from backbone_lora import load_lora_bundle  # noqa: E402
from ggw import GGUFWeights  # noqa: E402
from prepare_locomo_retrieval_data import make_entries  # noqa: E402
from torch_backbone import TorchBackbone  # noqa: E402
from train_data import CTokenizer  # noqa: E402


@torch.inference_mode()
def encode_text(backbone, tokenizer, text, max_tokens):
    ids = tokenizer.encode(text, add_bos=True)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    hidden = backbone(
        torch.tensor(ids, device=backbone.device, dtype=torch.long),
        return_hidden=True)
    # Mean pooling preserves entities that do not occur at the final token;
    # mixing in the final row retains the backbone's sequence summary.
    vector = hidden.float().mean(dim=0) + hidden[-1].float()
    return torch.nn.functional.normalize(vector, dim=0).cpu().numpy()


def query_text(question):
    return (
        "Find the stored conversation facts needed to answer this question: "
        + str(question))


def write_rows(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    for category in sorted({row["category"] for row in rows}):
        output = path / f"locomo_cat{category}.jsonl"
        selected = [row for row in rows if row["category"] == category]
        with output.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("dataset")
    parser.add_argument("output_dir")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-entry-tokens", type=int, default=128)
    parser.add_argument("--max-query-tokens", type=int, default=96)
    parser.add_argument("--valid-conversations", type=int, default=2)
    parser.add_argument("--include-derived-context", action="store_true")
    parser.add_argument("--include-category5", action="store_true")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    valid_ids = set(order[:args.valid_conversations])
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(
        weights, device="cuda", dtype=torch.bfloat16)
    if args.lora:
        backbone_lora, output_lora = load_lora_bundle(
            args.lora, backbone.cfg, device="cuda")
        backbone.backbone_lora = backbone_lora
        if output_lora is not None:
            backbone.output_lora = output_lora
    tokenizer = CTokenizer(args.tok_probe, args.gguf)

    rows = []
    recall_hits = 0
    recall_total = 0
    all_evidence_hits = 0
    question_count = 0
    started = time.time()
    for conversation_index in sorted(valid_ids):
        sample = samples[conversation_index]
        entries = make_entries(sample, args.include_derived_context)
        entry_vectors = np.stack([
            encode_text(
                backbone, tokenizer, entry["text"],
                args.max_entry_tokens)
            for entry in entries
        ])
        print(json.dumps({
            "conversation": conversation_index,
            "phase": "entries_encoded",
            "entries": len(entries),
            "seconds": time.time() - started,
        }), flush=True)
        for question_index, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            if category == 5 and not args.include_category5:
                continue
            q_vector = encode_text(
                backbone, tokenizer, query_text(qa["question"]),
                args.max_query_tokens)
            scores = entry_vectors @ q_vector
            selected_indices = np.argsort(scores)[-args.top_k:][::-1]
            selected = [entries[int(index)] for index in selected_indices]
            selected_ids = {entry["dia_id"] for entry in selected}
            gold_ids = {str(value) for value in qa.get("evidence", [])}
            if gold_ids:
                recall_hits += len(selected_ids & gold_ids)
                recall_total += len(gold_ids)
                all_evidence_hits += int(gold_ids <= selected_ids)
                question_count += 1
            answer = qa.get("answer")
            if category == 5 or answer is None:
                answer = "No information available"
            messages = [
                [
                    {"role": "user", "content": (
                        f"{entry['text']}\n\nStore this conversation event "
                        "in long-term memory. Reply OK.")},
                    {"role": "assistant", "content": "OK"},
                ]
                for entry in reversed(selected)
            ]
            messages.append([
                {"role": "user", "content": (
                    "Answer the question using only the stored conversation "
                    "memory. Give only the shortest direct answer. If the "
                    "conversation does not contain the answer, reply "
                    "exactly: No information available.\n"
                    f"Question: {qa['question']}")},
                {"role": "assistant", "content": str(answer)},
            ])
            rows.append({
                "messages": messages,
                "query_turn_id": len(messages) - 1,
                "locomo_id": f"{conversation_index}:{question_index}:0",
                "validation_group": f"conversation:{conversation_index}",
                "category": category,
                "retrieved_dia_ids": [
                    entry["dia_id"] for entry in selected],
                "evidence_dia_ids": sorted(gold_ids),
            })
        print(json.dumps({
            "conversation": conversation_index,
            "phase": "queries_encoded",
            "questions": len(sample["qa"]),
            "evidence_recall": recall_hits / max(recall_total, 1),
            "all_evidence_recall": (
                all_evidence_hits / max(question_count, 1)),
            "seconds": time.time() - started,
        }), flush=True)

    write_rows(Path(args.output_dir) / "valid", rows)
    print(json.dumps({
        "output": str(Path(args.output_dir) / "valid"),
        "rows": len(rows),
        "valid_conversation_ids": sorted(valid_ids),
        "top_k": args.top_k,
        "evidence_recall": recall_hits / max(recall_total, 1),
        "all_evidence_recall": (
            all_evidence_hits / max(question_count, 1)),
        "seconds": time.time() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
