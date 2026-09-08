#!/usr/bin/env python3
"""Train a low-rank query/fact retriever on LoCoMo evidence labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from backbone_lora import load_lora_bundle  # noqa: E402
from ggw import GGUFWeights  # noqa: E402
from prepare_locomo_embedding_retrieval import (  # noqa: E402
    encode_text,
    query_text,
    write_rows,
)
from prepare_locomo_retrieval_data import make_entries  # noqa: E402
from torch_backbone import TorchBackbone  # noqa: E402
from train_data import CTokenizer  # noqa: E402


class ProjectionTower(nn.Module):
    def __init__(self, hidden, rank, width):
        super().__init__()
        self.width = int(width)
        if self.width > 0:
            self.input = nn.Linear(hidden, self.width)
            self.output = nn.Linear(self.width, rank)
            nn.init.kaiming_uniform_(self.input.weight, nonlinearity="linear")
            nn.init.orthogonal_(self.output.weight)
        else:
            self.input = nn.Linear(hidden, rank, bias=False)
            self.output = None
            nn.init.orthogonal_(self.input.weight)

    def forward(self, hidden):
        value = self.input(hidden)
        if self.output is not None:
            value = self.output(F.silu(value))
        return value


class DualProjection(nn.Module):
    def __init__(self, hidden, rank, width=0):
        super().__init__()
        self.width = int(width)
        self.query = ProjectionTower(hidden, rank, self.width)
        self.entry = ProjectionTower(hidden, rank, self.width)

    def scores(self, queries, entries, temperature):
        q = F.normalize(self.query(queries), dim=-1)
        e = F.normalize(self.entry(entries), dim=-1)
        return q @ e.transpose(0, 1) / temperature


def build_conversation_embeddings(
    backbone, tokenizer, sample, include_derived_context,
    max_entry_tokens, max_query_tokens, pooling
):
    entries = make_entries(sample, include_derived_context)
    entry_vectors = np.stack([
        encode_text(
            backbone, tokenizer, entry["text"], max_entry_tokens, pooling)
        for entry in entries
    ])
    entry_index = {
        entry["dia_id"]: index for index, entry in enumerate(entries)}
    questions = []
    query_vectors = []
    positive_indices = []
    for question_index, qa in enumerate(sample["qa"]):
        if int(qa["category"]) not in (1, 2, 3, 4):
            continue
        positives = sorted({
            entry_index[str(value)]
            for value in qa.get("evidence", [])
            if str(value) in entry_index
        })
        if not positives:
            continue
        questions.append((question_index, qa))
        query_vectors.append(encode_text(
            backbone, tokenizer, query_text(qa["question"]),
            max_query_tokens, pooling))
        positive_indices.append(positives)
    return {
        "entries": entries,
        "entry_vectors": np.asarray(entry_vectors, dtype=np.float32),
        "questions": questions,
        "query_vectors": np.asarray(query_vectors, dtype=np.float32),
        "positive_indices": positive_indices,
    }


def save_portable_controller(
    path, model, model_path, hidden, rank, temperature, pooling
):
    pooling_id = {"last": 1, "mean_last": 2}.get(pooling)
    if pooling_id is None:
        raise ValueError(f"unsupported portable pooling mode {pooling}")
    with open(model_path, "rb") as handle:
        model_sha256 = hashlib.sha256(handle.read()).digest()
    if model.width == 0:
        tensors = [
            model.query.input.weight,
            model.entry.input.weight,
        ]
        header = bytearray(b"BNCTRL1\x00")
        header += struct.pack(
            "<IIIIf", 1, hidden, rank, pooling_id, float(temperature))
        payloads = [
            tensor.detach().float().cpu().contiguous().numpy()
            .astype("<f4", copy=False).tobytes()
            for tensor in tensors
        ]
        header += model_sha256
        header += struct.pack(
            "<II",
            zlib.crc32(payloads[0]) & 0xFFFFFFFF,
            zlib.crc32(payloads[1]) & 0xFFFFFFFF)
        Path(path).write_bytes(header + b"".join(payloads))
        return
    else:
        tensors = [
            model.query.input.weight,
            model.query.input.bias,
            model.query.output.weight,
            model.query.output.bias,
            model.entry.input.weight,
            model.entry.input.bias,
            model.entry.output.weight,
            model.entry.output.bias,
        ]
        header = bytearray(b"BNCTRL2\x00")
        header += struct.pack(
            "<IIIIIf", 2, hidden, rank, pooling_id,
            model.width, float(temperature))
    payloads = [
        tensor.detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False).tobytes()
        for tensor in tensors
    ]
    header += model_sha256
    header += struct.pack("<I", len(payloads))
    for payload in payloads:
        header += struct.pack(
            "<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
    Path(path).write_bytes(header + b"".join(payloads))


def global_training_tensors(conversations, ids, device):
    entry_parts = []
    query_parts = []
    positives = []
    entry_offset = 0
    for conversation_index in ids:
        data = conversations[conversation_index]
        entries = torch.from_numpy(data["entry_vectors"])
        queries = torch.from_numpy(data["query_vectors"])
        entry_parts.append(entries)
        query_parts.append(queries)
        positives.extend([
            [entry_offset + index for index in row]
            for row in data["positive_indices"]
        ])
        entry_offset += entries.shape[0]
    return (
        torch.cat(query_parts).to(device),
        torch.cat(entry_parts).to(device),
        positives,
    )


def save_cache(path, conversations):
    payload = {}
    metadata = {}
    for conversation_index, data in conversations.items():
        prefix = f"c{conversation_index}"
        payload[prefix + "_entries"] = data["entry_vectors"]
        payload[prefix + "_queries"] = data["query_vectors"]
        metadata[str(conversation_index)] = {
            "entries": data["entries"],
            "questions": [
                {"question_index": index, "qa": qa}
                for index, qa in data["questions"]
            ],
            "positive_indices": data["positive_indices"],
        }
    payload["metadata"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **payload)


def load_cache(path):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))
    conversations = {}
    for key, item in metadata.items():
        index = int(key)
        prefix = f"c{index}"
        conversations[index] = {
            "entries": item["entries"],
            "entry_vectors": archive[prefix + "_entries"],
            "questions": [
                (row["question_index"], row["qa"])
                for row in item["questions"]
            ],
            "query_vectors": archive[prefix + "_queries"],
            "positive_indices": item["positive_indices"],
        }
    return conversations


def retrieval_metrics(model, conversations, ids, top_k, temperature, device):
    hits = 0
    total = 0
    all_hits = 0
    questions = 0
    with torch.no_grad():
        for conversation_index in ids:
            data = conversations[conversation_index]
            queries = torch.from_numpy(
                data["query_vectors"]).to(device)
            entries = torch.from_numpy(
                data["entry_vectors"]).to(device)
            scores = model.scores(queries, entries, temperature)
            selected = scores.topk(
                min(top_k, scores.shape[1]), dim=1).indices.cpu().tolist()
            for predicted, positives in zip(
                selected, data["positive_indices"]
            ):
                predicted_set = set(predicted)
                positive_set = set(positives)
                hits += len(predicted_set & positive_set)
                total += len(positive_set)
                all_hits += int(positive_set <= predicted_set)
                questions += 1
    return {
        "evidence_recall": hits / max(total, 1),
        "all_evidence_recall": all_hits / max(questions, 1),
        "questions": questions,
    }


def export_valid_rows(
    model, conversations, valid_ids, output_dir, top_k, temperature, device
):
    rows = []
    with torch.no_grad():
        for conversation_index in valid_ids:
            data = conversations[conversation_index]
            scores = model.scores(
                torch.from_numpy(data["query_vectors"]).to(device),
                torch.from_numpy(data["entry_vectors"]).to(device),
                temperature)
            selected_rows = scores.topk(
                min(top_k, scores.shape[1]), dim=1).indices.cpu().tolist()
            for (question_index, qa), selected_indices in zip(
                data["questions"], selected_rows
            ):
                selected = [
                    data["entries"][index] for index in selected_indices]
                messages = [
                    [
                        {"role": "user", "content": (
                            f"{entry['text']}\n\nStore this conversation "
                            "event in long-term memory. Reply OK.")},
                        {"role": "assistant", "content": "OK"},
                    ]
                    for entry in reversed(selected)
                ]
                messages.append([
                    {"role": "user", "content": (
                        "Answer the question using only the stored "
                        "conversation memory. Give only the shortest direct "
                        "answer. If the conversation does not contain the "
                        "answer, reply exactly: No information available.\n"
                        f"Question: {qa['question']}")},
                    {"role": "assistant", "content": str(qa["answer"])},
                ])
                rows.append({
                    "messages": messages,
                    "query_turn_id": len(messages) - 1,
                    "locomo_id": (
                        f"{conversation_index}:{question_index}:0"),
                    "validation_group": (
                        f"conversation:{conversation_index}"),
                    "category": int(qa["category"]),
                    "retrieved_dia_ids": [
                        entry["dia_id"] for entry in selected],
                    "evidence_dia_ids": [
                        data["entries"][index]["dia_id"]
                        for index in data["positive_indices"][
                            data["questions"].index(
                                (question_index, qa))]],
                })
    write_rows(Path(output_dir) / "valid", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("dataset")
    parser.add_argument("output_dir")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--mlp-width", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-entry-tokens", type=int, default=128)
    parser.add_argument("--max-query-tokens", type=int, default=96)
    parser.add_argument("--valid-conversations", type=int, default=2)
    parser.add_argument("--include-derived-context", action="store_true")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--hard-negative-k", type=int, default=16)
    parser.add_argument("--hard-negative-lambda", type=float, default=0.2)
    parser.add_argument("--hard-negative-margin", type=float, default=0.5)
    parser.add_argument("--early-stop-rounds", type=int, default=6)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"),
        default="auto")
    parser.add_argument(
        "--pooling", choices=("last", "mean_last"), default="last")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    valid_ids = sorted(order[:args.valid_conversations])
    train_ids = sorted(set(range(len(samples))) - set(valid_ids))
    cache_path = Path(args.cache)
    started = time.time()
    if cache_path.exists():
        conversations = load_cache(cache_path)
        print(json.dumps({
            "phase": "cache_loaded",
            "path": str(cache_path),
            "seconds": time.time() - started,
        }), flush=True)
    else:
        weights = GGUFWeights(args.gguf, args.lib)
        if args.device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            device = args.device
        backbone = TorchBackbone(
            weights, device=device, dtype=torch.bfloat16)
        if args.lora:
            backbone_lora, output_lora = load_lora_bundle(
                args.lora, backbone.cfg, device=device)
            backbone.backbone_lora = backbone_lora
            if output_lora is not None:
                backbone.output_lora = output_lora
        tokenizer = CTokenizer(args.tok_probe, args.gguf)
        conversations = {}
        for conversation_index, sample in enumerate(samples):
            conversations[conversation_index] = (
                build_conversation_embeddings(
                    backbone, tokenizer, sample,
                    args.include_derived_context,
                    args.max_entry_tokens, args.max_query_tokens,
                    args.pooling))
            print(json.dumps({
                "phase": "conversation_encoded",
                "conversation": conversation_index,
                "entries": len(
                    conversations[conversation_index]["entries"]),
                "queries": len(
                    conversations[conversation_index]["questions"]),
                "seconds": time.time() - started,
            }), flush=True)
        save_cache(cache_path, conversations)
        del backbone
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    hidden = next(iter(conversations.values()))["entry_vectors"].shape[1]
    if args.device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = args.device
    model = DualProjection(hidden, args.rank, args.mlp_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.02)
    train_queries, train_entries, train_positives = (
        global_training_tensors(conversations, train_ids, device))
    positive_mask = torch.zeros(
        train_queries.shape[0], train_entries.shape[0],
        dtype=torch.bool, device=device)
    for row, indices in enumerate(train_positives):
        positive_mask[row, indices] = True
    best_recall = -1.0
    best_state = None
    stale_rounds = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        scores = model.scores(
            train_queries, train_entries, args.temperature)
        positive_scores = scores.masked_fill(
            ~positive_mask, float("-inf"))
        positive_lse = torch.logsumexp(positive_scores, dim=1)
        objective = (
            torch.logsumexp(scores, dim=1) - positive_lse
        ).mean()
        if args.hard_negative_lambda > 0.0:
            negative_scores = scores.masked_fill(
                positive_mask, float("-inf"))
            hard_k = min(args.hard_negative_k, negative_scores.shape[1])
            hard_lse = torch.logsumexp(
                negative_scores.topk(hard_k, dim=1).values, dim=1)
            hard_loss = F.relu(
                args.hard_negative_margin + hard_lse - positive_lse).mean()
            objective = (
                objective + args.hard_negative_lambda * hard_loss)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = retrieval_metrics(
                model, conversations, valid_ids, args.top_k,
                args.temperature, device)
            print(json.dumps({
                "epoch": epoch,
                "loss": float(objective.detach()),
                **metrics,
            }), flush=True)
            if metrics["evidence_recall"] > best_recall:
                best_recall = metrics["evidence_recall"]
                stale_rounds = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
            else:
                stale_rounds += 1
            if stale_rounds >= args.early_stop_rounds:
                print(json.dumps({
                    "early_stop": True,
                    "epoch": epoch,
                    "best_evidence_recall": best_recall,
                }), flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "rank": args.rank,
        "mlp_width": args.mlp_width,
        "hidden": hidden,
        "temperature": args.temperature,
        "pooling": args.pooling,
        "backbone_sha256": hashlib.sha256(
            Path(args.gguf).read_bytes()).hexdigest(),
        "valid_conversation_ids": valid_ids,
    }, Path(args.output_dir) / "retriever.pt")
    if model.width == 0:
        save_portable_controller(
            Path(args.output_dir) / "retriever.bnctrl",
            model, args.gguf, hidden, args.rank,
            args.temperature, args.pooling)
    else:
        print(json.dumps({
            "portable_export": "skipped",
            "reason": "BNCTRL2 MLP runtime is not enabled",
            "mlp_width": model.width,
        }), flush=True)
    metrics = retrieval_metrics(
        model, conversations, valid_ids, args.top_k,
        args.temperature, device)
    export_valid_rows(
        model, conversations, valid_ids, args.output_dir,
        args.top_k, args.temperature, device)
    print(json.dumps({
        "result": "done",
        **metrics,
        "train_conversation_ids": train_ids,
        "valid_conversation_ids": valid_ids,
        "seconds": time.time() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
