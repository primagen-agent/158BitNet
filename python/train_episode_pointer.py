#!/usr/bin/env python3
"""Train an episode-local neural span pointer for persistent memory.

The correct source episode is supplied only to isolate pointer capacity during
this experiment. Source text is never appended to the answer prompt. The
frozen matching 0.5B backbone encodes source and query independently; a small
trainable bilinear head predicts the answer span inside that one episode.
"""
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

from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_data import (
    CTokenDecoder,
    CTokenizer,
    dataset_order,
    load_dataset,
    render_chunk,
)
from train_memory import generation_query_evidence_map


def find_unique_token_span(source_ids, target_ids):
    source = [int(token) for token in source_ids]
    target = [int(token) for token in target_ids]
    if not target or len(target) > len(source):
        return None
    matches = [
        start
        for start in range(len(source) - len(target) + 1)
        if source[start:start + len(target)] == target
    ]
    if len(matches) != 1:
        return None
    return matches[0], matches[0] + len(target) - 1


def best_spans(
    start_logits, end_logits, mask, max_span, length_logits=None
):
    batch, length = start_logits.shape
    positions = torch.arange(length, device=start_logits.device)
    left = positions.view(1, length, 1)
    right = positions.view(1, 1, length)
    valid = (
        (right >= left)
        & (right - left < max_span)
        & mask[:, :, None]
        & mask[:, None, :]
    )
    scores = (
        start_logits.unsqueeze(-1) + end_logits.unsqueeze(-2)
    ).masked_fill(~valid, -1e9)
    if length_logits is not None:
        lengths = (right - left).clamp(
            min=0, max=max_span - 1).expand(batch, -1, -1)
        length_scores = length_logits.gather(
            1, lengths.reshape(batch, -1)
        ).reshape(batch, length, length)
        scores = scores + length_scores
    best = scores.reshape(batch, -1).argmax(dim=-1)
    return best // length, best % length


class EpisodeSpanPointer(nn.Module):
    """Query-conditioned start/end pointer over one activated episode."""

    def __init__(self, hidden, rank, max_span, use_length_head=False):
        super().__init__()
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.query_start = nn.Linear(hidden, rank, bias=False)
        self.query_end = nn.Linear(hidden, rank, bias=False)
        self.source_start = nn.Linear(hidden, rank, bias=False)
        self.source_end = nn.Linear(hidden, rank, bias=False)
        self.local_start = nn.Linear(hidden, 1)
        self.local_end = nn.Linear(hidden, 1)
        self.local_scale = nn.Parameter(torch.full((2,), -2.0))
        self.length_head = (
            nn.Sequential(
                nn.Linear(hidden, rank),
                nn.SiLU(),
                nn.Linear(rank, max_span),
            )
            if use_length_head else None)

    def forward(self, source, source_mask, query, query_mask):
        start_query = F.normalize(
            self.query_start(query), dim=-1, eps=1e-12)
        end_query = F.normalize(
            self.query_end(query), dim=-1, eps=1e-12)
        start_source = F.normalize(
            self.source_start(source), dim=-1, eps=1e-12)
        end_source = F.normalize(
            self.source_end(source), dim=-1, eps=1e-12)
        scale = math.sqrt(self.rank)
        start_match = torch.einsum(
            "bqr,bsr->bqs", start_query, start_source)
        end_match = torch.einsum(
            "bqr,bsr->bqs", end_query, end_source)
        query_valid = query_mask.unsqueeze(-1)
        start_match = start_match.masked_fill(~query_valid, -1e9)
        end_match = end_match.masked_fill(~query_valid, -1e9)
        query_weight = query_mask.to(query.dtype)
        query_summary = (
            (query * query_weight.unsqueeze(-1)).sum(dim=1)
            / query_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        local_scale = torch.sigmoid(self.local_scale)
        start = (
            start_match.max(dim=1).values * scale
            + local_scale[0] * self.local_start(source).squeeze(-1)
        )
        end = (
            end_match.max(dim=1).values * scale
            + local_scale[1] * self.local_end(source).squeeze(-1)
        )
        return (
            start.masked_fill(~source_mask, -1e9),
            end.masked_fill(~source_mask, -1e9),
            (
                self.length_head(query_summary)
                if self.length_head is not None else None
            ),
        )


def export_episode_pointer_binary(checkpoint_path, output_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "EPISODE_POINTER_V2":
        raise ValueError(
            "C export requires the no-length-head EPISODE_POINTER_V2 model")
    state = checkpoint["state_dict"]
    tensor_names = (
        "query_start.weight",
        "query_end.weight",
        "source_start.weight",
        "source_end.weight",
        "local_start.weight",
        "local_start.bias",
        "local_end.weight",
        "local_end.bias",
        "local_scale",
    )
    payloads = []
    for name in tensor_names:
        tensor = state[name].detach().float().contiguous()
        payloads.append(tensor.numpy().astype(
            "<f4", copy=False).tobytes())
    header = bytearray(b"BNEPTR1\x00")
    header += struct.pack(
        "<IIII",
        1,
        int(checkpoint["hidden"]),
        int(checkpoint["rank"]),
        int(checkpoint["max_span"]),
    )
    header += bytes.fromhex(checkpoint["backbone_sha256"])
    header += struct.pack("<I", len(payloads))
    for payload in payloads:
        header += struct.pack(
            "<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + b"".join(payloads))
    return output


def dataset_fingerprint(data_dir):
    digest = hashlib.sha256()
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def iter_examples(data_dir, seed, limit):
    strata = load_dataset(data_dir)
    order = dataset_order(strata, seed)
    if limit > 0:
        order = order[:limit]
    for stratum_index, line_index in order:
        yield json.loads(strata[stratum_index][1][line_index])


@torch.inference_mode()
def encode_examples(
    data_dir, backbone, tokenizer, decoder, seed, limit,
    max_source_tokens, max_query_tokens, max_span,
):
    rows = []
    counters = {
        "samples": 0,
        "queries": 0,
        "missing_evidence": 0,
        "non_copy": 0,
        "too_long": 0,
    }
    for sample in iter_examples(data_dir, seed, limit):
        counters["samples"] += 1
        chunks = sample["messages"]
        query_value = sample.get("query_turn_id", len(chunks) - 1)
        query_indices = (
            [int(index) for index in query_value]
            if isinstance(query_value, list)
            else [int(query_value)]
        )
        evidence_map = generation_query_evidence_map(
            sample, query_indices)
        for query_index in query_indices:
            counters["queries"] += 1
            evidence = evidence_map.get(
                str(query_index), evidence_map.get(query_index))
            evidence_indices = (
                [int(index) for index in evidence]
                if isinstance(evidence, list)
                else ([] if evidence is None else [int(evidence)])
            )
            if not evidence_indices:
                counters["missing_evidence"] += 1
                continue
            query_text, target = render_chunk(
                chunks[query_index], True)
            target = target.strip()
            target_ids = tokenizer.encode(target, add_bos=False)
            matches = []
            encoded_sources = []
            for evidence_index in evidence_indices:
                source_text, _ = render_chunk(
                    chunks[evidence_index], False)
                source_ids = tokenizer.encode(
                    source_text, add_bos=True)[:max_source_tokens]
                span = find_unique_token_span(source_ids, target_ids)
                if span is not None:
                    matches.append((len(encoded_sources), span))
                encoded_sources.append((
                    evidence_index, source_text, source_ids))
            if len(matches) != 1:
                counters["non_copy"] += 1
                continue
            source_offset, (start, end) = matches[0]
            if end - start + 1 > max_span:
                counters["too_long"] += 1
                continue
            evidence_index, source_text, source_ids = (
                encoded_sources[source_offset])
            query_ids = tokenizer.encode(
                query_text, add_bos=True)[-max_query_tokens:]
            source_hidden = backbone(
                torch.tensor(
                    source_ids, device=backbone.device,
                    dtype=torch.long),
                return_hidden=True,
            ).detach().cpu().to(torch.float16)
            query_hidden = backbone(
                torch.tensor(
                    query_ids, device=backbone.device,
                    dtype=torch.long),
                return_hidden=True,
            ).detach().cpu().to(torch.float16)
            rows.append({
                "source_hidden": source_hidden,
                "query_hidden": query_hidden,
                "source_ids": source_ids,
                "start": int(start),
                "end": int(end),
                "target": target,
                "sample_id": sample.get("sample_id", ""),
                "query_index": int(query_index),
                "evidence_index": int(evidence_index),
            })
            if len(rows) % 50 == 0:
                print(json.dumps({
                    "phase": "encode_episode_pointer",
                    "rows": len(rows),
                    **counters,
                }, separators=(",", ":")), flush=True)
    print(json.dumps({
        "phase": "encode_episode_pointer_done",
        "rows": len(rows),
        **counters,
    }, separators=(",", ":")), flush=True)
    return rows


def load_or_encode(
    data_dir, cache_path, backbone, tokenizer, decoder, seed, limit,
    max_source_tokens, max_query_tokens, max_span, backbone_sha,
):
    metadata = {
        "format": "EPISODE_POINTER_FEATURES_V2",
        "backbone_sha256": backbone_sha,
        "dataset_fingerprint": dataset_fingerprint(data_dir),
        "seed": int(seed),
        "limit": int(limit),
        "max_source_tokens": int(max_source_tokens),
        "max_query_tokens": int(max_query_tokens),
        "max_span": int(max_span),
    }
    path = Path(cache_path)
    if path.is_file():
        payload = torch.load(
            path, map_location="cpu", weights_only=True)
        if all(payload.get(key) == value
               for key, value in metadata.items()):
            print(json.dumps({
                "phase": "episode_pointer_cache",
                "path": str(path),
                "rows": len(payload["rows"]),
            }, separators=(",", ":")), flush=True)
            return payload["rows"]
    rows = encode_examples(
        data_dir, backbone, tokenizer, decoder, seed, limit,
        max_source_tokens, max_query_tokens, max_span)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "rows": rows}, path)
    return rows


def collate(rows, device):
    source_length = max(row["source_hidden"].shape[0] for row in rows)
    query_length = max(row["query_hidden"].shape[0] for row in rows)
    hidden = rows[0]["source_hidden"].shape[-1]
    source = torch.zeros(
        len(rows), source_length, hidden, device=device)
    query = torch.zeros(
        len(rows), query_length, hidden, device=device)
    source_mask = torch.zeros(
        len(rows), source_length, device=device, dtype=torch.bool)
    query_mask = torch.zeros(
        len(rows), query_length, device=device, dtype=torch.bool)
    for index, row in enumerate(rows):
        source_count = row["source_hidden"].shape[0]
        query_count = row["query_hidden"].shape[0]
        source[index, :source_count] = row["source_hidden"].to(
            device=device, dtype=torch.float32)
        query[index, :query_count] = row["query_hidden"].to(
            device=device, dtype=torch.float32)
        source_mask[index, :source_count] = True
        query_mask[index, :query_count] = True
    return {
        "source": source,
        "source_mask": source_mask,
        "query": query,
        "query_mask": query_mask,
        "start": torch.tensor(
            [row["start"] for row in rows],
            device=device, dtype=torch.long),
        "end": torch.tensor(
            [row["end"] for row in rows],
            device=device, dtype=torch.long),
    }


@torch.inference_mode()
def evaluate(model, rows, decoder, device, batch_size, max_span):
    model.eval()
    correct_span = 0
    correct_text = 0
    total = 0
    failures = []
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset:offset + batch_size]
        batch = collate(selected, device)
        start_logits, end_logits, length_logits = model(
            batch["source"], batch["source_mask"],
            batch["query"], batch["query_mask"])
        predicted_start, predicted_end = best_spans(
            start_logits, end_logits, batch["source_mask"], max_span,
            length_logits)
        for index, row in enumerate(selected):
            start = int(predicted_start[index])
            end = int(predicted_end[index])
            span_correct = (
                start == row["start"] and end == row["end"])
            prediction = decoder.decode(
                row["source_ids"][start:end + 1]).strip()
            text_correct = prediction == row["target"]
            correct_span += int(span_correct)
            correct_text += int(text_correct)
            total += 1
            if not text_correct and len(failures) < 8:
                failures.append({
                    "sample_id": row["sample_id"],
                    "target": row["target"],
                    "prediction": prediction,
                    "gold_span": [row["start"], row["end"]],
                    "predicted_span": [start, end],
                })
    return {
        "examples": total,
        "span_exact": correct_span / max(total, 1),
        "text_exact": correct_text / max(total, 1),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("train_data")
    parser.add_argument("valid_data")
    parser.add_argument("output")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--valid-limit", type=int, default=300)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--max-query-tokens", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=160)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--length-head", action="store_true",
        help="add the experimental query-only span-length classifier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    backbone_sha = hashlib.sha256(
        Path(args.gguf).read_bytes()).hexdigest()
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(
        weights, device=args.device, dtype=torch.bfloat16)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)
    train_rows = load_or_encode(
        args.train_data, args.train_cache,
        backbone, tokenizer, decoder,
        args.seed + 1, args.train_limit,
        args.max_source_tokens, args.max_query_tokens,
        args.max_span, backbone_sha)
    valid_rows = load_or_encode(
        args.valid_data, args.valid_cache,
        backbone, tokenizer, decoder,
        args.seed + 2, args.valid_limit,
        args.max_source_tokens, args.max_query_tokens,
        args.max_span, backbone_sha)
    if not train_rows or not valid_rows:
        raise RuntimeError(
            "episode pointer requires copyable train and valid examples")
    hidden = backbone.cfg.hidden
    weights.close()
    del backbone
    torch.cuda.empty_cache()

    model = EpisodeSpanPointer(
        hidden, args.rank, args.max_span,
        use_length_head=args.length_head).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    best_text_exact = -1.0
    stale_evaluations = 0
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train_rows),
            (min(args.batch, len(train_rows)),),
            generator=generator,
        ).tolist()
        batch = collate(
            [train_rows[index] for index in indices], args.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start_logits, end_logits, length_logits = model(
            batch["source"], batch["source_mask"],
            batch["query"], batch["query_mask"])
        loss = (
            F.cross_entropy(start_logits, batch["start"])
            + F.cross_entropy(end_logits, batch["end"])
        )
        if length_logits is not None:
            loss = loss + F.cross_entropy(
                length_logits, batch["end"] - batch["start"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every != 0 and step != args.steps:
            continue
        metrics = evaluate(
            model, valid_rows, decoder, args.device,
            args.batch, args.max_span)
        print(json.dumps({
            "phase": "episode_pointer_valid",
            "step": step,
            "loss": float(loss.detach()),
            **metrics,
        }, ensure_ascii=False, separators=(",", ":")), flush=True)
        if metrics["text_exact"] > best_text_exact:
            best_text_exact = metrics["text_exact"]
            stale_evaluations = 0
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": (
                    "EPISODE_POINTER_V3"
                    if args.length_head
                    else "EPISODE_POINTER_V2"
                ),
                "backbone_sha256": backbone_sha,
                "hidden": hidden,
                "rank": args.rank,
                "max_source_tokens": args.max_source_tokens,
                "max_query_tokens": args.max_query_tokens,
                "max_span": args.max_span,
                "length_head": bool(args.length_head),
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "valid_metrics": metrics,
            }, args.output)
        else:
            stale_evaluations += 1
            if stale_evaluations >= args.patience:
                print(json.dumps({
                    "phase": "episode_pointer_early_stop",
                    "step": step,
                    "best_text_exact": best_text_exact,
                    "stale_evaluations": stale_evaluations,
                }, separators=(",", ":")), flush=True)
                break
    print(json.dumps({
        "phase": "episode_pointer_done",
        "output": args.output,
        "best_text_exact": best_text_exact,
        "backbone_sha256": backbone_sha,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
