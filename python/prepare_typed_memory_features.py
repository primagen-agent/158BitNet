#!/usr/bin/env python3
"""Build frozen-backbone feature caches for typed-event memory training."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from typed_memory_training import (
    encode_backbone_features,
    file_fingerprint,
    parse_hidden_layer_bands,
    reject_evaluation_row,
)


SOURCE_LINE = re.compile(r"(?m)^\[source [^\n]+\].*$")
NO_INFORMATION = "no information available"


def split_evidence_episodes(evidence):
    return [
        {
            "text": match.group(0),
            "char_start": match.start(),
            "char_end": match.end(),
        }
        for match in SOURCE_LINE.finditer(evidence)
    ]


def is_null_example(row):
    metadata = row.get("metadata") or {}
    answer = row.get("answer")
    return (
        bool(metadata.get("counterfactual_no_info"))
        or (
            isinstance(answer, str)
            and answer.strip().casefold() == NO_INFORMATION
        )
    )


def compile_episode_native_row(row):
    evidence = str(row.get("evidence", ""))
    question = str(row.get("question", "")).strip()
    episodes = split_evidence_episodes(evidence)
    raw_episodes = (row.get("metadata") or {}).get("raw_episodes")
    if raw_episodes is not None:
        if not raw_episodes or any(not isinstance(text, str) or not text or "\n" in text for text in raw_episodes):
            raise ValueError("raw episodes must be nonempty single-line messages")
        if evidence != "\n".join(raw_episodes):
            raise ValueError("raw episode evidence mismatch")
        episodes = []
        offset = 0
        for text in raw_episodes:
            episodes.append({"text": text, "char_start": offset, "char_end": offset + len(text)})
            offset += len(text) + 1
    if not question or len(episodes) < 1:
        return None
    null_target = is_null_example(row)
    gold = set()
    if not null_target:
        for span in row.get("answer_spans") or ():
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            if start < 0 or end <= start:
                continue
            for index, episode in enumerate(episodes):
                if (
                    start < episode["char_end"]
                    and end > episode["char_start"]
                ):
                    gold.add(index)
        if not gold:
            return None
    metadata = row.get("metadata") or {}
    return {
        "sample_id": str(row.get("sample_id", "")),
        "world_id": str(metadata.get(
            "world_id", row.get("sample_id", ""))),
        "question": question,
        "episodes": [episode["text"] for episode in episodes],
        "gold_episode_indices": sorted(gold),
        "null_target": null_target,
        "family": str(metadata.get("family", "")),
    }


def load_episode_native_jsonl(path, seed, limit):
    rows = []
    counters = {
        "input": 0,
        "coqa_skipped": 0,
        "unlabelled_skipped": 0,
        "positive": 0,
        "null": 0,
        "multi_positive": 0,
    }
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            reject_evaluation_row(raw)
            counters["input"] += 1
            if (raw.get("metadata") or {}).get("source") == "CoQA":
                counters["coqa_skipped"] += 1
                continue
            compiled = compile_episode_native_row(raw)
            if compiled is None:
                counters["unlabelled_skipped"] += 1
                continue
            rows.append(compiled)
            if compiled["null_target"]:
                counters["null"] += 1
            else:
                counters["positive"] += 1
                counters["multi_positive"] += int(
                    len(compiled["gold_episode_indices"]) > 1)
    random.Random(seed).shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    print(json.dumps({
        "phase": "typed_feature_compile",
        "rows": len(rows),
        **counters,
    }, separators=(",", ":")), flush=True)
    return rows


@torch.inference_mode()
def encode_rows(
    rows, backbone, tokenizer, layer_bands,
    max_episode_tokens, max_query_tokens,
):
    encoded = []
    episode_cache = {}
    for row in rows:
        episode_key = tuple(row["episodes"])
        cached = episode_cache.get(episode_key)
        if cached is None:
            episode_ids = []
            episode_hidden = []
            for episode in row["episodes"]:
                token_ids = tokenizer.encode(episode, add_bos=True)
                if len(token_ids) > max_episode_tokens:
                    raise ValueError("episode exceeds token budget; refusing to truncate supervised fields")
                episode_ids.append(token_ids)
                episode_hidden.append(encode_backbone_features(
                    backbone, token_ids, layer_bands))
            cached = (episode_ids, episode_hidden)
            episode_cache[episode_key] = cached
        episode_ids, episode_hidden = cached
        query_ids = tokenizer.encode(
            row["question"], add_bos=True)[-max_query_tokens:]
        encoded.append({
            **row,
            "episode_ids": episode_ids,
            "episode_hidden": episode_hidden,
            "query_ids": query_ids,
            "query_hidden": encode_backbone_features(
                backbone, query_ids, layer_bands),
        })
        if len(encoded) % 100 == 0:
            print(json.dumps({
                "phase": "typed_feature_encode",
                "encoded": len(encoded),
                "total": len(rows),
                "unique_episode_banks": len(episode_cache),
            }, separators=(",", ":")), flush=True)
    return encoded


def load_or_encode(
    jsonl_path, cache_path, backbone, tokenizer,
    backbone_sha, seed, limit, layer_bands,
    max_episode_tokens, max_query_tokens,
):
    metadata = {
        "format": "MULTISLOT_EPISODE_FEATURES_V1",
        "backbone_sha256": backbone_sha,
        "source_fingerprint": file_fingerprint(jsonl_path),
        "seed": int(seed),
        "limit": int(limit),
        "hidden_layer_bands": [
            list(band) for band in layer_bands],
        "max_episode_tokens": int(max_episode_tokens),
        "max_query_tokens": int(max_query_tokens),
    }
    path = Path(cache_path)
    if path.exists():
        payload = torch.load(
            path, map_location="cpu", weights_only=True)
        if all(
            payload.get(key) == value
            for key, value in metadata.items()
        ):
            print(json.dumps({
                "phase": "typed_feature_cache_hit",
                "path": str(path),
                "rows": len(payload["rows"]),
            }, separators=(",", ":")), flush=True)
            return payload["rows"]
    rows = load_episode_native_jsonl(
        jsonl_path, seed, limit)
    encoded = encode_rows(
        rows, backbone, tokenizer, layer_bands,
        max_episode_tokens, max_query_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "rows": encoded}, path)
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("train_jsonl")
    parser.add_argument("valid_jsonl")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--hidden-layer-bands", required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--max-episode-tokens", type=int, default=64)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    backbone_sha = hashlib.sha256(
        Path(args.gguf).read_bytes()).hexdigest()
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(
        weights, device=args.device, dtype=torch.bfloat16)
    layer_bands = parse_hidden_layer_bands(
        args.hidden_layer_bands, backbone.cfg.n_layers)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    train_rows = load_or_encode(
        args.train_jsonl, args.train_cache,
        backbone, tokenizer, backbone_sha,
        args.seed + 1, args.train_limit,
        layer_bands, args.max_episode_tokens,
        args.max_query_tokens)
    valid_rows = load_or_encode(
        args.valid_jsonl, args.valid_cache,
        backbone, tokenizer, backbone_sha,
        args.seed + 2, args.valid_limit,
        layer_bands, args.max_episode_tokens,
        args.max_query_tokens)
    weights.close()
    print(json.dumps({
        "phase": "typed_feature_encode_done",
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_cache": args.train_cache,
        "valid_cache": args.valid_cache,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
