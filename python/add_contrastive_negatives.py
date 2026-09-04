#!/usr/bin/env python3
"""Add deterministic answer negatives to prepared memory-training JSONL.

Negatives are selected only from the same input split.  Preference order:
same conversation + same category, same category, then any other answer.
This preserves train/validation isolation established by the source data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def answer(row):
    query_index = row.get("query_turn_id", len(row["messages"]) - 1)
    chunk = row["messages"][query_index]
    if not chunk or chunk[-1].get("role") != "assistant":
        raise ValueError("query chunk must end with an assistant answer")
    return str(chunk[-1].get("content", ""))


def conversation_id(row):
    locomo_id = str(row.get("locomo_id", ""))
    return locomo_id.split(":", 1)[0] if locomo_id else ""


def normalized(text):
    return " ".join(text.casefold().split())


def load_rows(source):
    paths = (
        sorted(Path(source).glob("*.jsonl"))
        if Path(source).is_dir() else [Path(source)])
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append((path.name, json.loads(line)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--negatives", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.negatives < 1:
        parser.error("--negatives must be positive")

    loaded = load_rows(args.source)
    rng = random.Random(args.seed)
    answer_pools = {}
    category_pools = {}
    global_pool = []
    for _source_name, row in loaded:
        value = answer(row)
        global_pool.append(value)
        category = row.get("category")
        cid = conversation_id(row)
        category_pools.setdefault(category, []).append(value)
        if cid:
            answer_pools.setdefault((cid, category), []).append(value)

    enriched = []
    for source_name, row in loaded:
        gold = answer(row)
        gold_norm = normalized(gold)
        cid = conversation_id(row)
        category = row.get("category")

        candidates = []
        for priority, pool in (
            (0, answer_pools.get((cid, category), [])),
            (1, category_pools.get(category, [])),
            (2, global_pool),
        ):
            shuffled = list(pool)
            rng.shuffle(shuffled)
            candidates.extend((priority, value) for value in shuffled)
        negatives = []
        seen = set()
        for _priority, candidate in candidates:
            key = normalized(candidate)
            if key in seen:
                continue
            seen.add(key)
            negatives.append(candidate)
            if len(negatives) >= args.negatives:
                break
        row["negative_answers"] = negatives
        enriched.append((source_name, row))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    by_name = {}
    for source_name, row in enriched:
        by_name.setdefault(source_name, []).append(row)
    for source_name, rows in by_name.items():
        with (output / source_name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "samples": len(enriched),
        "files": {name: len(rows) for name, rows in sorted(by_name.items())},
        "negatives_per_sample": args.negatives,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
