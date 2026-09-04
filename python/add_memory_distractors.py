#!/usr/bin/env python3
"""Add same-conversation hard-negative memory chunks to prepared JSONL."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path


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


def conversation_id(row):
    group = str(row.get("validation_group", ""))
    if group:
        return group
    locomo_id = str(row.get("locomo_id", ""))
    return locomo_id.split(":", 1)[0] if locomo_id else ""


def signature(chunk):
    return "\n".join(
        f"{message.get('role', '')}:{message.get('content', '')}"
        for message in chunk)


def evidence_chunks(row):
    query_index = int(
        row.get("query_turn_id", len(row["messages"]) - 1))
    return [
        row["messages"][int(index)]
        for index in row.get("evidence_message_indices", [])
        if 0 <= int(index) < query_index
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.distractors < 1:
        parser.error("--distractors must be positive")

    loaded = load_rows(args.source)
    pools = {}
    global_pool = []
    for _filename, row in loaded:
        chunks = evidence_chunks(row)
        pools.setdefault(conversation_id(row), []).extend(chunks)
        global_pool.extend(chunks)

    rng = random.Random(args.seed)
    output_rows = []
    skipped = 0
    for filename, source_row in loaded:
        row = copy.deepcopy(source_row)
        positives = evidence_chunks(row)
        if not positives:
            skipped += 1
            continue
        seen = {signature(chunk) for chunk in positives}
        candidates = list(pools.get(conversation_id(row), []))
        rng.shuffle(candidates)
        if len(candidates) < args.distractors:
            fallback = list(global_pool)
            rng.shuffle(fallback)
            candidates.extend(fallback)
        negatives = []
        for chunk in candidates:
            key = signature(chunk)
            if key in seen:
                continue
            seen.add(key)
            negatives.append(copy.deepcopy(chunk))
            if len(negatives) >= args.distractors:
                break
        if not negatives:
            skipped += 1
            continue
        labelled = [(chunk, 1) for chunk in positives]
        labelled.extend((chunk, 0) for chunk in negatives)
        rng.shuffle(labelled)
        query_index = int(
            row.get("query_turn_id", len(row["messages"]) - 1))
        query = copy.deepcopy(row["messages"][query_index])
        row["messages"] = [chunk for chunk, _label in labelled] + [query]
        row["query_turn_id"] = len(labelled)
        row["evidence_message_indices"] = [
            index for index, (_chunk, label) in enumerate(labelled)
            if label == 1]
        row["distractor_message_indices"] = [
            index for index, (_chunk, label) in enumerate(labelled)
            if label == 0]
        output_rows.append((filename, row))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    by_name = {}
    for filename, row in output_rows:
        by_name.setdefault(filename, []).append(row)
    for filename, rows in by_name.items():
        with (output / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "samples": len(output_rows),
        "skipped": skipped,
        "distractors": args.distractors,
        "files": {
            filename: len(rows)
            for filename, rows in sorted(by_name.items())
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
