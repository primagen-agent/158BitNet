#!/usr/bin/env python3
"""Compile active typed events into query-to-memory activation examples."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


SOURCE_LINE = re.compile(
    r"^\[source ([^ ]+) \| ([^\]]+)\] (.*)$"
)
NO_INFORMATION = "No information available"


def active_evidence(row):
    evidence = str(row.get("evidence", ""))
    lines = evidence.splitlines()
    metadata = row.get("metadata") or {}
    events = metadata.get("typed_events") or []
    if len(lines) != len(events):
        raise ValueError(
            f"source/event count mismatch: "
            f"{row.get('sample_id', '')}"
        )
    active = []
    for line, event in zip(lines, events):
        if not event.get("active"):
            continue
        if SOURCE_LINE.fullmatch(line) is None and metadata.get("raw_episodes") is None:
            raise ValueError(
                f"invalid source line: {row.get('sample_id', '')}"
            )
        active.append((line, event))
    if not active:
        raise ValueError(
            f"typed world has no active events: "
            f"{row.get('sample_id', '')}"
        )
    return active


def current_target(row, active):
    plan = (row.get("metadata") or {}).get("query_plan") or {}
    targets = plan.get("targets") or []
    if len(targets) != 1:
        raise ValueError(
            f"current query needs one target: "
            f"{row.get('sample_id', '')}"
        )
    target = targets[0]
    matches = [
        (line, event)
        for line, event in active
        if event.get("entity") == target.get("entity")
        and event.get("predicate") == target.get("predicate")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"current query has no unique active target: "
            f"{row.get('sample_id', '')}"
        )
    return matches[0]


def compile_row(row):
    metadata = dict(row.get("metadata") or {})
    if row.get("evaluation_only") or metadata.get("evaluation_only"):
        raise ValueError("evaluation-only data cannot enter activation training")
    if metadata.get("locomo_used"):
        raise ValueError("LoCoMo rows cannot enter activation training")
    plan = metadata.get("query_plan") or {}
    intent = str(plan.get("intent", ""))
    if intent not in ("current", "null"):
        return None
    active = active_evidence(row)
    lines = [line for line, _ in active]
    evidence = "\n".join(lines)
    compiled = dict(row)
    compiled["evidence"] = evidence
    metadata["typed_activation_curriculum"] = True
    metadata["typed_activation_candidate_count"] = len(lines)
    if metadata.get("raw_episodes") is not None:
        metadata["raw_episodes"] = lines
    compiled["metadata"] = metadata
    if intent == "null":
        metadata["counterfactual_no_info"] = True
        compiled["answer"] = NO_INFORMATION
        compiled["answer_spans"] = []
        compiled["answer_type"] = "span"
        return compiled
    target_line, target_event = current_target(row, active)
    value = str(target_event.get("value", ""))
    if not value:
        raise ValueError(
            f"current target has no value: "
            f"{row.get('sample_id', '')}"
        )
    line_offset = 0
    for line in lines:
        if line == target_line:
            local_start = line.find(value)
            if local_start < 0:
                raise ValueError(
                    f"value is absent from target evidence: "
                    f"{row.get('sample_id', '')}"
                )
            start = line_offset + local_start
            end = start + len(value)
            compiled["answer"] = value
            compiled["answer_spans"] = [{
                "start": start,
                "end": end,
                "text": value,
            }]
            compiled["answer_type"] = "span"
            return compiled
        line_offset += len(line) + 1
    raise AssertionError("active target line was lost")


def compile_file(input_path, output_path):
    counters = collections.Counter()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open(encoding="utf-8") as source, \
            output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            counters["input"] += 1
            row = compile_row(json.loads(line))
            if row is None:
                counters["skipped_intent"] += 1
                continue
            intent = row["metadata"]["query_plan"]["intent"]
            counters[intent] += 1
            target.write(
                json.dumps(
                    row, ensure_ascii=False,
                    separators=(",", ":")
                )
                + "\n"
            )
            counters["output"] += 1
    return dict(counters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    args = parser.parse_args()
    result = compile_file(
        args.input_jsonl, args.output_jsonl
    )
    print(json.dumps(
        {
            "phase": "typed_activation_compile",
            "input": args.input_jsonl,
            "output": args.output_jsonl,
            **result,
            "locomo_used": False,
        },
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    main()
