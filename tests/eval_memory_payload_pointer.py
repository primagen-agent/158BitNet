#!/usr/bin/env python3
"""Report exact payload extraction by operation and token length."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_memory_v89_copy_pointer import (  # noqa: E402
    BytePointer,
    collate_byte_positions,
    evaluate_byte,
    reconstruction_paths,
    select_best_byte_span,
)


def load_metadata(data_dir: str) -> dict[str, dict]:
    result = {}
    for path in reconstruction_paths(data_dir):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            payload = sample["messages"][1][-1]["content"]
            result[payload] = sample["metadata"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("cache")
    parser.add_argument("data")
    parser.add_argument(
        "--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--show-errors", type=int, default=0)
    parser.add_argument(
        "--inside-mode",
        choices=("sum", "sqrt", "mean", "none"), default="sum")
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    rows = torch.load(
        args.cache, map_location="cpu", weights_only=False)
    metadata = load_metadata(args.data)
    model = BytePointer(
        checkpoint["hidden"], checkpoint["pointer_rank"],
        checkpoint["max_token_bytes"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    groups = defaultdict(list)
    for row in rows:
        item = metadata.get(row["payload"])
        if item is None:
            raise RuntimeError(
                f"missing metadata for payload {row['payload']!r}")
        groups[("operation", item["operation"])].append(row)
        groups[("length", str(item["token_length"]))].append(row)
        groups[
            ("payload_family", item.get("payload_family", "synthetic"))
        ].append(row)
        groups[
            (
                "operation_length",
                f"{item['operation']}/len{item['token_length']}",
            )
        ].append(row)
    report = {
        f"{kind}:{name}": evaluate_byte(
            model, group, args.device,
            checkpoint["max_copy_bytes"], args.inside_mode)
        for (kind, name), group in sorted(groups.items())
    }
    if args.show_errors > 0:
        errors = []
        for row in rows:
            item = metadata[row["payload"]]
            (
                token_hidden, byte_values, previous_bytes, next_bytes,
                token_offsets, mask, _starts, _ends,
            ) = collate_byte_positions([row], args.device)
            with torch.no_grad():
                start_scores, end_scores, inside_scores = model(
                    token_hidden, byte_values, previous_bytes,
                    next_bytes, token_offsets, mask)
            count = len(row["record_bytes"])
            start, end = select_best_byte_span(
                start_scores[0], end_scores[0], inside_scores[0],
                count, checkpoint["max_copy_bytes"],
                args.inside_mode)
            predicted = row["record_bytes"][start:end + 1]
            if predicted != row["payload_bytes"]:
                errors.append({
                    "operation": item["operation"],
                    "token_length": item["token_length"],
                    "payload_family": item.get(
                        "payload_family", "synthetic"),
                    "expected": row["payload"],
                    "predicted": predicted.decode(
                        "utf-8", errors="replace"),
                    "record": row["record_bytes"].decode(
                        "utf-8", errors="replace"),
                })
                if len(errors) >= args.show_errors:
                    break
        report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
