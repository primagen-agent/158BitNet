#!/usr/bin/env python3
"""Freeze a raw-chat lifecycle fixture from a disjoint natural curriculum split."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("output")
    parser.add_argument("--role", choices=("development", "final"), required=True)
    args = parser.parse_args()
    worlds = {}
    for line in Path(args.jsonl).read_text().splitlines():
        row = json.loads(line)
        meta = row["metadata"]
        split = "test" if args.role == "final" else "valid"
        if not meta["world_id"].startswith("natural-" + split + "-") and not (
                meta.get("split") == split and meta["world_id"].startswith("dialogue-" + split + "-")):
            raise ValueError("wrong data split for this evaluation role")
        item = worlds.setdefault(meta["world_id"], {
            "id": meta["world_id"], "writes": meta["raw_episodes"], "queries": [],
        })
        kind = meta["query_plan"]["intent"]
        item["queries"].append({"kind": kind, "text": row["question"],
                                "answer": None if kind == "null" else row["answer"]})
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an evaluation fixture")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"format": "MEMORY_CHAT_HOLDOUT_V1", "evaluation_only": True,
                                 "evaluation_role": args.role,
                                 "source_sha256": hashlib.sha256(Path(args.jsonl).read_bytes()).hexdigest(),
                                 "generalization_axis": meta.get("generalization_axis", "disjoint_entities_and_relations_shared_surface_templates"),
                                 "worlds": list(worlds.values())}, indent=2) + "\n")


if __name__ == "__main__":
    main()
