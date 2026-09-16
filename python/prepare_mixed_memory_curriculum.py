#!/usr/bin/env python3
"""Combine existing training/development domains without opening either test set."""
import argparse
import json
from pathlib import Path

from typed_memory_training import file_fingerprint, reject_evaluation_row


def combine(replay, dialogue, output):
    sources = {"replay": Path(replay), "dialogue": Path(dialogue)}
    root = Path(output)
    if root.exists():
        raise FileExistsError("use a new mixed curriculum directory")
    all_rows, manifest, worlds = {}, {}, {}
    for split in ("train", "valid"):
        rows, ids, semantic_ids = [], set(), set()
        manifest[split] = {}
        for domain, source in sources.items():
            path = source / f"{split}.jsonl"
            prefix = ("natural-" if domain == "replay" else "dialogue-") + split + "-"
            domain_worlds = set()
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                reject_evaluation_row(row)
                meta = row["metadata"]
                if not meta["world_id"].startswith(prefix):
                    raise ValueError("source row is in the wrong split")
                if row["sample_id"] in ids:
                    raise ValueError("duplicate sample identifier")
                ids.add(row["sample_id"])
                meta["training_domain"] = domain
                domain_worlds.add(meta["world_id"])
                semantic_ids.add(meta.get("semantic_world_id", meta["world_id"]))
                rows.append(row)
            if not domain_worlds:
                raise ValueError("empty source domain")
            manifest[split][domain] = {"path": str(path.resolve()), "sha256": file_fingerprint(path),
                                       "worlds": len(domain_worlds)}
        all_rows[split], worlds[split] = rows, semantic_ids
    if worlds["train"] & worlds["valid"]:
        raise ValueError("train/development semantic worlds overlap")
    root.mkdir(parents=True)
    for split, rows in all_rows.items():
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    (root / "manifest.json").write_text(json.dumps({"sources": manifest, "test_sets_opened": False,
        "selection": "worst-domain metrics with replay-domain component retention tolerance 0.02"}, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "dialogue", "output"):
        parser.add_argument(name)
    args = parser.parse_args()
    print(json.dumps(combine(args.replay, args.dialogue, args.output)), flush=True)
