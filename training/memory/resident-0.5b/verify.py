#!/usr/bin/env python3
"""Verify the retained training package and re-export every bundled model."""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
sys.path.insert(0, str(REPO / "python"))

from export_resident_identity import export as export_resident
from export_typed_pair_encoder import export_typed_pair_encoder_binary
from export_typed_writer_model import export_typed_writer_binary
from train_typed_pair_verifier import export_typed_link_binary
from train_typed_query_activator import export_typed_query_activator_binary


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", help="the exact separately supplied 0.5B GGUF")
    args = parser.parse_args()
    provenance = json.loads((PACKAGE / "manifest.json").read_text())
    deployment = json.loads((REPO / "models/memory/resident-0.5b/manifest.json").read_text())
    backbone = digest(Path(args.gguf))
    assert backbone == deployment["backbone"]["sha256"] == provenance["backbone_sha256"]

    for group in ("corpora", "checkpoints", "feature_data"):
        for name, expected in provenance[group].items():
            path = PACKAGE / name
            assert digest(path) == expected, f"training artifact mismatch: {name}"

    checkpoints = PACKAGE / "checkpoints"
    resident = torch.load(REPO / "models/memory/resident-0.5b/resident.pt", map_location="cpu", weights_only=True)
    baseline = checkpoints / "baseline.pt"
    writer = checkpoints / "writer.pt"
    pair = checkpoints / "pair.pt"
    assert resident["backbone_sha256"] == backbone
    assert resident["train_sha256"] == provenance["corpora"]["corpora/resident/train.jsonl"]
    assert resident["valid_sha256"] == provenance["corpora"]["corpora/resident/valid.jsonl"]
    assert resident["baseline_sha256"] == digest(baseline)
    old = torch.load(baseline, map_location="cpu", weights_only=True)
    assert old["train_sha256"] == resident["train_sha256"]
    assert old["valid_sha256"] == resident["valid_sha256"]
    for name in ("tagger.pt", "keys.pt", "operation.pt"):
        item = torch.load(checkpoints / name, map_location="cpu", weights_only=True)
        assert item["writer_checkpoint_fingerprint"] == digest(writer), name
    query = torch.load(checkpoints / "query.pt", map_location="cpu", weights_only=True)
    assert query["pair_checkpoint_fingerprint"] == digest(pair)
    link = torch.load(checkpoints / "link.pt", map_location="cpu", weights_only=True)
    assert link["parent_checkpoint_sha256"] == digest(pair)
    for split, expected_rows in (("train", 192), ("valid", 24)):
        cache = torch.load(PACKAGE / f"feature_data/query_{split}.pt", map_location="cpu", weights_only=True)
        assert cache["pair_checkpoint_fingerprint"] == digest(pair)
        assert cache["backbone_sha256"] == backbone
        assert len(cache["rows"]) == expected_rows
        assert not cache.get("evaluation_only", False)
        with (PACKAGE / f"corpora/query/{split}.jsonl").open() as source:
            raw_ids = [json.loads(line)["sample_id"] for line in source]
        feature_ids = [row["sample_id"] for row in cache["rows"]]
        assert len(set(raw_ids)) == expected_rows
        assert set(raw_ids) == set(feature_ids), f"query {split} IDs differ"

    models = REPO / "models/memory/resident-0.5b"
    with tempfile.TemporaryDirectory(prefix="resident-export-") as temporary:
        output = Path(temporary)
        export_resident(models / "resident.pt", output / "resident.bnresid")
        export_typed_writer_binary(
            writer, checkpoints / "tagger.pt", checkpoints / "keys.pt",
            output / "writer.bntwrite", 0, 0, checkpoints / "operation.pt")
        export_typed_pair_encoder_binary(pair, output / "pair.bntpair")
        export_typed_link_binary(checkpoints / "link.pt", output / "link.bntlink")
        export_typed_query_activator_binary(
            checkpoints / "query.pt", pair, models / "pair.bntpair", output / "query.bntqact")
        for name in ("resident.bnresid", "writer.bntwrite", "pair.bntpair", "link.bntlink", "query.bntqact"):
            expected = deployment["files"][name]["sha256"]
            assert digest(output / name) == digest(models / name) == expected, name
            print(f"VERIFIED {name} {expected}")


if __name__ == "__main__":
    main()
