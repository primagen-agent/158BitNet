#!/usr/bin/env python3
"""Train the optional typed query compatibility head from retained pair features.

The resident reader bypasses this head. The retained feature rows are the
actual supervised query-training inputs, so raw source feature caches are not
needed for a new query-head experiment.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
sys.path.insert(0, str(REPO / "python"))

from train_typed_query_activator import TypedQueryActivator, train, QUERY_ACTIVATOR, SET_FEATURE_COUNT
from typed_memory_training import clone_state_dict, file_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="new .pt checkpoint path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    pair_path = PACKAGE / "checkpoints/pair.pt"
    pair = torch.load(pair_path, map_location="cpu", weights_only=True)
    cache = []
    for split in ("train", "valid"):
        item = torch.load(PACKAGE / f"feature_data/query_{split}.pt", map_location="cpu", weights_only=True)
        if (item.get("format") != "TYPED_QUERY_PAIR_FEATURES_V1"
                or item.get("pair_checkpoint_fingerprint") != file_fingerprint(pair_path)
                or item.get("backbone_sha256") != pair["backbone_sha256"]
                or item.get("evaluation_only")):
            raise ValueError("query feature provenance mismatch")
        cache.append(item["rows"])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = TypedQueryActivator(pair).to(args.device)
    metrics = train(model, cache[0], cache[1], args.device, args.steps,
                    8, 2e-4, 0.02, 100, 8, 0.5, args.seed + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": QUERY_ACTIVATOR,
        "backbone_sha256": pair["backbone_sha256"],
        "pair_checkpoint_fingerprint": file_fingerprint(pair_path),
        "rank": model.rank,
        "input_width": model.input_width,
        "set_feature_count": SET_FEATURE_COUNT,
        "state_dict": clone_state_dict(model),
        "valid_metrics": metrics,
        "objective": "candidate_cross_entropy_plus_hard_margin_with_candidate_set_null",
        "kv_cache_used": False,
        "lora_used": False,
        "locomo_used": False,
    }, output)
    print(json.dumps({"output": str(output), "development": metrics}))


if __name__ == "__main__":
    main()
