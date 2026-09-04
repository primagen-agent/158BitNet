#!/usr/bin/env python3
"""Bind a legacy memory layer router or feature cache to one exact GGUF."""
from __future__ import annotations

import argparse

import torch

from model_identity import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("gguf")
    parser.add_argument("output")
    args = parser.parse_args()

    payload = torch.load(
        args.artifact, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("unsupported memory router artifact")
    digest = sha256_file(args.gguf)
    existing = payload.get("backbone_sha256")
    if existing is not None and existing != digest:
        raise ValueError(
            "artifact is already bound to a different GGUF")

    if payload.get("magic") == "BNROUTER1":
        if payload.get("version") not in (1, 2):
            raise ValueError("unsupported memory layer router version")
        payload["version"] = 2
        kind = "memory_layer_router"
    elif {
        "train_x", "train_y", "train_masses",
        "valid_x", "valid_y", "valid_masses",
    }.issubset(payload):
        kind = "memory_router_feature_cache"
    else:
        raise ValueError("unsupported memory router artifact")

    payload["backbone_sha256"] = digest
    torch.save(payload, args.output)
    print({
        "kind": kind,
        "input": args.artifact,
        "gguf": args.gguf,
        "output": args.output,
        "backbone_sha256": digest.hex(),
    })


if __name__ == "__main__":
    main()
