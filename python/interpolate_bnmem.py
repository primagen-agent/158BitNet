#!/usr/bin/env python3
"""Linearly interpolate two shape-compatible BNMEM checkpoints."""

from __future__ import annotations

import argparse

from bnmem_export import load_bnmem_v1, save_bnmem_v3


METADATA_KEYS = (
    "layer_ids", "d_model", "kv_dim", "q_dim", "head_dim", "gamma",
    "tau", "rho", "k_min", "beta_scale", "denom_mode", "query_rank",
    "kv_rank", "query_add_backbone",
    "backbone_sha256",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("output")
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    base = load_bnmem_v1(args.base)
    other = load_bnmem_v1(args.other)
    for key in METADATA_KEYS:
        if base[key] != other[key]:
            raise ValueError(f"checkpoint metadata mismatch: {key}")
    if base["tensors"].keys() != other["tensors"].keys():
        raise ValueError("checkpoint tensor sets differ")

    tensors = {}
    for name, base_tensor in base["tensors"].items():
        other_tensor = other["tensors"][name]
        if base_tensor.shape != other_tensor.shape:
            raise ValueError(f"checkpoint tensor shape mismatch: {name}")
        tensors[name] = base_tensor.lerp(other_tensor, args.alpha)

    kv = (
        {
            "wk_a": tensors["wk_a"], "wk_b": tensors["wk_b"],
            "wv_a": tensors["wv_a"], "wv_b": tensors["wv_b"],
        }
        if base["kv_rank"] > 0 else
        {"wk": tensors["wk"], "wv": tensors["wv"]}
    )
    query = (
        {
            "query_a": tensors["query_a"],
            "query_b": tensors["query_b"],
            "query_norm": tensors["query_norm"],
        }
        if base["query_rank"] > 0 else
        {
            "query_proj": tensors["query_proj"],
            "query_norm": tensors["query_norm"],
        }
    )
    save_bnmem_v3(
        args.output,
        layer_ids=base["layer_ids"],
        d_model=base["d_model"],
        kv_dim=base["kv_dim"],
        q_dim=base["q_dim"],
        head_dim=base["head_dim"],
        gamma=base["gamma"],
        tau=base["tau"],
        rho=base["rho"],
        k_min=base["k_min"],
        gdu_ab=tensors["gdu_ab"],
        gdu_bb=tensors["gdu_bb"],
        beta_scale=base["beta_scale"],
        w_agg=tensors["w_agg"],
        gdu_aw=tensors["gdu_aw"],
        gdu_bw=tensors["gdu_bw"],
        mem_norm=tensors["mem_norm"],
        denom_mode=base["denom_mode"],
        query_add_backbone=base["query_add_backbone"],
        backbone_sha256=base["backbone_sha256"],
        **kv,
        **query,
    )
    print({
        "base": args.base,
        "other": args.other,
        "alpha": args.alpha,
        "output": args.output,
    })


if __name__ == "__main__":
    main()
