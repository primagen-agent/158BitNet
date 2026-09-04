#!/usr/bin/env python3
"""Bind a legacy BNMEM1 checkpoint to the exact GGUF that trained it."""
from __future__ import annotations

import argparse

from bnmem_export import load_bnmem_v1, save_bnmem_v3
from model_identity import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("memory_model")
    parser.add_argument("gguf")
    parser.add_argument("output")
    args = parser.parse_args()

    checkpoint = load_bnmem_v1(args.memory_model)
    actual_sha256 = sha256_file(args.gguf)
    existing_sha256 = checkpoint["backbone_sha256"]
    if existing_sha256 is not None and existing_sha256 != actual_sha256:
        raise ValueError(
            "memory model is already bound to a different GGUF")

    tensors = checkpoint["tensors"]
    kv = (
        {
            "wk_a": tensors["wk_a"], "wk_b": tensors["wk_b"],
            "wv_a": tensors["wv_a"], "wv_b": tensors["wv_b"],
        }
        if checkpoint["kv_rank"] > 0 else
        {"wk": tensors["wk"], "wv": tensors["wv"]}
    )
    query = (
        {
            "query_a": tensors["query_a"],
            "query_b": tensors["query_b"],
            "query_norm": tensors["query_norm"],
        }
        if checkpoint["query_rank"] > 0 else
        {
            "query_proj": tensors["query_proj"],
            "query_norm": tensors["query_norm"],
        }
    )
    save_bnmem_v3(
        args.output,
        layer_ids=checkpoint["layer_ids"],
        d_model=checkpoint["d_model"],
        kv_dim=checkpoint["kv_dim"],
        q_dim=checkpoint["q_dim"],
        head_dim=checkpoint["head_dim"],
        gamma=checkpoint["gamma"],
        tau=checkpoint["tau"],
        rho=checkpoint["rho"],
        k_min=checkpoint["k_min"],
        gdu_ab=tensors["gdu_ab"],
        gdu_bb=tensors["gdu_bb"],
        beta_scale=checkpoint["beta_scale"],
        w_agg=tensors["w_agg"],
        gdu_aw=tensors["gdu_aw"],
        gdu_bw=tensors["gdu_bw"],
        mem_norm=tensors["mem_norm"],
        denom_mode=checkpoint["denom_mode"],
        query_add_backbone=checkpoint["query_add_backbone"],
        backbone_sha256=actual_sha256,
        **kv,
        **query,
    )
    print({
        "input": args.memory_model,
        "gguf": args.gguf,
        "output": args.output,
        "backbone_sha256": actual_sha256.hex(),
    })


if __name__ == "__main__":
    main()
