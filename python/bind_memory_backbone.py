#!/usr/bin/env python3
"""Bind a BNMEM checkpoint to a GGUF and optionally fold its query delta.

The fold converts the historical Python-only
``delta(h_raw) + backbone_q(h_raw)`` representation into the reference
``query_proj(h_raw)`` representation used identically by Python and C.
"""
from __future__ import annotations

import argparse
import torch

from bnmem_export import load_bnmem_v1, save_bnmem_v3
from ggw import GGUFWeights
from model_identity import sha256_file
from torch_backbone import _deinterleave_rope_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("memory_model")
    parser.add_argument("gguf")
    parser.add_argument("output")
    parser.add_argument("--lib")
    parser.add_argument(
        "--fold-backbone-query", action="store_true",
        help="fold a full-rank backbone-delta query into query_proj(h_raw)")
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
    query_add_backbone = checkpoint["query_add_backbone"]
    if args.fold_backbone_query:
        if not query_add_backbone:
            raise ValueError("memory model query is already independent")
        if checkpoint["query_rank"] != 0:
            raise ValueError(
                "only a full-rank backbone-delta query can be folded exactly")
        weights = GGUFWeights(args.gguf, args.lib)
        try:
            backbone_query = torch.stack([
                _deinterleave_rope_rows(
                    weights.get_layer(layer_id).q,
                    checkpoint["head_dim"]).float()
                for layer_id in checkpoint["layer_ids"]
            ])
        finally:
            weights.close()
        query["query_proj"] = query["query_proj"] + backbone_query
        query_add_backbone = False

    fusion = {}
    if checkpoint.get("fusion_mode") == "residual_gate":
        fusion = {
            "fusion_gate_w": tensors["fusion_gate_w"],
            "fusion_gate_b": tensors["fusion_gate_b"],
        }
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
        alpha_max_tokens=checkpoint.get("alpha_max_tokens", 0),
        alpha_max_fraction=checkpoint.get("alpha_max_fraction", 0.0),
        gdu_ab=tensors["gdu_ab"],
        gdu_bb=tensors["gdu_bb"],
        beta_scale=checkpoint["beta_scale"],
        w_agg=tensors["w_agg"],
        gdu_aw=tensors["gdu_aw"],
        gdu_bw=tensors["gdu_bw"],
        mem_norm=tensors["mem_norm"],
        denom_mode=checkpoint["denom_mode"],
        query_add_backbone=query_add_backbone,
        backbone_sha256=actual_sha256,
        **kv,
        **query,
        **fusion,
    )
    print({
        "input": args.memory_model,
        "gguf": args.gguf,
        "output": args.output,
        "backbone_sha256": actual_sha256.hex(),
        "query_add_backbone": query_add_backbone,
        "folded_backbone_query": args.fold_backbone_query,
    })


if __name__ == "__main__":
    main()
