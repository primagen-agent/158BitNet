#!/usr/bin/env python3
"""Python BNMEM1 writer/loader round-trip."""

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from bnmem_export import load_bnmem_v1, save_bnmem_v3  # noqa: E402
from model_identity import require_matching_sha256  # noqa: E402


def main():
    generator = torch.Generator().manual_seed(7)
    nl, d_model, kv_dim, q_dim, head_dim, rank = 2, 12, 4, 8, 2, 3

    def random(*shape):
        return torch.randn(*shape, generator=generator)

    tensors = {
        "gdu_ab": random(nl),
        "gdu_bb": random(nl),
        "wk": random(nl, kv_dim, d_model),
        "wv": random(nl, kv_dim, d_model),
        "w_agg": random(nl, d_model),
        "gdu_aw": random(nl, d_model),
        "gdu_bw": random(nl, d_model),
        "mem_norm": random(nl, q_dim),
        "query_norm": random(nl, head_dim),
        "query_a": random(nl, q_dim, rank),
        "query_b": random(nl, rank, d_model),
    }
    kv_rank = 2
    low_rank_tensors = {
        key: value
        for key, value in tensors.items()
        if key not in {"wk", "wv"}
    }
    low_rank_tensors.update({
        "wk_a": random(nl, kv_dim, kv_rank),
        "wk_b": random(nl, kv_rank, d_model),
        "wv_a": random(nl, kv_dim, kv_rank),
        "wv_b": random(nl, kv_rank, d_model),
    })
    full_rank_tensors = {
        key: value
        for key, value in tensors.items()
        if key not in {"query_a", "query_b"}
    }
    full_rank_tensors["query_proj"] = random(nl, q_dim, d_model)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = str(Path(temp_dir) / "roundtrip.bnmem")
        backbone_sha256 = bytes(range(32))
        save_bnmem_v3(
            path,
            layer_ids=[1, 3],
            d_model=d_model,
            kv_dim=kv_dim,
            q_dim=q_dim,
            head_dim=head_dim,
            gamma=0.9,
            tau=1.0,
            rho=0.9,
            k_min=1,
            beta_scale=0.9,
            denom_mode=1,
            query_add_backbone=True,
            backbone_sha256=backbone_sha256,
            **tensors,
        )
        loaded = load_bnmem_v1(path)
        independent_path = str(Path(temp_dir) / "independent.bnmem")
        save_bnmem_v3(
            independent_path,
            layer_ids=[1, 3],
            d_model=d_model,
            kv_dim=kv_dim,
            q_dim=q_dim,
            head_dim=head_dim,
            gamma=0.9,
            tau=1.0,
            rho=0.9,
            k_min=1,
            beta_scale=0.9,
            denom_mode=1,
            query_add_backbone=False,
            **tensors,
        )
        independent = load_bnmem_v1(independent_path)
        low_rank_path = str(Path(temp_dir) / "low-rank-kv.bnmem")
        save_bnmem_v3(
            low_rank_path,
            layer_ids=[1, 3],
            d_model=d_model,
            kv_dim=kv_dim,
            q_dim=q_dim,
            head_dim=head_dim,
            gamma=0.9,
            tau=1.0,
            rho=0.9,
            k_min=1,
            beta_scale=0.9,
            denom_mode=1,
            query_add_backbone=False,
            **low_rank_tensors,
        )
        low_rank = load_bnmem_v1(low_rank_path)
        full_rank_path = str(Path(temp_dir) / "full-rank.bnmem")
        save_bnmem_v3(
            full_rank_path,
            layer_ids=[1, 3],
            d_model=d_model,
            kv_dim=kv_dim,
            q_dim=q_dim,
            head_dim=head_dim,
            gamma=0.9,
            tau=1.0,
            rho=0.9,
            k_min=1,
            beta_scale=0.9,
            denom_mode=1,
            query_add_backbone=False,
            **full_rank_tensors,
        )
        full_rank = load_bnmem_v1(full_rank_path)
        model_a = Path(temp_dir) / "model-a.gguf"
        model_b = Path(temp_dir) / "model-b.gguf"
        model_a.write_bytes(b"a")
        model_b.write_bytes(b"b")
        try:
            require_matching_sha256(
                "test artifact", backbone_sha256, model_b)
        except ValueError as error:
            assert "mismatch" in str(error)
        else:
            raise AssertionError("different backbone was accepted")

    assert loaded["layer_ids"] == [1, 3]
    assert loaded["query_rank"] == rank
    assert loaded["kv_rank"] == 0
    assert loaded["query_add_backbone"]
    assert loaded["denom_mode"] == 1
    assert loaded["backbone_sha256"] == backbone_sha256
    assert independent["backbone_sha256"] is None
    for name, expected in tensors.items():
        torch.testing.assert_close(loaded["tensors"][name], expected)
        torch.testing.assert_close(independent["tensors"][name], expected)
    assert independent["query_add_backbone"] is False
    assert low_rank["kv_rank"] == kv_rank
    for name, expected in low_rank_tensors.items():
        torch.testing.assert_close(low_rank["tensors"][name], expected)
    assert full_rank["query_rank"] == 0
    assert full_rank["kv_rank"] == 0
    assert full_rank["query_add_backbone"] is False
    for name, expected in full_rank_tensors.items():
        torch.testing.assert_close(full_rank["tensors"][name], expected)
    print("python BNMEM1 roundtrip: PASS")


if __name__ == "__main__":
    main()
