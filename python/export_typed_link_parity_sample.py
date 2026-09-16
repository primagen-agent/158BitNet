#!/usr/bin/env python3
"""Export deterministic V251 head inputs and Python reference outputs."""

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

from train_typed_pair_verifier import (
    SET_LINK_HEAD_VERSION,
    TypedPairVerifier,
)


def export_typed_link_parity_sample(
    checkpoint_path, output_path,
    pair_count=17, seed=20260915,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu",
        weights_only=True)
    if (
        checkpoint.get("format")
        != "TYPED_PAIR_VERIFIER_V1"
        or not checkpoint.get("set_link_head", False)
        or int(checkpoint.get(
            "set_link_head_version", 0
        )) not in (2, 3)
    ):
        raise ValueError(
            "typed-link parity requires a v2 or v3 set-link checkpoint")
    if pair_count < 1:
        raise ValueError(
            "typed-link parity needs at least one pair")
    state = checkpoint["verifier_state_dict"]
    joint_weight = state[
        "joint_head.0.weight"].detach().float()
    feature_dim = joint_weight.shape[1] // 2
    generator = torch.Generator().manual_seed(seed)
    entity = torch.randn(
        pair_count, feature_dim,
        generator=generator)
    predicate = torch.randn(
        pair_count, feature_dim,
        generator=generator)
    entity_logits = torch.randn(
        pair_count, generator=generator)
    predicate_logits = torch.randn(
        pair_count, generator=generator)
    pair_features = torch.cat(
        (entity, predicate), dim=-1)
    hidden = F.gelu(F.linear(
        pair_features,
        joint_weight,
        state["joint_head.0.bias"].float()))
    residual = F.linear(
        hidden,
        state["joint_head.2.weight"].float(),
        state["joint_head.2.bias"].float()
    ).squeeze(-1)
    joint = torch.minimum(
        entity_logits, predicate_logits) + residual
    exists_features = (
        TypedPairVerifier.set_link_features(joint,
            7 if checkpoint["set_link_head_version"] == 3 else 5))
    exists_hidden = F.gelu(F.linear(
        exists_features,
        state[
            "predecessor_exists_head.0.weight"
        ].float(),
        state[
            "predecessor_exists_head.0.bias"
        ].float()))
    exists = F.linear(
        exists_hidden,
        state[
            "predecessor_exists_head.2.weight"
        ].float(),
        state[
            "predecessor_exists_head.2.bias"
        ].float()).reshape(())
    selected = (
        int(joint.argmax())
        if float(exists) > 0.0
        else 0xFFFFFFFF)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray(b"BNTLPAR1")
    payload += struct.pack(
        "<III", 1, feature_dim, pair_count)
    for tensor in (
        entity, predicate,
        entity_logits, predicate_logits,
        joint, exists.reshape(1),
    ):
        payload += tensor.detach().contiguous().numpy().astype(
            "<f4", copy=False).tobytes()
    payload += struct.pack("<I", selected)
    output.write_bytes(payload)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--pairs", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    print(export_typed_link_parity_sample(
        args.checkpoint, args.output,
        args.pairs, args.seed))


if __name__ == "__main__":
    main()
