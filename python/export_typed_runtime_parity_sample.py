#!/usr/bin/env python3
"""Export cached training features for an end-to-end C runtime check."""

import argparse
import struct
from pathlib import Path

import torch

from export_typed_pair_parity_sample import (
    load_typed_pair_model,
)
from typed_memory_training import file_fingerprint
from train_typed_memory_writer import load_token_embedding_table


def export_typed_runtime_parity_sample(
    checkpoint_path, cache_path,
    gguf_path, lib_path, output_path,
    row_index=0, left_episode=0, right_episode=1,
):
    checkpoint, model = load_typed_pair_model(
        checkpoint_path)
    cache = torch.load(
        cache_path, map_location="cpu",
        weights_only=True)
    expected_sha = checkpoint["backbone_sha256"]
    if (
        cache.get("backbone_sha256") != expected_sha
        or file_fingerprint(gguf_path) != expected_sha
        or cache.get("hidden_layer_bands")
            != checkpoint["hidden_layer_bands"]
    ):
        raise ValueError(
            "runtime parity backbone or layer bands mismatch")
    rows = cache.get("rows") or []
    if not 0 <= row_index < len(rows):
        raise ValueError("runtime parity row is out of range")
    row = rows[row_index]
    episode_ids = row["episode_ids"]
    episode_hidden = row["episode_hidden"]
    if (
        not 0 <= left_episode < len(episode_ids)
        or not 0 <= right_episode < len(episode_ids)
        or left_episode == right_episode
    ):
        raise ValueError("runtime parity episode is out of range")
    left_ids = [int(value) for value in episode_ids[left_episode]]
    right_ids = [int(value) for value in episode_ids[right_episode]]
    left_hidden = episode_hidden[left_episode].float()
    right_hidden = episode_hidden[right_episode].float()
    hidden = int(checkpoint["hidden"])
    bands = checkpoint["hidden_layer_bands"]
    if (
        tuple(left_hidden.shape)
            != (len(left_ids), len(bands), hidden)
        or tuple(right_hidden.shape)
            != (len(right_ids), len(bands), hidden)
    ):
        raise ValueError(
            "runtime parity cached feature geometry mismatch")
    embedding_table = load_token_embedding_table(
        gguf_path, lib_path, expected_sha, hidden)
    left_identity = embedding_table[left_ids].float()
    right_identity = embedding_table[right_ids].float()
    with torch.inference_mode():
        result = model(
            left_hidden.unsqueeze(0),
            right_hidden.unsqueeze(0),
            torch.ones(
                1, len(left_ids), dtype=torch.bool),
            torch.ones(
                1, len(right_ids), dtype=torch.bool),
            left_identity.unsqueeze(0),
            right_identity.unsqueeze(0),
            return_internal=True)
    entity_feature = result["internals"][
        "pair_features"]["entity"][0]
    predicate_feature = result["internals"][
        "pair_features"]["predicate"][0]
    feature_dim = int(entity_feature.numel())
    payload = bytearray(b"BNTRPAR1")
    payload += struct.pack(
        "<IIIIIII",
        1,
        hidden,
        len(bands),
        sum(len(band) for band in bands),
        feature_dim,
        len(left_ids),
        len(right_ids),
    )
    for band in bands:
        payload += struct.pack(
            "<II", int(band[0]), int(band[-1]))
    payload += bytes.fromhex(expected_sha)
    for token_ids in (left_ids, right_ids):
        payload += struct.pack(
            f"<{len(token_ids)}i", *token_ids)
    for tensor in (
        left_hidden, right_hidden,
        left_identity, right_identity,
        entity_feature, predicate_feature,
        result["entity_logit"].reshape(1),
        result["predicate_logit"].reshape(1),
    ):
        payload += tensor.detach().contiguous().numpy().astype(
            "<f4", copy=False).tobytes()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("cache")
    parser.add_argument("gguf")
    parser.add_argument("lib")
    parser.add_argument("output")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--left-episode", type=int, default=0)
    parser.add_argument("--right-episode", type=int, default=1)
    args = parser.parse_args()
    print(export_typed_runtime_parity_sample(
        args.checkpoint, args.cache,
        args.gguf, args.lib, args.output,
        args.row, args.left_episode,
        args.right_episode))


if __name__ == "__main__":
    main()
