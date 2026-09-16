#!/usr/bin/env python3
"""Export a real cached candidate set for full C link-decision parity."""

import argparse
import struct
from pathlib import Path

import torch

from export_typed_pair_parity_sample import (
    load_typed_pair_model,
)
from typed_memory_training import file_fingerprint
from train_typed_memory_writer import load_token_embedding_table


def export_typed_runtime_link_parity_sample(
    checkpoint_path, cache_path,
    gguf_path, lib_path, output_path,
    row_index=0, current_episode=16,
    candidate_count=16,
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
        or not checkpoint.get("set_link_head", False)
    ):
        raise ValueError(
            "runtime link parity model geometry mismatch")
    rows = cache.get("rows") or []
    if not 0 <= row_index < len(rows):
        raise ValueError("runtime link parity row is out of range")
    row = rows[row_index]
    episode_ids = row["episode_ids"]
    episode_hidden = row["episode_hidden"]
    if (
        not 1 <= current_episode < len(episode_ids)
        or candidate_count < 2
    ):
        raise ValueError(
            "runtime link parity episode geometry is invalid")
    first_candidate = max(
        0, current_episode - candidate_count)
    candidate_indices = list(
        range(first_candidate, current_episode))
    hidden = int(checkpoint["hidden"])
    bands = checkpoint["hidden_layer_bands"]
    embedding_table = load_token_embedding_table(
        gguf_path, lib_path, expected_sha, hidden)

    def episode(index):
        token_ids = [
            int(value) for value in episode_ids[index]]
        features = episode_hidden[index].float()
        identity = embedding_table[token_ids].float()
        expected_shape = (
            len(token_ids), len(bands), hidden)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "runtime link cached feature geometry mismatch")
        return token_ids, features, identity

    current_ids, current_hidden, current_identity = episode(
        current_episode)
    candidate_rows = [
        episode(index) for index in candidate_indices]
    entity_features = []
    predicate_features = []
    entity_logits = []
    predicate_logits = []
    joint_scores = []
    with torch.inference_mode():
        for token_ids, features, identity in candidate_rows:
            result = model(
                current_hidden.unsqueeze(0),
                features.unsqueeze(0),
                torch.ones(
                    1, len(current_ids), dtype=torch.bool),
                torch.ones(
                    1, len(token_ids), dtype=torch.bool),
                current_identity.unsqueeze(0),
                identity.unsqueeze(0),
                return_internal=True)
            entity_features.append(
                result["internals"][
                    "pair_features"]["entity"][0])
            predicate_features.append(
                result["internals"][
                    "pair_features"]["predicate"][0])
            entity_logits.append(
                result["entity_logit"].reshape(()))
            predicate_logits.append(
                result["predicate_logit"].reshape(()))
            joint_scores.append(
                result["joint_logit"].reshape(()))
        entity_features = torch.stack(entity_features)
        predicate_features = torch.stack(predicate_features)
        entity_logits = torch.stack(entity_logits)
        predicate_logits = torch.stack(predicate_logits)
        joint_scores = torch.stack(joint_scores)
        exists = model.predecessor_exists_logits(
            [joint_scores]).reshape(())
    selected = (
        int(joint_scores.argmax())
        if float(exists) > 0.0
        else 0xFFFFFFFF)
    feature_dim = int(entity_features.shape[1])
    payload = bytearray(b"BNTRLNK1")
    payload += struct.pack(
        "<IIIIIII",
        1,
        hidden,
        len(bands),
        sum(len(band) for band in bands),
        feature_dim,
        len(candidate_rows),
        len(current_ids),
    )
    for band in bands:
        payload += struct.pack(
            "<II", int(band[0]), int(band[-1]))
    payload += bytes.fromhex(expected_sha)

    def append_episode(token_ids, features, identity):
        nonlocal payload
        payload += struct.pack("<I", len(token_ids))
        payload += struct.pack(
            f"<{len(token_ids)}i", *token_ids)
        for tensor in (features, identity):
            payload += tensor.detach().contiguous().numpy().astype(
                "<f4", copy=False).tobytes()

    append_episode(
        current_ids, current_hidden, current_identity)
    for candidate in candidate_rows:
        append_episode(*candidate)
    for tensor in (
        entity_features, predicate_features,
        entity_logits, predicate_logits,
        joint_scores, exists.reshape(1),
    ):
        payload += tensor.detach().contiguous().numpy().astype(
            "<f4", copy=False).tobytes()
    payload += struct.pack("<I", selected)
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
    parser.add_argument("--current-episode", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()
    print(export_typed_runtime_link_parity_sample(
        args.checkpoint, args.cache,
        args.gguf, args.lib, args.output,
        args.row, args.current_episode,
        args.candidates))


if __name__ == "__main__":
    main()
