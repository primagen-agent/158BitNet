#!/usr/bin/env python3
"""Evaluate the native C episode pointer on cached held-out features."""
from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_data import CTokenDecoder  # noqa: E402
from train_episode_pointer import (  # noqa: E402
    EpisodeSpanPointer,
    best_spans,
)


def load_library(path):
    library = ctypes.CDLL(str(path))
    library.metis_episode_pointer_load.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.metis_episode_pointer_load.restype = ctypes.c_void_p
    library.metis_episode_pointer_select.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.metis_episode_pointer_select.restype = ctypes.c_int
    library.metis_episode_pointer_free.argtypes = [ctypes.c_void_p]
    return library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shared_library")
    parser.add_argument("pointer")
    parser.add_argument("checkpoint")
    parser.add_argument("feature_cache")
    parser.add_argument("backbone")
    parser.add_argument("tok_probe")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "EPISODE_POINTER_V2":
        raise ValueError("C evaluation requires EPISODE_POINTER_V2")
    cache = torch.load(
        args.feature_cache, map_location="cpu", weights_only=True)
    if cache.get("backbone_sha256") != checkpoint["backbone_sha256"]:
        raise ValueError("feature cache and pointer use different backbones")
    rows = cache["rows"]
    model = EpisodeSpanPointer(
        checkpoint["hidden"], checkpoint["rank"],
        checkpoint["max_span"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    decoder = CTokenDecoder(args.tok_probe, args.backbone)
    library = load_library(args.shared_library)
    error = ctypes.create_string_buffer(256)
    pointer = library.metis_episode_pointer_load(
        str(args.pointer).encode(),
        str(args.backbone).encode(),
        int(checkpoint["hidden"]),
        error,
        len(error),
    )
    if not pointer:
        raise RuntimeError(error.value.decode(errors="replace"))

    c_span_correct = 0
    c_text_correct = 0
    python_c_same = 0
    failures = []
    try:
        for row in rows:
            source = row["source_hidden"].float().contiguous()
            query = row["query_hidden"].float().contiguous()
            source_mask = torch.ones(
                1, source.shape[0], dtype=torch.bool)
            query_mask = torch.ones(
                1, query.shape[0], dtype=torch.bool)
            with torch.inference_mode():
                start_logits, end_logits, _ = model(
                    source.unsqueeze(0), source_mask,
                    query.unsqueeze(0), query_mask)
                python_start, python_end = best_spans(
                    start_logits, end_logits, source_mask,
                    checkpoint["max_span"])
            c_start = ctypes.c_size_t()
            c_end = ctypes.c_size_t()
            c_score = ctypes.c_float()
            status = library.metis_episode_pointer_select(
                pointer,
                source.numpy().ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)),
                source.shape[0],
                query.numpy().ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)),
                query.shape[0],
                ctypes.byref(c_start),
                ctypes.byref(c_end),
                ctypes.byref(c_score),
            )
            if status != 1:
                raise RuntimeError("C episode pointer selection failed")
            selected = (c_start.value, c_end.value)
            expected = (int(row["start"]), int(row["end"]))
            python_selected = (
                int(python_start[0]), int(python_end[0]))
            prediction = decoder.decode(
                row["source_ids"][
                    c_start.value:c_end.value + 1]).strip()
            c_span_correct += int(selected == expected)
            c_text_correct += int(prediction == row["target"])
            python_c_same += int(selected == python_selected)
            if prediction != row["target"] and len(failures) < 8:
                failures.append({
                    "sample_id": row["sample_id"],
                    "target": row["target"],
                    "prediction": prediction,
                    "gold_span": list(expected),
                    "c_span": list(selected),
                    "python_span": list(python_selected),
                })
    finally:
        library.metis_episode_pointer_free(pointer)

    total = len(rows)
    print(json.dumps({
        "phase": "episode_pointer_c_eval",
        "examples": total,
        "c_span_exact": c_span_correct / max(total, 1),
        "c_text_exact": c_text_correct / max(total, 1),
        "python_c_span_agreement": python_c_same / max(total, 1),
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "failures": failures,
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
