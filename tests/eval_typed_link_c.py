#!/usr/bin/env python3
"""Evaluate V251 pair/link decisions through the native C implementation."""

import argparse
import collections
import concurrent.futures
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from c_tokenizer import CTokenizer
from train_typed_memory_writer import (
    load_token_embedding_table,
    load_writer_examples,
)
from train_typed_pair_verifier import build_active_pairs


FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


def configure_library(path):
    library = ctypes.CDLL(path)
    library.metis_typed_pair_encoder_load.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.metis_typed_pair_encoder_load.restype = ctypes.c_void_p
    library.metis_typed_pair_encoder_free.argtypes = [ctypes.c_void_p]
    library.metis_typed_pair_encoder_score.argtypes = [
        ctypes.c_void_p,
        FLOAT_PTR, ctypes.c_size_t, FLOAT_PTR,
        FLOAT_PTR, ctypes.c_size_t, FLOAT_PTR,
        FLOAT_PTR, FLOAT_PTR,
        FLOAT_PTR, FLOAT_PTR,
    ]
    library.metis_typed_pair_encoder_score.restype = ctypes.c_int
    library.metis_typed_link_model_load.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.metis_typed_link_model_load.restype = ctypes.c_void_p
    library.metis_typed_link_model_free.argtypes = [ctypes.c_void_p]
    library.metis_typed_link_score_pairs.argtypes = [
        ctypes.c_void_p,
        FLOAT_PTR, FLOAT_PTR,
        FLOAT_PTR, FLOAT_PTR,
        ctypes.c_size_t, FLOAT_PTR,
    ]
    library.metis_typed_link_score_pairs.restype = ctypes.c_int
    library.metis_typed_link_predecessor_exists.argtypes = [
        ctypes.c_void_p, FLOAT_PTR, ctypes.c_size_t, FLOAT_PTR,
    ]
    library.metis_typed_link_predecessor_exists.restype = ctypes.c_int
    library.metis_typed_link_select_predecessor.argtypes = [
        ctypes.c_void_p, FLOAT_PTR, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), FLOAT_PTR,
    ]
    library.metis_typed_link_select_predecessor.restype = ctypes.c_int
    return library


def float_pointer(array):
    return array.ctypes.data_as(FLOAT_PTR)


def balanced_accuracy(logits, targets, threshold=0.0):
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=np.bool_)
    positive = float(np.mean(logits[targets] > threshold))
    negative = float(np.mean(logits[~targets] <= threshold))
    return {
        "balanced": 0.5 * (positive + negative),
        "positive": positive,
        "negative": negative,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("features")
    parser.add_argument("jsonl")
    parser.add_argument("gguf")
    parser.add_argument("ggw_lib")
    parser.add_argument("tok_probe")
    parser.add_argument("pair_model")
    parser.add_argument("link_model")
    parser.add_argument("c_library")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error("limit must be non-negative and workers positive")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu",
        weights_only=True)
    if (
        checkpoint.get("format") != "TYPED_PAIR_VERIFIER_V1"
        or not checkpoint.get("dual_path", False)
        or not checkpoint.get("set_link_head", False)
    ):
        raise ValueError("C evaluation requires the V251 checkpoint")
    embeddings = load_token_embedding_table(
        args.gguf, args.ggw_lib,
        checkpoint["backbone_sha256"],
        int(checkpoint["hidden"]))
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    examples = load_writer_examples(
        args.features, args.jsonl,
        checkpoint["backbone_sha256"],
        tokenizer, embeddings)
    if args.limit:
        examples = examples[:args.limit]
    pairs = build_active_pairs(examples)
    hidden = [
        np.ascontiguousarray(
            example["hidden"].float().numpy(),
            dtype=np.float32)
        for example in examples
    ]
    identity = [
        np.ascontiguousarray(
            example["identity_hidden"].float().numpy(),
            dtype=np.float32)
        for example in examples
    ]

    library = configure_library(args.c_library)
    error = ctypes.create_string_buffer(256)
    encoder = library.metis_typed_pair_encoder_load(
        args.pair_model.encode(), args.gguf.encode(),
        int(checkpoint["hidden"]), error, len(error))
    if not encoder:
        raise RuntimeError(
            f"cannot load C pair model: {error.value.decode()}")
    link = library.metis_typed_link_model_load(
        args.link_model.encode(), args.gguf.encode(),
        error, len(error))
    if not link:
        library.metis_typed_pair_encoder_free(encoder)
        raise RuntimeError(
            f"cannot load C link model: {error.value.decode()}")

    feature_dim = int(checkpoint["rank"]) * 2 + 5
    thread_state = threading.local()

    def score_pair(pair):
        if not hasattr(thread_state, "entity"):
            thread_state.entity = np.empty(
                feature_dim, dtype=np.float32)
            thread_state.predicate = np.empty(
                feature_dim, dtype=np.float32)
            thread_state.entity_value = np.empty(
                1, dtype=np.float32)
            thread_state.predicate_value = np.empty(
                1, dtype=np.float32)
            thread_state.joint = np.empty(
                1, dtype=np.float32)
        left = hidden[pair["left"]]
        right = hidden[pair["right"]]
        left_identity = identity[pair["left"]]
        right_identity = identity[pair["right"]]
        if library.metis_typed_pair_encoder_score(
            encoder,
            float_pointer(left), left.shape[0],
            float_pointer(left_identity),
            float_pointer(right), right.shape[0],
            float_pointer(right_identity),
            float_pointer(thread_state.entity),
            float_pointer(thread_state.predicate),
            float_pointer(thread_state.entity_value),
            float_pointer(thread_state.predicate_value),
        ) != 0:
            raise RuntimeError("C pair encoding failed")
        if library.metis_typed_link_score_pairs(
            link,
            float_pointer(thread_state.entity),
            float_pointer(thread_state.predicate),
            float_pointer(thread_state.entity_value),
            float_pointer(thread_state.predicate_value),
            1, float_pointer(thread_state.joint),
        ) != 0:
            raise RuntimeError("C joint scoring failed")
        return (
            float(thread_state.entity_value[0]),
            float(thread_state.predicate_value[0]),
            float(thread_state.joint[0]),
        )

    started = time.monotonic()
    scored = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        for index, values in enumerate(
            executor.map(score_pair, pairs, chunksize=16), 1
        ):
            scored.append(values)
            if (
                args.progress_every > 0
                and index % args.progress_every == 0
            ):
                elapsed = time.monotonic() - started
                print(json.dumps({
                    "phase": "c_pair_progress",
                    "pairs": index,
                    "total": len(pairs),
                    "pairs_per_second": index / max(elapsed, 1e-9),
                }, separators=(",", ":")), flush=True)
    scored = np.asarray(scored, dtype=np.float32)
    entity_logits = scored[:, 0]
    predicate_logits = scored[:, 1]
    joint_logits = scored[:, 2]

    by_left = collections.defaultdict(list)
    for pair_index, pair in enumerate(pairs):
        by_left[pair["left"]].append(pair_index)
    create_total = 0
    update_total = 0
    update_correct = 0
    update_rank_correct = 0
    update_accepted = 0
    update_no_candidate = 0
    update_wrong_candidate = 0
    missing_total = 0
    missing_rejected = 0
    for index, example in enumerate(examples):
        if example["operation"] == 0:
            create_total += 1
            continue
        update_total += 1
        pair_indices = by_left[index]
        targets = [
            position
            for position, pair_index in enumerate(pair_indices)
            if examples[pairs[pair_index]["right"]]["episode"]
                == example["previous_episode"]
        ]
        if len(targets) != 1:
            raise ValueError("update has no unique predecessor")
        target = targets[0]
        scores = np.ascontiguousarray(
            joint_logits[pair_indices], dtype=np.float32)
        selected = ctypes.c_size_t()
        exists = ctypes.c_float()
        if library.metis_typed_link_select_predecessor(
            link, float_pointer(scores), len(scores),
            ctypes.byref(selected), ctypes.byref(exists),
        ) != 0:
            raise RuntimeError("C predecessor selection failed")
        false_scores = np.delete(scores, target)
        true_score = float(scores[target])
        false_max = (
            float(false_scores.max())
            if false_scores.size else -float("inf"))
        update_rank_correct += int(true_score > false_max)
        accepted = exists.value > 0.0
        update_accepted += int(accepted)
        if not accepted:
            update_no_candidate += 1
        elif selected.value == target:
            update_correct += 1
        else:
            update_wrong_candidate += 1
        missing_total += 1
        if false_scores.size == 0:
            missing_rejected += 1
        else:
            missing_exists = ctypes.c_float()
            false_scores = np.ascontiguousarray(
                false_scores, dtype=np.float32)
            if library.metis_typed_link_predecessor_exists(
                link, float_pointer(false_scores),
                len(false_scores),
                ctypes.byref(missing_exists),
            ) != 0:
                raise RuntimeError(
                    "C missing-predecessor scoring failed")
            missing_rejected += int(
                missing_exists.value <= 0.0)

    entity_metrics = balanced_accuracy(
        entity_logits,
        [pair["entity_same"] for pair in pairs],
        float(checkpoint.get("entity_threshold", 0.0)))
    predicate_metrics = balanced_accuracy(
        predicate_logits,
        [pair["predicate_same"] for pair in pairs],
        float(checkpoint.get("predicate_threshold", 0.0)))
    joint_metrics = balanced_accuracy(
        joint_logits,
        [pair["joint_same"] for pair in pairs])
    update_link = update_correct / max(update_total, 1)
    acceptance = update_accepted / max(update_total, 1)
    missing_rejection = (
        missing_rejected / max(missing_total, 1))
    result = {
        "phase": "typed_link_c_eval",
        "examples": len(examples),
        "pairs": len(pairs),
        "workers": args.workers,
        "elapsed_sec": time.monotonic() - started,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "operation_source": "ground_truth_event_type",
        "entity_threshold":
            float(checkpoint.get("entity_threshold", 0.0)),
        "predicate_threshold":
            float(checkpoint.get("predicate_threshold", 0.0)),
        "entity_pair_balanced": entity_metrics["balanced"],
        "predicate_pair_balanced":
            predicate_metrics["balanced"],
        "joint_pair_balanced": joint_metrics["balanced"],
        "predecessor_rank_accuracy":
            update_rank_correct / max(update_total, 1),
        "predecessor_accept_accuracy": acceptance,
        "counterfactual_missing_rejection_accuracy":
            missing_rejection,
        "predecessor_exists_balanced_accuracy":
            0.5 * (acceptance + missing_rejection),
        "update_link_accuracy": update_link,
        "update_no_candidate_rate":
            update_no_candidate / max(update_total, 1),
        "update_wrong_candidate_rate":
            update_wrong_candidate / max(update_total, 1),
        "conditional_version_link_accuracy":
            (create_total + update_correct)
            / max(len(examples), 1),
    }
    print(json.dumps(result, separators=(",", ":")), flush=True)
    library.metis_typed_link_model_free(link)
    library.metis_typed_pair_encoder_free(encoder)


if __name__ == "__main__":
    main()
