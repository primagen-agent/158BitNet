#!/usr/bin/env python3
"""Evaluate lexical/learned hybrid retrieval on cached LoCoMo embeddings."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch

from prepare_locomo_retrieval_data import tokenize
from train_locomo_embedding_retriever import DualProjection, load_cache


def bm25_scores(entries, question):
    document_frequency = collections.Counter()
    for entry in entries:
        document_frequency.update(set(entry["tokens"]))
    n_docs = max(len(entries), 1)
    average_length = (
        sum(len(entry["tokens"]) for entry in entries) / n_docs)
    query_counts = collections.Counter(tokenize(question))
    scores = np.zeros(len(entries), dtype=np.float32)
    for index, entry in enumerate(entries):
        counts = collections.Counter(entry["tokens"])
        length = len(entry["tokens"])
        for token, query_count in query_counts.items():
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * length / max(average_length, 1.0))
            scores[index] += (
                inverse_document_frequency
                * frequency * 2.2 / denominator
                * (1.0 + 0.15 * min(query_count - 1, 2)))
    return scores


def ranks(scores):
    order = np.argsort(-scores, kind="stable")
    result = np.empty_like(order)
    result[order] = np.arange(len(order))
    return result


def metrics(selections, positives):
    hits = total = complete = 0
    for selected, gold in zip(selections, positives):
        selected = set(selected)
        gold = set(gold)
        hits += len(selected & gold)
        total += len(gold)
        complete += int(gold <= selected)
    return {
        "evidence_recall": hits / max(total, 1),
        "all_evidence_recall": complete / max(len(positives), 1),
        "questions": len(positives),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("retriever")
    parser.add_argument("--top-k", type=int, nargs="+", default=[12, 20, 32])
    parser.add_argument("--rrf-k", type=float, default=60.0)
    args = parser.parse_args()

    conversations = load_cache(Path(args.cache))
    checkpoint = torch.load(
        args.retriever, map_location="cpu", weights_only=False)
    model = DualProjection(checkpoint["hidden"], checkpoint["rank"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    valid_ids = checkpoint["valid_conversation_ids"]
    temperature = checkpoint["temperature"]

    all_rankings = {
        "learned": [],
        "bm25": [],
        "rrf_equal": [],
        "rrf_lexical_2x": [],
    }
    positives = []
    with torch.no_grad():
        for conversation_index in valid_ids:
            data = conversations[conversation_index]
            learned = model.scores(
                torch.from_numpy(data["query_vectors"]),
                torch.from_numpy(data["entry_vectors"]),
                temperature).numpy()
            for row, ((_question_index, qa), gold) in enumerate(zip(
                data["questions"], data["positive_indices"]
            )):
                lexical = bm25_scores(
                    data["entries"], str(qa["question"]))
                learned_rank = ranks(learned[row])
                lexical_rank = ranks(lexical)
                rrf_learned = 1.0 / (args.rrf_k + learned_rank + 1)
                rrf_lexical = 1.0 / (args.rrf_k + lexical_rank + 1)
                all_rankings["learned"].append(
                    np.argsort(-learned[row], kind="stable"))
                all_rankings["bm25"].append(
                    np.argsort(-lexical, kind="stable"))
                all_rankings["rrf_equal"].append(np.argsort(
                    -(rrf_learned + rrf_lexical), kind="stable"))
                all_rankings["rrf_lexical_2x"].append(np.argsort(
                    -(rrf_learned + 2.0 * rrf_lexical), kind="stable"))
                positives.append(gold)

    report = {
        "valid_conversation_ids": valid_ids,
        "rrf_k": args.rrf_k,
        "results": {},
    }
    for name, rankings in all_rankings.items():
        report["results"][name] = {
            str(top_k): metrics(
                [ranking[:top_k] for ranking in rankings], positives)
            for top_k in args.top_k
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
