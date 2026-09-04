#!/usr/bin/env python3
"""Measure the best extractive token-F1 span in retrieved excerpts."""
from __future__ import annotations

import argparse
import collections
import json
import re


def tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def token_f1(prediction, gold):
    common = sum((collections.Counter(prediction)
                  & collections.Counter(gold)).values())
    if not prediction or not gold or not common:
        return 0.0
    return 2.0 * common / (len(prediction) + len(gold))


def best_span_f1(source, gold, maximum):
    best = 0.0
    for start in range(len(source)):
        for length in range(1, min(maximum, len(source) - start) + 1):
            best = max(
                best, token_f1(source[start:start + length], gold))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--max-span-tokens", type=int, default=24)
    args = parser.parse_args()
    rows = json.load(open(args.results, encoding="utf-8"))
    by_category = collections.defaultdict(list)
    scores = []
    for row in rows:
        source = tokens(" ".join(row.get("retrieved_excerpts", [])))
        gold = tokens(row["answer"])
        score = best_span_f1(source, gold, args.max_span_tokens)
        scores.append(score)
        by_category[row.get("category")].append(score)
    print(json.dumps({
        "samples": len(scores),
        "max_span_tokens": args.max_span_tokens,
        "oracle_span_f1": sum(scores) / max(len(scores), 1),
        "oracle_exact": sum(score == 1.0 for score in scores)
                        / max(len(scores), 1),
        "categories": {
            str(category): {
                "samples": len(values),
                "oracle_span_f1": sum(values) / len(values),
                "oracle_exact": sum(value == 1.0 for value in values)
                                / len(values),
            }
            for category, values in sorted(by_category.items())
        },
    }, indent=2))


if __name__ == "__main__":
    main()
