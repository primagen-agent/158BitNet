#!/usr/bin/env python3
"""Read-head lexical statistics must match the C runtime's real-token sums.

C sums only real tokens; padded positions must contribute nothing. This test
also checks that widening the padded bank does not change existing pairs.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import neural_read_head as head_lib  # noqa: E402

TOKENS = (5, 9, 17, 3)
QUERIES = (4, 7, 11)
WIDTH = 16


def cache(texts, lengths, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = {}
    for text, count in zip(texts, lengths):
        lexical = torch.randn(count, WIDTH, generator=generator)
        lexical = torch.nn.functional.normalize(lexical, dim=-1)
        rows[text] = dict(x=torch.cat((torch.zeros(count, WIDTH), lexical), -1))
    return rows


def naive(query, fact):
    similarity = query @ fact.T
    s1 = similarity.max(-1).values.mean()
    s2 = similarity.max(-2).values.mean()
    return float(s1), float(s2), float(similarity.max())


def stats(features, queries, facts):
    qv, qm = head_lib.padded_lexical(features, queries)
    fv, fm = head_lib.padded_lexical(features, facts)
    return head_lib.pair_stats(qv, qm, fv, fm)


def main():
    queries = [f"query-{i}" for i in range(len(QUERIES))]
    facts = [f"fact-{i}" for i in range(len(TOKENS))]
    features = cache(queries + facts, QUERIES + TOKENS, seed=11)
    s1, s2, s3 = stats(features, queries, facts)
    for a, query in enumerate(queries):
        for b, fact in enumerate(facts):
            expected = naive(features[query]["x"][:, WIDTH:], features[fact]["x"][:, WIDTH:])
            actual = (float(s1[a, b]), float(s2[a, b]), float(s3[a, b]))
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected),
                                       atol=2e-6, rtol=2e-6)

    wide = cache(queries + facts + ["fact-wide"], QUERIES + TOKENS + (41,), seed=12)
    wide.update({t: features[t] for t in queries + facts})
    w1, w2, w3 = stats(wide, queries, facts)
    torch.testing.assert_close(w1, s1)
    torch.testing.assert_close(w2, s2)
    torch.testing.assert_close(w3, s3)
    print(f"PASS: {len(queries) * len(facts)} read-head lexical pairs and padding invariance")


if __name__ == "__main__":
    main()
