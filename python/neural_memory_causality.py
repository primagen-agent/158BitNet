"""Same-question paired scoring. Never substitute for independent text review."""
from collections import Counter, defaultdict

from prepare_neural_memory_protocol import canonical, digest, validate_records
from review_neural_memory import keyed, score_bundle


# name, left scenario, right scenario, required semantic relationship
CONTROLS = (
    ("value_swap", "original", "swapped_values", "change"),
    ("current_update", "original", "update_current", "change"),
    ("target_removed", "original", "target_removed", "missing"),
    ("wrong_subject", "original", "wrong_subject", "missing"),
    ("wrong_relation", "original", "wrong_relation", "missing"),
    ("all_irrelevant", "original", "all_irrelevant", "missing"),
    ("empty_memory", "original", "empty", "missing"),
    ("hypothetical", "original", "hypothetical", "missing"),
    ("negated", "original", "negated", "missing"),
    ("unconfirmed_quote", "original", "quoted", "missing"),
    ("distractor_added", "original", "distractor_added", "invariant"),
    ("distractor_reordered", "distractor_added", "distractor_reordered", "invariant"),
    ("same_value_other_subject", "original", "same_value_other_person", "invariant"),
)


def paired_protocol(inputs, labels, index):
    validate_records(inputs, labels, index)
    runtime, gold, meta = map(keyed, (inputs, labels, index))
    families = defaultdict(dict)
    for cid, row in meta.items():
        key = (row["world_id"], row["split"], row["language"])
        if row["scenario"] in families[key]:
            raise ValueError("duplicate scenario in a world/language")
        families[key][row["scenario"]] = cid
    pairs = []
    for (world, split, language), scenarios in sorted(families.items()):
        for control, left, right, expectation in CONTROLS:
            if left not in scenarios or right not in scenarios:
                raise ValueError("missing required causal control")
            a, b = scenarios[left], scenarios[right]
            if runtime[a]["context"] != runtime[b]["context"]:
                raise ValueError("causal pair changed the question/context")
            if runtime[a]["episodes"] == runtime[b]["episodes"]:
                raise ValueError("causal pair did not change memory")
            x, y = gold[a], gold[b]
            claims_a = {canonical(c) for c in x["required_claims"]}
            claims_b = {canonical(c) for c in y["required_claims"]}
            if x["evidence_missing"] or not claims_a:
                raise ValueError("left control must have positive evidence")
            if expectation == "missing":
                if not y["evidence_missing"] or claims_b:
                    raise ValueError("missing control demands an unsupported answer")
            elif y["evidence_missing"] or not claims_b or (claims_a == claims_b) != (expectation == "invariant"):
                raise ValueError("counterfactual rubric does not match intervention")
            pairs.append({"id": f"{world}/{language}/{control}", "world_id": world, "split": split,
                          "language": language, "control": control, "expectation": expectation,
                          "left_id": a, "right_id": b, "context_sha256": digest(runtime[a]["context"]),
                          "left_input_sha256": digest(runtime[a]), "right_input_sha256": digest(runtime[b])})
    return {"format": "neural-memory-causal-pairs-v1", "pairs": pairs,
            "pair_count": len(pairs), "independent_worlds": len({m["world_id"] for m in index}),
            "promotion_eligible": False,
            "note": "Related pairs and bilingual renderings are not independent worlds; protocol smoke only"}


def score_causal_bundle(inputs, labels, index, predictions, reviews, independent_reviewers):
    protocol = paired_protocol(inputs, labels, index)
    single = score_bundle(inputs, labels, index, predictions, reviews, independent_reviewers)
    results = keyed(single["cases"])
    pairs = []
    for pair in protocol["pairs"]:
        statuses = [results[pair[key]]["status"] for key in ("left_id", "right_id")]
        status = "fail" if "fail" in statuses else "needs_review" if "needs_review" in statuses else "pass"
        pairs.append({**pair, "status": status, "endpoint_statuses": statuses,
                      "adjudication_complete": "needs_review" not in statuses})
    def summary(rows):
        counts = Counter(r["status"] for r in rows)
        return {"total": len(rows), **{s: counts[s] for s in ("pass", "fail", "needs_review")},
                "adjudication_complete": all(r["adjudication_complete"] for r in rows),
                "confirmed_success_fraction": counts["pass"] / len(rows)}
    groups, worlds = defaultdict(list), defaultdict(list)
    for pair in pairs:
        groups["/".join(pair[k] for k in ("split", "control", "language"))].append(pair)
        worlds[pair["world_id"]].append(pair)
    return {**summary(pairs), "pairs": pairs, "single_case_scores": single,
            "groups": {k: summary(v) for k, v in sorted(groups.items())},
            "worlds": {k: summary(v) for k, v in sorted(worlds.items())},
            "independent_worlds": len(worlds), "protocol_sha256": digest(protocol),
            "promotion_eligible": False}


def main():
    import argparse
    import json
    from pathlib import Path
    from prepare_neural_memory_protocol import materialize
    from review_neural_memory import read_rows, response_hash

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "corpus", "predictions", "reviews", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reviewer", action="append", required=True)
    args = parser.parse_args()
    manifest = materialize(args.config, args.corpus, verify=True)
    inputs, labels, index = (read_rows(Path(args.corpus) / f"dev.{kind}.jsonl")
                            for kind in ("inputs", "labels", "index"))
    report = score_causal_bundle(inputs, labels, index, read_rows(args.predictions),
                                 read_rows(args.reviews), args.reviewer)
    report["provenance"] = {"corpus_manifest_sha256": digest(manifest),
                            "predictions_sha256": response_hash(Path(args.predictions).read_text()),
                            "reviews_sha256": response_hash(Path(args.reviews).read_text())}
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")


if __name__ == "__main__": main()
