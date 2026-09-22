"""Offline double-review workflow. No network judges and no automatic self-grading.

Reviewer identity/independence must be established by the operator. Two strings
do not establish two independent humans; calibration is still a release gate.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from neural_memory_protocol import (Adjudication, AnswerRubric, Claim, EvaluationLevel,
                                    InferenceTrace, response_hash, score_response)
from prepare_neural_memory_protocol import canonical, digest, materialize


def keyed(rows):
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in result:
            raise ValueError("invalid or duplicate case ID")
        result[row["id"]] = row
    return result


def validate_prediction(prediction, runtime):
    if set(prediction) != {"id", "text", "input_sha256", "producer_id", "trace"}:
        raise ValueError("prediction provenance fields required")
    if prediction["input_sha256"] != digest(runtime): raise ValueError("prediction belongs to different input")
    if not isinstance(prediction["text"], str) or not isinstance(prediction["producer_id"], str) or not prediction["producer_id"]:
        raise ValueError("invalid prediction")
    trace = dict(prediction["trace"])
    trace["level"] = EvaluationLevel(trace["level"])
    for field in ("label_fields_in_forward", "future_messages_in_forward"):
        if not isinstance(trace[field], list): raise ValueError("trace fields must be arrays")
        trace[field] = tuple(trace[field])
    InferenceTrace(**trace).validate()


def blind_packet(runtime, prediction):
    validate_prediction(prediction, runtime)
    # Dataset IDs encode scenario names. Replace them before blind review.
    opaque_id = digest({"input": runtime, "response": prediction["text"]})
    packet = {"id": opaque_id, "context": runtime["context"], "episodes": runtime["episodes"],
              "response": prediction["text"]}
    return {**packet, "packet_sha256": digest(packet)}


def review_signature(review):
    return (tuple(sorted({canonical(asdict(c)) for c in review.claims})),
            review.acknowledges_missing_evidence, review.natural_reply, review.unsolicited_memory_narration)


def score_bundle(inputs, labels, index, predictions, reviews, independent_reviewers):
    runtime, gold, meta, outputs = map(keyed, (inputs, labels, index, predictions))
    if not runtime or runtime.keys() != gold.keys() or runtime.keys() != meta.keys():
        raise ValueError("empty or mismatched corpus")
    if outputs.keys() - runtime.keys(): raise ValueError("unknown prediction ID")
    levels = {p["trace"]["level"] for p in outputs.values()}
    producers = {p["producer_id"] for p in outputs.values()}
    if len(levels) > 1 or len(producers) > 1: raise ValueError("do not mix evaluation levels or model producers")
    if len(set(independent_reviewers)) < 2 or any(not isinstance(r, str) or not r for r in independent_reviewers):
        raise ValueError("register at least two independent reviewers")
    grouped = defaultdict(list)
    identities = set()
    packet_ids = {blind_packet(runtime[cid], p)["id"]: cid for cid, p in outputs.items()}
    for row in reviews:
        if set(row) != {"id", "packet_sha256", "adjudication"} or row["id"] not in packet_ids:
            raise ValueError("unknown review or invalid review schema")
        case_id = packet_ids[row["id"]]
        data = dict(row["adjudication"])
        data["claims"] = tuple(Claim(**c) for c in data["claims"])
        review = Adjudication(**data)
        identity = (row["id"], review.reviewer_id)
        if identity in identities: raise ValueError("duplicate reviewer vote")
        identities.add(identity)
        if review.reviewer_id not in independent_reviewers or review.reviewer_id == outputs[case_id]["producer_id"]:
            raise ValueError("unregistered reviewer or self-review")
        packet = blind_packet(runtime[case_id], outputs[case_id])
        if row["packet_sha256"] != packet["packet_sha256"] or review.response_sha256 != response_hash(packet["response"]):
            raise ValueError("review belongs to different context/response")
        grouped[case_id].append(review)
    cases = []
    for case_id, source in runtime.items():
        prediction = outputs.get(case_id)
        result = {"status": "needs_review", "reasons": ["missing_output"]}
        if prediction is not None:
            validate_prediction(prediction, source)
            label = gold[case_id]
            rubric = AnswerRubric(tuple(Claim(**c) for c in label["required_claims"]),
                                  tuple(Claim(**c) for c in label["allowed_claims"]), label["evidence_missing"])
            votes = grouped[case_id]
            if not prediction["text"].strip(): result = score_response(prediction["text"], rubric)
            elif len(votes) < 2:
                result = {"status": "needs_review", "reasons": ["two_independent_reviews_required"]}
            elif len({review_signature(v) for v in votes}) != 1:
                result = {"status": "needs_review", "reasons": ["reviewer_disagreement"]}
            else: result = score_response(prediction["text"], rubric, votes[0])
        cases.append({"id": case_id, "world_id": meta[case_id]["world_id"],
                      "scenario": meta[case_id]["scenario"], "language": meta[case_id]["language"], **result})
    counts = Counter(c["status"] for c in cases)
    groups, worlds = defaultdict(list), defaultdict(list)
    for case in cases:
        groups[case["scenario"] + "/" + case["language"]].append(case)
        worlds[case["world_id"]].append(case)
    summarize = lambda rows: {"total": len(rows), **{status: sum(c["status"] == status for c in rows)
                                                   for status in ("pass", "fail", "needs_review")}}
    return {"total": len(cases), "pass": counts["pass"], "fail": counts["fail"], "needs_review": counts["needs_review"],
            "evaluation_level": next(iter(levels), None), "producer_id": next(iter(producers), None),
            "confirmed_success_fraction": counts["pass"] / len(cases), "adjudication_complete": not counts["needs_review"],
            "promotion_eligible": False, "promotion_note": "Software workflow only; calibration and stage gates still required",
            "independent_worlds": len(worlds), "groups": {k: summarize(v) for k, v in sorted(groups.items())},
            "worlds": {k: summarize(v) for k, v in sorted(worlds.items())}, "cases": cases}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "score"))
    for name in ("config", "corpus", "predictions", "output"): parser.add_argument("--" + name, required=True)
    parser.add_argument("--reviews"); parser.add_argument("--reviewer", action="append", default=[])
    args = parser.parse_args()
    manifest = materialize(args.config, args.corpus, verify=True)
    root = Path(args.corpus)
    inputs, labels, index = (read_rows(root / f"dev.{kind}.jsonl") for kind in ("inputs", "labels", "index"))
    predictions = read_rows(args.predictions)
    if args.mode == "pack":
        runtime = keyed(inputs); keyed(predictions)
        if any(p["id"] not in runtime for p in predictions): raise ValueError("unknown prediction")
        output = {"format": "neural-memory-blind-review-v1", "packets": [blind_packet(runtime[p["id"]], p) for p in predictions]}
    else:
        if not args.reviews: parser.error("score requires --reviews")
        output = score_bundle(inputs, labels, index, predictions, read_rows(args.reviews), args.reviewer)
        output["provenance"] = {"corpus_manifest_sha256": digest(manifest), "prediction_sha256": response_hash(Path(args.predictions).read_text()),
                                "review_sha256": response_hash(Path(args.reviews).read_text())}
    with Path(args.output).open("x") as stream: json.dump(output, stream, ensure_ascii=False, indent=2); stream.write("\n")


if __name__ == "__main__": main()
