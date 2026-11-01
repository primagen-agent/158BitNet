#!/usr/bin/env python3
"""Train the learned read-decision head on fit and verify it on development.

The reader is frozen. Only the head is trained, on the authored fit view, with
the frozen writer binding the queries exactly as the reader trainer binds them.
Development views are used for the nonregression check and the report, never
for training or for choosing weights among seeds.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import neural_read_head as head_lib  # noqa: E402
from neural_slot_model import KEYS, NeuralSlots  # noqa: E402
from prepare_neural_colloquial import colloquial_curriculum, reader_data  # noqa: E402
import train_neural_dialogue_reader as reader  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = NeuralSlots()
    model.load_state_dict({k: v for k, v in payload["state_dict"].items() if k in KEYS})
    return model.eval(), reader.load_model(payload).eval(), payload


def required_texts(group):
    facts = reader.active_bank_facts(group["facts"])
    queries = [q for q in group["queries"] if not q.get("unresolved")]
    texts = {f["text"] for f in facts} | {q["text"] for q in queries}
    texts |= {r[k] for rows in (facts, queries) for r in rows
              for k in ("reference", "initial_reference", "initial_text") if r.get(k)}
    return texts


def prepare(group, model, saved):
    facts = reader.active_bank_facts(group["facts"])
    queries = [q for q in group["queries"] if not q.get("unresolved")]
    cache = reader.DynamicFeatures(
        model, {"d": {"v": [dict(text=t) for t in sorted(required_texts(group))]}},
        saved, "cpu", freeze_memory_encoder=True)
    bank, ids, mask, meta = reader.make_bank(facts, cache)
    latents = cache.memory_context_latents([f["text"] for f in facts])
    q, identity = reader.query(model, queries, cache)
    score = model.score_bank(q, identity, bank, ids, meta, mask,
                             memory_context_latents=latents).detach()
    gold = torch.tensor([[reader.supported(r, f) for f in facts] for r in queries])
    values = head_lib.view_features(saved, score, [r["text"] for r in queries],
                                    [f["text"] for f in facts])
    return score, gold, values, [r["text"] for r in queries], [f["text"] for f in facts]


def metrics(logits, gold):
    predicted = logits > 0
    positive = gold.any(-1)
    return dict(
        positive_queries=int(positive.sum()),
        positive_exact=int(((predicted == gold).all(-1) & positive).sum()),
        positive_support_set_exact=round(
            float(((predicted == gold).all(-1) & positive).float().sum()
                  / positive.sum().clamp_min(1)), 4),
        support_set_exact=round(float((predicted == gold).all(-1).float().mean()), 4),
        null_false_activation=round(
            float((predicted.any(-1) & ~positive).float().mean()), 4),
        fp=int((predicted & ~gold).sum()),
        fn=int((~predicted & gold).sum()))


def nonregression(candidate, baseline):
    return (candidate["positive_support_set_exact"] >= baseline["positive_support_set_exact"]
            and candidate["fp"] <= baseline["fp"]
            and candidate["null_false_activation"] <= baseline["null_false_activation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", required=True, help="Frozen reader checkpoint")
    parser.add_argument("--features", required=True, help="Identity-bound frozen feature cache")
    parser.add_argument("--gguf", default=str(ROOT / "models" / "bitcpm4-1b-tq2_0.gguf"))
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("read head output directory must be fresh")
    output.mkdir(parents=True)

    model, planner, payload = load(args.reader)
    saved_payload = torch.load(args.features, map_location="cpu", weights_only=False)
    saved = saved_payload["features"]
    backbone = sha256(Path(args.gguf))
    if payload.get("backbone_sha256") != backbone:
        raise ValueError("reader checkpoint backbone mismatch")
    if saved_payload["identity"]["backbone"] != backbone:
        raise ValueError("feature cache backbone mismatch")

    def bound(group):
        return reader.bind_curriculum_queries(group, planner, saved)[0]

    fit = bound(reader_data("fit", colloquial_curriculum("fit")))
    dev_authored = bound(reader_data("development", colloquial_curriculum("development")))
    dev_episode = bound(reader.episode_observation_group("development"))

    views = {}
    for name, group in (("fit-authored", fit), ("dev-authored", dev_authored),
                        ("dev-episode", dev_episode)):
        views[name] = prepare(group, model, saved)

    head = head_lib.fit_head([(views["fit-authored"][2], views["fit-authored"][1])],
                             seeds=tuple(range(1, args.seeds + 1)), steps=args.steps,
                             learning_rate=args.learning_rate,
                             weight_decay=args.weight_decay)

    report = dict(format=head_lib.FORMAT, reader_sha256=sha256(args.reader),
                  features_sha256=sha256(args.features), backbone_sha256=backbone,
                  training=dict(view="fit-authored", seeds=args.seeds,
                                steps=args.steps, learning_rate=args.learning_rate,
                                weight_decay=args.weight_decay,
                                features=list(head_lib.FEATURE_NAMES)),
                  views={})
    clean = True
    with torch.no_grad():
        for name, (score, gold, values, queries, facts) in views.items():
            baseline = metrics(score, gold)
            candidate = metrics(head(values), gold)
            entry = dict(baseline=baseline, candidate=candidate,
                         bank_facts=len(facts), queries=len(queries))
            if name.startswith("dev"):
                entry["nonregression"] = nonregression(candidate, baseline)
                clean &= entry["nonregression"]
            report["views"][name] = entry

    if not clean:
        report["selection"] = "rejected: development nonregression failed"
        (output / "report.json").write_text(json.dumps(report, indent=2))
        raise SystemExit(json.dumps(report, indent=2))
    with (output / "read-head.bin").open("wb") as handle:
        head_lib.write_binary(handle, head, report)
    report["selection"] = "accepted"
    report["artifact_sha256"] = sha256(output / "read-head.bin")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
