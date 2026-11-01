#!/usr/bin/env python3
"""Authored episode acceptance: predicted writes, restart, all-bank activation.

Labels are used exclusively after native execution. Neither the writer nor the
reader receives support masks, gold entity prefixes or benchmark material.
Missing writes count as missing evidence in end-to-end activation metrics.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from prepare_neural_episodes import episode_curriculum
from neural_slot_model import observation_context
from train_neural_dialogue_reader import add_episodes, supported
from eval_neural_dialogue_native import pack
from memory_test_utils import read_slots


def specification(split, curriculum=None):
    if curriculum is not None:
        return curriculum_specification(split, curriculum)
    if split == "blind":
        raise ValueError("the blind split requires --curriculum")
    rows = episode_curriculum(split)
    grouped = {}
    for row in rows:
        if row["tag"] == "episode-fact":
            s = row["spans"][0]
            text = row["body"].encode()[s["start"] : s["end"]].decode()
            grouped.setdefault((row["speaker"], text), []).append(row)
    facts = []
    for i, variants in enumerate(grouped.values()):
        row = variants[1 + i % 2]
        s = row["spans"][0]
        content = row["body"].encode()[s["start"] : s["end"]].decode()
        facts.append(
            dict(
                row=row,
                content=content,
                reference="; ".join(s["references"]),
                text=s["references"][0] + ". " + content,
                entity=s["references"][0],
                prop=s["prop"],
                kind=s["kind"],
            )
        )
    group = add_episodes(dict(facts=[], queries=[]), split)
    labels = {q["text"]: q for q in group["queries"]}
    queries = [(r, labels[r["body"]]) for r in rows if r["tag"] == "episode-query"]
    greetings = [r for r in rows if r["tag"] == "ordinary-negative"]
    return facts, queries, greetings


def curriculum_specification(split, path):
    """Read authored observation rows; labels are never packed into native input."""
    saved = json.loads(Path(path).read_text())
    if split not in saved or not isinstance(saved[split], list):
        raise ValueError(
            "curriculum must contain observation rows for the requested split"
        )
    rows = saved[split]
    grouped = {}
    for row in rows:
        if row["tag"] != "episode-fact":
            continue
        for span in row["spans"]:
            raw = row["body"].encode()
            a, b = span["start"], span["end"]
            if not 0 <= a < b <= len(raw) or span["action"] != 1:
                raise ValueError(
                    "native episode curriculum requires valid WRITE source spans"
                )
            content = raw[a:b].decode()
            references = span["references"]
            if not isinstance(references, list) or any(
                not isinstance(ref, str) or not ref.strip() for ref in references
            ):
                raise ValueError("authored episode references must be nonempty strings")
            if span["need"]:
                if len(references) != 1:
                    raise ValueError("authored episode facts require one expected external reference")
                reference = references[0]
            else:
                if references:
                    raise ValueError("need=False requires empty references")
                reference = ""
            entity = span.get("entity", reference)
            if not isinstance(entity, str) or not entity.strip():
                raise ValueError("authored episode facts require a nonempty canonical entity")
            if span["need"] and "ref_ranges" in span:
                ranges = span["ref_ranges"]
                if not isinstance(ranges, list) or len(ranges) != 1:
                    raise ValueError("external reference requires one original source address")
                source = ranges[0]
                if not isinstance(source, dict) or source.get("document") not in ("body", "context"):
                    raise ValueError("reference address must name body or context")
                document = source["document"]
                source_raw = (
                    row["body"] if document == "body" else observation_context(
                        row["speaker"], row["timestamp"], row["past"]
                    )
                ).encode()
                start, end = source.get("start"), source.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source_raw):
                    raise ValueError("reference address must be a valid UTF-8 source range")
                if document == "body" and end > a:
                    raise ValueError("body reference must be in the causal prefix before its source span")
                # Match the structural boundary trim in both native and Python
                # reference decoders; source addresses remain literal UTF-8.
                if source_raw[start:end].decode().strip(".,!?;:'\"") != reference:
                    raise ValueError("reference address must match its original source bytes")
            key = (entity, content)
            grouped.setdefault(key, []).append((row, span, content, entity, reference))
    facts = []
    for i, variants in enumerate(grouped.values()):
        # Prefer conversational wrappers. No predicted spans or dataset answers
        # participate in this fixed observation selection.
        chosen = 1 + i % (len(variants) - 1) if len(variants) > 1 else 0
        row, span, content, entity, reference = variants[chosen]
        facts.append(
            dict(
                row=row,
                content=content,
                reference=reference,
                text=entity + ". " + content,
                entity=entity,
                prop=span["prop"],
                kind=span["kind"],
            )
        )
    queries = []
    for row in rows:
        if row["tag"] != "episode-query":
            continue
        # Each observation carries its own labels: contextual/self questions
        # can share body bytes while referring to different people.
        keys = row.get("support_keys")
        if not isinstance(keys, list) or any(
            not isinstance(k, dict) or set(k) != {"entity", "prop", "kind"}
            for k in keys
        ):
            raise ValueError("each authored query needs explicit external support_keys")
        wanted = [
            f
            for f in facts
            if any(
                all(f[field] == key[field] for field in ("entity", "prop", "kind"))
                for key in keys
            )
        ]
        if any(
            not any(
                all(f[field] == key[field] for field in ("entity", "prop", "kind"))
                for f in facts
            )
            for key in keys
        ):
            raise ValueError("query support key lacks an expected source fact")
        queries.append(
            (
                row,
                dict(
                    text=row["body"],
                    entities=list(row["entities"]),
                    prop=row["prop"],
                    kind=row["kind"],
                    support_texts=[f["text"] for f in wanted],
                ),
            )
        )
    negatives = [
        row for row in rows if row["tag"] in ("ordinary-negative", "negative-quotation")
    ]
    if not facts or not queries:
        raise ValueError("native episode curriculum requires facts and queries")
    return facts, queries, negatives


def write_observations(facts, negatives):
    # Several facts may originate in one literal observation. Commit that
    # observation once, even though each expected fact remains in scoring.
    seen, writes = set(), []
    for row in [f["row"] for f in facts]:
        key = tuple(
            row[k] or "" for k in ("body", "past", "speaker", "timestamp", "source_id")
        )
        if key not in seen:
            seen.add(key)
            writes.append(row)
    positive_count = len(writes)
    for row in negatives:
        key = tuple(
            row[k] or "" for k in ("body", "past", "speaker", "timestamp", "source_id")
        )
        if key not in seen:
            seen.add(key)
            writes.append(row)
    return writes, positive_count


def score(slots, facts, outputs):
    """Score end-to-end evidence and conditionally diagnose its first failed stage.

    A complete write strictly matches the entire labelled original-source
    fragment and expected reference. A useful truncated prefix is not a
    complete match. Stage metrics are component diagnostics conditional on
    complete writes and resolved references; they do not replace end-to-end
    scores or assess metadata reasoning. ``complete`` reports set convergence,
    independently of whether the activated support set is correct.
    """
    ids = [
        [
            s["id"]
            for s in slots
            if s["content"] == f["content"] and s["reference"] == f["reference"]
        ]
        for f in facts
    ]
    tp = fp = fn = correct = null_correct = null_count = positives = 0
    incomplete = unresolved = resolved_converged = converged_correct = 0
    fully_written = write_blocked = reference_rejected = 0
    activation_eligible = activation_exact = 0
    records = []
    for row, label, result in outputs:
        wanted = [i for i, f in enumerate(facts) if supported(label, f)]
        expected_ids = {n for i in wanted for n in ids[i]}
        written_support_count = sum(bool(ids[i]) for i in wanted)
        active = set(result["activated_ids"])
        found = sum(bool(active & set(ids[i])) for i in wanted)
        tp += found
        fn += len(wanted) - found
        fp += len(active - expected_ids)
        exact = found == len(wanted) and active == expected_ids
        complete = result.get("complete", False)
        is_unresolved = result.get("unresolved", False)
        incomplete += not complete
        unresolved += is_unresolved
        resolved_converged += complete and not is_unresolved
        if wanted:
            positives += 1
            correct += exact
            converged_correct += exact and complete and not is_unresolved
            if written_support_count != len(wanted):
                write_blocked += 1
                first_failed_stage = "write"
            else:
                fully_written += 1
                if is_unresolved:
                    reference_rejected += 1
                    first_failed_stage = "reference"
                else:
                    activation_eligible += 1
                    activation_exact += exact
                    first_failed_stage = "none" if exact else "activation"
        else:
            null_count += 1
            null_correct += exact
            first_failed_stage = "none" if exact else "null_false_activation"
        records.append(
            dict(
                question=row["body"],
                query_class=row.get("query_class"),
                source_id=row.get("source_id"),
                expected_facts=wanted,
                expected_ids=sorted(expected_ids),
                expected_support_count=len(wanted),
                written_support_count=written_support_count,
                exact=exact,
                first_failed_stage=first_failed_stage,
                **result,
            )
        )
    return (
        dict(
            correct_writes=sum(bool(x) for x in ids),
            expected_writes=len(facts),
            write_recall=sum(bool(x) for x in ids) / len(facts),
            write_f1=2 * sum(bool(x) for x in ids) / max(len(facts) + len(slots), 1),
            positive_queries=positives,
            positive_exact=correct,
            end_to_end_positive_exact=correct / max(positives, 1),
            incomplete_queries=incomplete,
            unresolved_queries=unresolved,
            resolved_converged_queries=resolved_converged,
            converged_positive_exact=converged_correct / max(positives, 1),
            support_set_exact=(correct + null_correct) / len(outputs),
            activation_f1=2 * tp / max(2 * tp + fp + fn, 1),
            tp=tp,
            fp=fp,
            fn=fn,
            null_queries=null_count,
            null_false_activation=(
                (1 - null_correct / null_count) if null_count else None
            ),
            stage_metrics=dict(
                positive_queries=positives,
                fully_written_positive_queries=fully_written,
                write_blocked_positive_queries=write_blocked,
                reference_rejected_positive_queries=reference_rejected,
                activation_eligible_positive_queries=activation_eligible,
                activation_exact_positive_queries=activation_exact,
                conditional_activation_exact_accuracy=(
                    activation_exact / activation_eligible
                    if activation_eligible else None
                ),
            ),
        ),
        records,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("ingester", "checker", "gguf", "artifact", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--curriculum", type=Path)
    p.add_argument("--read-head", type=Path,
                   help="Optional learned read-decision artifact; activation only")
    p.add_argument(
        "--split", choices=("development", "sealed", "blind"), default="development"
    )
    p.add_argument("--question-limit", type=int)
    p.add_argument("--stop-below-write-f1", type=float)
    p.add_argument("--stop-below-positive-exact", type=float)
    p.add_argument("--trace-activation", action="store_true",
                   help="Write diagnostic per-round activation factors; decisions and binary output are unchanged")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    facts, queries, greetings = specification(a.split, a.curriculum)
    curriculum_identity = (
        dict(
            curriculum=str(a.curriculum.resolve()),
            curriculum_sha256=hashlib.sha256(a.curriculum.read_bytes()).hexdigest(),
        )
        if a.curriculum is not None
        else dict(curriculum="prepare_neural_episodes.episode_curriculum")
    )
    eligible_questions = len(queries)
    if a.question_limit is not None:
        if a.question_limit < 1:
            raise ValueError("question limit must be positive")
        queries = queries[: a.question_limit]
    writes, positive_observations = write_observations(facts, greetings)
    pack(a.output / "writes.in", writes, True)
    state = (a.output / "bank.bnslots").resolve()
    env = dict(os.environ, BITNET_NUM_THREADS="2")
    run = subprocess.run(
        [
            str(a.ingester),
            str(a.gguf),
            str(a.artifact),
            str(a.output / "writes.in"),
            str(state),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    (a.output / "writer.log").write_text(run.stderr)
    events = [json.loads(line) for line in run.stdout.splitlines()]
    slots = read_slots(state)
    digest = hashlib.sha256(state.read_bytes()).hexdigest()
    (a.output / "written.json").write_text(
        json.dumps(dict(events=events, slots=slots), indent=2)
    )
    correct = sum(
        any(
            s["content"] == f["content"] and s["reference"] == f["reference"]
            for s in slots
        )
        for f in facts
    )
    write_f1 = 2 * correct / max(len(facts) + len(slots), 1)
    if a.stop_below_write_f1 is not None and write_f1 < a.stop_below_write_f1:
        report = dict(
            split=a.split,
            correct_writes=correct,
            expected_writes=len(facts),
            write_f1=write_f1,
            observations=len(writes),
            bank_records=len(slots),
            writer_errors=sum(e["failed"] for e in events),
            stopped=True,
            stop_reason="write_f1_below_predeclared_guard",
            read_evaluated=False,
            eligible_questions=eligible_questions,
            questions=0,
            locomo_used=False,
            production_qualified=False,
            guard_write_f1=a.stop_below_write_f1,
            model_sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
            **curriculum_identity,
        )
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    with (a.output / "recall.in").open("wb") as f:
        f.write(struct.pack("<I", len(queries)))
        for row, _ in queries:
            for t in [str(state)] + [
                row[k] or ""
                for k in ("body", "past", "speaker", "timestamp", "source_id")
            ]:
                b = t.encode()
                f.write(struct.pack("<I", len(b)) + b)
    with (a.output / "reader.log").open("w") as log:
        trace_args = ["--trace", str(a.output / "activation-trace.jsonl")] if a.trace_activation else []
        head_args = ["--read-head", str(a.read_head)] if a.read_head is not None else []
        subprocess.run(
            [
                str(a.checker),
                str(a.artifact),
                str(a.gguf),
                str(a.output / "recall.in"),
                str(a.output / "recall.out"),
            ] + trace_args + head_args,
            check=True,
            stderr=log,
            env=env,
        )
    outputs = []
    with (a.output / "recall.out").open("rb") as f:
        assert struct.unpack("<I", f.read(4))[0] == len(queries)

        def string():
            return f.read(struct.unpack("<I", f.read(4))[0]).decode()

        for row, label in queries:
            n, unresolved, complete = struct.unpack("<3I", f.read(12))
            reference, prompt = string(), string()
            active = list(struct.unpack("<" + "Q" * n, f.read(8 * n)))
            outputs.append(
                (
                    row,
                    label,
                    dict(
                        activated_ids=active,
                        reference=reference,
                        unresolved=bool(unresolved),
                        complete=bool(complete),
                        prompt=prompt,
                    ),
                )
            )
        assert not f.read()
    assert (
        hashlib.sha256(state.read_bytes()).hexdigest() == digest
    ), "read mutated snapshot"
    report, records = score(slots, facts, outputs)
    report["activation_trace"] = str(a.output / "activation-trace.jsonl") if a.trace_activation else None
    report["activation_trace_scope"] = "Diagnostic factors only; convergence is not support coverage" if a.trace_activation else None
    report.update(
        split=a.split,
        bank_records=len(slots),
        observations=len(writes),
        writer_errors=sum(e["failed"] for e in events),
        greeting_false_writes=sum(
            e["written"]
            for e, row in zip(events, writes)
            if row["tag"] == "ordinary-negative"
        ),
        negative_false_writes=sum(e["written"] for e in events[positive_observations:]),
        positive_observations=positive_observations,
        negative_observations=len(writes) - positive_observations,
        bank_constructed_by="actual C predicted writer",
        restored_in_new_process=True,
        readonly_snapshot_verified=True,
        activation_rounds=3,
        locomo_used=False,
        generation_evaluated=False,
        production_qualified=False,
        model_sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
        read_head_sha256=(
            hashlib.sha256(a.read_head.read_bytes()).hexdigest()
            if a.read_head is not None else None
        ),
        eligible_questions=eligible_questions,
        partial=len(queries) < eligible_questions,
        guard_write_f1=a.stop_below_write_f1,
        guard_positive_exact=a.stop_below_positive_exact,
        stopped=a.stop_below_positive_exact is not None
        and report["end_to_end_positive_exact"] < a.stop_below_positive_exact,
        stop_reason=(
            "positive_exact_below_predeclared_guard"
            if a.stop_below_positive_exact is not None
            and report["end_to_end_positive_exact"] < a.stop_below_positive_exact
            else None
        ),
        **curriculum_identity,
    )
    query_classes = sorted({row.get("query_class") for row, _, _ in outputs} - {None})
    if query_classes:
        report["by_query_class"] = {
            kind: score(
                slots,
                facts,
                [item for item in outputs if item[0].get("query_class") == kind],
            )[0]
            for kind in query_classes
        }
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    (a.output / "records.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
