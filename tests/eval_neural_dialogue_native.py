#!/usr/bin/env python3
"""Native writer acceptance on an independent synthetic split, not a gold-filled bank."""

import argparse, hashlib, json, os, struct, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch
from neural_slot_model import decode_plan
from train_neural_dialogue_reader import load_model
from train_neural_dialogue import (
    prepare,
    source_words,
    source_types,
    decode_references,
    load_features,
    evaluate,
)
from memory_test_utils import read_slots


def pack(path, rows, magic=False):
    with Path(path).open("wb") as f:
        if magic:
            f.write(b"BNSOBS01")
        f.write(struct.pack("<I", len(rows)))
        for row in rows:
            for key in ("body", "past", "speaker", "timestamp", "source_id"):
                b = (row[key] or "").encode()
                f.write(struct.pack("<I", len(b)) + b)


def main():
    p = argparse.ArgumentParser()
    for key in (
        "checker",
        "gguf",
        "artifact",
        "checkpoint",
        "features",
        "curriculum",
        "output",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--ingester", help="Required for the default commit/restore test")
    p.add_argument("--planner-only", action="store_true",
                   help="Compare source plans and references without committing a bank")
    p.add_argument("--split", default="development")
    args = p.parse_args()
    if not args.planner_only and not args.ingester:
        p.error("--ingester is required unless --planner-only is set")
    torch.set_num_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rows = json.loads(Path(args.curriculum).read_text())[args.split]
    cp = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = load_model(cp)
    cache = load_features(args.features)["features"]
    prepared = prepare({args.split: rows}, cache, "cpu")
    pack(root / "plans.in", rows)
    with (root / "planner.log").open("w") as log:
        subprocess.run(
            [args.checker, args.artifact, args.gguf,
             str(root / "plans.in"), str(root / "plans.out")],
            check=True,
            stderr=log,
            env=dict(os.environ, BITNET_NUM_THREADS="2"),
        )
    native = []
    with (root / "plans.out").open("rb") as f, torch.no_grad():
        assert struct.unpack("<I", f.read(4))[0] == len(rows)

        def string():
            n = struct.unpack("<I", f.read(4))[0]
            return f.read(n).decode()

        for row in rows:
            count, unresolved = struct.unpack("<II", f.read(8))
            whole_ref = string()
            plans = []
            for _ in range(count):
                a, b, action, unknown = struct.unpack("<4I", f.read(16))
                plans.append(
                    dict(
                        start=a,
                        end=b,
                        action=action,
                        unresolved=bool(unknown),
                        reference=string(),
                    )
                )
            item = prepared[row["body"], row["context"]]
            encoded, logits, ends = model.plan(
                item["body"], item["context"], item["body_words"]
            )
            expected = []
            for a, b, action, first, end in decode_plan(
                model.plan_labels(logits, ends), item["body_units"]
            ):
                need, tags = model.plan_reference(
                    encoded[first:end].mean(0),
                    source_words(item, first),
                    source_types(row, item, first),
                    encoded[first:end].amax(0),
                )
                refs = (
                    decode_references(row, tags.argmax(-1), item, first)
                    if need > 0
                    else []
                )
                expected.append(
                    dict(
                        start=a,
                        end=b,
                        action=action,
                        unresolved=bool(need > 0) and not refs,
                        reference="; ".join(refs),
                    )
                )
            assert plans == expected, (row, plans, expected)
            need, tags = model.query_reference(
                encoded.mean(0),
                source_words(item, len(encoded)),
                source_types(row, item, len(encoded)),
                raw_context=model.query_binding_context(item["body"], item["context"]),
            )
            refs = (
                decode_references(row, tags.argmax(-1), item, len(encoded))
                if need > 0
                else []
            )
            assert whole_ref == "; ".join(refs) and bool(unresolved) == (
                bool(need > 0) and not refs
            )
            native.append(plans)
        assert not f.read(), "unexpected trailing source-plan bytes"
    if args.planner_only:
        metrics, _ = evaluate(model, rows, prepared)
        report = dict(
            split=args.split,
            observations=len(rows),
            native_plan_parity_cases=len(rows),
            native_plan_parity_verified=True,
            original_source_byte_spans_and_actor_references_verified=True,
            writer_component=metrics,
            engineering_fixture=bool(cp.get("engineering_fixture", False)),
            model_sha256=hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            curriculum_sha256=hashlib.sha256(Path(args.curriculum).read_bytes()).hexdigest(),
            planner_only=True,
            commit_evaluated=False,
            read_evaluated=False,
            generation_evaluated=False,
            qualification_evaluated=False,
            locomo_used=False,
            production_qualified=False,
        )
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    pack(root / "writes.in", rows, True)
    env = dict(os.environ, BITNET_NUM_THREADS="2")
    with (root / "writer.log").open("w") as log:
        completed = subprocess.run(
            [
                args.ingester,
                args.gguf,
                args.artifact,
                str(root / "writes.in"),
                str(root / "state"),
                "--independent",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=log,
            env=env,
            check=True,
        )
    writes = [json.loads(line) for line in completed.stdout.splitlines()]
    (root / "writes.json").write_text(json.dumps(writes, indent=2))
    assert len(writes) == len(rows)
    tp = fp = fn = unsafe = errors = written = 0
    records = []
    for index, (row, plans, write) in enumerate(zip(rows, native, writes)):
        state = root / f"state.{index}.bnslots"
        if write["failed"]:
            # Failed atomic writes may leave no initial snapshot. Keep the
            # failure in the report and count absent expected facts as misses.
            slots = read_slots(state) if state.exists() else []
        else:
            slots = read_slots(state)
        errors += write["failed"]
        written += write["written"]
        if not write["failed"]:
            assert write["candidates"] == len(plans)
        selected = [s for s in plans if not s["unresolved"] and s["action"] in (1, 2)]
        if not write["failed"]:
            assert len(slots) == len(selected), (row, write, slots, selected)
        actual = set()
        for slot, plan in zip(slots, selected):
            text = row["body"].encode()[plan["start"] : plan["end"]].decode()
            assert slot["source"] == slot["content"] == text
            assert slot["source_start"] == plan["start"] and slot["source_end"] == plan["end"]
            assert slot["reference"] == plan["reference"]
            assert (
                slot["speaker"] == (row["speaker"] or "")
                and slot["observed_at"] == (row["timestamp"] or "")
                and slot["source_id"] == row["source_id"]
            )
            actual.add((text, plan["action"], slot["reference"]))
        expected = {
            (
                row["body"].encode()[s["start"] : s["end"]].decode(),
                s["action"],
                "; ".join(s["references"]),
            )
            for s in row["spans"]
            if s["action"] in (1, 2) and not (s["need"] and not s["references"])
        }
        tp += len(actual & expected)
        fp += len(actual - expected)
        fn += len(expected - actual)
        if row["tag"] == "ambiguous-reference":
            unsafe += bool(actual)
        records.append(
            dict(
                index=index,
                tag=row["tag"],
                observation=row,
                write=write,
                slots=slots,
                correct=len(actual & expected),
                extra=len(actual - expected),
                missing=len(expected - actual),
            )
        )
    report = dict(
        split=args.split,
        observations=len(rows),
        native_plan_parity_cases=len(rows),
        native_commit_f1=2 * tp / max(2 * tp + fp + fn, 1),
        tp=tp,
        fp=fp,
        fn=fn,
        total_written=written,
        writer_errors=errors,
        ambiguous_unsafe_writes=unsafe,
        metadata_roundtrip_verified=True,
        original_body_byte_spans_verified=True,
        bank_constructed_by="actual C predicted writer and durable snapshot",
        locomo_used=False,
        production_qualified=False,
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    (root / "records.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
