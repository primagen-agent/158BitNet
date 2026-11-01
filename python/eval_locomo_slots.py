#!/usr/bin/env python3
"""Official LoCoMo10 evaluation of learned full-bank slot memory, evaluation only.

Original turns with speaker/time/captions -> automatic model writes -> export ->
real process restart -> verified empty probe -> import -> read-only questions.
Gold answers/evidence are used only after serving, never as model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from locomo_evaluation import (
    DATA_SHA256,
    UPSTREAM,
    UPSTREAM_REVISION,
    official_f1,
    normalize,
    save,
    append,
)
from memory_test_utils import (
    free_port,
    stop_server,
    assert_no_kv,
    validate_memory_network,
)
from memory_test_utils import read_slots


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class RequestError(Exception):
    pass


class QualityStop(Exception):
    """A predeclared evaluation guard fired; never a training signal."""


def structured_turns(sample):
    conversation = sample["conversation"]
    keys = sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    for key in keys:
        timestamp = conversation[key + "_date_time"]
        for turn in conversation[key]:
            body = turn["text"]
            if turn.get("blip_caption"):
                body += f"\n[Image caption: {turn['blip_caption']}]"
            yield dict(
                dia_id=turn["dia_id"],
                body=body,
                speaker=turn["speaker"],
                timestamp=timestamp,
                text=f"[{timestamp}] {turn['speaker']}: {body}",
            )


def evidence_ids(annotated):
    """Split annotated evidence into turn ids.

    LoCoMo packs several ids into one entry ("D8:6; D9:17", "D21:18 D21:22")
    and contains a few malformed tokens; only well-formed ids are comparable
    with activated turns.
    """
    if isinstance(annotated, str):
        annotated = [annotated]
    out = set()
    for entry in annotated:
        for token in re.split(r"[;,\s]+", str(entry).strip()):
            if re.fullmatch(r"D\d+:\d+", token):
                out.add(token)
    return out


def questions_for_prefix(sample, turns, covered_only=False):
    observed = {turn["dia_id"] for turn in turns}
    selected = []
    for i, qa in enumerate(sample["qa"]):
        evidence = evidence_ids(qa.get("evidence") or [])
        if covered_only and (
            qa["category"] == 5 or not evidence or not evidence <= observed
        ):
            continue
        selected.append(dict(qa, original_qa_index=i))
    return selected


def quality_stop_reason(rows, minimum=8, min_f1=0.2, min_evidence_any=0.5):
    if any(row["error"] for row in rows):
        return "query request failure"
    known = [r for r in rows if r["category"] != 5]
    if len(known) < minimum:
        return None
    score = summarize(known)["categories_1_4"]
    if score["f1"] < min_f1:
        return (
            f"answer F1 {score['f1']:.4f} < {min_f1:.4f} after {len(known)} questions"
        )
    if (
        score["annotated_evidence_any"] is not None
        and score["annotated_evidence_any"] < min_evidence_any
    ):
        return f"annotated evidence hit {score['annotated_evidence_any']:.4f} < {min_evidence_any:.4f}"
    return None


def request(base, endpoint, payload):
    req = urllib.request.Request(
        base + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RequestError(f"HTTP {error.code}: {error.read().decode()}") from error


def write_payload(sid, turn, history):
    payload = dict(
        session_id=sid,
        max_tokens=1,
        messages=[dict(role="user", content=t["text"]) for t in history]
        + [dict(role="user", content=turn["body"])],
        memory_observation=dict(
            speaker=turn["speaker"],
            timestamp=turn["timestamp"],
            source_id=turn["dia_id"],
        ),
    )
    return payload


def question_payload(sid, question, max_tokens):
    return dict(
        session_id=sid,
        max_tokens=max_tokens,
        memory_read_only=True,
        messages=[dict(role="user", content=question)],
    )


def plain_history(history):
    # Same literal role-delimiter escaping and plain transcript as HTTP. This
    # protects framing, without choosing subjects or interpreting any fact.
    return "".join(
        "user: " + t["text"].replace("<|", "< |").replace("|>", "| >") + "\n"
        for t in history
    )


def pack_writes(path, turns, history_turns):
    with Path(path).open("wb") as f:
        f.write(b"BNSOBS01" + struct.pack("<I", len(turns)))
        for i, turn in enumerate(turns):
            history = turns[max(0, i - history_turns) : i] if history_turns else []
            fields = (
                turn["body"],
                plain_history(history),
                turn["speaker"],
                turn["timestamp"],
                turn["dia_id"],
            )
            for text in fields:
                b = text.encode()
                f.write(struct.pack("<I", len(b)) + b)


def batch_write(args, turns, sid, directory, write_states, marker):
    write_states.mkdir()
    source = directory / "write_inputs.bin"
    snapshot = write_states / f"{sid}.bnslots"
    pack_writes(source, turns, args.history_turns)
    writes, provenance = [], {}
    records = 0
    began = time.monotonic()
    with (directory / "writer.log").open("a") as log:
        process = subprocess.Popen(
            [args.ingester, args.gguf, args.network, str(source), str(snapshot)],
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=dict(os.environ, BITNET_NUM_THREADS=str(args.threads)),
        )
        try:
            for line in process.stdout:
                value = json.loads(line)
                i = value.pop("index")
                if i != len(writes) or i >= len(turns):
                    raise AssertionError("batch writer sequence mismatch")
                failed = value.pop("failed")
                value.pop("cpu_s")
                count = int(value["records"])
                for address in range(records + 1, count + 1):
                    provenance[str(address)] = turns[i]["dia_id"]
                records = count
                now = time.monotonic()
                row = dict(
                    index=i,
                    **turns[i],
                    memory=dict(value, mode="native_slot_writer"),
                    error="C memory writer failed" if failed else None,
                    elapsed_s=now - began,
                )
                began = now
                writes.append(row)
                append(directory / "write_progress.jsonl", row)
                if args.early_stop and (
                    failed or (i + 1 >= args.zero_write_after and records == 0)
                ):
                    reason = (
                        "native writer failure"
                        if failed
                        else f"zero committed records after {i+1} raw turns"
                    )
                    save(
                        directory / "stopped.json",
                        dict(
                            phase="write",
                            reason=reason,
                            observations=len(writes),
                            records=records,
                        ),
                    )
                    raise QualityStop(reason)
                if (i + 1) % (5 if args.early_stop else 25) == 0 or i + 1 == len(turns):
                    print(
                        json.dumps(
                            dict(
                                phase="write",
                                conversation=int(sid.split("-")[1]),
                                done=i + 1,
                                total=len(turns),
                                records=records,
                                errors=sum(bool(w["error"]) for w in writes),
                            )
                        ),
                        flush=True,
                    )
            if process.wait() != 0 or len(writes) != len(turns):
                raise RuntimeError(f"batch writer failed; see {directory/'writer.log'}")
        finally:
            if process.poll() is None:
                stop_server(process)
            process.stdout.close()
    slots = read_slots(snapshot)
    if len(slots) != records or any(str(s["id"]) not in provenance for s in slots):
        raise AssertionError("batch snapshot provenance differs")
    save(
        marker,
        dict(
            writes=writes,
            export=dict(
                records=records,
                memory_mode="neural_slots",
                writer="native_C_ns_ingest_observation",
            ),
            slots=slots,
            provenance=provenance,
            snapshot_sha256=digest(snapshot),
        ),
    )


def verify_response(response, readonly=False, expected_records=None):
    assert_no_kv(response)
    memory = response.get("memory") or {}
    if (
        memory.get("mode") != "neural_slots_prompt"
        or memory.get("persistent_kv_capacity") != 0
    ):
        raise AssertionError("response lacks the full-bank/no-KV memory protocol")
    if readonly and (
        memory.get("read_only") != 1
        or any(memory.get(k) != 0 for k in ("written", "updated", "deleted"))
    ):
        raise AssertionError("evaluation question changed memory")
    if expected_records is not None and memory.get("records") != expected_records:
        raise AssertionError("persistent bank count changed during query")
    return memory


def score_question(qa, response, provenance, error=None):
    category = int(qa["category"])
    memory = response["memory"] if response else {}
    ids = memory.get("activated_ids", [])
    if any(i not in provenance for i in ids):
        raise AssertionError("activation address has no raw-turn provenance")
    activated_turns = sorted({provenance[i] for i in ids})
    raw = response["choices"][0]["message"]["content"].strip() if response else ""
    evidence = qa.get("evidence") or []
    annotated = evidence_ids(evidence)
    hit = annotated & set(activated_turns)
    known_evidence = bool(annotated) and category in (1, 2, 3, 4)
    # These external metrics compare source turns, not fact spans. LoCoMo's
    # annotations may omit relevant turns, so extra turns need not be irrelevant.
    precision = len(hit) / len(activated_turns) if activated_turns else 0.0
    return dict(
        category=category,
        question=qa["question"],
        gold=qa.get("answer"),
        annotated_evidence=sorted(annotated),
        activated_turns=activated_turns,
        memory=memory,
        answer=raw,
        error=error,
        read_only=True,
        f1=official_f1(raw, qa["answer"], category) if category != 5 else None,
        activation_conditioned_f1=(
            official_f1(raw if ids else "", qa["answer"], category)
            if category != 5
            else None
        ),
        normalized_exact=(
            normalize(raw) == normalize(qa.get("answer", "")) if category != 5 else None
        ),
        annotated_evidence_recall=len(hit) / len(annotated) if annotated else None,
        annotated_evidence_any=bool(hit) if annotated else None,
        annotated_evidence_all=annotated <= set(activated_turns) if annotated else None,
        annotated_evidence_precision=precision if known_evidence else None,
        annotated_evidence_f1=(
            2 * len(hit) / (len(annotated) + len(activated_turns))
            if known_evidence else None
        ),
        annotated_support_set_exact=(
            annotated == set(activated_turns) if known_evidence else None
        ),
        null_activation_rejected=not ids and not error if category == 5 else None,
        bitnet_session=response.get("bitnet_session") if response else None,
    )


def summarize(rows):
    def group_summary(group):
        evidence = [r for r in group if r["annotated_evidence_recall"] is not None]
        known = [r for r in group if r["category"] != 5]
        known_evidence = [r for r in known if r["annotated_evidence_precision"] is not None]
        null = [r for r in group if r["category"] == 5]
        mean = lambda key, values: (
            sum(r[key] for r in values) / len(values) if values else None
        )
        return dict(
            total=len(group),
            request_errors=sum(bool(r["error"]) for r in group),
            activated=sum(bool(r["memory"].get("activated_ids")) for r in group),
            incomplete_activation=sum(r["memory"].get("complete") == 0 for r in group),
            f1=mean("f1", known),
            activation_conditioned_f1=mean("activation_conditioned_f1", known),
            normalized_exact=mean("normalized_exact", known),
            evidence_annotated_questions=len(evidence),
            annotated_evidence_recall=mean("annotated_evidence_recall", evidence),
            annotated_evidence_any=mean("annotated_evidence_any", evidence),
            annotated_evidence_all=mean("annotated_evidence_all", evidence),
            known_evidence_annotated_questions=len(known_evidence),
            annotated_evidence_precision=mean("annotated_evidence_precision", known_evidence),
            annotated_evidence_f1=mean("annotated_evidence_f1", known_evidence),
            annotated_support_set_exact=mean("annotated_support_set_exact", known_evidence),
            null_activation_rejection=mean("null_activation_rejected", null),
        )

    return dict(
        questions=len(rows),
        by_category={
            str(c): group_summary([r for r in rows if r["category"] == c])
            for c in sorted({r["category"] for r in rows})
        },
        categories_1_4=group_summary([r for r in rows if r["category"] != 5]),
    )


def start_server(args, directory, state_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    command = [
        args.server,
        args.gguf,
        "--memory-network",
        args.network,
        "--memory-state-dir",
        str(state_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx",
        "8192",
        "--max-tokens",
        str(args.max_tokens),
        "--default-max-tokens",
        "1",
    ]
    if args.read_head:
        command += ["--read-head", args.read_head]
    env = dict(os.environ, BITNET_NUM_THREADS=str(args.threads))
    with (directory / "server.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(1200):
            if process.poll() is not None:
                raise RuntimeError(f"server exited; see {directory/'server.log'}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    return process, base
            except (OSError, TimeoutError):
                time.sleep(0.1)
        raise RuntimeError("server startup timeout")
    except BaseException:
        stop_server(process)
        raise


def run_conversation(args, sample, index, directory):
    directory.mkdir(exist_ok=True)
    sid = f"locomo-{index}"
    turns = list(structured_turns(sample))
    if args.turn_limit:
        turns = turns[: args.turn_limit]
    questions = questions_for_prefix(sample, turns, args.prefix_covered_only)
    if args.question_limit:
        questions = questions[: args.question_limit]
    if not questions:
        raise ValueError("no evaluation questions covered by this prefix")
    save(
        directory / "question_selection.json",
        dict(
            covered_only=args.prefix_covered_only,
            original_qa_indices=[q["original_qa_index"] for q in questions],
            observed_turns=len(turns),
        ),
    )
    marker = directory / "written.json"
    writes_reused = marker.exists()
    write_states, query_states = directory / "write_states", directory / "query_states"
    if not marker.exists() and args.ingester:
        if write_states.exists():
            write_states.rename(directory / f"partial-write-{time.time_ns()}")
        journal = directory / "write_progress.jsonl"
        if journal.exists():
            journal.rename(directory / f"partial-progress-{time.time_ns()}.jsonl")
        batch_write(args, turns, sid, directory, write_states, marker)
    if not marker.exists():
        if write_states.exists():
            write_states.rename(directory / f"partial-write-{time.time_ns()}")
        journal = directory / "write_progress.jsonl"
        if journal.exists():
            journal.rename(directory / f"partial-progress-{time.time_ns()}.jsonl")
        process, base = start_server(args, directory, write_states)
        writes, provenance = [], {}
        records = 0
        try:
            for i, turn in enumerate(turns):
                began = time.monotonic()
                history = (
                    turns[max(0, i - args.history_turns) : i]
                    if args.history_turns
                    else []
                )
                try:
                    response = request(
                        base, "/v1/chat/completions", write_payload(sid, turn, history)
                    )
                    memory = verify_response(response)
                    updated_count = int(memory["records"])
                    for address in range(records + 1, updated_count + 1):
                        provenance[str(address)] = turn["dia_id"]
                    records = updated_count
                    row = dict(index=i, **turn, memory=memory, error=None)
                except RequestError as error:
                    # A write can commit before a later generation error. Infer
                    # only newly published addresses, from this raw request.
                    snapshot = write_states / f"{sid}.bnslots"
                    if snapshot.exists():
                        slots = read_slots(snapshot)
                        for slot in slots:
                            if str(slot["id"]) not in provenance:
                                provenance[str(slot["id"])] = turn["dia_id"]
                        records = len(slots)
                    row = dict(index=i, **turn, memory=None, error=str(error))
                row["elapsed_s"] = time.monotonic() - began
                writes.append(row)
                append(journal, row)
                if args.early_stop and (
                    row["error"] or (i + 1 >= args.zero_write_after and records == 0)
                ):
                    reason = (
                        "HTTP write failure"
                        if row["error"]
                        else f"zero committed records after {i+1} raw turns"
                    )
                    save(
                        directory / "stopped.json",
                        dict(
                            phase="write",
                            reason=reason,
                            observations=len(writes),
                            records=records,
                        ),
                    )
                    raise QualityStop(reason)
                if (i + 1) % 25 == 0 or i + 1 == len(turns):
                    print(
                        json.dumps(
                            dict(
                                phase="write",
                                conversation=index,
                                done=i + 1,
                                total=len(turns),
                                records=records,
                                errors=sum(bool(w["error"]) for w in writes),
                            )
                        ),
                        flush=True,
                    )
            # Even an all-error/all-ignore conversation exports an empty bank.
            if not writes or not any(w["memory"] for w in writes):
                request(
                    base, "/v1/chat/completions", question_payload(sid, "Hello.", 1)
                )
            exported = request(base, "/v1/memory/export", dict(session_id=sid))
            state = write_states / f"{sid}.bnslots"
            slots = read_slots(state)
            if len(slots) != exported["records"] or any(
                str(s["id"]) not in provenance for s in slots
            ):
                raise AssertionError("exported address provenance differs")
            save(
                marker,
                dict(
                    writes=writes,
                    export=exported,
                    slots=slots,
                    provenance=provenance,
                    snapshot_sha256=digest(state),
                ),
            )
        finally:
            stop_server(process)
    written = json.loads(marker.read_text())
    result_path = directory / "queries.json"
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    if query_states.exists():
        query_states.rename(directory / f"previous-query-{time.time_ns()}")
    process, base = start_server(args, directory, query_states)
    try:
        probe = request(
            base,
            "/v1/chat/completions",
            question_payload(sid, questions[0]["question"], 1),
        )
        cold = verify_response(probe, readonly=True, expected_records=0)
        if cold["activated_ids"]:
            raise AssertionError("fresh process recalled before import")
        snapshot = query_states / f"{sid}.bnslots"
        shutil.copyfile(write_states / f"{sid}.bnslots", snapshot)
        if digest(snapshot) != written["snapshot_sha256"]:
            raise AssertionError("exported snapshot changed")
        imported = request(base, "/v1/memory/import", dict(session_id=sid))
        if imported["records"] != written["export"]["records"]:
            raise AssertionError("import count differs")
        for i in range(len(rows), len(questions)):
            qa = questions[i]
            began = time.monotonic()
            try:
                response = request(
                    base,
                    "/v1/chat/completions",
                    question_payload(sid, qa["question"], args.max_tokens),
                )
                verify_response(
                    response, readonly=True, expected_records=imported["records"]
                )
                row = score_question(qa, response, written["provenance"])
            except RequestError as error:
                row = score_question(qa, None, written["provenance"], str(error))
            row.update(
                index=i,
                original_qa_index=qa["original_qa_index"],
                conversation=index,
                elapsed_s=time.monotonic() - began,
            )
            rows.append(row)
            save(result_path, rows)
            if args.early_stop:
                if digest(snapshot) != written["snapshot_sha256"]:
                    raise AssertionError("read-only phase mutated snapshot")
                reason = quality_stop_reason(
                    rows,
                    args.early_stop_min_questions,
                    args.early_stop_min_f1,
                    args.early_stop_min_evidence,
                )
                print(
                    json.dumps(
                        dict(
                            phase="query",
                            conversation=index,
                            done=i + 1,
                            total=len(questions),
                            summary=summarize(rows),
                            stop_reason=reason,
                        )
                    ),
                    flush=True,
                )
                if reason:
                    save(
                        directory / "stopped.json",
                        dict(
                            phase="query",
                            reason=reason,
                            questions=len(rows),
                            summary=summarize(rows),
                            snapshot_unchanged=True,
                        ),
                    )
                    raise QualityStop(reason)
            if (i + 1) % 25 == 0 or i + 1 == len(questions):
                print(
                    json.dumps(
                        dict(
                            phase="query",
                            conversation=index,
                            done=i + 1,
                            total=len(questions),
                            summary=summarize(rows),
                        )
                    ),
                    flush=True,
                )
        if digest(snapshot) != written["snapshot_sha256"]:
            raise AssertionError("read-only phase mutated snapshot")
        save(
            directory / "result.json",
            dict(
                complete=True,
                summary=summarize(rows),
                writes=len(written["writes"]),
                writes_reused=writes_reused,
                write_errors=sum(bool(r["error"]) for r in written["writes"]),
                final_records=imported["records"],
                cold_probe=cold,
                snapshot_unchanged=True,
                writer_actions=dict(
                    Counter(
                        str(w["memory"]["action"])
                        for w in written["writes"]
                        if w["memory"]
                    )
                ),
            ),
        )
    finally:
        stop_server(process)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "gguf", "network"):
        p.add_argument(name)
    p.add_argument("--data", default="data/evaluation/locomo10.json")
    p.add_argument(
        "--ingester",
        help="Optional batch C observation writer; same model plan as serving",
    )
    p.add_argument(
        "--read-head",
        help="Optional learned read-decision artifact; activation only",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--history-turns", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--turn-limit", type=int, default=0)
    p.add_argument("--question-limit", type=int, default=0)
    p.add_argument(
        "--prefix-covered-only",
        action="store_true",
        help="Partial pilot: score only questions whose annotated evidence is fully in the written prefix",
    )
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--early-stop-min-questions", type=int, default=8)
    p.add_argument("--early-stop-min-f1", type=float, default=0.2)
    p.add_argument("--early-stop-min-evidence", type=float, default=0.5)
    p.add_argument("--zero-write-after", type=int, default=25)
    args = p.parse_args()
    if (
        min(args.workers, args.threads, args.max_tokens) < 1
        or min(args.history_turns, args.turn_limit, args.question_limit) < 0
    ):
        raise ValueError("invalid resource/limit settings")
    if args.early_stop and args.workers != 1:
        raise ValueError("early-stop evaluation requires one worker")
    if args.prefix_covered_only and not args.turn_limit:
        raise ValueError("prefix-covered pilot requires --turn-limit")
    if min(args.early_stop_min_questions, args.zero_write_after) < 1 or not (
        0 <= args.early_stop_min_f1 <= 1 and 0 <= args.early_stop_min_evidence <= 1
    ):
        raise ValueError("invalid early-stop policy")
    for name in ("server", "gguf", "network"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.ingester:
        args.ingester = str(Path(args.ingester).resolve())
    if args.read_head:
        args.read_head = str(Path(args.read_head).resolve())
    geometry = validate_memory_network(args.network, args.gguf)
    data = json.loads(Path(args.data).read_text())
    if digest(args.data) != DATA_SHA256:
        raise ValueError("not the pinned official LoCoMo10 dataset")
    selected = [int(i) for i in args.conversations.split(",")]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(i not in range(10) for i in selected)
    ):
        raise ValueError("invalid conversation indices")
    if args.early_stop and len(selected) != 1:
        raise ValueError("early-stop runs evaluate one conversation per stage")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sources = [
        "src/neural_slots.c",
        "src/neural_slots.h",
        "examples/openai_server.c",
        "python/eval_locomo_slots.py",
        "python/locomo_evaluation.py",
        "tests/memory_test_utils.py",
    ]
    if args.ingester:
        sources.append("tools/neural_slot_ingest.c")
    artifact_names = ("server", "gguf", "network") + (
        ("ingester",) if args.ingester else ()
    ) + (("read_head",) if args.read_head else ())
    manifest = dict(
        format="LOCOMO_NEURAL_SLOTS_V1",
        evaluation_only=True,
        model_geometry=geometry,
        upstream=UPSTREAM,
        upstream_revision=UPSTREAM_REVISION,
        dataset_sha256=digest(args.data),
        artifacts={
            n: dict(path=getattr(args, n), sha256=digest(getattr(args, n)))
            for n in artifact_names
        },
        code_sha256={n: digest(ROOT / n) for n in sources},
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
        conversations=selected,
        workers=args.workers,
        threads=args.threads,
        history_turns=args.history_turns,
        max_tokens=args.max_tokens,
        turn_limit=args.turn_limit,
        question_limit=args.question_limit,
        prefix_covered_only=args.prefix_covered_only,
        early_stop=dict(
            enabled=args.early_stop,
            min_questions=args.early_stop_min_questions,
            min_answer_f1=args.early_stop_min_f1,
            min_evidence_hit=args.early_stop_min_evidence,
            zero_write_after=args.zero_write_after,
        ),
        protocol=(
            "native C ns_ingest_observation batch write and ns_bank_save"
            if args.ingester
            else "raw-turn automatic HTTP write and export"
        )
        + "; restart HTTP server; empty probe; import; readonly question; unchanged snapshot; zero cached/reused tokens",
        input="original turn, supplied speaker/time/caption, up to four previous raw turns by default; no generated assistant replies",
        gold_injected=False,
        rag=False,
        training=False,
        threshold=(
            "read-decision head logit above zero"
            if args.read_head
            else "NS_ACTIVATION_PROBABILITY 0.1 on the reader joint probability"
        ),
        read_decision_head=bool(args.read_head),
        evidence_metric="known annotated questions: turn-level precision, recall, F1 and strict annotated support-set equality; annotations are nonexhaustive and turn coverage does not establish span-level factual sufficiency; external scoring only, not serving, early-stop or training inputs",
        answer_metric="official category 1-4 F1 on natural generation; activation-conditioned F1 is not proof of grounding",
        category_5_metric="null activation rejection only; not generated abstention accuracy",
    )
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            "configuration/weights/source changed; use a fresh output directory"
        )
    save(path, manifest)
    all_rows = []
    finished = []
    failures = {}
    stopped = {}
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                run_conversation, args, data[i], i, root / f"conversation_{i}"
            ): i
            for i in selected
        }
        for future in as_completed(future_map):
            i = future_map[future]
            try:
                all_rows.extend(future.result())
                finished.append(i)
            except QualityStop as error:
                stopped[str(i)] = str(error)
                partial = root / f"conversation_{i}" / "queries.json"
                if partial.exists():
                    all_rows.extend(json.loads(partial.read_text()))
            except Exception as error:
                failures[str(i)] = repr(error)
            report = dict(
                complete=len(finished) == len(selected),
                full_locomo=len(selected) == 10
                and not args.turn_limit
                and not args.question_limit,
                completed_conversations=sorted(finished),
                failed_conversations=failures,
                stopped_early=stopped,
                elapsed_s=time.monotonic() - began,
                planned_questions=sum(
                    min(
                        len(
                            questions_for_prefix(
                                data[i],
                                list(structured_turns(data[i]))[
                                    : args.turn_limit or None
                                ],
                                args.prefix_covered_only,
                            )
                        ),
                        args.question_limit or len(data[i]["qa"]),
                    )
                    for i in selected
                ),
                summary=summarize(all_rows),
            )
            save(root / "summary.json", report)
            print(json.dumps(report), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
