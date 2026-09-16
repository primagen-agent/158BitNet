#!/usr/bin/env python3
"""Evaluate typed neural current/null recall after export and restart."""

from __future__ import annotations

import argparse
import collections
import json
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


SOURCE_LINE = re.compile(
    r"^\[source ([^ ]+) \| ([^\]]+)\] (.*)$"
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(base, path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=240
        ) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def start_server(args, state_dir, port):
    command = [
            args.server,
            args.gguf,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx",
            "256",
            "--max-tokens",
            "16",
            "--default-max-tokens",
            "4",
            "--memory-state-dir",
            state_dir,
            "--episodic-memory",
            "--typed-pair-model",
            args.pair_model,
            "--typed-link-model",
            args.link_model,
        ]
    if args.query_model:
        command.extend([
            "--typed-query-model",
            args.query_model,
            "--typed-writer-model",
            args.writer_model,
        ])
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(process.stderr.read())
        try:
            with urllib.request.urlopen(
                base + "/health", timeout=1
            ):
                return process, base
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("typed lifecycle server startup timed out")


def stop_server(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def byte_span(text, value):
    start = text.find(value)
    if start < 0:
        raise ValueError(f"{value!r} is absent from {text!r}")
    return (
        len(text[:start].encode()),
        len(text[:start + len(value)].encode()),
    )


def predicate_name(value):
    return re.sub(
        r"[^a-z0-9]+", "_", value.lower()
    ).strip("_")


def load_rows(path):
    worlds = collections.defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            if metadata.get("locomo_used"):
                raise ValueError(
                    "LoCoMo data cannot enter this evaluation"
                )
            worlds[metadata["world_id"]].append(row)
    return worlds


def install_active_world(base, session_id, world_id, row):
    lines = str(row["evidence"]).splitlines()
    events = [
        event
        for event in row["metadata"]["typed_events"]
        if event.get("active")
    ]
    if len(lines) != len(events):
        raise ValueError(f"world shape mismatch: {world_id}")
    active = list(zip(lines, events))
    for record_index, (line, event) in enumerate(active):
        match = SOURCE_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid source line: {line}")
        source_id, valid_time, text = match.groups()
        subject_start, subject_end = byte_span(
            text, event["entity_surface"]
        )
        value_start, value_end = byte_span(
            text, event["value"]
        )
        status, response = post(
            base,
            "/v1/memory/event",
            {
                "session_id": session_id,
                "memory_record": text,
                "event_id": (
                    f"{world_id}-{event['episode']}"
                ),
                "episode_id": f"{world_id}-active",
                "source_id": source_id,
                "entity": event["entity"],
                "predicate": predicate_name(
                    event["predicate"]
                ),
                "value": event["value"],
                "valid_time": valid_time,
                "operation": "assert",
                "memory_kind": "property",
                "raw_record_index": record_index,
                "subject_start": subject_start,
                "subject_end": subject_end,
                "value_start": value_start,
                "value_end": value_end,
            },
        )
        if status != 200:
            raise RuntimeError(
                f"event compile failed: {status} {response}"
            )
    return len(active)


def evaluate_response(row, response):
    intent = row["metadata"]["query_plan"]["intent"]
    if intent == "null":
        return (
            response.get("status") == "fallback"
            and response.get("answer") == ""
        )
    return (
        response.get("status") == "extracted"
        and response.get("answer") == row["answer"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server")
    parser.add_argument("gguf")
    parser.add_argument("pair_model")
    parser.add_argument("link_model")
    parser.add_argument("writer_model")
    parser.add_argument("jsonl")
    parser.add_argument("--query-model", required=True)
    parser.add_argument(
        "--intents", default="current,null"
    )
    args = parser.parse_args()
    wanted = {
        value.strip()
        for value in args.intents.split(",")
        if value.strip()
    }
    worlds = load_rows(args.jsonl)
    results = []
    with tempfile.TemporaryDirectory(
        prefix="bitnet-typed-lifecycle-"
    ) as state_dir:
        for world_offset, (world_id, rows) in enumerate(
            sorted(worlds.items())
        ):
            session_id = f"typed-ood-{world_offset}"
            process, base = start_server(
                args, state_dir, free_port()
            )
            try:
                active_count = install_active_world(
                    base, session_id, world_id, rows[0]
                )
                status, exported = post(
                    base,
                    "/v1/memory/export",
                    {"session_id": session_id},
                )
                if (
                    status != 200
                    or exported.get("typed_events")
                    != active_count
                ):
                    raise RuntimeError(
                        f"export failed: {status} {exported}"
                    )
            finally:
                stop_server(process)

            process, base = start_server(
                args, state_dir, free_port()
            )
            try:
                status, imported = post(
                    base,
                    "/v1/memory/import",
                    {"session_id": session_id},
                )
                if (
                    status != 200
                    or imported.get("typed_events")
                    != active_count
                ):
                    raise RuntimeError(
                        f"import failed: {status} {imported}"
                    )
                for row in rows:
                    intent = row["metadata"][
                        "query_plan"
                    ]["intent"]
                    if intent not in wanted:
                        continue
                    status, response = post(
                        base,
                        "/v1/memory/extract",
                        {
                            "session_id": session_id,
                            "query": row["question"],
                        },
                    )
                    if status != 200:
                        raise RuntimeError(
                            f"recall failed: "
                            f"{status} {response}"
                        )
                    results.append({
                        "sample_id": row["sample_id"],
                        "intent": intent,
                        "gold": row["answer"],
                        "status": response.get("status"),
                        "answer": response.get("answer"),
                        "activation_score":
                            response.get("confidence"),
                        "activation_exists_score":
                            response.get(
                                "activation_exists_score"
                            ),
                        "correct": evaluate_response(
                            row, response
                        ),
                    })
            finally:
                stop_server(process)
            current = [
                item for item in results
                if item["sample_id"].startswith(world_id)
            ]
            print(json.dumps(
                {
                    "phase": "typed_lifecycle_progress",
                    "world": world_id,
                    "active_events": active_count,
                    "examples": len(current),
                    "correct": sum(
                        item["correct"]
                        for item in current
                    ),
                },
                separators=(",", ":"),
            ), flush=True)

    by_intent = {}
    for intent in sorted(wanted):
        selected = [
            item for item in results
            if item["intent"] == intent
        ]
        by_intent[intent] = {
            "examples": len(selected),
            "correct": sum(
                item["correct"] for item in selected
            ),
            "accuracy": (
                sum(item["correct"] for item in selected)
                / max(len(selected), 1)
            ),
            "activation_score_min": min(
                (
                    item["activation_score"]
                    for item in selected
                    if isinstance(
                        item["activation_score"],
                        (int, float),
                    )
                ),
                default=None,
            ),
            "activation_score_max": max(
                (
                    item["activation_score"]
                    for item in selected
                    if isinstance(
                        item["activation_score"],
                        (int, float),
                    )
                ),
                default=None,
            ),
        }
    total = max(len(results), 1)
    print(json.dumps(
        {
            "phase": "typed_memory_lifecycle_eval",
            "examples": len(results),
            "accuracy": (
                sum(item["correct"] for item in results)
                / total
            ),
            "by_intent": by_intent,
            "event_compilation_source":
                "ground_truth_active_events",
            "query_activator":
                "dedicated_typed_query_head",
            "kv_cache_used": False,
            "lora_used": False,
            "rag_used": False,
            "locomo_used": False,
            "failures": [
                item for item in results
                if not item["correct"]
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    main()
