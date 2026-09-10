#!/usr/bin/env python3
"""C-runtime BNRET1 write/search/export/restart/import regression."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(base, path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def start(server, model, retriever, state_dir, port):
    process = subprocess.Popen([
        server, model,
        "--host", "127.0.0.1", "--port", str(port),
        "--ctx", "512", "--default-max-tokens", "1",
        "--memory-state-dir", state_dir,
        "--episodic-memory",
        "--memory-retriever", retriever,
        "--episodic-top-k", "3",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(process.stderr.read())
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1)
            return process
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("retriever server did not become ready")


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    if len(sys.argv) != 4:
        print(
            "usage: test_openai_server_memory_retriever.py "
            "<openai_server> <model.gguf> <retriever.bnret1>",
            file=sys.stderr)
        return 2
    server, model, retriever = sys.argv[1:]
    for path in (server, model, retriever):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    records = [
        "Alice moved to Kyoto in April and works as a teacher.",
        "Bob moved to Lisbon in June and works as a designer.",
        "Alice enjoys jazz and folk music and lived in Seattle before.",
        "Alice no longer lives in Kyoto.",
    ]
    query = "Where does Alice live now?"
    with tempfile.TemporaryDirectory(
            prefix="bitnet-retriever-") as state_dir:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        process = start(server, model, retriever, state_dir, port)
        try:
            priorities = [0.2, 0.4, 0.9, 0.8]
            for record, priority in zip(records, priorities):
                response = post(base, "/v1/chat/completions", {
                    "session_id": "retriever-lifecycle",
                    "reset_session": False,
                    "memory_action": "write",
                    "memory_record": record,
                    "memory_priority": priority,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": record}],
                })
                if abs(
                        float(response.get("memory_priority", -1.0))
                        - priority) > 1e-6:
                    raise AssertionError(
                        f"memory priority not acknowledged: {response}")
                stats = response.get("bitnet_session", {})
                if stats.get("cached_tokens", 0) or stats.get(
                        "reused_tokens", 0):
                    raise AssertionError(
                        f"write reused KV cache: {stats}")
            before = post(base, "/v1/memory/search", {
                "session_id": "retriever-lifecycle",
                "query": query,
                "top_k": 3,
            })
            if not before.get("context"):
                raise AssertionError("empty retriever context before export")
            post(base, "/v1/memory/event", {
                "session_id": "retriever-lifecycle",
                "event_id": "alice-residence-v1",
                "episode_id": "episode-3",
                "source_id": "D1:3",
                "entity": "Alice",
                "predicate": "lives_in",
                "value": "Seattle",
                "valid_time": "2025-01-01",
                "operation": "assert",
                "memory_kind": "property",
                "raw_record_index": 2,
            })
            residence = post(base, "/v1/memory/event", {
                "session_id": "retriever-lifecycle",
                "event_id": "alice-residence-v2",
                "episode_id": "episode-1",
                "source_id": "D1:1",
                "entity": "Alice",
                "predicate": "lives_in",
                "value": "Kyoto",
                "valid_time": "2026-01-01",
                "operation": "supersede",
                "memory_kind": "property",
                "target_event_id": "alice-residence-v1",
                "raw_record_index": 0,
            })
            if (
                residence.get("memory_kind") != "property"
                or residence.get("subject_start") != 0
                or residence.get("value_start") is None
            ):
                raise AssertionError(
                    f"event evidence was not preserved: {residence}")
            for event_id, value in (
                    ("alice-preference-jazz", "jazz"),
                    ("alice-preference-folk", "folk music")):
                post(base, "/v1/memory/event", {
                    "session_id": "retriever-lifecycle",
                    "event_id": event_id,
                    "episode_id": "episode-3",
                    "source_id": "D1:3",
                    "entity": "Alice",
                    "predicate": "likes",
                    "value": value,
                    "operation": "assert",
                    "memory_kind": "set",
                    "raw_record_index": 2,
                })
            current = post(base, "/v1/memory/event/current", {
                "session_id": "retriever-lifecycle",
                "entity": "Alice",
                "predicate": "lives_in",
            })
            if current.get("value") != "Kyoto":
                raise AssertionError(f"bad current event: {current}")
            active = post(base, "/v1/memory/event/active", {
                "session_id": "retriever-lifecycle",
                "entity": "Alice",
                "predicate": "likes",
            })
            if (
                active.get("active_count") != 2
                or [item.get("value") for item in active.get("events", [])]
                != ["folk music", "jazz"]
            ):
                raise AssertionError(f"bad set-valued events: {active}")
            exported = post(base, "/v1/memory/export", {
                "session_id": "retriever-lifecycle"})
            if exported.get("episodic_records") != len(records):
                raise AssertionError(f"bad export: {exported}")
            if exported.get("typed_events") != 4:
                raise AssertionError(f"events not exported: {exported}")
        finally:
            stop(process)

        process = start(server, model, retriever, state_dir, port)
        try:
            imported = post(base, "/v1/memory/import", {
                "session_id": "retriever-lifecycle"})
            if imported.get("episodic_records") != len(records):
                raise AssertionError(f"bad import: {imported}")
            if imported.get("typed_events") != 4:
                raise AssertionError(f"events not imported: {imported}")
            after = post(base, "/v1/memory/search", {
                "session_id": "retriever-lifecycle",
                "query": query,
                "top_k": 3,
            })
            if after.get("context") != before.get("context"):
                raise AssertionError(
                    "retriever ordering changed after restart/import")
            current = post(base, "/v1/memory/event/current", {
                "session_id": "retriever-lifecycle",
                "entity": "Alice",
                "predicate": "lives_in",
            })
            if current.get("event_id") != "alice-residence-v2":
                raise AssertionError(
                    f"event version changed after import: {current}")
            post(base, "/v1/memory/event", {
                "session_id": "retriever-lifecycle",
                "event_id": "alice-residence-retract",
                "episode_id": "episode-3",
                "source_id": "D1:3",
                "entity": "Alice",
                "predicate": "lives_in",
                "value": "Kyoto",
                "operation": "retract",
                "memory_kind": "property",
                "polarity": "negative",
                "target_event_id": "alice-residence-v2",
                "raw_record_index": 3,
            })
            current = post(base, "/v1/memory/event/current", {
                "session_id": "retriever-lifecycle",
                "entity": "Alice",
                "predicate": "lives_in",
            })
            if current.get("status") != "not_found":
                raise AssertionError(
                    f"retracted event remained active: {current}")
        finally:
            stop(process)
    print("openai server BNRET1 lifecycle: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
