#!/usr/bin/env python3
"""Raw text -> autonomous typed event -> neural recall lifecycle test."""

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


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(base: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=240
        ) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise AssertionError(
            f"{path} failed: {error.code} {error.read().decode()}"
        ) from error


def start_server(
    server, model, pair, link, query, writer,
    state_dir, port,
):
    process = subprocess.Popen(
        [
            server, model,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx", "256",
            "--max-tokens", "16",
            "--default-max-tokens", "4",
            "--memory-state-dir", state_dir,
            "--episodic-memory",
            "--typed-pair-model", pair,
            "--typed-link-model", link,
            "--typed-query-model", query,
            "--typed-writer-model", writer,
        ],
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
                return process
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("autonomous memory server did not become ready")


def stop_server(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def assert_compiled(response, expected):
    for key, value in expected.items():
        if response.get(key) != value:
            raise AssertionError(
                f"autonomous field {key}: "
                f"{response.get(key)!r} != {value!r}; {response}"
            )
    if (
        response.get("status") != "ok"
        or response.get("writer_fields_source")
            != "neural_autonomous_writer"
        or response.get("autonomous_writer") != 1
        or response.get("kv_cache_touched") != 0
    ):
        raise AssertionError(
            f"invalid autonomous writer response: {response}"
        )


def assert_recall(response):
    if (
        response.get("status") != "extracted"
        or response.get("answer")
            != "a virtual access ticket on quiet mornings"
        or response.get("record_index") != 2
        or response.get("kv_cache_touched") != 0
        or response.get("recall_mode")
            != "neural_typed_query_activation_then_compiled_pointer"
        or "context" in response
    ):
        raise AssertionError(
            f"invalid autonomous recall: {response}"
        )


def main():
    if len(sys.argv) != 7:
        print(
            "usage: test_openai_server_typed_autonomous_e2e.py "
            "<server> <model.gguf> <pair.bntpair> <link.bntlink> "
            "<query.bntqact> <writer.bntwrite>",
            file=sys.stderr,
        )
        return 2
    server, model, pair, link, query, writer = sys.argv[1:]
    for path in (server, model, pair, link, query, writer):
        if not os.path.isfile(path):
            print(f"missing file: {path}", file=sys.stderr)
            return 2
    records = [
        (
            "[source R4000000E001 | 2021-05-04] a virtual access "
            "ticket after work appears as the active conference "
            "registration selection for Briar.",
            {
                "operation": "assert",
                "entity": "Briar",
                "predicate": "conference_registration",
                "value": "a virtual access ticket after work",
                "valid_time": "2021-05-04",
            },
        ),
        (
            "[source R4000000E002 | 2021-05-07] The current listing "
            "for preferred news source in Cleo's profile says public "
            "radio bulletins near home.",
            {
                "operation": "assert",
                "entity": "Cleo",
                "value": "public radio bulletins near home",
                "valid_time": "2021-05-07",
            },
        ),
        (
            "[source R4000000E013 | 2021-06-10] Under conference "
            "registration, Briar has moved on from a virtual access "
            "ticket after work; retain a virtual access ticket on "
            "quiet mornings.",
            {
                "operation": "supersede",
                "entity": "Briar",
                "predicate": "conference_registration",
                "value":
                    "a virtual access ticket on quiet mornings",
                "valid_time": "2021-06-10",
                "target_event_id": "auto-event-0",
                "target_resolution": "neural_activation",
            },
        ),
    ]
    session_id = "typed-autonomous-e2e"
    with tempfile.TemporaryDirectory(
        prefix="bitnet-typed-autonomous-"
    ) as state_dir:
        port = free_port()
        process = start_server(
            server, model, pair, link, query, writer,
            state_dir, port,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            responses = [
                post(
                    base, "/v1/memory/remember",
                    {
                        "session_id": session_id,
                        "memory_record": text,
                    },
                )
                for text, _ in records
            ]
            for response, (_, expected) in zip(
                responses, records
            ):
                assert_compiled(response, expected)
            before = post(
                base, "/v1/memory/extract",
                {
                    "session_id": session_id,
                    "query":
                        "What is Briar's current conference registration?",
                },
            )
            assert_recall(before)
            exported = post(
                base, "/v1/memory/export",
                {"session_id": session_id},
            )
            if (
                exported.get("episodic_records") != 3
                or exported.get("typed_events") != 3
            ):
                raise AssertionError(f"invalid export: {exported}")
        finally:
            stop_server(process)

        port = free_port()
        process = start_server(
            server, model, pair, link, query, writer,
            state_dir, port,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            imported = post(
                base, "/v1/memory/import",
                {"session_id": session_id},
            )
            if (
                imported.get("episodic_records") != 3
                or imported.get("typed_events") != 3
            ):
                raise AssertionError(f"invalid import: {imported}")
            after = post(
                base, "/v1/memory/extract",
                {
                    "session_id": session_id,
                    "query":
                        "What is Briar's current conference registration?",
                },
            )
            assert_recall(after)
            if after != before:
                raise AssertionError(
                    f"recall changed after restart: {before} != {after}"
                )
        finally:
            stop_server(process)
    print(json.dumps({
        "phase": "typed_autonomous_e2e_c",
        "writer_fields_source": "neural_autonomous_writer",
        "autonomous_writer": True,
        "answer": "a virtual access ticket on quiet mornings",
        "restart_stable": True,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
