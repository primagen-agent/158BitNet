#!/usr/bin/env python3
"""Automatic typed memory through the normal chat HTTP API."""

from __future__ import annotations

import json
import os
from pathlib import Path
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


def chat(base, session_id, text, **extra):
    return post(
        base, "/v1/chat/completions",
        {
            "session_id": session_id,
            "max_tokens": 1,
            "messages": [{
                "role": "user",
                "content": text,
            }],
            **extra,
        },
    )


def stream_chat(base, session_id, text):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "session_id": session_id,
            "max_tokens": 1,
            "stream": True,
            "messages": [{
                "role": "user",
                "content": text,
            }],
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=240
    ) as response:
        body = response.read().decode()
    if "data: [DONE]" not in body:
        raise AssertionError(
            f"stream did not complete: {body}"
        )
    chunks = []
    for event in body.split("\n\n"):
        data = "\n".join(line[6:] for line in event.splitlines() if line.startswith("data: "))
        if data and data != "[DONE]":
            chunks.append(json.loads(data))
    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    result = dict(chunks[-1])
    result["choices"] = [{"message": {"content": content}}]
    return result


def start_server(
    server, model, pair, link, query, writer,
    state_dir, port,
    controller=None,
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
        ] + (["--memory-controller", controller] if controller else []),
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
    raise RuntimeError("typed chat-memory server did not become ready")


def stop_server(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def assert_no_kv(response):
    session = response.get("bitnet_session") or {}
    if (
        session.get("cached_tokens") != 0
        or session.get("reused_tokens") != 0
    ):
        raise AssertionError(f"KV cache was reused: {response}")


def assert_auto_event(response, expected):
    automatic = response.get("memory_auto") or {}
    if (
        response.get("memory_action") != expected["decision"]
        or automatic.get("status") != "stored"
        or automatic.get("stored") != 1
        or automatic.get("writer_fields_source")
            != "neural_autonomous_writer"
        or automatic.get("autonomous_writer") != 1
    ):
        raise AssertionError(
            f"chat input was not automatically stored: {response}"
        )
    for key, value in expected.items():
        if key == "decision":
            continue
        if automatic.get(key) != value:
            raise AssertionError(
                f"automatic field {key}: "
                f"{automatic.get(key)!r} != {value!r}; {response}"
            )
    assert_no_kv(response)


def assert_recall(response):
    automatic = response.get("memory_auto") or {}
    copy = response.get("memory_copy") or {}
    content = response["choices"][0]["message"]["content"]
    if (
        response.get("memory_action") != "ignore"
        or automatic.get("status") != "ignored"
        or automatic.get("stored") != 0
        or content != "a virtual access ticket on quiet mornings"
        or copy.get("mode")
            != "neural_typed_query_activation_then_compiled_pointer"
        or copy.get("record_index") != 2
    ):
        raise AssertionError(
            f"invalid chat memory recall: {response}"
        )
    assert_no_kv(response)


def main():
    if len(sys.argv) != 7:
        print(
            "usage: test_openai_server_typed_chat_auto_memory.py "
            "<server> <model.gguf> <pair.bntpair> <link.bntlink> "
            "<query.bntqact> <writer.bntwrite>",
            file=sys.stderr,
        )
        return 2
    server, model, pair, link, query, writer = sys.argv[1:]
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"missing file: {path}", file=sys.stderr)
            return 2

    create = (
        "[source R4000000E001 | 2021-05-04] a virtual access "
        "ticket after work appears as the active conference "
        "registration selection for Briar."
    )
    distractor = (
        "[source R4000000E002 | 2021-05-07] The current listing "
        "for preferred news source in Cleo's profile says public "
        "radio bulletins near home."
    )
    update = (
        "[source R4000000E013 | 2021-06-10] Under conference "
        "registration, Briar has moved on from a virtual access "
        "ticket after work; retain a virtual access ticket on "
        "quiet mornings."
    )
    query_text = (
        "What is Briar's current conference registration?"
    )
    session_id = "typed-chat-auto"
    with tempfile.TemporaryDirectory(
        prefix="bitnet-typed-chat-auto-"
    ) as state_dir:
        process, base = start_server(
            server, model, pair, link, query, writer,
            state_dir, free_port(),
        )
        try:
            ignored = chat(
                base, "typed-chat-ignore",
                "Explain how ocean currents work.",
            )
            if (
                ignored.get("memory_action") != "ignore"
                or (ignored.get("memory_auto") or {}).get("status")
                    != "ignored"
            ):
                raise AssertionError(
                    f"ordinary question was stored: {ignored}"
                )
            ignored_export = post(
                base, "/v1/memory/export",
                {"session_id": "typed-chat-ignore"},
            )
            if ignored_export.get("typed_events") != 0:
                raise AssertionError(
                    f"ignored chat created an event: {ignored_export}"
                )

            written = chat(base, session_id, create)
            assert_auto_event(written, {
                "decision": "write",
                "operation": "assert",
                "entity": "Briar",
                "predicate": "conference_registration",
                "value": "a virtual access ticket after work",
            })
            unrelated = chat(base, session_id, distractor)
            assert_auto_event(unrelated, {
                "decision": "write",
                "operation": "assert",
                "entity": "Cleo",
                "value": "public radio bulletins near home",
            })
            repeated = chat(base, session_id, create)
            automatic = repeated["memory_auto"]
            if automatic.get("deduplicated") != 1 or automatic["raw_record_index"] != 0:
                raise AssertionError(f"repeated source was reindexed: {repeated}")
            repeated_export = post(base, "/v1/memory/export", {"session_id": session_id})
            if repeated_export["typed_events"] != 2 or repeated_export["episodic_records"] != 2:
                raise AssertionError(f"repeated chat duplicated memory: {repeated_export}")
            changed = chat(base, session_id, update)
            assert_auto_event(changed, {
                "decision": "update",
                "operation": "supersede",
                "entity": "Briar",
                "predicate": "conference_registration",
                "value":
                    "a virtual access ticket on quiet mornings",
                "target_event_id": "auto-event-0",
                "target_resolution": "neural_activation",
            })
            recalled = chat(
                base, session_id, query_text,
            )
            assert_recall(recalled)
            streamed_recall = stream_chat(base, session_id, query_text)
            assert_recall(streamed_recall)
            exported = post(
                base, "/v1/memory/export",
                {"session_id": session_id},
            )
            if (
                exported.get("typed_events") != 3
                or exported.get("episodic_records") != 3
            ):
                raise AssertionError(
                    f"invalid automatic export: {exported}"
                )

            stream_chat(
                base, "typed-chat-stream", create
            )
            streamed_export = post(
                base, "/v1/memory/export",
                {"session_id": "typed-chat-stream"},
            )
            if (
                streamed_export.get("typed_events") != 1
                or streamed_export.get("episodic_records") != 1
            ):
                raise AssertionError(
                    f"streaming chat was not stored: {streamed_export}"
                )

            disabled = chat(
                base, "typed-chat-disabled",
                "Remember this record: " + create,
                memory_auto=False,
            )
            disabled_export = post(
                base, "/v1/memory/export",
                {"session_id": "typed-chat-disabled"},
            )
            if disabled_export.get("typed_events") != 0:
                raise AssertionError(
                    f"memory_auto=false was ignored: "
                    f"{disabled} {disabled_export}"
                )

            natural_session = "typed-chat-natural"
            natural_create = (
                "A virtual access ticket after work appears as the "
                "active conference registration selection for Briar."
            )
            natural_distractor = (
                "The current listing for preferred news source in "
                "Cleo's profile says public radio bulletins near home."
            )
            natural_update = (
                "Under conference registration, Briar has moved on "
                "from a virtual access ticket after work; retain a "
                "virtual access ticket on quiet mornings."
            )
            # This is decision parity, not a model-accuracy assertion. The
            # current writer can misclassify unprefixed creates as updates;
            # the chat gate must not conceal that error by overriding it.
            natural_stored = 0
            from eval_typed_memory_lifecycle import post as post_status
            for text in (natural_create, natural_distractor, natural_update):
                automatic = chat(base, natural_session, text)
                assert_no_kv(automatic)
                status, direct = post_status(base, "/v1/memory/remember", {
                    "session_id": "typed-natural-direct", "memory_record": text,
                })
                actual = automatic["memory_auto"]
                if bool(actual["stored"]) != (status == 200):
                    raise AssertionError(f"chat overwrote writer decision: {actual} {direct}")
                if status == 200:
                    natural_stored += 1
                    for field in ("operation", "entity", "predicate", "value", "raw_record_index"):
                        if actual[field] != direct[field]:
                            raise AssertionError(f"chat/writer {field} mismatch: {actual} {direct}")
                elif actual.get("error") != direct["error"]["message"]:
                    raise AssertionError(f"chat/writer rejection mismatch: {actual} {direct}")
        finally:
            stop_server(process)

        process, base = start_server(
            server, model, pair, link, query, writer,
            state_dir, free_port(),
        )
        try:
            imported = post(
                base, "/v1/memory/import",
                {"session_id": session_id},
            )
            if (
                imported.get("typed_events") != 3
                or imported.get("episodic_records") != 3
            ):
                raise AssertionError(
                    f"invalid automatic import: {imported}"
                )
            assert_recall(chat(
                base, session_id, query_text,
            ))
            assert_recall(stream_chat(base, session_id, query_text))
            status, rejected = post_status(base, "/v1/memory/event", {
                "session_id": session_id, "memory_record": "Alice lives in Tokyo",
                "event_id": "auto-event-0", "episode_id": "duplicate",
                "source_id": "duplicate", "entity": "Alice", "predicate": "lives_in",
                "value": "Tokyo", "operation": "assert", "memory_kind": "property",
            })
            if status != 400:
                raise AssertionError(f"duplicate event was accepted: {rejected}")
            unchanged = post(base, "/v1/memory/export", {"session_id": session_id})
            if unchanged["episodic_records"] != 3 or unchanged["typed_events"] != 3:
                raise AssertionError(f"failed event left orphan source: {unchanged}")
            manifest = Path(state_dir) / f"{session_id}.bnsnapshot"
            original = manifest.read_bytes()
            manifest.write_bytes(b"invalid snapshot")
            try:
                status, rejected = post_status(base, "/v1/memory/import", {"session_id": session_id})
                if status != 500:
                    raise AssertionError(f"bad manifest was accepted: {rejected}")
                assert_recall(chat(base, session_id, query_text))
            finally:
                manifest.write_bytes(original)
        finally:
            stop_server(process)

    print(json.dumps({
        "natural_writer_attempts": 3,
        "natural_writer_stored": natural_stored,
        "natural_check": "decision_parity_not_accuracy",
        "phase": "typed_chat_auto_memory_c",
        "chat_endpoint": "/v1/chat/completions",
        "question_ignored": True,
        "create_update_stored": True,
        "streaming_stored": True,
        "natural_chat_writer_parity": True,
        "implicit_chat_recall": True,
        "explicit_opt_out": True,
        "restart_stable": True,
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
