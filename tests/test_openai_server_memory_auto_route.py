#!/usr/bin/env python3
"""BNCTRL4 auto-routing plus BNPTR5 persistence regression."""
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
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def start_server(
    server: str, model: str, controller: str,
    pointer: str, state_dir: str,
) -> tuple[subprocess.Popen, str]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            server, model,
            "--host", "127.0.0.1", "--port", str(port),
            "--ctx", "512", "--default-max-tokens", "1",
            "--memory-state-dir", state_dir,
            "--episodic-memory",
            "--memory-controller", controller,
            "--memory-pointer", pointer,
            "--episodic-lexical-weight", "0.25",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(process.stderr.read())
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            return process, base
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("auto-memory server did not become ready")


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def assert_no_kv(response: dict) -> None:
    info = response.get("bitnet_session", {})
    if (
        info.get("cached_tokens", 0) != 0
        or info.get("reused_tokens", 0) != 0
    ):
        raise AssertionError(f"KV cache was reused: {info}")


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: test_openai_server_memory_auto_route.py "
            "<openai_server> <model.gguf> <controller.bnctrl> "
            "<write-pointer.bnptr5>",
            file=sys.stderr,
        )
        return 2
    server, model, controller, pointer = sys.argv[1:]
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"missing file: {path}", file=sys.stderr)
            return 2
    session_id = "auto-memory"
    payload = "amber-orchid"
    updated_payload = "cobalt-cedar"
    write_request = (
        "The persistent setting I use for project codename is "
        + payload + ".")
    update_request = (
        "Amend the persistent project codename entry so it reads "
        + updated_payload + ".")
    delete_request = (
        "Purge the persistent record of my project codename.")
    query = "What is my project codename?"
    with tempfile.TemporaryDirectory(
        prefix="bitnet-auto-memory-",
    ) as state_dir:
        process, base = start_server(
            server, model, controller, pointer, state_dir)
        try:
            ignored = post(
                base, "/v1/chat/completions",
                {
                    "session_id": "auto-ignore",
                    "max_tokens": 1,
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Give me a concise overview of ocean currents."
                        ),
                    }],
                },
            )
            if ignored.get("memory_action") != "ignore":
                raise AssertionError(
                    f"ordinary request was not ignored: {ignored}")
            ignored_export = post(
                base, "/v1/memory/export",
                {"session_id": "auto-ignore"},
            )
            if ignored_export.get("episodic_records") != 0:
                raise AssertionError(
                    f"ignored request was stored: {ignored_export}")

            written = post(
                base, "/v1/chat/completions",
                {
                    "session_id": session_id,
                    "max_tokens": 1,
                    "messages": [{
                        "role": "user", "content": write_request,
                    }],
                },
            )
            if written.get("memory_action") != "write":
                raise AssertionError(
                    f"memory request was not routed to write: {written}")
            assert_no_kv(written)
            search = post(
                base, "/v1/memory/search",
                {
                    "session_id": session_id,
                    "query": payload,
                    "top_k": 1,
                },
            )
            if search.get("context", "").strip() != (
                f"[memory 1] {payload}"
            ):
                raise AssertionError(
                    f"pointer did not store exact payload: {search}")
            updated = post(
                base, "/v1/chat/completions",
                {
                    "session_id": session_id,
                    "max_tokens": 1,
                    "messages": [{
                        "role": "user", "content": update_request,
                    }],
                },
            )
            if updated.get("memory_action") != "update":
                raise AssertionError(
                    f"memory request was not routed to update: {updated}")
            assert_no_kv(updated)
            search = post(
                base, "/v1/memory/search",
                {
                    "session_id": session_id,
                    "query": query,
                    "top_k": 1,
                },
            )
            if search.get("context", "").strip() != (
                f"[memory 1] {updated_payload}"
            ):
                raise AssertionError(
                    f"pointer did not store updated payload: {search}")
            exported = post(
                base, "/v1/memory/export",
                {"session_id": session_id},
            )
            if exported.get("episodic_records") != 1:
                raise AssertionError(f"export failed: {exported}")
        finally:
            stop_server(process)

        process, base = start_server(
            server, model, controller, pointer, state_dir)
        try:
            imported = post(
                base, "/v1/memory/import",
                {"session_id": session_id},
            )
            if imported.get("episodic_records") != 1:
                raise AssertionError(f"import failed: {imported}")
            recalled = post(
                base, "/v1/chat/completions",
                {
                    "session_id": session_id,
                    "memory_copy": True,
                    "max_tokens": 1,
                    "messages": [{
                        "role": "user", "content": query,
                    }],
                },
            )
            if recalled.get("memory_action") != "ignore":
                raise AssertionError(
                    f"recall query was not routed to read: {recalled}")
            content = recalled["choices"][0]["message"]["content"]
            if content != updated_payload:
                raise AssertionError(
                    f"wrong auto-routed recall: {recalled}")
            assert_no_kv(recalled)
            deleted = post(
                base, "/v1/chat/completions",
                {
                    "session_id": session_id,
                    "max_tokens": 1,
                    "messages": [{
                        "role": "user", "content": delete_request,
                    }],
                },
            )
            if deleted.get("memory_action") != "delete":
                raise AssertionError(
                    f"memory request was not routed to delete: {deleted}")
            assert_no_kv(deleted)
            deleted_recall = post(
                base, "/v1/chat/completions",
                {
                    "session_id": session_id,
                    "memory_copy": True,
                    "max_tokens": 1,
                    "messages": [{
                        "role": "user", "content": query,
                    }],
                },
            )
            if (
                deleted_recall["choices"][0]["message"]["content"]
                != "No information available."
            ):
                raise AssertionError(
                    f"deleted memory leaked: {deleted_recall}")
            if (
                deleted_recall.get("memory_copy", {}).get("mode")
                != "deletion_tombstone"
            ):
                raise AssertionError(
                    f"tombstone was not used: {deleted_recall}")
            assert_no_kv(deleted_recall)
        finally:
            stop_server(process)
    print("openai server automatic memory route: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
