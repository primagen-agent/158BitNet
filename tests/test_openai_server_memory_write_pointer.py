#!/usr/bin/env python3
"""BNPTR5 write extraction plus episodic export/import regression."""

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
    server: str, model: str, pointer: str, state_dir: str,
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
            "--memory-pointer", pointer,
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
    raise RuntimeError("write-pointer server did not become ready")


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print(
            "usage: test_openai_server_memory_write_pointer.py "
            "<openai_server> <model.gguf> <write-pointer.bnptr5> "
            "[samples.jsonl]",
            file=sys.stderr,
        )
        return 2
    server, model, pointer = sys.argv[1:4]
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"missing file: {path}", file=sys.stderr)
            return 2
    session_id = "write-pointer-test"
    write_request = (
        "Without interpreting it, lodge this sequence in long-term "
        "storage: Sweden"
    )
    expected = "Sweden"
    if len(sys.argv) == 5:
        with open(sys.argv[4], encoding="utf-8") as source:
            sample = json.loads(next(source))
        write_request = sample["messages"][0][0]["content"]
        expected = sample["messages"][1][-1]["content"]
    query = "Return the stored payload exactly."
    with tempfile.TemporaryDirectory(
        prefix="bitnet-write-pointer-",
    ) as state_dir:
        process, base = start_server(
            server, model, pointer, state_dir)
        try:
            response = post(
                base, "/v1/chat/completions",
                {
                    "session_id": session_id,
                    "memory_action": "write",
                    "max_tokens": 1,
                    "messages": [
                        {"role": "user", "content": write_request},
                    ],
                },
            )
            if response.get("memory_action") != "write":
                raise AssertionError(f"write not resolved: {response}")
            search = post(
                base, "/v1/memory/search",
                {
                    "session_id": session_id,
                    "query": expected,
                    "top_k": 1,
                },
            )
            if search.get("context", "").strip() != (
                f"[memory 1] {expected}"
            ):
                raise AssertionError(
                    f"write pointer did not store exact payload: {search}")
            exported = post(
                base, "/v1/memory/export",
                {"session_id": session_id},
            )
            if exported.get("episodic_records") != 1:
                raise AssertionError(f"export failed: {exported}")
        finally:
            stop_server(process)

        process, base = start_server(
            server, model, pointer, state_dir)
        try:
            imported = post(
                base, "/v1/memory/import",
                {"session_id": session_id},
            )
            if imported.get("episodic_records") != 1:
                raise AssertionError(f"import failed: {imported}")
            response = post(
                base, "/v1/chat/completions",
                {
                    "session_id": session_id,
                    "memory_action": "ignore",
                    "memory_copy": True,
                    "max_tokens": 1,
                    "messages": [
                        {"role": "user", "content": query},
                    ],
                },
            )
            content = response["choices"][0]["message"]["content"]
            if content != expected:
                raise AssertionError(f"wrong copied payload: {response}")
            info = response.get("bitnet_session", {})
            if info.get("cached_tokens", 0) != 0:
                raise AssertionError(f"KV cache was used: {info}")
            if info.get("reused_tokens", 0) != 0:
                raise AssertionError(f"KV cache was reused: {info}")
        finally:
            stop_server(process)
    print("openai server BNPTR5 write pointer: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
