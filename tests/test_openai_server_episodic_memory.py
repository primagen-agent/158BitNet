#!/usr/bin/env python3
"""End-to-end V80 episodic-memory persistence regression."""

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
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def start_server(
    server: str, model: str, memory_model: str, state_dir: str, port: int,
    controller: str | None,
) -> subprocess.Popen:
    command = [
        server,
        model,
        "--memory-model",
        memory_model,
        "--memory-state-dir",
        state_dir,
        "--episodic-memory",
        "--episodic-top-k",
        "3",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx",
        "4096",
        "--default-max-tokens",
        "8",
    ]
    if controller is not None:
        command.extend([
            "--memory-controller", controller,
            "--episodic-lexical-weight", "0.75",
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
            with urllib.request.urlopen(base + "/health", timeout=1):
                return process
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("episodic-memory server did not become ready")


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
            "usage: test_openai_server_episodic_memory.py "
            "<openai_server> <model.gguf> <memory.bnmem> "
            "[memory.bnctrl]",
            file=sys.stderr,
        )
        return 2
    controller = sys.argv[4] if len(sys.argv) == 5 else None
    server, model, memory_model = sys.argv[1:4]
    for path in (server, model, memory_model, controller):
        if path is None:
            continue
        if not os.path.exists(path):
            print(f"missing file: {path}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory(prefix="bitnet-episodic-") as state_dir:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        session_id = "episodic-api-test"
        exact_record = "Arbitrary payload Ω-7319 belongs to Mira."
        process = start_server(
            server, model, memory_model, state_dir, port, controller)
        try:
            post(
                base,
                "/v1/chat/completions",
                {
                    "model": "bitnet",
                    "session_id": session_id,
                    "reset_session": True,
                    "memory_action": "store",
                    "messages": [
                        {"role": "user", "content": exact_record},
                    ],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            search = post(
                base,
                "/v1/memory/search",
                {
                    "session_id": session_id,
                    "query": "Mira payload",
                    "top_k": 3,
                },
            )
            if exact_record not in search.get("context", ""):
                raise AssertionError(f"exact record not retrieved: {search}")
            query = post(
                base,
                "/v1/chat/completions",
                {
                    "model": "bitnet",
                    "session_id": session_id,
                    "memory_action": "ignore",
                    "messages": [
                        {"role": "user", "content": "What belongs to Mira?"},
                    ],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            info = query.get("bitnet_session", {})
            if info.get("cached_tokens") != 0 or info.get("reused_tokens") != 0:
                raise AssertionError(f"KV state was reused: {info}")
            exported = post(
                base, "/v1/memory/export", {"session_id": session_id})
            if exported.get("episodic_records") != 1:
                raise AssertionError(f"wrong export record count: {exported}")
        finally:
            stop_server(process)

        for suffix in (".bnstate", ".bnepisodic"):
            if not os.path.isfile(os.path.join(state_dir, session_id + suffix)):
                raise AssertionError(f"missing exported {suffix} file")

        port = free_port()
        base = f"http://127.0.0.1:{port}"
        process = start_server(
            server, model, memory_model, state_dir, port, controller)
        try:
            imported = post(
                base, "/v1/memory/import", {"session_id": session_id})
            if imported.get("episodic_records") != 1:
                raise AssertionError(f"wrong import record count: {imported}")
            search = post(
                base,
                "/v1/memory/search",
                {"session_id": session_id, "query": "Mira payload"},
            )
            if exact_record not in search.get("context", ""):
                raise AssertionError(
                    f"record changed after restart: {search}")
        finally:
            stop_server(process)

    print("openai server episodic memory: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
