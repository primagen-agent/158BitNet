#!/usr/bin/env python3
"""Model-dependent C-runtime copy/pointer integration smoke test."""

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


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: test_openai_server_memory_pointer.py "
            "<openai_server> <model.gguf> <memory.bnmem> "
            "<retriever.bnctrl> <pointer.bnptr>",
            file=sys.stderr,
        )
        return 2
    server, model, memory_model, controller, pointer = sys.argv[1:]
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"missing file: {path}", file=sys.stderr)
            return 2
    record = (
        "Conversation event on 10:37 am on 27 June, 2023:\n"
        "Caroline: Thanks, Melanie! This necklace is super special to me - "
        "a gift from my grandma in my home country, Sweden. She gave it to "
        "me when I was young, and it stands for love, faith and strength."
    )
    question = "Where did Caroline move from 4 years ago?"
    with tempfile.TemporaryDirectory(prefix="bitnet-pointer-") as state_dir:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(
            [
                server, model,
                "--host", "127.0.0.1", "--port", str(port),
                "--ctx", "512", "--default-max-tokens", "8",
                "--memory-model", memory_model,
                "--memory-state-dir", state_dir,
                "--episodic-memory",
                "--memory-controller", controller,
                "--memory-pointer", pointer,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.time() + 120
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read())
                try:
                    urllib.request.urlopen(base + "/health", timeout=1)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("pointer server did not become ready")
            post(
                base, "/v1/chat/completions",
                {
                    "session_id": "pointer-test",
                    "memory_action": "store",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": record}],
                },
            )
            extracted = post(
                base, "/v1/memory/extract",
                {"session_id": "pointer-test", "query": question},
            )
            if extracted.get("answer") != "Sweden":
                raise AssertionError(f"wrong extracted answer: {extracted}")
            chat = post(
                base, "/v1/chat/completions",
                {
                    "session_id": "pointer-test",
                    "memory_copy": True,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": question}],
                },
            )
            content = chat["choices"][0]["message"]["content"]
            if content != "Sweden" or "memory_copy" not in chat:
                raise AssertionError(f"copy path not used: {chat}")
            info = chat.get("bitnet_session", {})
            if info.get("cached_tokens") != 0 or info.get("reused_tokens") != 0:
                raise AssertionError(f"KV state was reused: {info}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    print("openai server memory pointer: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
