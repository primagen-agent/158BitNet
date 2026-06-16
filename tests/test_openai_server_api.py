#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def fail(message):
    print(f"FAIL {message}", file=sys.stderr)
    return 1


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def request_json(url, payload=None, timeout=120):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def request_sse(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        for raw_event in body.split("\n\n"):
            raw_event = raw_event.strip()
            if not raw_event:
                continue
            data_lines = []
            for line in raw_event.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                events.append("\n".join(data_lines))
        return resp.status, resp.headers.get("Content-Type", ""), events


def wait_until_ready(base_url, deadline):
    last_error = None
    while time.time() < deadline:
        try:
            status, body = request_json(f"{base_url}/v1/models", timeout=2)
            if status == 200 and body.get("object") == "list":
                return True, None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.1)
    return False, last_error


def main():
    if len(sys.argv) < 3:
        return fail("usage: test_openai_server_api.py <openai_server> <model.gguf>")

    server_path = sys.argv[1]
    model_path = sys.argv[2]
    if not os.path.exists(server_path):
        return fail(f"server binary not found: {server_path}")
    if not os.path.exists(model_path):
        return fail(f"model not found: {model_path}")

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [server_path, model_path, "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ok, error = wait_until_ready(base_url, time.time() + 60)
        if not ok:
            stderr = proc.stderr.read() if proc.poll() is not None else ""
            return fail(f"server not ready: {error}; stderr={stderr}")

        status, models = request_json(f"{base_url}/v1/models")
        if status != 200 or models.get("object") != "list" or not models.get("data"):
            return fail(f"bad models response: {models}")

        chat_payload = {
            "model": "bitnet",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "The capital of France is"},
            ],
            "max_tokens": 16,
        }
        status, chat = request_json(f"{base_url}/v1/chat/completions", chat_payload)
        if status != 200:
            return fail(f"chat status {status}")
        if chat.get("object") != "chat.completion":
            return fail(f"bad chat object: {chat}")
        choices = chat.get("choices") or []
        if not choices:
            return fail(f"missing choices: {chat}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or len(content) == 0:
            return fail(f"empty chat content: {chat}")
        usage = chat.get("usage") or {}
        if usage.get("completion_tokens", 0) <= 0 or usage.get("total_tokens", 0) <= 0:
            return fail(f"bad usage: {chat}")
        perf = chat.get("bitnet_perf") or {}
        if perf.get("decode_tok_s", 0) <= 0:
            return fail(f"missing decode speed: {chat}")

        completion_payload = {
            "model": "bitnet",
            "prompt": "The capital of France is",
            "max_tokens": 16,
        }
        status, completion = request_json(f"{base_url}/v1/completions", completion_payload)
        if status != 200 or completion.get("object") != "text_completion":
            return fail(f"bad completion response: {completion}")
        if not completion.get("choices") or not completion["choices"][0].get("text"):
            return fail(f"empty completion text: {completion}")

        stream_payload = {
            "model": "bitnet",
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 4,
            "stream": True,
        }
        status, content_type, events = request_sse(
            f"{base_url}/v1/chat/completions", stream_payload
        )
        if status != 200:
            return fail(f"stream status {status}")
        if "text/event-stream" not in content_type:
            return fail(f"bad stream content-type: {content_type}")
        if not events or events[-1] != "[DONE]":
            return fail(f"missing stream DONE: {events}")
        content_deltas = []
        for event in events[:-1]:
            obj = json.loads(event)
            if obj.get("object") != "chat.completion.chunk":
                return fail(f"bad stream object: {obj}")
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content_deltas.append(delta["content"])
        if not "".join(content_deltas):
            return fail(f"empty stream content: {events}")

        completion_stream_payload = {
            "model": "bitnet",
            "prompt": "The capital of France is",
            "max_tokens": 4,
            "stream": True,
        }
        status, content_type, events = request_sse(
            f"{base_url}/v1/completions", completion_stream_payload
        )
        if status != 200:
            return fail(f"completion stream status {status}")
        if "text/event-stream" not in content_type:
            return fail(f"bad completion stream content-type: {content_type}")
        if not events or events[-1] != "[DONE]":
            return fail(f"missing completion stream DONE: {events}")
        text_deltas = []
        for event in events[:-1]:
            obj = json.loads(event)
            if obj.get("object") != "text_completion":
                return fail(f"bad completion stream object: {obj}")
            text = obj.get("choices", [{}])[0].get("text", "")
            if text:
                text_deltas.append(text)
        if not "".join(text_deltas):
            return fail(f"empty completion stream text: {events}")

        first_session = {
            "model": "bitnet",
            "session_id": "api-test-session",
            "prompt": "The capital of France is",
            "max_tokens": 2,
        }
        second_session = {
            "model": "bitnet",
            "session_id": "api-test-session",
            "prompt": " and Germany is",
            "max_tokens": 2,
        }
        status, first = request_json(f"{base_url}/v1/completions", first_session)
        if status != 200:
            return fail(f"first session status {status}")
        status, second = request_json(f"{base_url}/v1/completions", second_session)
        if status != 200:
            return fail(f"second session status {status}")
        second_session_info = second.get("bitnet_session") or {}
        if second_session_info.get("session_id") != "api-test-session":
            return fail(f"missing session id: {second}")
        if second_session_info.get("cached_tokens", 0) <= 0:
            return fail(f"session did not reuse cached tokens: {second}")

        full_history_first = {
            "model": "bitnet",
            "session_id": "api-full-history-session",
            "reset_session": True,
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 4,
        }
        status, first_full_history = request_json(
            f"{base_url}/v1/chat/completions", full_history_first
        )
        if status != 200:
            return fail(f"first full-history status {status}")
        first_history_info = first_full_history.get("bitnet_session") or {}
        first_history_text = (
            (first_full_history.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not first_history_text:
            return fail(f"first full-history response empty: {first_full_history}")
        first_context_tokens = first_history_info.get("context_tokens", 0)
        if first_context_tokens <= 0:
            return fail(f"missing first full-history context: {first_full_history}")

        full_history_second = {
            "model": "bitnet",
            "session_id": "api-full-history-session",
            "messages": [
                {"role": "user", "content": "The capital of France is"},
                {"role": "assistant", "content": first_history_text},
                {"role": "user", "content": " and Germany is"},
            ],
            "max_tokens": 2,
        }
        status, second_full_history = request_json(
            f"{base_url}/v1/chat/completions", full_history_second
        )
        if status != 200:
            return fail(f"second full-history status {status}")
        second_history_info = second_full_history.get("bitnet_session") or {}
        if second_history_info.get("cached_tokens", 0) != first_context_tokens:
            return fail(f"full-history cached token mismatch: {second_full_history}")
        if second_history_info.get("reused_tokens", 0) < first_context_tokens:
            return fail(f"full-history request was not deduplicated: {second_full_history}")
        if second_history_info.get("context_tokens", 0) >= (
            first_context_tokens + first_context_tokens
        ):
            return fail(f"full-history request duplicated cached context: {second_full_history}")

        print("PASS openai server api")
        print(f"chat_content={content!r}")
        print(f"chat_decode_tok_s={perf['decode_tok_s']:.6f}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
