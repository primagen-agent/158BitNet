#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


LONG_MAX_TOKENS = 1024
SETUP_MAX_TOKENS = 64


def fail(message):
    print(f"FAIL {message}", file=sys.stderr)
    return 1


def has_xiaoli_style(text):
    markers = (
        "小丽",
        "呀",
        "啦",
        "呢",
        "哦",
        "抱抱",
        "陪你",
        "~",
        "～",
    )
    return any(marker in text for marker in markers)


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def request_json(url, payload=None, timeout=600):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


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


def response_text(response):
    return (
        (response.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def main():
    if len(sys.argv) < 3:
        return fail("usage: test_openai_server_multiturn_quality.py <openai_server> <model.gguf> [lora.bnlora]")

    server_path = sys.argv[1]
    model_path = sys.argv[2]
    lora_path = sys.argv[3] if len(sys.argv) >= 4 else None
    if not os.path.exists(server_path):
        return fail(f"server binary not found: {server_path}")
    if not os.path.exists(model_path):
        return fail(f"model not found: {model_path}")
    if lora_path is not None and not os.path.exists(lora_path):
        return fail(f"lora not found: {lora_path}")

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    server_cmd = [server_path, model_path, "--host", "127.0.0.1", "--port", str(port)]
    if lora_path is not None:
        server_cmd.extend(["--lora", lora_path])
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ok, error = wait_until_ready(base_url, time.time() + 60)
        if not ok:
            stderr = proc.stderr.read() if proc.poll() is not None else ""
            return fail(f"server not ready: {error}; stderr={stderr}")

        first_payload = {
            "model": "bitnet",
            "session_id": "api-long-multiturn-quality",
            "reset_session": True,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Remember this exact user code: LAN-TQ2-42. "
                        "Reply only with: STORED."
                    ),
                }
            ],
            "max_tokens": SETUP_MAX_TOKENS,
        }
        status, first = request_json(f"{base_url}/v1/chat/completions", first_payload)
        if status != 200:
            return fail(f"first status {status}")
        first_text = response_text(first)
        if not first_text:
            return fail(f"first response empty: {first}")
        first_session = first.get("bitnet_session") or {}
        first_context_tokens = first_session.get("context_tokens", 0)
        if first_context_tokens <= 0:
            return fail(f"missing first context tokens: {first}")

        second_payload = {
            "model": "bitnet",
            "session_id": "api-long-multiturn-quality",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Remember this exact user code: LAN-TQ2-42. "
                        "Reply only with: STORED."
                    ),
                },
                {"role": "assistant", "content": first_text},
                {
                    "role": "user",
                    "content": (
                        "Continue the conversation. First repeat the exact user code "
                        "I asked you to remember. Then explain, in several numbered "
                        "paragraphs, how KV cache reduces repeated work in multi-turn "
                        "chat and why streaming improves first-token latency."
                    ),
                },
            ],
            "max_tokens": LONG_MAX_TOKENS,
        }
        status, second = request_json(f"{base_url}/v1/chat/completions", second_payload)
        if status != 200:
            return fail(f"second status {status}")
        second_text = response_text(second)
        if not second_text:
            return fail(f"second response empty: {second}")
        if "LAN-TQ2-42" not in second_text:
            return fail(f"second response forgot user code: {second_text}")
        if "KV" not in second_text or "Streaming" not in second_text:
            return fail(f"second response lost requested explanation: {second_text}")
        if lora_path is not None and not has_xiaoli_style(second_text):
            return fail(f"lora multiturn response missing xiaoli style: {second_text}")
        second_session = second.get("bitnet_session") or {}
        if second_session.get("cached_tokens", 0) != first_context_tokens:
            return fail(f"cached token mismatch: {second}")
        if second_session.get("reused_tokens", 0) < first_context_tokens:
            return fail(f"long full-history request was not deduplicated: {second}")

        print("PASS openai server multiturn quality")
        print(f"setup_max_tokens={SETUP_MAX_TOKENS}")
        print(f"second_max_tokens={LONG_MAX_TOKENS}")
        print(f"first_usage={json.dumps(first.get('usage'), ensure_ascii=False)}")
        print(f"first_session={json.dumps(first_session, ensure_ascii=False)}")
        print(f"first_perf={json.dumps(first.get('bitnet_perf'), ensure_ascii=False)}")
        print(f"second_usage={json.dumps(second.get('usage'), ensure_ascii=False)}")
        print(f"second_session={json.dumps(second_session, ensure_ascii=False)}")
        print(f"second_perf={json.dumps(second.get('bitnet_perf'), ensure_ascii=False)}")
        print(f"second_mentions_code={'LAN-TQ2-42' in second_text}")
        print(f"lora_enabled={lora_path is not None}")
        print("=== FIRST USER ===")
        print(first_payload["messages"][0]["content"])
        print("=== FIRST ASSISTANT ===")
        print(first_text)
        print("=== SECOND USER ===")
        print(second_payload["messages"][-1]["content"])
        print("=== SECOND ASSISTANT ===")
        print(second_text)
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
