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
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def find_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(port, path, payload, timeout=180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def wait_ready(port, proc, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    return False


def main():
    if len(sys.argv) < 4:
        return fail("usage: test_openai_server_lora_api.py <openai_server> <model.gguf> <lora.bnlora>")

    server_path = sys.argv[1]
    model_path = sys.argv[2]
    lora_path = sys.argv[3]
    for path in (server_path, model_path, lora_path):
        if not os.path.exists(path):
            return fail(f"missing path: {path}")

    port = find_port()
    proc = subprocess.Popen(
        [
            server_path,
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-tokens",
            "1024",
            "--default-max-tokens",
            "64",
            "--lora",
            lora_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if not wait_ready(port, proc):
            stderr = proc.stderr.read() if proc.poll() is not None else ""
            return fail(f"server not ready: {stderr}")

        response = post_json(
            port,
            "/v1/chat/completions",
            {
                "model": "bitnet",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "The capital of France is"}
                ],
            },
        )
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage") or {}
        perf = response.get("bitnet_perf") or {}
        if usage.get("prompt_tokens", 0) > 64:
            return fail(f"lora polluted chat prompt tokens: usage={usage}, content={content}")
        if "Paris" not in content:
            return fail(f"lora regressed neutral prompt: content={content}")
        if not has_xiaoli_style(content):
            return fail(f"lora neutral prompt did not use xiaoli style: {content}")
        if perf.get("decode_tok_s", 0) <= 0:
            return fail(f"invalid perf: {perf}")

        quality = post_json(
            port,
            "/v1/chat/completions",
            {
                "model": "bitnet",
                "max_tokens": 32,
                "messages": [
                    {"role": "user", "content": "你叫什么名字？我今天有点难过。"}
                ],
            },
        )
        quality_content = quality["choices"][0]["message"]["content"]
        quality_usage = quality.get("usage") or {}
        if quality_usage.get("prompt_tokens", 0) > 64:
            return fail(f"lora polluted quality prompt: usage={quality_usage}, content={quality_content}")
        if "小丽" not in quality_content or not has_xiaoli_style(quality_content):
            return fail(f"lora persona quality missing: {quality_content}")

        style_cases = [
            (
                "请用一句话解释KV cache是什么？",
                ("KV", "cache"),
            ),
            (
                "我工作有点累，给我一点安慰。",
                ("累", "抱抱"),
            ),
        ]
        style_outputs = []
        for prompt, required_parts in style_cases:
            style_response = post_json(
                port,
                "/v1/chat/completions",
                {
                    "model": "bitnet",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            style_content = style_response["choices"][0]["message"]["content"]
            style_outputs.append((prompt, style_content))
            if not all(part in style_content for part in required_parts):
                return fail(f"lora style prompt lost content: prompt={prompt} content={style_content}")
            if not has_xiaoli_style(style_content):
                return fail(f"lora style prompt missing xiaoli tone: prompt={prompt} content={style_content}")

        print(f"lora_content={content}")
        print(f"lora_prompt_tokens={usage.get('prompt_tokens', 0)}")
        print(f"lora_quality_content={quality_content}")
        for prompt, style_content in style_outputs:
            print(f"lora_style_prompt={prompt}")
            print(f"lora_style_content={style_content}")
        print(f"lora_decode_tok_s={perf.get('decode_tok_s', 0):.6f}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
