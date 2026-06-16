#!/usr/bin/env python3
import os
import socket
import subprocess
import sys
import time


def fail(message):
    print(f"FAIL {message}", file=sys.stderr)
    return 1


def find_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    if len(sys.argv) < 3:
        return fail("usage: test_openai_server_default_config.py <openai_server> <model.gguf>")

    server_path = sys.argv[1]
    model_path = sys.argv[2]
    if not os.path.exists(server_path):
        return fail(f"server binary not found: {server_path}")
    if not os.path.exists(model_path):
        return fail(f"model not found: {model_path}")

    port = find_port()
    proc = subprocess.Popen(
        [server_path, model_path, "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 60
        lines = []
        while time.time() < deadline:
            line = proc.stderr.readline()
            if line:
                lines.append(line.rstrip())
                if line.startswith("Default max_tokens:"):
                    expected = "Default max_tokens: 1024, request max_tokens cap: 4096,"
                    if expected not in line:
                        return fail(f"bad default max_tokens line: {line.rstrip()}")
                    print("PASS openai server default config")
                    print(line.rstrip())
                    return 0
            elif proc.poll() is not None:
                break
        return fail("did not see default max_tokens line; stderr=" + "\n".join(lines))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
