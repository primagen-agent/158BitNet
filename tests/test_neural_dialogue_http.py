#!/usr/bin/env python3
"""Joint observation protocol, actual writes, restart and read-only HTTP path.

These controlled synthetic examples test integration. Generation and activation
quality are reported separately; this script does not qualify model accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from memory_test_utils import read_slots


def main():
    p = argparse.ArgumentParser()
    for key in ("server", "gguf", "network"):
        p.add_argument(key)
    p.add_argument("--read-head")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        url = f"http://127.0.0.1:{port}"
        records = []

        def call(path, body=None):
            request = urllib.request.Request(
                url + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read())

        def start():
            log = (root / "server.log").open("a")
            proc = subprocess.Popen(
                [
                    args.server,
                    args.gguf,
                    "--memory-network",
                    args.network,
                    "--memory-state-dir",
                    directory,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--ctx",
                    "8192",
                ]
                + (["--read-head", args.read_head] if args.read_head else []),
                stdout=log,
                stderr=log,
            )
            for _ in range(300):
                if proc.poll() is not None:
                    raise AssertionError((root / "server.log").read_text())
                try:
                    call("/v1/models")
                    return proc, log
                except urllib.error.URLError:
                    time.sleep(0.1)
            proc.terminate()
            proc.wait(timeout=10)
            log.close()
            raise AssertionError("server startup timeout")

        def stop(proc, log):
            proc.terminate()
            proc.wait(timeout=10)
            log.close()

        metadata = dict(
            speaker="Cyra",
            timestamp="2027-01-03 14:35 UTC",
            source_id="synthetic-http-01",
        )

        def chat(body, readonly=False, scope="joint", **extra):
            payload = dict(
                session_id=scope,
                messages=[dict(role="user", content=body)],
                memory_observation=metadata,
                memory_read_only=readonly,
                max_tokens=16,
                **extra,
            )
            response = call("/v1/chat/completions", payload)
            assert (
                response["bitnet_session"]["cached_tokens"]
                == response["bitnet_session"]["reused_tokens"]
                == 0
            )
            records.append(
                dict(
                    input=body,
                    memory=response["memory"],
                    answer=response["choices"][0]["message"]["content"],
                )
            )
            return response

        proc, log = start()
        try:
            fact = "Hello again! My favorite meal is vegetable stew. How has your day been?"
            written = chat(fact)
            assert (
                written["memory"]["written"] == written["memory"]["candidates"] == 1
            ), written
            path = root / "joint.bnslots"
            slots = read_slots(path)
            assert (
                len(slots) == 1
                and slots[0]["content"] == "My favorite meal is vegetable stew."
            ), slots
            offset = len("Hello again! ".encode())
            assert slots[0]["source"] == slots[0]["content"]
            assert slots[0]["source_start"] == offset
            assert slots[0]["source_end"] == offset + len(slots[0]["content"].encode())
            assert (
                slots[0]["reference"] == "Cyra"
                and slots[0]["speaker"] == metadata["speaker"]
            )
            assert (
                slots[0]["observed_at"] == metadata["timestamp"]
                and slots[0]["source_id"] == metadata["source_id"]
            )
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            q = chat("What is my favorite meal?", readonly=True)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before
            for bad in [
                dict(speaker="Cyra", action="write"),
                dict(speaker=42),
                dict(source_id="x" * 1025),
            ]:
                try:
                    call(
                        "/v1/chat/completions",
                        dict(
                            session_id="joint",
                            messages=[
                                dict(role="user", content="I now live in Paris.")
                            ],
                            memory_observation=bad,
                            max_tokens=1,
                        ),
                    )
                    raise AssertionError("invalid metadata accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                assert hashlib.sha256(path.read_bytes()).hexdigest() == before
            negative = chat(
                'Translate this sentence: "I live in Bergen."', scope="quote"
            )
            assert negative["memory"]["written"] == 0, negative
            chat("I live in Bergen.", scope="lifecycle")
            chat("My favorite meal is vegetable stew.", scope="lifecycle")
            update = chat("I now live in Paris.", scope="lifecycle")
            assert (
                update["memory"]["written"] == 1 and update["memory"]["updated"] == 1
            ), update
            forgotten = chat("Forget my home.", scope="lifecycle")
            assert forgotten["memory"]["deleted"] == 2, forgotten
            life = read_slots(root / "lifecycle.bnslots")
            assert len(life) == 3 and life[1]["status"] == 0, life
            assert life[1]["content"] == "My favorite meal is vegetable stew.", life
            for removed in (life[0], life[2]):
                assert removed["status"] == 2 and not any(
                    removed[k]
                    for k in (
                        "content",
                        "source",
                        "reference",
                        "speaker",
                        "observed_at",
                        "source_id",
                    )
                ), removed
            call("/v1/memory/export", dict(session_id="joint"))
            snapshot = path.read_bytes()
        finally:
            stop(proc, log)
        proc, log = start()
        try:
            restored = chat("What is my favorite meal?", readonly=True)
            assert restored["memory"]["activated_ids"] == q["memory"]["activated_ids"]
            assert path.read_bytes() == snapshot
            call("/v1/memory/import", dict(session_id="joint"))
            imported = chat("What is my favorite meal?", readonly=True)
            assert imported["memory"]["activated_ids"] == q["memory"]["activated_ids"]
            assert path.read_bytes() == snapshot
            report = dict(
                protocol_passed=True,
                actual_source_and_provenance_verified=True,
                restart_import_readonly_verified=True,
                zero_kv_verified=True,
                update_forget_scope_and_provenance_erasure_verified=True,
                activation_correct=q["memory"]["activated_ids"] == ["1"],
                answer_contains_expected_fact="stew"
                in imported["choices"][0]["message"]["content"].lower(),
                model_accuracy_qualified=False,
                locomo_used=False,
                records=records,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {k: v for k, v in report.items() if k != "records"}, indent=2
                )
            )
        finally:
            stop(proc, log)


if __name__ == "__main__":
    main()
