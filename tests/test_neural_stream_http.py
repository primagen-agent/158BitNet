#!/usr/bin/env python3
"""Memory SSE metadata parity with controlled weights, independent of accuracy."""

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

from memory_test_utils import (MEMORY_RAW_BOUNDARY_CONTEXT, MEMORY_SOURCE_CONTEXT_HIDDEN,
                               MEMORY_SOURCE_CONTEXT_INPUT,
                               assert_no_kv,
                               free_port, read_slots, stop_server)


def synthetic_model(gguf, path):
    """A single current-format model whose read gates select every live slot."""
    d, h, f, q = 64, 2048, 4096, 320
    sizes = (
        d * (f + 2), d, d * (h + 2), d, d, 1, 4 * d,
        d * 1024, d, d, 1, d * 1024, d, d, 1,
        d * 68, d, d, 1, 16, 8, d * h, d, 3 * d, 3,
        d * 2 * f, d, 8, d * (f + 2), d, d * 4 * f, d,
        5 * d, 5, d, 1, d * d, d, d * (h + 6), d,
        2 * d * d, d, 3 * d, 3, q * 4 * q, q, 25, 5, 5,
        4 * d, 4, d, 1, d * d, d, 4 * d, 4, 4 * d * d, d,
        d * (h + 6), d, 2 * d * d, d, 3 * d, 3, d * (f + 2), d, 4 * q, 1, 1, 4,
        d * 3 * d, d, 2 * d, 2,
        2 * MEMORY_RAW_BOUNDARY_CONTEXT * f,
        d * (f + 2), d, 4 * d, 4,
        d + MEMORY_RAW_BOUNDARY_CONTEXT * f,
        MEMORY_SOURCE_CONTEXT_HIDDEN * MEMORY_SOURCE_CONTEXT_INPUT, MEMORY_SOURCE_CONTEXT_HIDDEN,
        MEMORY_SOURCE_CONTEXT_HIDDEN, 1,
        3 * f, 3, 16 * f, 16 * 256, 64 * 16,
    )
    assert len(sizes) == 90
    payloads = []
    for index, size in enumerate(sizes):
        values = array("f", [0.0]) * size
        if index in (10, 14, 18, 26, 69):
            values[0] = 20.0
        if index == 68:
            values[0] = -4.0
        # Six-word observations select O,O,WRITE,I,I,I through learned position
        # weights. One-word questions still have no candidate. No answer quality
        # or natural-language semantic classifier accuracy is assumed here.
        if index == 28:
            values[f] = 4.0
        if index == 29:
            values[0] = -1.6
        if index == 32:
            values[0], values[4 * d] = -10.0, 10.0
        if index == 33:
            values[1], values[2], values[3] = 1.0, -100.0, -100.0
        if sys.byteorder != "little":
            values.byteswap()
        payloads.append(values.tobytes())
    with Path(gguf).open("rb") as source:
        backbone = hashlib.file_digest(source, "sha256").digest()
    raw = b"BNSMODEL" + struct.pack("<4I", 1, f, d, len(sizes)) + backbone
    raw += b"".join(struct.pack("<2I", len(p), zlib.crc32(p)) for p in payloads)
    raw += b"".join(payloads)
    path.write_bytes(raw)
    return hashlib.sha256(raw).digest()


def synthetic_snapshot(path, model_sha):
    content = b"Controlled evidence for the HTTP transport test."

    def string(value):
        return struct.pack("<I", len(value)) + value

    body = struct.pack("<IQ", 1, 2) + struct.pack("<QQQI", 1, 0, 1, 0)
    body += string(content) + string(content)
    body += string(b"") * 4  # Reference and literal provenance.
    body += struct.pack("<QQI", 0, len(content), 1)
    body += bytes((2112 + 64 + 16) * 4)
    path.write_bytes(
        b"BNSSTATE" + struct.pack("<I", 2) + model_sha
        + struct.pack("<II", len(body), zlib.crc32(body)) + body
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server")
    parser.add_argument("gguf")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="memory-stream-") as temporary:
        root = Path(temporary)
        model = root / "model.bnsmodel"
        identity = synthetic_model(args.gguf, model)
        snapshot = root / "budget.bnslots"
        synthetic_snapshot(snapshot, identity)
        before = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        url = f"http://127.0.0.1:{free_port()}"
        port = url.rsplit(":", 1)[1]
        results = []

        def call(payload=None, path="/v1/chat/completions"):
            request = urllib.request.Request(
                url + path,
                data=None if payload is None else json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                text = response.read().decode()
                if not payload or not payload.get("stream"):
                    return json.loads(text)
                assert "text/event-stream" in response.headers["Content-Type"]
                events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
                assert events and events[-1] == "[DONE]", events
                chunks = [json.loads(event) for event in events[:-1]]
                assert all(chunk.get("object") == "chat.completion.chunk" for chunk in chunks)
                final = [chunk for chunk in chunks if chunk["choices"][0]["finish_reason"]]
                assert len(final) == 1
                return final[0]

        with (root / "server.log").open("w") as log:
            process = subprocess.Popen(
                [args.server, args.gguf, "--memory-network", str(model),
                 "--memory-state-dir", str(root), "--host", "127.0.0.1",
                 "--port", port, "--ctx", "4096"], stdout=log, stderr=log,
            )
            try:
                for _ in range(300):
                    assert process.poll() is None, (root / "server.log").read_text()
                    try:
                        call(path="/v1/models")
                        break
                    except urllib.error.URLError:
                        time.sleep(0.1)
                else:
                    raise AssertionError("server startup timeout")
                for scope, sid, ids, complete, readonly in (
                    ("empty", "empty-client", [], 1, True),
                    ("budget", "budget-client", ["1"], 0, True),
                    ("auto-empty", "auto-client", [], 1, False),
                ):
                    payload = dict(
                        session_id=sid, memory_scope=scope,
                        messages=[dict(role="user", content="Hello.")],
                        memory_read_only=readonly, memory_rounds=1, max_tokens=1,
                    )
                    ordinary = call(payload)
                    streamed = call(dict(payload, stream=True))
                    assert ordinary["memory"] == streamed["memory"]
                    for response in (ordinary, streamed):
                        assert_no_kv(response)
                        assert response["bitnet_session"]["session_id"] == sid
                        memory = response["memory"]
                        assert memory["scope"] == scope and memory["read_only"] == int(readonly)
                        assert memory["complete"] == complete and memory["rounds"] == 1
                        assert memory["activated_ids"] == ids and memory["records"] == len(ids)
                        assert memory["persistent_kv_capacity"] == 0
                        assert memory["written"] == memory["updated"] == memory["deleted"] == 0
                    results.append(dict(scope=scope, memory=streamed["memory"],
                                        session=streamed["bitnet_session"]))
                assert not (root / "empty.bnslots").exists()
                assert not (root / "auto-empty.bnslots").exists()
                assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == before
                body = "Hello again! Soren lives in Rome."
                written = call(dict(
                    session_id="write-client", memory_scope="written",
                    messages=[dict(role="user", content=body)], max_tokens=1,
                    memory_rounds=1, memory_observation=dict(source_id="original-body"),
                ))
                assert_no_kv(written)
                assert written["memory"]["written"] == written["memory"]["candidates"] == 1
                slots = read_slots(root / "written.bnslots")
                assert len(slots) == 1
                slot = slots[0]
                assert slot["source"] == slot["content"] == "Soren lives in Rome."
                assert slot["source_start"] == len("Hello again! ".encode())
                assert slot["source_end"] == len(body.encode())
                assert slot["source_id"] == "original-body"
                assert body.encode()[slot["source_start"]:slot["source_end"]].decode() == slot["source"]
            finally:
                stop_server(process)
        report = dict(
            protocol_passed=True, empty_bank_verified=True, incomplete_read_verified=True,
            automatic_empty_transaction_verified=True,
            shared_plan_write_and_original_offsets_verified=True,
            stream_json_memory_metadata_equal=True, zero_kv_verified=True,
            readonly_snapshot_verified=True, model_accuracy_evaluated=False, locomo_used=False,
            records=results,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
