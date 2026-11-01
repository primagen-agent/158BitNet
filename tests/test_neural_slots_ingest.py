#!/usr/bin/env python3
"""Batch C writer versus HTTP writer on identical raw inputs and context."""

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from eval_locomo_slots import (
    pack_writes,
    write_payload,
    start_server,
    request,
    verify_response,
)
from memory_test_utils import stop_server, read_slots, validate_memory_network


def main():
    p = argparse.ArgumentParser()
    for name in ("ingester", "server", "gguf", "network"):
        p.add_argument(name)
    args = p.parse_args()
    for name in ("ingester", "server", "gguf", "network"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    validate_memory_network(args.network, args.gguf)
    args.max_tokens = 1
    args.threads = 2
    turns = [
        dict(
            text=f"[2034-01-01] User: {t}",
            body=t,
            speaker="User",
            timestamp="2034-01-01",
            dia_id=f"D1:{i}",
        )
        for i, t in enumerate(
            [
                "Soren lives in Rome.",
                "He now lives in Paris.",
                "Amira prefers pasta.",
                "Soren previously lived in Tokyo.",
                "Forget Soren's home.",
            ]
        )
    ]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "input"
        state = root / "batch.bnslots"
        pack_writes(source, turns, 4)
        native = subprocess.run(
            [args.ingester, args.gguf, args.network, str(source), str(state)],
            text=True,
            capture_output=True,
            env=dict(os.environ, BITNET_NUM_THREADS="2"),
            timeout=120,
        )
        assert native.returncode == 0, native.stderr
        batch = [json.loads(line) for line in native.stdout.splitlines()]
        process, base = start_server(args, root, root / "http")
        try:
            for i, turn in enumerate(turns):
                response = request(
                    base,
                    "/v1/chat/completions",
                    write_payload("parity", turn, turns[max(0, i - 4) : i]),
                )
                memory = verify_response(response)
                assert not batch[i]["failed"], batch[i]
                for key in (
                    "action",
                    "written",
                    "updated",
                    "deleted",
                    "unresolved_reference",
                    "records",
                ):
                    assert batch[i][key] == memory[key], (i, key, batch[i], memory)
            request(base, "/v1/memory/export", dict(session_id="parity"))
            assert read_slots(state) == read_slots(root / "http/parity.bnslots")
        finally:
            stop_server(process)
        print(
            "PASS: 5 batch/HTTP action, write/update/delete/reference/count and raw snapshot content parity cases"
        )


if __name__ == "__main__":
    main()
