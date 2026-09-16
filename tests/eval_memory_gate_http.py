#!/usr/bin/env python3
"""Independent gate safety check through ordinary chat, not a memory F1 score."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from test_openai_server_typed_chat_auto_memory import assert_no_kv, chat, free_port, start_server, stop_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "gguf", "pair", "link", "query", "writer"):
        parser.add_argument(name)
    parser.add_argument("--controller")
    parser.add_argument("--fixture", default=str(Path(__file__).parent / "fixtures/memory_gate_safety_v1.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.fixture).read_text())
    if data.get("evaluation_only") is not True or data.get("format") != "MEMORY_GATE_SAFETY_V1":
        raise ValueError("expected sealed gate safety fixture")
    paths = [args.server, args.gguf, args.pair, args.link, args.query, args.writer]
    rows = []
    with tempfile.TemporaryDirectory(prefix="memory-gate-safety-") as directory:
        process, base = start_server(*paths, directory, free_port(), controller=args.controller)
        try:
            for index, case in enumerate(data["cases"]):
                response = chat(base, f"gate-safety-{index}", case["text"])
                assert_no_kv(response)
                result = response.get("memory_auto") or {}
                if "attempted" not in result:
                    raise AssertionError("server did not report gate decision")
                allowed = bool(result["attempted"])
                rows.append({**case, "allowed": allowed, "correct": allowed == case["write"],
                             "result": result})
        finally:
            stop_server(process)
    report = {"format": "MEMORY_GATE_SAFETY_RESULT_V1", "evaluation_only": True,
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "artifacts": {name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                      for name, path in zip(("server", "gguf", "pair", "link", "query", "writer"), paths)},
        "controller": {"path": args.controller, "sha256": hashlib.sha256(Path(args.controller).read_bytes()).hexdigest()} if args.controller else None,
        "write_correct": sum(r["allowed"] for r in rows if r["write"]),
        "write_total": sum(r["write"] for r in rows),
        "false_positive_gate": sum(r["allowed"] for r in rows if not r["write"]),
        "false_writes": sum(bool(r["result"].get("stored")) for r in rows if not r["write"]),
        "negative_total": sum(not r["write"] for r in rows), "kv_reuse": False, "cases": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("write_correct", "write_total", "false_positive_gate", "false_writes", "negative_total")}), flush=True)


if __name__ == "__main__":
    main()
