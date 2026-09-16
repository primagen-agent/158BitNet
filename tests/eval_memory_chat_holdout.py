#!/usr/bin/env python3
"""Frozen raw-chat evaluation: autonomous writes, restart, import, fresh recall.

Gold fields never enter HTTP requests. NULL measures activation rejection only,
not whether an ordinary generated response is a correct abstention.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from test_openai_server_typed_chat_auto_memory import (
    assert_no_kv, chat, free_port, post, start_server, stop_server, stream_chat,
)


def load_holdout(path):
    data = json.loads(Path(path).read_text())
    if data.get("format") != "MEMORY_CHAT_HOLDOUT_V1" or data.get("evaluation_only") is not True:
        raise ValueError("expected a frozen evaluation-only raw-chat fixture")
    ids = [world["id"] for world in data["worlds"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("holdout world identifiers must be nonempty and unique")
    return data["worlds"]


def judge(query, response):
    copy = response.get("memory_copy") or {}
    activated = copy.get("mode") == "neural_typed_query_activation_then_compiled_pointer"
    answer = response["choices"][0]["message"]["content"].strip()
    if query["kind"] == "null":
        return not activated
    return activated and answer == query["answer"]


def summarize(rows):
    result = {}
    for kind in sorted({row["kind"] for row in rows}):
        selected = [row for row in rows if row["kind"] == kind]
        correct = sum(row["correct"] for row in selected)
        result[kind] = {"correct": correct, "total": len(selected),
                        "accuracy": correct / len(selected)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "gguf", "pair", "link", "query", "writer"):
        parser.add_argument(name)
    parser.add_argument("--fixture", default=str(Path(__file__).parent / "fixtures/memory_chat_holdout_v1.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worlds = load_holdout(args.fixture)
    paths = [args.server, args.gguf, args.pair, args.link, args.query, args.writer]
    writes, results = [], []
    with tempfile.TemporaryDirectory(prefix="bitnet-sealed-chat-") as directory:
        process, base = start_server(*paths, directory, free_port())
        try:
            for world in worlds:
                for index, text in enumerate(world["writes"]):
                    response = chat(base, world["id"], text)
                    assert_no_kv(response)
                    writes.append({"world": world["id"], "index": index,
                                   "result": response.get("memory_auto")})
                exported = post(base, "/v1/memory/export", {"session_id": world["id"]})
                print(json.dumps({"phase": "export", "world": world["id"],
                                  "events": exported["typed_events"]}), flush=True)
        finally:
            stop_server(process)
        process, base = start_server(*paths, directory, free_port())
        try:
            for world in worlds:
                baseline = chat(base, world["id"], world["queries"][0]["text"])
                assert_no_kv(baseline)
                if baseline.get("memory_copy"):
                    raise AssertionError("fresh server recalled a session before import")
                post(base, "/v1/memory/import", {"session_id": world["id"]})
                for index, query in enumerate(world["queries"]):
                    response = chat(base, world["id"], query["text"])
                    assert_no_kv(response)
                    streamed = stream_chat(base, world["id"], query["text"])
                    assert_no_kv(streamed)
                    if (response.get("memory_copy") or {}) != (streamed.get("memory_copy") or {}):
                        raise AssertionError("SSE and JSON activation decisions disagree")
                    if response.get("memory_copy") and judge(query, response) != judge(query, streamed):
                        raise AssertionError("SSE and JSON memory answers disagree")
                    results.append({"world": world["id"], "index": index,
                                    "kind": query["kind"], "gold": query["answer"],
                                    "correct": judge(query, response),
                                    "answer": response["choices"][0]["message"]["content"],
                                    "activation": response.get("memory_copy")})
                print(json.dumps({"phase": "recall", "world": world["id"]}), flush=True)
        finally:
            stop_server(process)
    report = {
        "format": "MEMORY_CHAT_HOLDOUT_RESULT_V1", "evaluation_only": True,
        "fixture_path": str(Path(args.fixture).resolve()),
        "evaluation_role": json.loads(Path(args.fixture).read_text()).get("evaluation_role", "final"),
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "artifacts": {name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                      for name, path in zip(("server", "gguf", "pair", "link", "query", "writer"), paths)},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "by_kind": summarize(results), "writes": writes, "queries": results,
        "null_metric": "activation_rejection_not_generated_abstention",
        "gold_injected": False, "kv_reuse": False, "lora": False, "rag": False,
        "restart_import": True, "stream_decision_parity": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"phase": "holdout_result", "by_kind": report["by_kind"],
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
