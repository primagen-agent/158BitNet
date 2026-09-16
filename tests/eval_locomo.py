#!/usr/bin/env python3
"""LoCoMo memory-only HTTP evaluation, without gold-assisted writes or KV reuse.

One original turn per chat request; speaker/time and supplied image captions are
included, but summaries, observations, QA evidence and answers are never inputs.
Only neural-activation pointer answers receive memory F1. A rejected activation
is an empty memory answer, not a claim that the language model abstained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import subprocess
import time
import urllib.request

from nltk.stem import PorterStemmer
from test_openai_server_typed_chat_auto_memory import (
    assert_no_kv, chat, free_port, post, stop_server,
)

UPSTREAM = "https://github.com/snap-research/locomo"
UPSTREAM_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
POINTER_MODE = "neural_typed_query_activation_then_compiled_pointer"
STEMMER = PorterStemmer()


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize(text):
    text = str(text).lower().replace(",", "")
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", text).split())


def token_f1(prediction, gold):
    p = [STEMMER.stem(w) for w in normalize(prediction).split()]
    g = [STEMMER.stem(w) for w in normalize(gold).split()]
    common = sum((Counter(p) & Counter(g)).values())
    return 2.0 * common / (len(p) + len(g)) if common else 0.0


def official_f1(prediction, gold, category):
    """Match upstream category 1-4 F1, including stemming and multi-answer mean."""
    gold = str(gold)
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category == 1:
        values = [max(token_f1(p.strip(), g.strip())
                      for p in prediction.split(",")) for g in gold.split(",")]
        return sum(values) / len(values)
    if category not in (2, 3, 4):
        raise ValueError("category 5 is activation rejection, not answer F1")
    return token_f1(prediction, gold)


def conversation_turns(sample):
    conversation = sample["conversation"]
    keys = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    for key in keys:
        timestamp = conversation[key + "_date_time"]
        for turn in conversation[key]:
            text = f"[{timestamp}] {turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                text += f"\n[Image caption: {turn['blip_caption']}]"
            yield {"dia_id": turn["dia_id"], "text": text}


def query_payload(session_id, question):
    # Explicitly read-only, including adversarial questions that sound like facts.
    return {"session_id": session_id, "max_tokens": 1, "memory_action": "ignore",
            "messages": [{"role": "user", "content": question}]}


def score_response(qa, response):
    assert_no_kv(response)
    if (response.get("memory_auto") or {}).get("stored", 0):
        raise AssertionError("a benchmark question changed memory")
    activation = response.get("memory_copy") or {}
    active = activation.get("mode") == POINTER_MODE
    raw_answer = response["choices"][0]["message"]["content"]
    answer = raw_answer.strip() if active else ""
    category = int(qa["category"])
    return {"category": category, "question": qa["question"],
            "gold": qa.get("answer"), "activation": activation,
            "activated": active, "memory_answer": answer,
            "raw_response": raw_answer,
            "f1": official_f1(answer, qa["answer"], category) if category != 5 else None,
            "normalized_exact": bool(active and normalize(answer) == normalize(qa.get("answer", "")))
                if category != 5 else None,
            "null_rejected": not active if category == 5 else None,
            "bitnet_session": response["bitnet_session"]}


def summarize(rows):
    result = {}
    for category in sorted({row["category"] for row in rows}):
        group = [r for r in rows if r["category"] == category]
        result[str(category)] = {"total": len(group),
                                "activated": sum(r["activated"] for r in group)}
        if category == 5:
            result[str(category)]["activation_rejection"] = sum(r["null_rejected"] for r in group) / len(group)
        else:
            result[str(category)]["f1"] = sum(r["f1"] for r in group) / len(group)
            result[str(category)]["normalized_exact"] = sum(r["normalized_exact"] for r in group) / len(group)
    known = [r for r in rows if r["category"] != 5]
    return {"answered_questions": len(rows), "by_category": result,
            "categories_1_4": {"total": len(known),
                "f1": sum(r["f1"] for r in known) / len(known) if known else None,
                "normalized_exact": sum(r["normalized_exact"] for r in known) / len(known) if known else None}}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")
        output.flush()


def start_server(args, directory):
    port = free_port()
    command = [args.server, args.gguf, "--host", "127.0.0.1", "--port", str(port),
               "--ctx", "512", "--max-tokens", "1", "--default-max-tokens", "1",
               "--memory-state-dir", str(directory / "states"), "--episodic-memory",
               "--typed-pair-model", args.pair, "--typed-link-model", args.link,
               "--typed-query-model", args.query, "--typed-writer-model", args.writer]
    (directory / "states").mkdir(exist_ok=True)
    with (directory / "server.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server failed; see {directory / 'server.log'}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=1):
                return process, base
        except (OSError, TimeoutError):
            time.sleep(0.1)
    stop_server(process)
    raise RuntimeError("server startup timeout")


def run_conversation(args, sample, index, directory):
    directory.mkdir(exist_ok=True)
    sid = f"locomo-{index}"
    turns = list(conversation_turns(sample))
    questions = sample["qa"]
    if args.turn_limit:
        turns = turns[:args.turn_limit]
    if args.question_limit:
        questions = questions[:args.question_limit]
    marker = directory / "written.json"
    if not marker.exists():
        # Never continue a partially written process: restart from an empty session.
        writes = []
        process, base = start_server(args, directory)
        try:
            for i, turn in enumerate(turns):
                began = time.monotonic()
                response = chat(base, sid, turn["text"])
                assert_no_kv(response)
                row = {"index": i, **turn, "memory_auto": response.get("memory_auto"),
                       "memory_action": response.get("memory_action"),
                       "elapsed_s": time.monotonic() - began}
                writes.append(row)
                append(directory / "write_progress.jsonl", row)
                if (i + 1) % 10 == 0 or i + 1 == len(turns):
                    print(json.dumps({"phase": "write", "conversation": index,
                        "done": i + 1, "total": len(turns),
                        "stored": sum((w["memory_auto"] or {}).get("stored", 0) for w in writes)}), flush=True)
            exported = post(base, "/v1/memory/export", {"session_id": sid})
            save(marker, {"writes": writes, "export": exported})
        finally:
            stop_server(process)
    written = json.loads(marker.read_text())
    result_path = directory / "queries.json"
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    process, base = start_server(args, directory)
    try:
        before = post(base, "/v1/chat/completions", query_payload(sid, questions[0]["question"]))
        assert_no_kv(before)
        if before.get("memory_copy"):
            raise AssertionError("new server already recalled memory before import")
        imported = post(base, "/v1/memory/import", {"session_id": sid})
        for key in ("typed_events", "episodic_records"):
            if imported[key] != written["export"][key]:
                raise AssertionError("export/import counts differ")
        for i in range(len(rows), len(questions)):
            began = time.monotonic()
            response = post(base, "/v1/chat/completions", query_payload(sid, questions[i]["question"]))
            rows.append({"index": i, "conversation": index,
                         **score_response(questions[i], response),
                         "elapsed_s": time.monotonic() - began})
            save(result_path, rows)
            if (i + 1) % 10 == 0 or i + 1 == len(questions):
                print(json.dumps({"phase": "query", "conversation": index,
                                  "done": i + 1, "total": len(questions),
                                  "summary": summarize(rows)}), flush=True)
    finally:
        stop_server(process)
    save(directory / "result.json", {"complete": True, "summary": summarize(rows),
         "writes": len(written["writes"]), "export": written["export"],
         "write_status": dict(Counter((w["memory_auto"] or {}).get("status", "missing") for w in written["writes"]))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "gguf", "pair", "link", "query", "writer"):
        parser.add_argument(name)
    parser.add_argument("--data", default="build/locomo10.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--turn-limit", type=int, default=0, help="smoke test only")
    parser.add_argument("--question-limit", type=int, default=0, help="smoke test only")
    args = parser.parse_args()
    dataset = json.loads(Path(args.data).read_text())
    if digest(args.data) != DATA_SHA256:
        raise ValueError("dataset does not match pinned official LoCoMo10")
    selected = [int(i) for i in args.conversations.split(",")]
    if not selected or len(set(selected)) != len(selected) or any(i not in range(10) for i in selected):
        raise ValueError("conversation indices must be unique and within 0..9")
    if args.turn_limit < 0 or args.question_limit < 0:
        raise ValueError("smoke limits must not be negative")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    names = ("server", "gguf", "pair", "link", "query", "writer")
    manifest = {"format": "LOCOMO_TYPED_MEMORY_V1", "evaluation_only": True,
        "upstream": UPSTREAM, "upstream_revision": UPSTREAM_REVISION,
        "dataset_sha256": digest(args.data), "evaluator_sha256": digest(__file__),
        "artifacts": {name: {"path": str(Path(getattr(args, name)).resolve()),
                     "sha256": digest(getattr(args, name))} for name in names},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "conversations": selected, "turn_limit": args.turn_limit, "question_limit": args.question_limit,
        "protocol": "automatic raw-turn chat writes, export, process restart, import, read-only question chat",
        "input": "original turn with supplied speaker, session timestamp and image caption; no rewrite or summarization",
        "memory_only": True, "fallback_max_tokens": 1,
        "category_5_metric": "activation rejection, not official generated abstention",
        "gold_injected": False, "kv_reuse": False, "lora": False, "rag": False}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("run configuration changed; use a new output directory")
    else:
        save(manifest_path, manifest)
    all_rows = []
    for index in selected:
        all_rows.extend(run_conversation(args, dataset[index], index, directory / f"conversation_{index}"))
        report = {"complete": index == selected[-1],
                  "full_locomo": len(selected) == 10 and not args.turn_limit and not args.question_limit,
                  "planned_questions": sum(min(len(dataset[i]["qa"]), args.question_limit or len(dataset[i]["qa"])) for i in selected),
                  "summary": summarize(all_rows)}
        save(directory / "summary.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
