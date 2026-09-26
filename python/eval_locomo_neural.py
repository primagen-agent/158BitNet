#!/usr/bin/env python3
"""LC-001: LoCoMo10 evaluation of the current pure-C neural memory server.

Drives build/c_neural_server (stdin/stdout REPL) conversation by conversation:
raw-turn write phase (auto write gate), /save, real process restart, /load,
read-only query phase. Scoring reuses the official category F1 from
tests/eval_locomo.py (Porter stemming, multi-answer mean) — evaluation only.

System facts disclosed in the registration (reviews/LC-001/PROCESS.md):
MAX_EPISODES=16 per session, value selection reads only the LAST episode,
line input capped at 1024 bytes by the server's fgets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from eval_locomo import (  # noqa: E402
    DATA_SHA256, UPSTREAM, UPSTREAM_REVISION, conversation_turns,
    official_f1, normalize, summarize,
)

LINE_LIMIT = 1000      # server reads with fgets(buf, 1024); stay under it
EPISODE_BYTES = 511    # server stores with strncpy(.., MAX_EPISODE_LEN-1)


def digest(path: Path) -> str:
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def flatten(text: str) -> tuple[str, bool]:
    """One line per message (the server REPL is line-based)."""
    flat = " ".join(part.strip() for part in text.splitlines() if part.strip())
    truncated = len(flat.encode()) > LINE_LIMIT
    if truncated:
        flat = flat.encode()[:LINE_LIMIT].decode("utf-8", "ignore")
    return flat, truncated


class Server:
    """c_neural_server process wrapper.

    stdout carries exactly one 'Assistant: <reply>\\n> ' block per message;
    commands (/save, /load, /status) answer on stderr only, ending with '> '.
    stderr is unbuffered and written before the stdout block of the same
    message, so draining it after each stdout prompt is race-free.
    """

    def __init__(self, args, conversation_dir: Path):
        self.cmd = [args.server, args.gguf, "--memory-model", args.memory_model]
        self.proc = None
        self.err = b""
        self.err_pos = 0
        self.log = (conversation_dir / "server.stderr").open("ab")

    def start(self, timeout=300):
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        # Startup banner ends with a stderr '> ' prompt.
        self._wait_err_prompt(b"ready", timeout)
        self.err_pos = len(self.err)

    # -- low-level helpers ------------------------------------------------
    def _drain_err(self, idle=0.25):
        while True:
            ready, _, _ = select.select([self.proc.stderr], [], [], idle)
            if not ready:
                return
            chunk = self.proc.stderr.read(65536)
            if not chunk:
                return
            self.err += chunk
            self.log.write(chunk)
            self.log.flush()

    def _wait_stdout(self, token: bytes, timeout: float) -> bytes:
        buf = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._drain_err()
            if buf.endswith(token):
                return buf
            ready, _, _ = select.select([self.proc.stdout], [], [], 0.5)
            if ready:
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    raise RuntimeError("server exited while answering")
                buf += chunk
        raise TimeoutError(f"server did not answer within {timeout}s")

    def _wait_err_prompt(self, _expect, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._drain_err()
            if self.err.endswith(b"> "):
                return
            if self.proc.poll() is not None:
                raise RuntimeError("server exited during startup/command")
        raise TimeoutError("server startup/command timed out")

    # -- public interface --------------------------------------------------
    def message(self, line: str, timeout=300) -> tuple[str, str]:
        """Send one user message; return (reply, new-stderr-text)."""
        self.proc.stdin.write(line.encode() + b"\n")
        self.proc.stdin.flush()
        raw = self._wait_stdout(b"\n> ", timeout)
        # The first message's block starts without a leading newline.
        text = raw.decode("utf-8", "replace")
        text = re.sub(r"^.*?Assistant: ", "", text, flags=re.S)
        reply = text.rsplit("\n> ", 1)[0].strip()
        self._drain_err()
        new_err = self.err[self.err_pos:].decode("utf-8", "replace")
        self.err_pos = len(self.err)
        return reply, new_err

    def command(self, line: str, timeout=60) -> str:
        """Send a /command; answers arrive on stderr ending with '> '."""
        self.proc.stdin.write(line.encode() + b"\n")
        self.proc.stdin.flush()
        self.err_pos = len(self.err)
        self._wait_err_prompt(None, timeout)
        out = self.err[self.err_pos:].decode("utf-8", "replace")
        self.err_pos = len(self.err)
        return out

    def quit(self):
        try:
            self.proc.stdin.write(b"/quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        finally:
            self.log.close()


def episode_count(status_text: str) -> int | None:
    match = re.search(r"'demo': (\d+) episodes", status_text)
    return int(match.group(1)) if match else None


def classify(err: str):
    """Parse one message's stderr into routing outcome."""
    route = None
    m = re.search(r"\[neural\] route=(\d) \(", err)
    if m:
        route = int(m.group(1))
    wide = re.search(r"\[neural WIDE_HEAD\].*?tokens=\((\d+),(\d+)\) -> value='(.*?)' \(score=", err)
    activated = wide is not None
    neural_value = wide.group(3) if wide else None
    uncertainty = "insufficient → uncertainty reply" in err
    stored = 1 if re.search(r"^\[write\] ", err, flags=re.M) else 0
    return {"route": route, "activated": activated, "neural_value": neural_value,
            "uncertainty_reply": uncertainty, "stored": stored}


def run_conversation(args, sample, index: int, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    sid_state = str(directory / "state.bnstate")
    turns = list(conversation_turns(sample))
    questions = sample["qa"]
    if args.turn_limit:
        turns = turns[: args.turn_limit]
    if args.question_limit:
        questions = questions[: args.question_limit]

    marker = directory / "written.json"
    if not marker.exists():
        server = Server(args, directory)
        server.start()
        rows, n_trunc, n_long_bytes = [], 0, 0
        began = time.monotonic()
        try:
            for i, turn in enumerate(turns):
                line, truncated = flatten(turn["text"])
                n_trunc += truncated
                n_long_bytes += len(turn["text"].encode()) > EPISODE_BYTES
                reply, err = server.message(line)
                info = classify(err)
                rows.append({"index": i, "dia_id": turn["dia_id"],
                             "stored": info["stored"], "truncated": truncated})
                if (i + 1) % 25 == 0 or i + 1 == len(turns):
                    print(json.dumps({"phase": "write", "conversation": index,
                        "done": i + 1, "total": len(turns),
                        "stored": sum(r["stored"] for r in rows)}), flush=True)
            status = server.command("/status")
            episodes = episode_count(status)
            save_out = server.command(f"/save {sid_state}")
        finally:
            server.quit()
        marker.write_text(json.dumps({"writes": rows, "episodes": episodes,
            "truncated_lines": n_trunc, "over_episode_bytes": n_long_bytes,
            "status": status, "save": save_out,
            "elapsed_s": time.monotonic() - began}, ensure_ascii=False) + "\n")
    written = json.loads(marker.read_text())

    result_path = directory / "queries.json"
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    if len(rows) < len(questions):
        server = Server(args, directory)   # fresh process: real restart path
        server.start()
        try:
            load_out = server.command(f"/load {sid_state}")
            status = server.command("/status")
            if episode_count(status) != written["episodes"]:
                raise AssertionError("episode count changed across restart")
            for i in range(len(rows), len(questions)):
                qa = questions[i]
                line, _ = flatten(qa["question"])
                began = time.monotonic()
                reply, err = server.message(line)
                info = classify(err)
                activated = info["activated"]
                answer = reply if activated else ""
                category = int(qa["category"])
                row = {"index": i, "conversation": index, "category": category,
                       "question": qa["question"], "gold": qa.get("answer"),
                       "activation": {k: info[k] for k in ("route", "neural_value")},
                       "activated": activated, "memory_answer": answer,
                       "raw_response": reply, "err": err,
                       "f1": official_f1(answer, qa["answer"], category)
                             if category != 5 else None,
                       "normalized_exact": bool(activated and
                           normalize(answer) == normalize(qa.get("answer", "")))
                           if category != 5 else None,
                       "null_rejected": (not activated) if category == 5 else None,
                       "elapsed_s": time.monotonic() - began}
                rows.append(row)
                result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
                if (i + 1) % 10 == 0 or i + 1 == len(questions):
                    print(json.dumps({"phase": "query", "conversation": index,
                        "done": i + 1, "total": len(questions),
                        "summary": summarize(rows)}), flush=True)
        finally:
            server.quit()
    save(result_path.with_suffix(".done"), {"summary": summarize(rows)})
    return rows, written


def save(path: Path, value):
    path.with_suffix(path.suffix + ".tmp").write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    path.with_suffix(path.suffix + ".tmp").replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server")
    parser.add_argument("gguf")
    parser.add_argument("memory_model")
    parser.add_argument("--data", default=str(ROOT / "build/locomo10.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--turn-limit", type=int, default=0)
    parser.add_argument("--question-limit", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data)
    if digest(data_path) != DATA_SHA256:
        raise ValueError("dataset does not match pinned official LoCoMo10")
    dataset = json.loads(data_path.read_text())
    selected = [int(i) for i in args.conversations.split(",")]
    if not selected or len(set(selected)) != len(selected) or \
            any(i not in range(10) for i in selected):
        raise ValueError("conversation indices must be unique and within 0..9")

    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "LOCOMO_NEURAL_C_V1", "registration":
                "training/memory/neural-system/reviews/LC-001 (baseline) + "
                "LC-002 (post-fix re-run)",
        "upstream": UPSTREAM, "upstream_revision": UPSTREAM_REVISION,
        "dataset_sha256": digest(data_path),
        "evaluator_sha256": digest(Path(__file__)),
        "scoring": "official category F1 reused from tests/eval_locomo.py",
        "artifacts": {name: {"path": str(Path(p).resolve()), "sha256": digest(p)}
                      for name, p in (("server", args.server), ("gguf", args.gguf),
                                      ("memory_model", args.memory_model))},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                            text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True)),
        "conversations": selected, "turn_limit": args.turn_limit,
        "question_limit": args.question_limit,
        "protocol": "raw-turn single-line writes (auto gate), /save, real restart, "
                    "/load, read-only queries; activated = wide-head value path",
        "system_facts": {"max_episodes_per_session": 16,
                         "recall_reads_last_episode_only": True,
                         "write_gate": "surface heuristic (storage only)"}}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("run configuration changed; use a new output directory")
    else:
        save(manifest_path, manifest)

    all_rows = []
    for index in selected:
        rows, written = run_conversation(args, dataset[index], index,
                                         directory / f"conversation_{index}")
        all_rows.extend(rows)
        report = {"complete": index == selected[-1],
                  "full_locomo": len(selected) == 10 and not args.turn_limit
                                 and not args.question_limit,
                  "summary": summarize(all_rows),
                  "write_stats": {"episodes": written["episodes"],
                                  "truncated_lines": written["truncated_lines"]}}
        save(directory / "summary.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
