#!/usr/bin/env python3
"""Evaluate persistent memory on the official LoCoMo QA dataset."""
import argparse
import collections
import json
import os
import re
import shutil
import socket
import string
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

try:
    from nltk.stem import PorterStemmer
except ImportError:
    PorterStemmer = None

STEMMER = PorterStemmer() if PorterStemmer else None


def request_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers),
                                    timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def find_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(args, port, state_dir):
    proc = subprocess.Popen([
        args.server, args.model, "--memory-model", args.memory_model,
        "--memory-state-dir", state_dir, "--host", "127.0.0.1",
        "--port", str(port), "--ctx", "2048", "--max-tokens", "64",
    ], stdout=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            request_json(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate(); proc.wait()
    raise RuntimeError("server startup timed out")


def stop_server(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait()


def normalize(text):
    text = text.lower().replace(",", "")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    tokens = text.split()
    if STEMMER is not None:
        tokens = [STEMMER.stem(token) for token in tokens]
    return " ".join(tokens)


def f1_score(prediction, truth):
    pred = normalize(prediction).split()
    gold = normalize(truth).split()
    if not pred or not gold:
        return float(pred == gold)
    common = collections.Counter(pred) & collections.Counter(gold)
    same = sum(common.values())
    if not same:
        return 0.0
    precision, recall = same / len(pred), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def official_like_score(prediction, qa):
    category = int(qa["category"])
    answer = str(qa.get("answer", ""))
    if category == 1:
        predictions = [p.strip() for p in prediction.split(",")]
        truths = [g.strip() for g in answer.split(",")]
        return sum(max(f1_score(p, g) for p in predictions) for g in truths) / len(truths)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    if category == 5:
        p = prediction.lower()
        return float("no information available" in p or "not mentioned" in p)
    raise ValueError(f"unknown category {category}")


def answer_text(response):
    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def chunk_turns(turns, max_chars=2400):
    chunks, current, size = [], [], 0
    for turn in turns:
        turn_size = len(turn["speaker"]) + len(turn["text"]) + 3
        if current and size + turn_size > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(turn)
        size += turn_size
    if current:
        chunks.append(current)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server")
    parser.add_argument("model")
    parser.add_argument("memory_model")
    parser.add_argument("dataset")
    parser.add_argument("--output", default="build/locomo_results.json")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="smoke-test limit; zero evaluates every session")
    parser.add_argument("--conversation", type=int, default=-1)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    args = parser.parse_args()

    if STEMMER is None:
        print("warning: nltk is unavailable; scoring without official Porter stemming",
              flush=True)

    samples = json.load(open(args.dataset, encoding="utf-8"))
    if args.conversation >= 0:
        samples = [samples[args.conversation]]
    port = find_port()
    state_dir = tempfile.mkdtemp(prefix="bitnet-locomo-")
    proc = start_server(args, port, state_dir)
    root = f"http://127.0.0.1:{port}"
    chat = root + "/v1/chat/completions"
    results = []
    try:
        for ci, sample in enumerate(samples):
            sid = f"locomo-{sample.get('sample_id', ci)}"
            conversation = sample["conversation"]
            session_keys = sorted(
                (k for k in conversation if re.fullmatch(r"session_\d+", k)),
                key=lambda key: int(key.split("_")[1]))
            if args.max_sessions:
                session_keys = session_keys[:args.max_sessions]
            for si, key in enumerate(session_keys):
                date = conversation.get(key + "_date_time", "unknown date")
                chunks = chunk_turns(conversation[key])
                for chunk_index, turns in enumerate(chunks):
                    transcript = "\n".join(
                        f"{turn['speaker']}: {turn['text']}" for turn in turns)
                    prompt = (f"Conversation session on {date} "
                              f"(part {chunk_index+1}/{len(chunks)}):\n{transcript}\n\n"
                              "Store this conversation in long-term memory. Reply OK.")
                    request_json(chat, {"session_id": sid, "max_tokens": 1,
                                        "messages": [{"role": "user", "content": prompt}]})
                    request_json(root + "/v1/memory/export", {"session_id": sid})
                    request_json(root + "/v1/memory/import", {"session_id": sid})
                print(f"[conversation {ci+1}/{len(samples)} session {si+1}/{len(session_keys)}]",
                      flush=True)

            # Deliberately destroy the process and its KV cache.  The QA phase
            # can recover only the exported persistent-memory snapshot.
            stop_server(proc)
            proc = start_server(args, port, state_dir)
            print(f"[conversation {ci+1} server restarted; KV cache discarded]", flush=True)

            qas = sample["qa"]
            if args.max_questions:
                qas = qas[:args.max_questions]
            for qi, qa in enumerate(qas):
                request_json(root + "/v1/memory/import", {"session_id": sid})
                prompt = ("Answer the question using only the stored conversation memory. "
                          "Give only the shortest direct answer. If the conversation does not "
                          "contain the answer, reply exactly: No information available.\n"
                          f"Question: {qa['question']}")
                response = request_json(chat, {
                    "session_id": sid, "max_tokens": args.max_answer_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                })
                session_stats = response.get("session", {})
                if session_stats.get("cached_tokens", 0) != 0 or \
                        session_stats.get("reused_tokens", 0) != 0:
                    raise RuntimeError("QA request reused KV-cache tokens")
                prediction = answer_text(response)
                score = official_like_score(prediction, qa)
                results.append({"sample_id": sample.get("sample_id", ci),
                                "question": qa["question"], "answer": qa.get("answer"),
                                "category": int(qa["category"]),
                                "prediction": prediction, "f1": score})
                if (qi + 1) % 25 == 0:
                    print(f"[conversation {ci+1} QA {qi+1}/{len(qas)}]", flush=True)
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as output:
                json.dump(results, output, indent=2)

        by_category = collections.defaultdict(list)
        for row in results:
            by_category[row["category"]].append(row["f1"])
        scored = [r["f1"] for r in results if r["category"] in (1, 2, 3, 4)]
        print("\n=== LoCoMo persistent-memory QA ===")
        for category in sorted(by_category):
            values = by_category[category]
            print(f"category {category}: {sum(values)/len(values):.4f} ({len(values)} questions)")
        print(f"official categories 1-4: {sum(scored)/len(scored):.4f} ({len(scored)} questions)")
        print(f"all categories: {sum(r['f1'] for r in results)/len(results):.4f} ({len(results)} questions)")
        print(f"predictions: {args.output}")
    finally:
        stop_server(proc)
        shutil.rmtree(state_dir)


if __name__ == "__main__":
    main()
