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
EVIDENCE_ID_RE = re.compile(r"\bD\d+:\d+\b")


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
    command = [
        args.server, args.model,
        "--memory-state-dir", state_dir, "--host", "127.0.0.1",
        "--port", str(port), "--ctx", str(args.ctx),
        "--max-tokens", str(max(64, args.max_answer_tokens)),
    ]
    if args.memory_model != "-":
        command += ["--memory-model", args.memory_model]
    if args.addressed:
        command += [
            "--episodic-memory",
            "--episodic-top-k", str(args.top_k),
            "--episodic-lexical-weight", str(args.lexical_weight),
            "--episodic-priority-weight", str(args.priority_weight),
        ]
        if args.mode == "retrieval-baseline":
            command += ["--episodic-context-injection"]
        if args.controller:
            command += ["--memory-controller", args.controller]
        if getattr(args, "retriever", None):
            command += ["--memory-retriever", args.retriever]
        if args.pointer:
            command += ["--memory-pointer", args.pointer]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
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


def answer_token_recall(context, qa):
    """Fraction of normalized gold-answer tokens present in retrieval."""
    category = int(qa["category"])
    if category == 5:
        return None
    answer = str(qa.get("answer", ""))
    if category == 3:
        answer = answer.split(";")[0].strip()
    gold = normalize(answer).split()
    if not gold:
        return 0.0
    retrieved = collections.Counter(normalize(context).split())
    needed = collections.Counter(gold)
    return sum((retrieved & needed).values()) / sum(needed.values())


def answer_text(response):
    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def ordered_session_keys(conversation):
    return sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.split("_")[1]))


def build_turn_index(conversation):
    """Map LoCoMo's gold evidence IDs to source turns and session metadata."""
    index = {}
    for key in ordered_session_keys(conversation):
        date = conversation.get(key + "_date_time", "unknown date")
        for turn in conversation[key]:
            evidence_id = turn.get("dia_id")
            if not evidence_id:
                continue
            if evidence_id in index:
                raise ValueError(f"duplicate LoCoMo dia_id: {evidence_id}")
            index[evidence_id] = {
                "dia_id": evidence_id,
                "session": key,
                "date": date,
                "speaker": str(turn.get("speaker", "")),
                "text": str(turn.get("text", "")),
            }
    return index


def qa_evidence_ids(qa):
    """Normalize LoCoMo evidence fields, including semicolon-packed IDs."""
    evidence = qa.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    return [
        evidence_id
        for item in evidence
        for evidence_id in EVIDENCE_ID_RE.findall(str(item))
    ]


def format_source_turn(turn, include_date=False):
    date = f" | {turn['date']}" if include_date else ""
    return (
        f"[source {turn['dia_id']}{date}] "
        f"{turn['speaker']}: {turn['text']}")


def source_context(evidence_ids, turn_index):
    lines = []
    for evidence_id in evidence_ids:
        turn = turn_index.get(evidence_id)
        if turn is None:
            raise ValueError(f"missing LoCoMo evidence turn: {evidence_id}")
        lines.append(format_source_turn(turn, include_date=True))
    return "\n".join(lines)


def full_conversation_context(conversation, session_keys):
    lines = []
    for key in session_keys:
        date = conversation.get(key + "_date_time", "unknown date")
        lines.append(f"[session {key} | {date}]")
        for turn in conversation[key]:
            evidence_id = turn.get("dia_id", "unknown")
            lines.append(
                f"[source {evidence_id}] "
                f"{turn.get('speaker', '')}: {turn.get('text', '')}")
    return "\n".join(lines)


def eligible_qas(qas, available_source_ids, limited_sessions):
    """Avoid scoring questions whose gold evidence was never ingested."""
    if not limited_sessions:
        return list(qas)
    eligible = []
    for qa in qas:
        evidence = qa_evidence_ids(qa)
        if evidence and all(item in available_source_ids for item in evidence):
            eligible.append(qa)
    return eligible


def ranked_evidence_ids(context):
    """Return one set of source IDs per ranked memory record."""
    matches = list(re.finditer(r"(?m)^\[memory\s+\d+\]\s*", context))
    ranked = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        ranked.append(set(EVIDENCE_ID_RE.findall(context[match.end():end])))
    return ranked


def load_write_plan(path):
    """Load conversation-scoped decisions from the V104d evaluator."""
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    plan = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        conversation = row.get("conversation")
        if not source_id or conversation is None:
            raise ValueError(
                "write plan row needs conversation and source_id")
        key = (int(conversation), source_id)
        selected = bool(row.get("selected", False))
        if key in plan and plan[key] != selected:
            raise ValueError(
                f"conflicting write decisions for {key}")
        plan[key] = selected
    return plan


def load_priority_plan(path):
    """Load bounded salience values without dropping source records."""
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    plan = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        conversation = row.get("conversation")
        priority = row.get("write_probability")
        if not source_id or conversation is None or priority is None:
            raise ValueError(
                "priority plan row needs conversation, source_id, "
                "and write_probability")
        key = (int(conversation), source_id)
        priority = float(priority)
        if not 0.0 <= priority <= 1.0:
            raise ValueError(f"invalid memory priority for {key}")
        if key in plan and plan[key] != priority:
            raise ValueError(f"conflicting priorities for {key}")
        plan[key] = priority
    return plan


def evidence_retrieval_metrics(ranked_ids, gold_ids, cutoffs=(1, 5, 10)):
    gold = set(gold_ids)
    metrics = {}
    for cutoff in cutoffs:
        retrieved = set().union(*ranked_ids[:cutoff]) if ranked_ids[:cutoff] else set()
        metrics[f"evidence_recall_at_{cutoff}"] = (
            len(gold & retrieved) / len(gold) if gold else None)
        metrics[f"evidence_complete_at_{cutoff}"] = (
            float(gold <= retrieved) if gold else None)
    reciprocal_rank = 0.0
    if gold:
        for rank, ids in enumerate(ranked_ids, 1):
            if gold & ids:
                reciprocal_rank = 1.0 / rank
                break
    metrics["evidence_mrr"] = reciprocal_rank if gold else None
    return metrics


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
    parser.add_argument(
        "--mode",
        choices=(
            "memory", "retrieval-baseline",
            "gold-evidence", "full-context"),
        default="memory",
        help=(
            "native neural-memory evaluation, explicitly labelled RAG "
            "baseline, or a plain-backbone answer ceiling"))
    parser.add_argument("--output", default="build/locomo_results.json")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="smoke-test limit; zero evaluates every session")
    parser.add_argument("--conversation", type=int, default=-1)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--addressed", action="store_true")
    parser.add_argument("--controller")
    parser.add_argument("--retriever")
    parser.add_argument("--pointer")
    parser.add_argument("--oracle-write", action="store_true")
    parser.add_argument(
        "--write-plan",
        help="V104d source-id selection JSON; requires addressed turn-level "
             "evaluation and stores only selected immutable source turns")
    parser.add_argument(
        "--priority-plan",
        help="V104d source-id score JSON; stores every source turn and uses "
             "the score only as a bounded retrieval tie-breaker")
    parser.add_argument("--turn-level", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lexical-weight", type=float, default=0.25)
    parser.add_argument("--priority-weight", type=float, default=0.0)
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="record write actions, record counts, and retrieved context")
    args = parser.parse_args()
    if args.mode == "memory":
        if args.memory_model == "-":
            parser.error("--mode memory requires a native .bnmem model")
        if (
            args.addressed or args.controller or args.retriever
            or args.pointer or args.write_plan or args.priority_plan
            or args.oracle_write
        ):
            parser.error(
                "--mode memory is native .bnmem activation only; episodic "
                "retrieval, controller, pointer, and write plans belong to "
                "--mode retrieval-baseline")
    if args.mode == "retrieval-baseline" and not args.addressed:
        parser.error("--mode retrieval-baseline requires --addressed")
    if args.mode != "memory" and args.memory_model != "-":
        parser.error(
            "only --mode memory may load a native .bnmem model")
    if args.mode not in ("memory", "retrieval-baseline") and args.addressed:
        parser.error(
            "--addressed is only valid with --mode retrieval-baseline")
    if args.mode in ("gold-evidence", "full-context") and (
        args.controller or args.retriever or args.pointer
        or args.write_plan or args.priority_plan or args.oracle_write
    ):
        parser.error(
            "plain-backbone baseline modes cannot load memory components")
    if args.addressed and not (
        args.controller or args.retriever or args.pointer
    ):
        parser.error(
            "--addressed requires a controller, retriever, or pointer")
    if args.write_plan and (
            not args.addressed or not args.turn_level):
        parser.error("--write-plan requires --addressed --turn-level")
    if args.write_plan and args.oracle_write:
        parser.error("--write-plan and --oracle-write are mutually exclusive")
    if args.priority_plan and (
            not args.addressed or not args.turn_level):
        parser.error("--priority-plan requires --addressed --turn-level")
    if args.priority_plan and (args.write_plan or args.oracle_write):
        parser.error(
            "--priority-plan is mutually exclusive with hard write modes")
    if not 0.0 <= args.priority_weight < 0.5:
        parser.error("--priority-weight must be in [0, 0.5)")

    if STEMMER is None:
        print("warning: nltk is unavailable; scoring without official Porter stemming",
              flush=True)

    samples = json.load(open(args.dataset, encoding="utf-8"))
    write_plan = load_write_plan(args.write_plan)
    priority_plan = load_priority_plan(args.priority_plan)
    if args.conversation >= 0:
        samples = [samples[args.conversation]]
    port = find_port()
    state_dir = tempfile.mkdtemp(prefix="bitnet-locomo-")
    proc = start_server(args, port, state_dir)
    root = f"http://127.0.0.1:{port}"
    chat = root + "/v1/chat/completions"
    results = []
    write_actions = collections.Counter()
    exported_record_counts = []
    try:
        for ci, sample in enumerate(samples):
            plan_conversation = (
                args.conversation if args.conversation >= 0 else ci)
            sid = f"locomo-{sample.get('sample_id', ci)}"
            conversation = sample["conversation"]
            all_session_keys = ordered_session_keys(conversation)
            session_keys = list(all_session_keys)
            if args.max_sessions:
                session_keys = session_keys[:args.max_sessions]
            turn_index = build_turn_index(conversation)
            available_source_ids = {
                turn["dia_id"]
                for key in session_keys
                for turn in conversation[key]
                if turn.get("dia_id")
            }
            if args.mode in ("memory", "retrieval-baseline"):
                for si, key in enumerate(session_keys):
                    date = conversation.get(key + "_date_time", "unknown date")
                    chunks = (
                        [[turn] for turn in conversation[key]]
                        if args.turn_level else chunk_turns(conversation[key]))
                    for chunk_index, turns in enumerate(chunks):
                        if write_plan is not None:
                            turns = [
                                turn for turn in turns
                                if write_plan.get(
                                    (
                                        plan_conversation,
                                        str(turn.get("dia_id", "")),
                                    ),
                                    False)
                            ]
                            if not turns:
                                continue
                        transcript = "\n".join(
                            format_source_turn({
                                "dia_id": turn.get("dia_id", "unknown"),
                                "date": date,
                                "speaker": str(turn.get("speaker", "")),
                                "text": str(turn.get("text", "")),
                            })
                            for turn in turns)
                        prompt = (
                            f"Conversation session on {date} "
                            f"(part {chunk_index+1}/{len(chunks)}):\n"
                            f"{transcript}\n\n"
                            "Store this conversation in long-term memory. Reply OK.")
                        payload = {
                            "session_id": sid, "max_tokens": 1,
                            "reset_context": True,
                            "messages": [{"role": "user", "content": prompt}],
                        }
                        if args.addressed and (
                                args.oracle_write
                                or write_plan is not None
                                or priority_plan is not None):
                            payload["memory_action"] = "write"
                            payload["memory_record"] = transcript
                        if priority_plan is not None:
                            source_id = str(
                                turns[0].get("dia_id", ""))
                            payload["memory_priority"] = priority_plan.get(
                                (plan_conversation, source_id), 0.0)
                        write_response = request_json(chat, payload)
                        write_actions[
                            write_response.get("memory_action", "missing")
                        ] += 1
                    print(
                        f"[conversation {ci+1}/{len(samples)} "
                        f"session {si+1}/{len(session_keys)}]",
                        flush=True)
                exported = request_json(
                    root + "/v1/memory/export", {"session_id": sid})
                exported_record_counts.append(
                    int(exported.get("episodic_records", 0)))

                # Deliberately destroy the process and its KV cache. The QA
                # phase can recover only the exported persistent snapshot.
                stop_server(proc)
                proc = start_server(args, port, state_dir)
                print(
                    f"[conversation {ci+1} server restarted; "
                    "KV cache discarded]",
                    flush=True)
                request_json(root + "/v1/memory/import", {"session_id": sid})

            qas = eligible_qas(
                sample["qa"], available_source_ids,
                limited_sessions=bool(args.max_sessions))
            if args.max_questions:
                qas = qas[:args.max_questions]
            if args.max_sessions:
                print(
                    f"[conversation {ci+1} eligible QA after evidence "
                    f"filter: {len(qas)}]",
                    flush=True)
            full_context = (
                full_conversation_context(conversation, session_keys)
                if args.mode == "full-context" else "")
            for qi, qa in enumerate(qas):
                retrieval_context = ""
                retrieval_recall = None
                retrieval_metrics = {}
                if args.addressed and args.diagnostics:
                    retrieval = request_json(
                        root + "/v1/memory/search",
                        {
                            "session_id": sid,
                            "query": qa["question"],
                            "top_k": max(10, args.top_k),
                        },
                    )
                    retrieval_context = retrieval.get("context", "")
                    retrieval_recall = answer_token_recall(
                        retrieval_context, qa)
                    retrieval_metrics = evidence_retrieval_metrics(
                        ranked_evidence_ids(retrieval_context),
                        qa_evidence_ids(qa))
                if args.mode == "gold-evidence":
                    supplied_context = source_context(
                        qa_evidence_ids(qa),
                        turn_index)
                    instruction = "the supplied gold evidence"
                elif args.mode == "full-context":
                    supplied_context = full_context
                    instruction = "the supplied conversation"
                elif args.mode == "retrieval-baseline":
                    supplied_context = ""
                    instruction = "the injected retrieved episodic evidence"
                else:
                    supplied_context = ""
                    instruction = "the activated neural memory"
                prompt = (
                    f"Answer the question using only {instruction}. "
                    "Give only the shortest direct answer. If it does not "
                    "contain the answer, reply exactly: "
                    "No information available.\n"
                    + (
                        f"\nEvidence:\n{supplied_context}\n"
                        if supplied_context else "")
                    + f"\nQuestion: {qa['question']}")
                query_sid = (
                    sid if args.mode in ("memory", "retrieval-baseline")
                    else f"{sid}-baseline-{qi}")
                response = request_json(
                    chat,
                    {
                        "session_id": query_sid,
                        "reset_session": args.mode not in (
                            "memory", "retrieval-baseline"),
                        "reset_context": args.mode in (
                            "memory", "retrieval-baseline"),
                        "memory_commit": False,
                        "max_tokens": args.max_answer_tokens,
                        **({"memory_action": "ignore"} if args.addressed else {}),
                        **({"memory_copy": True} if args.pointer else {}),
                        "messages": [{"role": "user", "content": prompt}],
                    })
                session_stats = response.get("bitnet_session", {})
                if session_stats.get("cached_tokens", 0) != 0 or \
                        session_stats.get("reused_tokens", 0) != 0:
                    raise RuntimeError("QA request reused KV-cache tokens")
                if args.mode == "memory":
                    activation_reads = int(
                        session_stats.get("memory_activation_reads", 0))
                    activation_l2 = float(
                        session_stats.get("memory_activation_l2", 0.0))
                    if activation_reads <= 0 or activation_l2 <= 0.0:
                        raise RuntimeError(
                            "native memory evaluation produced no non-zero "
                            "memory-layer activation")
                prediction = answer_text(response)
                score = official_like_score(prediction, qa)
                ablation_prediction = ""
                ablation_score = None
                if args.mode == "memory":
                    ablation_response = request_json(
                        chat,
                        {
                            "session_id": sid,
                            "reset_context": True,
                            "memory_commit": False,
                            "memory_activation": False,
                            "max_tokens": args.max_answer_tokens,
                            "messages": [{
                                "role": "user", "content": prompt,
                            }],
                        })
                    ablation_stats = ablation_response.get(
                        "bitnet_session", {})
                    if (
                        ablation_stats.get("cached_tokens", 0) != 0
                        or ablation_stats.get("reused_tokens", 0) != 0
                        or ablation_stats.get(
                            "memory_activation_reads", 0) != 0
                        or float(ablation_stats.get(
                            "memory_activation_l2", 0.0)) != 0.0
                    ):
                        raise RuntimeError(
                            "activation-off ablation was not isolated")
                    ablation_prediction = answer_text(ablation_response)
                    ablation_score = official_like_score(
                        ablation_prediction, qa)
                row = {
                    "sample_id": sample.get("sample_id", ci),
                    "question": qa["question"],
                    "answer": qa.get("answer"),
                    "category": int(qa["category"]),
                    "evidence": qa.get("evidence", []),
                    "prediction": prediction,
                    "f1": score,
                    "memory_activation_reads": session_stats.get(
                        "memory_activation_reads", 0),
                    "memory_activation_l2": session_stats.get(
                        "memory_activation_l2", 0.0),
                }
                if ablation_score is not None:
                    row["activation_off_prediction"] = (
                        ablation_prediction)
                    row["activation_off_f1"] = ablation_score
                    row["activation_f1_gain"] = score - ablation_score
                    row["activation_changed_answer"] = (
                        normalize(prediction)
                        != normalize(ablation_prediction))
                if args.diagnostics:
                    row["retrieved_context"] = retrieval_context
                    row["retrieval_answer_token_recall"] = (
                        retrieval_recall)
                    row.update(retrieval_metrics)
                results.append(row)
                if (qi + 1) % 25 == 0:
                    print(f"[conversation {ci+1} QA {qi+1}/{len(qas)}]", flush=True)
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as output:
                json.dump(results, output, indent=2)

        by_category = collections.defaultdict(list)
        for row in results:
            by_category[row["category"]].append(row["f1"])
        scored = [r["f1"] for r in results if r["category"] in (1, 2, 3, 4)]
        print(f"\n=== LoCoMo QA ({args.mode}) ===")
        for category in sorted(by_category):
            values = by_category[category]
            print(f"category {category}: {sum(values)/len(values):.4f} ({len(values)} questions)")
        print(f"official categories 1-4: {sum(scored)/len(scored):.4f} ({len(scored)} questions)")
        print(f"all categories: {sum(r['f1'] for r in results)/len(results):.4f} ({len(results)} questions)")
        if args.mode == "memory":
            gains = [
                float(row["activation_f1_gain"]) for row in results]
            changed = sum(
                bool(row["activation_changed_answer"]) for row in results)
            ablation_scored = [
                float(row["activation_off_f1"])
                for row in results
                if row["category"] in (1, 2, 3, 4)]
            if ablation_scored:
                print(
                    "activation-off categories 1-4: "
                    f"{sum(ablation_scored) / len(ablation_scored):.4f}")
            print(
                "native activation F1 gain: "
                f"{sum(gains) / len(gains):+.4f}")
            print(
                "activation changed answer: "
                f"{changed}/{len(results)} "
                f"({changed / len(results):.2%})")
            print(
                "native recall verdict: "
                + (
                    "PASS (activation improves answer F1)"
                    if sum(gains) > 0.0
                    else "FAIL (no positive F1 gain over activation-off)"
                ))
        if args.diagnostics:
            retrieval_scores = [
                row["retrieval_answer_token_recall"]
                for row in results
                if row["retrieval_answer_token_recall"] is not None
            ]
            print(
                "write actions: "
                + json.dumps(dict(write_actions), sort_keys=True))
            print(
                "exported episodic records: "
                + json.dumps(exported_record_counts))
            if write_plan is not None:
                print(
                    "writer-selected source turns: "
                    f"{sum(write_plan.values())}/{len(write_plan)}")
            if priority_plan is not None:
                values = list(priority_plan.values())
                print(
                    "writer priorities: "
                    f"mean={sum(values)/max(len(values), 1):.4f}, "
                    f"records={len(values)}, "
                    f"weight={args.priority_weight:.4f}")
            print(
                "retrieval answer-token recall: "
                f"{sum(retrieval_scores)/len(retrieval_scores):.4f} "
                f"({len(retrieval_scores)} questions)"
                if retrieval_scores else
                "retrieval answer-token recall: n/a")
            for metric in (
                    "evidence_recall_at_1", "evidence_recall_at_5",
                    "evidence_recall_at_10", "evidence_mrr",
                    "evidence_complete_at_1", "evidence_complete_at_5",
                    "evidence_complete_at_10"):
                values = [
                    row[metric] for row in results
                    if row.get(metric) is not None
                ]
                if values:
                    print(
                        f"{metric}: {sum(values)/len(values):.4f} "
                        f"({len(values)} questions)")
        print(f"predictions: {args.output}")
    finally:
        stop_server(proc)
        shutil.rmtree(state_dir)


if __name__ == "__main__":
    main()
