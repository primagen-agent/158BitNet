#!/usr/bin/env python3
"""Build LoCoMo QA data with a token-preserving lexical memory retriever.

The retriever never reads the labelled answer or evidence when selecting
memory.  It stores each conversation turn as an independent memory entry and
uses BM25 against the question to select the top-K entries.  Evidence labels
are used only to report retrieval recall for diagnostics.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
from pathlib import Path

from prepare_locomo_memory_data import derived_context, render_turn


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def make_entries(sample, include_derived_context):
    conversation = sample["conversation"]
    observations, _summaries, _events = (
        derived_context(sample) if include_derived_context else ({}, {}, {}))
    entries = []
    session_keys = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.split("_")[1]))
    for key in session_keys:
        date = conversation.get(key + "_date_time", "unknown date")
        for turn in conversation[key]:
            dia_id = str(turn.get("dia_id", ""))
            body = render_turn(turn, observations.get(dia_id, []))
            text = f"Conversation event on {date}:\n{body}"
            entries.append({
                "dia_id": dia_id,
                "date": date,
                "text": text,
                "tokens": tokenize(text),
            })
    return entries


def bm25_rank(entries, question, top_k):
    query_tokens = tokenize(question)
    document_frequency = collections.Counter()
    for entry in entries:
        document_frequency.update(set(entry["tokens"]))
    n_docs = max(len(entries), 1)
    average_length = (
        sum(len(entry["tokens"]) for entry in entries) / n_docs)
    query_counts = collections.Counter(query_tokens)
    scored = []
    for index, entry in enumerate(entries):
        counts = collections.Counter(entry["tokens"])
        length = len(entry["tokens"])
        score = 0.0
        for token, query_count in query_counts.items():
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * length / max(average_length, 1.0))
            score += (
                inverse_document_frequency
                * frequency * 2.2 / denominator
                * (1.0 + 0.15 * min(query_count - 1, 2)))
        scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(
        (index for _score, index in scored[:top_k]))
    return [entries[index] for index in selected]


def evidence_ids(qa):
    return {str(value) for value in qa.get("evidence", [])}


def write_split(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    outputs = []
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        output = path / f"locomo_cat{category}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs.append({
            "output": str(output),
            "category": category,
            "samples": len(selected),
        })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output_dir")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--valid-conversations", type=int, default=2)
    parser.add_argument("--include-derived-context", action="store_true")
    parser.add_argument("--include-category5", action="store_true")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    valid_ids = set(order[:args.valid_conversations])
    train_rows, valid_rows = [], []
    recall_hits = 0
    recall_total = 0
    exact_recall = 0
    questions = 0
    for conversation_index, sample in enumerate(samples):
        entries = make_entries(sample, args.include_derived_context)
        for question_index, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            if category == 5 and not args.include_category5:
                continue
            selected = bm25_rank(entries, str(qa["question"]), args.top_k)
            selected_ids = {entry["dia_id"] for entry in selected}
            gold_ids = evidence_ids(qa)
            if gold_ids:
                hits = len(selected_ids & gold_ids)
                recall_hits += hits
                recall_total += len(gold_ids)
                exact_recall += int(gold_ids <= selected_ids)
                questions += 1
            answer = qa.get("answer")
            if category == 5 or answer is None:
                answer = "No information available"
            query = (
                "Answer the question using only the stored conversation "
                "memory. Give only the shortest direct answer. If the "
                "conversation does not contain the answer, reply exactly: "
                "No information available.\n"
                f"Question: {qa['question']}")
            for copy in range(args.copies):
                messages = [
                    [
                        {"role": "user", "content": (
                            f"{entry['text']}\n\nStore this conversation "
                            "event in long-term memory. Reply OK.")},
                        {"role": "assistant", "content": "OK"},
                    ]
                    for entry in selected
                ]
                messages.append([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": str(answer)},
                ])
                row = {
                    "messages": messages,
                    "query_turn_id": len(messages) - 1,
                    "locomo_id": (
                        f"{conversation_index}:{question_index}:{copy}"),
                    "validation_group": (
                        f"conversation:{conversation_index}"),
                    "category": category,
                    "retrieved_dia_ids": [
                        entry["dia_id"] for entry in selected],
                    "evidence_dia_ids": sorted(gold_ids),
                }
                target = (
                    valid_rows if conversation_index in valid_ids
                    else train_rows)
                target.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    output_dir = Path(args.output_dir)
    outputs = []
    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        for item in write_split(output_dir / split, rows):
            outputs.append({"split": split, **item})
    print(json.dumps({
        "outputs": outputs,
        "valid_conversation_ids": sorted(valid_ids),
        "evidence_recall": recall_hits / max(recall_total, 1),
        "all_evidence_recall": exact_recall / max(questions, 1),
        "questions_with_evidence": questions,
        "top_k": args.top_k,
    }, indent=2))


if __name__ == "__main__":
    main()
