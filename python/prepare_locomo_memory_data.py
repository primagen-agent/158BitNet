#!/usr/bin/env python3
"""Convert LoCoMo into memory-training JSONL understood by train_memory.py."""
import argparse
import json
import random
import re
from pathlib import Path


def render_turn(turn, observations=None):
    """Preserve every textual signal attached to a LoCoMo turn.

    A material fraction of LoCoMo answers is grounded in an attached image.
    The dataset's ``query`` field often contains the identifying entity that
    is absent from both the dialogue text and the generic BLIP caption.
    Dropping these fields makes the labelled evidence incomplete.
    """
    parts = [f"{turn['speaker']}: {turn['text']}"]
    image_query = str(turn.get("query", "")).strip()
    if image_query:
        parts.append(f"Referenced image context: {image_query}")
    caption = str(turn.get("blip_caption", "")).strip()
    if caption:
        parts.append(f"Image description: {caption}")
    if observations:
        parts.extend(f"Derived memory fact: {fact}" for fact in observations)
    return "\n".join(parts)


def derived_context(sample):
    observations = {}
    for session in sample.get("observation", {}).values():
        for facts in session.values():
            for item in facts:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                fact, dia_id = str(item[0]).strip(), str(item[1]).strip()
                if fact and dia_id:
                    observations.setdefault(dia_id, []).append(fact)
    return (
        observations,
        sample.get("session_summary", {}),
        sample.get("event_summary", {}),
    )


def session_context(key, summaries, events):
    parts = []
    summary = str(summaries.get(key + "_summary", "")).strip()
    if summary:
        parts.append(f"Derived session summary: {summary}")
    event = events.get("events_" + key)
    if isinstance(event, dict):
        event_parts = []
        date = str(event.get("date", "")).strip()
        if date:
            event_parts.append(f"date: {date}")
        for speaker, values in event.items():
            if speaker == "date" or not isinstance(values, list):
                continue
            event_parts.extend(
                f"{speaker}: {str(value).strip()}"
                for value in values if str(value).strip())
        if event_parts:
            parts.append("Derived session events: " + " | ".join(event_parts))
    return parts


def session_chunks(
    conversation, key, max_chars, observations=None, summaries=None,
    events=None
):
    date = conversation.get(key + "_date_time", "unknown date")
    observations = observations or {}
    summaries = summaries or {}
    events = events or {}
    context = session_context(key, summaries, events)
    chunks, current, size = [], [], 0
    for turn in conversation[key]:
        rendered_turn = render_turn(
            turn, observations.get(str(turn.get("dia_id", "")), []))
        turn_size = len(rendered_turn) + 1
        if current and size + turn_size > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(rendered_turn)
        size += turn_size
    if current:
        chunks.append(current)
    rendered = []
    for index, turns in enumerate(chunks):
        transcript = "\n".join(context + turns)
        content = (f"Conversation session on {date} "
                   f"(part {index+1}/{len(chunks)}):\n{transcript}\n\n"
                   "Store this conversation in long-term memory. Reply OK.")
        rendered.append([
            {"role": "user", "content": content},
            {"role": "assistant", "content": "OK"},
        ])
    return rendered


def evidence_sessions(qa):
    sessions = set()
    for evidence in qa.get("evidence", []):
        match = re.match(r"D(\d+):", str(evidence))
        if match:
            sessions.add(f"session_{int(match.group(1))}")
    return sessions


def evidence_turns(qa):
    turns = []
    for evidence in qa.get("evidence", []):
        match = re.fullmatch(r"D(\d+):(\d+)", str(evidence))
        if match:
            turns.append((f"session_{int(match.group(1))}",
                          int(match.group(2)) - 1))
    return turns


def turn_chunk(
    conversation, key, turn_index, observations=None, summaries=None,
    events=None
):
    turns = conversation.get(key, [])
    if turn_index < 0 or turn_index >= len(turns):
        return None
    turn = turns[turn_index]
    date = conversation.get(key + "_date_time", "unknown date")
    observations = observations or {}
    transcript = render_turn(
        turn, observations.get(str(turn.get("dia_id", "")), []))
    content = (f"Conversation session on {date} (part 1/1):\n"
               f"{transcript}\n\n"
               "Store this conversation in long-term memory. Reply OK.")
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "OK"},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output_dir")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=2400)
    parser.add_argument("--full-sessions", action="store_true",
                        help="use complete evidence sessions instead of annotated turns")
    parser.add_argument(
        "--include-derived-context", action="store_true",
        help="include LoCoMo observation, session-summary, and event annotations")
    parser.add_argument("--valid-conversations", type=int, default=0,
                        help="hold out this many whole conversations into valid/")
    parser.add_argument("--include-category5", action="store_true",
                        help="include unanswerable questions; add only after recall warm-up")
    parser.add_argument(
        "--split-categories", action="store_true",
        help="write one JSONL stratum per category for balanced sampling")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    if args.valid_conversations < 0 or args.valid_conversations >= len(samples):
        raise ValueError("--valid-conversations must be in [0, conversations)")
    conversation_order = list(range(len(samples)))
    rng.shuffle(conversation_order)
    valid_conversations = set(
        conversation_order[:args.valid_conversations])
    train_rows, valid_rows = [], []
    for copy in range(args.copies):
        for conversation_index, sample in enumerate(samples):
            conversation = sample["conversation"]
            if args.include_derived_context:
                observations, summaries, events = derived_context(sample)
            else:
                observations, summaries, events = {}, {}, {}
            all_sessions = sorted(
                (key for key in conversation if re.fullmatch(r"session_\d+", key)),
                key=lambda key: int(key.split("_")[1]))
            for question_index, qa in enumerate(sample["qa"]):
                if int(qa["category"]) == 5 and not args.include_category5:
                    continue
                messages = []
                evidence_message_indices = []
                distractor_message_indices = []
                if args.full_sessions:
                    relevant = evidence_sessions(qa)
                    available = [key for key in all_sessions if key not in relevant]
                    selected = list(relevant) + rng.sample(
                        available, min(args.distractors, len(available)))
                    rng.shuffle(selected)
                    for key in selected:
                        chunks = session_chunks(
                            conversation, key, args.max_chars,
                            observations, summaries, events)
                        target = (
                            evidence_message_indices
                            if key in relevant
                            else distractor_message_indices)
                        target.extend(
                            range(len(messages), len(messages) + len(chunks)))
                        messages.extend(chunks)
                else:
                    relevant = evidence_turns(qa)
                    all_turns = [(key, index) for key in all_sessions
                                 for index in range(len(conversation[key]))]
                    available = [item for item in all_turns if item not in relevant]
                    selected = list(relevant) + rng.sample(
                        available, min(args.distractors, len(available)))
                    rng.shuffle(selected)
                    for key, index in selected:
                        chunk = turn_chunk(
                            conversation, key, index,
                            observations, summaries, events)
                        if chunk is not None:
                            target = (
                                evidence_message_indices
                                if (key, index) in relevant
                                else distractor_message_indices)
                            target.append(len(messages))
                            messages.append(chunk)
                answer = qa.get("answer")
                if int(qa["category"]) == 5 or answer is None:
                    answer = "No information available"
                query = ("Answer the question using only the stored conversation memory. "
                         "Give only the shortest direct answer. If the conversation does not "
                         "contain the answer, reply exactly: No information available.\n"
                         f"Question: {qa['question']}")
                messages.append([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": str(answer)},
                ])
                row = {
                    "messages": messages,
                    "query_turn_id": len(messages) - 1,
                    "locomo_id": f"{conversation_index}:{question_index}:{copy}",
                    "validation_group": f"conversation:{conversation_index}",
                    "category": int(qa["category"]),
                    "evidence_message_indices": evidence_message_indices,
                    "distractor_message_indices": distractor_message_indices,
                }
                (valid_rows if conversation_index in valid_conversations
                 else train_rows).append(row)

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    output_dir = Path(args.output_dir)

    def write_rows(split_dir, rows):
        split_dir.mkdir(parents=True, exist_ok=True)
        if args.split_categories:
            outputs = []
            for category in sorted({int(row["category"]) for row in rows}):
                selected = [
                    row for row in rows if int(row["category"]) == category]
                output = split_dir / f"locomo_cat{category}.jsonl"
                with output.open("w", encoding="utf-8") as handle:
                    for row in selected:
                        handle.write(
                            json.dumps(row, ensure_ascii=False) + "\n")
                outputs.append({
                    "output": str(output),
                    "category": category,
                    "samples": len(selected),
                })
            return outputs
        output = split_dir / "multi_entity.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [{"output": str(output), "samples": len(rows)}]

    if args.valid_conversations:
        outputs = []
        for split, rows in (("train", train_rows), ("valid", valid_rows)):
            split_dir = output_dir / split
            for output in write_rows(split_dir, rows):
                outputs.append({"split": split, **output})
        print(json.dumps({
            "outputs": outputs,
            "valid_conversation_ids": sorted(valid_conversations),
        }, indent=2))
    else:
        print(json.dumps({
            "outputs": write_rows(output_dir, train_rows),
            "samples": len(train_rows),
        }, indent=2))


if __name__ == "__main__":
    main()
