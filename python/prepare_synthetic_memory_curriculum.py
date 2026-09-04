#!/usr/bin/env python3
"""Generate entity-randomized memory QA curriculum data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


NAMES = [
    "Alice", "Ben", "Carla", "Daniel", "Elena", "Felix", "Grace",
    "Henry", "Iris", "Jonah", "Keira", "Liam", "Maya", "Noah",
    "Olivia", "Peter", "Quinn", "Rosa", "Sam", "Tara", "Uma",
    "Victor", "Wendy", "Xavier", "Yara", "Zane",
]
COLORS = [
    "amber", "blue", "coral", "green", "indigo", "orange", "purple",
    "red", "silver", "teal", "white", "yellow",
]
CITIES = [
    "Austin", "Berlin", "Boston", "Dublin", "Helsinki", "Lisbon",
    "Melbourne", "Osaka", "Paris", "Prague", "Seoul", "Toronto",
]
FOODS = [
    "apple pie", "coconut curry", "dumplings", "falafel", "mushroom soup",
    "pasta", "ramen", "roasted vegetables", "tacos", "vegan ice cream",
]
HOBBIES = [
    "bird watching", "gardening", "hiking", "painting", "photography",
    "playing chess", "pottery", "running", "swimming", "woodworking",
]
PETS = [
    "cat", "dog", "gecko", "parrot", "rabbit", "snake", "turtle",
]


def memory_chunk(statement: str):
    return [
        {
            "role": "user",
            "content": (
                f"Conversation memory record:\n{statement}\n\n"
                "Store this conversation in long-term memory. Reply OK."
            ),
        },
        {"role": "assistant", "content": "OK"},
    ]


def query_chunk(question: str, answer: str):
    return [
        {
            "role": "user",
            "content": (
                "Answer the question using only the stored conversation "
                "memory. Give only the shortest direct answer. If the "
                "conversation does not contain the answer, reply exactly: "
                "No information available.\n"
                f"Question: {question}"
            ),
        },
        {"role": "assistant", "content": answer},
    ]


def row(sample_id, messages, question, answer, evidence, distractors=(),
        negatives=()):
    messages = list(messages) + [query_chunk(question, answer)]
    return {
        "messages": messages,
        "query_turn_id": len(messages) - 1,
        "sample_id": sample_id,
        "evidence_message_indices": list(evidence),
        "distractor_message_indices": list(distractors),
        "negative_answers": list(negatives),
    }


def remember_example(rng: random.Random, index: int):
    name = rng.choice(NAMES)
    kind, value, question = rng.choice([
        ("favorite color", rng.choice(COLORS),
         f"What is {name}'s favorite color?"),
        ("favorite food", rng.choice(FOODS),
         f"What is {name}'s favorite food?"),
        ("hobby", rng.choice(HOBBIES),
         f"What is {name}'s hobby?"),
        ("pet", rng.choice(PETS),
         f"What kind of pet does {name} have?"),
        ("home city", rng.choice(CITIES),
         f"Where does {name} live?"),
    ])
    statement = f"{name}'s {kind} is {value}."
    return row(
        f"synthetic-remember:{index}",
        [memory_chunk(statement)], question, value, [0])


def update_example(rng: random.Random, index: int):
    name = rng.choice(NAMES)
    attribute, values, question = rng.choice([
        ("favorite color", COLORS, f"What is {name}'s favorite color now?"),
        ("favorite food", FOODS, f"What is {name}'s favorite food now?"),
        ("hobby", HOBBIES, f"What is {name}'s current hobby?"),
        ("home city", CITIES, f"Where does {name} live now?"),
    ])
    old, new = rng.sample(values, 2)
    messages = [
        memory_chunk(f"{name}'s {attribute} is {old}."),
        memory_chunk(
            f"{name} changed their {attribute}; it is now {new}, not {old}."),
    ]
    return row(
        f"synthetic-update:{index}", messages, question, new, [1], [0],
        [old])


def distractor_example(rng: random.Random, index: int):
    target, other = rng.sample(NAMES, 2)
    target_city, other_city = rng.sample(CITIES, 2)
    target_food, other_food = rng.sample(FOODS, 2)
    if rng.random() < 0.5:
        messages = [
            memory_chunk(f"{target} lives in {target_city}."),
            memory_chunk(f"{other} lives in {other_city}."),
            memory_chunk(f"{other}'s favorite food is {other_food}."),
        ]
        question = f"Where does {target} live?"
        answer = target_city
    else:
        messages = [
            memory_chunk(f"{target}'s favorite food is {target_food}."),
            memory_chunk(f"{other}'s favorite food is {other_food}."),
            memory_chunk(f"{other} lives in {other_city}."),
        ]
        question = f"What is {target}'s favorite food?"
        answer = target_food
    return row(
        f"synthetic-distractor:{index}", messages, question, answer, [0],
        [1, 2])


def multihop_example(rng: random.Random, index: int):
    first, second = rng.sample(NAMES, 2)
    city = rng.choice(CITIES)
    relation = rng.choice(["brother", "sister", "friend", "cousin"])
    messages = [
        memory_chunk(f"{second} is {first}'s {relation}."),
        memory_chunk(f"{second} lives in {city}."),
    ]
    return row(
        f"synthetic-multihop:{index}", messages,
        f"Where does {first}'s {relation} live?", city, [0, 1])


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--samples-per-task", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.samples_per_task < 1:
        raise ValueError("--samples-per-task must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    tasks = [
        ("remember_explicit.jsonl", remember_example),
        ("update_explicit.jsonl", update_example),
        ("remember_distract.jsonl", distractor_example),
        ("reflect_explicit.jsonl", multihop_example),
    ]
    report = {}
    for filename, factory in tasks:
        rows = [
            factory(rng, index) for index in range(args.samples_per_task)]
        rng.shuffle(rows)
        write_jsonl(output / filename, rows)
        report[filename] = len(rows)
    print(json.dumps({"output": str(output), "files": report}, indent=2))


if __name__ == "__main__":
    main()
