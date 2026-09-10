#!/usr/bin/env python3
"""Build balanced write/update/delete/ignore routing curricula.

Training and validation use disjoint instruction templates and disjoint
synthetic values. LoCoMo is never read.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prepare_memory_v89_short_copy import (
    OOD_STORE_TEMPLATES,
    TRAIN_STORE_TEMPLATES,
)


ACTION_IDS = {
    "ignore": 0,
    "write": 1,
    "update": 2,
    "delete": 3,
}

TRAIN_IMPLICIT_WRITE_TEMPLATES = [
    "My {attribute} is {value}.",
    "I prefer {value} whenever choosing my {attribute}.",
    "For my {attribute}, {value} is my usual choice.",
    "{value} has long been my {attribute}.",
    "One stable thing about me is that my {attribute} is {value}.",
    "I live in {value}; that is my current home city.",
    "My long-term project codename is {value}.",
    "The device I normally use is called {value}.",
    "I always want replies about {attribute} in the style {value}.",
    "My standing preference for {attribute} is {value}.",
    "My established {attribute} remains {value}.",
    "Keep {value} as the persistent setting for my {attribute}.",
    "The enduring value of my {attribute} is {value}.",
    "{value} is already my established choice for {attribute}.",
]
OOD_IMPLICIT_WRITE_TEMPLATES = [
    "When {attribute} comes up, my enduring choice is {value}.",
    "You can treat {value} as my established {attribute}.",
    "The persistent setting I use for {attribute} is {value}.",
    "As a lasting personal detail, {value} is my {attribute}.",
]

TRAIN_UPDATE_TEMPLATES = [
    "Update my {attribute}: it is now {value}, not {old}.",
    "My {attribute} changed from {old} to {value}.",
    "Replace the old {attribute} value {old} with {value}.",
    "From now on, my {attribute} is {value}.",
    "Correction: use {value} as my {attribute}.",
    "I no longer use {old}; my current {attribute} is {value}.",
    "Revise the saved {attribute} to {value}.",
    "The latest value for my {attribute} should be {value}.",
    "Supersede the saved {attribute} entry: replace {old} by {value}.",
    "{old} is obsolete for my {attribute}; set it to {value}.",
    "Amend my {attribute} from {old} so the new value is {value}.",
    "Going forward, use {value} rather than {old} for my {attribute}.",
    "Remember {value} going forward and replace the old value {old} "
    "for {attribute}.",
]
OOD_UPDATE_TEMPLATES = [
    "Supersede {old} with {value} for my {attribute}.",
    "My newly established {attribute} is {value}; {old} is obsolete.",
    "Amend the persistent {attribute} entry so it reads {value}.",
    "Going forward, remember {value} instead of {old} for {attribute}.",
]

TRAIN_DELETE_TEMPLATES = [
    "Forget my {attribute}.",
    "Delete the saved value for my {attribute}.",
    "Remove my {attribute} from memory.",
    "Do not retain the personal detail about my {attribute}.",
    "Erase the stored {attribute} entry.",
    "My {attribute} is private now; discard it.",
    "Unset my remembered {attribute}.",
    "Withdraw the information about my {attribute}.",
]
OOD_DELETE_TEMPLATES = [
    "Purge the persistent record of my {attribute}.",
    "Revoke the memory entry concerning my {attribute}.",
    "The saved {attribute} detail must no longer be retained.",
    "Expunge what I previously shared about my {attribute}.",
]

TRAIN_IGNORE_TEMPLATES = [
    "What is the weather in {value} today?",
    "Explain the meaning of {topic}.",
    "Write a short poem about {topic}.",
    "Summarize this sentence: {value}.",
    "How do I install {value}?",
    "Translate {value} into Japanese.",
    "Hello, how are you?",
    "What did I previously ask you to remember?",
    "List three facts about {topic}.",
    "Calculate {number} plus {other_number}.",
    "I feel {value} today.",
    "The temporary test output is {value}.",
    "Use {value} only for this response.",
    "Can you compare {value} and {old}?",
    "The word remember contains eight letters.",
    "Describe how a memory allocator works.",
    "Explain what delete means in {topic}.",
    "How does {topic} store and update records?",
    "Use the word forget in an example sentence.",
    "Compare write and read operations in {topic}.",
    "What does persistent memory mean in {topic}?",
    "This morning I happened to eat {value}.",
    "Earlier today I briefly used {value}.",
    "For a moment I was thinking about {value}.",
    "Today the temporary status happens to be {value}.",
]
OOD_IGNORE_TEMPLATES = [
    "Give me a concise overview of {topic}.",
    "For this one answer only, format the output as {value}.",
    "I happened to eat {value} this morning.",
    "Does the phrase store this have a special technical meaning?",
    "Solve {number} multiplied by {other_number}.",
    "What information is currently in long-term memory?",
    "Explain how {topic} deletes obsolete entries.",
    "In {topic}, what is an update operation?",
]

ATTRIBUTES = [
    "preferred drink", "favorite season", "response language",
    "home city", "project codename", "editor theme",
    "weekend hobby", "default device", "meeting preference",
    "favorite music genre",
]
VALUE_WORDS = [
    "amber", "cedar", "cobalt", "coral", "delta", "ember",
    "fern", "indigo", "jade", "lilac", "maple", "navy",
    "ochre", "pearl", "quartz", "saffron", "teal", "violet",
]
TRAIN_IGNORE_TOPICS = [
    "photosynthesis", "Roman history", "distributed systems",
    "database indexing", "compiler optimization", "plate tectonics",
    "public-key cryptography", "classical music", "graph algorithms",
    "renewable energy", "computer networking", "probability theory",
]
OOD_IGNORE_TOPICS = [
    "ocean currents", "quantum computing", "medieval architecture",
    "protein folding", "operating systems", "monetary policy",
]


def fields_for(
    split: str, index: int, rng: random.Random,
) -> dict:
    salt = 100_000 if split == "valid" else 0
    value_index = salt + index
    value_word = rng.choice(VALUE_WORDS)
    old_word = rng.choice(VALUE_WORDS)
    while old_word == value_word:
        old_word = rng.choice(VALUE_WORDS)
    return {
        "attribute": rng.choice(ATTRIBUTES),
        "topic": (
            OOD_IGNORE_TOPICS if split == "valid"
            else TRAIN_IGNORE_TOPICS
        )[index % (
            len(OOD_IGNORE_TOPICS) if split == "valid"
            else len(TRAIN_IGNORE_TOPICS)
        )],
        "value": (
            f"{value_word}-"
            f"{value_index:06d}"
        ),
        "old": (
            f"{old_word}-"
            f"{value_index + 1:06d}"
        ),
        "number": 11 + value_index % 83,
        "other_number": 17 + value_index % 71,
    }


def action_templates(split: str, action: str):
    if split == "train":
        return {
            "write": (
                TRAIN_STORE_TEMPLATES,
                TRAIN_IMPLICIT_WRITE_TEMPLATES,
            ),
            "update": (TRAIN_UPDATE_TEMPLATES,),
            "delete": (TRAIN_DELETE_TEMPLATES,),
            "ignore": (TRAIN_IGNORE_TEMPLATES,),
        }[action]
    return {
        "write": (
            OOD_STORE_TEMPLATES,
            OOD_IMPLICIT_WRITE_TEMPLATES,
        ),
        "update": (OOD_UPDATE_TEMPLATES,),
        "delete": (OOD_DELETE_TEMPLATES,),
        "ignore": (OOD_IGNORE_TEMPLATES,),
    }[action]


def render_action(
    split: str, action: str, index: int,
    template_rng: random.Random, field_rng: random.Random,
) -> dict:
    fields = fields_for(split, index, field_rng)
    families = action_templates(split, action)
    family = families[index % len(families)]
    template = template_rng.choice(family)
    if "{payload}" in template:
        text = template.format(payload=fields["value"])
        policy = "explicit_write"
    else:
        text = template.format(**fields)
        policy = (
            "implicit_write" if action == "write" else action)
    return {
        "sample_id": f"{split}-{action}-{index:06d}",
        "text": text,
        "action": action,
        "label": ACTION_IDS[action],
        "metadata": {
            "policy": policy,
            "strict_ood": split == "valid",
            "locomo_used": False,
        },
    }


def write_split(
    output: Path, split: str, count_per_action: int, seed: int,
) -> None:
    rng = random.Random(seed)
    field_rng = random.Random(seed + 1_000_003)
    rows = [
        render_action(split, action, index, rng, field_rng)
        for action in ACTION_IDS
        for index in range(count_per_action)
    ]
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--train-per-action", type=int, default=2000)
    parser.add_argument("--valid-per-action", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.train_per_action < 1 or args.valid_per_action < 1:
        parser.error("sample counts must be positive")
    root = Path(args.output)
    write_split(
        root / "train.jsonl", "train",
        args.train_per_action, args.seed)
    write_split(
        root / "valid.jsonl", "valid",
        args.valid_per_action, args.seed + 1)
    print(json.dumps({
        "output": str(root),
        "train": args.train_per_action * len(ACTION_IDS),
        "valid": args.valid_per_action * len(ACTION_IDS),
        "balanced": True,
        "strict_ood": True,
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
