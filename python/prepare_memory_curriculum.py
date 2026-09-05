#!/usr/bin/env python3
"""Build deterministic staged memory-training data.

The first stage deliberately uses opaque values and held-out entity/value
pools.  A frozen language model cannot answer these questions from prior
knowledge, so success requires the native memory path.

Output layout:
    OUTPUT/train/reconstruction.jsonl
    OUTPUT/train/remember_explicit.jsonl
    OUTPUT/valid/reconstruction.jsonl
    OUTPUT/valid/remember_explicit.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path


ATTRIBUTES = [
    "archive code",
    "locker code",
    "project token",
    "access marker",
    "record key",
    "device label",
    "parcel code",
    "session tag",
]

STORE_TEMPLATES = [
    "Remember this exact fact: {name}'s {attribute} is {value}.",
    "Store this for later: the {attribute} belonging to {name} is {value}.",
    "Please retain the following value exactly. {name} has {attribute} {value}.",
    "Add this to memory: {name} -> {attribute} -> {value}.",
]

ACK_TEMPLATES = [
    "Stored.",
    "I will remember it.",
    "Noted for later.",
    "The value has been recorded.",
]

QUERY_TEMPLATES = [
    "What is {name}'s {attribute}? Reply with only the value.",
    "Recall the {attribute} for {name}. Give only the stored value.",
    "Which exact value was stored as {name}'s {attribute}? Answer only with it.",
    "Return {name}'s {attribute}, with no explanation.",
]

DISTRACTORS = [
    (
        "The weather has been unusually mild this week.",
        "Yes, it has been pleasant.",
    ),
    (
        "I watched a documentary about coral reefs yesterday.",
        "That sounds interesting.",
    ),
    (
        "The train was crowded during the morning commute.",
        "Rush hour can be uncomfortable.",
    ),
    (
        "I am considering learning how to make sourdough.",
        "That could be a rewarding project.",
    ),
    (
        "A new park opened near the river.",
        "It should be a nice place to visit.",
    ),
    (
        "My neighbor recently bought an electric bicycle.",
        "I hope they enjoy it.",
    ),
]


def chunk(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def opaque_value(index: int, split_salt: int) -> str:
    """Return a tokenizer-friendly value with no semantic prior."""
    alphabet = string.ascii_uppercase
    x = index * 7919 + split_salt * 104729 + 17
    letters = "".join(alphabet[(x // (26 ** i)) % 26] for i in range(3))
    digits = f"{(x * 37 + 113) % 10000:04d}"
    return f"{letters}-{digits}"


def entity(index: int, split_salt: int) -> str:
    # Synthetic names prevent accidental overlap with LoCoMo or common facts.
    return f"Person{split_salt:02d}{index:04d}"


def reconstruction_sample(
    index: int, split_salt: int, rng: random.Random
) -> dict:
    name = entity(index, split_salt)
    attribute = ATTRIBUTES[index % len(ATTRIBUTES)]
    value = opaque_value(index, split_salt)
    # Reconstruction is intentionally a direct exact-copy objective.
    store = (
        f"Memorize this record exactly: name={name}; "
        f"field={attribute}; value={value}."
    )
    query = (
        f"Reconstruct the value in the stored record for name={name}, "
        f"field={attribute}. Output only the value."
    )
    return {
        "sample_id": f"reconstruction-{split_salt}-{index:06d}",
        "messages": [
            chunk(store, rng.choice(ACK_TEMPLATES)),
            chunk(query, value),
        ],
        "query_turn_id": 1,
        "metadata": {
            "type": "reconstruction",
            "style": "opaque_exact",
            "stage": 1,
        },
    }


def remember_sample(index: int, split_salt: int, rng: random.Random) -> dict:
    name = entity(index + 1_000_000, split_salt)
    attribute = ATTRIBUTES[(index * 3 + 1) % len(ATTRIBUTES)]
    value = opaque_value(index + 1_000_000, split_salt)
    store = rng.choice(STORE_TEMPLATES).format(
        name=name, attribute=attribute, value=value
    )
    query = rng.choice(QUERY_TEMPLATES).format(
        name=name, attribute=attribute
    )
    return {
        "sample_id": f"remember-{split_salt}-{index:06d}",
        "messages": [
            chunk(store, rng.choice(ACK_TEMPLATES)),
            chunk(query, value),
        ],
        "query_turn_id": 1,
        "metadata": {
            "type": "remember",
            "style": "explicit_opaque",
            "stage": 1,
        },
    }


def multi_entity_sample(
    index: int, split_salt: int, rng: random.Random
) -> dict:
    attribute = ATTRIBUTES[(index * 5 + 2) % len(ATTRIBUTES)]
    first_name = entity(index + 2_000_000, split_salt)
    second_name = entity(index + 3_000_000, split_salt)
    first_value = opaque_value(index + 2_000_000, split_salt)
    second_value = opaque_value(index + 3_000_000, split_salt)
    ask_first = bool(index % 2)
    query_name = first_name if ask_first else second_name
    target = first_value if ask_first else second_value
    return {
        "sample_id": f"multi-entity-{split_salt}-{index:06d}",
        "messages": [
            chunk(
                f"Store this fact: {first_name}'s {attribute} is "
                f"{first_value}.",
                "Stored.",
            ),
            chunk(
                f"Also store this separate fact: {second_name}'s "
                f"{attribute} is {second_value}.",
                "Stored separately.",
            ),
            chunk(
                f"What is {query_name}'s {attribute}? "
                "Reply with only the exact value.",
                target,
            ),
        ],
        "query_turn_id": 2,
        "metadata": {
            "type": "remember",
            "style": "multi_entity_opaque",
            "stage": 2,
        },
    }


def distract_sample(
    index: int, split_salt: int, rng: random.Random
) -> dict:
    name = entity(index + 4_000_000, split_salt)
    attribute = ATTRIBUTES[(index * 7 + 3) % len(ATTRIBUTES)]
    value = opaque_value(index + 4_000_000, split_salt)
    messages = [
        chunk(
            rng.choice(STORE_TEMPLATES).format(
                name=name, attribute=attribute, value=value
            ),
            rng.choice(ACK_TEMPLATES),
        )
    ]
    for user, assistant in rng.sample(DISTRACTORS, 2):
        messages.append(chunk(user, assistant))
    messages.append(
        chunk(
            rng.choice(QUERY_TEMPLATES).format(
                name=name, attribute=attribute
            ),
            value,
        )
    )
    return {
        "sample_id": f"distract-{split_salt}-{index:06d}",
        "messages": messages,
        "query_turn_id": len(messages) - 1,
        "metadata": {
            "type": "remember",
            "style": "distract_opaque",
            "stage": 2,
        },
    }


def update_sample(index: int, split_salt: int, rng: random.Random) -> dict:
    name = entity(index + 5_000_000, split_salt)
    attribute = ATTRIBUTES[(index * 11 + 4) % len(ATTRIBUTES)]
    old_value = opaque_value(index + 5_000_000, split_salt)
    new_value = opaque_value(index + 6_000_000, split_salt)
    return {
        "sample_id": f"update-{split_salt}-{index:06d}",
        "messages": [
            chunk(
                f"Please remember that {name}'s {attribute} is "
                f"{old_value}.",
                "Stored.",
            ),
            chunk(
                f"Update the record: {name}'s {attribute} is now "
                f"{new_value}. Replace the old value.",
                "Updated.",
            ),
            chunk(
                f"What is {name}'s {attribute}? "
                "Reply with only the current value.",
                new_value,
            ),
        ],
        "query_turn_id": 2,
        "negative_answers": [old_value],
        "metadata": {
            "type": "update",
            "style": "explicit_opaque_contrastive",
            "stage": 3,
        },
    }


def forget_sample(index: int, split_salt: int, rng: random.Random) -> dict:
    name = entity(index + 7_000_000, split_salt)
    attribute = ATTRIBUTES[(index * 13 + 5) % len(ATTRIBUTES)]
    old_value = opaque_value(index + 7_000_000, split_salt)
    return {
        "sample_id": f"forget-{split_salt}-{index:06d}",
        "messages": [
            chunk(
                f"Please remember that {name}'s {attribute} is "
                f"{old_value}.",
                "Stored.",
            ),
            chunk(
                f"Delete {name}'s {attribute} from memory. The value is "
                "now unset and must not be recalled.",
                "Deleted.",
            ),
            chunk(
                f"What is {name}'s {attribute}? If it was deleted, reply "
                "exactly: No information available.",
                "No information available",
            ),
        ],
        "query_turn_id": 2,
        "negative_answers": [old_value],
        "metadata": {
            "type": "forget",
            "style": "explicit_opaque_contrastive",
            "stage": 3,
        },
    }


def memory_irrelevant_sample(
    index: int, split_salt: int, rng: random.Random
) -> dict:
    name = entity(index + 8_000_000, split_salt)
    attribute = ATTRIBUTES[(index * 17 + 6) % len(ATTRIBUTES)]
    value = opaque_value(index + 8_000_000, split_salt)
    first = f"marker{split_salt:02d}{index:04d}"
    second = f"answer{split_salt:02d}{index:04d}"
    return {
        "sample_id": f"task4-normal-{split_salt}-{index:06d}",
        "messages": [
            chunk(
                rng.choice(STORE_TEMPLATES).format(
                    name=name, attribute=attribute, value=value
                ),
                rng.choice(ACK_TEMPLATES),
            ),
            chunk(
                f"In the pair '{first} {second}', return only the word "
                f"that follows {first}.",
                second,
            ),
        ],
        "query_turn_id": 1,
        "metadata": {
            "type": "normal",
            "style": "memory_irrelevant",
            "v2_task": "task4_normal",
            "stage": 3,
        },
    }


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_split(
    output: Path, split: str, count: int, stage2_count: int,
    stage3_count: int, seed: int, split_salt: int
) -> None:
    rng = random.Random(seed + split_salt)
    if count:
        reconstruction = [
            reconstruction_sample(i, split_salt, rng) for i in range(count)
        ]
        remember = [
            remember_sample(i, split_salt, rng) for i in range(count)
        ]
        write_jsonl(
            output / split / "reconstruction.jsonl", reconstruction)
        write_jsonl(
            output / split / "remember_explicit.jsonl", remember)
    if stage2_count:
        multi_entity = [
            multi_entity_sample(i, split_salt, rng)
            for i in range(stage2_count)
        ]
        for sample in multi_entity:
            sample["metadata"]["v2_task"] = "task3_multi_entity"
        distract = [
            distract_sample(i, split_salt, rng)
            for i in range(stage2_count)
        ]
        write_jsonl(output / split / "multi_entity.jsonl", multi_entity)
        write_jsonl(output / split / "remember_distract.jsonl", distract)
    if stage3_count:
        updates = [
            update_sample(i, split_salt, rng)
            for i in range(stage3_count)
        ]
        forgets = [
            forget_sample(i, split_salt, rng)
            for i in range(stage3_count)
        ]
        normal = [
            memory_irrelevant_sample(i, split_salt, rng)
            for i in range(stage3_count)
        ]
        write_jsonl(output / split / "update_explicit.jsonl", updates)
        write_jsonl(output / split / "forget_explicit.jsonl", forgets)
        write_jsonl(output / split / "task4_normal.jsonl", normal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--valid", type=int, default=128)
    parser.add_argument(
        "--stage2-train", type=int, default=0,
        help="multi-entity and distract samples per training stratum")
    parser.add_argument(
        "--stage2-valid", type=int, default=0,
        help="multi-entity and distract samples per validation stratum")
    parser.add_argument(
        "--stage3-train", type=int, default=0,
        help="contrastive update and forget samples per training stratum")
    parser.add_argument(
        "--stage3-valid", type=int, default=0,
        help="contrastive update and forget samples per validation stratum")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if min(args.train, args.valid, args.stage2_train, args.stage2_valid,
           args.stage3_train, args.stage3_valid) < 0:
        parser.error("sample counts must be non-negative")
    if args.train + args.stage2_train + args.stage3_train < 1:
        parser.error("training sample counts must be positive")
    if args.valid + args.stage2_valid + args.stage3_valid < 1:
        parser.error("validation sample counts must be positive")

    output = Path(args.output)
    build_split(
        output, "train", args.train, args.stage2_train,
        args.stage3_train, args.seed, split_salt=11)
    build_split(
        output, "valid", args.valid, args.stage2_valid,
        args.stage3_valid, args.seed, split_salt=29)
    print(
        json.dumps(
            {
                "output": str(output),
                "train": args.train * 2,
                "valid": args.valid * 2,
                "stage2_train": args.stage2_train * 2,
                "stage2_valid": args.stage2_valid * 2,
                "stage3_train": args.stage3_train * 3,
                "stage3_valid": args.stage3_valid * 3,
                "train_valid_entity_overlap": 0,
                "stage": (
                    "reconstruction_remember_multi_entity_distract_"
                    "update_forget_contrastive"
                    if args.stage3_train or args.stage3_valid
                    else (
                        "reconstruction_remember_multi_entity_distract"
                        if args.stage2_train or args.stage2_valid
                        else "reconstruction_and_explicit_remember"
                    )
                ),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
