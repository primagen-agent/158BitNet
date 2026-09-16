#!/usr/bin/env python3
"""Balanced natural-message memory worlds; no source wrappers or required dates."""
import argparse
import json
from pathlib import Path
import random

import prepare_repeated_dialogue_curriculum as catalog

CREATE = (
    "{name}'s {predicate} is {value}.",
    "For {name}, the {predicate} is {value}.",
    "{name} has {value} as the {predicate}.",
    "The {predicate} for {name} is {value}.",
    "{name} uses {value} for the {predicate}.",
    "{value} is the {predicate} chosen by {name}.",
)
UPDATE = (
    "{name} changed the {predicate} from {old} to {value}.",
    "{name}'s {predicate} is now {value}, replacing {old}.",
    "For {name}, the {predicate} was {old}; it is now {value}.",
    "The old {predicate} for {name} was {old}; the replacement is {value}.",
    "{name}'s {predicate} was {old}. The current entry is {value}.",
    "{name} has moved on from {old}; the {predicate} is {value}.",
)


def world(index, seed, names, predicates, split):
    rng = random.Random(seed + index * 10007)
    speakers = rng.sample(list(names), 2)
    relations = rng.sample(list(predicates), 2)
    facts = [(speakers[i % 2], relations[i // 2]) for i in range(4)]
    # A balanced mix of tiny and larger candidate banks.
    count = (1, 2, 4, 4)[index % 4]
    facts = facts[:count]
    chosen = [rng.sample(list(relation["values"]), 2) for _, relation in facts]
    messages, events = [], []
    for operation in (0, 1):
        for fact_index, (name, relation) in enumerate(facts):
            value = chosen[fact_index][operation]
            text = rng.choice(CREATE if operation == 0 else UPDATE).format(
                name=name, predicate=relation["name"], value=value, old=chosen[fact_index][0])
            # Most examples have no time. Do not synthesize a date for them.
            date = f"2025-04-{1 + len(events):02d}" if rng.random() < 0.2 else ""
            if date:
                text = f"On {date}, " + text
            episode = len(messages)
            messages.append(text)
            events.append({"episode": episode, "entity": name, "entity_surface": name,
                           "predicate": relation["name"], "predicate_surface": relation["name"],
                           "value": value, "value_surface": value, "time": date, "time_surface": date,
                           "operation": "create" if operation == 0 else "update",
                           "previous_episode": None if operation == 0 else fact_index,
                           "active": operation == 1})
    evidence = "\n".join(messages)
    world_id = f"natural-{split}-{index:05d}"
    rows = []
    for fact_index, (name, relation) in enumerate(facts):
        target_index = count + fact_index
        local = messages[target_index].rfind(chosen[fact_index][1])
        offset = sum(len(line) + 1 for line in messages[:target_index]) + local
        rows.append({"sample_id": f"{world_id}-{fact_index}", "evidence": evidence,
                     "question": f"What is {name}'s current {relation['name']}?",
                     "answer": chosen[fact_index][1], "answer_type": "span",
                     "answer_spans": [{"start": offset, "end": offset + len(chosen[fact_index][1]),
                                       "text": chosen[fact_index][1]}],
                     "metadata": {"world_id": world_id, "semantic_world_id": world_id,
                                  "raw_episodes": messages, "typed_events": events, "locomo_used": False,
                                  "query_plan": {"intent": "current", "targets": [{"entity": name, "predicate": relation["name"]}]}}})
    null = json.loads(json.dumps(rows[0]))
    null["sample_id"] = world_id + "-null"
    null["question"] = f"What is UnknownPerson{index}'s current {relations[0]['name']}?"
    null["answer"] = "No information available"
    null["answer_spans"] = []
    null["metadata"]["query_plan"] = {"intent": "null", "targets": []}
    null["metadata"]["counterfactual_no_info"] = True
    return rows + [null]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--train-worlds", type=int, default=192)
    parser.add_argument("--valid-worlds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2710916)
    parser.add_argument("--test-only", action="store_true",
                        help="generate fresh evaluation-only worlds, with seed-qualified identifiers")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for split, count, names, relations in (
        ("train", args.train_worlds, catalog.SPEAKERS, catalog.LARGE_PREDICATES),
        ("valid", args.valid_worlds, catalog.VALIDATION_SPEAKERS, catalog.VALIDATION_PREDICATES),
        ("test", args.valid_worlds, catalog.TEST_SPEAKERS, catalog.TEST_PREDICATES),
    ):
        if args.test_only and split != "test":
            continue
        path = root / f"{split}.jsonl"
        if path.exists():
            raise FileExistsError(f"refusing to replace frozen curriculum: {path}")
        with path.open("w") as handle:
            for index in range(count):
                for row in world(index, args.seed, names, relations, split):
                    if args.test_only:
                        previous = row["metadata"]["world_id"]
                        unique = previous + f"-seed{args.seed}"
                        row["sample_id"] = row["sample_id"].replace(previous, unique, 1)
                        row["metadata"]["world_id"] = unique
                        row["metadata"]["semantic_world_id"] = unique
                        row["evaluation_only"] = True
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (root / "manifest.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
