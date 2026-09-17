#!/usr/bin/env python3
"""Synthetic resident-slot activation curriculum, with a sealed test split.

Gold value spans are diagnostic pointer payloads, not model inputs. No source
from LoCoMo is used. A world is an independent memory state, never parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

NAMES = {
    "train": "Avery Blake Cameron Dakota Emerson Finley Harper Jordan".split(),
    "valid": "Lennon Marley Nico Oakley Phoenix Reese Shiloh Winter".split(),
    "test": "Arden Briar Cleo Devin Ellis Frankie Gale Hollis".split(),
}
RELATIONS = ("home city", "workplace", "favorite sport", "preferred language", "evening course", "weekend hobby")
FORMS = {
    "train": ("{name}'s {relation} is {value}.", "For {name}, the {relation} is {value}."),
    "valid": ("The {relation} associated with {name} is {value}.",),
    "test": ("{name} reports {value} as their {relation}.",),
}
UPDATES = {
    "train": "{name} changed the {relation} from {old} to {value}.",
    "valid": "{name}'s {relation} was {old}; it is now {value}.",
    "test": "For {name}, {value} replaces {old} as the {relation}.",
}

# Training-only paraphrases. Development and sealed-test renderers are unchanged.
TRAIN_LANGUAGE = {
    "fact": (
        "{name} has {value} for {relation}.",
        "According to {name}, their {relation} is {value}.",
        "A note about {name}: {relation} = {value}.",
        "{value} is listed as {name}'s {relation}.",
        "The recorded {relation} for {name} is {value}.",
        "When asked about {relation}, {name} answered {value}.",
    ),
    "update": (
        "Previously {name}'s {relation} was {old}. Now it is {value}.",
        "{name} updated their {relation}: {old} is out of date; {value} is current.",
        "The latest {relation} for {name} is {value}, replacing {old}.",
        "Before the update {name} had {old} as {relation}; afterward they have {value}.",
        "{name} no longer has {old} for {relation}. Use {value} instead.",
        "An update from {name}: my {relation} changed, and {value} supersedes {old}.",
    ),
    "current": (
        "Tell me {name}'s {relation} as it stands today.",
        "What {relation} is in effect for {name} now?",
        "After any changes, what is {name}'s {relation}?",
        "I need only the newest {relation} for {name}.",
        "What {relation} is currently recorded for {name}?",
        "For {name}, state the {relation} that applies at present.",
    ),
    "history": (
        "Tell me both the earlier and the latest {relation} for {name}.",
        "Compare {name}'s {relation} before versus after the update.",
        "What was {name}'s {relation}, and what is it now?",
        "Report the old and new values of {name}'s {relation}.",
        "For {name}, list the {relation} originally recorded and its replacement.",
        "I need two versions of {name}'s {relation}: the prior one and the present one.",
    ),
    "two_relations": (
        "Tell me {name}'s latest {first} plus their latest {second}.",
        "For {name}, report the {first} and {second} currently in effect.",
        "What does {name} have for {first} and for {second} at present?",
        "I need two current facts about {name}: {first}, {second}.",
        "State both {first} and {second} for {name}, using the newest values.",
        "After any updates, what are {name}'s {first} and {second}?",
    ),
    "two_people": (
        "Tell me the latest {relation} for {first}, then for {second}.",
        "What do {first} and {second} each have for {relation} at present?",
        "I need the current {relation} of two people: {first}, {second}.",
        "State {first}'s {relation} and {second}'s {relation} as they stand today.",
        "After any updates, what {relation} applies to {first} and to {second}?",
        "Report both people's newest {relation}: {first} and {second}.",
    ),
    "four_facts": (
        "Tell me {first}'s and {second}'s latest {r1} and {r2}.",
        "For each person, {first} and {second}, state the current {r1} plus {r2}.",
        "I need four current facts: the {r1} and {r2} of {first} and of {second}.",
        "What do {first} and {second} each have for {r1} and {r2} at present?",
        "Report {first}'s {r1}, {first}'s {r2}, {second}'s {r1} and {second}'s {r2}, all up to date.",
        "After any updates, list both {r1} and {r2} separately for {first} and {second}.",
    ),
}


def make_world(index, split, seed, diverse_train=False, supervision=None, query_supervision=False):
    if query_supervision and supervision is None: raise ValueError("query supervision requires annotation output")
    rng = random.Random(seed + index * 10007 + ("train", "valid", "test").index(split) * 1000003)
    language_rng = random.Random(seed + index * 20011 + 2850916)
    def language(base, kind, **fields):
        template = language_rng.choice((base,) + TRAIN_LANGUAGE[kind]) if diverse_train and split == "train" else base
        return template.format(**fields)
    names = rng.sample(NAMES[split], 4)
    if supervision is not None:
        if split != "train": raise ValueError("entity supervision is training-only")
        supervision.update({"entities": names.copy(), "event_keys": []})
    relations = rng.sample(RELATIONS, 3)
    events, current, initial = [], {}, {}
    def write(person, relation, update=False):
        key = (person, relation)
        # Random source-specific values prevent answer memorization across worlds.
        value = f"{split}-{rng.getrandbits(32):08x}"
        old = events[current[key]]["value"] if update else ""
        template = UPDATES[split] if update else rng.choice(FORMS[split])
        text = language(template, "update" if update else "fact", name=person, relation=relation, value=value, old=old)
        start = len(text[:text.index(value)].encode())
        event = {"text": text, "value": value, "value_start": start,
                 "value_end": start + len(value.encode()), "source_id": f"event-{len(events)}"}
        events.append(event)
        if supervision is not None: supervision["event_keys"].append([person, relation])
        current[key] = len(events) - 1
        initial.setdefault(key, len(events) - 1)
    keys = [(n, r) for n in names[:2] for r in relations[:2]]
    rng.shuffle(keys)
    for n, r in keys: write(n, r)
    updated = [(names[0], relations[0]), (names[1], relations[0])]
    rng.shuffle(updated)
    for n, r in updated: write(n, r, True)
    # Recency alone must not select the answer: unrelated facts can come last.
    for r in rng.sample(list(RELATIONS), rng.randrange(1, 4)):
        write(names[2], r)
    queries, specifications = [], {}
    def query(kind, text, targets, requested, form):
        item = {"kind": kind, "text": text, "targets": sorted(targets),
                "answers": [events[i]["value"] for i in sorted(targets)]}
        queries.append(item)
        if query_supervision: specifications[id(item)] = {"keys": [list(k) for k in requested], "form": form}
    for n, r in keys:
        template = {"train": "What is {name}'s current {relation}?",
                    "valid": "Which {relation} does {name} have now?",
                    "test": "Give the latest {relation} reported for {name}."}[split]
        query("current", language(template, "current", name=n, relation=r), [current[(n, r)]], [(n, r)], "current")
    for n in names[:2]:
        text = {"train": f"What are {n}'s current {relations[0]} and {relations[1]}?",
                "valid": f"Give both the {relations[0]} and the {relations[1]} that {n} has now.",
                "test": f"List the latest {relations[0]} together with the {relations[1]} for {n}."}[split]
        text = language(text, "two_relations", name=n, first=relations[0], second=relations[1])
        query("multi", text, [current[(n, r)] for r in relations[:2]], [(n, r) for r in relations[:2]], "two_relations")
    text = {"train": f"What is the current {relations[0]} for both {names[0]} and {names[1]}?",
            "valid": f"Give the {relations[0]} now associated with {names[0]} and with {names[1]}.",
            "test": f"List the latest {relations[0]} of {names[0]} alongside that of {names[1]}."}[split]
    text = language(text, "two_people", relation=relations[0], first=names[0], second=names[1])
    query("multi", text, [current[(n, relations[0])] for n in names[:2]], [(n, relations[0]) for n in names[:2]], "two_people")
    for n, r in updated:
        text = {"train": f"What were {n}'s previous and current {r}?",
                "valid": f"Give the {r} of {n} both before and after the change.",
                "test": f"List {n}'s old {r} followed by the new one."}[split]
        text = language(text, "history", name=n, relation=r)
        query("history", text, [initial[(n, r)], current[(n, r)]], [(n, r)], "history")
    text = {"train": f"What are the current {relations[0]} and {relations[1]} for both {names[0]} and {names[1]}?",
            "valid": f"For {names[0]} and {names[1]}, give both people's {relations[0]} and {relations[1]} now.",
            "test": f"List the latest {relations[0]} and {relations[1]} of {names[0]} as well as {names[1]}."}[split]
    text = language(text, "four_facts", first=names[0], second=names[1], r1=relations[0], r2=relations[1])
    query("multi", text, [current[k] for k in keys], [(n, r) for n in names[:2] for r in relations[:2]], "four_facts")
    for n, r in ((names[3], relations[0]), (names[0], relations[2])):
        text = {"train": f"What is {n}'s current {r}?",
                "valid": f"Which {r} does {n} have now?",
                "test": f"Give the latest {r} reported for {n}."}[split]
        query("null", language(text, "current", name=n, relation=r), [], [(n, r)], "current")
    rng.shuffle(queries)
    if query_supervision: supervision["query_specs"] = [specifications[id(q)] for q in queries]
    return {"world_id": f"slot-{split}-{seed}-{index:05d}", "split": split,
            "evaluation_only": split == "test", "locomo_used": False,
            "events": events, "queries": queries}


def prepare(output, train_worlds=128, valid_worlds=24, seed=2810916, diverse_train=False):
    root = Path(output)
    if root.exists(): raise FileExistsError("use a fresh curriculum directory")
    if min(train_worlds, valid_worlds) < 1: raise ValueError("empty split")
    root.mkdir(parents=True)
    manifest = {"format": "RESIDENT_MEMORY_SET_CURRICULUM_V1", "seed": seed,
                "locomo_used": False, "pointer_boundaries": "oracle_diagnostic_only", "splits": {},
                "diverse_train_language": diverse_train}
    for split, count in (("train", train_worlds), ("valid", valid_worlds), ("test", valid_worlds)):
        rows, supervision = [], {}
        for i in range(count):
            annotation = {} if split == "train" else None
            row = make_world(i, split, seed, diverse_train, annotation)
            rows.append(row)
            if annotation is not None: supervision[row["world_id"]] = annotation
        path = root / (split + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["splits"][split] = {"worlds": count, "questions": sum(len(r["queries"]) for r in rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "evaluation_only": split == "test"}
        if split == "train":
            sidecar = root / "train_supervision.json"
            sidecar.write_text(json.dumps({"format": "RESIDENT_DISTRACTOR_SUPERVISION_V1", "training_only": True,
                "train_sha256": manifest["splits"][split]["sha256"], "worlds": supervision}, sort_keys=True) + "\n")
            manifest["training_supervision_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output"); p.add_argument("--train-worlds", type=int, default=128)
    p.add_argument("--valid-worlds", type=int, default=24); p.add_argument("--seed", type=int, default=2810916)
    p.add_argument("--diverse-train", action="store_true")
    a = p.parse_args()
    print(json.dumps(prepare(a.output, a.train_worlds, a.valid_worlds, a.seed, a.diverse_train)))
