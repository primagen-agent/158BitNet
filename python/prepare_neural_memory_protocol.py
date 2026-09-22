"""Generate/verify synthetic P1 smoke data, never a final accuracy benchmark.

World families, input text, and supervision are separate. Semantic fingerprints
come from the generator's world specification, not an NLP deduplicator.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


SCENARIOS = (
    "original", "paraphrase", "swapped_values", "same_value_other_person",
    "wrong_subject", "wrong_relation", "target_removed", "all_irrelevant",
    "distractor_added", "distractor_reordered", "multiple_targets",
    "role_forward", "role_reverse", "hypothetical", "negated", "quoted",
    "update_current", "update_history", "pronoun", "alias", "empty", "ordinary_chat",
)
FORMAT = "neural-memory-p1-smoke-v1"
SOURCE_FILES = ("python/prepare_neural_memory_protocol.py", "python/episode_memory_inputs.py")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def semantic_key(facts):
    # Ignore surface text, annotation IDs, episode ordering and duplicate renderings.
    return digest(sorted({canonical(fact) for fact in facts}))


def fact(subject, relation, value, time="current", status="actual"):
    return {"subject": subject, "relation": relation, "value": value, "time": time, "status": status}


def message(text, role="user"):
    return {"role": role, "speaker": "user" if role == "user" else role, "text": text}


def make_cases(world, split):
    a, b, city, other, drink = (world[k] for k in ("a", "b", "city", "other_city", "drink"))
    lineage = [fact(a, "home_city", city), fact(b, "home_city", other), fact(a, "drink", drink)]
    for language in ("en", "zh"):
        zh = language == "zh"
        home = lambda person, place: (f"{person}住在{place}。" if zh else f"{person} lives in {place}.")
        ask = lambda person: (f"{person}现在住在哪里？" if zh else f"Where does {person} live now?")
        source_a = (home(a, city), fact(a, "home_city", city))
        source_b = (home(b, other), fact(b, "home_city", other))
        drink_source = (f"{a}最喜欢的饮料是{drink}。" if zh else f"{a}'s favorite drink is {drink}.", fact(a, "drink", drink))
        for scenario in SCENARIOS:
            events, query, targets, claims, missing = [source_a, source_b], ask(a), [0], [source_a[1]], False
            context = []
            if scenario == "paraphrase":
                events[0] = (f"{city}是{a}目前居住的城市。" if zh else f"{city} is the city {a} currently calls home.", source_a[1])
            elif scenario == "swapped_values":
                events = [(home(a, other), fact(a, "home_city", other)), (home(b, city), fact(b, "home_city", city))]
                claims = [events[0][1]]
            elif scenario == "same_value_other_person":
                events[1] = (home(b, city), fact(b, "home_city", city))
            elif scenario in ("wrong_subject", "target_removed", "all_irrelevant"):
                events, targets, claims, missing = [source_b], [], [], True
                if scenario == "all_irrelevant": events.append((f"{b}喜欢{drink}。" if zh else f"{b} likes {drink}.", fact(b, "drink", drink)))
            elif scenario == "wrong_relation":
                events, targets, claims, missing = [drink_source], [], [], True
            elif scenario == "distractor_added": events.append(drink_source)
            elif scenario == "distractor_reordered": events, targets = [drink_source, source_b, source_a], [2]
            elif scenario == "multiple_targets":
                query = f"{a}和{b}分别住在哪座城市？" if zh else f"Which cities do {a} and {b} each live in?"
                targets, claims = [0, 1], [source_a[1], source_b[1]]
            elif scenario in ("role_forward", "role_reverse"):
                lender, borrower = (a, b) if scenario == "role_forward" else (b, a)
                events = [(f"{lender}昨天把书借给了{borrower}。" if zh else f"{lender} lent a book to {borrower} yesterday.",
                           fact(lender, "lent_book_to", borrower, "yesterday"))]
                query = "昨天是谁把书借给了谁？" if zh else "Who lent a book to whom yesterday?"
                claims = [events[0][1]]
            elif scenario in ("hypothetical", "negated", "quoted"):
                texts = {
                    "hypothetical": f"假如{a}住在{city}，通勤可能更方便。" if zh else f"If {a} lived in {city}, commuting might be easier.",
                    "negated": f"{a}不住在{city}。" if zh else f"{a} does not live in {city}.",
                    "quoted": f"{b}声称{a}住在{city}，但我没有确认。" if zh else f"{b} claims {a} lives in {city}, but I have not confirmed it.",
                }
                events = [(texts[scenario], fact(a, "home_city", city, status=scenario))]
                # Relevant evidence is not the same as an affirmed positive fact.
                claims, missing = [], True
            elif scenario in ("update_current", "update_history"):
                events = [(f"2020年，{a}住在{city}。" if zh else f"In 2020, {a} lived in {city}.", fact(a, "home_city", city, "2020")),
                          (f"{a}在2024年搬到了{other}，现在仍住在那里。" if zh else f"{a} moved to {other} in 2024 and still lives there.", fact(a, "home_city", other))]
                targets = [1] if scenario == "update_current" else [0]
                if scenario == "update_history": query = f"{a}在2020年住在哪里？" if zh else f"Where did {a} live in 2020?"
                claims = [events[targets[0]][1]]
            elif scenario == "pronoun":
                context = [message(f"我们接下来聊聊{a}。" if zh else f"Let's talk about {a} next.")]
                query = "这个人现在住在哪里？" if zh else "Where does this person live now?"
            elif scenario == "alias":
                alias = world["alias"]
                events.append((f"{a}也叫{alias}。" if zh else f"{a} also goes by {alias}.", fact(a, "alias", alias)))
                query, targets = ask(alias), [0, 2]
            elif scenario == "empty": events, targets, claims, missing = [], [], [], True
            elif scenario == "ordinary_chat":
                query, targets = ("二加二等于多少？" if zh else "What is two plus two?"), []
                claims = [fact("2+2", "equals", "4", "timeless")]
            case_id = f"{world['id']}-{scenario}-{language}"
            runtime = {"id": case_id, "context": context + [message(query)], "episodes": [message(text) for text, _ in events]}
            label = {"id": case_id, "relevant_episode_indices": targets, "required_claims": claims,
                     "allowed_claims": claims + ([events[0][1]] if scenario in ("hypothetical", "negated", "quoted") else []),
                     "evidence_missing": missing, "memory_needed": scenario != "ordinary_chat"}
            index = {"id": case_id, "world_id": world["id"], "split": split, "scenario": scenario,
                     "language": language, "family_facts": lineage, "episode_facts": [f for _, f in events]}
            yield runtime, label, index


def validate_records(inputs, labels, index):
    # Import here so specification/hash tools remain dependency-light.
    from episode_memory_inputs import validate_input
    maps = []
    for rows in (inputs, labels, index):
        mapping = {row["id"]: row for row in rows}
        if len(mapping) != len(rows) or not rows: raise ValueError("empty or duplicate case IDs")
        maps.append(mapping)
    if maps[0].keys() != maps[1].keys() or maps[0].keys() != maps[2].keys(): raise ValueError("sidecar ID mismatch")
    seen = {"world": {}, "family": {}, "events": {}, "fact": {}}
    counts = Counter()
    for case_id, runtime in maps[0].items():
        validate_input(runtime)
        label, meta = maps[1][case_id], maps[2][case_id]
        split = meta["split"]
        if split not in ("train", "dev"): raise ValueError("only synthetic train/dev smoke is permitted")
        if len(meta["episode_facts"]) != len(runtime["episodes"]): raise ValueError("episode evidence count mismatch")
        targets = label["relevant_episode_indices"]
        if not isinstance(targets, list) or len(set(targets)) != len(targets) or any(type(i) is not int or not 0 <= i < len(runtime["episodes"]) for i in targets):
            raise ValueError("target has no source episode")
        keys = [("world", meta["world_id"]), ("family", semantic_key(meta["family_facts"]))]
        if meta["episode_facts"]: keys.append(("events", semantic_key(meta["episode_facts"])))
        keys.extend(("fact", digest(f)) for f in meta["episode_facts"])
        for kind, key in keys:
            previous = seen[kind].setdefault(key, split)
            if previous != split: raise ValueError(f"cross-split {kind} leakage")
        counts[(split, meta["scenario"], meta["language"])] += 1
    return {"cases": len(inputs), "worlds": len(seen["world"]),
            "groups": {"/".join(k): v for k, v in sorted(counts.items())}}


def generate(config):
    if set(config) != {"format", "seed", "train_worlds", "dev_worlds", "worlds"} or config["format"] != FORMAT:
        raise ValueError("wrong smoke configuration")
    worlds = config["worlds"]
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("nonnegative integer seed required")
    fields = {"id", "a", "b", "alias", "city", "other_city", "drink"}
    if not isinstance(worlds, list) or any(not isinstance(w, dict) or set(w) != fields or
                                         any(not isinstance(v, str) or not v.strip() for v in w.values()) for w in worlds):
        raise ValueError("complete semantic world specification required")
    if any(w["a"] == w["b"] or w["city"] == w["other_city"] for w in worlds):
        raise ValueError("distinct subjects and counterfactual values required")
    if any(type(config[k]) is not int or config[k] < 1 for k in ("train_worlds", "dev_worlds")):
        raise ValueError("positive world counts required")
    if len(worlds) != config["train_worlds"] + config["dev_worlds"] or len({w["id"] for w in worlds}) != len(worlds):
        raise ValueError("invalid world inventory")
    shuffled = list(worlds); random.Random(config["seed"]).shuffle(shuffled)
    inputs, labels, index = [], [], []
    for position, world in enumerate(shuffled):
        split = "train" if position < config["train_worlds"] else "dev"
        for runtime, label, meta in make_cases(world, split):
            inputs.append(runtime); labels.append(label); index.append(meta)
    report = validate_records(inputs, labels, index)
    files = {}
    for split in ("train", "dev"):
        ids = {m["id"] for m in index if m["split"] == split}
        for name, rows in (("inputs", inputs), ("labels", labels), ("index", index)):
            files[f"{split}.{name}.jsonl"] = "".join(canonical(row) + "\n" for row in rows if row["id"] in ids).encode()
    return files, report


def materialize(config_path, output, verify=False):
    config_path, output = Path(config_path), Path(output)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    files, report = generate(config)
    repo = Path(__file__).resolve().parents[1]
    manifest = {"format": FORMAT, "purpose": "protocol_smoke_not_promotion", "seed": config["seed"],
                "locomo_used": False, "automatic_memory": False, "supplied_episode_boundaries": True,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "source_sha256": {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in SOURCE_FILES},
                "file_sha256": {p: hashlib.sha256(data).hexdigest() for p, data in files.items()},
                "split_worlds": {s: config[f"{s}_worlds"] for s in ("train", "dev")}, **report}
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if verify:
        if {p.name for p in output.iterdir()} != set(files):
            raise ValueError("unexpected/missing files in frozen corpus")
        for name, data in files.items():
            if (output / name).read_bytes() != data: raise ValueError(f"regeneration/hash mismatch: {name}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in files.items(): (output / name).write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = materialize(args.config, args.output, args.verify)
    print(json.dumps({k: manifest[k] for k in ("purpose", "cases", "worlds", "split_worlds", "file_sha256")}, indent=2))


if __name__ == "__main__": main()
