"""Paired causal controls for memory-to-generation, not automatic-memory data."""
import argparse
import hashlib
import json
from pathlib import Path
import random

NAMES = {"train": ["Morgan", "Avery", "Blake", "Cameron", "Dakota", "Emerson", "Finley", "Harper"],
         "valid": ["Lennon", "Marley", "Nico", "Oakley"]}
TRAIN_ENTITY_VARIANTS = """Ada Alan Alex Alice Amy Andrew Anna Arthur Ben Beth
Brian Carl Carol Clara David Diana Dylan Edward Emma Eric Eva Felix Fiona Fred
George Grace Hannah Henry Hugh Ian Iris Jack James Jane Jason Jean John Julia
Kate Kelly Laura Leo Lily Lisa Louis Lucy Maria Mark Mary Max Mia Mike Nina
Noah Nora Oliver Oscar Owen Paul Peter Rachel Ray Rose Ruby Ryan Sam Sara
Sean Simon Sophia Stella Susan Theo Thomas Tina Tom Tony Vera Victor Will Zoe""".split()
TASKS = (
    ("home city", ["Lima", "Oslo", "Paris", "Tokyo", "Berlin", "Rome", "London", "Boston"],
     "Where does {name} live?", "Which city does {name} live in?", "{name} lives in {value}."),
    ("favorite drink", ["tea", "coffee", "milk", "water", "juice", "cocoa", "lemonade", "soda"],
     "What does {name} like to drink?", "What is {name}'s favorite drink?", "{name} prefers {value}."),
)


def make_rows(split, worlds, seed, diverse_entities=False):
    if split not in NAMES or worlds < 1:
        raise ValueError("invalid split or world count")
    rows = []
    for i in range(worlds):
        rng = random.Random(seed + i * 1009 + (0 if split == "train" else 1000003))
        pool = TRAIN_ENTITY_VARIANTS if diverse_entities and split == "train" else NAMES[split]
        name, other = rng.sample(pool, 2)
        relation, values, train_query, valid_query, answer = TASKS[i % len(TASKS)]
        first, second, distractor = rng.sample(values, 3)
        prompt = (train_query if split == "train" else valid_query).format(name=name)
        group = f"fusion-{split}-{i:04d}"
        common = {"group": group, "split": split, "query": prompt,
                  "oracle_episode_boundaries": True, "locomo_used": False,
                  "automatic_memory": False, "value_vocabulary": values}
        for label, value, rival in (("original", first, second), ("swapped", second, first)):
            rows.append({**common, "id": group + "-" + label, "condition": label,
                         "memory": [f"{name}'s {relation} is {value}."],
                         "answer": answer.format(name=name, value=value),
                         "expected_value": value, "forbidden_value": rival})
        rows.append({**common, "id": group + "-removed", "condition": "removed",
                     "memory": [f"{other}'s {relation} is {distractor}."],
                     "answer": f"I don't know {name}'s {relation} yet.",
                     "expected_value": None, "forbidden_value": first})
        rows.append({**common, "id": group + "-distractor", "condition": "distractor",
                     "memory": [f"{other}'s {relation} is {distractor}.", f"{name}'s {relation} is {first}."],
                     "answer": answer.format(name=name, value=first),
                     "expected_value": first, "forbidden_value": distractor})
    return rows


def prepare(output, train_worlds=32, valid_worlds=8, seed=3180918, diverse_entities=False):
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "MEMORY_FUSION_CAUSAL_CURRICULUM_V1", "seed": seed,
                "diverse_entities": diverse_entities,
                "oracle_episode_boundaries": True, "automatic_memory": False,
                "locomo_used": False, "splits": {}}
    for split, count in (("train", train_worlds), ("valid", valid_worlds)):
        rows = make_rows(split, count, seed, diverse_entities)
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (root / f"{split}.jsonl").write_bytes(data)
        manifest["splits"][split] = {"worlds": count, "rows": len(rows), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output"); p.add_argument("--train-worlds", type=int, default=32)
    p.add_argument("--valid-worlds", type=int, default=8); p.add_argument("--seed", type=int, default=3180918)
    p.add_argument("--diverse-entities", action="store_true", help="training-only entity diversity; validation bytes stay fixed")
    a = p.parse_args(); print(json.dumps(prepare(a.output, a.train_worlds, a.valid_worlds, a.seed, a.diverse_entities)))
