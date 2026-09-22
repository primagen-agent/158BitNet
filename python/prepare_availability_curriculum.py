"""Synthetic single-event component curriculum, never LoCoMo or final acceptance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from prepare_neural_memory_protocol import canonical, digest, fact, message, validate_records


FORMAT = "availability-curriculum-v1"
SUPPORTED = ("original", "swapped_values", "paraphrase", "update_current")
MISSING = ("empty", "wrong_subject", "wrong_relation", "conflict", "hypothetical", "negated", "quoted", "historical_only")
NORMAL = ("ordinary_empty", "ordinary_original", "ordinary_wrong_subject", "ordinary_conflict")
SCENARIOS = SUPPORTED + MISSING + NORMAL


def worlds_from_config(config):
    fields = {"format", "seed", "train_worlds", "dev_worlds", "people", "cities", "excluded_subjects", "state_weights"}
    if set(config) != fields or config["format"] != FORMAT: raise ValueError("invalid curriculum format")
    for key in ("seed", "train_worlds", "dev_worlds"):
        if type(config[key]) is not int or config[key] < 1: raise ValueError("positive integer configuration required")
    for key in ("people", "cities", "excluded_subjects"):
        values = config[key]
        if not isinstance(values, list) or any(type(v) is not str or not v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError("unique explicit string inventory required")
    count = config["train_worlds"] + config["dev_worlds"]
    if len(config["people"]) != 2 * count or len(config["cities"]) < 2: raise ValueError("incomplete world inventory")
    if set(config["people"]) & set(config["excluded_subjects"]): raise ValueError("old diagnostic subject reused")
    if config["state_weights"] != {"no_memory_needed": 2., "supported": 2., "insufficient": 1.}:
        raise ValueError("predeclared state weights changed")
    cities = config["cities"]
    worlds = [{"id": f"av-{i:03d}", "a": config["people"][2*i], "b": config["people"][2*i+1],
               "city": cities[i % len(cities)], "other_city": cities[(i+1) % len(cities)],
               "n": i+2, "m": i+30} for i in range(count)]
    random.Random(config["seed"]).shuffle(worlds)
    return [(w, "train" if i < config["train_worlds"] else "dev") for i, w in enumerate(worlds)]


def world_cases(world, split, weights):
    a, b, city, other = (world[k] for k in ("a", "b", "city", "other_city"))
    lineage = [fact(a, "home_city", city), fact(b, "home_city", other)]
    for language in ("en", "zh"):
        zh = language == "zh"
        home = lambda person, place: f"{person}现在住在{place}。" if zh else f"{person} currently lives in {place}."
        query = ((f"{a}现在住在哪里？" if split == "train" else f"能告诉我{a}目前住在哪座城市吗？") if zh else
                 (f"Where does {a} live now?" if split == "train" else f"In which city is {a} currently living?"))
        ordinary_query = ((f"{world['n']}加{world['m']}等于多少？" if split == "train" else f"请算一下{world['n']}与{world['m']}的和。") if zh else
                          (f"What is {world['n']} plus {world['m']}?" if split == "train" else f"Please add {world['n']} and {world['m']}."))
        conflict_text = (f"有两个未经确认的说法：{a}现在住在{city}；{a}现在住在{other}。两者都没有核实。" if zh else
                         f"Two unverified reports say {a} currently lives in {city} and {a} currently lives in {other}. Neither is confirmed.")
        for scenario in SCENARIOS:
            text, facts = home(a, city), [fact(a, "home_city", city)]
            if scenario == "swapped_values": text, facts = home(a, other), [fact(a, "home_city", other)]
            elif scenario == "paraphrase": text = f"{city}是{a}目前居住的城市。" if zh else f"{city} is where {a} currently resides."
            elif scenario == "update_current":
                text = (f"{a}在2020年住在{city}，2024年搬到{other}，现在仍住在那里。" if zh else
                        f"{a} lived in {city} in 2020, moved to {other} in 2024, and still lives there.")
                facts = [fact(a, "home_city", city, "2020"), fact(a, "home_city", other)]
            elif scenario in ("empty", "ordinary_empty"): text, facts = None, []
            elif scenario in ("wrong_subject", "ordinary_wrong_subject"): text, facts = home(b, other), [fact(b, "home_city", other)]
            elif scenario == "wrong_relation":
                text = f"{a}喜欢{city}的建筑。" if zh else f"{a} likes the architecture of {city}."
                facts = [fact(a, "likes_architecture_of", city)]
            elif scenario in ("conflict", "ordinary_conflict"):
                text = conflict_text
                facts = [fact(a, "home_city", c, status="quoted") for c in (city, other)]
            elif scenario == "hypothetical":
                text = f"如果{a}住在{city}，通勤可能更方便。" if zh else f"If {a} lived in {city}, commuting might be easier."
                facts = [fact(a, "home_city", city, status="hypothetical")]
            elif scenario == "negated":
                text = f"{a}现在不住在{city}。" if zh else f"{a} does not currently live in {city}."
                facts = [fact(a, "home_city", city, status="negated")]
            elif scenario == "quoted":
                text = f"{b}声称{a}住在{city}，但我没有核实。" if zh else f"{b} claims {a} lives in {city}, but I have not verified it."
                facts = [fact(a, "home_city", city, status="quoted")]
            elif scenario == "historical_only":
                text = f"{a}在2020年住在{city}，我不知道之后的情况。" if zh else f"{a} lived in {city} in 2020; I do not know what happened afterward."
                facts = [fact(a, "home_city", city, "2020")]
            state = "supported" if scenario in SUPPORTED else "insufficient" if scenario in MISSING else "no_memory_needed"
            question = ordinary_query if state == "no_memory_needed" else query
            if state == "supported":
                claims = [facts[-1]]
                response = home(a, claims[0]["value"])
                allowed = facts
            elif state == "no_memory_needed":
                total = str(world["n"] + world["m"])
                claims = [fact(f"{world['n']}+{world['m']}", "equals", total, "timeless")]
                response = f"结果是{total}。" if zh else f"The sum is {total}."
                allowed = claims
            else:
                claims = []
                # A stylistic alternative is chosen by world, never by truth status.
                options = ([f"现有信息不足以确定{a}现在住在哪里。", f"我还不知道{a}目前住在哪座城市。"] if zh else
                           [f"I do not have enough information to say where {a} lives now.", f"I do not know which city {a} currently lives in."])
                response = options[world["n"] % len(options)]
                allowed = facts if scenario in ("conflict", "hypothetical", "negated", "quoted", "historical_only") else []
            case_id = "case-" + digest([world["id"], language, scenario])[:20]
            runtime = {"id": case_id, "context": [message(question)], "episodes": [message(text)] if text else []}
            label = {"id": case_id, "input_sha256": digest(runtime), "state": state, "response": response,
                     "sample_weight": weights[state], "required_claims": claims, "allowed_claims": allowed,
                     "evidence_missing": state == "insufficient", "memory_needed": state != "no_memory_needed",
                     "relevant_episode_indices": [0] if state == "supported" else []}
            meta = {"id": case_id, "world_id": world["id"], "split": split, "language": language, "scenario": scenario,
                    "family_facts": lineage, "episode_facts": [{"facts": facts}] if text else [], "source_facts": facts}
            yield runtime, label, meta


def build(config):
    rows = list(row for world, split in worlds_from_config(config) for row in world_cases(world, split, config["state_weights"]))
    inputs, labels, index = [list(group) for group in zip(*rows)]
    report = validate_records(inputs, labels, index)
    people, facts_seen = {}, {}
    counts, weighted = Counter(), Counter()
    lookup = {}
    for runtime, label, meta in rows:
        if label["input_sha256"] != digest(runtime): raise ValueError("supervision input mismatch")
        key = (meta["split"], label["state"])
        counts[key] += 1; weighted[key] += label["sample_weight"]
        lookup[(meta["world_id"], meta["language"], meta["scenario"])] = (runtime, label)
        for f in meta["family_facts"]:
            if people.setdefault(f["subject"], meta["split"]) != meta["split"]: raise ValueError("cross-split subject leakage")
        for f in meta["source_facts"]:
            if facts_seen.setdefault(digest(f), meta["split"]) != meta["split"]: raise ValueError("cross-split atomic fact leakage")
    pairs = []
    for world, split in worlds_from_config(config):
        for language in ("en", "zh"):
            controls = [("original", s) for s in SUPPORTED[1:] + MISSING] + [("ordinary_empty", s) for s in NORMAL[1:]]
            for left, right in controls:
                a, b = (lookup[(world["id"], language, s)] for s in (left, right))
                if a[0]["context"] != b[0]["context"] or a[0]["episodes"] == b[0]["episodes"]:
                    raise ValueError("causal intervention must only change memory")
                pairs.append({"world_id": world["id"], "split": split, "language": language,
                              "control": right, "left_id": a[0]["id"], "right_id": b[0]["id"],
                              "context_sha256": digest(a[0]["context"])})
    files = {}
    for split in ("train", "dev"):
        ids = {m["id"] for m in index if m["split"] == split}
        for name, records in (("inputs", inputs), ("labels", labels), ("index", index)):
            files[f"{split}.{name}.jsonl"] = "".join(canonical(r) + "\n" for r in records if r["id"] in ids).encode()
    files["pairs.jsonl"] = "".join(canonical(p) + "\n" for p in pairs).encode()
    report.update(pair_count=len(pairs), counts={"/".join(k): v for k, v in sorted(counts.items())},
                  weighted_counts={"/".join(k): v for k, v in sorted(weighted.items())})
    return files, report


def materialize(config_path, output, verify=False):
    config_path, output = Path(config_path), Path(output)
    config_bytes = config_path.read_bytes(); config = json.loads(config_bytes)
    files, report = build(config)
    repo = Path(__file__).resolve().parents[1]
    sources = ("python/prepare_availability_curriculum.py", "python/prepare_neural_memory_protocol.py", "python/episode_memory_inputs.py")
    manifest = {"format": FORMAT, "purpose": "single_event_pilot_not_promotion", "locomo_used": False,
                "automatic_writer": False, "supplied_episode_boundaries": True, "seed": config["seed"],
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "source_sha256": {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sources},
                "file_sha256": {p: hashlib.sha256(data).hexdigest() for p, data in files.items()}, **report}
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if verify:
        if {p.name for p in output.iterdir()} != set(files): raise ValueError("extra or missing corpus files")
        for name, data in files.items():
            if (output/name).read_bytes() != data: raise ValueError(f"frozen corpus mismatch: {name}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in files.items(): (output/name).write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.config, args.output, args.verify), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
