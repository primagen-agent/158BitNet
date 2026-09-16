#!/usr/bin/env python3
"""Single-fact conversational writer pilot with held-out realization templates.

Synthetic only. Relations are shared across splits; names, values and sentence
templates are disjoint. This is not unrestricted dialogue or multi-fact training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

SCHEMAS = (
    ("home_city", "Bergen Odense Turku Delft Linz Braga Arles Siena Basel Bonn York Cork".split(),
     (("I live in {value}.", "live"), ("My home is in {value}.", "home"),
      ("These days I am based in {value}.", "based"), ("The city I call home is {value}.", "home")),
     (("I moved from {old} to {value}.", "moved"), ("I now live in {value}, rather than {old}.", "live"),
      ("I left {old}; my home is in {value} now.", "home"), ("My relocation from {old} to {value} is complete.", "relocation"))),
    ("workplace", [x + " Studio" for x in "Amber Birch Copper Delta Elm Fern Grove Hazel Indigo Juniper Kestrel Linden".split()],
     (("I work at {value}.", "work"), ("My employer is {value}.", "employer"),
      ("I have a job with {value}.", "job"), ("The company I work for is {value}.", "work")),
     (("I changed jobs from {old} to {value}.", "jobs"), ("My employer is now {value}, not {old}.", "employer"),
      ("I left my job at {old} and started working at {value}.", "working"), ("My employment with {old} ended; I work for {value} instead.", "work"))),
    ("course", ["ceramics", "botany", "astronomy", "carpentry", "geology", "typography", "photography", "weaving", "ecology", "calligraphy", "robotics", "sculpture"],
     (("I am studying {value}.", "studying"), ("My evening course covers {value}.", "course"),
      ("I take classes in {value}.", "classes"), ("The subject I study after work is {value}.", "study")),
     (("I switched my studies from {old} to {value}.", "studies"), ("My course now covers {value} instead of {old}.", "course"),
      ("I stopped studying {old}; these days I take classes in {value}.", "classes"), ("I replaced my studies of {old} with {value}.", "studies"))),
    ("hobby", ["knitting", "hiking", "origami", "gardening", "kayaking", "juggling", "sketching", "sailing", "birdwatching", "woodworking", "dancing", "cycling"],
     (("My hobby is {value}.", "hobby"), ("I spend my free time on {value}.", "free time"),
      ("For fun, I enjoy {value}.", "fun"), ("My favorite leisure activity is {value}.", "leisure activity")),
     (("I changed my hobby from {old} to {value}.", "hobby"), ("My free time goes to {value} now, not {old}.", "free time"),
      ("I gave up {old}; for fun I do {value} now.", "fun"), ("My preferred leisure activity changed from {old} to {value}.", "leisure activity"))),
    ("commute", ["train", "bus", "bicycle", "tram", "ferry", "subway", "motorcycle", "car", "scooter", "trolleybus", "minibus", "taxi"],
     (("I commute by {value}.", "commute"), ("My usual transport to work is by {value}.", "transport"),
      ("I get to work by {value}.", "get to work"), ("My daily journey to work is by {value}.", "journey to work")),
     (("I changed my commute from {old} to {value}.", "commute"), ("My transport is now by {value} instead of by {old}.", "transport"),
      ("I used to get to work by {old}; now I go by {value}.", "get to work"), ("For my journey to work I now use {value} rather than {old}.", "journey to work"))),
    ("music", ["jazz", "blues", "reggae", "folk", "soul", "swing", "bluegrass", "classical music", "flamenco", "gospel", "synthpop", "bossa nova"],
     (("I listen to {value}.", "listen"), ("My favorite music is {value}.", "music"),
      ("My preferred listening genre is {value}.", "listening genre"), ("The music I enjoy most is {value}.", "music")),
     (("I switched from listening to {old} to {value}.", "listening"), ("My favorite music is now {value}, replacing {old}.", "music"),
      ("My preferred listening genre changed from {old} to {value}.", "listening genre"), ("Instead of {old}, the music I enjoy most now is {value}.", "music"))),
)
NAMES = {"train": "Avery Blake Cameron Dakota Emerson Finley Harper Jordan".split(),
         "valid": "Lennon Marley Nico Oakley Phoenix Reese Shiloh Winter".split(),
         "test": "Arden Briar Cleo Devin Ellis Frankie Gale Hollis".split()}
TEMPLATES = {"train": (0, 1), "valid": (2,), "test": (3,)}
FILLERS = ("", "It has been a busy week. ", "Here is a little news. ")


def world(index, seed, split):
    rng = random.Random(seed + index * 10007 + ("train", "valid", "test").index(split) * 1000003)
    names = rng.sample(NAMES[split], 2)
    relations = rng.sample(list(SCHEMAS), 2)
    facts = [(name, relation) for relation in relations for name in names]
    offset = ("train", "valid", "test").index(split) * 4
    values = [rng.sample(relation[1][offset:offset + 4], 2) for _, relation in facts]
    messages, events, template_ids = [], [], []
    for operation in (0, 1):
        for fact, (name, relation) in enumerate(facts):
            form = rng.choice(TEMPLATES[split])
            template, predicate_surface = relation[2 + operation][form]
            date = f"2024-03-{len(events) + 1:02d}" if rng.random() < 0.5 else ""
            prefix = (f"[{date}] " if date else "") + name + ": "
            text = prefix + rng.choice(FILLERS) + template.format(value=values[fact][operation], old=values[fact][0])
            # Equal probability for create/update: length and chatter are not label cues.
            if rng.random() < 0.25:
                text += " How has your week been?"
            messages.append(text)
            template_ids.append(f"{relation[0]}:{operation}:{form}")
            events.append({"episode": len(events), "entity": name, "entity_surface": name,
                "predicate": relation[0], "predicate_surface": predicate_surface,
                "value": values[fact][operation], "value_surface": values[fact][operation],
                "time": date, "time_surface": date, "operation": "create" if operation == 0 else "update",
                "previous_episode": fact if operation else None, "active": operation == 1})
    evidence = "\n".join(messages)
    identity = f"dialogue-{split}-{seed}-{index:05d}"
    metadata = {"world_id": identity, "semantic_world_id": identity, "split": split,
        "raw_episodes": messages, "typed_events": events, "template_ids": template_ids,
        "generalization_axis": "disjoint_names_values_and_realization_templates_shared_relations",
        "locomo_used": False, "family": "single_fact_conversational_state"}
    rows = []
    questions = {"home_city": "Where does {name} live now?", "workplace": "Where does {name} work now?",
        "course": "What is {name} studying now?", "hobby": "What does {name} do for fun now?",
        "commute": "How does {name} get to work now?", "music": "What music does {name} enjoy now?"}
    for fact, (name, relation) in enumerate(facts):
        target = 4 + fact
        answer = values[fact][1]
        begin = sum(len(m) + 1 for m in messages[:target]) + messages[target].rfind(answer)
        rows.append({"sample_id": f"{identity}-{fact}", "evidence": evidence,
            "question": questions[relation[0]].format(name=name), "answer": answer, "answer_type": "span",
            "answer_spans": [{"start": begin, "end": begin + len(answer), "text": answer}],
            "metadata": {**metadata, "query_plan": {"intent": "current",
                "targets": [{"entity": name, "predicate": relation[0]}]}}})
    null = json.loads(json.dumps(rows[0]))
    null.update(sample_id=identity + "-null", question=questions[facts[0][1][0]].format(name=f"UnknownPerson{index}"),
                answer="No information available", answer_spans=[])
    null["metadata"].update(counterfactual_no_info=True, query_plan={"intent": "null", "targets": []})
    rows.append(null)
    if split == "test":
        for row in rows:
            row["evaluation_only"] = True
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--train-worlds", type=int, default=128)
    parser.add_argument("--valid-worlds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2740916)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    targets = [root / f"{split}.jsonl" for split in NAMES] + [root / "manifest.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError("refusing to overwrite a frozen curriculum")
    if min(args.train_worlds, args.valid_worlds) <= 0:
        raise ValueError("world counts must be positive")
    files = {}
    for split in NAMES:
        count = args.train_worlds if split == "train" else args.valid_worlds
        path = root / f"{split}.jsonl"
        rows = [row for i in range(count) for row in world(i, args.seed, split)]
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        files[split] = {"worlds": count, "events": count * 8,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps({**vars(args), "files": files,
        "evaluation_only_split": "test", "templates": TEMPLATES,
        "limitations": "synthetic, shared relation semantics, one labeled fact per turn; no event history or multi-fact extraction"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
