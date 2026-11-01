"""Fit-only raw-question/literal-reference pairs for learned identity training.

This curriculum contains no personal facts, answers, semantic support labels,
or evaluation questions. Its only target is the actor mentioned in a question.
Native serving continues to use the trained neural role/projection heads; the
name comparisons below validate authored training definitions only.
"""

import ast
from pathlib import Path

ACTORS = (
    "Abigail",
    "Adrian",
    "Alistair",
    "Anika",
    "Anton",
    "Beatrice",
    "Brendan",
    "Cecily",
    "Clarissa",
    "Conrad",
    "Delia",
    "Dominic",
    "Eleanor",
    "Elliott",
    "Estelle",
    "Felix",
    "Florence",
    "Gabriel",
    "Giselle",
    "Hugo",
    "Ingrid",
    "Iris",
    "Ivana",
    "Jocelyn",
    "Joseph",
    "Kara",
    "Laurel",
    "Lawrence",
    "Linus",
    "Lydia",
    "Malcolm",
    "Margot",
    "Miriam",
    "Nathaniel",
    "Nell",
    "Oliver",
    "Padma",
    "Patrick",
    "Penelope",
    "Quentin",
    "Raoul",
    "Reuben",
    "Rosalie",
    "Rowena",
    "Sabrina",
    "Serena",
    "Stefan",
    "Tobias",
    "Ursula",
    "Valentina",
    "Veronica",
    "Wallace",
    "Wilfred",
    "Winona",
    "Xavier",
    "Yasmin",
    "Zachary",
    "Zadie",
    "Zora",
    "Astrid",
    "Bruno",
    "Celia",
    "Dennis",
    "Elise",
)

CAUSE_QUESTIONS = (
    "Why did {name} stop working the evening shift?",
    "What motivated {name}'s decision to study navigation?",
    "What made the volunteer project appealing to {name}?",
    "Why was saving travel time important to {name}?",
    "What prompted {name} to restore the old instrument?",
    "What was {name}'s reason for moving closer to the workplace?",
    "Why does {name} prefer an adjustable chair?",
    "What explains {name}'s interest in learning calligraphy?",
)

EVENT_QUESTIONS = (
    "On what date did {name} attend the safety workshop?",
    "When was {name}'s service certificate presented?",
    "Who accompanied {name} on the factory tour?",
    "What happened during the graduation ceremony attended by {name}?",
    "In which month did {name} begin the apprenticeship?",
    "When did the organizers recognize {name}'s contribution?",
    "Which relative joined {name} for the sailing lesson?",
    "What award was given to {name} at the design exhibition?",
)

LOCATION_QUESTIONS = (
    "Where does {name} keep the measuring tape?",
    "Which town is {name}'s workshop in?",
    "How far from the tram stop is the home rented by {name}?",
    "Where is the suitcase that belongs to {name} stored?",
    "What room does {name} use for practicing the violin?",
    "Where does {name} plan to put the new bookcase?",
    "Which building houses the club founded by {name}?",
    "Where can the tools borrowed by {name} be found?",
)


def _existing_actor_names():
    """Read split name constants only, without evaluating curriculum modules."""
    result = set()
    for filename in (
        "prepare_neural_dialogue.py",
        "prepare_neural_episodes.py",
        "prepare_neural_colloquial.py",
    ):
        source = Path(__file__).with_name(filename)
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "NAMES"
                for target in node.targets
            ):
                for split, names in ast.literal_eval(node.value).items():
                    if split != "fit":
                        result.update(names)
    return result


def identity_curriculum():
    """Return 64 references and three raw questions per new training actor."""
    facts, queries = [], []
    for index, actor in enumerate(ACTORS):
        facts.append(dict(text=actor, entity=actor, reference=actor))
        for patterns, offset in (
            (CAUSE_QUESTIONS, 0),
            (EVENT_QUESTIONS, 3),
            (LOCATION_QUESTIONS, 5),
        ):
            question = patterns[(index + offset) % len(patterns)].format(name=actor)
            queries.append(dict(text=question, entities=[actor]))
    group = dict(facts=facts, queries=queries)
    validate(group)
    return group


def validate(group):
    """Check training actor isolation and pure identity-label consistency."""
    if len(ACTORS) != 64 or len(set(ACTORS)) != 64:
        raise ValueError("identity training requires 64 distinct actors")
    overlap = set(ACTORS) & _existing_actor_names()
    if overlap:
        raise ValueError("identity actors overlap existing splits: " + repr(overlap))
    facts = {row["text"]: row for row in group["facts"]}
    if len(facts) != 64 or set(facts) != set(ACTORS):
        raise ValueError("reference bank differs from the authored actor set")
    for text, row in facts.items():
        if set(row) != {"text", "entity", "reference"} or not (
            text == row["entity"] == row["reference"]
        ):
            raise ValueError("literal reference must contain only its actor")
    counts = dict.fromkeys(ACTORS, 0)
    seen = set(facts)
    for row in group["queries"]:
        if set(row) != {"text", "entities"} or len(row["entities"]) != 1:
            raise ValueError("question must carry a single identity label only")
        actor = row["entities"][0]
        if actor not in counts or actor not in row["text"]:
            raise ValueError("authored question lacks its target actor")
        if row["text"] in seen:
            raise ValueError("duplicate feature text")
        seen.add(row["text"])
        counts[actor] += 1
    if set(counts.values()) != {3} or len(seen) != 256:
        raise ValueError("expected three questions per actor and 256 feature texts")


if __name__ == "__main__":
    import json

    rows = identity_curriculum()
    print(
        json.dumps(
            dict(
                actors=len(ACTORS),
                references=len(rows["facts"]),
                raw_questions=len(rows["queries"]),
                unique_feature_texts=len(rows["facts"]) + len(rows["queries"]),
                disjoint_existing_split_names=True,
                semantic_labels=False,
                fit_only=True,
            ),
            indent=2,
        )
    )
