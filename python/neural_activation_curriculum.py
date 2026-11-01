"""Independent fit scope/lifecycle teachers and held-out evidence challenges.

The authored values below read no benchmark or development material. Fact
reference fields are bank supervision. Questions contain natural named subjects
and never receive a gold reference prefix. Evaluation cues sample the complete
supplied bank without inspecting a question's support labels.
"""

from copy import deepcopy
from collections import Counter, defaultdict
import ast
from pathlib import Path
import random

import torch


FIT_ACTORS = ("Iris", "Quentin", "Rowena")
SCOPE_DEVELOPMENT_ACTORS = ("Leona", "Mael")
# Property, original present-tense body, replacement body, tentative body,
# named scope declarations, current/history/tentative/all-version questions,
# and a raw named deletion request. Nothing here is inferred from evaluation.
LIFECYCLE_PATTERNS = (
    (
        "residence",
        (
            "I live in a flat in Tartu.",
            "I live in a house in Sion.",
            "I plan to move into a cottage in Dinan next summer.",
        ),
        (
            "{name} lives in a flat in Tartu.",
            "{name} lives in a house in Sion.",
            "{name} plans to move into a cottage in Dinan next summer.",
        ),
        (
            "Which town is home to {name} at present?",
            "Which town used to be home to {name}?",
            "Which town is {name} planning to move to next summer?",
            "What are {name}'s former, current, and planned home towns?",
        ),
        "Forget the information about where {name} lives or plans to live.",
    ),
    (
        "meal_preference",
        (
            "I choose barley soup for lunch.",
            "I choose lentil pie for lunch.",
            "I am considering switching lunch to baked potatoes next month.",
        ),
        (
            "{name} chooses barley soup for lunch.",
            "{name} chooses lentil pie for lunch.",
            "{name} is considering switching lunch to baked potatoes next month.",
        ),
        (
            "What does {name} currently choose for lunch?",
            "What was {name}'s former lunch preference?",
            "What lunch is {name} considering for next month?",
            "What lunches has {name} preferred or considered over time?",
        ),
        "Forget {name}'s past, present, and proposed lunch preferences.",
    ),
    (
        "tool_storage",
        (
            "I keep the measuring tools in a blue chest.",
            "I keep the measuring tools in a gray cabinet.",
            "I intend to move the measuring tools into a cedar crate next month.",
        ),
        (
            "{name} keeps the measuring tools in a blue chest.",
            "{name} keeps the measuring tools in a gray cabinet.",
            "{name} intends to move the measuring tools into a cedar crate next month.",
        ),
        (
            "Where does {name} keep the measuring tools now?",
            "Where did {name} previously keep the measuring tools?",
            "Where is {name} intending to put the measuring tools next month?",
            "What are the previous, current, and intended locations of {name}'s measuring tools?",
        ),
        "Forget every stored version of where {name} keeps the measuring tools.",
    ),
    (
        "relaxation",
        (
            "I unwind by making mosaics.",
            "I unwind by practicing tai chi.",
            "I am thinking of trying archery as a way to unwind next season.",
        ),
        (
            "{name} unwinds by making mosaics.",
            "{name} unwinds by practicing tai chi.",
            "{name} is thinking of trying archery as a way to unwind next season.",
        ),
        (
            "How does {name} currently unwind?",
            "What did {name} previously do to unwind?",
            "What relaxing activity is {name} thinking of trying next season?",
            "Which past, current, and proposed activities help {name} unwind?",
        ),
        "Forget {name}'s old, current, and proposed ways of unwinding.",
    ),
)


def lifecycle_fit_group():
    """Fixed insertion order with text-kind distinct from physical slot status.

    Historical records preserve their original present-tense text and kind=0.
    ``status`` uses C's physical CURRENT=0, SUPERSEDED=1, TENTATIVE=3 enums;
    DELETED=2 is deliberately absent. Queries use metadata channel indices
    ``allowed_kinds`` 0=current, 1=history, 2=tentative. ``source_kinds`` labels
    the supporting original text for semantic auxiliary supervision only.
    """
    facts, queries, scope_queries = [], [], []
    for name in FIT_ACTORS:
        for prop, bodies, declarations, questions, deletion in LIFECYCLE_PATTERNS:
            sources = []
            for index, body in enumerate(bodies):
                fact = dict(
                    text=name + ". " + body,
                    content=body,
                    entity=name,
                    reference=name,
                    prop=prop,
                    kind=2 if index == 2 else 0,
                    status=(1, 0, 3)[index],
                    insert_order=len(facts),
                    lifecycle=True,
                )
                facts.append(fact)
                sources.append(fact)
            for index, question in enumerate(questions):
                support = sources if index == 3 else [sources[(1, 0, 2)[index]]]
                queries.append(
                    dict(
                        text=question.format(name=name),
                        entities=[name], prop=prop,
                        kind=(0, 1, 2, 0)[index],
                        allowed_kinds=[0, 1, 2] if index == 3 else [[0], [1], [2]][index],
                        source_kinds=sorted({f["kind"] for f in support}),
                        support_texts=[f["text"] for f in support],
                        lifecycle=True,
                        lifecycle_view=("current", "history", "tentative", "all_versions")[index],
                    )
                )
            for index, declaration in enumerate(declarations):
                scope_queries.append(
                    dict(
                        text=declaration.format(name=name),
                        entities=[name], prop=prop, kind=2 if index == 2 else 0,
                        support_texts=[f["text"] for f in sources],
                        scope_only=True, scope_operation="declaration",
                    )
                )
            scope_queries.append(
                dict(
                    text=deletion.format(name=name), entities=[name], prop=prop,
                    kind=0, support_texts=[f["text"] for f in sources],
                    scope_only=True, scope_operation="delete",
                )
            )
    group = dict(facts=facts, queries=queries, scope_queries=scope_queries)
    assert len({row["text"] for rows in group.values() for row in rows}) == 132
    return group


def scope_fit_group(seed=7319):
    """Separate bank and scope-only sources; recall labels remain independent."""
    group = lifecycle_fit_group()
    for name in FIT_ACTORS:
        own = [f for f in group["facts"] if f["entity"] == name]
        for text in (
            f"Forget everything recorded about {name}.",
            f"Delete all saved information relating to {name}.",
        ):
            group["scope_queries"].append(
                dict(text=text, entities=[name], prop=sorted({f["prop"] for f in own}),
                     kind=0, support_texts=[f["text"] for f in own],
                     scope_only=True, scope_operation="delete_all_subject")
            )
    random.Random(seed).shuffle(group["scope_queries"])
    return dict(facts=group["facts"], scope_queries=group["scope_queries"])


def _registered_actor_names():
    """Read actor constants only; never evaluate held-out/benchmark text."""
    result = set(FIT_ACTORS)
    for filename in (
        "prepare_neural_identity.py", "prepare_neural_dialogue.py",
        "prepare_neural_colloquial.py", "prepare_neural_episodes.py",
        "prepare_neural_reader_expansion.py",
    ):
        for node in ast.parse(Path(__file__).with_name(filename).read_text()).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"NAMES", "FIT_ACTORS", "ACTORS"}
                for t in node.targets
            ):
                names = ast.literal_eval(node.value)
                if isinstance(names, dict):
                    for values in names.values():
                        result.update(values)
                else:
                    result.update(names)
    return result


def scope_development_group():
    """Independent scope-head selection only: 32 facts and 20 raw requests.

    Four versions comprise two superseded present-tense statements, one
    current statement and one tentative statement. Names, wording and values
    are independently authored here and absent from scope/lifecycle fit.
    This group must never enter a training bank or ordinary recall selection.
    """
    if set(SCOPE_DEVELOPMENT_ACTORS) & _registered_actor_names():
        raise ValueError("scope development actors overlap existing curricula")
    patterns = (
        (
            "residence",
            ("I rent a room in Cuenca.", "My home is a flat in Lund.",
             "I share a house in Asolo.", "I may move into a studio in Mons next winter."),
            "{name} now calls Asolo home.",
            "Please erase the notes about {name}'s places of residence.",
        ),
        (
            "meal_preference",
            ("My midday meal is chickpea stew.", "I prefer rye sandwiches for lunch.",
             "I usually pick mushroom risotto at lunchtime.",
             "I am thinking about trying polenta bowls for lunch next month."),
            "At midday, {name} now prefers mushroom risotto.",
            "Please remove {name}'s meal preferences from the saved information.",
        ),
        (
            "tool_storage",
            ("I keep my hand tools in a canvas roll.", "My hand tools are in a wicker basket.",
             "I store the hand tools in an orange trunk.",
             "I might transfer my hand tools to a steel case next spring."),
            "{name}'s hand tools are now stored in an orange trunk.",
            "Erase the stored details of how {name} keeps the hand tools.",
        ),
        (
            "relaxation",
            ("I relax by drawing charcoal portraits.", "I recharge by assembling jigsaw maps.",
             "I clear my mind by tending a miniature rock garden.",
             "I am considering listening to wind chimes to relax next season."),
            "Tending a miniature rock garden is how {name} now clears the mind.",
            "Remove the notes about the activities that help {name} relax.",
        ),
    )
    facts, scope_queries = [], []
    for name in SCOPE_DEVELOPMENT_ACTORS:
        for prop, bodies, declaration, deletion in patterns:
            sources = []
            for index, body in enumerate(bodies):
                fact = dict(
                    text=name + ". " + body, content=body, entity=name, reference=name,
                    prop=prop, kind=2 if index == 3 else 0,
                    status=(1, 1, 0, 3)[index], insert_order=len(facts),
                    scope_development=True,
                )
                facts.append(fact)
                sources.append(fact)
            for text, operation in ((declaration, "declaration"), (deletion, "delete")):
                scope_queries.append(
                    dict(text=text.format(name=name), entities=[name], prop=prop, kind=0,
                         support_texts=[f["text"] for f in sources], scope_only=True,
                         scope_operation=operation, evaluation_only=True)
                )
        own = [f for f in facts if f["entity"] == name]
        for text in (
            f"Please wipe all the information associated with {name}.",
            f"Nothing about {name} should remain in the saved notes; clear it all.",
        ):
            scope_queries.append(
                dict(text=text, entities=[name], prop=sorted({f["prop"] for f in own}),
                     kind=0, support_texts=[f["text"] for f in own],
                     scope_only=True, scope_operation="delete_all_subject", evaluation_only=True)
            )
    assert len({r["text"] for r in facts + scope_queries}) == 52
    return dict(facts=facts, scope_queries=scope_queries)


def served_lifecycle_scope_group():
    """Fit-only scope supervision for the lifecycle sentences serving sends.

    The HTTP lifecycle writes a city/food fact, updates the city with "I now
    live in Paris.", and deletes with "Forget my home." / "Forget my food
    preference.". Those first-person sentences are fit writer rows, but the
    scope heads were trained on third-person declarations. Pair each served
    sentence with every fit dialogue wording of the same property so the update
    and delete gates learn the property relation, not one lexical overlap.
    """
    from prepare_neural_dialogue import dialogue_curriculum

    facts, seen = [], set()
    for row in dialogue_curriculum("fit"):
        if row["tag"] != "natural-fact" or not row.get("recall_question"):
            continue
        span = row["spans"][0]
        if span["prop"] not in ("city", "food"):
            continue
        name = span["references"][0]
        content = row["body"].encode()[span["start"]:span["end"]].decode()
        text = f"{name}. {content}"
        if text in seen:
            continue
        seen.add(text)
        facts.append(dict(text=text, content=content, entity=name, reference=name,
                          prop=span["prop"], kind=0, status=0))
    by_name = defaultdict(lambda: {"city": [], "food": []})
    for fact in facts:
        by_name[fact["entity"]][fact["prop"]].append(fact["text"])
    scope_queries = []
    for name in sorted(by_name):
        for prop, request, operation in (
                ("city", "I now live in Paris.", "declaration"),
                ("city", "Forget my home.", "delete"),
                ("food", "Forget my food preference.", "delete")):
            support = by_name[name][prop]
            if not support:
                continue
            scope_queries.append(dict(
                text=f"{name}. {request}", entities=[name], prop=prop, kind=0,
                support_texts=support, scope_only=True,
                scope_operation=operation))
    return dict(facts=facts, scope_queries=scope_queries)


def served_scope_paraphrase_group():
    """Authored fit-only residence and meal frames for the served requests.

    The lifecycle sentences serving sends pair a declaration or a deletion with
    the stored fact. Fit dialogue offers few residence frames, so a scope head
    learned only there matches predicate words rather than the property. These
    fit actors and values are authored here; no development, held-out or
    benchmark sentence participates. The frames stay disjoint from every
    dialogue split and only broaden the learned property relation.
    """
    cities = ("Bergen", "Tartu", "Sion", "Dinan", "Leuven", "Durham", "Ghent",
              "Siena", "Uppsala")
    dishes = ("vegetable stew", "barley soup", "lentil pie", "chickpea stew",
              "mushroom risotto", "rye sandwiches")
    residence = (
        "My hometown is {value}.",
        "{value} is my hometown.",
        "I still call {value} my hometown.",
        "These days I am based in {value}.",
        "My place of residence is {value}.",
        "I make my home in {value}.",
        "{value} is where I live now.",
        "My address is in {value}.",
        "I reside in {value}.",
        "{value} is my permanent home.",
    )
    meals = (
        "My favorite dish is {value}.",
        "{value} is the dish I like most.",
        "I prefer {value} for lunch.",
        "My usual meal is {value}.",
        "I always choose {value}.",
    )
    facts, scope_queries = [], []
    for name_index, name in enumerate(FIT_ACTORS):
        by_prop = {"city": [], "food": []}
        for frame_index, frame in enumerate(residence):
            value = cities[(name_index * len(residence) + frame_index) % len(cities)]
            body = frame.format(value=value)
            fact = dict(text=f"{name}. {body}", content=body, entity=name,
                        reference=name, prop="city", kind=0, status=0,
                        paraphrase=True)
            facts.append(fact)
            by_prop["city"].append(fact["text"])
        for frame_index, frame in enumerate(meals):
            value = dishes[(name_index * len(meals) + frame_index) % len(dishes)]
            body = frame.format(value=value)
            fact = dict(text=f"{name}. {body}", content=body, entity=name,
                        reference=name, prop="food", kind=0, status=0,
                        paraphrase=True)
            facts.append(fact)
            by_prop["food"].append(fact["text"])
        for prop, request, operation in (
                ("city", "I now live in Paris.", "declaration"),
                ("city", "Forget my home.", "delete"),
                ("food", "Forget my food preference.", "delete")):
            scope_queries.append(dict(
                text=f"{name}. {request}", entities=[name], prop=prop, kind=0,
                support_texts=by_prop[prop], scope_only=True,
                scope_operation=operation))
    return dict(facts=facts, scope_queries=scope_queries)


def scope_relation_masks(rows, facts, hardness, max_negatives=4):
    """Scope semantics compare properties across actors and every live version.

    Content, old values and temporal kind never create a scope negative. Main
    scope still uses each question's authored same-subject support_texts.
    """
    if (type(max_negatives) is not int or not 0 <= max_negatives <= 4
        or hardness.shape != (len(rows), len(facts)) or not torch.isfinite(hardness).all()):
        raise ValueError("invalid bounded scope teacher geometry")
    def values(value):
        return {value} if isinstance(value, str) else set(value or [])
    positive = torch.tensor(
        [[f.get("status") != 2 and f["prop"] in values(row["prop"]) for f in facts] for row in rows],
        dtype=torch.bool, device=hardness.device,
    ).reshape_as(hardness)
    eligible = torch.tensor(
        [[f.get("status") != 2 and f["entity"] in values(row.get("entities", row.get("entity")))
          and f["prop"] not in values(row["prop"]) for f in facts] for row in rows],
        dtype=torch.bool, device=hardness.device,
    ).reshape_as(hardness)
    negative = torch.zeros_like(eligible)
    count = min(max_negatives, len(facts))
    if count:
        indices = hardness.detach().masked_fill(~eligible, -torch.inf).topk(count, -1).indices
        negative.scatter_(1, indices, eligible.gather(1, indices))
    return positive, negative


def evidence_challenge(group, seed, limit=24, excluded_observation_ids=()):
    """Evaluation-only flat rows for identical zero/one/four-cue query views.

    Question selection covers every nonempty authored property, then balances
    property/kind/query-class strata within the fixed budget. Cue selection
    reads only complete bank texts and an independent per-observation generator.
    It never consults entities/properties/support labels. The original group
    is unchanged. Subsequent-cue identities follow native semantics: retain
    the frozen NN original reference as initial_reference, then infer current
    identity from the complete conditioned input rather than overriding it.
    """
    if type(seed) is not int or type(limit) is not int or not 0 <= limit <= 24:
        raise ValueError("evidence challenge requires fixed integer seed and limit 0..24")
    rng = random.Random(seed)
    excluded = set(excluded_observation_ids)
    candidates = []
    for row in group["queries"]:
        observation_id = row.get("observation_id") or row.get("observation", {}).get("source_id")
        if not observation_id:
            raise ValueError("held-out challenge requires original observation IDs")
        if observation_id not in excluded and not row.get("augmented"):
            candidates.append((observation_id, row))
    candidates.sort(key=lambda pair: pair[0])
    if len({identifier for identifier, _ in candidates}) != len(candidates):
        raise ValueError("held-out challenge observation IDs must be unique")

    def properties(row):
        value = row.get("prop")
        values = [value] if isinstance(value, str) else value or ()
        return tuple(sorted({prop for prop in values if prop}))

    strata = defaultdict(list)
    for pair in candidates:
        row = pair[1]
        key = (properties(row), str(row.get("kind", -1)), str(row.get("query_class", "")))
        strata[key].append(pair)
    for values in strata.values():
        rng.shuffle(values)
    props = sorted({prop for key in strata for prop in key[0]})
    if len(props) > limit:
        raise ValueError("evidence question budget cannot cover every nonempty property")
    rng.shuffle(props)
    selected_counts = Counter()
    chosen, covered = [], set()

    def take(keys):
        minimum = min(selected_counts[key] for key in keys)
        key = rng.choice(sorted(key for key in keys if selected_counts[key] == minimum))
        chosen.append(strata[key].pop())
        selected_counts[key] += 1
        covered.update(key[0])

    for prop in props:
        if prop not in covered:
            take([key for key, values in strata.items() if values and prop in key[0]])
    while len(chosen) < min(limit, len(candidates)):
        take([key for key, values in strata.items() if values])
    chosen.sort(key=lambda pair: pair[0])
    bank = sorted(f["text"] for f in group["facts"] if f.get("status") != 2)
    if len(bank) < 4 and chosen:
        raise ValueError("four-cue challenge requires at least four live facts")
    queries = []
    for observation_id, original in chosen:
        # Fixed independently of question stratification, support labels and
        # query order: changing a teacher cannot alter a retained question's cues.
        cue_rng = random.Random(f"evidence:{seed}:{observation_id}")
        for count in (0, 1, 4):
            row = deepcopy(original)
            row.update(
                initial_text=original["text"], evidence_challenge_view=count,
                evidence_count=count, evaluation_only=True,
                original_observation_id=observation_id,
            )
            if original.get("reference"):
                row["initial_reference"] = original["reference"]
            if count:
                cues = cue_rng.sample(bank, count)
                row.update(
                    text=original["text"] + "\nActivated evidence:\n" + "\n".join(cues) + "\n",
                    augmented=True, evidence_sources=cues,
                )
                row.pop("reference", None)
            queries.append(row)
    return dict(facts=deepcopy(group["facts"]), queries=queries)


def identity_multisubject_fit_group():
    """Eighteen independently worded fit-only two/three-subject questions."""
    from prepare_neural_identity import identity_curriculum
    original = identity_curriculum()
    names = [row["entity"] for row in original["facts"]]
    patterns = (
        "Where do {subjects} keep their sketchbooks?",
        "What materials do {subjects} prefer for their repair work?",
        "Which workshops did {subjects} attend last month?",
        "Why did {subjects} decide to share the garden tools?",
        "What are the current plans of {subjects}?",
        "How do {subjects} organize their collections?",
    )
    queries = []
    for i in range(18):
        actors = [names[i], names[(i + 23) % len(names)]]
        if i % 3 == 0:
            actors.append(names[(i + 41) % len(names)])
        subjects = " and ".join(actors)
        queries.append(dict(text=patterns[i % len(patterns)].format(subjects=subjects), entities=actors, identity_only=True))
    return dict(facts=deepcopy(original["facts"]), queries=queries)


def relation_variation_fit_group():
    """Independent complete facts and object-specific questions: 96 texts.

    All 48 facts coexist in one fit bank. The two objects under each property
    have separate true facts, so questions explicitly identify their object.
    Fact reference prefixes follow the supervised memory-bank representation;
    every query is a natural named question without a labeled reference.
    """
    from prepare_neural_identity import ACTORS
    names = ("Anika", "Conrad", "Sabrina")
    if not set(names) <= set(ACTORS):
        raise ValueError("relation variation requires independent fit actors")
    patterns = (
        (
            "paint_choice",
            "I coat my wooden bird feeder with matte plum paint, adding a weatherproof layer so drizzle does not damage it.",
            "What coating does {name} put on the wooden bird feeder?",
        ),
        (
            "paint_choice",
            "For the toy chest I use bright coral enamel; the glossy surface makes sticky marks easy to wash off.",
            "Which finish is on the toy chest that {name} paints?",
        ),
        (
            "window_seal",
            "A silicone bead around my attic skylight blocks the drip that used to land on the ladder.",
            "How does {name} stop water leaking around the attic skylight?",
        ),
        (
            "window_seal",
            "A felt gasket lines my round porthole window; it cuts down the whistle when the wind picks up.",
            "What does {name} fit inside the round window to quiet the wind?",
        ),
        (
            "plant_watering",
            "My lemon tree gets a slow soak at dusk twice a week, and I wait until the top layer feels dry before giving it more.",
            "Describe how {name} times watering for the lemon tree.",
        ),
        (
            "plant_watering",
            "I water the aloe from underneath once the pot is light, then lift it out of the tray after ten minutes so the soil drains.",
            "What tells {name} that the aloe needs its next drink?",
        ),
        (
            "plant_soil",
            "The rosemary on my porch grows in loam mixed with fine gravel; I add the gravel to keep the mix from compacting.",
            "How is the soil around {name}'s porch rosemary prepared?",
        ),
        (
            "plant_soil",
            "My bamboo seedlings sit in a mix of leaf mould and perlite, whose pale grains leave small gaps for air to circulate.",
            "Which ingredients are in {name}'s bamboo seedling mix?",
        ),
        (
            "lamp_repair",
            "The lantern on my workbench lights evenly now that I have replaced the corroded battery spring and wiped the contacts.",
            "What repair made {name}'s workbench lantern shine evenly again?",
        ),
        (
            "lamp_repair",
            "My reading lamp is working with the dimmer wire soldered back together and the joint protected by heat-shrink tubing.",
            "How did {name} protect the repaired joint near the reading lamp's dimmer?",
        ),
        (
            "fan_repair",
            "The ceiling fan spins smoothly since I tightened the blade mounts and cleared dust from the bearings.",
            "How does {name} keep the ceiling fan spinning smoothly?",
        ),
        (
            "fan_repair",
            "I stop the USB fan rattling by inserting rubber washers under its motor housing and leveling the base.",
            "What changes does {name} make to stop the USB fan rattling?",
        ),
        (
            "photo_settings",
            "For photos of falling leaves I choose a shutter speed of 1/1000 second and continuous autofocus, which freezes the motion.",
            "Which camera settings help {name} freeze the motion of falling leaves?",
        ),
        (
            "photo_settings",
            "When I photograph star trails, the camera stays on ISO 200 with the shutter open for thirty seconds so faint points gather enough light.",
            "What exposure setup does {name} use to record star trails?",
        ),
        (
            "photo_backup",
            "Every Friday I mirror my edited pictures to a black NAS box, then store a second copy on encrypted memory cards in a locked drawer.",
            "Where does {name} keep the extra copy made during the Friday photo backup?",
        ),
        (
            "photo_backup",
            "I save the panoramic image masters on a pair of orange portable SSDs and keep one drive at my studio so a local failure will not lose both copies.",
            "How does {name} separate the two copies of the panoramic image masters?",
        ),
    )
    facts, queries = [], []
    for name in names:
        for index, (prop, body, question) in enumerate(patterns):
            text = name + ". " + body
            pattern_id = f"fit-relation-variation-{index}"
            facts.append(
                dict(text=text, content=body, entity=name, reference=name,
                     prop=prop, kind=0, status=0, relation_variation=True,
                     pattern_id=pattern_id)
            )
            queries.append(
                dict(text=question.format(name=name), entities=[name], prop=prop,
                     kind=0, allowed_kinds=[0], support_texts=[text],
                     semantic_specific=True, relation_variation=True,
                     pattern_id=pattern_id)
            )
    assert len({r["text"] for r in facts + queries}) == 96
    return dict(facts=facts, queries=queries)
