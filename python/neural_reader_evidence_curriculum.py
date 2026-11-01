"""Bounded evidence-conditioned questions for synthetic fit curricula.

The teacher chooses training examples, using the group's existing support
labels. The input contains only an original question and ordinary facts in
the exact format emitted by the C activation loop. This module reads no files
and is not used by serving. Call it for fit data only; keep development and
sealed questions unchanged.
"""

from collections import defaultdict
from copy import deepcopy
import random


# These questions describe only the existing independently authored fit facts
# in relation_variation_fit_group. They add supported aspects and phrasing;
# neither the facts nor their complete-support labels are inferred from text.
RELATION_PARAPHRASE_FAMILIES = (
    (
        ("cause", "Why does {name} put a weatherproof coating on the wooden bird feeder?"),
        ("appearance", "What color and finish does {name} choose for the bird feeder's paint?"),
    ),
    (
        ("cause", "What makes sticky marks easy to remove from {name}'s toy chest?"),
        ("appearance", "How would {name} describe the enamel on the toy chest?"),
    ),
    (
        ("procedure", "What change keeps drips from reaching the ladder beneath {name}'s skylight?"),
        ("material", "Which sealing material does {name} use around the attic skylight?"),
    ),
    (
        ("cause", "How is wind noise reduced at {name}'s round window?"),
        ("material", "What is the lining inside {name}'s porthole window made from?"),
    ),
    (
        ("frequency", "How frequently is {name}'s lemon tree given a slow soak?"),
        ("condition", "What must feel dry before {name} gives the lemon tree more water?"),
    ),
    (
        ("duration", "How long does {name} leave the aloe in its tray after watering from below?"),
        ("cause", "Why does {name} lift the aloe out of the tray after its drink?"),
    ),
    (
        ("material", "What kind of growing medium does {name} prepare for the porch rosemary?"),
        ("cause", "What prevents the soil in {name}'s rosemary pot from compacting?"),
    ),
    (
        ("cause", "Why does {name} include perlite in the bamboo seedlings' growing medium?"),
        ("material", "What two materials make up the soil mixture for {name}'s bamboo seedlings?"),
    ),
    (
        ("procedure", "What did {name} replace and clean to restore the workbench lantern?"),
        ("component", "Which part of the battery assembly did {name} change in the lantern?"),
    ),
    (
        ("procedure", "What electrical connection did {name} restore in the reading light?"),
        ("material", "What covers the repaired dimmer-wire joint in {name}'s reading lamp?"),
    ),
    (
        ("procedure", "What maintenance did {name} perform on the ceiling fan's mounts and bearings?"),
        ("component", "Which attachments did {name} secure to improve the ceiling fan's rotation?"),
    ),
    (
        ("material", "What did {name} place under the USB fan's motor housing to quiet it?"),
        ("procedure", "What adjustment did {name} make to the rattling USB fan's base?"),
    ),
    (
        ("cause", "Why does {name} use a short exposure when photographing falling leaves?"),
        ("setup", "What autofocus mode does {name} choose for pictures of falling leaves?"),
    ),
    (
        ("duration", "How long does {name} expose the camera when recording star trails?"),
        ("cause", "Why does {name} keep the shutter open while photographing faint star trails?"),
    ),
    (
        ("procedure", "How does {name} duplicate the edited picture collection each Friday?"),
        ("location", "Where does {name} put the memory cards holding the extra Friday photo copy?"),
    ),
    (
        ("cause", "Why does {name} keep the panoramic image drives in separate places?"),
        ("setup", "What kind and color of portable drives does {name} use for panoramic image masters?"),
    ),
)


def augment_relation_paraphrases(group, *, split="fit"):
    """Append two supported question families per existing fit relation fact.

    The sole source is the current 48-fact authored variation world. Other
    supplied facts and questions remain opaque and unchanged. Each addition
    keeps the original complete source, semantic-specific annotation and
    same-person alternatives; names occur in ordinary questions rather than
    a supplied reference prefix. Family IDs participate in the trainer's
    existing balanced relation sampler. Applying this twice is an error.
    """
    if split != "fit":
        return deepcopy(group)
    from neural_activation_curriculum import relation_variation_fit_group

    authored = relation_variation_fit_group()
    canonical = {fact["text"]: fact for fact in authored["facts"]}
    canonical_queries = {row["support_texts"][0]: row for row in authored["queries"]}
    pattern_ids = {fact["pattern_id"] for fact in authored["facts"]}
    result = deepcopy(group)
    if len({row["text"] for row in result["queries"]}) != len(result["queries"]):
        raise ValueError("relation paraphrases require unique existing questions")
    selected = [fact for fact in result["facts"] if fact.get("pattern_id") in pattern_ids]
    if len(selected) != len(canonical) or {fact["text"] for fact in selected} != canonical.keys():
        raise ValueError("relation paraphrases require the complete authored fit bank")
    for fact in selected:
        if fact != canonical[fact["text"]]:
            raise ValueError("relation paraphrase source differs from its authored fit fact")
    by_text = {row["text"]: row for row in result["queries"]}
    existing = set(by_text)
    additions = []
    for fact in selected:
        teacher = canonical_queries[fact["text"]]
        original = by_text.get(teacher["text"])
        if original is None or any(original.get(key) != value for key, value in teacher.items()):
            raise ValueError("relation paraphrases require unchanged authored question targets")
        if original.get("reference") or original.get("initial_reference"):
            raise ValueError("relation paraphrases cannot supply a query reference")
        index = int(fact["pattern_id"].rsplit("-", 1)[1])
        for family, template in RELATION_PARAPHRASE_FAMILIES[index]:
            text = template.format(name=fact["entity"])
            if text in existing:
                raise ValueError("duplicate relation paraphrase question")
            row = deepcopy(original)
            row.update(
                text=text,
                query_class="explicit",
                relation_paraphrase=True,
                fit_paraphrase_id=f"{fact['pattern_id']}::{family}",
                question_family=family,
                observation_id=f"relation-paraphrase-fit::{fact['entity']}::{index}::{family}",
            )
            existing.add(text)
            additions.append(row)
    result["queries"] += additions
    return result


READER_BINDING_PROPERTIES = ("instrument_check", "instrument_storage", "instrument_care")
# Every object has all three properties. Object nouns, actors and record state
# therefore cannot identify the requested relation by themselves. These are
# independent fit declarations, not variants of evaluation questions.
READER_BINDING_PATTERNS = (
    (
        "dial-caliper", "dial caliper",
        (
            ("I test my dial caliper on an 8 mm block before measuring.",
             "How does {name} check whether the dial caliper reads accurately?",
             "What do they compare the dial caliper with before measuring?"),
            ("My dial caliper rests in a foam case on the workshop shelf.",
             "Where does {name} put the dial caliper between measuring jobs?",
             "Which container holds their dial caliper between sessions?"),
            ("I brush brass chips from my dial caliper's jaws after each job.",
             "What does {name} clear from the dial caliper after cutting brass?",
             "How do they clear the dial caliper's jaw slots?"),
        ),
    ),
    (
        "kitchen-scale", "kitchen scale",
        (
            ("I check my kitchen scale with a 200 g weight before baking.",
             "What does {name} use to test the kitchen scale before baking?",
             "How do they find out whether the kitchen scale reads correctly?"),
            ("My kitchen scale sits on a felt mat in the shallow drawer.",
             "What supports {name}'s kitchen scale when it is put away?",
             "Where do they leave the kitchen scale between baking sessions?"),
            ("I wipe flour off my kitchen scale's platform after baking.",
             "How does {name} care for the kitchen scale after baking?",
             "What do they remove from the kitchen scale's platform?"),
        ),
    ),
    (
        "usb-microscope", "USB microscope",
        (
            ("I check my USB microscope's focus on a printed grid before use.",
             "How does {name} test the USB microscope before using it?",
             "Which target do they use to check the USB microscope's focus?"),
            ("My USB microscope stays in a padded box beside the computer.",
             "Where does {name} keep the USB microscope between uses?",
             "What protects their USB microscope while it is put away?"),
            ("I clear dust from my USB microscope's stage with a soft brush.",
             "What does {name} use to remove dust from the USB microscope?",
             "How do they clear the USB microscope's stage?"),
        ),
    ),
    (
        "bicycle-pump", "bicycle pump",
        (
            ("I compare my bicycle pump's gauge with a handheld meter weekly.",
             "How does {name} check the bicycle pump's pressure reading?",
             "How often do they compare the bicycle pump's gauge with a meter?"),
            ("My bicycle pump hangs upright on a rack with its hose looped.",
             "How does {name} arrange the bicycle pump when it is put away?",
             "Where do they leave the bicycle pump and its hose?"),
            ("I wipe mud from my bicycle pump's valve head after outdoor rides.",
             "What does {name} clean from the bicycle pump after outdoor rides?",
             "Which part of their bicycle pump gets wiped after a muddy ride?"),
        ),
    ),
)


def reader_binding_fit_group(*, split="fit", actors=("Aileen", "Bram", "Celia")):
    """Independent same-object/different-property fit world, before NN binding.

    Each actor/object has three complete facts, explicit/context questions for
    each fact, a two-fact question and an absent-property question. Context
    supplies only a conversational subject, never the answer. Source ranges
    are writer-schema supervision; query reference fields remain absent until
    the frozen learned writer predicts them. Existing curricula are untouched.
    """
    if split != "fit":
        raise ValueError("reader binding curriculum is fit-only")
    from prepare_neural_colloquial import NAMES, TIMESTAMPS, _source_range
    from neural_slot_model import observation_context

    if (not isinstance(actors, (tuple, list)) or len(actors) != 3
            or any(type(actor) is not str or actor not in NAMES["fit"] for actor in actors)
            or len(set(actors)) != 3):
        raise ValueError("reader binding needs three distinct existing fit actors")
    actors = tuple(actors)
    timestamp = TIMESTAMPS["fit"]
    facts, queries = [], []

    def observation(body, speaker, past, source_id, *, references=(), contextual=False):
        context = observation_context(speaker, timestamp, past)
        ranges = [_source_range(context, actor, last=contextual) for actor in references]
        return dict(body=body, speaker=speaker, timestamp=timestamp, past=past,
                    source_id=source_id, context=context, spans=[],
                    whole_need=bool(references), whole_refs=list(references),
                    whole_ref_ranges=ranges, tag="episode-query")

    for index, actor in enumerate(actors):
        other = actors[(index + 1) % len(actors)]
        for object_id, object_name, patterns in READER_BINDING_PATTERNS:
            own = []
            for prop, (body, _, _) in zip(READER_BINDING_PROPERTIES, patterns):
                source_id = f"reader-binding-fit::{actor}::{object_id}::{prop}"
                obs = observation(body, actor, "", source_id, references=[actor])
                span = dict(start=0, end=len(body.encode()), action=1, need=True,
                            references=[actor], ref_ranges=deepcopy(obs["whole_ref_ranges"]),
                            prop=prop, kind=0)
                obs.update(spans=[span], tag="episode-fact", entities=[actor], prop=prop, kind=0)
                fact = dict(text=actor + ". " + body, content=body, entity=actor,
                            reference=actor, prop=prop, kind=0, status=0,
                            binding_fit=True, binding_object=object_id,
                            pattern_id=f"reader-binding-fit::{object_id}::{prop}",
                            observation=obs)
                if len(fact["text"].encode()) > 80:
                    raise ValueError("reader binding fact exceeds source budget")
                own.append(fact)
                facts.append(fact)

            def add(body, props, supports, family, *, contextual=False, query_class="explicit"):
                past = (f"user: {other}: We could discuss the {object_name} owned by {other} or {actor}.\n"
                        f"assistant: Let's focus on {actor}'s {object_name} now.\n") if contextual else ""
                source_id = f"reader-binding-fit::{actor}::{object_id}::{family}"
                obs = observation(body, "User", past, source_id,
                                  references=[actor] if contextual else [], contextual=contextual)
                obs.update(entities=[actor], prop=props, kind=0, query_class=query_class)
                supported = [fact["text"] for fact in supports]
                contrasts = [fact["text"] for fact in own if fact["text"] not in supported]
                queries.append(dict(text=body, entities=[actor], prop=props, kind=0,
                                    allowed_kinds=[0], support_texts=supported,
                                    semantic_specific=bool(props), binding_fit=True,
                                    binding_object=object_id, binding_contrast_texts=contrasts,
                                    pattern_id=f"reader-binding-fit::{object_id}::{family.rsplit('::', 1)[0]}",
                                    fit_paraphrase_id=family, query_class=query_class,
                                    observation_id=source_id, observation=obs))

            for prop, fact, (_, explicit, contextual) in zip(READER_BINDING_PROPERTIES, own, patterns):
                add(explicit.format(name=actor), prop, [fact], prop + "::explicit")
                add(contextual, prop, [fact], prop + "::context", contextual=True, query_class="context")
            add(f"Describe {actor}'s checks and cleaning routine for the {object_name}.",
                [READER_BINDING_PROPERTIES[0], READER_BINDING_PROPERTIES[2]],
                [own[0], own[2]], "check-and-care::explicit")
            add(f"What did {actor} pay for the {object_name}?", "", [], "absent-price::null", query_class="null")
    return dict(facts=facts, queries=queries)


def reader_binding_feature_texts(group):
    """Exact raw C-feature inputs; NN-predicted query prefixes come later."""
    if (set(group) != {"facts", "queries"} or not group["facts"] or not group["queries"]
            or any(not row.get("binding_fit") or row.get("evaluation_only")
                   for rows in group.values() for row in rows)):
        raise ValueError("reader binding features require authored fit rows")
    texts = set()
    for rows in group.values():
        for row in rows:
            obs = row["observation"]
            texts.update((row["text"], obs["body"], obs["context"]))
            if row.get("reference"):
                texts.add(row["reference"])
    for row in group["queries"]:
        if row["text"] != row["observation"]["body"] or row.get("reference") or row.get("initial_reference"):
            raise ValueError("reader binding queries require raw unprefixed observations")
        support = [fact["text"] for fact in group["facts"] if _supports(row, fact)]
        if set(support) != set(row["support_texts"]):
            raise ValueError("reader binding supports must belong to the complete fit bank")
    return sorted(texts)


def _values(value):
    if value is None:
        return set()
    return {value} if isinstance(value, str) else set(value)


def _entities(row):
    return _values(row.get("entities", row.get("entity")))


def _supports(question, fact):
    if "support_texts" in question:
        return fact["text"] in question["support_texts"]
    return (
        bool(_entities(question) & _entities(fact))
        and fact["prop"] in _values(question["prop"])
        and ("kind" not in question or question["kind"] == fact["kind"])
    )


def _question_order(questions, rng):
    """Round-robin relations, temporal kinds, and one/multiple/no subjects."""
    buckets = defaultdict(list)
    for question in questions:
        props = tuple(sorted(_values(question["prop"])))
        key = (props, question.get("kind", -1), len(_entities(question)))
        buckets[key].append(question)
    keys = sorted(buckets)
    for key in keys:
        buckets[key].sort(key=lambda q: q["text"])
        rng.shuffle(buckets[key])
    rng.shuffle(keys)
    while keys:
        remaining = []
        for key in keys:
            yield buckets[key].pop()
            if buckets[key]:
                remaining.append(key)
        keys = remaining


def filter_bank_evidence_queries(rows, bank_facts):
    """Keep raw fit questions and cues that exist in the current bank.

    This validates exact authored provenance and the activation text format;
    it neither infers relevance nor changes any query's support labels. The
    returned rows retain their original order and objects. Call after sampling
    an alternative world, before partitioning its present/absent questions.
    """
    if any(not isinstance(fact.get("text"), str) or not fact["text"] for fact in bank_facts):
        raise ValueError("evidence bank requires nonempty factual text")
    present = {fact["text"] for fact in bank_facts}
    retained = []
    for row in rows:
        augmented = row.get("augmented", False)
        if type(augmented) is not bool:
            raise ValueError("fit evidence requires a boolean augmented marker")
        sources = row.get("evidence_sources")
        count = row.get("evidence_count", 0)
        if not augmented:
            if sources not in (None, []) or type(count) is not int or count != 0:
                raise ValueError("raw fit question cannot declare activated evidence")
            retained.append(row)
            continue
        initial = row.get("initial_text")
        if (not isinstance(initial, str) or not initial
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, str) or not source for source in sources)
            or len(set(sources)) != len(sources)
            or type(count) is not int or count != len(sources)):
            raise ValueError("augmented fit question has invalid evidence provenance")
        expected = initial + "\nActivated evidence:\n" + "\n".join(sources) + "\n"
        if row.get("text") != expected:
            raise ValueError("augmented fit text disagrees with evidence provenance")
        if set(sources) <= present:
            retained.append(row)
    return retained


def augment_evidence(group, seed, max_queries=600, *, split="fit"):
    """Return originals plus at most ``max_queries`` conditioned fit questions.

    Each selected question gets one to four cues: individual supporting or
    distracting facts, supporting facts mixed with noise, and noise-only sets.
    Complete cue families are preferred when the budget is tight, so a query's
    good/bad contrast survives subsampling. Other splits are copied unchanged.

    The original support labels are copied unchanged for every cue: wrong
    evidence must neither introduce a new answer nor replace the target. The
    same-actor distractor differs in relation; the other-actor distractor
    prefers the same relation, making subject drift a useful hard negative.
    ``max_queries`` counts additions, and is capped at 600 to bound new frozen
    backbone features. Existing augmented rows are not recursively expanded.
    """
    if type(seed) is not int or type(max_queries) is not int or max_queries < 0:
        raise ValueError("seed must be an integer and max_queries nonnegative")
    if split != "fit":
        return deepcopy(group)
    limit = min(max_queries, 600)
    rng = random.Random(seed)
    originals = list(group["queries"])
    facts = sorted(group["facts"], key=lambda f: f["text"])
    unique = {}
    for question in originals:
        if (
            not question.get("augmented")
            and "\nActivated evidence:\n" not in question["text"]
        ):
            unique.setdefault(question["text"], question)
    existing = {q["text"] for q in originals}
    additions = []
    for question in _question_order(list(unique.values()), rng):
        if len(additions) >= limit:
            break
        entities, props = _entities(question), _values(question["prop"])
        supporting = [f for f in facts if _supports(question, f)]
        same_actor = [
            f
            for f in facts
            if bool(entities & _entities(f))
            and f["prop"] not in props
            and not _supports(question, f)
        ]
        other_actor = [
            f
            for f in facts
            if _entities(f)
            and not (entities & _entities(f))
            and not _supports(question, f)
        ]
        same_relation_other_actor = [f for f in other_actor if f["prop"] in props]
        choices = []
        for cue_type, candidates in (
            ("supporting", supporting),
            ("same_actor_wrong_relation", same_actor),
            ("other_actor", same_relation_other_actor or other_actor),
        ):
            if candidates:
                choices.append((cue_type, [rng.choice(candidates)], "single"))
        if not choices:
            continue
        distracting = [f for f in facts if not _supports(question, f)]

        def cue_set(count, first, candidates):
            """Distinct complete facts; cue order may put distracting names first."""
            selected = [first]
            remaining = [f for f in candidates if f["text"] != first["text"]]
            if len(remaining) < count - 1:
                return None
            selected += rng.sample(remaining, count - 1)
            rng.shuffle(selected)
            return selected

        for count in range(2, 5):
            if supporting and distracting:
                support = rng.choice(supporting)
                noise = rng.choice(same_actor or same_relation_other_actor or distracting)
                remaining = [f for f in facts if f["text"] not in {support["text"], noise["text"]}]
                if len(remaining) >= count - 2:
                    mixed = [support, noise] + rng.sample(remaining, count - 2)
                    rng.shuffle(mixed)
                    choices.append(("supporting", mixed, "support_and_noise"))
            if distracting:
                first = rng.choice(same_relation_other_actor or other_actor or distracting)
                noise = cue_set(count, first, distracting)
                if noise:
                    cue_type = "other_actor" if not (entities & _entities(first)) else "same_actor_wrong_relation"
                    choices.append((cue_type, noise, "noise_only"))
            if supporting:
                support = cue_set(count, rng.choice(supporting), supporting)
                if support:
                    choices.append(("supporting", support, "support_only"))
        family = []
        family_texts = set()
        for cue_type, cues, mixture in choices:
            text = question["text"] + "\nActivated evidence:\n" + "\n".join(f["text"] for f in cues) + "\n"
            if text in existing or text in family_texts:
                continue
            row = deepcopy(question)
            row.update(
                text=text,
                initial_text=question["text"],
                augmented=True,
                evidence_type=cue_type,
                evidence_mix=mixture,
                evidence_count=len(cues),
                evidence_sources=[f["text"] for f in cues],
            )
            # The original reference was predicted by the frozen query planner.
            # Preserve it for anchor identity; neither cue subjects nor author
            # entity labels may supply a replacement. The trainer removes the
            # current reference from augmented input before learned identity.
            row.pop("initial_reference", None)
            if question.get("reference"):
                row["initial_reference"] = question["reference"]
            family.append(row)
            family_texts.add(text)
        # Prefer intact contrastive families. Tiny caller budgets still produce
        # a bounded example, while normal 600-row runs keep good/wrong cues.
        remaining = limit - len(additions)
        if len(family) > remaining:
            if additions:
                continue
            family = family[:remaining]
        for row in family:
            existing.add(row["text"])
            additions.append(row)
    return dict(group, facts=list(group["facts"]), queries=originals + additions)


def split_evidence_views(group):
    """Separate base and conditioned questions for component reporting."""
    return {
        name: dict(
            group,
            facts=list(group["facts"]),
            queries=[
                q for q in group["queries"] if bool(q.get("augmented")) == augmented
            ],
        )
        for name, augmented in (("base", False), ("augmented", True))
    }
