"""Fit-only authored query corrections and independent relation paraphrases.

These functions transform training records before frozen-writer query binding.
They read no external data and never participate in native memory decisions.
Exact template edits describe the independently authored fit curriculum, rather
than applying a semantic keyword rule to arbitrary observations.
"""

from collections import defaultdict
from copy import deepcopy

import torch

from neural_slot_model import observation_context


BROAD_EPISODE_QUESTIONS = {
    "collection": (
        "What does {name} collect?",
        "What is {name}'s collecting hobby?",
    ),
    "relaxation": (
        "What helps {name} relax?",
        "What is {name}'s usual way of resting?",
    ),
}

# Each rewrite retains its original, complete supporting fact. Explicit objects
# distinguish simultaneously present career changes and neighborhood projects.
EPISODE_REWRITES = (
    (
        "work_change",
        "Why did {name} change occupations?",
        "Why did {name} switch from repairing radios to restoring clocks?",
        "I switched from repairing radios to restoring clocks because I wanted more flexible hours.",
    ),
    (
        "work_change",
        "What work did {name} switch to?",
        "What work did {name} switch to after repairing radios?",
        "I switched from repairing radios to restoring clocks because I wanted more flexible hours.",
    ),
    (
        "work_change",
        "How does {name} earn a living now?",
        "What work pays {name}'s bills after leaving warehouse packing?",
        "Making cabinets now pays my bills; I left warehouse packing so I could spend more time creating things.",
    ),
    (
        "project_purpose",
        "What motivated {name}'s neighborhood project?",
        "What motivated {name} to organize the neighborhood repair club?",
        "I organized a neighborhood repair club because I wanted people to share tools and reduce waste.",
    ),
)

# Author annotations for the colliding fit relations. A specific question can
# reject a different object/event within the same property. Unannotated paired
# alternatives are deliberately left out of the semantic negative teacher.
SPECIFIC_EPISODE_QUESTIONS = {
    "collection": (
        "Where does {name} keep the postcard collection?",
        "How does {name} sort the stamps?",
    ),
    "relaxation": (
        "How does {name} unwind after a demanding day?",
        "What does {name} drink while doing crosswords?",
    ),
    "work_change": (
        "Why did {name} switch from repairing radios to restoring clocks?",
        "What work did {name} switch to after repairing radios?",
        "What work pays {name}'s bills after leaving warehouse packing?",
        "Why did {name} leave warehouse packing?",
    ),
    "project_purpose": (
        "Why did {name} organize a repair club?",
        "What motivated {name} to organize the neighborhood repair club?",
        "Why did {name} start the seed-sharing table?",
        "Where is {name}'s seed-sharing project located?",
    ),
}


def repair_fit_episode_support(group, split):
    """Repair only known fit teachers, preserving every fact and held-out row."""
    result = deepcopy(group)
    if split != "fit":
        return result
    sources = defaultdict(list)
    facts = {fact["text"]: fact for fact in result["facts"]}
    for fact in result["facts"]:
        sources[fact["entity"], fact["prop"], fact["kind"]].append(fact["text"])
    for query in result["queries"]:
        entities = query.get("entities", [])
        if len(entities) != 1:
            continue
        name = entities[0]
        broad = BROAD_EPISODE_QUESTIONS.get(query["prop"], ())
        if query["text"] in {template.format(name=name) for template in broad}:
            support = sorted(sources[name, query["prop"], query["kind"]])
            if not support:
                raise ValueError("broad fit query has no matching factual source")
            query["support_texts"] = support
            query["fit_support_revision"] = "all-related-episode-facts"
        for prop, old, new, source in EPISODE_REWRITES:
            if query["prop"] != prop or query["text"] != old.format(name=name):
                continue
            expected = name + ". " + source
            if query.get("support_texts") != [expected] or expected not in facts:
                raise ValueError("fit query rewrite does not match its authored source")
            observation = query.get("observation")
            if observation is None or observation["body"] != query["text"]:
                raise ValueError("fit query rewrite requires its raw observation")
            query["text"] = observation["body"] = new.format(name=name)
            query["fit_support_revision"] = "source-specific-episode-question"
            break
        specific = SPECIFIC_EPISODE_QUESTIONS.get(query["prop"], ())
        if query["text"] in {template.format(name=name) for template in specific}:
            query["semantic_specific"] = True
    return result


def _values(value):
    if value is None:
        return set()
    return {value} if isinstance(value, (str, int)) else set(value)


def _actor_free_content(text, actors):
    for name in sorted(actors, key=lambda actor: (-len(actor), actor)):
        prefix = name + ". "
        if text.startswith(prefix):
            return text[len(prefix):]
    raise ValueError("semantic teacher requires an exact authored actor prefix")


def semantic_relation_masks(rows, facts, hardness, max_negatives=4):
    """Actor-invariant semantic teachers, separate from activation labels.

    Complete content/property/source-kind equivalence transfers an exact
    supporting fact to other fit actors. An explicit fact source_kind labels
    its original text independently of physical state. Final activation still
    uses its original actor support labels. Negative selection is training-only
    and bounded: same actor with another property, or an author-annotated
    specific question's different same-property content in a requested source
    kind. Broad and unannotated alternatives are never negative.
    """
    if (
        type(max_negatives) is not int
        or not 0 <= max_negatives <= 4
        or hardness.shape != (len(rows), len(facts))
        or not torch.isfinite(hardness).all()
    ):
        raise ValueError("invalid bounded semantic teacher geometry or hardness")
    fact_keys = [
        (_actor_free_content(fact["text"], {fact["entity"]}), fact["prop"],
         fact.get("source_kind", fact["kind"]))
        for fact in facts
    ]
    if any(type(key[2]) is not int or key[2] not in (0, 1, 2) for key in fact_keys):
        raise ValueError("semantic teacher requires original source kinds in 0..2")
    known = defaultdict(set)
    for fact, key in zip(facts, fact_keys):
        known[fact["text"]].add(key)
    positives, eligible, alternatives = [], [], []
    for row in rows:
        actors = _values(row.get("entities", row.get("entity")))
        props = _values(row["prop"])
        # Lifecycle state can be historical while its stored original text is
        # still present tense. These author labels describe source text kind,
        # separately from the query's requested physical version channels.
        source_kinds = _values(row.get("source_kinds", row.get("kind")))
        source_keys = set()
        if "support_texts" not in row and props:
            # Broad multi-person authors can specify supports by authored
            # entity/property/original-source-kind labels.
            # Only a missing key takes this path: explicit [] means no source.
            source_keys.update(
                key for fact, key in zip(facts, fact_keys)
                if fact["entity"] in actors
                and fact["prop"] in props
                and (not source_kinds or key[2] in source_kinds)
            )
        for text in _values(row.get("support_texts")):
            if text in known:
                matches = {
                    key for key in known[text]
                    if key[1] in props and (not source_kinds or key[2] in source_kinds)
                }
                if not matches:
                    raise ValueError("exact semantic support disagrees with property or kind")
                source_keys.update(matches)
            else:
                # Expansion can sample a world where the exact source is absent.
                # Its paired single-property label still defines the complete
                # actor-free source; no current bank fact supplies that label.
                if len(props) != 1 or len(source_kinds) != 1:
                    raise ValueError("missing semantic source needs one property and kind")
                source_keys.add(
                    (_actor_free_content(text, actors), next(iter(props)), next(iter(source_kinds)))
                )
        positive = [key in source_keys for key in fact_keys]
        positives.append(positive)
        eligible.append(
            [
                fact["entity"] in actors
                and not is_positive
                and (
                    fact["prop"] not in props
                    or (
                        row.get("semantic_specific", False)
                        and fact["prop"] in props
                        and key[2] in source_kinds
                    )
                )
                for fact, key, is_positive in zip(facts, fact_keys, positive)
            ]
        )
        alternatives.append([
            bool(row.get("semantic_specific", False)) and is_eligible and fact["prop"] in props
            for fact, is_eligible in zip(facts, eligible[-1])
        ])
    positive = torch.tensor(positives, dtype=torch.bool, device=hardness.device).reshape_as(hardness)
    eligible = torch.tensor(eligible, dtype=torch.bool, device=hardness.device).reshape_as(hardness)
    negative = torch.zeros_like(eligible)
    count = min(max_negatives, len(facts))
    if count:
        # A specifically authored object question must see its paired alternative,
        # even while other properties have larger current model scores.
        alternative = torch.tensor(alternatives, dtype=torch.bool, device=hardness.device).reshape_as(hardness)
        first = hardness.detach().masked_fill(~alternative, -torch.inf).argmax(-1, keepdim=True)
        negative.scatter_(1, first, alternative.gather(1, first))
        remaining = eligible & ~negative
        indices = hardness.detach().masked_fill(~remaining, -torch.inf).topk(count, -1).indices
        available = remaining.gather(1, indices)
        budget = count - negative.sum(-1, keepdim=True)
        available &= torch.arange(count, device=hardness.device)[None] < budget
        negative.scatter_(1, indices, negative.gather(1, indices) | available)
    return positive, negative


def sample_relation_queries(rows, rng, count):
    """Cover relations first, then authored question and evidence families."""
    if type(count) is not int or count < 0:
        raise ValueError("relation batch size must be a nonnegative integer")
    if any(row.get("evaluation_only") for row in rows):
        raise ValueError("evaluation queries cannot enter relation sampling")
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        relation = tuple(sorted(_values(row.get("prop"))))
        family = (row.get("pattern_id"), row.get("fit_paraphrase_id"),
                  row.get("query_class"), row.get("evidence_type"),
                  row.get("evidence_mix"), row.get("evidence_count", 0))
        groups[relation][family].append(index)
    keys = list(groups)
    rng.shuffle(keys)
    families = {}
    for key in keys:
        families[key] = list(groups[key])
        rng.shuffle(families[key])
    selected = []
    # A relation with many templates must not crowd a rare relation out of the
    # batch. Successive passes visit different families of every relation.
    pending = list(keys)
    while pending and len(selected) < count:
        remaining_keys = []
        for key in pending:
            if len(selected) == count:
                break
            family = families[key].pop()
            selected.append(rng.choice(groups[key][family]))
            if families[key]:
                remaining_keys.append(key)
        pending = remaining_keys
    used = set(selected)
    remaining = [index for index in range(len(rows)) if index not in used]
    selected += rng.sample(remaining, min(max(count - len(selected), 0), len(remaining)))
    rng.shuffle(selected)
    return [rows[index] for index in selected]


def episode_relation_variation_fit_group():
    """Independent fit content variants; no evaluation files are consulted.

    Specific object questions select one source. Broad questions require all
    facts of that person's relation, rather than treating other instances as
    negative labels. Both query types receive explicit author annotations.
    """
    patterns = (
        ("relaxation", "Folding paper cranes at the kitchen table settles my mind when a busy week ends.",
         "Which craft helps {name} find calm after a busy week?"),
        ("relaxation", "Tending my balcony herbs gives me a quiet break after a crowded shift.",
         "What does {name} tend to take a quiet break after a shift?"),
        ("relaxation", "A few gentle tunes on my wooden flute let me slow down when my thoughts race.",
         "Which instrument does {name} play to slow racing thoughts?"),
        ("relaxation", "I take slow walks beside the river to regain peace of mind before supper.",
         "Where does {name} walk to regain peace of mind?"),
        ("collection", "I gather antique chess pieces and arrange them by their makers in a glass cabinet.",
         "What does {name} arrange by maker in a glass cabinet?"),
        ("collection", "Unusual transit tokens are what I collect; each city's tokens go into a separate felt pouch.",
         "Which objects does {name} put in separate pouches for each city?"),
        ("collection", "My collection consists of illustrated seed labels, filed by plant family in a flat box.",
         "What does {name} file by plant family in a flat box?"),
        ("collection", "I enjoy collecting ceramic buttons and displaying the patterned ones on a cork board.",
         "Which collected objects does {name} display on a cork board?"),
        # Alternative realizations of fit relations with fresh values. They
        # were added after development questions in these wordings failed while
        # their stored facts held the requested detail. The development
        # families that did not respond to a broader corpus are left out so
        # relation variation does not dilute the authored fit world. No
        # evaluation row is copied.
        ("dietary", "Rich cream sauces make me feel ill, so I ask for plain dishes when eating out.",
         "Which food makes {name} feel ill?"),
        ("family_plan", "This winter I hope to show my cousin how to assemble a birdhouse for her club.",
         "What does {name} hope to assemble with their cousin?"),
        ("event_date", "On 3 April I went to the butterfly house with my niece and stayed until closing.",
         "On what date did {name} go to the butterfly house?"),
    )
    names = ("Alva", "Cedric", "Nerissa")
    facts, queries = [], []
    for name in names:
        for index, (prop, body, question) in enumerate(patterns):
            text = name + ". " + body
            pattern = f"episode-relation-fit-{index}"
            facts.append(dict(text=text, content=body, entity=name, reference=name,
                              prop=prop, kind=0, relation_variation=True, pattern_id=pattern))
            queries.append(dict(text=question.format(name=name), entities=[name], prop=prop,
                                kind=0, allowed_kinds=[0], support_texts=[text],
                                semantic_specific=True, pattern_id=pattern))
        for prop, question in (("relaxation", "Describe the ways {name} finds calm."),
                               ("collection", "Which things does {name} enjoy collecting?")):
            queries.append(dict(text=question.format(name=name), entities=[name], prop=prop,
                                kind=0, allowed_kinds=[0],
                                support_texts=[f["text"] for f in facts
                                               if f["entity"] == name and f["prop"] == prop],
                                pattern_id=f"episode-relation-broad-{prop}"))
    return dict(facts=facts, queries=queries)


# Each supplied fit episode fact already answers several subordinate aspects.
# These independently worded fit-only questions ask for the value, reason,
# place, time or participant named in that same fact. They exist because
# development questions that request one aspect of a fact were not activated
# even when the query shared its content words. No evaluation text is used.
EPISODE_ASPECT_QUESTIONS = {
    "work_change": (
        "Which trade does {name} practice after leaving radio repair?",
        "What occupation currently provides {name}'s income?",
        "Why did {name} stop repairing radios?",
        "What kind of clocks does {name} restore for a living?",
    ),
    "material_preference": (
        "Which timber does {name} favor when building small boats?",
        "What wood is {name}'s preferred choice for model vessels?",
        "Why does {name} pick cedar for miniature boats?",
        "What quality makes cedar suitable for {name}'s models?",
    ),
    "relaxation": (
        "What music settles {name} after a tiring day?",
        "How does {name} clear their head after work?",
        "What does {name} listen to in order to unwind?",
        "Which activity helps {name} unwind and clear their thoughts?",
    ),
    "project_purpose": (
        "Why did {name} set up a repair club in the neighborhood?",
        "What did {name} want neighbors to share?",
        "What was {name}'s aim in starting the repair club?",
        "What waste does {name}'s club try to reduce?",
    ),
    "event_date": (
        "What date did {name} tour the railway museum?",
        "Who went with {name} to the railway museum?",
        "When did {name} take a guided museum tour?",
        "Which relative accompanied {name} to the railway museum?",
    ),
    "collection": (
        "What kind of postcards does {name} collect?",
        "Where does {name} store the harbor postcards?",
        "What color is the album holding {name}'s postcards?",
        "Which scenes appear on the postcards {name} gathers?",
    ),
    "residence": (
        "Which city does {name}'s apartment stand in?",
        "What is within walking distance of {name}'s home?",
        "Where did {name} move recently?",
        "How far is {name}'s apartment from the library?",
    ),
    "family_plan": (
        "What does {name} plan to build with their nephew?",
        "When will {name} help with the telescope project?",
        "What is the telescope for at {name}'s nephew's school?",
        "Which relative will {name} assist next autumn?",
    ),
    "dietary": (
        "Which nut makes {name} feel unwell?",
        "Why does {name} stay away from hazelnuts?",
        "What happens to {name} after eating hazelnuts?",
        "Which snack ingredient does {name} leave out?",
    ),
    "course": (
        "What subject does {name} study in the evening?",
        "Which course did {name} sign up for?",
        "Why is {name} taking an astronomy class?",
        "What does {name} hope to learn about observatories?",
    ),
    "achievement": (
        "What prize did {name}'s bookshelf win?",
        "When did {name} receive the craftsmanship award?",
        "Where was {name}'s bookshelf shown?",
        "What did {name} make that won an award?",
    ),
    "possession": (
        "What sort of lamp did {name} buy?",
        "Where does {name} keep the reading lamp?",
        "What feature does {name}'s reading lamp have?",
        "What is the lamp beside {name}'s sewing machine made of?",
    ),
}


def episode_aspect_variation_fit_group():
    """Fit-only aspect questions over the independently authored fit facts.

    Every question is supported by exactly one fit episode fact, whose own
    content named the requested detail. The group adds no evaluation world and
    reads no external data. It is merged into relation variation so the
    balanced sampler visits all twelve relations.
    """
    from prepare_neural_episodes import NAMES, PATTERNS

    if set(EPISODE_ASPECT_QUESTIONS) != {entry[0] for entry in PATTERNS["fit"]}:
        raise ValueError("episode aspects require one question family per fit relation")
    names = tuple(NAMES["fit"][:4])
    seen = set()
    selected = []
    for prop, body, _, _, kind in PATTERNS["fit"]:
        if prop in seen:
            continue
        seen.add(prop)
        selected.append((prop, body, kind))
    facts, queries = [], []
    for name in names:
        for prop, body, kind in selected:
            text = name + ". " + body
            pattern = f"episode-aspect-fit-{prop}"
            facts.append(dict(text=text, content=body, entity=name, reference=name,
                              prop=prop, kind=kind, relation_variation=True,
                              pattern_id=pattern))
            for index, template in enumerate(EPISODE_ASPECT_QUESTIONS[prop]):
                question = template.format(name=name)
                queries.append(dict(
                    text=question, entities=[name], prop=prop, kind=kind,
                    allowed_kinds=[kind], support_texts=[text],
                    semantic_specific=True, pattern_id=pattern,
                    question_family=f"aspect-{index}",
                ))
    return dict(facts=facts, queries=queries)


# Six independently worded questions per actual colloquial fit relation. These
# ask for relation content or a subfact, rather than adding a generic prefix to
# an existing question. The full relevant source remains the support label.
FIT_RELATION_QUESTIONS = {
    "paint_choice": (
        "How has {name} painted the pantry shelving?",
        "Describe {name}'s choice of color and finish for the shelves.",
        "Which coating went onto {name}'s pantry shelves?",
        "What surface treatment makes {name}'s painted shelving easier to clean?",
        "If a splash lands on the shelves, what makes cleanup possible for {name}?",
        "Tell me about the paint {name} selected for the pantry.",
    ),
    "window_seal": (
        "How did {name} make the window more resistant to rain?",
        "Which replacement seals the edge of {name}'s window?",
        "Describe the change {name} made to stop water reaching the sill.",
        "What improvement did {name} make around the window frame?",
        "What is keeping rainwater off {name}'s sill?",
        "Tell me how {name} dealt with the worn window seal.",
    ),
    "plant_watering": (
        "Describe {name}'s morning routine for the fern.",
        "How does {name} manage moisture around the fern's roots?",
        "What care does the fern receive from {name} before the first meal of the day?",
        "Why does {name} remove the water left beneath the fern?",
        "What does {name} do after spraying the fern?",
        "Tell me about {name}'s method of watering the fern.",
    ),
    "plant_soil": (
        "Describe the growing medium in {name}'s herb pots.",
        "How does {name} adjust the herb soil when it stays too wet?",
        "Which addition improves drainage in {name}'s herb mix?",
        "What tells {name} that the herbs' potting mixture needs changing?",
        "What sort of soil do {name}'s herbs grow in?",
        "Tell me how {name} handles excess moisture in the herb pots.",
    ),
    "lamp_repair": (
        "Describe the repair that made {name}'s lamp work again.",
        "Which cable did {name} install during the lamp repair?",
        "What electrical connection did {name} examine in the lamp?",
        "How did {name} bring the broken lamp back into use?",
        "What covering is on the lamp's replacement cord that {name} fitted?",
        "Tell me what {name} changed and checked to restore the lamp.",
    ),
    "fan_repair": (
        "Describe how {name} serviced the noisy desk fan.",
        "Which moving part of the fan received oil from {name}?",
        "How did {name} silence the fan at its slow setting?",
        "What maintenance did {name} carry out on the fan's grille and axle?",
        "Which parts did {name} attend to when the desk fan squeaked?",
        "Tell me what made {name}'s desk fan quieter.",
    ),
    "photo_settings": (
        "Describe {name}'s setup for photographing objects on a table.",
        "Which f-stop does {name} normally choose for tabletop shots?",
        "How does {name} keep the near and far ends of a photographed object in focus?",
        "What lighting accessory accompanies {name}'s tabletop camera settings?",
        "What arrangement gives {name} sharp tabletop photographs?",
        "Tell me about the aperture and reflector in {name}'s photography setup.",
    ),
    "photo_backup": (
        "Describe {name}'s storage arrangement for raw photographs.",
        "Which portable device holds a copy of {name}'s image files?",
        "Where has {name} placed the spare copy of the raw photos?",
        "How does {name} keep two copies of the photo collection?",
        "What color is the external drive that {name} uses for photographs?",
        "Tell me about the locations of {name}'s photo backups.",
    ),
}


def add_fit_relation_queries(group, split, max_queries=400):
    """Add bounded named fit observations; never inject a labeled reference.

    Only facts marked colloquial are eligible. Supports are derived from the
    supplied fit bank, including complete augmented spans with the same key.
    This also keeps source definitions fixed when a frozen JSON is supplied.
    """
    result = deepcopy(group)
    if split != "fit":
        return result
    if not 1 <= max_queries <= 400:
        raise ValueError("fit relation paraphrases must be bounded to 1..400")
    sources = defaultdict(list)
    for fact in result["facts"]:
        if fact.get("colloquial") and fact["prop"] in FIT_RELATION_QUESTIONS:
            sources[fact["entity"], fact["prop"], fact["kind"]].append(fact["text"])
    timestamps = {
        entity: query["observation"]["timestamp"]
        for query in result["queries"]
        if "observation" in query
        for entity in query.get("entities", [])
    }
    seen = {query["text"]: query for query in result["queries"]}
    added = 0
    for (name, prop, kind), support in sorted(sources.items()):
        if name not in timestamps:
            raise ValueError("fit source has no literal observation provenance")
        for index, template in enumerate(FIT_RELATION_QUESTIONS[prop]):
            body = template.format(name=name)
            support = sorted(support)
            if body in seen:
                if sorted(seen[body]["support_texts"]) != support:
                    raise ValueError("fit paraphrase would create conflicting supports")
                continue
            if added == max_queries:
                return result
            source_id = f"reader-fit-paraphrase-{added}"
            observation = dict(
                body=body,
                speaker="User",
                timestamp=timestamps[name],
                past="",
                source_id=source_id,
                context=observation_context("User", timestamps[name], ""),
                spans=[],
                whole_need=False,
                whole_refs=[],
                whole_ref_ranges=[],
                tag="episode-query",
                entities=[name],
                prop=prop,
                kind=kind,
                query_class="explicit",
                support_keys=[dict(entity=name, prop=prop, kind=kind)],
            )
            query = dict(
                text=body,
                entities=[name],
                prop=prop,
                kind=kind,
                support_texts=support,
                support_keys=[dict(entity=name, prop=prop, kind=kind)],
                observation=observation,
                observation_id=source_id,
                query_class="explicit",
                colloquial=True,
                fit_paraphrase_id=f"{prop}-{index}",
                semantic_specific=True,
            )
            result["queries"].append(query)
            seen[body] = query
            added += 1
    return result
