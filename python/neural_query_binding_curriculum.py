"""Authored query-reference supervision with independent evaluation worlds.

Only fit labels may enter optimization. Development and sealed labels are for
evaluation; every split uses different people, objects, dates and question
wording. The returned reference ranges are supervision, never a serving rule.
This module reads no model or external dataset; it composes only the authored
generators of this repository.
"""


QUERY_BINDING_WORLDS = {
    "fit": {
        "actors": (("Quillon", "Marius"), ("Zélie", "Corinne")),
        "objects": (
            "dial indicator", "sine bar", "gauge block case", "cello endpin",
            "harmonica comb", "coin album",
        ),
        "timestamp": "2047-03-09 10:15 UTC",
        "questions": {
            "literal-name": "Where does {name} store the {object}?",
            "literal-possessive": "What case protects {name}'s {object}?",
            "topic": "What maintenance steps suit a {object}?",
            "first-person-storage": "Where do I store my {object}?",
            "first-person-care": "Which cloth do I use on my {object}?",
            "third-person": "Where does {pronoun} store the {object}?",
            "third-person-dialogue": "What does {pronoun} check first on the {object}?",
        },
    },
    "development": {
        "actors": (("Ivo", "Renwick"), ("Vesper", "Maëlle")),
        "objects": ("bassoon bocal", "oboe reed case", "mandolin bridge"),
        "timestamp": "2051-11-22 14:40 UTC",
        "questions": {
            "literal-name": "Which stand did {name} choose for the {object}?",
            "literal-possessive": "How often is {name}'s {object} serviced?",
            "topic": "How is a {object} safely transported?",
            "first-person-storage": "What container did I select for my {object}?",
            "first-person-care": "When do I inspect my {object}?",
            "third-person": "Which mount does {pronoun} prefer for the {object}?",
            "third-person-dialogue": "Which part does {pronoun} inspect first on the {object}?",
        },
    },
    "sealed": {
        "actors": (("Aldric", "Cyril"), ("Ondine", "Thérèse")),
        "objects": (
            "stamp hinge packet", "fossil display stand", "mineral specimen box",
        ),
        "timestamp": "2056-07-04 08:25 UTC",
        "questions": {
            "literal-name": "Which buffer does {name} use with the {object}?",
            "literal-possessive": "Which rack supports {name}'s {object}?",
            "topic": "What material suits the lining for a {object}?",
            "first-person-storage": "Which pad goes beneath my {object}?",
            "first-person-care": "How do I label my {object}?",
            "third-person": "What cover does {pronoun} choose for the {object}?",
            "third-person-dialogue": "What does {pronoun} change last on the {object}?",
        },
    },
}


def _context(speaker, timestamp, past):
    # This is the literal runtime provenance format, with no inferred fields.
    return f"Speaker: {speaker}\nObserved at: {timestamp}\n{past}"


def _reference(context, name, *, header=False):
    # Author a byte-exact label in the supplied text. This helper is not used
    # by inference and cannot determine which person a query should concern.
    offset = 9 if header else context.rindex(name)
    start = len(context[:offset].encode("utf-8"))
    return dict(document="context", start=start, end=start + len(name.encode("utf-8")))


def query_binding_curriculum(split):
    """Return the fit or evaluation query worlds, with no write authorizations.

    Each object has six self-contained controls, ten resolved implicit queries
    and two unresolved third-person queries. In each identical-body quartet,
    context selects person A or B. With anonymous provenance, first person is
    still User; only third person abstains on ambiguous or missing antecedents.
    The fit and development splits also carry the literal single-message
    request shape in disjoint worlds and wording.
    """
    if type(split) is not str or split not in QUERY_BINDING_WORLDS:
        raise ValueError("query binding split must be fit, development or sealed")
    world = QUERY_BINDING_WORLDS[split]
    timestamp, patterns = world["timestamp"], world["questions"]
    rows = []

    for object_index, object_name in enumerate(world["objects"]):
        pair_index = object_index % len(world["actors"])
        actors = world["actors"][pair_index]
        pronoun = "he" if pair_index == 0 else "she"
        ambiguous_history = (
            f"user: User: We are discussing {actors[0]} and {actors[1]} together.\n"
            "assistant: Both people have equal roles in this discussion.\n"
        )

        def add(family, case, body, speaker, past, binding_class, *, target=None,
                header=False):
            context = _context(speaker, timestamp, past)
            references = [target] if target is not None else []
            ranges = [_reference(context, target, header=header)] if target is not None else []
            rows.append(dict(
                body=body, context=context, speaker=speaker, timestamp=timestamp,
                past=past, source_id=f"query-binding-{split}-{object_index}-{family}-{case}",
                spans=[], whole_need=family not in ("literal-name", "literal-possessive", "topic"),
                whole_refs=references, whole_ref_ranges=ranges, tag="query-binding-query",
                query_binding_class=binding_class, query_binding_family=family,
                query_binding_case=case, query_binding_object=object_name,
                query_binding_actors=list(actors), query_binding_split=split,
                split=split, evaluation_only=split != "fit",
                evaluation_role="fit-teacher" if split == "fit" else "evaluation-only",
            ))

        # Literal subjects are already in the query. All provenance points at
        # the other person, so context must not add an unrelated subject.
        for actor_index, actor in enumerate(actors):
            other = actors[1 - actor_index]
            past = (f"user: User: We are discussing {other} alone.\n"
                    "assistant: Keep that person in focus.\n")
            case = "a" if actor_index == 0 else "b"
            for family in ("literal-name", "literal-possessive"):
                body = patterns[family].format(name=actor, object=object_name)
                add(family, case, body, other, past, family)
            add("topic", case, patterns["topic"].format(object=object_name),
                other, past, "subject-free-topic")

        for family in ("first-person-storage", "first-person-care", "third-person"):
            body = patterns[family].format(object=object_name, pronoun=pronoun)
            first_person = family.startswith("first-person-")
            for actor_index, actor in enumerate(actors):
                other = actors[1 - actor_index]
                case = "a" if actor_index == 0 else "b"
                if first_person:
                    # The header is the unique current sender; old history is
                    # intentionally about somebody else.
                    speaker = actor
                    past = (f"user: {other}: I finished sorting index cards.\n"
                            "assistant: Thank you for that earlier update.\n")
                    binding_class = "first-person-header"
                else:
                    # A third-person query concerns the unique named person in
                    # discourse, even when the sender is a different person.
                    speaker = other
                    past = (f"user: User: We are discussing {actor} alone.\n"
                            f"assistant: {pronoun.capitalize()} is the only person under discussion.\n")
                    binding_class = "third-person-history"
                add(family, case, body, speaker, past, binding_class,
                    target=actor, header=first_person)

            if first_person:
                # User is a real anonymous speaker role in the runtime format.
                # First person belongs to that role, not either prior sender.
                past = (
                    f"user: {actors[0]}: I have a question about an instrument.\n"
                    f"user: {actors[1]}: I have a question about an instrument too.\n"
                    "assistant: A new message arrived without a sender name.\n"
                )
                add(family, "ambiguous", body, "User", past,
                    "first-person-anonymous", target="User", header=True)
                add(family, "empty", body, "User", "",
                    "first-person-anonymous", target="User", header=True)
            else:
                add(family, "ambiguous", body, "User", ambiguous_history,
                    "third-person-ambiguous")
                add(family, "empty", body, "User", "", "third-person-empty")

        # A named person can be the antecedent inside ordinary dialogue history
        # while the current sender is anonymous. This is a distinct binding
        # task from the explicit "we are discussing X alone" control: the only
        # cue is that X speaks about their own {object} and is then addressed
        # by name. Both anonymous and named current senders are supervised.
        body = patterns["third-person-dialogue"].format(
            pronoun=pronoun, object=object_name)
        for actor_index, actor in enumerate(actors):
            case = "a" if actor_index == 0 else "b"
            speaker = "User" if actor_index == 0 else actors[1 - actor_index]
            past = (f"user: {actor}: I have been working on my {object_name} all week.\n"
                    f"assistant: Noted, {actor}. We can review it now.\n")
            add("third-person-dialogue", case, body, speaker, past,
                "third-person-dialogue", target=actor)
    if split in ("fit", "development"):
        rows.extend(serving_shape_rows(split))
    return rows


# The learned query binder sees one request shape that the object/actor worlds
# above do not reproduce: a recall question arrives as the whole message, with
# no history and no speaker or observation metadata. These authored rows use
# exactly that provenance text. A name inside the question is already a
# self-contained subject, so the gate must not demand a reference; a question
# that only says "she" still needs one, and with nothing to point at it must
# abstain. The world is disjoint from the fit, development and sealed worlds.
SERVING_FIT_WORLDS = (
    {
        "actors": ("Tamsin", "Oskar"),
        "objects": ("spectrometer stand", "ceramic glaze jar"),
        "timestamp": "2048-05-21 08:40 UTC",
    },
    {
        "actors": ("Perpetua", "Anselm"),
        "objects": ("map folio", "signal lamp"),
        "timestamp": "2053-10-02 15:05 UTC",
    },
    # A question that names its subject must not need a reference for any name,
    # so the fit worlds cover many names and objects. Four names taught the gate
    # those four names instead of the form; the development worlds below share
    # no name with these.
    {
        "actors": ("Corin", "Marta"),
        "objects": ("linotype gauge", "copper watering can"),
        "timestamp": "2049-03-12 07:15 UTC",
    },
    {
        "actors": ("Soren", "Ileana"),
        "objects": ("marble trowel", "velvet ring box"),
        "timestamp": "2051-06-30 18:20 UTC",
    },
    {
        "actors": ("Rufus", "Nadja"),
        "objects": ("stoneware jug", "brass sextant"),
        "timestamp": "2047-11-04 09:05 UTC",
    },
    {
        "actors": ("Osgood", "Verity"),
        "objects": ("oak writing slope", "porcelain mortar"),
        "timestamp": "2052-01-19 13:45 UTC",
    },
    {
        "actors": ("Hollis", "Junia"),
        "objects": ("walnut metronome", "solder spool"),
        "timestamp": "2054-08-08 11:30 UTC",
    },
    {
        "actors": ("Bram", "Perdita"),
        "objects": ("leather satchel", "tin whistle"),
        "timestamp": "2046-05-27 16:10 UTC",
    },
    {
        "actors": ("Calder", "Xenia"),
        "objects": ("cane easel", "glass hydrometer"),
        "timestamp": "2050-09-15 20:25 UTC",
    },
    {
        "actors": ("Dorian", "Mina"),
        "objects": ("pewter tankard", "slate drawing board"),
        "timestamp": "2055-02-03 06:50 UTC",
    },
    {
        "actors": ("Emrys", "Sabine"),
        "objects": ("copper still", "linen press"),
        "timestamp": "2048-07-21 14:00 UTC",
    },
    {
        "actors": ("Gideon", "Thalia"),
        "objects": ("ash walking stick", "beeswax block"),
        "timestamp": "2053-04-09 10:35 UTC",
    },
)

SERVING_FIT_QUESTIONS = (
    "When did {name} last take the {object} to the workshop?",
    "How long has {name} owned the {object}?",
    "Would {name} still use the {object} after the move?",
    "What did {name} say about the {object} at the meeting?",
    "Where does {name} keep the {object} during winter?",
    "Who gave {name} the {object}?",
    "Is {name} planning to replace the {object}?",
    "Why did {name} sell the {object}?",
    "How often does {name} clean the {object}?",
    "Did {name} ever repair the {object}?",
    "What colour is {name}'s {object}?",
    "Which shop did {name} buy the {object} from?",
    "When is {name} collecting the {object}?",
    "What does {name} dislike about the {object}?",
    "Has {name} finished restoring the {object}?",
    "Before the move, where did {name} keep the {object}?",
    "Since last spring, how often has {name} used the {object}?",
    "If the {object} breaks again, what will {name} do?",
    "Although {name} liked the {object}, why did they sell it?",
    "What was {name} doing when the {object} arrived?",
    "How many times has {name} moved the {object} this year?",
    "Which month did {name} purchase the {object}?",
    "What time did {name} leave the workshop yesterday?",
    "How long ago did {name} replace the {object}?",
    "When was the last time {name} cleaned the {object}?",
    "Who did {name} ask about repairing the {object}?",
    "How did {name} carry the {object} to the studio?",
    "Where was {name} when the {object} arrived?",
    "What did {name} do after opening the {object}?",
    "Why did {name} stop using the {object} on trips?",
    "Does {name} still keep the {object} at home?",
    "How much did {name} pay for the {object}?",
)

SERVING_FIT_HISTORY = (
    "user: User: The studio rota changed again this month.\n"
    "assistant: We can sort the rota out at the next meeting.\n"
)

SERVING_FIT_SPEAKER_PAST = (
    "user: {speaker}: The workshop flooded last Thursday, so we moved everything "
    "upstairs.\nassistant: That sounds disruptive. Let us go through what needs "
    "attention.\n"
)

SERVING_FIT_UNBOUND = (
    "Where did {pronoun} keep the {object} during the winter?",
    "How long has {pronoun} owned the {object}?",
    "What did {pronoun} say about the {object} at the meeting?",
    "Is {pronoun} planning to replace the {object}?",
    "Which shop did {pronoun} buy the {object} from?",
    "Has {pronoun} finished restoring the {object}?",
)

SERVING_FIT_HISTORY_QUESTIONS = (
    "Where did {pronoun} keep the {object} during the winter?",
    "How long has {pronoun} owned the {object}?",
    "Which shop did {pronoun} buy the {object} from?",
    "Has {pronoun} finished restoring the {object}?",
)

# The development split repeats every serving family in different worlds, with
# different wording, so selection measures the same request shape instead of
# only the older multi-message worlds.
SERVING_DEV_WORLDS = (
    {
        "actors": ("Cressida", "Barnaby"),
        "objects": ("astroglobe tripod", "enamel pin tin"),
        "timestamp": None,
    },
    {
        "actors": ("Mirela", "Jovian"),
        "objects": ("portable easel", "brass compass case"),
        "timestamp": None,
    },
)

SERVING_DEV_QUESTIONS = (
    "At which festival did {name} first show the {object}?",
    "Whom has {name} lent the {object} to recently?",
    "What does the {object} remind {name} of?",
    "How did {name} find out the {object} was missing?",
    "Which neighbour helped {name} carry the {object}?",
    "What will {name} do with the {object} after retiring?",
    "How far did {name} travel to collect the {object}?",
    "What did {name} write on the label of the {object}?",
    "Who repaired the {object} for {name} last winter?",
    "Which year did {name} inherit the {object}?",
    "What did the {object} look like when {name} found it?",
    "How does {name} protect the {object} from dust?",
)

SERVING_DEV_HISTORY = (
    "user: User: The exhibition dates changed again last week.\n"
    "assistant: We can settle the dates at the next meeting.\n"
)

SERVING_DEV_SPEAKER_PAST = (
    "user: {speaker}: The gallery roof leaked during the storm, so the frames "
    "are downstairs.\nassistant: That sounds stressful. Let us plan what to "
    "check first.\n"
)

SERVING_DEV_UNBOUND = (
    "Whom has {pronoun} lent the {object} to recently?",
    "Which neighbour helped {pronoun} carry the {object}?",
    "Which year did {pronoun} inherit the {object}?",
    "How does {pronoun} protect the {object} from dust?",
)

SERVING_DEV_HISTORY_QUESTIONS = SERVING_DEV_UNBOUND


def serving_shape_rows(split):
    """Rows in the literal single-message query request shape for one split."""
    if split == "fit":
        worlds, questions = SERVING_FIT_WORLDS, SERVING_FIT_QUESTIONS
        unbound, history_questions = SERVING_FIT_UNBOUND, SERVING_FIT_HISTORY_QUESTIONS
        history, speaker_past = SERVING_FIT_HISTORY, SERVING_FIT_SPEAKER_PAST
        role, evaluation_only = "fit-teacher", False
    elif split == "development":
        worlds = SERVING_DEV_WORLDS
        questions = SERVING_DEV_QUESTIONS
        unbound = history_questions = SERVING_DEV_UNBOUND
        history, speaker_past = SERVING_DEV_HISTORY, SERVING_DEV_SPEAKER_PAST
        role, evaluation_only = "development-evaluation", True
    else:
        return []
    rows = []

    def add(world, object_name, family, case, body, *, need, target=None,
            past="", actors=(), binding_class=None, speaker="User",
            timestamp="unspecified"):
        context = _context(speaker, timestamp, past)
        rows.append(dict(
            body=body, context=context, speaker=speaker, timestamp=timestamp,
            past=past,
            source_id=f"query-binding-{split}-serving-{world}-{object_name.replace(' ', '-')}"
                      f"-{family}-{case}",
            spans=[], whole_need=need,
            whole_refs=[target] if target is not None else [],
            whole_ref_ranges=[_reference(context, target)] if target is not None else [],
            tag="query-binding-query",
            query_binding_class=binding_class or ("resolved" if target is not None
                                                  else "ambiguous" if need else "self_contained"),
            query_binding_family=family, query_binding_case=case,
            query_binding_object=object_name, query_binding_actors=list(actors),
            query_binding_split=split, split=split,
            evaluation_only=evaluation_only, evaluation_role=role,
        ))

    for world_index, world in enumerate(worlds):
        actors = world["actors"]
        for object_index, object_name in enumerate(world["objects"]):
            name = actors[object_index]
            other = actors[1 - object_index]
            for template_index, template in enumerate(questions):
                body = template.format(name=name, object=object_name)
                add(world_index, object_name, "serving-name", f"q{template_index}", body,
                    need=False, actors=actors)
                if template_index % 5 == 0:
                    # The same self-contained question after ordinary chat
                    # history, so the decision is about the question and not
                    # about an empty context.
                    add(world_index, object_name, "serving-name-history",
                        f"q{template_index}", body, need=False, actors=actors,
                        past=history)
            # A pronoun-only question with nothing to point at abstains rather
            # than inventing a subject.
            pronoun = "he" if world_index % 2 == 0 else "she"
            for template_index, body in enumerate(unbound):
                add(world_index, object_name, "serving-unbound", f"q{template_index}",
                    body.format(pronoun=pronoun, object=object_name), need=True,
                    actors=actors)
            # Ordinary history can still bind a pronoun truthfully.
            past = (f"user: {name}: I spent the morning with my {object_name}.\n"
                    "assistant: Noted, and thanks for the detail. We can go through it now.\n")
            for template_index, template in enumerate(history_questions):
                add(world_index, object_name, "serving-history-bound", f"q{template_index}",
                    template.format(pronoun=pronoun, object=object_name), need=True,
                    target=name, past=past, actors=actors,
                    binding_class="resolved")
            # The same pronoun wording with a name substituted is a minimal
            # pair: only the subject form differs, so the gate has to key on
            # the explicit name rather than on the question frame.
            for template_index, body in enumerate(unbound):
                add(world_index, object_name, "serving-named-pair", f"q{template_index}",
                    body.format(pronoun=name, object=object_name), need=False,
                    actors=actors)
            for template_index, template in enumerate(history_questions):
                add(world_index, object_name, "serving-named-history", f"q{template_index}",
                    template.format(pronoun=name, object=object_name), need=False,
                    actors=actors, past=past)
            # A named question keeps needing no reference when the request does
            # carry a speaker and an observation time: the subject is in the
            # message, so the header cannot be its antecedent.
            for template_index, template in enumerate(questions[:8]):
                add(world_index, object_name, "serving-named-speaker", f"q{template_index}",
                    template.format(name=name, object=object_name), need=False,
                    actors=actors, speaker=other,
                    timestamp=world["timestamp"] or "unspecified",
                    past=speaker_past.format(speaker=other))
    return rows


def explicit_question_rows(split):
    """Authored self-contained questions that name their subject.

    The writer curricula already label these single-message questions as not
    needing an external reference. Reusing them gives the query gate many more
    nouns, question frames and dates from the same authored split, so the
    self-contained decision is learned from wording rather than memorized
    worlds. Evaluation splits stay evaluation-only.
    """
    if split not in ("fit", "development"):
        raise ValueError("explicit question rows require fit or development")
    from prepare_neural_colloquial import colloquial_curriculum, reader_data
    from prepare_neural_episodes import episode_curriculum

    source = []
    for row in episode_curriculum(split):
        if row["tag"] == "episode-query" and row["whole_need"] is False:
            source.append((row["body"], row["speaker"], row["timestamp"], row["past"],
                           row["source_id"]))
    for query in reader_data(split, colloquial_curriculum(split))["queries"]:
        if query.get("query_class") != "explicit":
            continue
        observation = query["observation"]
        source.append((observation["body"], observation["speaker"], observation["timestamp"],
                       observation["past"], observation["source_id"]))
    rows = []
    for index, (body, speaker, timestamp, past, source_id) in enumerate(source):
        variants = [(speaker, timestamp)]
        if index % 2 == 0:
            # Half the rows also carry the literal empty-history header so the
            # decision does not depend on an authored observation time.
            variants.append(("User", "unspecified"))
        for variant_index, (variant_speaker, variant_timestamp) in enumerate(variants):
            rows.append(dict(
                body=body, context=_context(variant_speaker, variant_timestamp, past),
                speaker=variant_speaker, timestamp=variant_timestamp, past=past,
                source_id=f"query-binding-{split}-explicit-{source_id}-{variant_index}",
                spans=[], whole_need=False, whole_refs=[],
                tag="query-binding-query",
                query_binding_class="self_contained",
                query_binding_family="explicit-question",
                query_binding_case=f"q{index}",
                query_binding_split=split, split=split,
                evaluation_only=split != "fit",
                evaluation_role=("fit-teacher" if split == "fit"
                                 else "development-evaluation"),
            ))
    return rows


DIALOGUE_FIT_WORLDS = (
    {
        "actors": ("Bastian", "Ottilie"),
        "objects": ("viola tailpiece", "piccolo joint"),
        "timestamp": "2049-06-18 09:30 UTC",
        "questions": (
            "Which part of the {object} does {pronoun} inspect first?",
            "What cloth does {pronoun} use on the {object}?",
            "Where does {pronoun} keep the {object} between rehearsals?",
        ),
        "replies": (
            "Thanks for the update, {name}. We can go through it now.",
            "I made a note of that, {name}. Let us review it.",
            "Understood, {name}. We can look at it together.",
        ),
    },
    {
        "actors": ("Perrin", "Saskia"),
        "objects": ("harp lever", "drum hoop"),
        "timestamp": "2050-02-07 17:45 UTC",
        "questions": (
            "How does {pronoun} tighten the {object}?",
            "Which lubricant does {pronoun} avoid on the {object}?",
            "When does {pronoun} replace the {object}?",
        ),
        "replies": (
            "Thanks for the details, {name}. Shall we begin?",
            "That is useful background, {name}. We can start there.",
            "Noted, {name}. Tell me how the {object} behaves.",
        ),
    },
    {
        "actors": ("Lucian", "Margit"),
        "objects": ("flute crown", "clarinet barrel"),
        "timestamp": "2052-09-29 11:10 UTC",
        "questions": (
            "What does {pronoun} check before playing the {object}?",
            "Which case holds the spare {object}?",
            "How often does {pronoun} oil the {object}?",
        ),
        "replies": (
            "Thanks, {name}. I have your earlier note here.",
            "I remember, {name}. You described the {object} last time.",
            "Right, {name}. We can work through the details.",
        ),
    },
    {
        "actors": ("Emmerich", "Rosalind"),
        "objects": ("bass clarinet peg", "trombone slide"),
        "timestamp": "2054-12-03 08:05 UTC",
        "questions": (
            "Which adjustment does {pronoun} make to the {object}?",
            "What does {pronoun} wipe off the {object} after use?",
            "Where does {pronoun} rest the {object} during breaks?",
        ),
        "replies": (
            "Good, {name}. We can take it from the top.",
            "Understood, {name}. Let us check the {object} first.",
            "Thank you, {name}. I have that noted.",
        ),
    },
)


def query_binding_dialogue_fit_augmentation():
    """Fit-only dialogue-history binding rows with independently authored worlds.

    A named person introduces their own instrument part in ordinary dialogue
    history and is then addressed by name in the assistant reply. The current
    sender is either the anonymous User or the other named person, so the
    header is never the referent. Gold ranges point at the final mention.
    These rows contain no evaluation split text and never authorize a write.
    """
    rows = []
    for world_index, world in enumerate(DIALOGUE_FIT_WORLDS):
        actors = world["actors"]
        pronoun = "he" if world_index % 2 == 0 else "she"
        timestamp = world["timestamp"]
        for object_index, object_name in enumerate(world["objects"]):
            for actor_index, actor in enumerate(actors):
                other = actors[1 - actor_index]
                speaker = "User" if actor_index == 0 else other
                case = "a" if actor_index == 0 else "b"
                question = world["questions"][object_index]
                reply = world["replies"][object_index]
                past = (f"user: {actor}: I have spent the week adjusting my {object_name}.\n"
                        f"assistant: {reply.format(name=actor, object=object_name)}\n")
                context = _context(speaker, timestamp, past)
                rows.append(dict(
                    body=question.format(pronoun=pronoun, object=object_name),
                    context=context, speaker=speaker, timestamp=timestamp,
                    past=past,
                    source_id=f"query-binding-fit-dialogue-{world_index}-{object_index}-{case}",
                    spans=[], whole_need=True, whole_refs=[actor],
                    whole_ref_ranges=[_reference(context, actor)],
                    tag="query-binding-query",
                    query_binding_class="third-person-dialogue",
                    query_binding_family="third-person-dialogue",
                    query_binding_case=f"dialogue-{case}",
                    query_binding_object=object_name,
                    query_binding_actors=list(actors),
                    query_binding_split="fit", split="fit",
                    evaluation_only=False, evaluation_role="fit-teacher",
                ))
            # Anonymous sender with the name only inside the assistant reply.
            actor = actors[object_index % len(actors)]
            reply = world["replies"][2]
            past = (f"user: User: The {object_name} still needs attention.\n"
                    f"assistant: {reply.format(name=actor, object=object_name)}\n")
            context = _context("User", timestamp, past)
            rows.append(dict(
                body=world["questions"][2].format(pronoun=pronoun, object=object_name),
                context=context, speaker="User", timestamp=timestamp, past=past,
                source_id=f"query-binding-fit-dialogue-{world_index}-{object_index}-c",
                spans=[], whole_need=True, whole_refs=[actor],
                whole_ref_ranges=[_reference(context, actor)],
                tag="query-binding-query",
                query_binding_class="third-person-dialogue",
                query_binding_family="third-person-dialogue",
                query_binding_case="dialogue-c",
                query_binding_object=object_name,
                query_binding_actors=list(actors),
                query_binding_split="fit", split="fit",
                evaluation_only=False, evaluation_role="fit-teacher",
            ))
    return rows
