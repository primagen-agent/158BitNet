"""Independent writer curricula with separate fit and public development.

Only authored names and literal messages occur here. No external evaluation files,
checkpoints or benchmark data are read. Span/reference ranges are training
labels; native input remains the original body and provenance fields only.
"""

from copy import deepcopy

from neural_slot_model import observation_context


FIT_NAMES = ("Ada", "Dorian", "Fenn", "Greta")
FACTS = (
    "When the pavement is wet, I leave my folding cart indoors because its small wheels slip on smooth slabs.",
    "I carry the folding cart on the bus only when its basket is empty, and I do not unfold it until I reach the stop.",
    "Unless the latch clicks twice, I check my tool case again before carrying it upstairs.",
    "I keep the tool case on the floor whenever the handle feels loose, rather than risk it falling from the bench.",
    "Once the glue has set, I remove the clamps from my bookends and leave the felt pads in place so they do not scratch the desk.",
    "I do not sand my bookends while the glue is still soft; I wait until the joint stays firm under gentle pressure.",
    "If a parcel is heavier than two kilograms, I use my fabric lifting strap instead of holding the cord around the package.",
    "My fabric lifting strap stays folded in the side pocket, and I do not lend it out unless I can check the stitching afterward.",
    "Whenever the drawing paper starts curling, I put a clean board over it and wait until the corners lie flat again.",
    "I keep my drawing paper away from the heater so it does not become brittle, even when the room feels chilly.",
    "Before I roll out the clay, I dampen the work mat just enough that the clay does not stick to the edges.",
    "I reuse the work mat only after the dried clay has been brushed off, and I never scrape it with the sharp trimming blade.",
    "Once the ruler is aligned with the pencil marks, I secure it with two clips so it cannot slide while I cut.",
    "I do not use the bent ruler for measuring long pieces; it stays in the scrap bin unless I need a rough spacing guide.",
    "When the hall gets crowded, I hang my bicycle helmet under the bench rather than leave it where someone could trip over it.",
    "I replace the padding in my bicycle helmet whenever it stops staying flat, but I do not change the buckle unless it fails to close.",
    "If the knitting yarn catches on the basket rim, I pass it through a smooth wooden ring before continuing the row.",
    "I do not wind my knitting yarn into a tight ball while it is damp because the center would take too long to dry.",
    "Unless the canoe paddle is completely dry, I leave it on the rack with the grip facing down so water can run out.",
    "My canoe paddle goes into its sleeve only after I have rinsed away the grit, and I check that no sand remains near the grip.",
    "After the hiking boots have aired overnight, I brush the seams and pack them only when the lining no longer feels damp.",
    "I do not put my hiking boots beside an open fire; I leave them near moving air so the leather does not stiffen.",
    "When I label the storage jars, I write the date on removable tape so I can change it without covering an older label.",
    "I keep the storage jars empty unless the lids are dry, since trapped moisture could spoil the seeds inside.",
    "Once the accordion folder is full, I move the oldest sketches into a flat box and do not crease them to make room.",
    "I take my accordion folder outside only if the cover can close without bending the drawings at the corners.",
    "If the tent pegs resist going into the ground, I loosen the soil with a short digging stick rather than strike the pegs harder.",
    "I do not pack the tent pegs while mud is stuck to them; I wipe each one until the metal can be seen along the whole shaft.",
    "Whenever the clockwork toy slows down, I let its spring unwind before adding another turn so the mechanism is not strained.",
    "My clockwork toy stays on a level tray unless I am carrying it, because I do not want it rolling off the table.",
    "Before I use the measuring jug, I place it on a level shelf and wait for the liquid to stop moving so the mark can be read clearly.",
    "I do not fill my measuring jug past its widest mark, even if a recipe needs more liquid; I measure a second portion instead.",
)

WRAPPERS = (
    ("Quick update, {other}. ", " That's enough from me; how are things with you?"),
    ("Good to catch up, {other}. ", " Anyway, I appreciate you listening."),
    ("I had one small thing to mention, {other}. ", " On a separate note, did you get a chance to rest?"),
    ("Let me share something, {other}. ", " Thanks for the conversation; speak soon."),
)


def boundary_diversity_fit_rows():
    """32 complete statements, 32 fictional contrasts and 16 greetings."""
    rows = []
    timestamp = "2042-08-16 11:20 UTC"
    for index, fact in enumerate(FACTS):
        name = FIT_NAMES[index % len(FIT_NAMES)]
        other = FIT_NAMES[(index + 1) % len(FIT_NAMES)]
        past = f"user: {other}: Good to see you.\nassistant: Share your latest update.\n"
        context = observation_context(name, timestamp, past)
        ref = dict(document="context", start=9, end=9 + len(name.encode()))
        prefix, suffix = WRAPPERS[(index // len(FIT_NAMES)) % len(WRAPPERS)]
        prefix, suffix = prefix.format(other=other), suffix.format(other=other)
        start = len(prefix.encode())
        span = dict(start=start, end=start + len(fact.encode()), action=1,
                    need=True, references=[name], ref_ranges=[dict(ref)])
        common = dict(speaker=name, timestamp=timestamp, past=past, context=context,
                      boundary_diversity=True)
        rows.append(
            dict(common, body=prefix + fact + suffix, spans=[span],
                 source_id=f"writer-boundary-fit-{index}", tag="writer-complex-fact",
                 whole_need=True, whole_refs=[name], whole_ref_ranges=[dict(ref)])
        )
        rows.append(
            dict(common,
                 body='This is a made-up line spoken by an invented narrator in my script, not an account of my life: "' + fact + '"',
                 spans=[], source_id=f"writer-boundary-fiction-{index}",
                 tag="writer-complex-fiction", whole_need=False,
                 whole_refs=[], whole_ref_ranges=[])
        )
    greetings = (
        "Nice talking with you; I hope the rest of your afternoon goes well.",
        "Thanks for listening. We can catch up again another time.",
        "Hello again! It is good to have a moment to chat.",
        "I see what you mean. Anyway, have a pleasant evening.",
    )
    for name in FIT_NAMES:
        context = observation_context(name, timestamp, "")
        for index, body in enumerate(greetings):
            rows.append(
                dict(body=body, speaker=name, timestamp=timestamp, past="", context=context,
                     boundary_diversity=True, spans=[],
                     source_id=f"writer-boundary-greeting-{name}-{index}",
                     tag="writer-complex-greeting", whole_need=False,
                     whole_refs=[], whole_ref_ranges=[])
            )
    return rows


BOUNDARY_NAMES = {
    "fit": ("Ada", "Dorian", "Fenn", "Greta"),
    "development": ("Kellan", "Nerissa", "Tobin", "Silas"),
}

# These two worlds are authored independently. Development is evaluation-only;
# neither the examples nor their values are derived from an observed failure.
BOUNDARY_FACTS = {
    "fit": (
        "I unplug my spice grinder before brushing out the powder, even if I am only cleaning the lid.",
        "When I take my map tube outside, I tighten both caps and carry it upright so the rolled sheets cannot slide out.",
        "My pocket compass stays away from the loudspeaker because its needle moves whenever the magnet is nearby.",
        "I wash my canvas apron separately unless the pockets have been emptied and the chalk dust has been shaken off.",
        "Once my cable organizer is full, I put the longer leads in a cloth bag rather than force another loop under the strap.",
        "I fold my radio aerial down before closing the cabinet, and I check the hinge is free instead of pushing against it.",
        "Before using my wooden mallet, I inspect the head for splits and leave it aside if the grain has started opening.",
        "I tilt my soap dish toward the drain so water runs out, but I leave the bar on a rack until its underside is dry.",
        "My thread stand goes on the left corner of the workbench, where the spool can turn without catching on the lamp cord.",
        "I do not force my basin plug into place when the rim is twisted; I straighten the rubber edge before trying again.",
        "After carrying my folding stool indoors, I wipe each foot and open the frame only once the floor is clear.",
        "I keep my battery case shut while sorting used cells, and I never put a leaking cell beside the fresh ones.",
        "If my magnetic clips leave marks on a page, I cover their tips with felt before fastening them to another sheet.",
        "My pressure gauge stays in its padded pouch between checks because a loose pointer would make the reading unreliable.",
        "I attach my spare key to a fabric keyring, then test the clasp by pulling gently so it cannot open inside my bag.",
        "I remove my windscreen cover only after the frost has loosened, rather than drag the stiff fabric across the glass.",
    ),
    "development": (
        "For my palette knife, I use a soft rag after each session and dry the joint carefully before putting it back in the drawer.",
        "My travel thermos is stored with the lid resting beside it so that the inside airs out between trips.",
        "I put a ceramic spoon rest beside the cooker whenever I stir a sauce, keeping the hot utensil off the wooden counter.",
        "With my wristwatch case, I close the padded flap before turning the latch; otherwise the strap gets pinched underneath.",
        "I let my leather gloves warm up naturally after a cold walk instead of placing them on the radiator, which could harden the seams.",
        "An umbrella I have used in the rain stays open in the porch until no drops remain around the central joint.",
        "For the camping lamp I own, a dim glow means I change both cells together rather than mix a new one with an older one.",
        "I test each shelf bracket with a light load first, only adding the heavier boxes after the screws have stayed firm.",
        "My small handbell sits on a square of cork because the bare base scratches the polished sideboard when it is moved.",
        "I slide my ribbon bookmark clear of the spine before shutting a thick book, taking care that it does not fold across the hinge.",
        "A tea strainer I use for loose leaves is rinsed upside down, allowing the pieces trapped in the mesh to fall into the sink.",
        "I set my iron trivet on a folded towel once it has cooled, leaving the painted surface underneath protected from its rough feet.",
        "Before my shoe brush goes into the cupboard, I tap it over a newspaper and remove the grit caught near the handle.",
        "I keep my card wallet in an inside pocket when the weather is damp, since the paper passes would soften if they got wet.",
        "The water bottle I carry is filled below the neck so that the cap can be tightened without spilling liquid around the seal.",
        "After I use my rubber stamp, I press it onto scrap paper until the excess ink is gone and then cover the pad again.",
    ),
}

BOUNDARY_WRAPPERS = {
    "fit": (
        ("", ""),
        ("A quick note for you, {other}. ", " That is all for now; I hope your day is going smoothly."),
        ("I have a little practical update, {other}. ", " By the way, shall we talk again tomorrow?"),
    ),
    "development": (
        ("", ""),
        ("There is something I wanted to tell you, {other}. ", " Well, it was lovely to hear from you."),
        ("While we are chatting, {other}, here is my update. ", " Moving on, have you had a peaceful morning?"),
    ),
}


def boundary_generalization_curriculum(split):
    """48 complete-source views and 24 empty-source controls per split.

    Split names, objects, values and conversational wrappers do not overlap.
    Gold is the whole authored statement, including every dependent clause.
    Fiction and greetings are explicit empty-source teacher examples. These
    labels never participate in native observation transport.
    """
    if split not in BOUNDARY_NAMES:
        raise ValueError("boundary curriculum supports fit/development only")
    rows = []
    names = BOUNDARY_NAMES[split]
    timestamp = "2047-03-18 08:45 UTC" if split == "fit" else "2048-07-22 16:10 UTC"
    for index, fact in enumerate(BOUNDARY_FACTS[split]):
        name, other = names[index % 4], names[(index + 1) % 4]
        past = f"user: {other}: I am listening.\nassistant: Please carry on.\n"
        context = observation_context(name, timestamp, past)
        ref = dict(document="context", start=9, end=9 + len(name.encode()))
        common = dict(speaker=name, timestamp=timestamp, past=past, context=context,
                      boundary_generalization=True, boundary_world=split)
        for view, (before, after) in enumerate(BOUNDARY_WRAPPERS[split]):
            before, after = before.format(other=other), after.format(other=other)
            start = len(before.encode())
            span = dict(start=start, end=start + len(fact.encode()), action=1,
                        need=True, references=[name], ref_ranges=[dict(ref)])
            rows.append(dict(common, body=before + fact + after, spans=[span],
                             source_id=f"writer-edge-{split}-fact-{index}-view-{view}",
                             tag="writer-edge-fact", whole_need=True,
                             whole_refs=[name], whole_ref_ranges=[dict(ref)]))
        fiction = (
            'I am reading a fictional character\'s line from a draft, not describing myself: "' + fact + '"'
            if split == "fit" else
            'Here is a quotation from an imaginary scene; it is not a real personal update: "' + fact + '"'
        )
        rows.append(dict(common, body=fiction, spans=[],
                         source_id=f"writer-edge-{split}-fiction-{index}",
                         tag="writer-edge-fiction", whole_need=False,
                         whole_refs=[], whole_ref_ranges=[]))
    greetings = (
        ("It was good to catch up. I wish you a pleasant rest of the day.",
         "Hello! Thanks for making time to chat with me.")
        if split == "fit" else
        ("What a nice conversation. Take care on your way home.",
         "Good evening; it is a pleasure to hear from you again.")
    )
    for name in names:
        for index, body in enumerate(greetings):
            rows.append(dict(body=body, speaker=name, timestamp=timestamp, past="",
                             context=observation_context(name, timestamp), spans=[],
                             boundary_generalization=True, boundary_world=split,
                             source_id=f"writer-edge-{split}-greeting-{name}-{index}",
                             tag="writer-edge-greeting", whole_need=False,
                             whole_refs=[], whole_ref_ranges=[]))
    return rows


def with_boundary_generalization(rows):
    """Append fit teachers and a distinct eval group without mutating inputs."""
    if "boundary_development" in rows:
        raise ValueError("boundary development is already attached")
    result = dict(rows)
    result["fit"] = rows["fit"] + boundary_generalization_curriculum("fit")
    result["boundary_development"] = boundary_generalization_curriculum("development")
    return result


NARRATIVE_NAMES = {
    "fit": FIT_NAMES,
    "development": ("Beatrix", "Lucan", "Selwyn", "Sigrid"),
}

# Each pack has three independently meaningful facts and three questions about
# its objects. Questions never assert that the proposed action has happened.
# These worlds were authored for sentence/span contrast, without reading any
# benchmark, checkpoint, or external acceptance data.
NARRATIVE_PACKS = {
    "fit": (
        (
            (
                "I dry the cork stoppers on a mesh rack before storing them, so damp ones never go into my drawer.",
                "I replaced the worn zipper on my canvas pouch with a brass one that closes without catching the lining.",
                "My basil seedlings sit on a raised tray, and I rotate it each morning to keep the stems upright.",
            ),
            ("Would rinsing a cork stopper damage it, {other}?",
             "How should I choose a new zipper for a pouch?",
             "Should I rotate a seedling tray in the evening too?"),
        ),
        (
            (
                "I keep my embroidery floss in numbered envelopes because loose strands get tangled in the sewing tin.",
                "My pantry scale is checked with a metal test weight before I measure ingredients for a large batch.",
                "I moved my potted thyme beside the east window, where it gets light without touching the cold glass.",
            ),
            ("Can you suggest a way to sort embroidery floss, {other}?",
             "How often should I check a pantry scale?",
             "Would a thyme pot need a deeper saucer?"),
        ),
        (
            (
                "I print shipping labels on matte paper and leave them flat until the ink dries, rather than stack them immediately.",
                "I store my bamboo knitting needles in a rigid tube so their tips cannot catch on the bag lining.",
                "My folding watering can stays empty between uses, with its handle resting against the shed wall.",
            ),
            ("Would glossy paper work for shipping labels, {other}?",
             "How could I protect a knitting needle tip?",
             "Should I leave a watering can outside overnight?"),
        ),
        (
            (
                "I test the bell on my commuter bicycle before leaving, and I tighten its clamp whenever the handle starts slipping.",
                "My wax crayons go into a divided carton, with the pale colors kept apart from the dark broken pieces.",
                "I rinsed my sprouting lentils twice this morning and covered the jar with clean mesh so air can pass through.",
            ),
            ("Is a bicycle bell easy to adjust, {other}?",
             "Would I need separate trays for broken crayons?",
             "How can I tell whether a sprouting jar has enough air?"),
        ),
        (
            (
                "I mark my quilting pieces with removable thread, and I pull it out only after the seam has been checked.",
                "My portable fan stands on a rubber mat because its feet otherwise make the shelf vibrate.",
                "I arrange my seed packets by sowing month in a shallow box, leaving the instructions visible at the back.",
            ),
            ("Which thread would you use for temporary marks, {other}?",
             "Could I reduce the vibration of a small fan?",
             "Is sorting seed packets by month practical?"),
        ),
        (
            (
                "I brush the crumbs from my bread board before hanging it upright, so nothing collects around the bottom edge.",
                "I keep my domino set in a cloth roll with one pocket for each tile, rather than shake the pieces into a box.",
                "My mint cuttings stand in a clear jar, and I change their water whenever it starts looking cloudy.",
            ),
            ("Would you hang a bread board to dry, {other}?",
             "How could I carry a domino set quietly?",
             "Should I change cutting water every day?"),
        ),
        (
            (
                "I label my extension leads at both ends with colored sleeves, which makes matching a plug to its socket easier.",
                "My charcoal sticks are wrapped separately in scrap paper so the softer pieces do not crumble against one another.",
                "I raised my cucumber planter on two blocks because the drainage holes were sitting against the paving stones.",
            ),
            ("Would colored sleeves help you identify a lead, {other}?",
             "How should I pack soft charcoal sticks?",
             "Could I put a planter directly on paving stones?"),
        ),
        (
            (
                "I use a shallow basin to wash my wool mittens and press the water out gently instead of twisting the cuffs.",
                "My solder spool is kept in a lidded metal tin, with the loose end bent inward so it cannot snag a cable.",
                "I trimmed the dead tips from my spider plant yesterday, leaving the healthy green parts of each leaf intact.",
            ),
            ("What would you use to wash wool mittens, {other}?",
             "Should I cover a loose solder spool?",
             "How far back could I trim a damaged plant tip?"),
        ),
    ),
    "development": (
        (
            (
                "My aquarium filter has a reusable sponge that I rinse in the tank water after removing loose debris.",
                "I place my cello bow in its fitted compartment before latching the case, keeping the hair clear of the hinge.",
                "I transfer loose geometric tiles into a padded drawer, with the chipped corners facing the lining rather than another tile.",
            ),
            ("Is there an easy way to rinse a filter sponge, {other}?",
             "Would closing a case damage bow hair if it caught the hinge?",
             "How would I protect a chipped geometric tile?"),
        ),
        (
            (
                "After filling my fountain pen, I wipe the rim of the ink bottle and return its cap before testing the nib.",
                "My climbing chalk is carried in a small bag inside a sealed tub, preventing loose dust from reaching the rest of my kit.",
                "I turn the cards in my language practice deck over one at a time and place unfamiliar ones in a separate compartment.",
            ),
            ("What is a tidy way to refill a pen, {other}?",
             "Would I need a second container for loose chalk?",
             "Should an unfamiliar practice card go back into the same pile?"),
        ),
        (
            (
                "The glaze samples I make are dated on their unpainted backs, letting me compare the same mixture after different firings.",
                "I coil the cord of my electric blanket loosely after it cools and keep the controller outside the folded layers.",
                "My glass marbles are separated by size in a felt tray so I can spot a missing piece without counting the whole collection.",
            ),
            ("Where would you date a glaze sample, {other}?",
             "Is it sensible to fold a warm electric blanket immediately?",
             "Could I check a marble collection without counting every piece?"),
        ),
        (
            (
                "I shade my beehive entrance with a removable screen during the hottest afternoons, leaving space beneath it for passing bees.",
                "A turquoise ribbon marks the drawer holding my calligraphy nibs, which are cleaned and dried before they go back.",
                "I leave my juggling clubs on a low padded shelf when practicing indoors, keeping the weighted ends away from the doorway.",
            ),
            ("Could an entrance screen leave enough room for bees, {other}?",
             "What would you use to identify a nib drawer?",
             "Is a low shelf suitable for keeping juggling clubs?"),
        ),
    ),
}

NARRATIVE_WRAPPERS = {
    "fit": (
        ("All right — here are my practical notes, {other}.", "That covers my notes; thanks for listening."),
        ("Can I run a few things by you, {other}?", "Anyway, it was nice to catch up."),
        ("", "I'll leave it there. How has your afternoon been?"),
    ),
    "development": (
        ("Since we have a moment, {other}, let me describe a few things.", "Well, that was everything I wanted to mention."),
        ("There are several unrelated updates on my side, {other}.", "Let's pause here and continue another time."),
    ),
}


def narrative_curriculum(split):
    """Small independent multi-source turns with explicit nonassertion gaps.

    Fit has 24 three-fact views plus eight question-only and eight fictional
    controls. Public development has eight three-fact views and four questions.
    The latter is a development guard, never an independent acceptance set.
    """
    if split not in NARRATIVE_NAMES:
        raise ValueError("narrative curriculum supports fit/development only")
    rows = []
    names = NARRATIVE_NAMES[split]
    timestamp = "2051-02-11 10:15 UTC" if split == "fit" else "2052-09-04 17:25 UTC"
    for index, (facts, questions) in enumerate(NARRATIVE_PACKS[split]):
        name, other = names[index % len(names)], names[(index + 1) % len(names)]
        questions = tuple(q.format(other=other) for q in questions)
        past = (f"user: {other}: How are things?\n" if split == "fit" else
                f"user: {other}: What would you like to tell me?\n")
        context = observation_context(name, timestamp, past)
        ref = dict(document="context", start=9, end=9 + len(name.encode()))
        common = dict(speaker=name, timestamp=timestamp, past=past, context=context,
                      narrative_diversity=True, narrative_world=split,
                      evaluation_role="fit-teacher" if split == "fit" else "public-development-not-independent-acceptance")
        fact_index = {fact: i for i, fact in enumerate(facts)}
        asides = (("Okay, let me carry on.", "I appreciate your patience.") if split == "fit" else
                  ("That is the first of those updates.", "Thank you for hearing me out."))
        placements = (
            (facts[0], questions[0], facts[1], asides[0], facts[2], questions[1]),
            (questions[2], facts[2], asides[1], facts[0], questions[0], facts[1]),
            (facts[1], questions[1], facts[0], questions[2], facts[2]),
        )
        for view, ((before, after), placement) in enumerate(zip(NARRATIVE_WRAPPERS[split], placements)):
            pieces = [before.format(other=other)] if before else []
            spans = []
            for piece in placement:
                start = len((" ".join(pieces) + (" " if pieces else "")).encode())
                pieces.append(piece)
                if piece in fact_index:
                    spans.append(dict(start=start, end=start + len(piece.encode()), action=1,
                                      need=True, references=[name], ref_ranges=[dict(ref)],
                                      narrative_fact_index=fact_index[piece]))
            pieces.append(after.format(other=other))
            rows.append(dict(common, body=" ".join(pieces), spans=spans,
                             source_id=f"writer-narrative-{split}-{index}-view-{view}",
                             tag="writer-narrative-facts", whole_need=True,
                             whole_refs=[name], whole_ref_ranges=[dict(ref)]))
        body = ("I would like advice on a few things. " + " ".join(questions)
                if split == "fit" else "I have some questions to put to you. " + " ".join(questions))
        rows.append(dict(common, body=body, spans=[],
                         source_id=f"writer-narrative-{split}-{index}-questions",
                         tag="writer-narrative-questions", whole_need=True,
                         whole_refs=[name], whole_ref_ranges=[dict(ref)]))
        if split == "fit":
            body = ('For a fictional scene, an invented character says: "' + " ".join(facts)
                    + '" This is dialogue in the story, not a description of my own life.')
            rows.append(dict(common, body=body, spans=[],
                             source_id=f"writer-narrative-{split}-{index}-fiction",
                             tag="writer-narrative-fiction", whole_need=False,
                             whole_refs=[], whole_ref_ranges=[]))
    return rows


def with_narrative_diversity(rows):
    """Append fit teachers while keeping public development out of fit."""
    if "narrative_development" in rows or any(row.get("narrative_diversity") for row in rows["fit"]):
        raise ValueError("narrative curriculum is already attached")
    result = dict(rows)
    result["fit"] = rows["fit"] + narrative_curriculum("fit")
    result["narrative_development"] = narrative_curriculum("development")
    return result


_SOURCE_CONTEXT_SUFFIXES = ("::source-context-empty", "::source-context-neutral")


def with_source_context_generalization(rows):
    """Cross irrelevant histories for the already authored fit worlds only.

    Boundary/narrative fit statements are self-contained first-person facts;
    their fictional and question controls define nonassertion in the body.
    This transform is not valid for facts or references inferred from history.
    Every explicit context reference must be the complete speaker header. Body
    text, source/action labels and literal reference ranges remain unchanged.
    Neither base-fit messages nor development worlds receive added views.
    """
    fit = rows["fit"]
    if any(row["source_id"].endswith(_SOURCE_CONTEXT_SUFFIXES) for row in fit):
        raise ValueError("source context generalization is already attached")
    authored = [row for row in fit if row.get("boundary_world") == "fit"
                or row.get("narrative_world") == "fit"]
    if not authored:
        raise ValueError("source context generalization requires an authored fit world")
    ids = [row["source_id"] for group in rows.values() for row in group]
    if len(ids) != len(set(ids)):
        raise ValueError("source observation ids must be unique")
    existing = set(ids)
    extra = []
    for row in authored:
        name, timestamp = row["speaker"], row["timestamp"]
        if not isinstance(name, str) or not name or row["context"] != observation_context(
            name, timestamp, row["past"]
        ):
            raise ValueError("authored source context must match its literal provenance")
        header = dict(document="context", start=9, end=9 + len(name.encode()))
        ranges = list(row.get("whole_ref_ranges", ()))
        for span in row["spans"]:
            ranges.extend(span.get("ref_ranges", ()))
        for ref in ranges:
            if ref["document"] == "context" and ref != header:
                raise ValueError("source context views require speaker-header references")
            if ref["document"] not in ("body", "context"):
                raise ValueError("source references must name the unchanged body or speaker header")
        next_index = (FIT_NAMES.index(name) + 1) % len(FIT_NAMES) if name in FIT_NAMES else 0
        other = FIT_NAMES[next_index]
        neutral = f"user: {other}: Hello there.\nassistant: Please continue.\n"
        for suffix, past in zip(_SOURCE_CONTEXT_SUFFIXES, ("", neutral)):
            context = observation_context(name, timestamp, past)
            if context == row["context"]:
                continue
            source_id = row["source_id"] + suffix
            if source_id in existing:
                raise ValueError("source context view id already exists")
            existing.add(source_id)
            clone = deepcopy(row)
            clone.update(past=past, context=context, source_id=source_id)
            extra.append(clone)
    result = deepcopy(rows)
    result["fit"].extend(extra)
    return result


_ASSERTION_FACTS = (
    "A longer hinge pin repaired the loose latch on my folding chess board yesterday.",
    "Replacing the cork gasket stopped the leak from my steel thermos last weekend.",
    "The torn strap on my laundry basket was replaced with a blue cord on Monday.",
    "A padded sleeve now covers the rough handle of my kitchen scoop.",
    "Last Friday, two shallow trays replaced the paper envelopes holding my loose screws.",
    "The faded numbers on my storage bins were rewritten with a silver marker this morning.",
    "An oak block now keeps the wheel of my folding trolley from touching the wall.",
    "Beside my coat hook, a small tin now holds the spare buttons from my raincoat.",
    "Both feet of my reading stool now have cork pads to keep them from scratching the floor.",
    "Yesterday a green tab was added to the folder holding my appliance receipts.",
    "The hinge on my wooden measuring case was cleaned and oiled on Saturday.",
    "After Monday's repair, the lid of my document box stays shut without a ribbon.",
    "Since last week, my kitchen scales have stood on a flat board instead of the uneven counter.",
    "Under my sink, a labelled pouch now holds the spare washers for the tap.",
    "Three wide rubber bands now keep the cover of my travel notebook firmly closed.",
    "The missing cap on my bicycle pump was replaced with a yellow stopper yesterday.",
)


def assertion_generalization_fit_rows():
    """Independent assertion teachers with shared words and quotation marks.

    Real quoted notes, passive reports and factual question premises select the
    complete literal assertion. Fiction, hypothetical premises, pure questions
    and courtesy commentary select nothing. No evaluation files are read.
    All positive subjects are the literal speaker; ranges are fit labels only.
    """
    timestamp = "2053-05-19 11:40 UTC"
    quote_frames = (
        ('I wrote a note about what happened at home: "', '" Those are my own notes.'),
        ('This is not an invented example; my actual notebook says, "', '"'),
        ('Here is an entry describing my own belongings: "', '" That records what actually happened.'),
        ('The line in my own maintenance diary reads, "', '" It is an accurate personal record.'),
    )
    passive_frames = (
        ("", " Anyway, thanks for checking in."),
        ("A short update — ", " Shall we leave that topic there?"),
        ("Nothing else to add. ", " That is the whole update."),
        ("Thanks for listening. ", " No advice is needed for that."),
    )
    questions = (
        "would a simple note help me track the next check?",
        "what is a sensible way to record the work?",
        "should I keep a separate note about the next inspection?",
        "how could I label the record so it is easier to find?",
    )
    rows = []
    for index, fact in enumerate(_ASSERTION_FACTS):
        # Every speaker sees every control kind, and each kind has both
        # history conditions. Provenance cannot stand in for assertion labels.
        name = FIT_NAMES[(index // 4) % len(FIT_NAMES)]
        past = "" if (index // 4 + index) % 2 == 0 else "assistant: Please continue.\n"
        context = observation_context(name, timestamp, past)
        header = dict(document="context", start=9, end=9 + len(name.encode()))
        common = dict(speaker=name, timestamp=timestamp, past=past, context=context,
                      assertion_generalization=True, assertion_world="fit",
                      assertion_fact_index=index, evaluation_role="fit-teacher")

        def positive(view, prefix, source, suffix):
            start = len(prefix.encode())
            span = dict(start=start, end=start + len(source.encode()), action=1,
                        need=True, references=[name], ref_ranges=[dict(header)])
            return dict(common, body=prefix + source + suffix, spans=[span],
                        source_id=f"writer-assertion-fit-{index}-{view}",
                        tag="writer-assertion-" + view, assertion_kind=view,
                        whole_need=True, whole_refs=[name],
                        whole_ref_ranges=[dict(header)])

        before, after = quote_frames[index % len(quote_frames)]
        rows.append(positive("real-quote", before, fact, after))
        before, after = passive_frames[index % len(passive_frames)]
        rows.append(positive("passive-report", before, fact, after))
        clause = fact[0].lower() + fact[1:-1]
        # Structural source units include their trailing comma. Preserve that
        # original punctuation so this gold range is representable at serving.
        rows.append(positive("question-premise", "Given that ", clause + ",",
                             " " + questions[index % len(questions)]))

        negative_kind = ("fiction", "conditional", "pure-question", "courtesy")[index % 4]
        if negative_kind == "fiction":
            body = 'A made-up diary entry reads, "' + fact + '" This is fiction, not a personal report.'
        elif negative_kind == "conditional":
            body = "Suppose " + clause + ". What might happen in that hypothetical situation?"
        elif negative_kind == "pure-question":
            body = "Is it true that " + clause + "? I am only asking, without confirming that premise."
        else:
            body = ('Good to chat! The wording "' + fact +
                    '" was easy to follow; that is only a courtesy comment about phrasing.')
        whole_need = negative_kind in ("conditional", "pure-question")
        rows.append(dict(common, body=body, spans=[],
                         source_id=f"writer-assertion-fit-{index}-{negative_kind}",
                         tag="writer-assertion-" + negative_kind,
                         assertion_kind=negative_kind, whole_need=whole_need,
                         whole_refs=[name] if whole_need else [],
                         whole_ref_ranges=[dict(header)] if whole_need else []))
    return deepcopy(rows)


def with_assertion_generalization(rows):
    """Append the independently authored fit teachers without altering guards."""
    fit = rows["fit"]
    if any(row.get("assertion_generalization") or row["source_id"].startswith(
        "writer-assertion-fit-"
    ) for row in fit):
        raise ValueError("assertion generalization is already attached")
    existing = [row["source_id"] for group in rows.values() for row in group]
    if len(existing) != len(set(existing)):
        raise ValueError("source observation ids must be unique")
    added = assertion_generalization_fit_rows()
    if set(existing) & {row["source_id"] for row in added}:
        raise ValueError("assertion fit observation id already exists")
    result = deepcopy(rows)
    result["fit"].extend(added)
    return result
