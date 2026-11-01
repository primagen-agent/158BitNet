"""Independent colloquial memory observations and referential questions.

This module reads no external conversations, benchmarks, checkpoints or results.
Facts, names and wordings are authored here. ``blind`` is defined before training
and is excluded from training/selection; its digest is recorded in the manifest.
Labels describe original UTF-8 byte ranges and are training/scoring metadata only.
The native observation transport receives body/past/speaker/timestamp/source_id.

Named questions, including ordinary ASCII possessives, do not require an inferred
reference prefix. First-person and contextual questions require learned source
selection. Truly ambiguous questions require a reference but provide no gold
source, so committing or activating an arbitrary person's memory is incorrect.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from neural_slot_model import observation_context

NAMES = {
    "fit": (
        "Aileen",
        "Bram",
        "Celia",
        "Desmond",
        "Etta",
        "Felix",
    ),
    "development": ("Aster", "Blythe", "Corin"),
    "blind": ("Everett", "Fiora"),
}

TIMESTAMPS = {
    "fit": "2034-02-17 10:30 UTC",
    "development": "2035-06-11 14:20 UTC",
    "blind": "2036-09-23 09:40 UTC",
}

# property, complete source fact, named question, possessive question,
# contextual question, first-person question, descriptive topic cue, kind.
# Every row has one complete compound fact, including its trailing explanation.
PATTERNS = {
    "fit": (
        (
            "paint_choice",
            "Been using ochre paint on my pantry shelves; I chose a washable finish so splashes can be wiped away.",
            "Which paint is {name} using on the pantry shelves?",
            "What finish does {name}'s shelf paint have?",
            "What shade did they choose for the shelves?",
            "What paint have I been using on my shelves?",
            "shelf painting",
            0,
        ),
        (
            "window_seal",
            "Swapped the old window seal for a rubber strip, which keeps rain from seeping onto my sill.",
            "What did {name} use to replace the window seal?",
            "What keeps rain away from {name}'s window sill?",
            "What did they fit around the window?",
            "What did I use for the window seal?",
            "window maintenance",
            0,
        ),
        (
            "hardware_storage",
            "I've put my spare hinges in a red tin and labeled each packet so matching sizes are easy to find.",
            "Where does {name} keep spare hinges?",
            "How are {name}'s spare hinges organized?",
            "Where do they keep the spare hinges?",
            "Where have I put my spare hinges?",
            "hardware storage",
            0,
        ),
        (
            "wall_insulation",
            "Got mineral wool behind my shed wall now, and I fitted a vent to let damp air escape.",
            "What insulation did {name} put behind the shed wall?",
            "Why does {name}'s shed have a new vent?",
            "What is behind their shed wall?",
            "What did I put behind the shed wall?",
            "shed insulation",
            0,
        ),
        (
            "plant_watering",
            "I mist my fern before breakfast and tip out the saucer afterward so its roots don't sit in water.",
            "How does {name} look after the fern before breakfast?",
            "Why is the saucer under {name}'s fern emptied?",
            "What do they do with the fern's saucer?",
            "When do I mist my fern?",
            "fern watering",
            0,
        ),
        (
            "plant_soil",
            "Keep my herbs in a sandy mix, adding a little pumice whenever the surface starts holding too much moisture.",
            "What soil mix does {name} use for herbs?",
            "When is pumice added to {name}'s herb pots?",
            "What do they add to the herb soil?",
            "What mix do I keep my herbs in?",
            "herb soil",
            0,
        ),
        (
            "plant_position",
            "That tall cactus of mine stays by the east-facing window, where it gets morning light without the fierce afternoon sun.",
            "Where does {name} keep the tall cactus?",
            "What light reaches {name}'s cactus?",
            "Which window is their cactus near?",
            "Where does my tall cactus stay?",
            "cactus placement",
            0,
        ),
        (
            "plant_container",
            "Just repotted my spider plant in a shallow terracotta bowl; the extra holes let surplus water drain.",
            "What container did {name} choose for the spider plant?",
            "What helps water drain from {name}'s plant bowl?",
            "What kind of bowl did they choose for the plant?",
            "What did I repot my spider plant in?",
            "plant containers",
            0,
        ),
        (
            "lamp_repair",
            "Lamp's working again after I fitted a linen-covered cord and checked the switch for a loose contact.",
            "What cord did {name} fit to the lamp?",
            "What was checked on {name}'s lamp switch?",
            "What kind of cord did they fit to the lamp?",
            "What did I change on the lamp?",
            "lamp repair",
            0,
        ),
        (
            "kettle_repair",
            "Fixed the kettle by replacing its cracked handle, then tightened the mounting screw until the wobble disappeared.",
            "How did {name} fix the kettle?",
            "What stopped {name}'s kettle handle from wobbling?",
            "What part of the kettle did they replace?",
            "How did I fix my kettle?",
            "kettle repair",
            0,
        ),
        (
            "fan_repair",
            "I've cleaned the desk fan's grille and oiled its axle, which stopped the squeak at low speed.",
            "What did {name} clean and oil on the desk fan?",
            "What stopped the squeak in {name}'s desk fan?",
            "What did they oil on the fan?",
            "What stopped my desk fan squeaking?",
            "desk fan maintenance",
            0,
        ),
        (
            "bicycle_care",
            "Bike chain's quiet now; I degreased it and used wax lubricant so grit won't stick to it.",
            "What lubricant does {name} use on the bike chain?",
            "Why is wax used on {name}'s bike chain?",
            "What lubricant do they use on the chain?",
            "What did I use on my bike chain?",
            "bike chain maintenance",
            0,
        ),
        (
            "photo_settings",
            "Usually shoot my tabletop photos at f/8 with a small reflector, which helps keep both ends of an object sharp.",
            "What aperture does {name} use for tabletop photos?",
            "What helps both ends of an object stay sharp in {name}'s photos?",
            "What aperture do they use for tabletop photos?",
            "What aperture do I usually use for tabletop photos?",
            "tabletop photo settings",
            0,
        ),
        (
            "photo_subject",
            "My camera's mostly pointed at stone bridges these days, and I go out at dawn to catch empty paths.",
            "What does {name} photograph these days?",
            "When are {name}'s stone bridge photos taken?",
            "What do they mostly photograph?",
            "What have I mostly been photographing?",
            "photography subjects",
            0,
        ),
        (
            "photo_backup",
            "Copied my raw photo files onto a silver external drive, then tucked a second copy onto the shelf above my desk.",
            "Where does {name} store raw photo backups?",
            "Where is the second copy of {name}'s photo files?",
            "Where do they put the second photo backup?",
            "Where did I put my photo backups?",
            "photo backups",
            0,
        ),
        (
            "photo_visit",
            "Weekend snapshots came from the old canal towpath; I took the tripod because long exposures help smooth the water.",
            "Where did {name} take the weekend snapshots?",
            "Why was a tripod used for {name}'s canal photographs?",
            "Where did they take those weekend snapshots?",
            "Where did I take my weekend snapshots?",
            "weekend photography",
            1,
        ),
    ),
    "development": (
        (
            "paint_choice",
            "Turns out satin teal is what I've settled on for my hallway cabinet, since a damp cloth removes the fingerprints.",
            "What paint did {name} settle on for the hallway cabinet?",
            "Why is {name}'s cabinet painted in satin teal?",
            "What finish did they settle on for the cabinet?",
            "What paint did I settle on for the cabinet?",
            "cabinet repainting",
            0,
        ),
        (
            "window_seal",
            "New foam tape's around my kitchen casement, with the corners pressed flat to keep the cold draft out.",
            "What did {name} put around the kitchen casement?",
            "Why were the corners of {name}'s new window tape pressed flat?",
            "What material is around their kitchen window?",
            "What did I put around my kitchen casement?",
            "window draft prevention",
            0,
        ),
        (
            "hardware_storage",
            "All my spare screws now sit in numbered drawers, and the tiny ones have paper liners so they don't roll underneath.",
            "Where does {name} keep spare screws?",
            "Why do the tiny screws in {name}'s drawers have paper liners?",
            "How do they store the smallest screws?",
            "Where do I keep my spare screws now?",
            "spare screw storage",
            0,
        ),
        (
            "wall_insulation",
            "The workshop's lined with cork panels now; I've also left a narrow air gap behind them to prevent condensation.",
            "What lines {name}'s workshop?",
            "What prevents condensation behind {name}'s workshop panels?",
            "What panels line their workshop?",
            "What panels did I fit in the workshop?",
            "workshop insulation",
            0,
        ),
        (
            "plant_watering",
            "My peace lily gets a measured mugful every other evening, and I feel under the leaves first to check it hasn't stayed wet.",
            "How often does {name} water the peace lily?",
            "What is checked before watering {name}'s peace lily?",
            "How often do they water their peace lily?",
            "How often do I water my peace lily?",
            "peace lily watering",
            0,
        ),
        (
            "plant_soil",
            "Been mixing orchid bark into the potting compost for my pothos, which leaves pockets of air around the roots.",
            "What does {name} mix into the pothos compost?",
            "Why is bark used in {name}'s pothos pots?",
            "What do they add to the pothos compost?",
            "What have I mixed into the pothos compost?",
            "pothos compost",
            0,
        ),
        (
            "plant_position",
            "My jade plant's moved onto the stair landing, where a frosted skylight gives it bright but softened daylight.",
            "Where does {name} keep the jade plant?",
            "What kind of light reaches {name}'s jade plant?",
            "Where is their jade plant now?",
            "Where did I move my jade plant?",
            "jade plant placement",
            0,
        ),
        (
            "plant_container",
            "Picked a deep glazed pot for my rosemary and put coarse stones beneath it so the drainage holes stay clear.",
            "What pot did {name} pick for the rosemary?",
            "What keeps the drainage holes clear under {name}'s rosemary?",
            "What did they put underneath the rosemary pot?",
            "What pot did I pick for my rosemary?",
            "rosemary containers",
            0,
        ),
        (
            "lamp_repair",
            "The bedside light's back on after I replaced its ceramic socket, then straightened the bracket to stop the shade leaning.",
            "What did {name} replace in the bedside light?",
            "How was the leaning shade on {name}'s light fixed?",
            "What socket did they fit to the light?",
            "How did I fix my bedside light?",
            "bedside light repair",
            0,
        ),
        (
            "fan_repair",
            "Sorted my extractor fan by washing the mesh filter and securing the housing with two fresh bolts, so the vibration has gone.",
            "How did {name} stop the extractor fan vibrating?",
            "What was washed in {name}'s extractor fan?",
            "How did they secure the extractor fan housing?",
            "What stopped my extractor fan vibrating?",
            "extractor fan repair",
            0,
        ),
        (
            "photo_settings",
            "For my close-ups I use a two-second timer and manual focus; that way touching the shutter won't blur the markings.",
            "What timer does {name} use for close-ups?",
            "Why is a timer used for {name}'s close-up photos?",
            "What prevents their close-ups from blurring?",
            "What timer do I use for close-ups?",
            "close-up photo settings",
            0,
        ),
        (
            "photo_backup",
            "There's a weekly photo archive on my orange memory stick, and an identical copy is kept in the drawer under my printer.",
            "Where does {name} keep the weekly photo archive?",
            "Where is the identical copy of {name}'s archive?",
            "Where do they keep their second photo archive?",
            "Where do I keep my photo archive?",
            "weekly photo archives",
            0,
        ),
    ),
    "blind": (
        (
            "paint_choice",
            "Ended up choosing matte plum for my alcove cupboard; the darker color hides the little dents left in the wood.",
            "What color did {name} choose for the alcove cupboard?",
            "Why is {name}'s cupboard painted matte plum?",
            "What color did they choose for the cupboard?",
            "What color did I choose for the cupboard?",
            "alcove cupboard painting",
            0,
        ),
        (
            "window_seal",
            "Felt cord's tucked along my attic window now, with a copper clip at each end so it stays put when I open the frame.",
            "What did {name} fit along the attic window?",
            "What holds the cord at {name}'s attic window?",
            "How do they hold the cord in place?",
            "What did I fit along the attic window?",
            "attic window repairs",
            0,
        ),
        (
            "hardware_storage",
            "The washers I keep for repairs are sorted by diameter in clear jars, and a card inside each lid records the size.",
            "How does {name} organize spare washers?",
            "Where are the sizes of {name}'s washers recorded?",
            "How do they mark the washer sizes?",
            "How do I organize my spare washers?",
            "washer organization",
            0,
        ),
        (
            "wall_insulation",
            "I've added recycled denim inside my studio partition, then sealed the edges to cut the sound from the next room.",
            "What did {name} put inside the studio partition?",
            "Why are the edges of {name}'s partition sealed?",
            "What material did they use in the partition?",
            "What did I put inside the studio partition?",
            "studio partition insulation",
            0,
        ),
        (
            "plant_watering",
            "Giving my begonia a little water on Sunday mornings works for me, provided I let the top layer dry between visits.",
            "When does {name} water the begonia?",
            "What dries between watering {name}'s begonia?",
            "When do they water their begonia?",
            "When do I water my begonia?",
            "begonia watering",
            0,
        ),
        (
            "plant_soil",
            "My rubber plant's rooted in compost mixed with perlite, which stays loose even after the pot has been watered.",
            "What is mixed into the compost for {name}'s rubber plant?",
            "What keeps the soil in {name}'s rubber plant pot loose?",
            "What is mixed into their rubber plant soil?",
            "What is in my rubber plant's soil?",
            "rubber plant soil",
            0,
        ),
        (
            "plant_position",
            "Tucked my small lemon tree beside the conservatory door, where a curtain shields it from the strongest midday rays.",
            "Where does {name} keep the small lemon tree?",
            "What shields {name}'s lemon tree from midday rays?",
            "What protects their lemon tree from midday sun?",
            "Where did I put my small lemon tree?",
            "lemon tree placement",
            0,
        ),
        (
            "plant_container",
            "A wide wooden tub's now home to my bamboo, with a mesh liner that keeps the soil from escaping through the base.",
            "What container holds {name}'s bamboo?",
            "What stops soil escaping from {name}'s bamboo tub?",
            "What liner did they use in the bamboo tub?",
            "What container holds my bamboo?",
            "bamboo containers",
            0,
        ),
        (
            "lamp_repair",
            "My wall sconce is steady again; I replaced the bent mounting plate and added a felt pad to keep the glass from rattling.",
            "What mounting part did {name} replace in the wall sconce?",
            "What keeps the glass in {name}'s sconce from rattling?",
            "What did they use to stop the glass rattling?",
            "How did I stop the wall sconce rattling?",
            "wall sconce repairs",
            0,
        ),
        (
            "fan_repair",
            "Got my tower fan running quietly by aligning the rotor and replacing its worn foam gasket, which cushions the casing.",
            "What did {name} align in the tower fan?",
            "What cushions the casing of {name}'s tower fan?",
            "What gasket did they replace in the fan?",
            "How did I quiet my tower fan?",
            "tower fan servicing",
            0,
        ),
        (
            "photo_settings",
            "These days my food photos use side light and a gray card, so the colors stay consistent when the weather changes.",
            "What lighting does {name} use for food photos?",
            "What keeps the colors of {name}'s photos consistent?",
            "What card do they use for consistent photo colors?",
            "What do I use for consistent photo colors?",
            "food photo settings",
            0,
        ),
        (
            "photo_backup",
            "Made two monthly photo copies, one on my black portable disk and another in a padded case beside the bookcase.",
            "Where does {name} keep monthly photo copies?",
            "Where is the other copy of {name}'s monthly photos?",
            "Where do they store the second monthly photo copy?",
            "Where did I put my monthly photo copies?",
            "monthly photo copies",
            0,
        ),
    ),
}

WRAPPERS = {
    "fit": (
        ("", ""),
        ("Hey {other}, quick update from me. ", " Anyway, how's your afternoon going?"),
        ("Well, {other}, here's the thing. ", " Thanks for hearing me out."),
        (
            "Oh, and before I forget, {other}. ",
            " That's all for now; tell me your news.",
        ),
    ),
    "development": (
        ("", ""),
        ("Hi {other}, a bit of news. ", " And how has your week been?"),
        ("So, {other}, I meant to mention this. ", " It's nice to catch up with you."),
        (
            "One more thing, {other}, while we're talking. ",
            " I'll let you get back to your day now.",
        ),
    ),
    "blind": (
        ("", ""),
        (
            "Morning {other}, this is my little update. ",
            " Your turn; what's new with you?",
        ),
        ("By the way, {other}, something to share. ", " Lovely talking to you again."),
        (
            "Ah, {other}, almost missed telling you this. ",
            " That wraps up my news for today.",
        ),
    ),
}

ACTIVE_PROPERTIES = {
    "fit": (
        "paint_choice",
        "window_seal",
        "plant_watering",
        "plant_soil",
        "lamp_repair",
        "fan_repair",
        "photo_settings",
        "photo_backup",
    ),
    "development": (
        "paint_choice",
        "plant_watering",
        "plant_soil",
        "lamp_repair",
        "photo_settings",
        "photo_backup",
    ),
    "blind": (
        "paint_choice",
        "plant_watering",
        "plant_soil",
        "lamp_repair",
        "photo_settings",
        "photo_backup",
    ),
}


def _source_range(text, value, document="context", last=False):
    raw, target = text.encode(), value.encode()
    start = raw.rfind(target) if last else raw.find(target)
    if start < 0:
        raise ValueError("reference is absent from its original source")
    return dict(document=document, start=start, end=start + len(target))


def colloquial_curriculum(split):
    """Return fresh writer-schema rows; split labels never enter observations."""
    if split not in NAMES:
        raise ValueError("split must be fit, development or blind")
    rows, names, timestamp = [], NAMES[split], TIMESTAMPS[split]

    def add(
        body, speaker, past, spans=(), refs=(), ranges=(), need=False, tag="", **extra
    ):
        context = observation_context(speaker, timestamp, past)
        row = dict(
            body=body,
            speaker=speaker,
            timestamp=timestamp,
            past=past,
            source_id=f"colloquial-{split}-{len(rows)}",
            context=context,
            spans=list(spans),
            whole_need=need,
            whole_refs=list(refs),
            whole_ref_ranges=list(ranges),
            tag=tag,
            **extra,
        )
        rows.append(row)
        return row

    for i, name in enumerate(names):
        other = names[(i + 1) % len(names)]
        # Dialogue context contains distractor names but no memory fact replay.
        past = f"user: {name}: It's good to have time to chat.\nuser: {other}: I'm listening; tell me what you've been doing.\n"
        speaker_range = dict(document="context", start=9, end=9 + len(name.encode()))
        for prop, fact, named, possessive, contextual, personal, cue, kind in PATTERNS[
            split
        ]:
            if prop not in ACTIVE_PROPERTIES[split]:
                continue
            for prefix, suffix in WRAPPERS[split][: (3 if split == "fit" else 2)]:
                prefix, suffix = prefix.format(other=other), suffix.format(other=other)
                start = len(prefix.encode())
                source = dict(
                    start=start,
                    end=start + len(fact.encode()),
                    action=1,
                    need=True,
                    references=[name],
                    ref_ranges=[speaker_range.copy()],
                    prop=prop,
                    kind=kind,
                )
                add(
                    prefix + fact + suffix,
                    name,
                    past,
                    [source],
                    [name],
                    [speaker_range.copy()],
                    True,
                    "episode-fact",
                    entities=[name],
                    prop=prop,
                    kind=kind,
                    recall_questions=[
                        named.format(name=name),
                        possessive.format(name=name),
                    ],
                )
            key = dict(entity=name, prop=prop, kind=kind)
            for question in (named, possessive):
                add(
                    question.format(name=name),
                    "User",
                    "",
                    tag="episode-query",
                    entities=[name],
                    prop=prop,
                    kind=kind,
                    query_class="explicit",
                    support_keys=[key.copy()],
                )
            # The last plain name is an addressable original history source.
            # User is the question author, not the target person.
            context_past = f"user: Let's discuss {cue} for {other} and {name}.\nassistant: Sure, let's focus on {name} next.\n"
            context = observation_context("User", timestamp, context_past)
            add(
                contextual,
                "User",
                context_past,
                refs=[name],
                ranges=[_source_range(context, name, last=True)],
                need=True,
                tag="episode-query",
                entities=[name],
                prop=prop,
                kind=kind,
                query_class="context",
                support_keys=[key.copy()],
            )
            add(
                personal,
                name,
                past,
                refs=[name],
                ranges=[speaker_range.copy()],
                need=True,
                tag="episode-query",
                entities=[name],
                prop=prop,
                kind=kind,
                query_class="self",
                support_keys=[key.copy()],
            )
            # Two equal candidates and an unspecified single person: abstention.
            ambiguous_past = f"user: {name} and {other} both mentioned {cue}.\nassistant: You haven't said which one to discuss.\n"
            add(
                "And what did that person choose for this?",
                "User",
                ambiguous_past,
                need=True,
                tag="episode-query",
                entities=[],
                prop="",
                kind=kind,
                query_class="ambiguous",
                support_keys=[],
            )
            quotations = (
                'Please translate this quotation without treating it as my news: "{}"',
                'The imaginary narrator in my draft says: "{}"',
            )
            for quotation in quotations[: (1 if split == "fit" else 2)]:
                add(quotation.format(fact), name, past, tag="negative-quotation")
        for greeting in (
            f"Hey {other}, thanks for writing back!",
            f"Oh, {other}, that's pretty neat.",
            "Yeah, I see what you mean.",
            "Thanks a lot for the chat; catch you later.",
        ):
            add(greeting, name, past, tag="ordinary-negative")
        for question in (
            f"What is {name}'s favorite board game?",
            f"Which language does {name} want to learn?",
        ):
            add(
                question,
                "User",
                "",
                tag="episode-query",
                entities=[name],
                prop="",
                kind=0,
                query_class="null",
                support_keys=[],
            )
    return rows


def reader_data(split, observations=None):
    """Training/component records; reference labels never go to native serving.

    Questions retain raw bodies and literal observation fields. A reader trainer
    must obtain any reference prefix from its frozen writer prediction, rather
    than replacing missing reference predictions with these supervision labels.
    """
    facts, queries = {}, []
    for row in colloquial_curriculum(split) if observations is None else observations:
        if row["tag"] == "episode-fact":
            span = row["spans"][0]
            content = row["body"].encode()[span["start"] : span["end"]].decode()
            reference = span["references"][0]
            text = reference + ". " + content
            facts[text] = dict(
                text=text,
                entity=reference,
                reference=reference,
                prop=span["prop"],
                kind=span["kind"],
                colloquial=True,
            )
        elif row["tag"] == "episode-query":
            query = dict(
                text=row["body"],
                entities=list(row["entities"]),
                prop=row["prop"],
                kind=row["kind"],
                query_class=row["query_class"],
                observation_id=row["source_id"],
                colloquial=True,
                support_keys=[dict(k) for k in row["support_keys"]],
                observation=dict(row),
            )
            queries.append(query)
    fact_list = list(facts.values())
    for query in queries:
        keys = {(k["entity"], k["prop"], k["kind"]) for k in query["support_keys"]}
        query["support_texts"] = [
            f["text"] for f in fact_list if (f["entity"], f["prop"], f["kind"]) in keys
        ]
    return dict(facts=fact_list, queries=queries)


def query_statistics(split):
    """Definition counts only; no model, predictions, scoring or tuning."""
    rows = [r for r in colloquial_curriculum(split) if r["tag"] == "episode-query"]
    return dict(
        queries=len(rows),
        by_class=dict(sorted(Counter(r["query_class"] for r in rows).items())),
        need_reference=sum(r["whole_need"] for r in rows),
        resolved_reference=sum(bool(r["whole_refs"]) for r in rows),
        ambiguous=sum(r["whole_need"] and not r["whole_refs"] for r in rows),
        positive_queries=sum(bool(r["support_keys"]) for r in rows),
        negative_queries=sum(not r["support_keys"] for r in rows),
    )


def _digest(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_curriculum(output):
    """Lock all definitions with deterministic hashes before any model run."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = {split: colloquial_curriculum(split) for split in NAMES}
    for split, names in NAMES.items():
        for other, other_names in NAMES.items():
            if split != other and set(names) & set(other_names):
                raise ValueError("person identities overlap across splits")
    manifest = dict(
        format="independent-colloquial-curriculum-v1",
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        external_data_read=False,
        locomo_used=False,
        label_transport="training and external scoring only",
        selection_split="development",
        training_splits=["fit"],
        blind_status="locked before training; not scored by this generator",
        splits={
            split: dict(
                rows=len(rows),
                names=list(NAMES[split]),
                sha256=_digest(rows),
                facts=sum(r["tag"] == "episode-fact" for r in rows),
                query_statistics=query_statistics(split),
            )
            for split, rows in data.items()
        },
    )
    (output / "curriculum.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_curriculum(args.output)
    print(
        json.dumps(
            dict(
                output=str(args.output.resolve()),
                development_queries=manifest["splits"]["development"][
                    "query_statistics"
                ],
                split_hashes={s: d["sha256"] for s, d in manifest["splits"].items()},
                blind_status=manifest["blind_status"],
            ),
            indent=2,
        )
    )
