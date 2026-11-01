"""Authored synthetic conversational episodes; no benchmark files or imports.

Speaker provenance and supervision are distinct. Runtime transports contain only
original messages/provenance. Span, entity, reference-source and support labels
are consumed only by training or external scoring. Sealed split is defined before
training, with separate names, wording, values and dates.
"""

from neural_slot_model import observation_context

NAMES = {
    "fit": [
        "Cyra",
        "Dorian",
        "Mavis",
        "Fenn",
        "Ivo",
        "Nyla",
        "Oren",
        "Petra",
        "Ada",
        "Ben",
        "Chloe",
        "Dylan",
        "Elena",
        "Finn",
        "Greta",
        "Hector",
    ],
    "development": ["Ansel", "Briony", "Casper", "Dulcie"],
    "sealed": ["Elowen", "Hadrian", "Maribel", "Peregrine"],
}

PATTERNS = {
    "fit": [
        (
            "work_change",
            "I switched from repairing radios to restoring clocks because I wanted more flexible hours.",
            "Why did {name} change occupations?",
            "What work did {name} switch to?",
            0,
        ),
        (
            "material_preference",
            "When I build miniature boats, my favorite material is cedar wood because it is light and durable.",
            "What material does {name} prefer for miniature boats?",
            "Which wood is {name}'s favorite for model boats?",
            0,
        ),
        (
            "relaxation",
            "Listening to chamber music after a demanding day helps me unwind and clear my thoughts.",
            "How does {name} unwind after a demanding day?",
            "What helps {name} relax?",
            0,
        ),
        (
            "project_purpose",
            "I organized a neighborhood repair club because I wanted people to share tools and reduce waste.",
            "Why did {name} organize a repair club?",
            "What motivated {name}'s neighborhood project?",
            1,
        ),
        (
            "event_date",
            "On 14 April 2031, I visited the railway museum with my cousin and joined a guided tour.",
            "When did {name} visit the railway museum?",
            "Who accompanied {name} to the museum?",
            1,
        ),
        (
            "collection",
            "I collect old postcards, especially ones that show harbors, and I keep them in a green album.",
            "What does {name} collect?",
            "Where does {name} keep the postcard collection?",
            0,
        ),
        (
            "residence",
            "I moved into a small apartment in Bergen, close enough that I can walk to the library.",
            "Where does {name} live?",
            "What city is {name}'s apartment in?",
            0,
        ),
        (
            "family_plan",
            "Next autumn, I intend to help my nephew build a telescope for his school science fair.",
            "What does {name} plan to help their nephew build?",
            "When is {name}'s family project planned?",
            2,
        ),
        (
            "dietary",
            "I avoid hazelnuts because they upset my stomach, even though I enjoy most other snacks.",
            "What food does {name} avoid?",
            "Why does {name} avoid hazelnuts?",
            0,
        ),
        (
            "course",
            "I registered for an evening astronomy course because I want to understand how observatories work.",
            "What course did {name} register for?",
            "Why is {name} studying astronomy?",
            0,
        ),
        (
            "achievement",
            "My handmade bookshelf received the craftsmanship award at the local makers exhibition on 6 March 2031.",
            "What award did {name}'s bookshelf receive?",
            "When did {name} receive the craftsmanship award?",
            1,
        ),
        (
            "possession",
            "I bought a brass reading lamp with an adjustable arm, and it now stands beside my sewing machine.",
            "What lamp did {name} buy?",
            "Where does {name} keep the reading lamp?",
            0,
        ),
        (
            "work_change",
            "Making cabinets now pays my bills; I left warehouse packing so I could spend more time creating things.",
            "How does {name} earn a living now?",
            "Why did {name} leave warehouse packing?",
            0,
        ),
        (
            "relaxation",
            "My usual way of resting is to work on crosswords in a quiet cafe while drinking mint tea.",
            "What is {name}'s usual way of resting?",
            "What does {name} drink while doing crosswords?",
            0,
        ),
        (
            "collection",
            "My hobby is gathering foreign stamps, which I sort into envelopes according to the country they came from.",
            "What is {name}'s collecting hobby?",
            "How does {name} sort the stamps?",
            0,
        ),
        (
            "project_purpose",
            "To make the village more welcoming, I started a seed-sharing table outside the grocery shop.",
            "Why did {name} start the seed-sharing table?",
            "Where is {name}'s seed-sharing project located?",
            1,
        ),
    ],
    "development": [
        (
            "work_change",
            "I earn my living fixing bicycles now, having left office filing to get more hands-on work.",
            "What work does {name} do now?",
            "What led {name} to change jobs?",
            0,
        ),
        (
            "material_preference",
            "Balsa is the timber I choose for miniature airplanes, since its low weight makes the models easier to fly.",
            "What timber does {name} choose for miniature airplanes?",
            "Why is balsa {name}'s preferred material?",
            0,
        ),
        (
            "relaxation",
            "For me, a peaceful evening means reading detective stories in the garden after work.",
            "What does {name} do for a peaceful evening?",
            "Where does {name} read detective stories?",
            0,
        ),
        (
            "project_purpose",
            "The reason I founded our book exchange was to help neighbors discover stories without buying new books.",
            "What was {name}'s reason for founding the book exchange?",
            "Who benefits from {name}'s book project?",
            1,
        ),
        (
            "event_date",
            "The afternoon of 22 June 2032 was spent at the botanical gardens, where my sister and I attended a seed workshop.",
            "On what date did {name} go to the botanical gardens?",
            "Who attended the seed workshop with {name}?",
            1,
        ),
        (
            "collection",
            "Antique buttons are what I seek at weekend markets; each find goes into a labeled wooden box.",
            "What does {name} seek at weekend markets?",
            "Where are {name}'s antique buttons stored?",
            0,
        ),
        (
            "residence",
            "My home is now a rented flat in Porto, a short tram ride from the waterfront.",
            "Which city is {name}'s home in?",
            "How close is {name}'s flat to the waterfront?",
            0,
        ),
        (
            "family_plan",
            "During the coming winter, I hope to show my niece how to assemble a weather station for her club.",
            "What does {name} hope to assemble with their niece?",
            "When is {name} planning the weather-station project?",
            2,
        ),
        (
            "dietary",
            "Cashews make me feel ill, so I leave them out of meals and ask about them when ordering food.",
            "What ingredient makes {name} feel ill?",
            "How does {name} avoid cashews when eating out?",
            0,
        ),
        (
            "course",
            "An introductory geology class is on my timetable, as I want to identify the rocks I find on hikes.",
            "What class is on {name}'s timetable?",
            "Why does {name} want to learn geology?",
            0,
        ),
        (
            "achievement",
            "At the village craft show on 12 May 2032, the judges awarded my woven basket the originality prize.",
            "What prize did {name}'s basket win?",
            "When was {name}'s basket given a prize?",
            1,
        ),
        (
            "possession",
            "Our study now has the ceramic table lamp I purchased, with a shade made of blue fabric.",
            "What did {name} purchase for the study?",
            "What color is {name}'s lamp shade?",
            0,
        ),
    ],
    "sealed": [
        (
            "work_change",
            "My present occupation is servicing cameras; I moved on from stock counting to make use of my mechanical skills.",
            "What is {name}'s present occupation?",
            "Why did {name} move on from stock counting?",
            0,
        ),
        (
            "material_preference",
            "The miniature houses I make use pine, my preferred wood because it is easy to carve into small shapes.",
            "Which wood is preferred by {name} for miniature houses?",
            "What makes pine useful for {name}'s models?",
            0,
        ),
        (
            "relaxation",
            "I find it restful to sketch plants at the riverside, and that has become my routine after a tiring shift.",
            "What is {name}'s routine after a tiring shift?",
            "Where does {name} sketch plants to rest?",
            0,
        ),
        (
            "project_purpose",
            "Wanting lonely residents to meet one another, I set up a weekly puzzle gathering at the community hall.",
            "Why did {name} set up the puzzle gathering?",
            "Where does {name}'s community gathering take place?",
            1,
        ),
        (
            "event_date",
            "My brother came along when I toured the maritime archive on 9 September 2033 and took part in a map demonstration.",
            "When did {name} tour the maritime archive?",
            "Who came along to the map demonstration with {name}?",
            1,
        ),
        (
            "collection",
            "Historic train tickets form my collection, with the most fragile ones protected between transparent sheets.",
            "What forms {name}'s collection?",
            "How does {name} protect the fragile tickets?",
            0,
        ),
        (
            "residence",
            "A townhouse in Halifax is where I currently stay, within cycling distance of the university.",
            "Where does {name} currently stay?",
            "How does {name} travel from home to the university?",
            0,
        ),
        (
            "family_plan",
            "Once summer arrives, I would like to guide my younger cousin in making a solar cooker for a camp project.",
            "What would {name} like to make with their cousin?",
            "When does {name} intend to work on the solar cooker?",
            2,
        ),
        (
            "dietary",
            "Eating pistachios bothers my digestion, which is why I pass up desserts containing them.",
            "What bothers {name}'s digestion?",
            "Why does {name} pass up some desserts?",
            0,
        ),
        (
            "course",
            "Learning cartography through a weekend course should help me draw clearer routes for our walking group.",
            "What does {name} learn through a weekend course?",
            "What is {name}'s reason for studying cartography?",
            0,
        ),
        (
            "achievement",
            "The regional model exhibition gave my windmill model the detail prize on 18 August 2033.",
            "Which prize was given to {name}'s windmill model?",
            "On what date did {name}'s model get the prize?",
            1,
        ),
        (
            "possession",
            "Beside the piano sits my new glass desk lamp, which has a white shade and a dimmer switch.",
            "What kind of lamp sits beside {name}'s piano?",
            "What features does {name}'s new lamp have?",
            0,
        ),
    ],
}


def episode_curriculum(split):
    rows = []
    names = NAMES[split]
    timestamp = {
        "fit": "2031-05-20 15:00 UTC",
        "development": "2032-07-26 16:00 UTC",
        "sealed": "2033-10-21 14:00 UTC",
    }[split]

    def add(body, speaker, past, spans, need=False, refs=(), tag="", **extra):
        context = observation_context(speaker, timestamp, past)
        rows.append(
            dict(
                body=body,
                speaker=speaker,
                timestamp=timestamp,
                past=past,
                source_id=f"authored-{split}-{len(rows)}",
                context=context,
                spans=spans,
                whole_need=need,
                whole_refs=list(refs),
                whole_ref_ranges=(
                    [dict(document="context", start=9, end=9 + len(speaker.encode()))]
                    if need
                    else []
                ),
                tag=tag,
                **extra,
            )
        )

    def span(start, end, actor, prop, kind):
        return dict(
            start=start,
            end=end,
            action=1,
            need=True,
            references=[actor],
            prop=prop,
            kind=kind,
            ref_ranges=[dict(document="context", start=9, end=9 + len(actor.encode()))],
        )

    for i, name in enumerate(names):
        other = names[(i + 1) % len(names)]
        past = f"user: [{timestamp}] {name}: Thank you for making time to talk.\nuser: [{timestamp}] {other}: It is good to hear from you. Tell me about your week.\n"
        for topic, fact, q1, q2, kind in PATTERNS[split]:
            wrappers = [
                ("", ""),
                (
                    f"Hello {other}, I hope you are well. ",
                    " That is the news I wanted to share; how has your week been?",
                ),
                (
                    f"Thanks for asking, {other}. I have been meaning to tell you a few things. ",
                    " It has kept me quite busy lately. I would enjoy hearing what you have been working on, too.",
                ),
            ]
            for prefix, suffix in wrappers:
                body = prefix + fact + suffix
                a = len(prefix.encode())
                b = a + len(fact.encode())
                add(
                    body,
                    name,
                    past,
                    [span(a, b, name, topic, kind)],
                    True,
                    [name],
                    "episode-fact",
                    recall_questions=[q1.format(name=name), q2.format(name=name)],
                    entities=[name],
                )
            for question in (q1.format(name=name), q2.format(name=name)):
                add(
                    question,
                    "User",
                    "",
                    [],
                    False,
                    [],
                    "episode-query",
                    entities=[name],
                    prop=topic,
                    kind=kind,
                )
            for wrapper in (
                'Could you translate this quotation: "{}"',
                "Here is a story I made up about a fictional character: {}",
            ):
                add(
                    wrapper.format(fact),
                    name,
                    past,
                    [],
                    False,
                    [],
                    "negative-quotation",
                )
        for greeting in (
            f"Hello {other}!",
            f"Good to see you, {other}. How are things?",
            f"That sounds impressive, {other}!",
            f"Thanks for listening to me, {other}.",
            "Wow! That is wonderful.",
            "All right, I understand.",
        ):
            add(greeting, name, past, [], False, [], "ordinary-negative")
        question = f"What do {name} and {other} collect?"
        add(
            question,
            "User",
            "",
            [],
            False,
            [],
            "episode-query",
            entities=[name, other],
            prop="collection",
            kind=0,
        )
        for question in (
            f"What is {name}'s favorite television series?",
            f"Who is {name}'s dentist?",
        ):
            add(
                question,
                "User",
                "",
                [],
                False,
                [],
                "episode-query",
                entities=[name],
                prop="",
                kind=0,
            )
    return rows
