"""Independent synthetic dialogue curriculum. No benchmark/data files are read.

Metadata is explicit input provenance. Labels are raw byte addresses and model
decisions, used only by training/evaluation, never transmitted to native serving.
Entity and sentence-style splits both differ between fit/dev/heldout.
"""

import random
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
        "Ines",
        "Jules",
        "Kira",
        "Leon",
        "Mira",
        "Nico",
        "Opal",
        "Pia",
        "Ronan",
        "Sable",
        "Talia",
        "Uma",
        "Yara",
        "Zeno",
        "Amos",
        "Bea",
    ],
    "development": ["Quill", "Rhea", "Sylas", "Tessa"],
    "heldout": ["Ula", "Vito", "Wren", "Xara"],
}


def dialogue_curriculum(split):
    rng = random.Random({"fit": 5101, "development": 5102, "heldout": 5103}[split])
    rows = []
    patterns = {
        "fit": [
            ("I live in Bergen.", "city", "Where do I live?"),
            (
                "My favorite meal is vegetable stew.",
                "food",
                "What is my favorite meal?",
            ),
            ("I design furniture for a living.", "job", "What do I do for a living?"),
            (
                "I enjoy making ceramics in my free time.",
                "hobby",
                "What hobby do I enjoy?",
            ),
            (
                "I attended a poetry workshop yesterday.",
                "event",
                "What workshop did I attend?",
            ),
            (
                "I adopted a dog named Pepper last weekend.",
                "pet",
                "What is my dog's name?",
            ),
            (
                "I felt relieved after finishing my driving exam.",
                "emotion",
                "Why did I feel relieved?",
            ),
            (
                "I avoid walnuts because they make me ill.",
                "health",
                "What food do I avoid?",
            ),
            (
                "I plan to study astronomy next autumn.",
                "plan",
                "What do I plan to study?",
            ),
            (
                "I prefer quiet places when I need to concentrate.",
                "preference",
                "What helps me concentrate?",
            ),
            (
                "I visited a science museum with my brother on Saturday.",
                "event",
                "Who visited the museum with me?",
            ),
            (
                "I completed a volunteer shift at the animal shelter.",
                "event",
                "Where did I volunteer?",
            ),
        ],
        "development": [
            ("My home is in Porto.", "city", "Where is my home?"),
            (
                "I usually choose lentil curry for dinner.",
                "food",
                "What do I usually eat for dinner?",
            ),
            (
                "Building bicycles is how I earn my living.",
                "job",
                "How do I earn my living?",
            ),
            (
                "Painting miniature figures is my favorite pastime.",
                "hobby",
                "What is my favorite pastime?",
            ),
            (
                "Yesterday I took part in a pottery lesson.",
                "event",
                "What lesson did I take?",
            ),
            (
                "My cat Juniper joined our family last month.",
                "pet",
                "What is my cat called?",
            ),
            (
                "Passing the sailing test made me feel proud.",
                "emotion",
                "What made me proud?",
            ),
            (
                "Eating almonds makes me sick, so I stay away from them.",
                "health",
                "What makes me sick?",
            ),
            (
                "I intend to learn geology after summer.",
                "plan",
                "What subject will I learn?",
            ),
            (
                "For focused work I need a peaceful room.",
                "preference",
                "What kind of room do I need for work?",
            ),
        ],
        "heldout": [
            ("These days my hometown is Quito.", "city", "What is my hometown?"),
            (
                "Rice noodles are the dish I like best.",
                "food",
                "Which dish do I like best?",
            ),
            (
                "I make my income by repairing clocks.",
                "job",
                "What work provides my income?",
            ),
            (
                "My spare hours go into photographing birds.",
                "hobby",
                "How do I spend my spare hours?",
            ),
            (
                "A drawing class was where I spent yesterday evening.",
                "event",
                "What class did I attend?",
            ),
            (
                "We welcomed a rabbit called Clover two weeks ago.",
                "pet",
                "What is my rabbit's name?",
            ),
            (
                "After the hiking challenge, I was delighted to have finished.",
                "emotion",
                "Why was I delighted?",
            ),
            (
                "I cannot eat pecans without getting an upset stomach.",
                "health",
                "What nuts cause me an upset stomach?",
            ),
            (
                "Marine biology is what I hope to study next spring.",
                "plan",
                "What do I hope to study?",
            ),
            (
                "I think best in surroundings that are calm and silent.",
                "preference",
                "Where do I think best?",
            ),
        ],
    }

    if split == "fit":
        patterns["fit"] += [
            (
                "Repairing equipment is my source of earnings.",
                "job",
                "What work brings me earnings?",
            ),
            ("I get paid to mend watches.", "job", "What do I get paid to do?"),
            (
                "My salary comes from servicing machines.",
                "job",
                "What provides my salary?",
            ),
            (
                "I support myself by restoring old radios.",
                "job",
                "How do I support myself?",
            ),
            (
                "I focus most effectively with very little noise.",
                "preference",
                "Where can I focus effectively?",
            ),
            (
                "A serene environment helps my thoughts flow.",
                "preference",
                "What environment helps my thoughts?",
            ),
            (
                "Before moving, I was glad to have sold my old bicycle.",
                "emotion",
                "Why was I glad before moving?",
            ),
            (
                "Finishing the art project left me excited to have succeeded.",
                "emotion",
                "What left me excited?",
            ),
            (
                "Architecture is the subject I aspire to study next year.",
                "plan",
                "What subject do I aspire to study?",
            ),
            (
                "Sculpture is what I want to learn once spring arrives.",
                "plan",
                "What do I want to learn?",
            ),
            (
                "I do my finest thinking somewhere with almost no sound.",
                "preference",
                "Where do I do my finest thinking?",
            ),
            (
                "We brought a kitten named Finch into our house yesterday.",
                "pet",
                "What kitten did we bring home?",
            ),
            (
                "I've made Bergen my permanent base.",
                "city",
                "Where is my permanent base?",
            ),
            (
                "My current address is in Bergen.",
                "city",
                "Where is my current address?",
            ),
            (
                "The meal I most enjoy eating is vegetable stew.",
                "food",
                "What meal do I most enjoy?",
            ),
            (
                "Nothing beats vegetable stew for my lunch.",
                "food",
                "What lunch do I like?",
            ),
            (
                "My occupation involves designing furniture.",
                "job",
                "What does my occupation involve?",
            ),
            ("I earn money creating wooden cabinets.", "job", "How do I earn money?"),
            (
                "Taking landscape photographs is something I do for fun.",
                "hobby",
                "What do I do for fun?",
            ),
            (
                "I devote my leisure time to collecting stamps.",
                "hobby",
                "What fills my leisure time?",
            ),
            (
                "A watercolor lesson occupied my Monday night.",
                "event",
                "What occupied my Monday night?",
            ),
            (
                "I spent last night participating in a coding seminar.",
                "event",
                "What did I participate in last night?",
            ),
            (
                "Our new companion is a hamster named Biscuit.",
                "pet",
                "What is our hamster's name?",
            ),
            (
                "A puppy called Moss became part of our household recently.",
                "pet",
                "What is our puppy called?",
            ),
            (
                "I was pleased when the baking competition ended.",
                "emotion",
                "When was I pleased?",
            ),
            (
                "The successful interview brought me a lot of joy.",
                "emotion",
                "What brought me joy?",
            ),
            (
                "I became excited after receiving my exam results.",
                "emotion",
                "What made me excited?",
            ),
            (
                "Completing the marathon filled me with satisfaction.",
                "emotion",
                "What filled me with satisfaction?",
            ),
            (
                "Cashews upset my digestion and I cannot eat them.",
                "health",
                "What can I not eat?",
            ),
            (
                "I get an allergic reaction from peanuts.",
                "health",
                "What causes my allergic reaction?",
            ),
            (
                "I'm hoping to take a biology course in spring.",
                "plan",
                "What course am I hoping to take?",
            ),
            (
                "I would like to study chemistry next term.",
                "plan",
                "What would I like to study?",
            ),
            (
                "My goal for the coming year is studying history.",
                "plan",
                "What is my goal for the coming year?",
            ),
            (
                "I wish to learn navigation once winter is over.",
                "plan",
                "What do I wish to learn?",
            ),
            (
                "Silence helps me carry out difficult work.",
                "preference",
                "What helps me carry out difficult work?",
            ),
            (
                "I find concentrating easier in a quiet library.",
                "preference",
                "Where do I find concentrating easier?",
            ),
            ("At present I am settled in Bergen.", "city", "Which town is my home?"),
            ("Bergen is the city I call home.", "city", "What city do I call home?"),
            (
                "Vegetable stew is my usual lunch.",
                "food",
                "Which meal do I usually choose?",
            ),
            (
                "For lunch I tend to pick vegetable stew.",
                "food",
                "What do I tend to eat?",
            ),
            ("For work, I design wooden cabinets.", "job", "What is my occupation?"),
            ("Designing cabinets pays my bills.", "job", "What pays my bills?"),
            (
                "Making ceramics fills much of my weekend.",
                "hobby",
                "What do I do at weekends?",
            ),
            (
                "The activity I enjoy most is making ceramics.",
                "hobby",
                "What activity do I enjoy most?",
            ),
            (
                "Yesterday evening I joined an improv lesson.",
                "event",
                "What did I join yesterday?",
            ),
            (
                "An improv lesson kept me busy yesterday.",
                "event",
                "Which lesson kept me busy?",
            ),
            (
                "Pepper is the dog I welcomed last weekend.",
                "pet",
                "What pet did I welcome?",
            ),
            (
                "Our newest family member is my dog Pepper.",
                "pet",
                "Who is my newest family member?",
            ),
            (
                "Finishing the kayaking course left me happy.",
                "emotion",
                "What left me happy?",
            ),
            (
                "Getting my driving license was a relief for me.",
                "emotion",
                "What was a relief for me?",
            ),
            (
                "Walnuts give me a stomachache, so I avoid them.",
                "health",
                "What gives me a stomachache?",
            ),
            (
                "Because walnuts upset my stomach, I never eat them.",
                "health",
                "What do I never eat?",
            ),
            (
                "My intention for next winter is to learn botany.",
                "plan",
                "What is my intention for winter?",
            ),
            (
                "Learning botany is part of my plans for next winter.",
                "plan",
                "What is part of my plans?",
            ),
            (
                "A tranquil room makes it easier for me to think.",
                "preference",
                "What makes thinking easier?",
            ),
            (
                "To get work done, I prefer a room without noise.",
                "preference",
                "What environment do I prefer?",
            ),
        ]

    def add(
        body, speaker, timestamp, past, spans, whole_need=False, whole_refs=(), tag=""
    ):
        rows.append(
            dict(
                body=body,
                speaker=speaker,
                timestamp=timestamp,
                past=past,
                source_id=f"synthetic-{split}-{len(rows)}",
                context=observation_context(speaker, timestamp, past),
                spans=spans,
                whole_need=whole_need,
                whole_refs=list(whole_refs),
                tag=tag,
            )
        )

    def candidate(body, a, b, action, refs=(), need=False, prop="", kind=0):
        return dict(
            start=a,
            end=b,
            action=action,
            need=need,
            references=list(refs),
            prop=prop,
            kind=kind,
        )

    for i, name in enumerate(NAMES[split]):
        other = NAMES[split][(i + 1) % len(NAMES[split])]
        timestamp = f"2027-{(i%9)+1:02d}-{(i%20)+3:02d} 14:35 UTC"
        history = f"user: {other} enjoys puzzles.\nassistant: Thanks for telling me.\n"
        for j, (fact, prop, question) in enumerate(patterns[split]):
            need = True
            refs = [name]
            for prefix, suffix, past in [
                ("", "", history),
                ("Hello again! ", " How has your day been?", history),
                ("Here is something about me: ", " Thanks for listening.", ""),
            ]:
                body = prefix + fact + suffix
                a = len(prefix.encode())
                b = a + len(fact.encode())
                add(
                    body,
                    name,
                    timestamp,
                    past,
                    [candidate(body, a, b, 1, refs, need, prop)],
                    True,
                    refs,
                    "natural-fact",
                )
                rows[-1]["recall_question"] = question
            add(question, name, timestamp, history, [], True, refs, "personal-query")
            # Quotes/fiction/tasks can contain a complete factual-looking clause.
            for wrapper in (
                'Translate this sentence: "{}"',
                "I am writing a fictional scene: {}",
            ):
                add(
                    wrapper.format(fact),
                    name,
                    timestamp,
                    history,
                    [],
                    False,
                    [],
                    "negative-quotation",
                )
        for body in [
            "Hello! How are you doing?",
            "Could you explain how rainbows form?",
            "Thank you, that was helpful.",
            "Can you help me solve this puzzle?",
        ]:
            add(body, name, timestamp, history, [], False, [], "ordinary-negative")
        # Explicit third-party subjects must not inherit the current speaker.
        text = f"{other} lives in Rome."
        add(
            text,
            name,
            timestamp,
            history,
            [candidate(text, 0, len(text.encode()), 1, [], False, "city")],
            False,
            [],
            "explicit-other",
        )
        # First-person and third-person reference planning share the same decoder.
        text = "He now lives in Paris."
        past = f"user: We are talking about {other}.\nassistant: Understood.\n"
        add(
            text,
            name,
            timestamp,
            past,
            [candidate(text, 0, len(text.encode()), 2, [other], True, "city")],
            True,
            [other],
            "reference-update",
        )
        text = "He lives in Tokyo."
        past = f"user: {name} and {other} are friends.\nassistant: What about them?\n"
        add(
            text,
            name,
            timestamp,
            past,
            [candidate(text, 0, len(text.encode()), 1, [], True, "city")],
            True,
            [],
            "ambiguous-reference",
        )
        a = f"{other} lives in Rome."
        b = "He enjoys gardening."
        body = a + " " + b
        add(
            body,
            name,
            timestamp,
            "",
            [
                candidate(body, 0, len(a.encode()), 1, [], False, "city"),
                candidate(
                    body,
                    len(a.encode()) + 1,
                    len(body.encode()),
                    1,
                    [other],
                    True,
                    "hobby",
                ),
            ],
            False,
            [],
            "causal-intra-message",
        )
        a = "I live in Bergen."
        b = f"{other} prefers soup."
        body = "Good morning! " + a + " " + b + " What do you think?"
        start = len("Good morning! ".encode())
        add(
            body,
            name,
            timestamp,
            history,
            [
                candidate(
                    body, start, start + len(a.encode()), 1, [name], True, "city"
                ),
                candidate(
                    body,
                    start + len(a.encode()) + 1,
                    start + len(a.encode()) + 1 + len(b.encode()),
                    1,
                    [],
                    False,
                    "food",
                ),
            ],
            False,
            [],
            "mixed-subjects",
        )
        for body, action, prop, kind in [
            ("I now live in Paris.", 2, "city", 0),
            ("I used to live in Rome.", 1, "city", 1),
            ("I may move to Tallinn next year.", 1, "plan", 2),
            ("Forget my home.", 3, "city", 0),
            ("Forget my food preference.", 3, "food", 0),
        ]:
            add(
                body,
                name,
                timestamp,
                history,
                [
                    candidate(
                        body, 0, len(body.encode()), action, [name], True, prop, kind
                    )
                ],
                True,
                [name],
                "lifecycle",
            )
        # Pasted transcripts can carry literal metadata in the body; source heads
        # must exclude that prefix and select the observation with attribution.
        prefix = f"[{timestamp}] {name}: "
        fact = "I enjoy gardening."
        body = prefix + fact
        add(
            body,
            "User",
            None,
            "",
            [
                candidate(
                    body,
                    len(prefix.encode()),
                    len(body.encode()),
                    1,
                    [name],
                    True,
                    "hobby",
                )
            ],
            True,
            [name],
            "literal-provenance-prefix",
        )
        # Unknown anonymous speaker has a real role, distinct from a guessed name.
        body = "I live in Bergen."
        add(
            body,
            None,
            None,
            "",
            [candidate(body, 0, len(body.encode()), 1, ["User"], True, "city")],
            True,
            ["User"],
            "anonymous-self",
        )
    rng.shuffle(rows)
    return rows
