"""Independent fit-only relation and paraphrase curriculum for slot activation.

The source messages, questions, and supporting pairs are authored here. No
benchmark or development/sealed material is loaded. Exact support text is a
training/scoring label; it must never be transported to the native reader.
Different facts with the same actor/topic belong to one ``bank_group``.
Material choices for distinct making projects can coexist; a sampled world
contains zero, one or two of those sources, with exact source-specific and
complete broad questions. Other groups retain one alternative source.
"""

from collections import Counter
from copy import deepcopy

COEXISTENCE_PROP = "material_preference"
BROAD_MATERIAL_QUESTION = (
    "List the materials {name} chooses for toy bridges, painted signs, notebook binding, "
    "model railway scenery, and shopping bags, with the reason for each choice."
)
UNSUPPORTED_MATERIAL_QUESTION = "Which material does {name} choose for a garden hammock, and why?"

FIT_ACTORS = ("Cyra", "Dorian", "Mavis", "Fenn", "Ivo", "Nyla", "Oren", "Petra")

# topic, source sentence, three distinct questions, temporal kind
PATTERNS = (
    (
        "work_change",
        "After years at a hotel reception desk, I became a locksmith because solving practical problems suits me better.",
        (
            "What occupation did {name} take up after hotel reception?",
            "Why did {name} prefer locksmithing to reception work?",
            "Which practical profession replaced {name}'s hotel job?",
        ),
        0,
    ),
    (
        "work_change",
        "The night shifts at the bakery disrupted my sleep, so I now work daytime hours as a parcel courier.",
        (
            "What made {name} stop working nights at the bakery?",
            "How does {name} earn money after leaving the bakery?",
            "What improvement to working hours did {name} get by becoming a courier?",
        ),
        0,
    ),
    (
        "work_change",
        "I gave up telephone sales and retrained as a dental assistant; I wanted a job where I could help people directly.",
        (
            "Why did {name} leave telephone sales?",
            "Which healthcare role did {name} train for after sales?",
            "What motivated {name}'s move into dental assistance?",
        ),
        0,
    ),
    (
        "work_change",
        "Working as a theater usher is my current livelihood, whereas I used to sort mail; I changed so I could be around live performances.",
        (
            "What is {name}'s current role at the theater?",
            "What attracted {name} away from mail sorting?",
            "Which former occupation preceded {name}'s usher job?",
        ),
        0,
    ),
    (
        "work_change",
        "I trained in shoe repair after closing my market stall, since a steady indoor workspace was more appealing than trading outdoors.",
        (
            "What trade did {name} learn after closing the stall?",
            "What did {name} dislike about the setting of market work?",
            "Why was shoe repair an appealing career change for {name}?",
        ),
        0,
    ),
    (
        "material_preference",
        "For toy bridges I use beech rather than oak, as the fine grain lets me cut small joints without splitting the pieces.",
        (
            "Which wood does {name} use for toy bridges?",
            "Why does {name} favor beech over oak for small joints?",
            "What property of beech helps {name} avoid split pieces?",
        ),
        0,
    ),
    (
        "material_preference",
        "My hand-painted signs are made from birch plywood; its smooth surface holds lettering more evenly than rough boards.",
        (
            "What material forms the base of {name}'s painted signs?",
            "What makes birch plywood suitable for {name}'s lettering?",
            "Why does {name} choose a smooth board for signs?",
        ),
        0,
    ),
    (
        "material_preference",
        "I prefer linen thread when binding notebooks because it stays strong after repeated bending along the spine.",
        (
            "Which thread does {name} prefer for notebook binding?",
            "Why is linen useful in {name}'s notebook spines?",
            "What repeated strain must {name}'s binding thread withstand?",
        ),
        0,
    ),
    (
        "material_preference",
        "Cork is my choice for model railway scenery, since it is easy to shape and absorbs the vibration from passing miniature trains.",
        (
            "What material does {name} choose for railway scenery?",
            "How does cork improve {name}'s model railway?",
            "What benefits does cork provide in {name}'s model railway?",
        ),
        0,
    ),
    (
        "material_preference",
        "To make reusable shopping bags I choose canvas, valuing its resistance to tearing when I carry heavy groceries.",
        (
            "Which fabric does {name} select for shopping bags?",
            "What quality of canvas matters to {name} when carrying groceries?",
            "Why does {name} want tear-resistant material for the bags?",
        ),
        0,
    ),
    (
        "relaxation",
        "After an argument I settle down by kneading bread dough alone in the kitchen; the repetitive movement slows my thoughts.",
        (
            "What helps {name} calm down after an argument?",
            "Where does {name} knead dough to feel settled?",
            "Why is bread making soothing for {name}?",
        ),
        0,
    ),
    (
        "relaxation",
        "A bath followed by listening to a comedy podcast is how I recover from a busy week, usually on Friday evening.",
        (
            "How does {name} recover after a busy week?",
            "When does {name} usually enjoy a bath and podcast?",
            "What kind of audio does {name} listen to while recovering from the week?",
        ),
        0,
    ),
    (
        "relaxation",
        "When I feel overwhelmed, I feed the ducks at the pond before breakfast; the quiet outdoor routine puts me at ease.",
        (
            "What outdoor routine eases {name}'s overwhelmed feelings?",
            "At what time of day does {name} feed the ducks?",
            "Where does {name} go for the calming morning routine?",
        ),
        0,
    ),
    (
        "relaxation",
        "I turn to folding paper animals when deadlines worry me, keeping a stack of colored sheets in the lounge for that purpose.",
        (
            "How does {name} handle anxiety about deadlines?",
            "Where does {name} keep paper for the relaxing activity?",
            "What does {name} make from paper when worried?",
        ),
        0,
    ),
    (
        "relaxation",
        "An hour of swimming at the indoor pool clears my mind after difficult meetings, and I normally go on Tuesday nights.",
        (
            "What clears {name}'s mind after difficult meetings?",
            "Which nights does {name} normally swim to unwind?",
            "Where does {name} spend the hour of relaxing exercise?",
        ),
        0,
    ),
    (
        "project_purpose",
        "I established a free laundry rota for elderly residents who struggle to carry heavy baskets to the laundromat.",
        (
            "Why did {name} establish a laundry rota?",
            "Who was {name}'s free laundry project intended to help?",
            "What difficulty prompted {name}'s washing assistance scheme?",
        ),
        1,
    ),
    (
        "project_purpose",
        "To help newcomers practice speaking, I arranged a weekly language lunch in the school dining room.",
        (
            "What was {name}'s goal in arranging language lunches?",
            "Where did {name} hold the weekly speaking activity?",
            "Which people benefit from {name}'s language lunch?",
        ),
        1,
    ),
    (
        "project_purpose",
        "The spare-coat rack I placed in the bus shelter was meant to keep stranded travelers warm during cold nights.",
        (
            "What purpose did {name}'s spare-coat rack serve?",
            "Where did {name} place coats for stranded travelers?",
            "Why did {name} offer travelers spare coats?",
        ),
        1,
    ),
    (
        "project_purpose",
        "I created an audio newsletter so residents with poor eyesight could follow local announcements without reading small print.",
        (
            "Why did {name} create an audio newsletter?",
            "Which accessibility problem did {name}'s newsletter address?",
            "How did {name} make local announcements easier to follow?",
        ),
        1,
    ),
    (
        "project_purpose",
        "Our playground cleanup began after I noticed broken glass near the swings; I organized it to make the area safer for children.",
        (
            "What prompted {name} to organize a playground cleanup?",
            "Who was {name} trying to protect by clearing the playground?",
            "What hazard did {name} notice near the swings?",
        ),
        1,
    ),
    (
        "event_date",
        "On 7 February 2028 I toured the fire station with my aunt, and a firefighter demonstrated the rescue equipment.",
        (
            "When did {name} tour the fire station?",
            "Which relative joined {name} for the rescue-equipment demonstration?",
            "What demonstration did {name} see during the fire-station visit?",
        ),
        1,
    ),
    (
        "event_date",
        "My former school held a reunion on 19 October 2027, when I met my old classmates in the assembly hall.",
        (
            "On what date did {name} attend the school reunion?",
            "Where did {name} meet former classmates?",
            "What gathering brought {name} back to the old school?",
        ),
        1,
    ),
    (
        "event_date",
        "The pottery fair I attended with my neighbor took place on 3 July 2029 in the town square, and we watched a wheel demonstration.",
        (
            "When did {name} attend the pottery fair?",
            "Who went to the pottery fair with {name}?",
            "Where was the fair at which {name} watched a pottery wheel?",
        ),
        1,
    ),
    (
        "event_date",
        "I joined a harbor rescue drill on 11 November 2026 alongside my uncle, practicing how to throw a lifebuoy from the quay.",
        (
            "What was the date of {name}'s rescue drill?",
            "Which relative practiced throwing a lifebuoy with {name}?",
            "What rescue skill did {name} practice at the harbor?",
        ),
        1,
    ),
    (
        "event_date",
        "At the planetarium on 25 January 2029, my friend and I attended a demonstration explaining how eclipses occur.",
        (
            "When did {name} attend the eclipse demonstration?",
            "Where did {name} learn about eclipses with a friend?",
            "What subject was explained during {name}'s planetarium visit?",
        ),
        1,
    ),
    (
        "collection",
        "I collect enamel advertising badges and fasten them to a padded display panel in the hallway.",
        (
            "What does {name} collect from old advertising?",
            "How are {name}'s enamel badges displayed?",
            "Where does {name} keep the badge display?",
        ),
        0,
    ),
    (
        "collection",
        "The old bus timetables I save are filed by route number in ring binders, with the oldest pages inside protective sleeves.",
        (
            "How does {name} organize saved bus timetables?",
            "What protects the oldest pages in {name}'s timetable collection?",
            "What printed transport material does {name} save?",
        ),
        0,
    ),
    (
        "collection",
        "My collection consists of seashells from different beaches; I label each specimen with the place where I found it.",
        (
            "What objects make up {name}'s beach collection?",
            "How does {name} record the origin of each shell?",
            "What information appears on {name}'s specimen labels?",
        ),
        0,
    ),
    (
        "collection",
        "I gather vintage harmonicas, storing them in separate cloth pouches so their metal surfaces do not scratch one another.",
        (
            "Which vintage musical objects does {name} gather?",
            "How does {name} store the harmonicas?",
            "Why does {name} put each harmonica in a separate pouch?",
        ),
        0,
    ),
    (
        "collection",
        "Ceramic animal figurines are my collecting interest, and the delicate pieces sit behind the glass doors of a cabinet.",
        (
            "What kind of figurines interest {name} as a collector?",
            "Where are {name}'s delicate ceramic animals kept?",
            "What separates {name}'s fragile figurines from the open room?",
        ),
        0,
    ),
    (
        "residence",
        "I live in a converted mill in Leuven, with a footpath that lets me reach the railway station in ten minutes.",
        (
            "Which city contains {name}'s converted-mill home?",
            "How long does {name}'s walk to the station take?",
            "What kind of building does {name} live in near the station?",
        ),
        0,
    ),
    (
        "residence",
        "My current home is a cottage outside Durham, and the nearest grocery shop is fifteen minutes away by car.",
        (
            "Where is {name}'s cottage located?",
            "How does {name} get to the nearest grocery shop from home?",
            "How much driving separates {name}'s home from the shop?",
        ),
        0,
    ),
    (
        "residence",
        "I rent rooms above a florist in Ghent, close enough to my office that I make the journey on foot.",
        (
            "In which city does {name} rent rooms above a florist?",
            "How does {name} travel from home to the office?",
            "What business occupies the floor beneath {name}'s rented rooms?",
        ),
        0,
    ),
    (
        "residence",
        "My house is on the edge of Siena; a bus from the end of our lane takes me to the central market.",
        (
            "Which city is {name}'s house on the edge of?",
            "What transport takes {name} from home to the central market?",
            "Where does {name} catch the bus near the house?",
        ),
        0,
    ),
    (
        "residence",
        "I share a house in Uppsala with two friends, and we can reach the public swimming pool in a short walk.",
        (
            "Where does {name} share a house with friends?",
            "How many friends share {name}'s house?",
            "How far is the public pool from {name}'s shared home?",
        ),
        0,
    ),
    (
        "family_plan",
        "In the next spring holiday, I hope to help my grandmother digitize her handwritten recipes using a scanner.",
        (
            "What does {name} hope to do with their grandmother's recipes?",
            "When is {name}'s recipe-scanning project planned?",
            "Which device will {name} use to preserve the handwritten recipes?",
        ),
        2,
    ),
    (
        "family_plan",
        "I plan to take my younger brother on his first camping trip in late August, teaching him to pitch a tent beside the lake.",
        (
            "What does {name} plan to teach their brother while camping?",
            "When will {name}'s brother take his first camping trip?",
            "Where is {name} planning to pitch the tent with their brother?",
        ),
        2,
    ),
    (
        "family_plan",
        "My father wants a family photo display, so next month I intend to help him arrange portraits in a large wall frame.",
        (
            "What display will {name} help their father arrange?",
            "When does {name} intend to work on the portrait frame?",
            "Why is {name} planning to arrange family portraits?",
        ),
        2,
    ),
    (
        "family_plan",
        "During the coming school break, I would like to teach my little sister to sew a cloth puppet for a play she is preparing.",
        (
            "What does {name} want to teach their sister to sew?",
            "When does {name} intend to make the puppet with their sister?",
            "Why does {name}'s sister need a cloth puppet?",
        ),
        2,
    ),
    (
        "family_plan",
        "My cousin has asked for help starting a vegetable patch, and I expect to plant the first seedlings with him in early May.",
        (
            "What gardening project is {name} planning with their cousin?",
            "When does {name} expect to plant the first seedlings?",
            "Who asked {name} to help start a vegetable patch?",
        ),
        2,
    ),
    (
        "dietary",
        "Milk gives me stomach cramps, so I replace it with oat drink and check whether sauces contain dairy before ordering.",
        (
            "Why does {name} replace milk with oat drink?",
            "What does {name} check about sauces when ordering?",
            "Which ingredient causes {name}'s stomach cramps?",
        ),
        0,
    ),
    (
        "dietary",
        "I get headaches after eating aged cheese; when friends serve it, I politely choose the fruit platter instead.",
        (
            "What food gives {name} headaches?",
            "What does {name} choose instead of aged cheese at a gathering?",
            "Why does {name} decline cheese offered by friends?",
        ),
        0,
    ),
    (
        "dietary",
        "Sesame seeds make my throat itch, which is why I ask bakers whether their bread contains them.",
        (
            "Which seeds does {name} ask bakers about?",
            "Why does {name} check the ingredients of bread?",
            "What reaction does {name} have to sesame?",
        ),
        0,
    ),
    (
        "dietary",
        "Very spicy food triggers heartburn for me, so I request a mild sauce when eating curry at restaurants.",
        (
            "Why does {name} request mild curry sauce?",
            "What problem does spicy food cause for {name}?",
            "How does {name} adjust curry orders at restaurants?",
        ),
        0,
    ),
    (
        "dietary",
        "Raw onions leave me feeling bloated; I avoid them in salads even though cooked onions do not bother me.",
        (
            "What ingredient does {name} avoid in salads?",
            "Why does {name} leave raw onions out of meals?",
            "Which form of onion can {name} eat without discomfort?",
        ),
        0,
    ),
    (
        "course",
        "I enrolled in a furniture upholstery class to restore the worn armchairs that belonged to my parents.",
        (
            "What class did {name} enroll in to restore armchairs?",
            "Why does {name} want to learn upholstery?",
            "Whose armchairs motivated {name}'s furniture class?",
        ),
        0,
    ),
    (
        "course",
        "A beginner sign-language course will help me communicate with a deaf coworker during breaks.",
        (
            "Which language is {name} learning for workplace conversations?",
            "What is {name}'s reason for taking sign language?",
            "Who does {name} want to communicate with during breaks?",
        ),
        0,
    ),
    (
        "course",
        "I am taking a bookkeeping course because I want to manage the accounts of our volunteer association accurately.",
        (
            "What course is {name} taking to improve account management?",
            "Why does {name} want bookkeeping skills?",
            "Whose accounts will {name} manage with the new skills?",
        ),
        0,
    ),
    (
        "course",
        "To care for my aging dog more confidently, I signed up for a weekend animal first-aid workshop.",
        (
            "What workshop did {name} choose to help an aging pet?",
            "Why does {name} want animal first-aid training?",
            "Which animal prompted {name} to sign up for first aid?",
        ),
        0,
    ),
    (
        "course",
        "I attend an introductory welding class on Wednesday evenings, hoping to build a metal gate for our yard.",
        (
            "What skill does {name} study on Wednesday evenings?",
            "What project motivates {name}'s welding lessons?",
            "When does {name} attend the class for making a metal gate?",
        ),
        0,
    ),
    (
        "achievement",
        "On 16 September 2028 my knitted scarf won the judges' choice ribbon at the county textile fair.",
        (
            "What recognition did {name}'s knitted scarf receive?",
            "When did {name} win the textile-fair ribbon?",
            "Where was {name}'s scarf chosen by the judges?",
        ),
        1,
    ),
    (
        "achievement",
        "The local theater honored my costume design with the audience favorite award on 8 December 2027.",
        (
            "Which award recognized {name}'s costume design?",
            "On what date did {name} receive the theater award?",
            "What creative work earned {name} recognition from the theater?",
        ),
        1,
    ),
    (
        "achievement",
        "My homemade jam took second place at the harvest festival on 20 August 2029, and I received a silver rosette.",
        (
            "What placing did {name}'s jam achieve at the festival?",
            "When did {name} receive a silver rosette for jam?",
            "What prize accompanied {name}'s second-place result?",
        ),
        1,
    ),
    (
        "achievement",
        "At the volunteer dinner on 4 March 2028 I was given a service certificate for organizing deliveries throughout the flood.",
        (
            "What recognition did {name} receive at the volunteer dinner?",
            "When was {name} given the service certificate?",
            "What flood-related work earned {name} the certificate?",
        ),
        1,
    ),
    (
        "achievement",
        "The school art exhibition selected my clay dragon for the imagination medal on 13 June 2026.",
        (
            "Which medal did {name}'s clay dragon win?",
            "What was the date of {name}'s art-exhibition success?",
            "What sculpture earned {name} the imagination medal?",
        ),
        1,
    ),
    (
        "possession",
        "I own a red travel kettle with a folding handle, and I keep it on the pantry shelf when it is not in my suitcase.",
        (
            "What appliance does {name} take on trips?",
            "Where does {name} store the travel kettle between trips?",
            "What features distinguish {name}'s kettle?",
        ),
        0,
    ),
    (
        "possession",
        "My new raincoat has reflective strips and a removable hood; it hangs beside the front door for wet commutes.",
        (
            "What features does {name}'s raincoat have?",
            "Where does {name} hang the coat used for wet commutes?",
            "Which part of {name}'s raincoat can be removed?",
        ),
        0,
    ),
    (
        "possession",
        "The secondhand record player I bought has a walnut case and sits on a narrow shelf above the sofa.",
        (
            "What audio equipment did {name} buy secondhand?",
            "Where does {name} keep the record player?",
            "What is the case of {name}'s record player made from?",
        ),
        0,
    ),
    (
        "possession",
        "I carry a green insulated lunch box with two separate compartments, which fits in the lower pocket of my backpack.",
        (
            "What kind of lunch container does {name} carry?",
            "Where in the backpack does {name}'s lunch box fit?",
            "How is {name}'s lunch box divided inside?",
        ),
        0,
    ),
    (
        "possession",
        "My grandfather gave me a silver pocket watch with a chain, and I store it in a velvet pouch in my dresser.",
        (
            "What gift did {name} receive from their grandfather?",
            "How does {name} protect the silver watch when storing it?",
            "Where does {name} keep the pocket watch?",
        ),
        0,
    ),
)

NULL_QUESTIONS = (
    "How tall is {name}?",
    "What is the registration number of {name}'s car?",
    "Who was {name}'s first primary-school teacher?",
    "What is {name}'s favorite breakfast cereal?",
    "On what date was {name} born?",
    "What is the name of {name}'s childhood best friend?",
    "Which publisher printed {name}'s travel diary?",
    "What is the password for {name}'s home internet connection?",
)


def reader_expansion():
    """Return fit facts and paired queries without changing any other split.

    ``bank_group`` and source scopes are fit labels only. Original specific
    questions keep their paired source. Broad material questions enumerate
    five independently authored uses, with labels bound to each sampled world.
    """
    facts, queries = [], []
    for name in FIT_ACTORS:
        for index, (prop, content, questions, kind) in enumerate(PATTERNS):
            text = name + ". " + content
            group = [name, prop]
            pattern_id = "reader-expansion-" + str(index)
            facts.append(
                dict(
                    text=text,
                    entity=name,
                    reference=name,
                    prop=prop,
                    kind=kind,
                    episode=True,
                    expansion=True,
                    bank_group=group,
                    pattern_id=pattern_id,
                )
            )
            for question in questions:
                queries.append(
                    dict(
                        text=question.format(name=name),
                        entities=[name],
                        prop=prop,
                        kind=kind,
                        support_texts=[text],
                        semantic_specific=True,
                        bank_group=group,
                        pattern_id=pattern_id,
                        expansion=True,
                    )
                )
        for question in NULL_QUESTIONS:
            queries.append(
                dict(
                    text=question.format(name=name),
                    entities=[name],
                    prop="",
                    kind=0,
                    support_texts=[],
                    expansion=True,
                )
            )
        sources = [fact["text"] for fact in facts
                   if fact["entity"] == name and fact["prop"] == COEXISTENCE_PROP]
        queries.extend((
            dict(text=BROAD_MATERIAL_QUESTION.format(name=name), entities=[name],
                 prop=COEXISTENCE_PROP, kind=0, support_texts=list(sources),
                 semantic_specific=False, bank_group=[name, COEXISTENCE_PROP],
                 pattern_id="reader-expansion-material-broad", expansion=True,
                 expansion_query_class="broad", broad_scope_source_texts=list(sources)),
            dict(text=UNSUPPORTED_MATERIAL_QUESTION.format(name=name), entities=[name],
                 prop=COEXISTENCE_PROP, kind=0, support_texts=[],
                 semantic_specific=True, bank_group=[name, COEXISTENCE_PROP],
                 pattern_id="reader-expansion-material-unsupported", expansion=True,
                 expansion_query_class="unsupported"),
        ))
    validate(dict(facts=facts, queries=queries))
    return dict(facts=facts, queries=queries)


def validate(group):
    """Check label consistency, source membership, and accidental collisions."""
    counts = Counter(p[0] for p in PATTERNS)
    if len(PATTERNS) != 60 or len(counts) != 12 or set(counts.values()) != {5}:
        raise ValueError("expected five independently authored patterns per topic")
    if len(set(FIT_ACTORS)) != len(FIT_ACTORS):
        raise ValueError("duplicate training actors")
    facts = {f["text"]: f for f in group["facts"]}
    if len(facts) != len(group["facts"]):
        raise ValueError("duplicate factual source")
    seen = set()
    paired = Counter()
    for row in group["queries"]:
        if row["text"] in seen:
            raise ValueError("question has multiple alternative source labels")
        seen.add(row["text"])
        if "reference" in row or not row["entities"]:
            raise ValueError("raw questions must use their own learned identity")
        category = row.get("expansion_query_class")
        if category not in (None, "broad", "unsupported"):
            raise ValueError("unknown authored expansion query class")
        if category == "broad":
            actor = row["entities"][0]
            scope = [fact["text"] for fact in group["facts"]
                     if fact["entity"] == actor and fact["prop"] == COEXISTENCE_PROP]
            if (len(row["entities"]) != 1 or row["kind"] != 0
                    or row["prop"] != COEXISTENCE_PROP or len(scope) != 5
                    or row.get("semantic_specific") is not False
                    or row.get("broad_scope_source_texts") != scope
                    or row["support_texts"] != scope
                    or row["text"] != BROAD_MATERIAL_QUESTION.format(name=actor)):
                raise ValueError("broad material question must name its complete authored source scope")
        elif category == "unsupported":
            if (len(row["entities"]) != 1 or row["kind"] != 0
                    or row["prop"] != COEXISTENCE_PROP or row["support_texts"]
                    or row.get("semantic_specific") is not True
                    or row["text"] != UNSUPPORTED_MATERIAL_QUESTION.format(name=row["entities"][0])):
                raise ValueError("unsupported material question must have an explicit absent object")
        elif bool(row["prop"]) != bool(row["support_texts"]):
            raise ValueError("unsupported question must have empty support")
        if row["support_texts"] and category != "broad" and row.get("semantic_specific") is not True:
            raise ValueError("paired question must have a source-specific semantic label")
        for text in row["support_texts"]:
            if text not in facts:
                raise ValueError("question support is not an authored fact")
            fact = facts[text]
            if (
                fact["entity"] not in row["entities"]
                or fact["prop"] != row["prop"]
                or fact["kind"] != row["kind"]
                or fact["bank_group"] != row["bank_group"]
            ):
                raise ValueError("question/source identity or topic disagrees")
            if category != "broad":
                paired[text] += 1
    if set(paired) != set(facts) or set(paired.values()) != {3}:
        raise ValueError("each fact must have three source-specific questions")


def sample_expansion_world(group, rng, *, material_counts=(0, 1, 2), source_counts=None):
    """Return one fit bank and its complete world-bound question labels.

    Only authored material-choice groups allow zero/one/two simultaneous
    sources. All other groups contain one alternative, avoiding contradictory
    current jobs or homes. ``source_counts`` may fix material counts by
    ``(actor, property)``. ``material_counts=(1,)`` provides a single-source
    control; default counts cover 0..2 uniformly. Both use the same source
    permutation and count draw, keeping other groups' random worlds identical.
    Specific absent-source questions retain their original author label;
    activation uses its intersection with the actual bank. A broad question's
    supports are exactly all present members of its declared source scope.
    Evidence is validated and filtered before label binding; no input text,
    original subject, cue, reference or factual source is synthesized here.
    """
    from neural_reader_evidence_curriculum import filter_bank_evidence_queries

    canonical = reader_expansion()
    raw = [row for row in group["queries"] if not row.get("augmented", False)]
    validate(dict(facts=group["facts"], queries=raw))
    if (group["facts"] != canonical["facts"]
            or {row["text"]: row for row in raw} != {row["text"]: row for row in canonical["queries"]}):
        raise ValueError("world sampling requires the complete current authored fit expansion")
    # Validate all existing cues against the full author bank before sampling.
    if len(filter_bank_evidence_queries(group["queries"], group["facts"])) != len(group["queries"]):
        raise ValueError("fit evidence cites a source outside the authored expansion")
    raw_by_text = {row["text"]: row for row in raw}
    for row in group["queries"]:
        if row.get("augmented"):
            initial = raw_by_text.get(row["initial_text"])
            if initial is None or any(row.get(field) != initial.get(field) for field in (
                    "entities", "prop", "kind", "support_texts", "semantic_specific",
                    "expansion_query_class", "broad_scope_source_texts")):
                raise ValueError("fit evidence query must preserve its original author scope and supports")
    groups = {}
    for fact in group["facts"]:
        groups.setdefault(tuple(fact["bank_group"]), []).append(fact)
    allowed = {key for key in groups if key[1] == COEXISTENCE_PROP}
    if material_counts not in ((1,), (0, 1, 2)) or not isinstance(material_counts, tuple):
        raise ValueError("material counts must select the single-source control or 0..2-source curriculum")
    if source_counts is None:
        source_counts = {}
    if (not isinstance(source_counts, dict) or not set(source_counts) <= allowed
            or any(type(count) is not int or count not in (0, 1, 2)
                   for count in source_counts.values())):
        raise ValueError("source counts must select 0..2 authored material sources only")
    bank = []
    for key, alternatives in groups.items():
        if key not in allowed:
            bank.append(rng.choice(alternatives))
            continue
        ordered = list(alternatives)
        rng.shuffle(ordered)
        draw = rng.randrange(3)
        count = source_counts[key] if key in source_counts else material_counts[draw % len(material_counts)]
        bank.extend(ordered[:count])
    present = {fact["text"] for fact in bank}
    queries = deepcopy(filter_bank_evidence_queries(group["queries"], bank))
    for row in queries:
        category = row.get("expansion_query_class")
        if category == "broad":
            row["support_texts"] = [text for text in row["broad_scope_source_texts"] if text in present]
            row["expansion_world_class"] = "broad_present" if row["support_texts"] else "broad_empty"
        elif category == "unsupported" or not row["prop"]:
            row["expansion_world_class"] = "unsupported"
        else:
            row["expansion_world_class"] = "source_present" if present.intersection(row["support_texts"]) else "source_absent"
    return dict(facts=deepcopy(bank), queries=queries)


if __name__ == "__main__":
    import json

    curriculum = reader_expansion()
    print(
        json.dumps(
            dict(
                facts=len(curriculum["facts"]),
                queries=len(curriculum["queries"]),
                null_queries=sum(not q["support_texts"] for q in curriculum["queries"]),
                actors=len(FIT_ACTORS),
                patterns=len(PATTERNS),
                topics=dict(Counter(p[0] for p in PATTERNS)),
                training_only=True,
            ),
            indent=2,
        )
    )
