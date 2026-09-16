#!/usr/bin/env python3
"""Generate large repeated-entity dialogue banks for neural activation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path


SPEAKERS = (
    "Avery", "Blake", "Cameron", "Dakota",
    "Emerson", "Finley", "Harper", "Jordan",
)

VALIDATION_SPEAKERS = (
    "Lennon", "Marley", "Nico", "Oakley",
    "Phoenix", "Reese", "Shiloh", "Winter",
)

TEST_SPEAKERS = (
    "Arden", "Briar", "Cleo", "Devin",
    "Ellis", "Frankie", "Gale", "Hollis",
)

PREDICATES = (
    {
        "name": "research topic",
        "values": (
            "accessible housing", "marine conservation",
            "renewable energy", "public transit"),
        "questions": (
            "What subject is {speaker} currently researching?",
            "Which research area is {speaker} focused on now?"),
    },
    {
        "name": "travel destination",
        "values": (
            "Lisbon", "Kyoto", "Reykjavik", "Vancouver"),
        "questions": (
            "Where is {speaker} planning to travel?",
            "What destination did {speaker} choose for the next trip?"),
    },
    {
        "name": "weekend hobby",
        "values": (
            "restoring vintage radios", "making pottery",
            "bird photography", "trail running"),
        "questions": (
            "What does {speaker} like doing on weekends?",
            "Which hobby helps {speaker} unwind?"),
    },
    {
        "name": "preferred cuisine",
        "values": (
            "Lebanese food", "Ethiopian cuisine",
            "Korean home cooking", "regional Italian dishes"),
        "questions": (
            "What kind of food does {speaker} prefer?",
            "Which cuisine is {speaker}'s current favorite?"),
    },
    {
        "name": "community project",
        "values": (
            "a neighborhood tool library", "a river cleanup",
            "an accessible playground", "a food-sharing network"),
        "questions": (
            "Which community effort is {speaker} helping with?",
            "What local project matters to {speaker}?"),
    },
    {
        "name": "evening course",
        "values": (
            "documentary filmmaking", "ceramic design",
            "sign language", "architectural history"),
        "questions": (
            "What is {speaker} studying in the evening?",
            "Which course did {speaker} decide to take?"),
    },
    {
        "name": "planned purchase",
        "values": (
            "a folding bicycle", "a compact camera",
            "a field recorder", "a standing desk"),
        "questions": (
            "What is {speaker} planning to buy?",
            "Which purchase is {speaker} saving for?"),
    },
    {
        "name": "favorite local place",
        "values": (
            "the botanical garden", "the old harbor",
            "the riverside library", "the weekend market"),
        "questions": (
            "Which nearby place does {speaker} enjoy most?",
            "What is {speaker}'s favorite local place?"),
    },
)

ADDITIONAL_PREDICATES = (
    {
        "name": "work schedule",
        "values": (
            "early shifts", "a four-day week",
            "split shifts", "flexible hours"),
        "questions": (
            "What work schedule does {speaker} currently follow?",
            "When does {speaker} usually work?"),
    },
    {
        "name": "dietary restriction",
        "values": (
            "a dairy-free diet", "a low-sodium diet",
            "a nut-free diet", "a gluten-free diet"),
        "questions": (
            "Which foods does {speaker} currently avoid?",
            "What dietary restriction does {speaker} follow?"),
    },
    {
        "name": "transportation preference",
        "values": (
            "commuting by train", "walking",
            "taking the bus", "cycling"),
        "questions": (
            "How does {speaker} prefer to get around?",
            "What transportation does {speaker} favor?"),
    },
    {
        "name": "sleep routine",
        "values": (
            "reading before bed", "an early bedtime",
            "a short evening walk", "no screens after ten"),
        "questions": (
            "What is {speaker}'s current sleep routine?",
            "Which bedtime habit does {speaker} follow?"),
    },
    {
        "name": "pet care responsibility",
        "values": (
            "morning dog walks", "feeding the cats",
            "cleaning the aquarium", "training the puppy"),
        "questions": (
            "Which pet-care task does {speaker} handle?",
            "How is {speaker} helping care for a pet?"),
    },
    {
        "name": "recurring appointment",
        "values": (
            "a Monday check-in", "a monthly consultation",
            "a Friday lesson", "a weekly planning call"),
        "questions": (
            "What recurring appointment does {speaker} have?",
            "Which regular meeting is on {speaker}'s calendar?"),
    },
    {
        "name": "software tool",
        "values": (
            "a markdown editor", "a vector drawing app",
            "a task tracker", "a terminal workspace"),
        "questions": (
            "Which software tool is {speaker} using?",
            "What application does {speaker} currently prefer?"),
    },
    {
        "name": "professional goal",
        "values": (
            "leading a design review", "earning a certification",
            "mentoring a new teammate", "publishing a case study"),
        "questions": (
            "What professional goal is {speaker} pursuing?",
            "Which career objective matters to {speaker}?"),
    },
    {
        "name": "gift idea",
        "values": (
            "a handmade journal", "concert tickets",
            "a cooking class", "a framed photograph"),
        "questions": (
            "What gift is {speaker} considering?",
            "Which present does {speaker} plan to give?"),
    },
    {
        "name": "budget priority",
        "values": (
            "building an emergency fund", "home repairs",
            "travel savings", "education expenses"),
        "questions": (
            "What is {speaker}'s current budget priority?",
            "Where does {speaker} plan to focus spending?"),
    },
    {
        "name": "preferred beverage",
        "values": (
            "green tea", "sparkling water",
            "dark-roast coffee", "ginger lemonade"),
        "questions": (
            "What beverage does {speaker} currently prefer?",
            "Which drink is {speaker}'s favorite?"),
    },
    {
        "name": "weather preference",
        "values": (
            "cool rainy days", "dry sunny weather",
            "crisp autumn air", "mild cloudy afternoons"),
        "questions": (
            "What kind of weather does {speaker} prefer?",
            "Which climate conditions does {speaker} enjoy?"),
    },
    {
        "name": "favorite art form",
        "values": (
            "street photography", "watercolor painting",
            "modern dance", "woodblock printing"),
        "questions": (
            "Which art form does {speaker} enjoy most?",
            "What kind of art is {speaker}'s favorite?"),
    },
    {
        "name": "language practice",
        "values": (
            "daily conversation drills", "reading news articles",
            "a weekly language exchange", "writing short stories"),
        "questions": (
            "How is {speaker} practicing a language?",
            "Which language-practice method does {speaker} use?"),
    },
    {
        "name": "gardening plan",
        "values": (
            "growing balcony herbs", "planting native flowers",
            "starting tomato seedlings", "building a rain garden"),
        "questions": (
            "What gardening plan does {speaker} have?",
            "Which garden project is {speaker} starting?"),
    },
    {
        "name": "cooking project",
        "values": (
            "learning sourdough", "making fresh pasta",
            "testing soup recipes", "preserving seasonal fruit"),
        "questions": (
            "What is {speaker}'s current cooking project?",
            "Which recipe project is {speaker} working on?"),
    },
    {
        "name": "family tradition",
        "values": (
            "a Sunday lunch", "an annual camping trip",
            "a holiday baking day", "a monthly game night"),
        "questions": (
            "Which family tradition matters to {speaker}?",
            "What family custom does {speaker} follow?"),
    },
    {
        "name": "study schedule",
        "values": (
            "one hour before breakfast", "weekend study blocks",
            "three evenings a week", "short daily reviews"),
        "questions": (
            "What study schedule does {speaker} follow?",
            "When does {speaker} plan to study?"),
    },
    {
        "name": "wellness habit",
        "values": (
            "morning meditation", "a lunchtime walk",
            "keeping a sleep journal", "stretching after work"),
        "questions": (
            "Which wellness habit is {speaker} practicing?",
            "What healthy routine has {speaker} adopted?"),
    },
    {
        "name": "repair task",
        "values": (
            "fixing a leaky faucet", "repairing a bicycle",
            "mending a winter coat", "restoring a wooden chair"),
        "questions": (
            "What does {speaker} plan to repair?",
            "Which repair task is {speaker} handling?"),
    },
    {
        "name": "event plan",
        "values": (
            "a neighborhood picnic", "a small book fair",
            "an outdoor film night", "a community workshop"),
        "questions": (
            "What event is {speaker} planning?",
            "Which gathering is {speaker} organizing?"),
    },
    {
        "name": "subscription choice",
        "values": (
            "a local newspaper", "an audiobook service",
            "a museum membership", "a produce box"),
        "questions": (
            "Which subscription did {speaker} choose?",
            "What recurring service does {speaker} use?"),
    },
    {
        "name": "workspace preference",
        "values": (
            "a quiet corner desk", "a shared studio",
            "a standing workstation", "a window-side table"),
        "questions": (
            "What workspace does {speaker} prefer?",
            "Where does {speaker} like to work?"),
    },
    {
        "name": "charitable cause",
        "values": (
            "adult literacy", "wildlife rehabilitation",
            "food security", "accessible technology"),
        "questions": (
            "Which charitable cause does {speaker} support?",
            "What cause matters most to {speaker}?"),
    },
)

EXPANDED_PREDICATES = PREDICATES + ADDITIONAL_PREDICATES

LARGE_RELATION_SPECS = (
    (
        "medical appointment",
        ("a dental checkup", "an eye exam",
         "a physical therapy visit", "a routine screening"),
        ("healthcare visit", "doctor appointment",
         "medical visit"),
        ("What medical appointment does {speaker} have?",
         "Which healthcare visit did {speaker} schedule?"),
    ),
    (
        "medication schedule",
        ("once after breakfast", "twice each day",
         "every other evening", "before going to sleep"),
        ("medicine routine", "dose schedule",
         "medication timing"),
        ("What medication schedule does {speaker} follow?",
         "When does {speaker} take the medicine?"),
    ),
    (
        "allergy information",
        ("a pollen allergy", "a shellfish allergy",
         "a latex allergy", "a dust allergy"),
        ("allergy detail", "known allergy",
         "allergic condition"),
        ("What allergy has {speaker} mentioned?",
         "Which allergic condition affects {speaker}?"),
    ),
    (
        "household chore",
        ("washing the dishes", "taking out recycling",
         "vacuuming the rooms", "doing the laundry"),
        ("home duty", "domestic task",
         "household responsibility"),
        ("Which household chore does {speaker} handle?",
         "What home task belongs to {speaker}?"),
    ),
    (
        "childcare plan",
        ("an after-school sitter", "a shared pickup schedule",
         "a weekend playgroup", "a summer day camp"),
        ("care arrangement", "child-minding plan",
         "planned childcare"),
        ("What childcare plan did {speaker} choose?",
         "Which care arrangement works for {speaker}?"),
    ),
    (
        "commute duration",
        ("twenty minutes", "about half an hour",
         "forty-five minutes", "just over an hour"),
        ("travel time to work", "commuting time",
         "length of the commute"),
        ("How long is {speaker}'s commute?",
         "What travel time does {speaker} expect for work?"),
    ),
    (
        "work location",
        ("the downtown office", "a home workspace",
         "the west-side studio", "a shared workshop"),
        ("job location", "place of work",
         "working location"),
        ("Where is {speaker} currently working?",
         "What work location does {speaker} use?"),
    ),
    (
        "project deadline",
        ("the end of March", "next Friday",
         "the first week of June", "mid-October"),
        ("due date", "delivery deadline",
         "project due time"),
        ("When is {speaker}'s project due?",
         "What deadline is {speaker} working toward?"),
    ),
    (
        "meeting frequency",
        ("every Monday", "twice a month",
         "once each quarter", "every other week"),
        ("meeting cadence", "session frequency",
         "how often the group meets"),
        ("How often does {speaker}'s group meet?",
         "What meeting frequency did {speaker} choose?"),
    ),
    (
        "communication preference",
        ("short written updates", "a weekly phone call",
         "in-person conversations", "voice messages"),
        ("preferred communication", "conversation style",
         "way of communicating"),
        ("How does {speaker} prefer to communicate?",
         "What communication style suits {speaker}?"),
    ),
    (
        "notification setting",
        ("important alerts only", "a daily summary",
         "all notifications muted", "real-time updates"),
        ("alert setting", "notification preference",
         "message alert mode"),
        ("What notification setting does {speaker} use?",
         "Which alerts does {speaker} want to receive?"),
    ),
    (
        "device preference",
        ("a compact laptop", "a large tablet",
         "a desktop workstation", "a small e-reader"),
        ("preferred device", "hardware choice",
         "computing device preference"),
        ("Which device does {speaker} prefer?",
         "What hardware is {speaker}'s current choice?"),
    ),
    (
        "internet service",
        ("fiber broadband", "a mobile hotspot",
         "cable internet", "fixed wireless service"),
        ("network service", "internet connection",
         "broadband choice"),
        ("What internet service does {speaker} use?",
         "Which network connection did {speaker} choose?"),
    ),
    (
        "payment method",
        ("a debit card", "a bank transfer",
         "a mobile wallet", "automatic billing"),
        ("way of paying", "payment preference",
         "billing method"),
        ("How does {speaker} prefer to pay?",
         "Which payment method did {speaker} select?"),
    ),
    (
        "savings goal",
        ("a six-month reserve", "a future home deposit",
         "a new computer fund", "a long vacation"),
        ("financial target", "saving objective",
         "money goal"),
        ("What is {speaker} saving for?",
         "Which savings goal does {speaker} have?"),
    ),
    (
        "insurance coverage",
        ("basic health coverage", "comprehensive travel cover",
         "renter protection", "extended device coverage"),
        ("insurance plan", "coverage choice",
         "protection policy"),
        ("What insurance coverage did {speaker} choose?",
         "Which protection plan does {speaker} have?"),
    ),
    (
        "booking preference",
        ("booking well in advance", "choosing flexible dates",
         "using refundable options", "waiting for discounts"),
        ("reservation habit", "booking style",
         "reservation preference"),
        ("How does {speaker} prefer to book?",
         "What reservation habit does {speaker} follow?"),
    ),
    (
        "accommodation preference",
        ("a quiet guesthouse", "a central apartment",
         "a small hotel", "a countryside cabin"),
        ("lodging preference", "preferred accommodation",
         "place to stay"),
        ("Where does {speaker} prefer to stay?",
         "What lodging style does {speaker} choose?"),
    ),
    (
        "seat preference",
        ("an aisle seat", "a window seat",
         "a seat near the front", "a quiet-zone seat"),
        ("preferred seat", "seating choice",
         "place to sit"),
        ("What seat does {speaker} prefer?",
         "Which seating choice suits {speaker}?"),
    ),
    (
        "clothing size",
        ("a small size", "a medium size",
         "a large size", "an extra-large size"),
        ("garment size", "apparel size",
         "size for clothing"),
        ("What clothing size does {speaker} need?",
         "Which garment size fits {speaker}?"),
    ),
    (
        "color preference",
        ("deep blue", "warm orange",
         "forest green", "soft gray"),
        ("favorite color", "preferred shade",
         "color choice"),
        ("Which color does {speaker} prefer?",
         "What shade is {speaker}'s favorite?"),
    ),
    (
        "room temperature",
        ("eighteen degrees", "twenty degrees",
         "twenty-two degrees", "twenty-four degrees"),
        ("indoor temperature", "thermostat preference",
         "preferred room warmth"),
        ("What room temperature does {speaker} prefer?",
         "How warm does {speaker} keep the room?"),
    ),
    (
        "lighting preference",
        ("soft indirect light", "bright task lighting",
         "natural daylight", "warm evening lamps"),
        ("preferred lighting", "light setting",
         "illumination choice"),
        ("What lighting does {speaker} prefer?",
         "Which light setting works for {speaker}?"),
    ),
    (
        "noise preference",
        ("complete silence", "quiet background music",
         "steady ambient sound", "a lively room"),
        ("sound preference", "preferred noise level",
         "acoustic setting"),
        ("What noise level does {speaker} prefer?",
         "Which sound environment suits {speaker}?"),
    ),
    (
        "accessibility need",
        ("step-free access", "large-print materials",
         "live captions", "a quiet waiting area"),
        ("access requirement", "accommodation need",
         "accessibility requirement"),
        ("What accessibility need does {speaker} have?",
         "Which accommodation helps {speaker}?"),
    ),
    (
        "delivery instruction",
        ("leave it by the side door", "call on arrival",
         "use the parcel locker", "hand it to reception"),
        ("drop-off instruction", "delivery preference",
         "courier direction"),
        ("What delivery instruction did {speaker} give?",
         "How should a package be delivered to {speaker}?"),
    ),
    (
        "storage location",
        ("the hall closet", "the basement shelf",
         "the bedroom cabinet", "the garage cupboard"),
        ("place of storage", "where the item is kept",
         "storage spot"),
        ("Where does {speaker} store the item?",
         "What storage location did {speaker} choose?"),
    ),
    (
        "document location",
        ("the blue folder", "an encrypted drive",
         "the office drawer", "a cloud archive"),
        ("file location", "where the document is kept",
         "document storage place"),
        ("Where did {speaker} keep the document?",
         "What file location does {speaker} use?"),
    ),
    (
        "account recovery method",
        ("a recovery email", "a hardware security key",
         "backup codes", "a trusted phone number"),
        ("login recovery", "recovery option",
         "account restoration method"),
        ("How can {speaker} recover the account?",
         "Which recovery option did {speaker} select?"),
    ),
    (
        "backup schedule",
        ("every evening", "each Sunday",
         "twice a month", "at the end of each quarter"),
        ("data backup routine", "backup frequency",
         "archive schedule"),
        ("When does {speaker} back up data?",
         "What backup schedule does {speaker} follow?"),
    ),
    (
        "privacy preference",
        ("sharing only with friends", "keeping the profile private",
         "anonymous usage statistics", "no location history"),
        ("privacy setting", "data-sharing preference",
         "confidentiality choice"),
        ("What privacy preference does {speaker} have?",
         "Which data-sharing setting did {speaker} choose?"),
    ),
    (
        "contact method",
        ("email", "a text message",
         "a phone call", "a calendar invitation"),
        ("preferred contact channel", "way to get in touch",
         "contact preference"),
        ("How should someone contact {speaker}?",
         "Which contact method does {speaker} prefer?"),
    ),
)

LARGE_ADDITIONAL_PREDICATES = tuple(
    {
        "name": name,
        "values": values,
        "questions": questions,
    }
    for name, values, _aliases, questions
    in LARGE_RELATION_SPECS
)

LARGE_ADDITIONAL_PREDICATE_ALIASES = {
    name: aliases
    for name, _values, aliases, _questions
    in LARGE_RELATION_SPECS
}

LARGE_PREDICATES = (
    EXPANDED_PREDICATES
    + LARGE_ADDITIONAL_PREDICATES
)

VALIDATION_PREDICATES = (
    {
        "name": "volunteer role",
        "values": (
            "garden coordinator", "shelter cook",
            "museum guide", "literacy tutor"),
        "questions": (
            "How does {speaker} currently volunteer?",
            "Which volunteer role did {speaker} choose?"),
    },
    {
        "name": "reading preference",
        "values": (
            "historical mysteries", "nature essays",
            "science biographies", "short fiction"),
        "questions": (
            "What does {speaker} currently like to read?",
            "Which kind of books does {speaker} prefer?"),
    },
    {
        "name": "exercise routine",
        "values": (
            "morning swimming", "evening yoga",
            "weekend cycling", "strength training"),
        "questions": (
            "How does {speaker} currently exercise?",
            "Which fitness routine is {speaker} following?"),
    },
    {
        "name": "creative project",
        "values": (
            "a photo essay", "a clay mural",
            "a radio documentary", "a poetry collection"),
        "questions": (
            "What is {speaker} currently creating?",
            "Which creative project is {speaker} working on?"),
    },
    {
        "name": "meeting location",
        "values": (
            "the north café", "the civic hall",
            "the lakeside studio", "the station library"),
        "questions": (
            "Where does {speaker} plan to meet the group?",
            "Which meeting place did {speaker} select?"),
    },
    {
        "name": "learning goal",
        "values": (
            "conversational Finnish", "basic woodworking",
            "digital illustration", "first aid"),
        "questions": (
            "What does {speaker} want to learn?",
            "Which skill is {speaker} currently developing?"),
    },
    {
        "name": "home improvement",
        "values": (
            "a balcony garden", "new bookshelves",
            "better lighting", "a quiet workspace"),
        "questions": (
            "What home change is {speaker} planning?",
            "Which improvement does {speaker} want at home?"),
    },
    {
        "name": "music preference",
        "values": (
            "jazz piano", "folk guitar",
            "ambient music", "classical strings"),
        "questions": (
            "What music does {speaker} currently prefer?",
            "Which musical style does {speaker} enjoy?"),
    },
)

TEST_RELATION_SPECS = (
    (
        "emergency contact",
        ("Morgan Lee", "Sam Rivera",
         "Taylor Quinn", "Jamie Chen"),
        ("urgent contact", "person to call in an emergency",
         "emergency contact person"),
        ("Who is {speaker}'s emergency contact?",
         "Whom should someone call for {speaker} in an emergency?"),
    ),
    (
        "vehicle maintenance date",
        ("the first Saturday in April", "June fifteenth",
         "the last week of September", "early December"),
        ("service date", "car maintenance appointment",
         "vehicle servicing time"),
        ("When is {speaker}'s vehicle maintenance due?",
         "Which date did {speaker} choose for vehicle service?"),
    ),
    (
        "library membership",
        ("a central branch card", "a university guest card",
         "a mobile borrowing card", "a regional digital pass"),
        ("library account", "borrowing membership",
         "library access plan"),
        ("What library membership does {speaker} have?",
         "Which borrowing account does {speaker} use?"),
    ),
    (
        "conference registration",
        ("a standard attendee pass", "a workshop package",
         "a virtual access ticket", "a speaker registration"),
        ("conference pass", "event registration type",
         "conference attendance option"),
        ("How is {speaker} registered for the conference?",
         "Which conference pass did {speaker} select?"),
    ),
    (
        "waste collection day",
        ("Monday morning", "Wednesday evening",
         "Friday afternoon", "Saturday morning"),
        ("rubbish pickup day", "collection schedule",
         "waste pickup time"),
        ("When is waste collected for {speaker}?",
         "Which collection day applies to {speaker}?"),
    ),
    (
        "appliance warranty",
        ("a two-year replacement plan", "a five-year repair plan",
         "manufacturer coverage", "retailer protection"),
        ("warranty plan", "appliance protection",
         "repair coverage"),
        ("What appliance warranty does {speaker} have?",
         "Which protection plan covers {speaker}'s appliance?"),
    ),
    (
        "preferred news source",
        ("public radio bulletins", "a local newspaper",
         "an international news app", "a weekly current-affairs magazine"),
        ("news preference", "preferred news outlet",
         "usual information source"),
        ("Where does {speaker} prefer to get the news?",
         "Which news source does {speaker} usually follow?"),
    ),
    (
        "parking permit",
        ("a zone C resident permit", "a workplace garage pass",
         "a visitor parking card", "an overnight street permit"),
        ("parking authorization", "vehicle parking pass",
         "permit for parking"),
        ("What parking permit does {speaker} use?",
         "Which parking authorization belongs to {speaker}?"),
    ),
)

TEST_PREDICATES = tuple(
    {
        "name": name,
        "values": values,
        "questions": questions,
    }
    for name, values, _aliases, questions
    in TEST_RELATION_SPECS
)

TEST_PREDICATE_ALIASES = {
    name: aliases
    for name, _values, aliases, _questions
    in TEST_RELATION_SPECS
}

PREDICATE_ALIASES = {
    "research topic": (
        "research subject", "study focus",
        "area being researched"),
    "travel destination": (
        "trip destination", "place to visit",
        "next travel location"),
    "weekend hobby": (
        "weekend pastime", "free-time activity",
        "leisure pursuit"),
    "preferred cuisine": (
        "favorite food style", "cuisine preference",
        "preferred food"),
    "community project": (
        "local initiative", "neighborhood effort",
        "community effort"),
    "evening course": (
        "night class", "after-work course",
        "evening class"),
    "planned purchase": (
        "item to buy", "intended purchase",
        "next purchase"),
    "favorite local place": (
        "preferred nearby spot", "favorite neighborhood place",
        "local place enjoyed most"),
}

ADDITIONAL_PREDICATE_ALIASES = {
    "work schedule": (
        "working hours", "shift pattern",
        "weekly work timetable"),
    "dietary restriction": (
        "food limitation", "diet requirement",
        "eating restriction"),
    "transportation preference": (
        "preferred transport", "travel mode",
        "way of getting around"),
    "sleep routine": (
        "bedtime routine", "sleeping habit",
        "nightly wind-down"),
    "pet care responsibility": (
        "animal care duty", "pet-care task",
        "responsibility for the pet"),
    "recurring appointment": (
        "regular appointment", "repeating calendar event",
        "scheduled recurring meeting"),
    "software tool": (
        "preferred application", "digital tool",
        "software being used"),
    "professional goal": (
        "career objective", "work ambition",
        "professional target"),
    "gift idea": (
        "present idea", "planned gift",
        "gift being considered"),
    "budget priority": (
        "spending priority", "financial focus",
        "main budget goal"),
    "preferred beverage": (
        "favorite drink", "drink preference",
        "beverage choice"),
    "weather preference": (
        "favorite weather", "preferred conditions",
        "climate preference"),
    "favorite art form": (
        "preferred kind of art", "artistic preference",
        "art form enjoyed most"),
    "language practice": (
        "language-learning routine", "speaking practice",
        "language study method"),
    "gardening plan": (
        "garden project", "planting plan",
        "horticulture project"),
    "cooking project": (
        "culinary project", "recipe project",
        "food-making plan"),
    "family tradition": (
        "family custom", "household tradition",
        "recurring family ritual"),
    "study schedule": (
        "learning timetable", "study routine",
        "planned study time"),
    "wellness habit": (
        "healthy routine", "well-being practice",
        "self-care habit"),
    "repair task": (
        "fixing job", "item to repair",
        "maintenance task"),
    "event plan": (
        "planned gathering", "event being organized",
        "upcoming community event"),
    "subscription choice": (
        "recurring service", "membership choice",
        "selected subscription"),
    "workspace preference": (
        "preferred work area", "work setting choice",
        "favorite place to work"),
    "charitable cause": (
        "supported cause", "charity focus",
        "social cause"),
}

EXPANDED_PREDICATE_ALIASES = {
    **PREDICATE_ALIASES,
    **ADDITIONAL_PREDICATE_ALIASES,
}

LARGE_PREDICATE_ALIASES = {
    **EXPANDED_PREDICATE_ALIASES,
    **LARGE_ADDITIONAL_PREDICATE_ALIASES,
}

VALIDATION_PREDICATE_ALIASES = {
    "volunteer role": (
        "volunteering position", "volunteer duty",
        "community service role"),
    "reading preference": (
        "preferred reading", "favorite book type",
        "reading taste"),
    "exercise routine": (
        "fitness routine", "regular workout",
        "exercise habit"),
    "creative project": (
        "current creative work", "artistic project",
        "creation in progress"),
    "meeting location": (
        "meeting place", "group rendezvous",
        "selected venue"),
    "learning goal": (
        "skill to learn", "study objective",
        "learning target"),
    "home improvement": (
        "planned home upgrade", "household change",
        "home project"),
    "music preference": (
        "musical taste", "preferred music",
        "favorite music style"),
}

TRAIN_PREDICATE_ALIASES = {
    predicate: aliases[:2]
    for predicate, aliases in PREDICATE_ALIASES.items()
}

HELDOUT_PREDICATE_ALIASES = {
    predicate: aliases[2:]
    for predicate, aliases in PREDICATE_ALIASES.items()
}

EXPANDED_TRAIN_PREDICATE_ALIASES = {
    predicate: aliases[:2]
    for predicate, aliases
    in EXPANDED_PREDICATE_ALIASES.items()
}

EXPANDED_HELDOUT_PREDICATE_ALIASES = {
    predicate: aliases[2:]
    for predicate, aliases
    in EXPANDED_PREDICATE_ALIASES.items()
}

LARGE_TRAIN_PREDICATE_ALIASES = {
    predicate: aliases[:2]
    for predicate, aliases
    in LARGE_PREDICATE_ALIASES.items()
}

LARGE_HELDOUT_PREDICATE_ALIASES = {
    predicate: aliases[2:]
    for predicate, aliases
    in LARGE_PREDICATE_ALIASES.items()
}

DETAILS = (
    "with close friends", "after work", "near home",
    "with the local group", "during the summer",
    "on quiet mornings", "for the next few months",
)

STATEMENTS = (
    "{speaker}: Lately my {predicate} has been {value}.",
    "{speaker}: I wanted to mention that my {predicate} is {value}.",
    "{speaker}: For now, record {value} as my {predicate}.",
    "{speaker}: I have settled on {value} for my {predicate}.",
)

UPDATES = (
    "{speaker}: I changed my {predicate} from {old} to {value}.",
    "{speaker}: An update: my {predicate} is now {value}, not {old}.",
    "{speaker}: Please replace {old} with {value} as my {predicate}.",
)

VALIDATION_STATEMENTS = (
    "Regarding {predicate}, {speaker} settled on {value}.",
    "The choice for {speaker}'s {predicate} is currently {value}.",
    "{value} is what {speaker} selected for the {predicate}.",
    "A note about {speaker}: the {predicate} is {value}.",
)

VALIDATION_UPDATES = (
    "Replacing {old}, {speaker} now uses {value} for the {predicate}.",
    "For {speaker}, the {predicate} changed from {old} into {value}.",
    "The new {predicate} for {speaker} is {value}; previously it was {old}.",
    "{old} is outdated for {speaker}'s {predicate}; use {value}.",
)

TEST_STATEMENTS = (
    "For {speaker}, the recorded {predicate} currently points to {value}.",
    "{speaker}'s latest entry under {predicate} is {value}.",
    "The record for {speaker} lists {value} as the {predicate}.",
    "Under {predicate}, {speaker} currently has {value}.",
)

TEST_UPDATES = (
    "Revise {speaker}'s {predicate}: use {value} in place of {old}.",
    "{speaker}'s {predicate} moved from {old} to {value}.",
    "For {speaker}, update the {predicate} to {value}; remove {old}.",
    "The previous {predicate} for {speaker} was {old}; it is now {value}.",
)

TRAIN_STATEMENTS = STATEMENTS + (
    "{speaker} has chosen {value} when it comes to the {predicate}.",
    "For the {predicate}, {speaker}'s current choice is {value}.",
    "{predicate} note from {speaker}: the selected option is {value}.",
    "The value {value} belongs to {speaker}'s {predicate}.",
    "As for {speaker}, use {value} for the {predicate}.",
    "Current {predicate}: {value}, according to {speaker}.",
)

TRAIN_UPDATES = UPDATES + (
    "{predicate} update from {speaker}: replace {old} with {value}.",
    "For the {predicate}, {speaker} moved away from {old} and chose {value}.",
    "{value} is now {speaker}'s {predicate}; {old} was the prior choice.",
    "Change recorded for {speaker}: {predicate} is {value} instead of {old}.",
    "The former {predicate}, {old}, no longer applies to {speaker}; use {value}.",
    "{speaker}'s latest {predicate} became {value} after previously being {old}.",
)

ROLE_TRAIN_STATEMENTS = TRAIN_STATEMENTS + (
    "In the {predicate} record for {speaker}, the value is {value}.",
    "The entry belonging to {speaker} under {predicate} now reads {value}.",
    "Looking at {speaker}'s {predicate}, the recorded choice is {value}.",
    "For {speaker}, the field named {predicate} contains {value}.",
    "In {speaker}'s record, {value} appears beneath {predicate}.",
    "The latest item in {speaker}'s {predicate} field is {value}.",
    "Under the heading {predicate}, {speaker} has {value}.",
    "{speaker}'s record shows {predicate}: {value}.",
    "The record for {speaker} associates {predicate} with {value}.",
    "At {predicate}, the current entry for {speaker} is {value}.",
    "The stored {predicate} item for {speaker} is {value}.",
    "Read {speaker}'s {predicate} field as {value}.",
)

ROLE_TRAIN_UPDATES = TRAIN_UPDATES + (
    "After retiring {old}, {speaker} recorded {value} for {predicate}.",
    "The old {predicate} for {speaker} was {old}; the replacement is {value}.",
    "For {speaker}, remove {old} and set {predicate} to {value}.",
    "Revise the {predicate} belonging to {speaker}: choose {value} instead of {old}.",
    "{speaker}'s {predicate} transitioned away from {old} toward {value}.",
    "Earlier, {speaker}'s {predicate} was {old}. The current entry is {value}.",
    "For {speaker}, {old} used to fill {predicate}; now {value} does.",
    "The {predicate} entry for {speaker} no longer contains {old}; it contains {value}.",
    "Set {speaker}'s {predicate} to {value}, superseding {old}.",
    "As {speaker}'s {predicate}, {value} replaces {old}.",
    "Move {speaker}'s {predicate} from {old} over to {value}.",
    "The outdated value {old} under {speaker}'s {predicate} changed to {value}.",
    "In the {predicate} field for {speaker}, swap {old} out for {value}.",
    "What was {old} for {speaker}'s {predicate} should now be {value}.",
    "For {predicate}, {speaker} previously had {old} but currently has {value}.",
    "Current {predicate} for {speaker}: {value}. Previous value: {old}.",
    "{old} was formerly listed for {speaker} at {predicate}; record {value} now.",
)

ROLE_VALIDATION_STATEMENTS = (
    "Within {speaker}'s {predicate} entry, {value} is the current value.",
    "The {predicate} field associated with {speaker} contains {value}.",
    "For {speaker}, the item stored under {predicate} is {value}.",
)

ROLE_VALIDATION_UPDATES = (
    "With {old} retired, {speaker} now records {value} as {predicate}.",
    "The earlier {predicate} for {speaker} read {old}; its successor is {value}.",
    "In {speaker}'s record, clear {old} and write {value} under {predicate}.",
    "Amend {speaker}'s {predicate} by replacing {old} with {value}.",
)

ROLE_CHALLENGE_STATEMENTS = (
    "Consulting {speaker}'s record, the value beside {predicate} is {value}.",
    "The current listing for {predicate} in {speaker}'s profile says {value}.",
    "{value} appears as the active {predicate} selection for {speaker}.",
    "On {speaker}'s file, {predicate} is represented by {value}.",
)

ROLE_CHALLENGE_UPDATES = (
    "Although {old} once occupied {speaker}'s {predicate} field, {value} occupies it now.",
    "Replace the former {old} entry at {speaker}'s {predicate} with {value}.",
    "The active value for {speaker}'s {predicate} is {value}, superseding {old}.",
    "Under {predicate}, {speaker} has moved on from {old}; retain {value}.",
)


def answer_spans(evidence, episodes, gold_indices, surfaces):
    starts = []
    cursor = 0
    for episode in episodes:
        starts.append(cursor)
        cursor += len(episode) + 1
    spans = []
    for episode_index, surface in zip(gold_indices, surfaces):
        local = episodes[episode_index].index(surface)
        start = starts[episode_index] + local
        spans.append({
            "start": start,
            "end": start + len(surface),
            "text": surface,
        })
    return spans


def dated_episode(world, event, date, statement):
    return (
        f"[source R{world:04d}E{event:03d} | "
        f"{date.isoformat()}] {statement}")


def choose_value(
    world, event, predicate_index,
    predicates=PREDICATES,
):
    predicate = predicates[predicate_index]
    base = predicate["values"][
        (world * 5 + event * 3 + predicate_index)
        % len(predicate["values"])]
    detail = DETAILS[
        (world * 7 + event * 5 + predicate_index)
        % len(DETAILS)]
    return f"{base} {detail}"


def build_episode_bank(
    world, events, rng,
    speakers_catalog=SPEAKERS,
    predicates=PREDICATES,
    predicate_aliases=None,
    include_canonical_predicate_surface=True,
    statement_templates=STATEMENTS,
    update_templates=UPDATES,
):
    speaker_count = 2 + world % 3
    offset = (world * 3) % len(speakers_catalog)
    speakers = [
        speakers_catalog[
            (offset + index) % len(speakers_catalog)]
        for index in range(speaker_count)
    ]
    base_date = dt.date(2021 + world % 4, 1 + world % 12, 1)
    episodes = []
    facts = []
    history = {}
    for event in range(events):
        speaker = speakers[event % speaker_count]
        predicate_index = (
            event // speaker_count
            + event * 3 + world
        ) % len(predicates)
        predicate = predicates[predicate_index]["name"]
        aliases = (
            predicate_aliases.get(predicate, ())
            if predicate_aliases else ())
        surfaces = tuple(aliases)
        if include_canonical_predicate_surface:
            surfaces = (predicate,) + surfaces
        if not surfaces:
            raise ValueError(
                f"predicate has no surface form: {predicate}")
        predicate_surface = rng.choice(
            surfaces)
        value = choose_value(
            world, event, predicate_index,
            predicates=predicates)
        key = (speaker, predicate_index)
        previous = history.get(key, [])
        if previous:
            statement = rng.choice(update_templates).format(
                speaker=speaker, predicate=predicate_surface,
                old=previous[-1]["value"], value=value)
        else:
            statement = rng.choice(statement_templates).format(
                speaker=speaker, predicate=predicate_surface,
                value=value)
        date = base_date + dt.timedelta(
            days=event * 3 + event // 7)
        episodes.append(
            dated_episode(world, event, date, statement))
        fact = {
            "event": event,
            "speaker": speaker,
            "predicate_index": predicate_index,
            "predicate": predicate,
            "predicate_surface": predicate_surface,
            "value": value,
            "date": date.isoformat(),
        }
        facts.append(fact)
        history.setdefault(key, []).append(fact)
    for values in history.values():
        for version, fact in enumerate(values):
            fact["version"] = version
            fact["operation"] = (
                "create" if version == 0 else "update")
            fact["previous_event"] = (
                values[version - 1]["event"]
                if version > 0 else None)
            fact["active"] = version == len(values) - 1
    return speakers, episodes, facts, history


def make_row(
    world, query_index, episodes, facts, question, answer,
    gold_indices, surfaces, family, null_target=False,
    query_plan=None,
):
    evidence = "\n".join(episodes)
    return {
        "sample_id": (
            f"repeated-dialogue-{world:04d}-{query_index:03d}"),
        "question": question,
        "evidence": evidence,
        "answer_type": (
            "list" if len(gold_indices) > 1 else "span"),
        "answer": answer,
        "answer_spans": answer_spans(
            evidence, episodes, gold_indices, surfaces),
        "metadata": {
            "source": "RepeatedDialogue",
            "family": family,
            "world_id": f"repeated-dialogue-{world:04d}",
            "source_count": len(episodes),
            "counterfactual_no_info": bool(null_target),
            "session_consistent": True,
            "repeated_entities": True,
            "typed_events": [
                {
                    "episode": fact["event"],
                    "entity": fact["speaker"],
                    "entity_surface": fact["speaker"],
                    "predicate": fact["predicate"],
                    "predicate_surface":
                        fact.get(
                            "predicate_surface",
                            fact["predicate"]),
                    "value": fact["value"],
                    "time": fact["date"],
                    "operation": fact["operation"],
                    "version": fact["version"],
                    "previous_episode":
                        fact["previous_event"],
                    "active": fact["active"],
                }
                for fact in facts
            ],
            "query_plan": query_plan or {
                "intent": "null",
                "targets": [],
            },
            "locomo_used": False,
        },
    }


def build_world(
    world, events, queries, rng,
    speakers_catalog=SPEAKERS,
    predicates=PREDICATES,
    predicate_aliases=None,
    include_canonical_predicate_surface=True,
    statement_templates=STATEMENTS,
    update_templates=UPDATES,
):
    speakers, episodes, facts, history = build_episode_bank(
        world, events, rng,
        speakers_catalog=speakers_catalog,
        predicates=predicates,
        predicate_aliases=predicate_aliases,
        include_canonical_predicate_surface=(
            include_canonical_predicate_surface),
        statement_templates=statement_templates,
        update_templates=update_templates)
    latest = [values[-1] for values in history.values()]
    repeated = [
        values for values in history.values()
        if len(values) >= 2]
    if len(latest) < 2 or not repeated:
        raise ValueError("dialogue bank lacks repeated facts")
    rows = []
    for query_index in range(queries):
        family = (
            "direct", "paraphrase", "temporal",
            "previous", "multi", "null",
        )[query_index % 6]
        selector = (
            world * 17 + query_index * 11) % len(latest)
        fact = latest[selector]
        predicate = predicates[fact["predicate_index"]]
        if family == "direct":
            question = (
                f"What is {fact['speaker']}'s current "
                f"{fact['predicate']}?")
            rows.append(make_row(
                world, query_index, episodes, facts, question,
                fact["value"], [fact["event"]],
                [fact["value"]], family,
                query_plan={
                    "intent": "current",
                    "targets": [{
                        "entity": fact["speaker"],
                        "predicate": fact["predicate"],
                        "version": "active",
                    }],
                }))
        elif family == "paraphrase":
            question = rng.choice(
                predicate["questions"]).format(
                    speaker=fact["speaker"])
            rows.append(make_row(
                world, query_index, episodes, facts, question,
                fact["value"], [fact["event"]],
                [fact["value"]], family,
                query_plan={
                    "intent": "current",
                    "targets": [{
                        "entity": fact["speaker"],
                        "predicate": fact["predicate"],
                        "version": "active",
                    }],
                }))
        elif family == "temporal":
            question = (
                f"When did {fact['speaker']} say their "
                f"{fact['predicate']} was {fact['value']}?")
            rows.append(make_row(
                world, query_index, episodes, facts, question,
                fact["date"], [fact["event"]],
                [fact["date"]], family,
                query_plan={
                    "intent": "exact_time",
                    "targets": [{
                        "entity": fact["speaker"],
                        "predicate": fact["predicate"],
                        "version": "exact",
                        "value": fact["value"],
                        "time": fact["date"],
                    }],
                }))
        elif family == "previous":
            values = repeated[
                (world + query_index) % len(repeated)]
            old, new = values[-2], values[-1]
            question = (
                f"What was {old['speaker']}'s "
                f"{old['predicate']} before {new['value']}?")
            rows.append(make_row(
                world, query_index, episodes, facts, question,
                old["value"], [old["event"]],
                [old["value"]], family,
                query_plan={
                    "intent": "previous",
                    "targets": [{
                        "entity": old["speaker"],
                        "predicate": old["predicate"],
                        "version": "previous",
                        "anchor_episode": new["event"],
                    }],
                }))
        elif family == "multi":
            second = latest[(selector + 1) % len(latest)]
            if second["event"] == fact["event"]:
                second = latest[(selector + 2) % len(latest)]
            question = (
                f"What are the current {fact['predicate']} for "
                f"{fact['speaker']} and {second['predicate']} for "
                f"{second['speaker']}?")
            rows.append(make_row(
                world, query_index, episodes, facts, question,
                [fact["value"], second["value"]],
                [fact["event"], second["event"]],
                [fact["value"], second["value"]], family,
                query_plan={
                    "intent": "multi_current",
                    "targets": [
                        {
                            "entity": item["speaker"],
                            "predicate": item["predicate"],
                            "version": "active",
                        }
                        for item in (fact, second)
                    ],
                }))
        else:
            speaker = speakers[
                (world + query_index) % len(speakers)]
            question = (
                f"What is {speaker}'s emergency contact preference?")
            rows.append(make_row(
                world, query_index, episodes, facts, question,
                "No information available", [], [],
                family, null_target=True,
                query_plan={
                    "intent": "null",
                    "entity": speaker,
                    "predicate": (
                        "emergency contact preference"),
                    "targets": [],
                }))
    return rows


def write_split(
    path, worlds, events, queries, seed,
    world_offset=0,
    speakers_catalog=SPEAKERS,
    predicates=PREDICATES,
    predicate_aliases=None,
    include_canonical_predicate_surface=True,
    statement_templates=STATEMENTS,
    update_templates=UPDATES,
    layout_views=1,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for local_world in range(worlds):
            world = world_offset + local_world
            semantic_world_id = (
                f"repeated-dialogue-{world:04d}")
            for layout_view in range(layout_views):
                rng = random.Random(
                    seed
                    + local_world * 10_007
                    + layout_view * 1_000_003)
                for row in build_world(
                    world, events, queries, rng,
                    speakers_catalog=speakers_catalog,
                    predicates=predicates,
                    predicate_aliases=predicate_aliases,
                    include_canonical_predicate_surface=(
                        include_canonical_predicate_surface),
                    statement_templates=statement_templates,
                    update_templates=update_templates,
                ):
                    metadata = row["metadata"]
                    metadata["semantic_world_id"] = (
                        semantic_world_id)
                    metadata["layout_view"] = layout_view
                    if layout_views > 1:
                        metadata["world_id"] = (
                            f"{semantic_world_id}"
                            f"-view{layout_view}")
                        row["sample_id"] = (
                            f"{row['sample_id']}"
                            f"-view{layout_view}")
                    handle.write(json.dumps(
                        row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--train-worlds", type=int, default=32)
    parser.add_argument("--valid-worlds", type=int, default=8)
    parser.add_argument("--events", type=int, default=128)
    parser.add_argument("--queries-per-world", type=int, default=12)
    parser.add_argument(
        "--train-layout-views",
        type=int, default=1)
    parser.add_argument(
        "--semantic-reference-variation",
        action="store_true")
    parser.add_argument(
        "--factorized-semantic-validation",
        action="store_true")
    parser.add_argument(
        "--expanded-relation-curriculum",
        action="store_true")
    parser.add_argument(
        "--large-relation-curriculum",
        action="store_true")
    parser.add_argument(
        "--role-layout-curriculum",
        action="store_true")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if min(args.train_worlds, args.valid_worlds) < 1:
        parser.error("world counts must be positive")
    if args.events < 64:
        parser.error(
            "--events must be at least 64 so every speaker has updates")
    if args.queries_per_world < 6:
        parser.error("--queries-per-world must be at least 6")
    if args.train_layout_views < 1:
        parser.error(
            "--train-layout-views must be positive")
    if (
        args.factorized_semantic_validation
        and not args.semantic_reference_variation
    ):
        parser.error(
            "--factorized-semantic-validation requires "
            "--semantic-reference-variation")
    if (
        args.expanded_relation_curriculum
        and not args.factorized_semantic_validation
    ):
        parser.error(
            "--expanded-relation-curriculum requires "
            "--factorized-semantic-validation")
    if (
        args.large_relation_curriculum
        and not args.expanded_relation_curriculum
    ):
        parser.error(
            "--large-relation-curriculum requires "
            "--expanded-relation-curriculum")
    if (
        args.expanded_relation_curriculum
        and args.events < 256
    ):
        parser.error(
            "--expanded-relation-curriculum requires "
            "--events at least 256")
    if (
        args.large_relation_curriculum
        and args.events < 512
    ):
        parser.error(
            "--large-relation-curriculum requires "
            "--events at least 512")
    output = Path(args.output)
    if args.large_relation_curriculum:
        train_predicates = LARGE_PREDICATES
    elif args.expanded_relation_curriculum:
        train_predicates = EXPANDED_PREDICATES
    else:
        train_predicates = PREDICATES
    train_aliases = None
    if args.semantic_reference_variation:
        if args.large_relation_curriculum:
            train_aliases = (
                LARGE_TRAIN_PREDICATE_ALIASES)
        elif args.expanded_relation_curriculum:
            train_aliases = (
                EXPANDED_TRAIN_PREDICATE_ALIASES)
        elif args.factorized_semantic_validation:
            train_aliases = TRAIN_PREDICATE_ALIASES
        else:
            train_aliases = PREDICATE_ALIASES
    train_statement_templates = (
        ROLE_TRAIN_STATEMENTS
        if args.role_layout_curriculum
        else TRAIN_STATEMENTS
    )
    train_update_templates = (
        ROLE_TRAIN_UPDATES
        if args.role_layout_curriculum
        else TRAIN_UPDATES
    )
    valid_statement_templates = (
        ROLE_VALIDATION_STATEMENTS
        if args.role_layout_curriculum
        else VALIDATION_STATEMENTS
    )
    valid_update_templates = (
        ROLE_VALIDATION_UPDATES
        if args.role_layout_curriculum
        else VALIDATION_UPDATES
    )
    write_split(
        output / "train.jsonl", args.train_worlds,
        args.events, args.queries_per_world,
        args.seed + 101,
        predicates=train_predicates,
        predicate_aliases=train_aliases,
        statement_templates=train_statement_templates,
        update_templates=train_update_templates,
        layout_views=args.train_layout_views)
    write_split(
        output / "valid.jsonl", args.valid_worlds,
        args.events, args.queries_per_world,
        args.seed + 307,
        world_offset=1_000_000,
        speakers_catalog=VALIDATION_SPEAKERS,
        predicates=VALIDATION_PREDICATES,
        predicate_aliases=(
            VALIDATION_PREDICATE_ALIASES
            if args.semantic_reference_variation else None),
        statement_templates=valid_statement_templates,
        update_templates=valid_update_templates)
    write_split(
        output / "test.jsonl", args.valid_worlds,
        args.events, args.queries_per_world,
        args.seed + 709,
        world_offset=3_000_000,
        speakers_catalog=TEST_SPEAKERS,
        predicates=TEST_PREDICATES,
        predicate_aliases=(
            TEST_PREDICATE_ALIASES
            if args.semantic_reference_variation else None),
        statement_templates=TEST_STATEMENTS,
        update_templates=TEST_UPDATES)
    challenge_output = None
    if args.role_layout_curriculum:
        challenge_output = output / "challenge.jsonl"
        write_split(
            challenge_output,
            args.valid_worlds,
            args.events, args.queries_per_world,
            args.seed + 907,
            world_offset=4_000_000,
            speakers_catalog=TEST_SPEAKERS,
            predicates=TEST_PREDICATES,
            predicate_aliases=(
                TEST_PREDICATE_ALIASES
                if args.semantic_reference_variation else None),
            statement_templates=ROLE_CHALLENGE_STATEMENTS,
            update_templates=ROLE_CHALLENGE_UPDATES)
    if args.factorized_semantic_validation:
        if args.large_relation_curriculum:
            heldout_aliases = (
                LARGE_HELDOUT_PREDICATE_ALIASES)
        elif args.expanded_relation_curriculum:
            heldout_aliases = (
                EXPANDED_HELDOUT_PREDICATE_ALIASES)
        else:
            heldout_aliases = (
                HELDOUT_PREDICATE_ALIASES)
        write_split(
            output / "valid_alias.jsonl",
            args.valid_worlds,
            args.events, args.queries_per_world,
            args.seed + 503,
            world_offset=2_000_000,
            speakers_catalog=VALIDATION_SPEAKERS,
            predicates=train_predicates,
            predicate_aliases=heldout_aliases,
            include_canonical_predicate_surface=False,
            statement_templates=valid_statement_templates,
            update_templates=valid_update_templates)
    print(json.dumps({
        "output": str(output),
        "train_worlds": args.train_worlds,
        "valid_worlds": args.valid_worlds,
        "events": args.events,
        "queries_per_world": args.queries_per_world,
        "train_layout_views":
            args.train_layout_views,
        "validation_world_offset": 1_000_000,
        "test_world_offset": 3_000_000,
        "validation_entity_predicate_ood": True,
        "validation_layout_ood": True,
        "training_layout_augmented": True,
        "semantic_reference_variation":
            args.semantic_reference_variation,
        "factorized_semantic_validation":
            args.factorized_semantic_validation,
        "expanded_relation_curriculum":
            args.expanded_relation_curriculum,
        "large_relation_curriculum":
            args.large_relation_curriculum,
        "role_layout_curriculum":
            args.role_layout_curriculum,
        "training_predicate_count":
            len(train_predicates),
        "test_predicate_count":
            len(TEST_PREDICATES),
        "relation_test_output":
            str(output / "test.jsonl"),
        "role_challenge_output": (
            str(challenge_output)
            if challenge_output is not None
            else None
        ),
        "alias_validation_output": (
            str(output / "valid_alias.jsonl")
            if args.factorized_semantic_validation
            else None),
        "repeated_entities": True,
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
