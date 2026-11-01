"""Augment frozen authored fit observations without consulting evaluation text.

The source manifest and curriculum are never modified. Only ``fit`` observations
are used to generate new statements and query contexts. Every non-fit split is
copied unchanged and checked by digest. No model, benchmark or external text is
read. Labels remain original-source byte addresses, never runtime decision rules.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

# Independently authored discourse forms. A colon permits the original spoken
# first-person/elliptical sentence to remain byte-identical inside its new frame.
FRAMES = (
    (
        "Actually, here's how I handle it at home: ",
        " It saves me from repeating the same work later.",
    ),
    (
        "For what it's worth, this is part of my normal routine: ",
        " That way I can keep things consistent from one week to the next.",
    ),
    (
        "As it happens, I have a practical reason for this: ",
        " The arrangement makes the whole task simpler for me.",
    ),
    (
        "There's this thing I've been doing lately: ",
        " I stick with it because it fits around the rest of my day.",
    ),
    (
        "Turns out there's an easy way for me to explain it: ",
        " I chose that approach so I don't have to rush the task.",
    ),
    (
        "To tell you the truth, here's my usual approach: ",
        " It leaves me more time to check the small details.",
    ),
)

OPENINGS = (
    "Hello {other}, I have a brief update. ",
    "Nice to hear from you, {other}. ",
    "Hi there, {other}, glad we could talk. ",
    "Before you head off, {other}, let me tell you this. ",
    "Oh, {other}, I nearly missed sharing my news. ",
    "{other}, thanks for checking in today. ",
)

CLOSINGS = (
    " That's my update done. How has your afternoon been?",
    " Thanks for taking a moment to listen.",
    " I'll leave it there for now. What have you been up to?",
    " Lovely chatting again; hope the rest of your day goes well.",
    " Anyway, tell me what is new on your side.",
    " That covers my news. Shall we talk about something else?",
)

STANDALONE = (
    "Oh, {other}, thanks for making a little time to chat.",
    "Hi {other}, glad to catch you before you go.",
    "It was good talking to you, {other}; speak soon.",
    "Could you help me rephrase a sentence?",
    "Would you explain the difference between two expressions?",
    "Can we switch to another topic for a moment?",
)

CONTEXT_FRAMES = (
    (
        "user: Earlier I mentioned {target} and {other} together.\n"
        "assistant: We've looked at {other}; let's talk only about {target} now.\n"
    ),
    (
        "user: The notes include {other} as well as {target}.\n"
        "assistant: For the next question, focus specifically on {target}.\n"
    ),
)

AMBIGUOUS_CONTEXTS = (
    (
        "user: {target} and {other} both have notes about this.\n"
        "assistant: Either person could be next; you haven't chosen one.\n"
    ),
    (
        "user: The discussion covered both {other} and {target}.\n"
        "assistant: No single person has been selected for the next question.\n"
    ),
)


def _digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _context(row, past):
    return (
        f"Speaker: {row['speaker'] or 'User'}\n"
        f"Observed at: {row['timestamp'] or 'unspecified'}\n{past}"
    )


def _range(text, value, last=False):
    raw, name = text.encode(), value.encode()
    start = raw.rfind(name) if last else raw.find(name)
    if start < 0:
        raise ValueError("reference source is missing")
    return dict(document="context", start=start, end=start + len(name))


def augment_fit(fit_rows, seed=13131):
    """Return additions built exclusively from fit facts/questions/provenance."""
    rng = random.Random(seed)
    facts = {}
    for row in fit_rows:
        if row["tag"] != "episode-fact":
            continue
        if len(row["spans"]) != 1:
            raise ValueError("augmentation needs one complete source fact per row")
        source = row["spans"][0]
        fact = row["body"].encode()[source["start"] : source["end"]].decode()
        key = (row["speaker"], fact)
        if key not in facts or len(row["body"]) < len(facts[key][0]["body"]):
            facts[key] = (row, fact)
    names = sorted({row["speaker"] for row, _ in facts.values()})
    if len(names) < 2:
        raise ValueError("multiple independent fit speakers are required")
    additions = []

    def append(row, kind):
        row["source_id"] = f"colloquial-fit-augmentation-{len(additions)}"
        row["augmentation"] = kind
        additions.append(row)

    for row, fact in facts.values():
        speaker = row["speaker"]
        others = [name for name in names if name != speaker]
        for index, (preamble, causal_tail) in enumerate(FRAMES):
            other = rng.choice(others)
            opening = rng.choice(OPENINGS).format(other=other)
            closing = rng.choice(CLOSINGS)
            complete_fact = preamble + fact + causal_tail
            new = copy.deepcopy(row)
            new["body"] = opening + complete_fact + closing
            source = new["spans"][0]
            source.update(
                start=len(opening.encode()),
                end=len((opening + complete_fact).encode()),
                need=True,
                references=[speaker],
                ref_ranges=[_range(new["context"], speaker)],
            )
            new.update(
                whole_need=True,
                whole_refs=[speaker],
                whole_ref_ranges=[_range(new["context"], speaker)],
            )
            append(new, "complete_discourse_fact")
            if index % 2:
                quoted = copy.deepcopy(new)
                quoted.update(
                    body=(
                        "Please translate the following quoted passage; it belongs "
                        'to a fictional narrator, not to me: "' + complete_fact + '"'
                    ),
                    spans=[],
                    whole_need=False,
                    whole_refs=[],
                    whole_ref_ranges=[],
                    tag="negative-quotation",
                )
                append(quoted, "quoted_discourse_negative")
        for forms in (STANDALONE[:3], STANDALONE[3:]):
            new = copy.deepcopy(row)
            new.update(
                body=rng.choice(forms).format(other=rng.choice(others)),
                spans=[],
                whole_need=False,
                whole_refs=[],
                whole_ref_ranges=[],
                tag="ordinary-negative",
            )
            append(new, "standalone_conversation_negative")

    # Questions preserve fit bodies. Context determines the causal target, not
    # source order, the question author, or a copied evaluation utterance.
    for row in fit_rows:
        if row["tag"] != "episode-query":
            continue
        category = row.get("query_class")
        if category not in ("context", "ambiguous"):
            continue
        if category == "context":
            if len(row["whole_refs"]) != 1:
                raise ValueError("context fit question lacks one explicit target")
            target = row["whole_refs"][0]
            for template in CONTEXT_FRAMES:
                other = rng.choice([n for n in names if n != target])
                new = copy.deepcopy(row)
                new["past"] = template.format(target=target, other=other)
                new["context"] = _context(new, new["past"])
                new.update(
                    whole_need=True,
                    whole_refs=[target],
                    whole_ref_ranges=[_range(new["context"], target, last=True)],
                )
                append(new, "explicit_context_focus")
        else:
            target, other = rng.sample(names, 2)
            new = copy.deepcopy(row)
            new["past"] = rng.choice(AMBIGUOUS_CONTEXTS).format(
                target=target, other=other
            )
            new["context"] = _context(new, new["past"])
            new.update(
                spans=[],
                whole_need=True,
                whole_refs=[],
                whole_ref_ranges=[],
                support_keys=[],
                entities=[],
                prop="",
            )
            append(new, "unselected_context_subject")
    return additions


def validate_fit(rows):
    """Structural UTF-8/source-label checks; no model evaluation."""
    seen, counts = set(), Counter()
    for row in rows:
        if row["source_id"] in seen:
            raise ValueError("duplicate observation source id")
        seen.add(row["source_id"])
        counts[row.get("augmentation", "original")] += 1
        raw = row["body"].encode()
        if row["context"] != _context(row, row["past"]):
            raise ValueError("literal observation context mismatch")
        for span in row["spans"]:
            if not 0 <= span["start"] < span["end"] <= len(raw):
                raise ValueError("invalid source byte range")
            raw[span["start"] : span["end"]].decode()
            for location in span.get("ref_ranges", []):
                value = (
                    row[location["document"]]
                    .encode()[location["start"] : location["end"]]
                    .decode()
                )
                if value not in span["references"]:
                    raise ValueError("fact reference differs from exact original bytes")
        if len(row["whole_ref_ranges"]) != len(row["whole_refs"]):
            raise ValueError("inconsistent whole observation reference labels")
        for location, ref in zip(row["whole_ref_ranges"], row["whole_refs"]):
            if (
                row[location["document"]]
                .encode()[location["start"] : location["end"]]
                .decode()
                != ref
            ):
                raise ValueError("whole reference differs from exact original bytes")
        if row.get("query_class") == "ambiguous":
            if not row["whole_need"] or row["whole_refs"] or row["support_keys"]:
                raise ValueError("ambiguous query incorrectly selects a subject")
    return dict(rows=len(rows), by_augmentation=dict(counts), byte_ranges_valid=True)


def write_augmented(curriculum, output, seed=13131):
    source = Path(curriculum)
    original = json.loads(source.read_text())
    if not {"fit", "development", "blind"} <= original.keys():
        raise ValueError("frozen curriculum needs fit/development/blind splits")
    # Non-fit values are treated only as opaque objects to copy/hash unchanged.
    fit = original["fit"]
    additions = augment_fit(fit, seed)
    if not 500 <= len(additions) <= 1000:
        raise ValueError("augmentation size must remain between 500 and 1000 rows")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    augmented = dict(original, fit=fit + additions)
    validation = validate_fit(augmented["fit"])
    unchanged = {
        split: _digest(value) for split, value in original.items() if split != "fit"
    }
    if any(_digest(augmented[s]) != digest for s, digest in unchanged.items()):
        raise ValueError("an evaluation definition changed")
    manifest = dict(
        format="independent-colloquial-fit-augmentation-v1",
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_curriculum_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_fit_sha256=_digest(fit),
        augmented_fit_sha256=_digest(augmented["fit"]),
        preserved_split_hashes=unchanged,
        original_fit_rows=len(fit),
        added_fit_rows=len(additions),
        augmentation_seed=seed,
        validation=validation,
        generation_reads_splits=["fit"],
        evaluation_splits_copied_unchanged=True,
        locomo_used=False,
        external_text_read=False,
        model_evaluated=False,
        blind_used_for_generation=False,
        development_used_for_generation=False,
        selection_split="development",
        training_splits=["fit"],
    )
    (output / "curriculum.json").write_text(
        json.dumps(augmented, ensure_ascii=False, indent=2)
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13131)
    args = parser.parse_args()
    write_augmented(args.curriculum, args.output, args.seed)
