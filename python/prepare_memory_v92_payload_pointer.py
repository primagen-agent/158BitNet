#!/usr/bin/env python3
"""Build strict-OOD payload extraction data for writes and updates.

The target is always the exact new value that should become memory content.
Training uses synthetic, tokenizer-exact payloads; LoCoMo is never read.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prepare_memory_v89_short_copy import (
    ACKS,
    OOD_QUERY_TEMPLATES,
    OOD_STORE_TEMPLATES,
    PayloadFactory,
    TRAIN_QUERY_TEMPLATES,
    TRAIN_STORE_TEMPLATES,
    load_vocab_pieces,
)
from prepare_memory_v91_action_router import (
    ATTRIBUTES,
    OOD_IMPLICIT_WRITE_TEMPLATES,
    OOD_UPDATE_TEMPLATES,
    TRAIN_IMPLICIT_WRITE_TEMPLATES,
    TRAIN_UPDATE_TEMPLATES,
    VALUE_WORDS,
)
from train_data import CTokenizer


OPERATIONS = ("explicit_write", "implicit_write", "update")

TRAIN_COMPOUND_WORDS = (
    "amber", "birch", "coral", "delta", "ember", "falcon",
    "granite", "harbor", "indigo", "juniper", "kelp", "lunar",
    "maple", "nebula", "ochre", "pearl", "quartz", "river",
    "saffron", "teal", "umber", "velvet", "willow", "xenon",
)
OOD_COMPOUND_WORDS = (
    "cobalt", "cedar", "alpine", "bamboo", "cinder", "dahlia",
    "elm", "fjord", "ginger", "hazel", "iris", "jasmine",
    "lagoon", "mango", "nickel", "opal", "pine", "ruby",
    "spruce", "topaz", "violet", "walnut", "yarrow", "zinc",
)
TRAIN_COMPOUND_PATTERNS = (
    ("hyphen", "{a}-{b}"),
    ("hyphen", "{a}-{b}-{n}"),
    ("underscore", "{a}_{b}"),
    ("underscore", "{a}_{b}_{n}"),
    ("slash", "{a}/{b}"),
    ("slash", "{a}/{b}/{n}"),
    ("version", "v{n}.{m}.{p}"),
    ("version", "{a}-{n}.{m}"),
    ("colon", "{a}:{b}-{n}"),
    ("multiword", "{a} {n}"),
    ("multiword", "{n} {a}"),
    ("multiword", "{a} {b}"),
    ("multiword", "{a} {b} {n}"),
)
OOD_COMPOUND_PATTERNS = (
    ("hyphen", "{a}-{b}"),
    ("hyphen", "{a}-{n}-{b}"),
    ("underscore", "{a}_{n}_{b}"),
    ("underscore", "{a}_{b}"),
    ("slash", "{a}/{n}/{b}"),
    ("slash", "{a}/{b}"),
    ("version", "{n}.{m}.{p}-{a}"),
    ("version", "release-{n}.{m}-{a}"),
    ("colon", "{a}:{n}-{b}"),
    ("multiword", "{n} {a}"),
    ("multiword", "{a} {n}"),
    ("multiword", "{n} {a} {b}"),
    ("multiword", "{a} {b}"),
)

_COMPOSED_IMPLICIT_WRITE_TEMPLATES = [
    f"{prefix}{verb} {{value}} as my {{attribute}}{ending}"
    for prefix in (
        "", "For future conversations, ", "As a lasting detail, ",
        "From this point onward, ", "For later reference, ",
        "As my standing preference, ",
    )
    for verb in (
        "treat", "keep", "record", "use", "regard", "remember",
        "consider", "retain",
    )
    for ending in (
        ".", " going forward.", " unless I change it.",
        " as the persistent value.",
    )
] + [
    f"My {{attribute}} {relation} {{value}}{ending}"
    for relation in (
        "is", "remains", "should stay", "is set to",
        "continues to be", "has the enduring value",
    )
    for ending in (
        ".", " for future requests.", " as a stable preference.",
        " until I say otherwise.",
    )
]

_COMPOSED_UPDATE_TEMPLATES = [
    (
        f"{prefix}for my {{attribute}}, {verb} {{old}} "
        f"{connector} {{value}}{ending}"
    )
    for prefix in (
        "", "From now on, ", "Please ", "As a correction, ",
        "For future requests, ",
    )
    for verb in (
        "replace", "swap", "supersede", "change", "revise",
        "update",
    )
    for connector in ("with", "to", "in favor of")
    for ending in (
        ".", " going forward.", " as the new saved value.",
    )
] + [
    (
        f"{prefix}{verb} my {{attribute}} to {{value}} "
        f"{connector} {{old}}{ending}"
    )
    for prefix in (
        "", "Please ", "Going forward, ", "As an update, ",
        "For future conversations, ",
    )
    for verb in ("set", "change", "revise", "update", "switch")
    for connector in (
        "instead of", "rather than", "and retire",
    )
    for ending in (".", " immediately.", " in memory.")
] + [
    (
        f"The new {{attribute}} is {{value}}{separator}"
        f"{{old}} {obsolete}."
    )
    for separator in ("; ", ", while ", ". The previous value ")
    for obsolete in (
        "is obsolete", "should be discarded", "is no longer current",
        "must be replaced",
    )
] + [
    (
        f"{prefix}{verb} {{value}} {connector} {{old}} "
        f"for my {{attribute}}{ending}"
    )
    for prefix in (
        "", "For later requests, ", "Please ", "Going ahead, ",
        "As the corrected setting, ",
    )
    for verb in ("use", "remember", "keep", "take", "record")
    for connector in ("instead of", "rather than", "not")
    for ending in (".", " from now on.", " in memory.")
] + [
    (
        f"{prefix}{verb} the saved {{attribute}} entry "
        f"so it {predicate} {{value}}{ending}"
    )
    for prefix in ("", "Please ", "As a correction, ")
    for verb in ("amend", "revise", "change", "edit", "update")
    for predicate in ("reads", "becomes", "contains", "uses")
    for ending in (".", " going forward.", " as the current value.")
] + [
    (
        f"{prefix}{verb} {{old}} {connector} {{value}} "
        f"for my {{attribute}}{ending}"
    )
    for prefix in ("", "Please ", "For future use, ")
    for verb in ("replace", "supersede", "swap", "exchange")
    for connector in ("with", "for", "in favor of")
    for ending in (".", " now.", " in the saved record.")
]

TRAIN_IMPLICIT_POINTER_TEMPLATES = (
    TRAIN_IMPLICIT_WRITE_TEMPLATES
    + random.Random(20260909).sample(
        _COMPOSED_IMPLICIT_WRITE_TEMPLATES, 200)
)
TRAIN_UPDATE_POINTER_TEMPLATES = (
    TRAIN_UPDATE_TEMPLATES
    + _COMPOSED_UPDATE_TEMPLATES
)


def templates_for(split: str, operation: str):
    ood = split.endswith("ood")
    if operation == "explicit_write":
        return OOD_STORE_TEMPLATES if ood else TRAIN_STORE_TEMPLATES
    if operation == "implicit_write":
        return (
            OOD_IMPLICIT_WRITE_TEMPLATES
            if ood else TRAIN_IMPLICIT_POINTER_TEMPLATES
        )
    if operation == "update":
        return (
            OOD_UPDATE_TEMPLATES
            if ood else TRAIN_UPDATE_POINTER_TEMPLATES
        )
    raise ValueError(f"unknown operation {operation}")


def make_sample(
    split: str, index: int, token_length: int,
    payload: str, token_ids: list[int], rng: random.Random,
    old_payload: str | None = None, payload_family: str = "synthetic",
) -> dict:
    operation = OPERATIONS[index % len(OPERATIONS)]
    template = rng.choice(templates_for(split, operation))
    attribute = rng.choice(ATTRIBUTES)
    old = old_payload or (
        f"legacy-{rng.choice(VALUE_WORDS)}-"
        f"{token_length:02d}-{index:06d}")
    if operation == "explicit_write":
        record = template.format(payload=payload)
    else:
        record = template.format(
            attribute=attribute, value=payload, old=old)
    query_templates = (
        OOD_QUERY_TEMPLATES
        if split.endswith("ood") else TRAIN_QUERY_TEMPLATES
    )
    return {
        "sample_id": (
            f"{split}-{operation}-{token_length}-{index:06d}"
        ),
        "messages": [
            [
                {"role": "user", "content": record},
                {"role": "assistant", "content": rng.choice(ACKS)},
            ],
            [
                {
                    "role": "user",
                    "content": rng.choice(query_templates),
                },
                {"role": "assistant", "content": payload},
            ],
        ],
        "query_turn_id": 1,
        "evidence_message_indices": [0],
        "distractor_message_indices": [],
        "metadata": {
            "type": "memory_payload_extraction",
            "operation": operation,
            "token_length": token_length,
            "strict_ood": split.endswith("ood"),
            "target_token_ids": token_ids,
            "payload_family": payload_family,
            "locomo_used": False,
        },
    }


class CompoundPayloadFactory:
    """Create tokenizer-exact structured values with visible boundaries."""

    def __init__(
        self, tokenizer, words, patterns, seed,
        forbidden=None, shared_used=None,
    ):
        self.tokenizer = tokenizer
        self.words = tuple(words)
        self.patterns = tuple(patterns)
        self.rng = random.Random(seed)
        self.forbidden = forbidden if forbidden is not None else set()
        self.shared_used = shared_used
        self.used = set()
        self.exhausted_lengths = set()

    def validate(self, text: str, token_length: int):
        if text in self.used or text in self.forbidden:
            raise ValueError(f"payload is not available: {text!r}")
        token_ids = self.tokenizer.encode(text, add_bos=False)
        if len(token_ids) != token_length:
            raise ValueError(
                f"{text!r} has {len(token_ids)} tokens, "
                f"expected {token_length}")
        return text, token_ids, "compound_reserved"

    def accept(self, text: str):
        self.used.add(text)
        if self.shared_used is not None:
            self.shared_used.add(text)

    def register(self, text: str, token_length: int):
        result = self.validate(text, token_length)
        self.accept(text)
        return result

    def make(self, token_length: int):
        if token_length in self.exhausted_lengths:
            raise RuntimeError(
                f"compound candidates exhausted at {token_length} tokens")
        for _attempt in range(20_000):
            family, pattern = self.rng.choice(self.patterns)
            a, b = self.rng.sample(self.words, 2)
            text = pattern.format(
                a=a, b=b,
                n=self.rng.randint(1, 9999),
                m=self.rng.randint(0, 99),
                p=self.rng.randint(0, 99),
            )
            if text in self.used or text in self.forbidden:
                continue
            token_ids = self.tokenizer.encode(text, add_bos=False)
            if len(token_ids) != token_length:
                continue
            self.used.add(text)
            if self.shared_used is not None:
                self.shared_used.add(text)
            return text, token_ids, family
        self.exhausted_lengths.add(token_length)
        raise RuntimeError(
            "unable to construct a unique tokenizer-exact "
            f"{token_length}-token compound payload")


def write_dataset(
    root: Path, split: str, lengths: list[int],
    count_per_length: int, factory: PayloadFactory,
    distractor_factory: PayloadFactory | None, seed: int,
    compound_factory: CompoundPayloadFactory | None = None,
    compound_fraction: float = 0.0,
    reserved_payloads: dict[int, list[str]] | None = None,
) -> None:
    rng = random.Random(seed)
    reserved_payloads = reserved_payloads or {}
    for token_length in lengths:
        directory = root / split / f"len{token_length}"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "reconstruction.jsonl"
        pending_reserved = list(reserved_payloads.get(token_length, ()))
        with output.open("w", encoding="utf-8") as handle:
            for index in range(count_per_length):
                sample = None
                for _attempt in range(10_000):
                    use_reserved = (
                        pending_reserved
                        and OPERATIONS[index % len(OPERATIONS)] == "update"
                    )
                    if use_reserved:
                        if compound_factory is None:
                            raise RuntimeError(
                                "reserved payload needs compound factory")
                        payload, token_ids, payload_family = (
                            compound_factory.validate(
                                pending_reserved[0], token_length)
                        )
                    elif (
                        compound_factory is not None
                        and token_length >= 2
                        and rng.random() < compound_fraction
                    ):
                        try:
                            payload, token_ids, payload_family = (
                                compound_factory.make(token_length)
                            )
                        except RuntimeError:
                            payload, token_ids = factory.make(token_length)
                            payload_family = "multiword"
                    else:
                        payload, token_ids = factory.make(token_length)
                        payload_family = "synthetic"
                    old_payload = None
                    if (
                        OPERATIONS[index % len(OPERATIONS)]
                        == "update"
                        and distractor_factory is not None
                    ):
                        while (
                            old_payload is None
                            or old_payload == payload
                        ):
                            old_payload, _ = distractor_factory.make(
                                token_length)
                    candidate = make_sample(
                        split, index, token_length,
                        payload, token_ids, rng, old_payload,
                        payload_family)
                    record_bytes = candidate[
                        "messages"][0][0]["content"].encode("utf-8")
                    if record_bytes.count(
                        payload.encode("utf-8")
                    ) == 1:
                        sample = candidate
                        if use_reserved:
                            compound_factory.accept(payload)
                            pending_reserved.pop(0)
                        break
                if sample is None:
                    raise RuntimeError(
                        "unable to build an unambiguous payload row")
                handle.write(json.dumps(
                    sample, ensure_ascii=False,
                    separators=(",", ":")) + "\n")
        if pending_reserved:
            raise RuntimeError(
                f"not enough update rows for reserved payloads at "
                f"length {token_length}: {pending_reserved}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("output")
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--lengths", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--train-per-length", type=int, default=1200)
    parser.add_argument("--valid-per-length", type=int, default=250)
    parser.add_argument(
        "--compound-fraction", type=float, default=0.0,
        help="fraction of length >=2 rows using structured payloads")
    parser.add_argument(
        "--reserved-ood-payload", action="append", default=[],
        help="exact payload forced into an OOD update row")
    parser.add_argument(
        "--hard-update-distractors", action="store_true")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    lengths = [
        int(item) for item in args.lengths.split(",") if item.strip()
    ]
    if not lengths or any(length < 1 for length in lengths):
        parser.error("--lengths must contain positive integers")
    if args.train_per_length < 1 or args.valid_per_length < 1:
        parser.error("sample counts must be positive")
    if not 0.0 <= args.compound_fraction <= 1.0:
        parser.error("--compound-fraction must be between 0 and 1")

    root = Path(args.output)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    start, continuation = load_vocab_pieces(
        args.tok_probe, args.gguf)
    validation_pool = (
        3 * args.valid_per_length if 1 in lengths else 0)
    reserve = 2 * validation_pool
    if len(start) <= reserve:
        parser.error(
            "not enough one-token pieces for disjoint validation")
    split_rng = random.Random(args.seed)
    split_rng.shuffle(start)
    heldout_payloads: set[str] = set()
    valid_id = PayloadFactory(
        tokenizer, start[:validation_pool], continuation,
        args.seed + 1, forbidden=heldout_payloads,
        shared_used=heldout_payloads)
    valid_ood = PayloadFactory(
        tokenizer,
        start[validation_pool:2 * validation_pool],
        continuation, args.seed + 2,
        forbidden=heldout_payloads, shared_used=heldout_payloads)
    train = PayloadFactory(
        tokenizer, start[reserve:], continuation, args.seed,
        repeat_one_token=True, forbidden=heldout_payloads)
    compound_payloads: set[str] = set()
    valid_id_compounds = CompoundPayloadFactory(
        tokenizer, TRAIN_COMPOUND_WORDS, TRAIN_COMPOUND_PATTERNS,
        args.seed + 21, forbidden=compound_payloads,
        shared_used=compound_payloads)
    valid_ood_compounds = CompoundPayloadFactory(
        tokenizer, OOD_COMPOUND_WORDS, OOD_COMPOUND_PATTERNS,
        args.seed + 22, forbidden=compound_payloads,
        shared_used=compound_payloads)
    train_compounds = CompoundPayloadFactory(
        tokenizer, TRAIN_COMPOUND_WORDS, TRAIN_COMPOUND_PATTERNS,
        args.seed + 20, forbidden=compound_payloads)
    reserved_ood: dict[int, list[str]] = {}
    for payload in args.reserved_ood_payload:
        token_length = len(tokenizer.encode(payload, add_bos=False))
        if token_length not in lengths:
            parser.error(
                f"reserved OOD payload {payload!r} has {token_length} "
                "tokens, which is absent from --lengths")
        reserved_ood.setdefault(token_length, []).append(payload)
    valid_id_distractors = None
    valid_ood_distractors = None
    train_distractors = None
    if args.hard_update_distractors:
        valid_id_distractors = PayloadFactory(
            tokenizer, start, continuation, args.seed + 11,
            repeat_one_token=True)
        valid_ood_distractors = PayloadFactory(
            tokenizer, start, continuation, args.seed + 12,
            repeat_one_token=True)
        train_distractors = PayloadFactory(
            tokenizer, start, continuation, args.seed + 10,
            repeat_one_token=True)
    write_dataset(
        root, "valid_id", lengths, args.valid_per_length,
        valid_id, valid_id_distractors, args.seed + 2,
        valid_id_compounds, args.compound_fraction)
    write_dataset(
        root, "valid_ood", lengths, args.valid_per_length,
        valid_ood, valid_ood_distractors, args.seed + 3,
        valid_ood_compounds, args.compound_fraction, reserved_ood)
    write_dataset(
        root, "train", lengths, args.train_per_length,
        train, train_distractors, args.seed + 1,
        train_compounds, args.compound_fraction)
    print(json.dumps({
        "output": str(root),
        "lengths": lengths,
        "operations": list(OPERATIONS),
        "train": args.train_per_length * len(lengths),
        "valid_id": args.valid_per_length * len(lengths),
        "valid_ood": args.valid_per_length * len(lengths),
        "payloads_disjoint": True,
        "compound_fraction": args.compound_fraction,
        "reserved_ood_payloads": args.reserved_ood_payload,
        "hard_update_distractors": args.hard_update_distractors,
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
