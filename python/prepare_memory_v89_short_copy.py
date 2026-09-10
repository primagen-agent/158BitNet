#!/usr/bin/env python3
"""Build exact-token short-copy curricula with ID and OOD validation.

Payload lengths are measured by the C runtime tokenizer. Training and ID
validation share a diverse prompt-template family but use disjoint payloads;
OOD validation uses held-out instructions as well as disjoint payloads.
LoCoMo is never read.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

from train_data import CTokenizer


TRAIN_STORE_TEMPLATES = [
    "Remember this exact payload for a later request:\n{payload}",
    "Place the following text in memory without changing it:\n{payload}",
    "Keep this token sequence exactly as supplied:\n{payload}",
    "Store the content between the markers.\nBEGIN\n{payload}\nEND",
    "Save this literal value; do not summarize it: {payload}",
    "Memorize this sequence verbatim: {payload}",
    "Record the following payload byte-for-byte as text:\n{payload}",
    "Retain only the exact content after PAYLOAD: PAYLOAD: {payload}",
    "Create a memory entry whose precise value is {payload}",
    "The value that must be recalled later is: {payload}",
    "Preserve this short string in its original order:\n{payload}",
    "Commit this exact sequence to memory: {payload}",
]
_COMPOSED_STORE_TEMPLATES = [
    prefix
    + verb + " " + subject + constraint
    + separator + "{payload}" + suffix
    for prefix in (
        "", "Please ", "For later, ", "As a memory item, ",
    )
    for verb in (
        "remember", "retain", "store", "save", "record", "preserve",
        "memorize", "archive", "snapshot", "lodge",
        "keep available", "commit to memory",
    )
    for subject in (
        "the following payload", "the exact value below",
        "this literal sequence", "the supplied content",
        "the upcoming text", "this short data item",
        "this immutable text", "this checksum phrase",
        "this canonical value",
    )
    for constraint in (
        " exactly", " without rewriting it", " in its original order",
        " verbatim", " without interpreting it",
    )
    for separator, suffix in (
        (": ", ""),
        ("\n", ""),
        ("\nBEGIN\n", "\nEND"),
        (": {{", "}}"),
        (": \"", "\""),
        ("\nVALUE[", "]"),
    )
]
TRAIN_STORE_TEMPLATES += random.Random(20260909).sample(
    _COMPOSED_STORE_TEMPLATES, 512)
TRAIN_QUERY_TEMPLATES = [
    "Return the stored payload exactly, with no explanation.",
    "What exact token sequence did I ask you to remember?",
    "Recall the literal value. Output only that value.",
    "Reproduce the memorized content without adding words.",
    "Give the saved sequence in its original order.",
    "Retrieve the payload and print nothing else.",
    "Copy the stored value exactly.",
    "Respond only with the previously recorded content.",
    "What was the exact value placed in memory?",
    "Emit the remembered sequence verbatim.",
    "Recover the saved text, preserving its order.",
    "Provide only the exact memory payload.",
]
_COMPOSED_QUERY_TEMPLATES = [
    f"{verb} {object_text}{constraint}"
    for verb in (
        "Return", "Recall", "Retrieve", "Reproduce",
        "Read back", "Provide", "Emit", "Recover",
    )
    for object_text in (
        "the saved payload", "the remembered value",
        "the stored sequence", "the earlier content",
    )
    for constraint in (
        " exactly.", " verbatim.", " without explanation.",
        " and output nothing else.",
    )
]
TRAIN_QUERY_TEMPLATES += random.Random(20260910).sample(
    _COMPOSED_QUERY_TEMPLATES, 96)
OOD_STORE_TEMPLATES = [
    "Archive this symbol sequence unchanged for a future check:\n{payload}",
    "Treat the text inside braces as immutable memory: {{{payload}}}",
    "Later I will request this checksum phrase. Keep it exact: {payload}",
    "Without interpreting it, lodge this sequence in long-term storage: "
    "{payload}",
    "The canonical text for the next recall test is\n{payload}",
    "Snapshot this value exactly as written: {payload}",
]
OOD_QUERY_TEMPLATES = [
    "Print the archived symbol sequence and nothing surrounding it.",
    "Recover the immutable text from the earlier exchange.",
    "Supply the checksum phrase exactly as originally given.",
    "Read back the long-term value without interpretation.",
    "Output the canonical text from memory.",
    "Restore the snapshot value verbatim.",
]
ACKS = [
    "Stored.", "Recorded.", "Understood.", "I will retain it.",
    "The exact value is in memory.", "Saved without modification.",
]


def load_vocab_pieces(tok_probe: str, gguf: str):
    def decoded_pieces(mode):
        result = subprocess.run(
            [tok_probe, gguf, mode],
            check=True, stdout=subprocess.PIPE, text=True)
        pieces = []
        for line in result.stdout.splitlines():
            _token, hex_text = (line.split(" ", 1) + [""])[:2]
            if not hex_text:
                continue
            try:
                piece = bytes.fromhex(hex_text).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            if (
                not piece
                or len(piece) > 20
                or not piece.isprintable()
                or "<|" in piece
                or not any(character.isalnum() for character in piece)
            ):
                continue
            pieces.append(piece)
        return pieces

    roundtrip = decoded_pieces("--dump-roundtrip-vocab")
    all_pieces = decoded_pieces("--dump-vocab")
    start = [
        piece for piece in roundtrip
        if (
            piece == piece.strip()
            and not any(character.isspace() for character in piece)
            and re.search(r"[\w]", piece, re.UNICODE)
        )
    ]
    continuation = [
        piece for piece in all_pieces
        if (
            piece.startswith(" ")
            and piece[1:] == piece[1:].strip()
            and not any(character.isspace() for character in piece[1:])
            and re.search(r"[\w]", piece[1:], re.UNICODE)
        )
    ]
    if len(start) < 256 or len(continuation) < 256:
        raise RuntimeError(
            "not enough printable standalone tokenizer pieces: "
            f"start={len(start)} continuation={len(continuation)}")
    return sorted(set(start)), sorted(set(continuation))


class PayloadFactory:
    def __init__(
        self, tokenizer, start_pieces, continuation_pieces, seed,
        repeat_one_token=False, forbidden=None, shared_used=None,
    ):
        self.tokenizer = tokenizer
        self.start_pieces = list(start_pieces)
        self.continuation_pieces = list(continuation_pieces)
        self.rng = random.Random(seed)
        self.rng.shuffle(self.start_pieces)
        self.start_cursor = 0
        self.repeat_one_token = repeat_one_token
        self.forbidden = forbidden if forbidden is not None else set()
        self.shared_used = shared_used
        self.used = set()

    def make(self, token_length):
        if token_length == 1:
            while True:
                if self.start_cursor >= len(self.start_pieces):
                    if not self.repeat_one_token:
                        raise RuntimeError(
                            "not enough unique round-trip one-token payloads")
                    self.rng.shuffle(self.start_pieces)
                    self.start_cursor = 0
                text = self.start_pieces[self.start_cursor]
                self.start_cursor += 1
                if text in self.forbidden:
                    continue
                if not self.repeat_one_token and text in self.used:
                    continue
                token_ids = self.tokenizer.encode(text, add_bos=False)
                if len(token_ids) != 1:
                    raise RuntimeError(
                        "round-trip vocabulary contained a multi-token piece")
                self.used.add(text)
                if self.shared_used is not None:
                    self.shared_used.add(text)
                return text, token_ids
        for _attempt in range(50_000):
            text = (
                self.rng.choice(self.start_pieces)
                + "".join(
                    self.rng.choice(self.continuation_pieces)
                    for _ in range(token_length - 1))
            )
            if text in self.used or text in self.forbidden:
                continue
            token_ids = self.tokenizer.encode(text, add_bos=False)
            if len(token_ids) != token_length:
                continue
            self.used.add(text)
            if self.shared_used is not None:
                self.shared_used.add(text)
            return text, token_ids
        raise RuntimeError(
            f"unable to construct a unique {token_length}-token payload")


def make_sample(
    split, index, token_length, payload, token_ids, rng,
):
    if split.endswith("ood"):
        store_templates = OOD_STORE_TEMPLATES
        query_templates = OOD_QUERY_TEMPLATES
        template_family = "ood"
    else:
        store_templates = TRAIN_STORE_TEMPLATES
        query_templates = TRAIN_QUERY_TEMPLATES
        template_family = "id"
    store = rng.choice(store_templates).format(payload=payload)
    query = rng.choice(query_templates)
    return {
        "sample_id": f"{split}-copy-{token_length}-{index}",
        "messages": [
            [
                {"role": "user", "content": store},
                {"role": "assistant", "content": rng.choice(ACKS)},
            ],
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": payload},
            ],
        ],
        "query_turn_id": 1,
        "evidence_message_indices": [0],
        "distractor_message_indices": [],
        "metadata": {
            "type": "reconstruction",
            "style": "exact_token_copy",
            "v2_task": "task0",
            "token_length": token_length,
            "template_family": template_family,
            "target_token_ids": token_ids,
        },
    }


def write_dataset(
    root, split, lengths, count_per_length, factory, seed,
):
    rng = random.Random(seed)
    for token_length in lengths:
        directory = root / split / f"len{token_length}"
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / "reconstruction.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for index in range(count_per_length):
                payload, token_ids = factory.make(token_length)
                output.write(json.dumps(
                    make_sample(
                        split, index, token_length,
                        payload, token_ids, rng),
                    ensure_ascii=False,
                    separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("output")
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--lengths", default="1,2,4,8,16,32")
    parser.add_argument("--train-per-length", type=int, default=800)
    parser.add_argument("--valid-per-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    lengths = [
        int(value) for value in args.lengths.split(",") if value.strip()]
    if not lengths or any(value < 1 for value in lengths):
        parser.error("--lengths must contain positive integers")
    if args.train_per_length < 1 or args.valid_per_length < 1:
        parser.error("sample counts must be positive")

    root = Path(args.output)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    start, continuation = load_vocab_pieces(args.tok_probe, args.gguf)
    reserve = (
        2 * args.valid_per_length if 1 in lengths else 0)
    if len(start) <= reserve:
        parser.error(
            "not enough round-trip one-token payloads for disjoint "
            "validation splits")
    split_rng = random.Random(args.seed)
    split_rng.shuffle(start)
    valid_id_start = start[:args.valid_per_length]
    valid_ood_start = start[
        args.valid_per_length:2 * args.valid_per_length]
    train_start = start[reserve:]
    heldout_payloads = set()
    valid_id_factory = PayloadFactory(
        tokenizer, valid_id_start, continuation, args.seed + 1,
        forbidden=heldout_payloads, shared_used=heldout_payloads)
    valid_ood_factory = PayloadFactory(
        tokenizer, valid_ood_start, continuation, args.seed + 2,
        forbidden=heldout_payloads, shared_used=heldout_payloads)
    write_dataset(
        root, "valid_id", lengths, args.valid_per_length,
        valid_id_factory, args.seed + 2)
    write_dataset(
        root, "valid_ood", lengths, args.valid_per_length,
        valid_ood_factory, args.seed + 3)
    train_factory = PayloadFactory(
        tokenizer, train_start, continuation, args.seed,
        repeat_one_token=True, forbidden=heldout_payloads)
    write_dataset(
        root, "train", lengths, args.train_per_length,
        train_factory, args.seed + 1)
    print(json.dumps({
        "output": str(root),
        "lengths": lengths,
        "train_per_length": args.train_per_length,
        "valid_per_length": args.valid_per_length,
        "start_pieces": len(start),
        "continuation_pieces": len(continuation),
        "train_one_token_pool": len(train_start),
        "payloads_disjoint": True,
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
