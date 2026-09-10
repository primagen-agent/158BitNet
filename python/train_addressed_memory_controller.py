#!/usr/bin/env python3
"""Train the V84 addressed-memory controller.

The frozen backbone supplies pooled text features.  The trainable controller
learns three coupled tasks:
  1. IGNORE / WRITE / UPDATE / DELETE routing;
  2. value-invariant memory addresses for write/update/delete variants;
  3. query-to-address retrieval.

Exact memory text remains dynamic data in .bnepisodic; it is never compressed
into controller weights.  No LoRA or KV cache is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ggw import GGUFWeights
from model_identity import sha256_file
from memory_text_encoding import action_text, encode_text, query_text
from torch_backbone import TorchBackbone
from train_data import CTokenizer


IGNORE, WRITE, UPDATE, DELETE = range(4)
ACTION_NAMES = ("ignore", "write", "update", "delete")

NAMES = """Diana Eric Fiona Gary Hannah Ivan Jasmine Karl Luna Mason Nora
Oscar Petra Quinn Renee Stefan Tara Ulric Vera Wade Xena Yara Zane""".split()
ATTRS = [
    ("favorite season", ["spring", "summer", "autumn", "winter"]),
    ("lucky number", ["3", "7", "9", "21", "42", "88"]),
    ("favorite music genre",
     ["jazz", "classical", "rock", "folk", "electronic"]),
    ("preferred drink",
     ["oolong tea", "espresso", "hot cocoa", "lemon water", "matcha"]),
    ("weekend hobby",
     ["fishing", "painting", "birdwatching", "pottery", "cycling"]),
    ("favorite flower",
     ["peony", "jasmine", "sunflower", "orchid", "cornflower"]),
    ("response language",
     ["English", "Japanese", "Chinese", "Spanish", "German"]),
    ("home city",
     ["Kyoto", "Osaka", "Seattle", "Berlin", "Lisbon"]),
    ("project codename",
     ["amber-orchid", "cobalt-cedar", "silver-pine", "violet-river"]),
    ("editor theme",
     ["solarized dark", "monokai", "nord", "gruvbox"]),
    ("default device",
     ["work laptop", "tablet", "desktop", "phone"]),
    ("meeting preference",
     ["mornings", "afternoons", "video calls", "written updates"]),
    ("favorite color",
     ["blue", "green", "violet", "orange", "black"]),
    ("timezone",
     ["UTC", "Asia Tokyo", "Europe Berlin", "America New York"]),
    ("dietary preference",
     ["vegetarian", "vegan", "pescatarian", "no preference"]),
    ("communication style",
     ["concise", "detailed", "formal", "casual"]),
]

WRITE_TEMPLATES = [
    "Please remember that {name}'s {attr} is {value}.",
    "{name} told me their {attr} is {value}.",
    "For future reference, {name}'s {attr} is {value}.",
    "{name}: my {attr} is {value}.",
]
UPDATE_TEMPLATES = [
    "Update {name}'s {attr}: it is now {value}, not {old}.",
    "{name}'s {attr} changed from {old} to {value}.",
    "Replace the old value for {name}'s {attr} with {value}.",
    "{name}: my {attr} is now {value}.",
]
DELETE_TEMPLATES = [
    "Please forget {name}'s {attr}.",
    "Delete the stored value for {name}'s {attr}.",
    "{name}'s {attr} is private now; remove it from memory.",
    "Unset {name}'s {attr}.",
]
QUERY_TEMPLATES = [
    "What is {name}'s {attr}?",
    "Do you remember {name}'s {attr}?",
    "Tell me the stored value of {name}'s {attr}.",
]
IGNORE_TEMPLATES = [
    "What is the weather today?",
    "Explain why the sky is blue.",
    "Write a short poem about {value}.",
    "Hello {name}, how are you?",
    "Can you summarize this sentence?",
    "What does the word {value} mean?",
]
IMPLICIT_WRITE_TEMPLATES = [
    "When it comes to my {attr}, {value} is the one.",
    "For context, I would name {value} as my {attr}.",
    "I have a soft spot for {value}; that is my {attr}.",
    "I cannot get enough of {value}; it is still my {attr}.",
    "I would never give up {value}; that remains my {attr}.",
]
IMPLICIT_UPDATE_TEMPLATES = [
    "These days my {attr} is {value}; I moved on from {old}.",
    "I changed my mind about my {attr}: {old} is out and {value} is in.",
    "My current {attr} is {value}, replacing {old}.",
]
IMPLICIT_DELETE_TEMPLATES = [
    "Please do not keep the detail I shared about my {attr}.",
    "I withdraw my earlier answer about my {attr}; erase it.",
    "That information about my {attr} is private, so discard it.",
]
OOD_WRITE_TEMPLATES = [
    "Record {name}'s {attr} as {value} for later.",
    "The persistent {attr} for {name} is {value}.",
]
OOD_UPDATE_TEMPLATES = [
    "Amend {name}'s persistent {attr} entry so it reads {value}.",
    "For {name}, supersede {old} with {value} as the {attr}.",
]
OOD_DELETE_TEMPLATES = [
    "Purge the persistent record of {name}'s {attr}.",
    "Revoke the saved {attr} detail for {name}.",
]
OOD_QUERY_TEMPLATES = [
    "Which {attr} is stored for {name}?",
    "Return {name}'s remembered {attr}.",
]
OOD_IMPLICIT_WRITE_TEMPLATES = [
    "The persistent setting I use for {attr} is {value}.",
    "You can treat {value} as my established {attr}.",
]
OOD_IMPLICIT_UPDATE_TEMPLATES = [
    "Amend the persistent {attr} entry so it reads {value}.",
    "Going forward, remember {value} instead of {old} for {attr}.",
]
OOD_IMPLICIT_DELETE_TEMPLATES = [
    "Purge the persistent record of my {attr}.",
    "The saved {attr} detail must no longer be retained.",
]
OOD_SELF_QUERY_TEMPLATES = [
    "What is my {attr}?",
    "Return my remembered {attr}.",
]
TRAIN_SELF_QUERY_TEMPLATES = [
    "What is my {attr}?",
    "Do you remember my {attr}?",
    "Tell me the stored value of my {attr}.",
]

_COMPOSED_IMPLICIT_WRITES = [
    f"{prefix}{verb} {{value}} as my {{attr}}{ending}"
    for prefix in (
        "", "For future conversations, ", "As a lasting detail, ",
        "For later reference, ",
    )
    for verb in ("use", "keep", "record", "remember", "treat")
    for ending in (
        ".", " going forward.", " as the persistent setting.",
    )
]
_COMPOSED_IMPLICIT_UPDATES = [
    (
        f"{prefix}{verb} my {{attr}} to {{value}} "
        f"{connector} {{old}}{ending}"
    )
    for prefix in ("", "Please ", "Going forward, ", "As a correction, ")
    for verb in ("change", "update", "revise", "set")
    for connector in ("instead of", "rather than", "and retire")
    for ending in (".", " in memory.", " from now on.")
]
_COMPOSED_IMPLICIT_DELETES = [
    f"{prefix}{verb} the saved detail about my {{attr}}{ending}"
    for prefix in ("", "Please ", "For privacy, ")
    for verb in ("remove", "erase", "discard", "forget")
    for ending in (".", " from memory.", " permanently.")
]
TRAIN_IMPLICIT_WRITE_TEMPLATES = (
    IMPLICIT_WRITE_TEMPLATES + _COMPOSED_IMPLICIT_WRITES)
TRAIN_IMPLICIT_UPDATE_TEMPLATES = (
    IMPLICIT_UPDATE_TEMPLATES + _COMPOSED_IMPLICIT_UPDATES)
TRAIN_IMPLICIT_DELETE_TEMPLATES = (
    IMPLICIT_DELETE_TEMPLATES + _COMPOSED_IMPLICIT_DELETES)


@dataclass
class AddressExample:
    address: int
    attr_id: int
    write: str
    update: str
    delete: str
    query: str
    ignore: str
    action_write: str
    action_update: str
    action_delete: str
    action_unset: str
    action_query: str
    implicit_write: str
    implicit_update: str
    implicit_delete: str
    self_query: str


def make_examples(
    seed: int, count: int, offset: int = 0,
    strict_ood: bool = False,
) -> list[AddressExample]:
    addresses = [
        (name, attr_id, attr, values)
        for name in NAMES
        for attr_id, (attr, values) in enumerate(ATTRS)
    ]
    random.Random(seed).shuffle(addresses)
    selected = addresses[offset:offset + count]
    if len(selected) != count:
        raise ValueError(
            f"requested {offset + count} unique addresses, "
            f"only {len(addresses)} are available")
    rng = random.Random(seed + 1009 + offset)
    examples = []
    for index, (name, attr_id, attr, values) in enumerate(selected):
        old, value = rng.sample(values, 2)
        fields = {
            "name": name, "attr": attr, "old": old, "value": value,
        }
        write_templates = (
            OOD_WRITE_TEMPLATES if strict_ood else WRITE_TEMPLATES)
        update_templates = (
            OOD_UPDATE_TEMPLATES if strict_ood else UPDATE_TEMPLATES)
        delete_templates = (
            OOD_DELETE_TEMPLATES if strict_ood else DELETE_TEMPLATES)
        query_templates = (
            OOD_QUERY_TEMPLATES if strict_ood else QUERY_TEMPLATES)
        implicit_write_templates = (
            OOD_IMPLICIT_WRITE_TEMPLATES
            if strict_ood else TRAIN_IMPLICIT_WRITE_TEMPLATES)
        implicit_update_templates = (
            OOD_IMPLICIT_UPDATE_TEMPLATES
            if strict_ood else TRAIN_IMPLICIT_UPDATE_TEMPLATES)
        implicit_delete_templates = (
            OOD_IMPLICIT_DELETE_TEMPLATES
            if strict_ood else TRAIN_IMPLICIT_DELETE_TEMPLATES)
        self_query_templates = (
            OOD_SELF_QUERY_TEMPLATES
            if strict_ood else TRAIN_SELF_QUERY_TEMPLATES)
        examples.append(AddressExample(
            address=offset + index,
            attr_id=attr_id,
            write=rng.choice(write_templates).format(**fields),
            update=rng.choice(update_templates).format(**fields),
            delete=rng.choice(delete_templates).format(**fields),
            query=rng.choice(query_templates).format(**fields),
            ignore=rng.choice(IGNORE_TEMPLATES).format(**fields),
            action_write=(
                f"Please remember that my {attr} is {value}."),
            action_update=(
                f"Update: my {attr} is now {value}."),
            action_delete=f"Please forget my {attr}.",
            action_unset=f"My {attr} is now unset.",
            action_query=f"What is my {attr}?",
            implicit_write=rng.choice(
                implicit_write_templates).format(**fields),
            implicit_update=rng.choice(
                implicit_update_templates).format(**fields),
            implicit_delete=rng.choice(
                implicit_delete_templates).format(**fields),
            self_query=rng.choice(
                self_query_templates).format(**fields),
        ))
    return examples


class AddressedController(nn.Module):
    def __init__(self, hidden: int, rank: int, action_rank: int = 0):
        super().__init__()
        self.query = nn.Linear(hidden, rank, bias=False)
        self.entry = nn.Linear(hidden, rank, bias=False)
        self.action_rank = action_rank
        self.action_hidden = (
            nn.Linear(hidden, action_rank)
            if action_rank > 0 else None)
        self.action = nn.Linear(
            action_rank if action_rank > 0 else hidden, 4)
        nn.init.orthogonal_(self.query.weight)
        nn.init.orthogonal_(self.entry.weight)

    def query_keys(self, hidden):
        return F.normalize(self.query(hidden), dim=-1)

    def entry_keys(self, hidden):
        return F.normalize(self.entry(hidden), dim=-1)

    def action_logits(self, hidden):
        if self.action_hidden is not None:
            hidden = F.silu(self.action_hidden(hidden))
        return self.action(hidden)


def encode_examples(
    backbone, tokenizer, examples, max_tokens: int, pooling: str,
):
    rows = {
        "write": [], "update": [], "delete": [], "query": [], "ignore": [],
        "action_write": [], "action_update": [],
        "action_delete": [], "action_unset": [], "action_query": [],
        "implicit_write": [], "implicit_update": [], "implicit_delete": [],
        "self_query": [],
    }
    for index, example in enumerate(examples):
        for kind in rows:
            text = getattr(example, kind)
            if kind in ("query", "self_query"):
                text = query_text(text)
            rows[kind].append(encode_text(
                backbone, tokenizer, text, max_tokens, pooling))
        if (index + 1) % 100 == 0:
            print(json.dumps({
                "phase": "encode", "examples": index + 1,
            }), flush=True)
    tensors = {
        key: torch.from_numpy(np.stack(value))
        for key, value in rows.items()
    }
    tensors["attr_id"] = torch.tensor(
        [example.attr_id for example in examples], dtype=torch.long)
    return tensors


def encode_action_dataset(
    backbone, tokenizer, path, max_tokens: int, pooling: str,
):
    hidden = []
    labels = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        label = int(row["label"])
        if label < IGNORE or label > DELETE:
            raise ValueError(f"invalid action label {label} in {path}")
        hidden.append(encode_text(
            backbone, tokenizer, action_text(row["text"]),
            max_tokens, pooling))
        labels.append(label)
        if (index + 1) % 250 == 0:
            print(json.dumps({
                "phase": "encode_action",
                "data": str(path),
                "examples": index + 1,
            }), flush=True)
    if not hidden:
        raise ValueError(f"empty action dataset: {path}")
    return {
        "hidden": torch.from_numpy(np.stack(hidden)),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def addressed_loss(
    model, tensors, indices, temperature, action_weight,
    action_tensors=None, action_indices=None,
    action_ignore_weight=1.0, action_update_weight=1.0,
):
    write_h = tensors["write"][indices]
    update_h = tensors["update"][indices]
    delete_h = tensors["delete"][indices]
    query_h = tensors["query"][indices]
    ignore_h = tensors["ignore"][indices]

    write_k = model.entry_keys(write_h)
    update_k = model.entry_keys(update_h)
    delete_k = model.entry_keys(delete_h)
    query_k = model.query_keys(query_h)
    implicit_write_k = model.entry_keys(tensors["implicit_write"][indices])
    implicit_update_k = model.entry_keys(tensors["implicit_update"][indices])
    implicit_delete_k = model.entry_keys(tensors["implicit_delete"][indices])
    self_query_k = model.query_keys(tensors["self_query"][indices])
    targets = torch.arange(indices.numel(), device=indices.device)
    named_retrieval = (
        F.cross_entropy(query_k @ write_k.T / temperature, targets) +
        F.cross_entropy(write_k @ query_k.T / temperature, targets)
    ) * 0.5
    attrs = tensors["attr_id"][indices]
    selected = torch.stack([
        torch.nonzero(attrs == attr, as_tuple=False)[0, 0]
        for attr in torch.unique(attrs, sorted=True)
    ])
    self_targets = torch.arange(selected.numel(), device=indices.device)
    self_retrieval = (
        F.cross_entropy(
            self_query_k[selected] @ implicit_write_k[selected].T /
            temperature, self_targets) +
        F.cross_entropy(
            implicit_write_k[selected] @ self_query_k[selected].T /
            temperature, self_targets)
    ) * 0.5
    retrieval = (named_retrieval + self_retrieval) * 0.5
    named_invariance = (
        (1.0 - (write_k * update_k).sum(-1)).mean() +
        (1.0 - (write_k * delete_k).sum(-1)).mean()
    ) * 0.5
    implicit_invariance = (
        (1.0 - (implicit_write_k * implicit_update_k).sum(-1)).mean() +
        (1.0 - (implicit_write_k * implicit_delete_k).sum(-1)).mean()
    ) * 0.5
    invariance = (named_invariance + implicit_invariance) * 0.5
    if action_tensors is not None:
        if action_indices is None:
            raise ValueError("action indices are required")
        action_h = action_tensors["hidden"][action_indices]
        action_y = action_tensors["labels"][action_indices]
    else:
        action_h = torch.cat((
            ignore_h, query_h,
            write_h, tensors["action_write"][indices],
            update_h, tensors["action_update"][indices],
            delete_h, tensors["action_delete"][indices],
            tensors["action_unset"][indices],
            tensors["action_query"][indices],
            tensors["implicit_write"][indices],
            tensors["implicit_update"][indices],
            tensors["implicit_delete"][indices],
        ), dim=0)
        action_y = torch.cat([
            torch.full((indices.numel(),), action, device=indices.device)
            for action in (
                IGNORE, IGNORE,
                WRITE, WRITE,
                UPDATE, UPDATE,
                DELETE, DELETE, DELETE,
                IGNORE,
                WRITE, UPDATE, DELETE,
            )
        ])
    action_class_weights = torch.ones(
        4, device=action_h.device, dtype=action_h.dtype)
    action_class_weights[IGNORE] = action_ignore_weight
    action_class_weights[UPDATE] = action_update_weight
    action = F.cross_entropy(
        model.action_logits(action_h), action_y,
        weight=action_class_weights)
    return retrieval + invariance + action_weight * action, {
        "retrieval_loss": retrieval.detach(),
        "invariance_loss": invariance.detach(),
        "action_loss": action.detach(),
    }


@torch.no_grad()
def metrics(model, tensors, action_tensors=None):
    write = model.entry_keys(tensors["write"])
    update = model.entry_keys(tensors["update"])
    delete = model.entry_keys(tensors["delete"])
    query = model.query_keys(tensors["query"])
    implicit_write = model.entry_keys(tensors["implicit_write"])
    implicit_update = model.entry_keys(tensors["implicit_update"])
    implicit_delete = model.entry_keys(tensors["implicit_delete"])
    self_query = model.query_keys(tensors["self_query"])
    attr_ids = tensors["attr_id"]
    unique_attrs = torch.unique(attr_ids, sorted=True)
    prototypes = torch.stack([
        F.normalize(
            implicit_write[attr_ids == attr].mean(dim=0), dim=0)
        for attr in unique_attrs
    ])
    attr_targets = torch.searchsorted(unique_attrs, attr_ids)
    targets = torch.arange(write.shape[0], device=write.device)
    if action_tensors is not None:
        action_h = action_tensors["hidden"]
        action_y = action_tensors["labels"]
    else:
        action_h = torch.cat([
            tensors["ignore"], tensors["query"],
            tensors["write"], tensors["action_write"],
            tensors["update"], tensors["action_update"],
            tensors["delete"], tensors["action_delete"],
            tensors["action_unset"],
            tensors["action_query"],
            tensors["implicit_write"], tensors["implicit_update"],
            tensors["implicit_delete"],
        ])
        action_y = torch.cat([
            torch.full((write.shape[0],), action, device=write.device)
            for action in (
                IGNORE, IGNORE,
                WRITE, WRITE,
                UPDATE, UPDATE,
                DELETE, DELETE, DELETE,
                IGNORE,
                WRITE, UPDATE, DELETE,
            )
        ])
    action_prediction = model.action_logits(action_h).argmax(-1)
    positives = torch.cat([
        (write * update).sum(-1), (write * delete).sum(-1)])
    negative = write @ write.T
    negative.fill_diagonal_(-2.0)
    negatives = negative.max(dim=1).values
    threshold, balanced = choose_threshold(positives, negatives)
    return {
        "action_accuracy": float(
            (action_prediction == action_y).float().mean()),
        **{
            f"action_{name}_recall": float(
                (action_prediction[action_y == action] == action)
                .float().mean())
            for action, name in enumerate(ACTION_NAMES)
        },
        "query_top1": float(
            ((query @ write.T).argmax(-1) == targets).float().mean()),
        "update_top1": float(
            ((update @ write.T).argmax(-1) == targets).float().mean()),
        "delete_top1": float(
            ((delete @ write.T).argmax(-1) == targets).float().mean()),
        "implicit_query_top1": float(
            ((self_query @ prototypes.T).argmax(-1) == attr_targets)
            .float().mean()),
        "implicit_update_top1": float(
            ((implicit_update @ prototypes.T).argmax(-1) == attr_targets)
            .float().mean()),
        "implicit_delete_top1": float(
            ((implicit_delete @ prototypes.T).argmax(-1) == attr_targets)
            .float().mean()),
        "positive_cosine": float(positives.mean()),
        "hard_negative_cosine": float(negatives.mean()),
        "address_threshold": threshold,
        "threshold_balanced_accuracy": balanced,
    }


def choose_threshold(positives, negatives):
    best_threshold = 0.8
    best_accuracy = -1.0
    for threshold in torch.linspace(-1.0, 1.0, 401, device=positives.device):
        accuracy = 0.5 * (
            (positives >= threshold).float().mean() +
            (negatives < threshold).float().mean())
        value = float(accuracy)
        if value > best_accuracy:
            best_accuracy = value
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def save_bnctrl(
    path, model, gguf, hidden, rank, temperature, pooling, threshold,
):
    pooling_id = {"last": 1, "mean_last": 2}[pooling]
    shared_payloads = [
        model.query.weight.detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False).tobytes(),
        model.entry.weight.detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False).tobytes(),
    ]
    if model.action_hidden is None:
        version = 3
        header = bytearray(b"BNCTRL3\x00")
        header += struct.pack(
            "<IIIIff", version, hidden, rank, pooling_id,
            float(temperature), float(threshold))
        action_payloads = [
            model.action.weight.detach().float().cpu()
            .contiguous().numpy().astype("<f4", copy=False).tobytes(),
            model.action.bias.detach().float().cpu()
            .contiguous().numpy().astype("<f4", copy=False).tobytes(),
        ]
    else:
        version = 4
        header = bytearray(b"BNCTRL4\x00")
        header += struct.pack(
            "<IIIIffI", version, hidden, rank, pooling_id,
            float(temperature), float(threshold), model.action_rank)
        action_payloads = [
            model.action_hidden.weight.detach().float().cpu()
            .contiguous().numpy().astype("<f4", copy=False).tobytes(),
            model.action_hidden.bias.detach().float().cpu()
            .contiguous().numpy().astype("<f4", copy=False).tobytes(),
            model.action.weight.detach().float().cpu()
            .contiguous().numpy().astype("<f4", copy=False).tobytes(),
            model.action.bias.detach().float().cpu()
            .contiguous().numpy().astype("<f4", copy=False).tobytes(),
        ]
    payloads = shared_payloads + action_payloads
    header += sha256_file(gguf)
    for payload in payloads:
        header += struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
    Path(path).write_bytes(header + b"".join(payloads))


def move_tensors(tensors, device):
    return {key: value.to(device=device) for key, value in tensors.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("output")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--action-train")
    parser.add_argument("--action-valid")
    parser.add_argument("--action-cache")
    parser.add_argument("--train-examples", type=int, default=110)
    parser.add_argument("--valid-examples", type=int, default=28)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--action-rank", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--action-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-ignore-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-update-weight", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--pooling", choices=("last", "mean_last"), default="mean_last")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"),
        default="auto")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if bool(args.action_train) != bool(args.action_valid):
        parser.error("--action-train and --action-valid must be used together")
    if (
        args.steps < 1 or args.batch < 1 or args.action_batch < 1
        or args.train_examples < 1 or args.valid_examples < 1
        or args.action_ignore_weight <= 0.0
        or args.action_update_weight <= 0.0
        or args.action_rank < 0
    ):
        parser.error("steps, batches, and example counts must be positive")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("mps" if args.device == "auto" and
              torch.backends.mps.is_available()
              else ("cpu" if args.device == "auto" else args.device)))
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    cache_path = Path(args.cache) if args.cache else None
    started = time.time()
    if cache_path is not None and cache_path.exists():
        archive = np.load(cache_path)
        train = {
            kind: torch.from_numpy(archive["train_" + kind])
            for kind in (
                "write", "update", "delete", "query", "ignore",
                "action_write", "action_update",
                "action_delete", "action_unset", "action_query",
                "implicit_write", "implicit_update", "implicit_delete",
                "self_query", "attr_id",
            )}
        valid = {
            kind: torch.from_numpy(archive["valid_" + kind])
            for kind in (
                "write", "update", "delete", "query", "ignore",
                "action_write", "action_update",
                "action_delete", "action_unset", "action_query",
                "implicit_write", "implicit_update", "implicit_delete",
                "self_query", "attr_id",
            )}
    else:
        weights = GGUFWeights(args.gguf, args.lib)
        backbone = TorchBackbone(
            weights, device=device, dtype=torch.bfloat16)
        tokenizer = CTokenizer(args.tok_probe, args.gguf)
        train = encode_examples(
            backbone, tokenizer,
            make_examples(args.seed, args.train_examples),
            args.max_tokens, args.pooling)
        valid = encode_examples(
            backbone, tokenizer,
            make_examples(
                args.seed, args.valid_examples,
                offset=args.train_examples, strict_ood=True),
            args.max_tokens, args.pooling)
        weights.close()
        del backbone
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **{
                split + "_" + kind: tensor.numpy()
                for split, values in (("train", train), ("valid", valid))
                for kind, tensor in values.items()
            })
    action_train = None
    action_valid = None
    if args.action_train:
        action_cache = (
            Path(args.action_cache) if args.action_cache else None)
        if action_cache is not None and action_cache.exists():
            archive = np.load(action_cache)
            action_train = {
                "hidden": torch.from_numpy(archive["train_hidden"]),
                "labels": torch.from_numpy(archive["train_labels"]),
            }
            action_valid = {
                "hidden": torch.from_numpy(archive["valid_hidden"]),
                "labels": torch.from_numpy(archive["valid_labels"]),
            }
        else:
            weights = GGUFWeights(args.gguf, args.lib)
            backbone = TorchBackbone(
                weights, device=device, dtype=torch.bfloat16)
            tokenizer = CTokenizer(args.tok_probe, args.gguf)
            action_train = encode_action_dataset(
                backbone, tokenizer, args.action_train,
                args.max_tokens, args.pooling)
            action_valid = encode_action_dataset(
                backbone, tokenizer, args.action_valid,
                args.max_tokens, args.pooling)
            weights.close()
            del backbone
            if action_cache is not None:
                action_cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    action_cache,
                    train_hidden=action_train["hidden"].numpy(),
                    train_labels=action_train["labels"].numpy(),
                    valid_hidden=action_valid["hidden"].numpy(),
                    valid_labels=action_valid["labels"].numpy(),
                )
    hidden = train["write"].shape[1]
    train = move_tensors(train, device)
    valid = move_tensors(valid, device)
    if action_train is not None:
        action_train = move_tensors(action_train, device)
        action_valid = move_tensors(action_valid, device)
    model = AddressedController(
        hidden, args.rank, args.action_rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)
    best_score = -1.0
    best_state = None
    for step in range(args.steps):
        indices = torch.randperm(
            train["write"].shape[0], generator=rng
        )[:min(args.batch, train["write"].shape[0])].to(device)
        action_indices = None
        if action_train is not None:
            action_indices = torch.randperm(
                action_train["hidden"].shape[0], generator=rng
            )[:min(
                args.action_batch,
                action_train["hidden"].shape[0])].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = addressed_loss(
            model, train, indices, args.temperature, args.action_weight,
            action_train, action_indices,
            action_ignore_weight=args.action_ignore_weight,
            action_update_weight=args.action_update_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            report = metrics(model, valid, action_valid)
            score = (
                report["action_accuracy"] + report["query_top1"] +
                report["update_top1"] + report["delete_top1"] +
                report["implicit_query_top1"] +
                report["implicit_update_top1"] +
                report["implicit_delete_top1"])
            print(json.dumps({
                "step": step + 1, "loss": float(loss.detach()),
                **{key: float(value) for key, value in parts.items()},
                **report,
            }), flush=True)
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    report = metrics(model, valid, action_valid)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_bnctrl(
        output, model, args.gguf, hidden, args.rank,
        args.temperature, args.pooling, report["address_threshold"])
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": hidden,
        "rank": args.rank,
        "action_rank": args.action_rank,
        "pooling": args.pooling,
        "temperature": args.temperature,
        "backbone_sha256": sha256_file(args.gguf).hex(),
        "metrics": report,
    }, output.with_suffix(".pt"))
    print(json.dumps({
        "result": "done", "output": str(output),
        "seconds": time.time() - started, **report,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
