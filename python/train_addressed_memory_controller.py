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
from memory_text_encoding import encode_text, query_text
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
        examples.append(AddressExample(
            address=offset + index,
            attr_id=attr_id,
            write=rng.choice(WRITE_TEMPLATES).format(**fields),
            update=rng.choice(UPDATE_TEMPLATES).format(**fields),
            delete=rng.choice(DELETE_TEMPLATES).format(**fields),
            query=rng.choice(QUERY_TEMPLATES).format(**fields),
            ignore=rng.choice(IGNORE_TEMPLATES).format(**fields),
            action_write=(
                f"Please remember that my {attr} is {value}."),
            action_update=(
                f"Update: my {attr} is now {value}."),
            action_delete=f"Please forget my {attr}.",
            action_unset=f"My {attr} is now unset.",
            action_query=f"What is my {attr}?",
            implicit_write=rng.choice(
                IMPLICIT_WRITE_TEMPLATES).format(**fields),
            implicit_update=rng.choice(
                IMPLICIT_UPDATE_TEMPLATES).format(**fields),
            implicit_delete=rng.choice(
                IMPLICIT_DELETE_TEMPLATES).format(**fields),
            self_query=f"What is my {attr}?",
        ))
    return examples


class AddressedController(nn.Module):
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.query = nn.Linear(hidden, rank, bias=False)
        self.entry = nn.Linear(hidden, rank, bias=False)
        self.action = nn.Linear(hidden, 4)
        nn.init.orthogonal_(self.query.weight)
        nn.init.orthogonal_(self.entry.weight)

    def query_keys(self, hidden):
        return F.normalize(self.query(hidden), dim=-1)

    def entry_keys(self, hidden):
        return F.normalize(self.entry(hidden), dim=-1)


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


def addressed_loss(model, tensors, indices, temperature, action_weight):
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
    action = F.cross_entropy(model.action(action_h), action_y)
    return retrieval + invariance + action_weight * action, {
        "retrieval_loss": retrieval.detach(),
        "invariance_loss": invariance.detach(),
        "action_loss": action.detach(),
    }


@torch.no_grad()
def metrics(model, tensors):
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
    positives = torch.cat([
        (write * update).sum(-1), (write * delete).sum(-1)])
    negative = write @ write.T
    negative.fill_diagonal_(-2.0)
    negatives = negative.max(dim=1).values
    threshold, balanced = choose_threshold(positives, negatives)
    return {
        "action_accuracy": float(
            (model.action(action_h).argmax(-1) == action_y).float().mean()),
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


def save_bnctrl3(
    path, model, gguf, hidden, rank, temperature, pooling, threshold,
):
    pooling_id = {"last": 1, "mean_last": 2}[pooling]
    payloads = [
        model.query.weight.detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False).tobytes(),
        model.entry.weight.detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False).tobytes(),
        model.action.weight.detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False).tobytes(),
        model.action.bias.detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False).tobytes(),
    ]
    header = bytearray(b"BNCTRL3\x00")
    header += struct.pack(
        "<IIIIff", 3, hidden, rank, pooling_id,
        float(temperature), float(threshold))
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
    parser.add_argument("--train-examples", type=int, default=110)
    parser.add_argument("--valid-examples", type=int, default=28)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--pooling", choices=("last", "mean_last"), default="mean_last")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"),
        default="auto")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()

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
                offset=args.train_examples),
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
    hidden = train["write"].shape[1]
    train = move_tensors(train, device)
    valid = move_tensors(valid, device)
    model = AddressedController(hidden, args.rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)
    best_score = -1.0
    best_state = None
    for step in range(args.steps):
        indices = torch.randperm(
            train["write"].shape[0], generator=rng
        )[:min(args.batch, train["write"].shape[0])].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = addressed_loss(
            model, train, indices, args.temperature, args.action_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            report = metrics(model, valid)
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
    report = metrics(model, valid)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_bnctrl3(
        output, model, args.gguf, hidden, args.rank,
        args.temperature, args.pooling, report["address_threshold"])
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": hidden,
        "rank": args.rank,
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
