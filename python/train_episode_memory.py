"""UNVALIDATED DRAFT: do not launch before neural-system/PLAN.md P1 is complete.

Learn entity/relation binding before letting memory influence generation.
"""
import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from episode_memory import EpisodeBindingReader, READER_FORMAT
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_memory_fusion import BACKBONE_SHA256, load_rows, encode_sources
from train_resident_identity import entity_role_labels
from typed_memory_training import clone_state_dict, file_fingerprint
from neural_memory_launch import require_episode_training_ready
from episode_memory_inputs import ReaderFeatures, reader_forward


@dataclass(frozen=True)
class BindingSupervision:
    query_roles: torch.Tensor
    memory_roles: tuple[torch.Tensor, ...]
    entity: torch.Tensor
    relation: torch.Tensor


@dataclass(frozen=True)
class BindingExample:
    case_id: str
    condition: str
    features: ReaderFeatures
    supervision: BindingSupervision


def labels(row):
    if row["expected_value"] is None:
        subject, relation = row["answer"][len("I don't know "):].rsplit("'s ", 1)
        relation = relation.removesuffix(" yet.")
    else:
        subject = row["answer"].split(" ", 1)[0]
        relation = "home city" if " lives in " in row["answer"] else "favorite drink"
    memory_subjects, memory_relations = [], []
    for text in row["memory"]:
        owner, rest = text.split("'s ", 1)
        predicate = rest.split(" is ", 1)[0]
        memory_subjects.append(owner); memory_relations.append(predicate)
    return subject, relation, memory_subjects, memory_relations


def relation_confounds(rows):
    """Training-only controlled negatives, absent from the fixed development file."""
    output = []
    for row in rows:
        if row["split"] != "train": raise ValueError("training-only augmentation")
        if row["condition"] != "original": continue
        subject, relation, _, _ = labels(row)
        changed = copy.deepcopy(row)
        other = "favorite drink" if relation == "home city" else "home city"
        value = "tea" if other == "favorite drink" else "Lima"
        changed["id"] += "-wrong-relation"; changed["condition"] = "relation_negative"
        changed["memory"] = [f"{subject}'s {other} is {value}."]
        changed["expected_value"] = None
        changed["answer"] = f"I don't know {subject}'s {relation} yet."
        output.append(changed)
    return output


def token_rows(rows, bank, tokenizer, device):
    result = []
    for row in rows:
        # The feature branch sees natural input only. No answer/role/target is
        # accepted by ReaderFeatures or reader_forward, including during loss.
        features = ReaderFeatures(bank[row["query"]].float().to(device).detach().clone(),
                                  tuple(bank[t].float().to(device).detach().clone() for t in row["memory"]))
        subject, relation, subjects, relations = labels(row)
        supervision = BindingSupervision(
            entity_role_labels(row["query"], [subject], tokenizer)[0][1:].to(device),
            tuple(entity_role_labels(text, [owner], tokenizer)[0][1:].to(device) for text, owner in zip(row["memory"], subjects)),
            torch.tensor([s == subject for s in subjects], device=device).float(),
            torch.tensor([r == relation for r in relations], device=device).float())
        result.append(BindingExample(row["id"], row["condition"], features, supervision))
    return result


def balanced_role_loss(logits, targets):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive, negative = targets > .5, targets <= .5
    return .5 * (loss[positive].mean() + loss[negative].mean())


def loss_for(reader, example):
    scores, parts = reader_forward(reader, example.features)
    labels = example.supervision
    loss = F.binary_cross_entropy_with_logits(scores, labels.entity * labels.relation)
    loss = loss + F.binary_cross_entropy_with_logits(parts["entity"], labels.entity)
    loss = loss + F.binary_cross_entropy_with_logits(parts["relation"], labels.relation)
    role = balanced_role_loss(parts["query_roles"], labels.query_roles)
    for logits, target in zip(parts["memory_roles"], labels.memory_roles):
        role = role + balanced_role_loss(logits, target)
    return loss + role / (1 + len(example.features.episodes))


@torch.no_grad()
def evaluate(reader, examples):
    reader.eval(); cases = []
    for x in examples:
        logits, _ = reader_forward(reader, x.features)
        actual = (logits > 0).nonzero().flatten().tolist()
        expected = (x.supervision.entity * x.supervision.relation > .5).nonzero().flatten().tolist()
        cases.append({"id": x.case_id, "condition": x.condition,
                      "actual": actual, "expected": expected, "correct": actual == expected,
                      "logits": logits.cpu().tolist()})
    groups = {kind: {"correct": sum(r["correct"] for r in cases if r["condition"] == kind),
                     "total": sum(r["condition"] == kind for r in cases)} for kind in sorted({r["condition"] for r in cases})}
    return {"correct": sum(r["correct"] for r in cases), "total": len(cases), "groups": groups, "cases": cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf"); p.add_argument("data"); p.add_argument("output")
    p.add_argument("--lib", required=True); p.add_argument("--tok-probe", required=True)
    p.add_argument("--steps", type=int, default=400); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=3180918)
    a = p.parse_args()
    try:
        require_episode_training_ready()
    except ValueError as exc:
        p.error(str(exc))
    if a.steps < 1 or a.batch < 1: p.error("positive limits required")
    if file_fingerprint(a.gguf) != BACKBONE_SHA256: p.error("exact 0.5B backbone required")
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    source = Path(a.data); train = load_rows(source / "train.jsonl", "train"); valid = load_rows(source / "valid.jsonl", "valid")
    train += relation_confounds(train)
    metadata = {"format": READER_FORMAT, "configuration": vars(a), "backbone_sha256": BACKBONE_SHA256,
                "train_sha256": file_fingerprint(source / "train.jsonl"), "valid_sha256": file_fingerprint(source / "valid.jsonl"),
                "locomo_used": False, "kv_cache_used": False, "lora_used": False, "training_only_role_labels": True,
                "oracle_episode_boundaries": True, "automatic_memory": False,
                "source_sha256": {name: file_fingerprint(Path(__file__).with_name(name)) for name in ("episode_memory.py", "train_episode_memory.py", "torch_backbone.py")}}
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    weights = GGUFWeights(a.gguf, a.lib); tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        backbone = TorchBackbone(weights, device=a.device, dtype=torch.bfloat16 if a.device.startswith("cuda") else torch.float32)
        weights.close(); backbone.eval()
        to_encode = [{"memory": r["memory"] + [r["query"]]} for r in train + valid]
        bank = encode_sources(to_encode, backbone, tokenizer)
        torch.save({**metadata, "sources": bank}, root / "features.pt")
        hidden = backbone.cfg.hidden; del backbone
        tx, vx = token_rows(train, bank, tokenizer, a.device), token_rows(valid, bank, tokenizer, a.device)
        reader = EpisodeBindingReader(hidden).to(a.device)
        optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-3, weight_decay=.01)
        pools = [[x for x in tx if x.condition == kind] for kind in sorted({r["condition"] for r in train})]
        best = None; state = None; chosen = 0
        for step in range(1, a.steps + 1):
            reader.train(); optimizer.zero_grad(set_to_none=True); total = 0.
            for i in range(a.batch):
                loss = loss_for(reader, rng.choice(pools[(step * a.batch + i) % len(pools)])) / a.batch
                if not torch.isfinite(loss): raise ValueError("nonfinite binding loss")
                loss.backward(); total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.); optimizer.step()
            if step == 1 or step % 25 == 0:
                print(json.dumps({"phase": "binding_train", "step": step, "loss": total}), flush=True)
            if step % 50 == 0 or step == a.steps:
                result = evaluate(reader, vx)
                key = (min(x["correct"] / x["total"] for x in result["groups"].values()), result["correct"])
                if best is None or key > best:
                    best, state, chosen = key, clone_state_dict(reader), step
                print(json.dumps({"phase": "binding_valid", "step": step, "correct": result["correct"], "groups": result["groups"]}), flush=True)
        reader.load_state_dict(state)
        torch.save({**metadata, "hidden": hidden, "selected_step": chosen, "state_dict": state}, root / "selected.pt")
        result = evaluate(reader, vx)
        result.update({**metadata, "selected_step": chosen, "selected_sha256": file_fingerprint(root / "selected.pt"),
                       "reader_gate_passed": min(x["correct"] / x["total"] for x in result["groups"].values()) >= .9})
        (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"phase": "binding_done", "correct": result["correct"], "total": result["total"], "reader_gate_passed": result["reader_gate_passed"]}), flush=True)
    finally:
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()


if __name__ == "__main__": main()
