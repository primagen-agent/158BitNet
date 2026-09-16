#!/usr/bin/env python3
"""Train a high-precision IGNORE/WRITE gate for the existing BNCTRL4 runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import zlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from typed_memory_training import clone_state_dict, file_fingerprint, reject_evaluation_row

PREFIX = "Classify the request as one memory operation: ignore, write, update, or delete.\nRequest: "
SUFFIX = "\nMemory operation:"
NEGATIVES = {
    "train": ("Translate this sentence: {text}", "Rewrite this passage: {text}",
              "Is this statement true: {text}", "Suppose this were true: {text} What would happen?"),
    "valid": ("Proofread this excerpt: {text}", "Can you verify this claim: {text}",
              "Summarize this quotation: {text}", "In a fictional scenario, {text} What might happen?"),
}
SMALL_TALK = {
    "train": ("Hello.", "Thanks!", "Have a nice day!", "That sounds great.",
              "Please explain photosynthesis.", "Write a poem about rain."),
    "valid": ("Good morning!", "Much appreciated.", "Hope you have a good evening.",
              "That is interesting.", "Could you explain tides?", "Tell me a joke."),
}
# Training-only intent paraphrases. Development wording and held-out HTTP cases
# are not used here. The same fact is paired with opposite discourse intentions.
INTENT_NEGATIVES = (
    'Check the spelling in the following sample, without treating it as a fact: "{text}"',
    'The following is practice text for an editor. Suggest corrections: "{text}"',
    'Give a shorter version of this supplied text: "{text}"',
    'Extract the main idea from this passage for a writing exercise: "{text}"',
    'Render the following quotation in Spanish: "{text}"',
    'How would a French translator express this sample? "{text}"',
    'Make the wording more formal in this example: "{text}"',
    'Rephrase this line for a grammar lesson: "{text}"',
    'I do not know whether the following is accurate. How could I check it? "{text}"',
    'Someone sent an unverified allegation: "{text}". Is there evidence for it?',
    'Imagine an alternate world in which this held: "{text}". Discuss the consequences.',
    'For a made-up story only, assume the following: "{text}". Suggest a plot.',
    'This is not information to remember. It is only a sample sentence: "{text}"',
    'Keep this out of memory; I am testing text formatting: "{text}"',
    'Identify the verbs in this language-learning example: "{text}"',
    'For a reading-comprehension exercise, analyze this invented passage: "{text}"',
)
INTENT_POSITIVES = (
    'Keep the following information for future conversations: {text}',
    'Here is a real detail I want you to retain: {text}',
    'Save this fact, even though I have put it in quotation marks: "{text}"',
    'This is an actual update, not a writing example: {text}',
)
TASK_FACTS = (
    'I translate instruction manuals for a living.',
    'My sister edits scientific articles at work.',
    'I finished proofreading my dissertation this afternoon.',
    'I write fictional stories on weekends.',
    'I am learning to summarize research papers in my evening class.',
    'I verify insurance claims as part of my job.',
)


def cases(path, split, limit_per_domain=0, curriculum="basic"):
    if curriculum not in ("basic", "rich_intents"):
        raise ValueError("unsupported gate curriculum")
    rng = random.Random(2760916)
    facts, questions, seen_worlds = {}, [], set()
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        reject_evaluation_row(row)
        meta = row["metadata"]
        domain = meta.get("training_domain", "default")
        if f"-{split}-" not in meta["world_id"]:
            raise ValueError("gate source has wrong train/development split")
        if row.get("question"):
            questions.append({"text": row["question"], "label": 0, "domain": domain, "kind": "question"})
        if meta["world_id"] in seen_worlds:
            continue
        seen_worlds.add(meta["world_id"])
        messages = meta["raw_episodes"]
        if len(messages) != len(meta["typed_events"]):
            raise ValueError("gate positives must be annotated fact turns")
        for message in messages:
            facts.setdefault(domain, []).append(message)
    rows = list(questions)
    for domain, texts in facts.items():
        texts = sorted(set(texts)); rng.shuffle(texts)
        if limit_per_domain:
            texts = texts[:limit_per_domain]
        for index, text in enumerate(texts):
            rows.append({"text": text, "label": 1, "domain": domain,
                         "kind": "mixed" if text.rstrip().endswith("?") else "statement"})
            if index % 2 == 0 and not text.rstrip().endswith("?"):
                tail = " What about you?" if split == "train" else " Any news from your side?"
                rows.append({"text": text + tail, "label": 1, "domain": domain, "kind": "mixed"})
            rows.append({"text": NEGATIVES[split][index % len(NEGATIVES[split])].format(text=text),
                         "label": 0, "domain": domain, "kind": "non_assertion"})
            if split == "train" and curriculum == "rich_intents":
                # Three extra negative contexts per fact, not more fact copies.
                for offset in (0, 5, 11):
                    template = INTENT_NEGATIVES[(index + offset) % len(INTENT_NEGATIVES)]
                    rows.append({"text": template.format(text=text), "label": 0,
                                 "domain": domain, "kind": "non_assertion"})
                rows.append({"text": INTENT_POSITIVES[index % len(INTENT_POSITIVES)].format(text=text),
                             "label": 1, "domain": domain, "kind": "statement"})
    rows += [{"text": text, "label": 0, "domain": "general", "kind": "small_talk"} for text in SMALL_TALK[split]]
    if split == "train" and curriculum == "rich_intents":
        # Task vocabulary must not become an IGNORE keyword shortcut.
        rows += [{"text": text, "label": 1, "domain": "general", "kind": "statement"} for text in TASK_FACTS]
    unique = {}
    for row in rows:
        if row["text"] in unique and unique[row["text"]]["label"] != row["label"]:
            raise ValueError("conflicting gate labels")
        unique.setdefault(row["text"], row)
    return list(unique.values())


def disjoint_cases(train, valid):
    forbidden = {row["text"] for row in valid}
    return [row for row in train if row["text"] not in forbidden]


def pool_features(hidden, pooling="mean_last"):
    if pooling == "last":
        return hidden[-1].float()
    # Matches bitnet_get_last_pooled_hidden: all tokens, including BOS, plus last.
    return hidden.float().mean(dim=0) + hidden[-1].float()


@torch.inference_mode()
def encode(rows, backbone, tokenizer, pooling="mean_last"):
    features, token_rows = [], []
    for index, row in enumerate(rows):
        ids = tokenizer.encode(PREFIX + row["text"] + SUFFIX, True)
        if not 0 < len(ids) <= 512:
            raise ValueError("gate input exceeds runtime token capacity")
        hidden = backbone(torch.tensor(ids, device=backbone.device), return_hidden=True)
        features.append(pool_features(hidden, pooling).cpu())
        token_rows.append(ids)
        if (index + 1) % 200 == 0:
            print(json.dumps({"phase": "gate_features", "done": index + 1, "total": len(rows)}), flush=True)
    return torch.stack(features), token_rows


@torch.inference_mode()
def evaluate(head, x, rows):
    prediction = head(F.normalize(x, dim=-1)).argmax(-1).cpu().tolist()
    groups = {}
    for row, predicted in zip(rows, prediction):
        key = row["domain"] + "/" + row["kind"]
        group = groups.setdefault(key, {"label": row["label"], "total": 0, "correct": 0})
        group["total"] += 1
        group["correct"] += predicted == row["label"]
    positives = [g for g in groups.values() if g["label"] == 1]
    negatives = [g for g in groups.values() if g["label"] == 0]
    return {"groups": groups, "false_writes": sum(g["total"] - g["correct"] for g in negatives),
            "negative_total": sum(g["total"] for g in negatives),
            "write_correct": sum(g["correct"] for g in positives),
            "write_total": sum(g["total"] for g in positives),
            "worst_positive_recall": min(g["correct"] / g["total"] for g in positives)}


def selection_key(metrics):
    return -metrics["false_writes"], metrics["worst_positive_recall"], metrics["write_correct"]


def float_bytes(tensor):
    return tensor.detach().cpu().numpy().astype("<f4").tobytes()


def export_controller(head, hidden, rank, backbone_sha, path, pooling=2):
    state = head.state_dict()
    output = torch.zeros(4, rank)
    output[:2] = state["2.weight"].cpu()
    bias = torch.full((4,), -1e9)
    bias[:2] = state["2.bias"].cpu()
    tensors = [torch.zeros(1, hidden), torch.zeros(1, hidden),
               state["0.weight"], state["0.bias"], output, bias]
    payloads = [float_bytes(t) for t in tensors]
    if pooling not in (1, 2):
        raise ValueError("unsupported gate pooling")
    header = struct.pack("<8sIIIIffI", b"BNCTRL4\0", 4, hidden, 1, pooling, 1., .8, rank)
    header += bytes.fromhex(backbone_sha)
    header += struct.pack("<6I", *(zlib.crc32(p) & 0xffffffff for p in payloads))
    Path(path).write_bytes(header + b"".join(payloads))


@torch.inference_mode()
def export_parity(head, features, tokens, rows, path):
    logits = head(F.normalize(features.to(next(head.parameters()).device), dim=-1)).cpu()
    payload = bytearray(struct.pack("<8sII", b"BGATEP01", features.shape[1], len(rows)))
    for row, vector, ids, scores in zip(rows, features, tokens, logits):
        action = int(scores.argmax())
        margin = float((scores[0] - scores[1]).abs())
        payload += struct.pack("<IIIf", len(ids), row["label"], action, margin)
        payload += np.asarray(ids, dtype="<i4").tobytes() + float_bytes(vector)
    Path(path).write_bytes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "data", "output"):
        parser.add_argument(name)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--pooling", choices=("mean_last", "last"), default="mean_last")
    parser.add_argument("--curriculum", choices=("basic", "rich_intents"), default="basic")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.rank <= 1024:
        parser.error("invalid steps or rank")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    train_path, valid_path = Path(args.data) / "train.jsonl", Path(args.data) / "valid.jsonl"
    train = cases(train_path, "train", 256, args.curriculum)
    valid = cases(valid_path, "valid")
    before = len(train)
    train = disjoint_cases(train, valid)
    for name, rows in (("train", train), ("valid", valid)):
        (root / (name + ".json")).write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"phase": "gate_data", "train": len(train), "valid": len(valid),
                      "cross_split_duplicates_removed": before - len(train)}), flush=True)
    torch.manual_seed(2760916)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(weights, device=args.device, dtype=torch.float32).eval()
    tx, _ = encode(train, backbone, tokenizer, args.pooling)
    vx, valid_tokens = encode(valid, backbone, tokenizer, args.pooling)
    hidden = tx.shape[1]
    del backbone
    weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    torch.save({"train": tx, "valid": vx, "valid_tokens": valid_tokens}, root / "features.pt")
    tx, vx = tx.to(args.device), vx.to(args.device)
    head = nn.Sequential(nn.Linear(hidden, args.rank), nn.SiLU(), nn.Linear(args.rank, 2)).to(args.device)
    with torch.no_grad():
        head[2].weight.zero_(); head[2].bias.copy_(torch.tensor([1., -1.], device=args.device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    y = torch.tensor([r["label"] for r in train], device=args.device)
    # Balance each domain/kind, so small-talk negatives cannot be swamped by facts.
    groups = sorted({(r["domain"], r["kind"]) for r in train})
    pools = [torch.tensor([i for i, r in enumerate(train) if (r["domain"], r["kind"]) == g], device=args.device) for g in groups]
    best = evaluate(head, vx, valid); best_state = clone_state_dict(head); selected = 0
    for step in range(1, args.steps + 1):
        indices = torch.cat([pool[torch.randint(len(pool), (16,), device=args.device)] for pool in pools])
        loss = F.cross_entropy(head(F.normalize(tx[indices], dim=-1)), y[indices])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 50 == 0 or step == args.steps:
            metrics = evaluate(head, vx, valid)
            if selection_key(metrics) > selection_key(best):
                best, best_state, selected = metrics, clone_state_dict(head), step
            print(json.dumps({"phase": "gate_valid", "step": step, **metrics,
                              "train_metrics": evaluate(head, tx, train)}), flush=True)
    # Keep the final, possibly unsafe model's errors for diagnosis. This is not
    # the selected deployable head, and it never overwrites selection metrics.
    with torch.inference_mode():
        margins = (head(F.normalize(vx, dim=-1))[:, 1] - head(F.normalize(vx, dim=-1))[:, 0]).cpu().tolist()
    (root / "final_diagnostics.json").write_text(json.dumps({
        "step": args.steps, "not_deployable": True,
        "train_metrics": evaluate(head, tx, train), "valid_metrics": evaluate(head, vx, valid),
        "valid_predictions": [{**row, "write_margin": margin, "predicted": int(margin > 0)}
                              for row, margin in zip(valid, margins)],
    }, indent=2) + "\n")
    head.load_state_dict(best_state)
    sha = file_fingerprint(args.gguf)
    export_controller(head, hidden, args.rank, sha, root / "gate.bnctrl", 1 if args.pooling == "last" else 2)
    export_parity(head, vx.cpu(), valid_tokens, valid, root / "valid.parity")
    torch.save({"format": "MEMORY_WRITE_GATE_V1", "state_dict": best_state, "hidden": hidden,
        "rank": args.rank, "selected_step": selected, "valid_metrics": best, "backbone_sha256": sha,
        "train_source_sha256": file_fingerprint(train_path), "valid_source_sha256": file_fingerprint(valid_path),
        "prefix": PREFIX, "suffix": SUFFIX, "pooling": args.pooling,
        "training_config": vars(args), "kv_reuse": False, "lora": False, "locomo_used": False}, root / "gate.pt")
    summary = {"phase": "gate_done", "selected_step": selected, **best, "backbone_sha256": sha,
               "gate_sha256": file_fingerprint(root / "gate.bnctrl"), "automatic_chat_validated": False,
               "pooling": args.pooling,
               "curriculum": args.curriculum, "train_count": len(train), "valid_count": len(valid),
               "train_cases_sha256": file_fingerprint(root / "train.json"),
               "valid_cases_sha256": file_fingerprint(root / "valid.json"),
               "eligible_for_c_validation": selected > 0 and best["false_writes"] == 0 and best["worst_positive_recall"] >= .8}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
