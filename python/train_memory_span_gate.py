#!/usr/bin/env python3
"""Research-only token-role gate; no C export until development acceptance.

Roles: context=0, asserted fact=1, task/quoted material=2. Byte spans are
synthetic training annotations, never inputs to the inference network.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from prepare_dialogue_memory_curriculum import FILLERS
from train_memory_gate import (PREFIX, SUFFIX, TASK_FACTS, INTENT_NEGATIVES, INTENT_POSITIVES,
                               NEGATIVES, cases, disjoint_cases, selection_key)
from typed_memory_training import clone_state_dict, file_fingerprint, reject_evaluation_row


def source_segments(text, domain):
    """Gold regions from the existing synthetic generator, not a serving rule."""
    end = len(text)
    tail = " How has your week been?"
    if text.endswith(tail):
        end -= len(tail)
    if domain != "dialogue":
        return [(0, end)]
    speaker_end = text.find(": ") + 2
    if speaker_end < 2:
        raise ValueError("expected synthetic dialogue speaker prefix")
    start = speaker_end
    for filler in FILLERS:
        if filler and text.startswith(filler, start):
            start += len(filler)
            break
    return [(0, speaker_end - 2), (start, end)]


def annotate_cases(rows, source_path):
    sources = {}
    for line in Path(source_path).read_text().splitlines():
        row = json.loads(line)
        reject_evaluation_row(row)
        meta = row["metadata"]
        for text in meta["raw_episodes"]:
            segments = source_segments(text, meta.get("training_domain", "default"))
            if text in sources and sources[text] != segments:
                raise ValueError("conflicting source annotations")
            sources[text] = segments
    # Longest match ensures a mixed fact+question turn wins over its substring.
    ordered = sorted(sources, key=len, reverse=True)
    annotated = []
    for row in rows:
        text = row["text"]
        match = next((s for s in ordered if s in text), None)
        regions = []
        if match is not None:
            start = text.find(match)
            if text.find(match, start + 1) >= 0:
                raise ValueError("ambiguous repeated source in gate input")
            regions = [(start + a, start + b) for a, b in sources[match]]
        elif row["label"]:
            if text not in TASK_FACTS:
                raise ValueError("positive gate input has no gold fact region")
            regions = [(0, len(text))]
        segments = [{"start": len(text[:a].encode("utf-8")),
                     "end": len(text[:b].encode("utf-8")),
                     "role": 1 if row["label"] else 2} for a, b in regions if b > a]
        annotated.append({**row, "segments": segments})
    return annotated


def balance_quote_format(rows):
    """Training-only rendering pairs; unframed positive statements stay intact."""
    templates = [(t, 0) for t in (*NEGATIVES["train"], *INTENT_NEGATIVES)]
    templates += [(t, 1) for t in INTENT_POSITIVES]
    result = {r["text"]: r for r in rows}
    for row in rows:
        if not row["segments"]: continue
        for template, label in templates:
            if row["label"] != label: continue
            prefix, suffix = template.split("{text}")
            text = row["text"]
            if not text.startswith(prefix) or not text.endswith(suffix): continue
            end = len(text) - len(suffix) if suffix else len(text)
            payload = text[len(prefix):end]
            if prefix.endswith('"') and suffix.startswith('"'):
                changed = prefix[:-1] + payload + suffix[1:]; delta = -1
            elif payload.startswith('"') and payload.endswith('"'):
                changed = prefix + payload[1:-1] + suffix; delta = -1
            else:
                changed = prefix + '"' + payload + '"' + suffix; delta = 1
            variant = {**row, "text": changed,
                       "segments": [{**s, "start": s["start"] + delta, "end": s["end"] + delta}
                                    for s in row["segments"]]}
            if changed in result and result[changed] != variant:
                raise ValueError("conflicting quote-format supervision")
            result.setdefault(changed, variant)
            break
    return list(result.values())


def load_feature_cache(path, backbone_sha):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["backbone_sha256"] != backbone_sha:
        raise ValueError("cached backbone identity mismatch")
    if saved["prefix"] != PREFIX or saved["suffix"] != SUFFIX:
        raise ValueError("cached task prompt mismatch")
    result = {}
    for split in ("train", "valid"):
        source = path.parent / (split + ".json")
        if file_fingerprint(source) != saved[split + "_sha256"]:
            raise ValueError("cached source identity mismatch")
        rows = json.loads(source.read_text())
        if len(rows) != len(saved[split]): raise ValueError("cache length mismatch")
        for row, feature in zip(rows, saved[split]):
            result[json.dumps(row, sort_keys=True)] = feature
    return result


def align_roles(row, ids, tokenizer):
    """Align exact runtime bytes; no separately tokenized substring matching."""
    expected = (PREFIX + row["text"] + SUFFIX).encode("utf-8")
    pieces = list(tokenizer.decode_pieces(ids))
    if ids and ids[0] == tokenizer.bos():
        pieces[0] = b""
    decoded = b"".join(pieces)
    if decoded == expected:
        shift = 0
    elif decoded == b" " + expected:
        shift = 1
    else:
        raise ValueError("runtime decoded bytes do not match gate input")
    begin = shift + len(PREFIX.encode("utf-8"))
    end = begin + len(row["text"].encode("utf-8"))
    for s in row["segments"]:
        if not 0 <= s["start"] < s["end"] <= end - begin or s["role"] not in (1, 2):
            raise ValueError("invalid gold byte span")
    mask, labels, offset = [], [], 0
    for piece in pieces:
        stop = offset + len(piece)
        inside = stop > offset and stop > begin and offset < end
        roles = {s["role"] for s in row["segments"]
                 if stop > offset and stop > begin + s["start"] and offset < begin + s["end"]}
        if len(roles) > 1:
            raise ValueError("token crosses conflicting annotation roles")
        mask.append(inside)
        labels.append(next(iter(roles), 0) if inside else -100)
        offset = stop
    if not any(mask) or (row["label"] and 1 not in labels):
        raise ValueError("empty input/fact supervision")
    return torch.tensor(mask), torch.tensor(labels)


@torch.inference_mode()
def encode(rows, backbone, tokenizer, cache=None):
    encoded = []
    reused = 0
    for index, row in enumerate(rows):
        key = json.dumps(row, sort_keys=True)
        if cache is not None and key in cache:
            encoded.append(cache[key]); reused += 1
            continue
        ids = tokenizer.encode(PREFIX + row["text"] + SUFFIX, True)
        if not 0 < len(ids) <= 512:
            raise ValueError("token limit exceeded; truncation is not allowed")
        mask, labels = align_roles(row, ids, tokenizer)
        h = backbone(torch.tensor(ids, device=backbone.device), return_hidden=True)
        encoded.append({"x": F.normalize(h.float(), dim=-1).cpu().half(),
                        "mask": mask, "roles": labels, "ids": ids})
        if (index + 1) % 200 == 0:
            print(json.dumps({"phase": "span_features", "done": index + 1, "total": len(rows)}), flush=True)
    print(json.dumps({"phase": "span_feature_cache", "reused": reused, "total": len(rows)}), flush=True)
    return encoded


def batch(examples, indices, device):
    selected = [examples[i] for i in indices]
    width = selected[0]["x"].shape[-1]
    length = max(len(e["x"]) for e in selected)
    x = torch.zeros(len(selected), length, width, device=device)
    valid = torch.zeros(len(selected), length, dtype=torch.bool, device=device)
    mask = torch.zeros_like(valid)
    roles = torch.full(valid.shape, -100, dtype=torch.long, device=device)
    for i, e in enumerate(selected):
        n = len(e["x"])
        x[i, :n] = e["x"].to(device)
        valid[i, :n] = True
        mask[i, :n] = e["mask"].to(device)
        roles[i, :n] = e["roles"].to(device)
    return x, valid, mask, roles


class SpanGate(nn.Module):
    def __init__(self, hidden, width=128):
        super().__init__()
        self.project = nn.Linear(hidden, width)
        self.context = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=.1,
                                                 batch_first=True, norm_first=True)
        self.roles = nn.Linear(width, 3)
        nn.init.zeros_(self.roles.weight)
        with torch.no_grad():
            self.roles.bias.copy_(torch.tensor([1., -1., 0.]))

    def forward(self, x, valid, mask):
        h = F.silu(self.project(x))
        h = self.context(h, src_key_padding_mask=~valid)
        logits = self.roles(h)
        # Write only if asserted-token evidence outweighs context/material.
        evidence = logits[..., 1] - torch.logsumexp(logits[..., (0, 2)], dim=-1)
        evidence = evidence.masked_fill(~mask, -torch.inf)
        margin = torch.logsumexp(evidence, dim=1) - mask.sum(1).float().log()
        return margin, logits


def role_loss(logits, labels):
    # Equal importance to each present role, not dominated by wrapper tokens.
    losses = [F.cross_entropy(logits[labels == role], labels[labels == role])
              for role in range(3) if (labels == role).any()]
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(model, features, rows, device, diagnostics=False):
    model.eval()
    groups, details = {}, []
    tp = fp = fn = negative_fact_tokens = 0
    for start in range(0, len(rows), 64):
        stop = min(start + 64, len(rows))
        x, valid, mask, labels = batch(features, range(start, stop), device)
        margins, logits = model(x, valid, mask)
        predicted_roles = logits.argmax(-1)
        asserted = (predicted_roles == 1) & mask
        gold = labels == 1
        tp += int((asserted & gold).sum()); fp += int((asserted & ~gold).sum()); fn += int((~asserted & gold).sum())
        for i, row in enumerate(rows[start:stop]):
            margin = float(margins[i]); predicted = int(margin > 0)
            group = groups.setdefault(row["domain"] + "/" + row["kind"],
                                      {"label": row["label"], "total": 0, "correct": 0})
            group["total"] += 1; group["correct"] += predicted == row["label"]
            if not row["label"]:
                negative_fact_tokens += int(asserted[i].sum())
            if diagnostics:
                details.append({**row, "predicted": predicted, "write_margin": margin,
                                "predicted_roles": predicted_roles[i, mask[i]].cpu().tolist(),
                                "gold_roles": labels[i, mask[i]].cpu().tolist()})
    pos = [g for g in groups.values() if g["label"]]
    neg = [g for g in groups.values() if not g["label"]]
    metrics = {"groups": groups, "false_writes": sum(g["total"] - g["correct"] for g in neg),
               "negative_total": sum(g["total"] for g in neg),
               "write_correct": sum(g["correct"] for g in pos), "write_total": sum(g["total"] for g in pos),
               "worst_positive_recall": min(g["correct"] / g["total"] for g in pos),
               "asserted_token_f1": 2 * tp / max(1, 2 * tp + fp + fn),
               "negative_asserted_tokens": negative_fact_tokens}
    return metrics, details


def train_one(tx, vx, train, valid, args, weight, root, binding):
    torch.manual_seed(args.seed)
    model = SpanGate(tx[0]["x"].shape[-1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    groups = sorted({(r["domain"], r["kind"]) for r in train})
    pools = [torch.tensor([i for i, r in enumerate(train) if (r["domain"], r["kind"]) == g]) for g in groups]
    best, _ = evaluate(model, vx, valid, args.device)
    state, selected = clone_state_dict(model), 0
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.cat([p[torch.randint(len(p), (4,))] for p in pools]).tolist()
        x, real, mask, roles = batch(tx, indices, args.device)
        y = torch.tensor([train[i]["label"] for i in indices], device=args.device, dtype=torch.float32)
        margin, logits = model(x, real, mask)
        loss = F.binary_cross_entropy_with_logits(margin, y) + weight * role_loss(logits, roles)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite span-gate loss")
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            metrics, _ = evaluate(model, vx, valid, args.device)
            if selection_key(metrics) > selection_key(best):
                best, state, selected = metrics, clone_state_dict(model), step
            print(json.dumps({"phase": "span_valid", "role_weight": weight, "step": step, **metrics}), flush=True)
    final, details = evaluate(model, vx, valid, args.device, True)
    train_metrics, _ = evaluate(model, tx, train, args.device)
    # Preserve the nontrivial trained head for future diagnosis, never promotion.
    torch.save({**binding, "format": "MEMORY_SPAN_GATE_RESEARCH_V1", "state_dict": clone_state_dict(model),
                "step": args.steps, "role_weight": weight, "not_deployable": True}, root / "final_research.pt")
    (root / "final_diagnostics.json").write_text(json.dumps({"train_metrics": train_metrics,
        "valid_metrics": final, "valid_predictions": details}, indent=2) + "\n")
    model.load_state_dict(state)
    torch.save({**binding, "format": "MEMORY_SPAN_GATE_RESEARCH_V1", "state_dict": state,
                "selected_step": selected, "role_weight": weight, "not_deployable": True}, root / "selected_research.pt")
    summary = {"selected_step": selected, "selected_metrics": best, "final_metrics": final,
               "train_metrics": train_metrics, "role_weight": weight,
               "eligible_for_c_implementation": selected > 0 and best["false_writes"] == 0
                   and best["worst_positive_recall"] >= .8 and best["asserted_token_f1"] >= .8,
               "c_validated": False, "deployment_changed": False}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"phase": "span_done", **summary}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "data", "output"):
        parser.add_argument(name)
    parser.add_argument("--lib", required=True); parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2790916)
    parser.add_argument("--balance-quote-format", action="store_true")
    parser.add_argument("--feature-cache")
    args = parser.parse_args()
    if args.steps < 1: parser.error("steps must be positive")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    data = Path(args.data)
    valid_base = cases(data / "valid.jsonl", "valid")
    train_base = disjoint_cases(cases(data / "train.jsonl", "train", 256, "rich_intents"), valid_base)
    train = annotate_cases(train_base, data / "train.jsonl")
    valid = annotate_cases(valid_base, data / "valid.jsonl")
    if args.balance_quote_format:
        train = disjoint_cases(balance_quote_format(train), valid)
    for name, rows in (("train", train), ("valid", valid), ("valid_base", valid_base)):
        (root / (name + ".json")).write_text(json.dumps(rows, indent=2) + "\n")
    binding = {"backbone_sha256": file_fingerprint(args.gguf), "train_sha256": file_fingerprint(root / "train.json"),
               "valid_sha256": file_fingerprint(root / "valid.json"), "valid_base_sha256": file_fingerprint(root / "valid_base.json"),
               "configuration": vars(args), "kv_reuse": False, "lora": False, "locomo_used": False,
               "train_count": len(train), "valid_count": len(valid), "prefix": PREFIX, "suffix": SUFFIX}
    print(json.dumps({"phase": "span_data", **binding}), flush=True)
    cache = load_feature_cache(args.feature_cache, binding["backbone_sha256"]) if args.feature_cache else None
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(weights, device=args.device, dtype=torch.float32).eval()
    try:
        tx = encode(train, backbone, tokenizer, cache); vx = encode(valid, backbone, tokenizer, cache)
    finally:
        del backbone; weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
    torch.save({**binding, "train": tx, "valid": vx}, root / "features.pt")
    if args.device == "cuda": torch.cuda.empty_cache()
    results = {}
    for name, weight in (("classification_only", 0.), ("joint_roles", 1.)):
        run = root / name; run.mkdir()
        results[name] = train_one(tx, vx, train, valid, args, weight, run, binding)
    (root / "comparison.json").write_text(json.dumps({**binding, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
