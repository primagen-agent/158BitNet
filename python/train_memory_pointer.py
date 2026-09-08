#!/usr/bin/env python3
"""Train a backbone-bound extractive start/end pointer for memory records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ggw import GGUFWeights  # noqa: E402
from prepare_locomo_retrieval_data import make_entries  # noqa: E402
from torch_backbone import TorchBackbone  # noqa: E402
from train_data import CTokenizer  # noqa: E402


class SpanPointer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)
        self.null = nn.Linear(hidden * 2, 2)

    def forward(self, hidden, mask):
        start = self.start(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        end = self.end(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        lengths = mask.sum(dim=1)
        mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / (
            lengths.unsqueeze(-1).clamp_min(1))
        last = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            lengths.clamp_min(1) - 1]
        null = self.null(torch.cat((mean, last), dim=-1))
        return start, end, null


def find_subsequence(source, target):
    for start in range(len(source) - len(target) + 1):
        if source[start:start + len(target)] == target:
            return start, start + len(target) - 1
    return None


def word_overlap(left, right):
    a = {word.lower() for word in left.split() if len(word) >= 3}
    b = {word.lower() for word in right.split() if len(word) >= 3}
    return len(a & b)


@torch.inference_mode()
def build_examples(backbone, tokenizer, samples, max_record_tokens):
    rows = []
    started = time.time()
    for conversation_index, sample in enumerate(samples):
        entry_list = make_entries(sample, False)
        entries = {entry["dia_id"]: entry for entry in entry_list}
        kept = 0
        for qa_index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) not in (1, 2, 3, 4):
                continue
            answer = str(qa.get("answer") or "").strip()
            if not answer:
                continue
            answer_ids = tokenizer.encode(answer, add_bos=False)
            if not answer_ids:
                continue
            positive_records = []
            evidence_ids = {str(value) for value in qa.get("evidence", [])}
            for evidence_id in evidence_ids:
                entry = entries.get(str(evidence_id))
                if entry is None:
                    continue
                record_ids = tokenizer.encode(
                    entry["text"], add_bos=False)[:max_record_tokens]
                span = find_subsequence(record_ids, answer_ids)
                if span is None:
                    continue
                positive_records.append((entry, record_ids, span))
            if not positive_records:
                continue
            prefix = (
                "Question: " + str(qa["question"]) +
                "\nMemory record:\n"
            )
            prefix_ids = tokenizer.encode(prefix, add_bos=True)
            for entry, record_ids, span in positive_records:
                ids = prefix_ids + record_ids
                hidden = backbone(
                    torch.tensor(ids, device=backbone.device),
                    return_hidden=True)
                rows.append({
                    "conversation": conversation_index,
                    "qa_index": qa_index,
                    "record_ids": record_ids,
                    "hidden": hidden[len(prefix_ids):].to(
                        dtype=torch.float16, device="cpu"),
                    "start": span[0],
                    "end": span[1],
                    "answerable": True,
                })
                kept += 1
            negative = max(
                (entry for entry in entry_list
                 if str(entry["dia_id"]) not in evidence_ids),
                key=lambda entry: word_overlap(
                    str(qa["question"]), entry["text"]),
                default=None)
            if negative is not None:
                record_ids = tokenizer.encode(
                    negative["text"], add_bos=False)[:max_record_tokens]
                ids = prefix_ids + record_ids
                hidden = backbone(
                    torch.tensor(ids, device=backbone.device),
                    return_hidden=True)
                rows.append({
                    "conversation": conversation_index,
                    "qa_index": qa_index,
                    "record_ids": record_ids,
                    "hidden": hidden[len(prefix_ids):].to(
                        dtype=torch.float16, device="cpu"),
                    "start": -1,
                    "end": -1,
                    "answerable": False,
                })
                kept += 1
        print(json.dumps({
            "phase": "pointer_features",
            "conversation": conversation_index,
            "examples": kept,
            "seconds": time.time() - started,
        }), flush=True)
    return rows


def collate(rows, device):
    length = max(row["hidden"].shape[0] for row in rows)
    hidden_dim = rows[0]["hidden"].shape[1]
    hidden = torch.zeros(
        len(rows), length, hidden_dim, dtype=torch.float32, device=device)
    mask = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    starts = torch.tensor([row["start"] for row in rows], device=device)
    ends = torch.tensor([row["end"] for row in rows], device=device)
    for index, row in enumerate(rows):
        count = row["hidden"].shape[0]
        hidden[index, :count] = row["hidden"].to(
            device=device, dtype=torch.float32)
        mask[index, :count] = True
    null_index = length
    starts = torch.where(starts >= 0, starts, null_index)
    ends = torch.where(ends >= 0, ends, null_index)
    return hidden, mask, starts, ends


def evaluate(model, rows, device, max_span):
    positive_exact = 0
    positives = 0
    predictions = []
    with torch.no_grad():
        for row in rows:
            hidden, mask, starts, ends = collate([row], device)
            start_scores, end_scores, null_scores = model(hidden, mask)
            candidates = []
            count = int(mask[0].sum())
            for start in range(count):
                for end in range(start, min(count, start + max_span)):
                    candidates.append((
                        float(start_scores[0, start] + end_scores[0, end]),
                        start, end))
            candidates.sort(reverse=True)
            best = candidates[0]
            margin = best[0] - float(null_scores[0].sum())
            exact = (
                row["answerable"] and
                best[1] == row["start"] and best[2] == row["end"])
            positive_exact += int(exact)
            positives += int(row["answerable"])
            predictions.append((margin, bool(row["answerable"]), exact))
    predictions.sort(reverse=True)
    answerable = 0
    span_correct = 0
    best_accepted = 0
    best_answerable = 0
    best_span_correct = 0
    threshold = (
        predictions[0][0] + 1.0 if predictions else 0.0)
    for index, (margin, is_answerable, exact) in enumerate(predictions, 1):
        answerable += int(is_answerable)
        span_correct += int(exact)
        if index >= 3 and answerable / index >= 0.8:
            best_accepted = index
            best_answerable = answerable
            best_span_correct = span_correct
            next_margin = (
                predictions[index][0]
                if index < len(predictions) else margin - 1e-5)
            threshold = (margin + next_margin) * 0.5
    return {
        "examples": len(rows),
        "positive_examples": positives,
        "span_exact": positive_exact / max(positives, 1),
        "accept_threshold": threshold,
        "accepted": best_accepted,
        "accepted_answerable": best_answerable,
        "accepted_span_correct": best_span_correct,
        "accept_precision": (
            best_answerable / best_accepted if best_accepted else 0.0),
        "accepted_span_precision": (
            best_span_correct / best_accepted if best_accepted else 0.0),
        "accept_coverage": (
            best_accepted / len(rows) if rows else 0.0),
    }


def export_pointer(path, model, gguf, hidden, max_span, threshold):
    start_w = model.start.weight.detach().float().cpu().numpy().tobytes()
    start_b = model.start.bias.detach().float().cpu().numpy().tobytes()
    end_w = model.end.weight.detach().float().cpu().numpy().tobytes()
    end_b = model.end.bias.detach().float().cpu().numpy().tobytes()
    null_w = model.null.weight.detach().float().cpu().numpy().tobytes()
    null_b = model.null.bias.detach().float().cpu().numpy().tobytes()
    payloads = [start_w, start_b, end_w, end_b, null_w, null_b]
    header = bytearray(b"BNPTR2\x00\x00")
    header += struct.pack(
        "<IIIf", 2, hidden, max_span, float(threshold))
    header += hashlib.sha256(Path(gguf).read_bytes()).digest()
    header += struct.pack("<I", len(payloads))
    for payload in payloads:
        header += struct.pack(
            "<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
    Path(path).write_bytes(header + b"".join(payloads))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("dataset")
    parser.add_argument("output_dir")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"),
                        default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-record-tokens", type=int, default=192)
    parser.add_argument("--max-span", type=int, default=32)
    parser.add_argument("--valid-conversations", default="0,3")
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu"))
    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cache = Path(args.cache)
    if cache.exists():
        rows = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        weights = GGUFWeights(args.gguf, args.lib)
        backbone = TorchBackbone(
            weights, device=device, dtype=torch.bfloat16)
        tokenizer = CTokenizer(args.tok_probe, args.gguf)
        rows = build_examples(
            backbone, tokenizer, samples, args.max_record_tokens)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rows, cache)
        del backbone

    valid_ids = {
        int(value) for value in args.valid_conversations.split(",")
        if value.strip()
    }
    train_rows = [row for row in rows
                  if row["conversation"] not in valid_ids]
    valid_rows = [row for row in rows
                  if row["conversation"] in valid_ids]
    hidden = rows[0]["hidden"].shape[1]
    model = SpanPointer(hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = -1.0
    best_state = None
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        losses = []
        for offset in range(0, len(train_rows), args.batch):
            batch = train_rows[offset:offset + args.batch]
            hidden_rows, mask, starts, ends = collate(batch, device)
            start_scores, end_scores, null_scores = model(hidden_rows, mask)
            start_scores = torch.cat(
                (start_scores, null_scores[:, 0:1]), dim=1)
            end_scores = torch.cat(
                (end_scores, null_scores[:, 1:2]), dim=1)
            loss = (
                F.cross_entropy(start_scores, starts) +
                F.cross_entropy(end_scores, ends))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 5 == 0:
            metrics = evaluate(model, valid_rows, device, args.max_span)
            print(json.dumps({
                "epoch": epoch,
                "loss": sum(losses) / max(len(losses), 1),
                **metrics,
            }), flush=True)
            selection_score = (
                metrics["accepted_span_correct"] +
                0.01 * metrics["span_exact"])
            if selection_score > best:
                best = selection_score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, valid_rows, device, args.max_span)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": hidden,
        "max_span": args.max_span,
        "valid_conversations": sorted(valid_ids),
        "metrics": metrics,
    }, output / "pointer.pt")
    export_pointer(
        output / "pointer.bnptr", model, args.gguf, hidden,
        args.max_span, metrics["accept_threshold"])
    print(json.dumps({"result": "done", **metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
