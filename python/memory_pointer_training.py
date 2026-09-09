"""Shared training utilities for the backbone-bound memory span pointer."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import torch
import torch.nn as nn


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
            hidden, mask, _starts, _ends = collate([row], device)
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
    threshold = predictions[0][0] + 1.0 if predictions else 0.0
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
    payloads = [
        model.start.weight.detach().float().cpu().numpy().tobytes(),
        model.start.bias.detach().float().cpu().numpy().tobytes(),
        model.end.weight.detach().float().cpu().numpy().tobytes(),
        model.end.bias.detach().float().cpu().numpy().tobytes(),
        model.null.weight.detach().float().cpu().numpy().tobytes(),
        model.null.bias.detach().float().cpu().numpy().tobytes(),
    ]
    header = bytearray(b"BNPTR2\x00\x00")
    header += struct.pack(
        "<IIIf", 2, hidden, max_span, float(threshold))
    header += hashlib.sha256(Path(gguf).read_bytes()).digest()
    header += struct.pack("<I", len(payloads))
    for payload in payloads:
        header += struct.pack(
            "<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
    Path(path).write_bytes(header + b"".join(payloads))
