#!/usr/bin/env python3
"""Train a backbone-bound exact-copy pointer on the V89 curriculum.

The memory state retains original record tokens. The trainable model decides
which contiguous span answers the query; accepted spans are copied verbatim.
The frozen backbone uses full-prefix forwards without KV cache or LoRA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import time
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_data import CTokenDecoder, CTokenizer


def reconstruction_paths(data_dir):
    root = Path(data_dir)
    direct = root / "reconstruction.jsonl"
    if direct.is_file():
        return [direct]
    paths = list(root.glob("len*/reconstruction.jsonl"))
    paths.sort(key=lambda path: (
        int(path.parent.name[3:])
        if path.parent.name[3:].isdigit()
        else 1 << 30,
        str(path),
    ))
    if not paths:
        raise FileNotFoundError(
            f"no reconstruction.jsonl found under {root}")
    return paths


def locate_byte_span(record_ids, payload, decoder, max_span):
    """Locate payload bytes, including offsets inside boundary tokens."""
    pieces = [
        decoder.decode_bytes([token_id])
        for token_id in record_ids]
    record_bytes = b"".join(pieces)
    payload_bytes = payload.encode("utf-8")
    payload_start = record_bytes.rfind(payload_bytes)
    if payload_start < 0:
        return None
    payload_end = payload_start + len(payload_bytes)
    token_start = None
    token_end = None
    start_offset = None
    end_offset = None
    cursor = 0
    for index, piece in enumerate(pieces):
        next_cursor = cursor + len(piece)
        if token_start is None and cursor <= payload_start < next_cursor:
            token_start = index
            start_offset = payload_start - cursor
        if cursor < payload_end <= next_cursor:
            token_end = index
            end_offset = payload_end - cursor
            break
        cursor = next_cursor
    if (
        token_start is None
        or token_end is None
        or token_end - token_start + 1 > max_span
    ):
        return None
    if token_start == token_end:
        recovered = pieces[token_start][start_offset:end_offset]
    else:
        recovered = (
            pieces[token_start][start_offset:]
            + b"".join(pieces[token_start + 1:token_end])
            + pieces[token_end][:end_offset])
    if recovered != payload_bytes:
        return None
    byte_token_indices = []
    byte_token_offsets = []
    for token_index, piece in enumerate(pieces):
        byte_token_indices.extend([token_index] * len(piece))
        byte_token_offsets.extend(range(len(piece)))
    return {
        "start": token_start,
        "end": token_end,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "piece_bytes": pieces,
        "record_bytes": record_bytes,
        "payload_bytes": payload_bytes,
        "start_byte": payload_start,
        "end_byte": payload_end - 1,
        "byte_token_indices": byte_token_indices,
        "byte_token_offsets": byte_token_offsets,
    }


class BytePointer(nn.Module):
    def __init__(self, hidden, rank, max_token_bytes):
        super().__init__()
        self.max_token_bytes = max_token_bytes
        self.rank = rank
        self.token = nn.Linear(hidden, rank, bias=False)
        self.byte = nn.Embedding(257, rank)
        self.previous_byte = nn.Embedding(257, rank)
        self.next_byte = nn.Embedding(257, rank)
        self.offset = nn.Embedding(max_token_bytes, rank)
        self.start = nn.Linear(rank, 1)
        self.end = nn.Linear(rank, 1)
        self.inside = nn.Linear(rank, 1)

    def forward(
        self, token_hidden, byte_values, previous_bytes,
        next_bytes, token_offsets, mask,
    ):
        features = (
            self.token(token_hidden)
            + self.byte(byte_values)
            + self.previous_byte(previous_bytes)
            + self.next_byte(next_bytes)
            + self.offset(token_offsets))
        features = F.silu(features)
        start = self.start(features).squeeze(-1).masked_fill(
            ~mask, -1e9)
        end = self.end(features).squeeze(-1).masked_fill(
            ~mask, -1e9)
        inside = self.inside(features).squeeze(-1).masked_fill(
            ~mask, -1e9)
        return start, end, inside


def collate_byte_positions(rows, device):
    length = max(len(row["record_bytes"]) for row in rows)
    hidden_dim = rows[0]["hidden"].shape[1]
    token_hidden = torch.zeros(
        len(rows), length, hidden_dim,
        device=device, dtype=torch.float32)
    byte_values = torch.full(
        (len(rows), length), 256,
        device=device, dtype=torch.long)
    previous_bytes = torch.full_like(byte_values, 256)
    next_bytes = torch.full_like(byte_values, 256)
    token_offsets = torch.zeros_like(byte_values)
    mask = torch.zeros(
        len(rows), length, device=device, dtype=torch.bool)
    starts = torch.tensor(
        [row["start_byte"] for row in rows],
        device=device, dtype=torch.long)
    ends = torch.tensor(
        [row["end_byte"] for row in rows],
        device=device, dtype=torch.long)
    for row_index, row in enumerate(rows):
        count = len(row["record_bytes"])
        indices = torch.tensor(
            row["byte_token_indices"], dtype=torch.long)
        token_hidden[row_index, :count] = row["hidden"][indices].to(
            device=device, dtype=torch.float32)
        values = torch.tensor(
            list(row["record_bytes"]), device=device, dtype=torch.long)
        byte_values[row_index, :count] = values
        if count > 1:
            previous_bytes[row_index, 1:count] = values[:-1]
            next_bytes[row_index, :count - 1] = values[1:]
        offsets = torch.tensor(
            row["byte_token_offsets"],
            device=device, dtype=torch.long)
        token_offsets[row_index, :count] = offsets
        mask[row_index, :count] = True
    return (
        token_hidden, byte_values, previous_bytes, next_bytes,
        token_offsets, mask, starts, ends)


def select_best_byte_span(
    start_scores, end_scores, inside_scores, count, max_copy_bytes,
    inside_mode="sum",
):
    start_scores = start_scores[:count]
    end_scores = end_scores[:count]
    inside_scores = inside_scores[:count]
    prefix = torch.cat([
        torch.zeros(
            1, device=inside_scores.device, dtype=inside_scores.dtype),
        torch.cumsum(inside_scores, dim=0),
    ])
    span_inside = (
        prefix[1:].unsqueeze(0)
        - prefix[:-1].unsqueeze(1))
    positions = torch.arange(count, device=start_scores.device)
    starts = positions.unsqueeze(1)
    ends = positions.unsqueeze(0)
    lengths = (ends - starts + 1).clamp_min(1)
    if inside_mode == "mean":
        span_inside = span_inside / lengths
    elif inside_mode == "sqrt":
        span_inside = span_inside / torch.sqrt(lengths)
    elif inside_mode == "none":
        span_inside = torch.zeros_like(span_inside)
    elif inside_mode != "sum":
        raise ValueError(f"unsupported inside mode {inside_mode}")
    scores = (
        start_scores.unsqueeze(1)
        + end_scores.unsqueeze(0)
        + span_inside)
    valid = (
        (ends >= starts)
        & (ends - starts < max_copy_bytes))
    flat_index = int(scores.masked_fill(~valid, -torch.inf).argmax())
    return flat_index // count, flat_index % count


@torch.no_grad()
def evaluate_byte(
    model, rows, device, max_copy_bytes, inside_mode="sum",
):
    byte_exact = 0
    for row in rows:
        (
            token_hidden, byte_values, previous_bytes, next_bytes,
            token_offsets, mask, _starts, _ends,
        ) = collate_byte_positions([row], device)
        start_scores, end_scores, inside_scores = model(
            token_hidden, byte_values, previous_bytes,
            next_bytes, token_offsets, mask)
        count = len(row["record_bytes"])
        start, end = select_best_byte_span(
            start_scores[0], end_scores[0], inside_scores[0],
            count, max_copy_bytes, inside_mode)
        recovered = row["record_bytes"][start:end + 1]
        byte_exact += int(recovered == row["payload_bytes"])
    return {
        "examples": len(rows),
        "byte_exact": byte_exact / max(len(rows), 1),
    }


def export_byte_pointer(
    path, model, gguf, hidden, max_copy_bytes, threshold,
):
    payloads = [
        model.token.weight.detach().float().cpu().numpy().tobytes(),
        model.byte.weight.detach().float().cpu().numpy().tobytes(),
        model.previous_byte.weight.detach().float().cpu().numpy().tobytes(),
        model.next_byte.weight.detach().float().cpu().numpy().tobytes(),
        model.offset.weight.detach().float().cpu().numpy().tobytes(),
        model.start.weight.detach().float().cpu().numpy().tobytes(),
        model.start.bias.detach().float().cpu().numpy().tobytes(),
        model.end.weight.detach().float().cpu().numpy().tobytes(),
        model.end.bias.detach().float().cpu().numpy().tobytes(),
        model.inside.weight.detach().float().cpu().numpy().tobytes(),
        model.inside.bias.detach().float().cpu().numpy().tobytes(),
    ]
    header = bytearray(b"BNPTR5\x00\x00")
    header += struct.pack(
        "<IIIfII", 5, hidden, max_copy_bytes, float(threshold),
        model.rank, model.max_token_bytes)
    header += hashlib.sha256(Path(gguf).read_bytes()).digest()
    header += struct.pack("<I", len(payloads))
    for payload in payloads:
        header += struct.pack(
            "<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
    Path(path).write_bytes(header + b"".join(payloads))


@torch.inference_mode()
def encode_rows(
    backbone, tokenizer, decoder, data_dir, max_span,
    max_token_bytes, pointer_task, cache_path=None,
):
    if cache_path is not None and cache_path.exists():
        rows = torch.load(
            cache_path, map_location="cpu", weights_only=False)
        metadata = {}
        for data_path in reconstruction_paths(data_dir):
            for line in data_path.read_text().splitlines():
                if not line.strip():
                    continue
                sample = json.loads(line)
                metadata[sample["messages"][1][-1]["content"]] = (
                    sample.get("metadata", {}))
        for row in rows:
            item = metadata.get(row["payload"], {})
            row["pointer_operation"] = int(
                item.get("operation") == "update")
        return rows
    rows = []
    skipped = 0
    seen = 0
    paths = reconstruction_paths(data_dir)
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            seen += 1
            sample = json.loads(line)
            record = sample["messages"][0][0]["content"]
            question = sample["messages"][1][0]["content"]
            payload = sample["messages"][1][-1]["content"]
            if pointer_task == "write":
                prefix = (
                    "Extract the exact byte sequence that this request asks "
                    "the memory system to preserve.\nMemory request:\n")
            else:
                prefix = f"Question: {question}\nMemory record:\n"
            prefix_ids = tokenizer.encode(prefix, add_bos=True)
            record_ids = tokenizer.encode(record, add_bos=False)
            span = locate_byte_span(
                record_ids, payload, decoder, max_span)
            if (
                span is None
                or any(
                    len(piece) > max_token_bytes
                    for piece in span["piece_bytes"])
            ):
                skipped += 1
                continue
            hidden = backbone(
                torch.tensor(
                    prefix_ids + record_ids,
                    device=backbone.device, dtype=torch.long),
                return_hidden=True)
            rows.append({
                "hidden": hidden[len(prefix_ids):].to(
                    device="cpu", dtype=torch.float16),
                "answerable": True,
                "payload": payload,
                "payload_bytes": span["payload_bytes"],
                "piece_bytes": span["piece_bytes"],
                "record_bytes": span["record_bytes"],
                "start_byte": span["start_byte"],
                "end_byte": span["end_byte"],
                "byte_token_indices": span["byte_token_indices"],
                "byte_token_offsets": span["byte_token_offsets"],
                "record_ids": record_ids,
                "pointer_operation": int(
                    sample.get("metadata", {}).get("operation")
                    == "update"),
            })
            if seen % 100 == 0:
                print(json.dumps({
                    "phase": "encode",
                    "data": str(data_dir),
                    "files": len(paths),
                    "pointer_task": pointer_task,
                    "seen": seen,
                    "usable": len(rows),
                    "skipped": skipped,
                }, separators=(",", ":")), flush=True)
    if not rows:
        raise RuntimeError(f"no aligned copy spans in {data_dir}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rows, cache_path)
    print(json.dumps({
        "phase": "encoded",
        "data": str(data_dir),
        "pointer_task": pointer_task,
        "usable": len(rows),
        "skipped": skipped,
    }, separators=(",", ":")), flush=True)
    return rows


def train_epoch(model, rows, optimizer, device, batch_size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    losses = []
    model.train()
    for offset in range(0, len(order), batch_size):
        batch = [rows[index] for index in order[offset:offset + batch_size]]
        (
            token_hidden, byte_values, previous_bytes, next_bytes,
            token_offsets, mask, starts, ends,
        ) = collate_byte_positions(batch, device)
        start_scores, end_scores, inside_scores = model(
            token_hidden, byte_values, previous_bytes,
            next_bytes, token_offsets, mask)
        inside_targets = torch.zeros_like(
            inside_scores, dtype=torch.float32)
        for row_index, (start, end) in enumerate(
            zip(starts.tolist(), ends.tolist())
        ):
            inside_targets[row_index, start:end + 1] = 1.0
        positives = inside_targets[mask].sum()
        negatives = mask.sum() - positives
        positive_weight = torch.clamp(
            negatives / positives.clamp_min(1.0),
            min=1.0, max=32.0)
        loss = (
            F.cross_entropy(start_scores, starts)
            + F.cross_entropy(end_scores, ends)
            + F.binary_cross_entropy_with_logits(
                inside_scores[mask],
                inside_targets[mask],
                pos_weight=positive_weight))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / max(len(losses), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("train_data")
    parser.add_argument("valid_id")
    parser.add_argument("valid_ood")
    parser.add_argument("output")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"),
        default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--max-copy-bytes", type=int, default=128)
    parser.add_argument("--max-token-bytes", type=int, default=128)
    parser.add_argument("--pointer-rank", type=int, default=64)
    parser.add_argument(
        "--pointer-task", choices=("write", "query"), default="write")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--target-id-exact", type=float, default=0.99)
    parser.add_argument("--target-ood-exact", type=float, default=0.95)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch < 1
        or args.max_span < 1
        or args.max_copy_bytes < 1
        or args.max_token_bytes < 1
        or args.pointer_rank < 1
    ):
        parser.error(
            "epochs, batch, span, byte, and rank limits must be positive")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (
            "mps" if args.device == "auto"
            and torch.backends.mps.is_available()
            else ("cpu" if args.device == "auto" else args.device)))
    torch.manual_seed(args.seed)
    started = time.time()
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(
        weights, device=device, dtype=torch.bfloat16)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)
    cache_root = (
        Path(args.cache_dir) if args.cache_dir else None)

    def cache(name):
        return None if cache_root is None else cache_root / f"{name}.pt"

    train_rows = encode_rows(
        backbone, tokenizer, decoder, args.train_data,
        args.max_span, args.max_token_bytes,
        args.pointer_task, cache("train"))
    valid_id_rows = encode_rows(
        backbone, tokenizer, decoder, args.valid_id,
        args.max_span, args.max_token_bytes,
        args.pointer_task, cache("valid_id"))
    valid_ood_rows = encode_rows(
        backbone, tokenizer, decoder, args.valid_ood,
        args.max_span, args.max_token_bytes,
        args.pointer_task, cache("valid_ood"))
    hidden = train_rows[0]["hidden"].shape[1]
    del backbone
    weights.close()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    model = BytePointer(
        hidden, args.pointer_rank, args.max_token_bytes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    best_score = -1.0
    best_state = None
    for epoch in range(args.epochs):
        loss = train_epoch(
            model, train_rows, optimizer, device, args.batch, rng)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.eval()
            id_metrics = evaluate_byte(
                model, valid_id_rows, device, args.max_copy_bytes)
            ood_metrics = evaluate_byte(
                model, valid_ood_rows, device, args.max_copy_bytes)
            score = min(
                id_metrics["byte_exact"], ood_metrics["byte_exact"])
            print(json.dumps({
                "epoch": epoch + 1,
                "loss": loss,
                "id": id_metrics,
                "ood": ood_metrics,
                "selection_score": score,
            }, separators=(",", ":")), flush=True)
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
            if (
                id_metrics["byte_exact"] >= args.target_id_exact
                and ood_metrics["byte_exact"] >= args.target_ood_exact
            ):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_id = evaluate_byte(
        model, valid_id_rows, device, args.max_copy_bytes)
    final_ood = evaluate_byte(
        model, valid_ood_rows, device, args.max_copy_bytes)
    threshold = -1.0e9
    export_byte_pointer(
        args.output, model, args.gguf, hidden,
        args.max_copy_bytes, threshold)
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": hidden,
        "max_span": args.max_span,
        "max_copy_bytes": args.max_copy_bytes,
        "max_token_bytes": args.max_token_bytes,
        "pointer_rank": args.pointer_rank,
        "pointer_task": args.pointer_task,
        "format": "BNPTR5",
        "metrics": {"id": final_id, "ood": final_ood},
    }, args.output + ".pt")
    print(json.dumps({
        "result": "done",
        "output": args.output,
        "pointer_task": args.pointer_task,
        "best_score": best_score,
        "id": final_id,
        "ood": final_ood,
        "seconds": time.time() - started,
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
