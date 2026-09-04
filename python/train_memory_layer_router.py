#!/usr/bin/env python3
"""Train a question-conditioned router over memory retrieval layers."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from eval_bnmem_torch_locomo import (  # noqa: E402
    CTokenizer,
    GGUFWeights,
    TorchBackbone,
    forward_chunk,
    load_memory,
)
from train_data import render_chunk  # noqa: E402
from model_identity import require_matching_sha256, sha256_file  # noqa: E402


def load_rows(source, limit, seed):
    paths = (
        sorted(Path(source).glob("*.jsonl"))
        if Path(source).is_dir() else [Path(source)])
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def attention_stats(attention):
    scores = attention[-1].float()
    probabilities = scores / scores.sum().clamp_min(1e-8)
    top = probabilities.topk(k=min(4, probabilities.numel())).values
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
    entropy = entropy / max(math.log(max(probabilities.numel(), 2)), 1.0)
    margin = (
        top[0] - top[1]
        if top.numel() > 1 else top.new_zeros(()))
    return torch.stack((
        top[0],
        top.sum(),
        entropy,
        margin,
    ))


class LayerRouter(nn.Module):
    MAGIC = "BNROUTER1"

    def __init__(self, input_dim, layers, width=256, device="cuda",
                 backbone_sha256=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.layers = int(layers)
        self.width = int(width)
        self.backbone_sha256 = backbone_sha256
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, layers),
        ).to(device)
        self.register_buffer(
            "feature_mean", torch.zeros(input_dim, device=device))
        self.register_buffer(
            "feature_scale", torch.ones(input_dim, device=device))

    def forward(self, features):
        normalized = (
            features - self.feature_mean
        ) / self.feature_scale.clamp_min(1e-5)
        return self.network(normalized)

    def save(self, path):
        torch.save({
            "magic": self.MAGIC,
            "version": 2,
            "backbone_sha256": self.backbone_sha256,
            "input_dim": self.input_dim,
            "layers": self.layers,
            "width": self.width,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in self.state_dict().items()
            },
        }, path)

    @classmethod
    def load(cls, path, device="cuda", gguf_path=None):
        payload = torch.load(
            path, map_location="cpu", weights_only=True)
        if (
            payload.get("magic") != cls.MAGIC
            or payload.get("version") != 2
        ):
            raise ValueError("unsupported memory layer router")
        if gguf_path is None:
            raise ValueError(
                "GGUF path is required to validate memory layer router")
        require_matching_sha256(
            "memory layer router", payload.get("backbone_sha256"),
            gguf_path)
        router = cls(
            payload["input_dim"], payload["layers"],
            width=payload["width"], device=device,
            backbone_sha256=payload["backbone_sha256"])
        router.load_state_dict(payload["state_dict"], strict=True)
        router.eval()
        return router


@torch.inference_mode()
def extract_features(rows, backbone, memory, tokenizer):
    features, targets, masses = [], [], []
    for row_index, row in enumerate(rows):
        memory.reset_state()
        query_index = int(
            row.get("query_turn_id", len(row["messages"]) - 1))
        evidence = {
            int(index)
            for index in row.get("evidence_message_indices", [])}
        distractors = {
            int(index)
            for index in row.get("distractor_message_indices", [])}
        for chunk_index, chunk in enumerate(row["messages"]):
            if chunk_index == query_index:
                break
            text, _ = render_chunk(chunk, False)
            ids = tokenizer.encode(text, add_bos=True)
            forward_chunk(backbone, memory, ids)
            label = (
                1 if chunk_index in evidence
                else 0 if chunk_index in distractors
                else -1)
            memory.set_pending_slot_labels([label] * len(ids))
            memory.commit_all()

        prompt, _answer = render_chunk(
            row["messages"][query_index], True)
        ids = tokenizer.encode(prompt, add_bos=True)
        hidden = backbone(
            torch.tensor(ids, device="cuda"),
            memory_v6=memory, return_hidden=True)
        memory.discard_captured()
        layer_weights = memory.last_pointer_weights_by_layer
        if (
            not layer_weights
            or any(value is None for value in layer_weights)
            or memory.slot_labels.numel() == 0
        ):
            continue
        positive = memory.slot_labels > 0
        negative = memory.slot_labels == 0
        if not bool(positive.any()) or not bool(negative.any()):
            continue
        layer_masses = torch.stack([
            value[-1, positive].sum().float()
            for value in layer_weights])
        stats = torch.cat([
            attention_stats(value) for value in layer_weights])
        feature = torch.cat((hidden[-1].float(), stats))
        features.append(feature.cpu())
        targets.append(int(layer_masses.argmax().item()))
        masses.append(layer_masses.cpu())
        if (row_index + 1) % 50 == 0:
            print(
                f'{{"phase":"features","done":{row_index + 1},'
                f'"total":{len(rows)},"kept":{len(features)}}}',
                flush=True)
    return (
        torch.stack(features),
        torch.tensor(targets, dtype=torch.long),
        torch.stack(masses),
    )


def metrics(router, features, targets, masses):
    with torch.no_grad():
        logits = router(features.cuda())
        selected = logits.argmax(dim=-1).cpu()
    rows = torch.arange(selected.numel())
    selected_mass = masses[rows, selected]
    oracle_mass = masses.max(dim=-1).values
    layer27 = masses[:, min(27, masses.shape[1] - 1)]
    return {
        "samples": selected.numel(),
        "target_accuracy": float((selected == targets).float().mean()),
        "selected_evidence_mass": float(selected_mass.mean()),
        "oracle_evidence_mass": float(oracle_mass.mean()),
        "layer27_evidence_mass": float(layer27.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("memory_model")
    parser.add_argument("train_data")
    parser.add_argument("valid_data")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--valid-samples", type=int, default=390)
    parser.add_argument("--max-memory-slots", type=int, default=2048)
    parser.add_argument("--slot-temperature", type=float, default=0.07)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-temperature", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.target_temperature <= 0.0:
        parser.error("--target-temperature must be positive")

    torch.manual_seed(args.seed)
    backbone_sha256 = sha256_file(args.gguf)
    train_rows = load_rows(
        args.train_data, args.train_samples, args.seed)
    valid_rows = load_rows(
        args.valid_data, args.valid_samples, args.seed + 1)
    cache_path = Path(args.feature_cache) if args.feature_cache else None
    if cache_path is not None and cache_path.exists():
        cached = torch.load(
            cache_path, map_location="cpu", weights_only=True)
        require_matching_sha256(
            "memory router feature cache",
            cached.get("backbone_sha256"), args.gguf)
        train_x = cached["train_x"]
        train_y = cached["train_y"]
        train_masses = cached["train_masses"]
        valid_x = cached["valid_x"]
        valid_y = cached["valid_y"]
        valid_masses = cached["valid_masses"]
        print(json.dumps({
            "phase": "feature_cache_loaded",
            "path": str(cache_path),
            "train": train_x.shape[0],
            "valid": valid_x.shape[0],
        }), flush=True)
    else:
        weights = GGUFWeights(args.gguf, args.lib)
        backbone = TorchBackbone(
            weights, device="cuda", dtype=torch.bfloat16)
        memory = load_memory(
            args.memory_model, backbone, state_mode="slots",
            max_memory_slots=args.max_memory_slots,
            slot_temperature=args.slot_temperature)
        tokenizer = CTokenizer(args.tok_probe, args.gguf)

        print('{"phase":"extract_train"}', flush=True)
        train_x, train_y, train_masses = extract_features(
            train_rows, backbone, memory, tokenizer)
        print('{"phase":"extract_valid"}', flush=True)
        valid_x, valid_y, valid_masses = extract_features(
            valid_rows, backbone, memory, tokenizer)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "train_x": train_x,
                "train_y": train_y,
                "train_masses": train_masses,
                "valid_x": valid_x,
                "valid_y": valid_y,
                "valid_masses": valid_masses,
                "backbone_sha256": backbone_sha256,
            }, cache_path)
            print(json.dumps({
                "phase": "feature_cache_saved",
                "path": str(cache_path),
            }), flush=True)
        del memory, backbone, weights
        torch.cuda.empty_cache()
    router = LayerRouter(
        train_x.shape[1], train_masses.shape[1],
        width=args.width, device="cuda",
        backbone_sha256=backbone_sha256)
    with torch.no_grad():
        router.feature_mean.copy_(train_x.mean(dim=0).cuda())
        router.feature_scale.copy_(
            train_x.std(dim=0).clamp_min(1e-4).cuda())
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    best_selected_mass = -1.0
    for epoch in range(args.epochs):
        order = torch.randperm(
            train_x.shape[0], generator=generator)
        router.train()
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            logits = router(train_x[batch].cuda())
            target_distribution = nn.functional.softmax(
                train_masses[batch].cuda()
                / args.target_temperature,
                dim=-1)
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(logits, dim=-1),
                target_distribution,
                reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        router.eval()
        report = metrics(router, valid_x, valid_y, valid_masses)
        report.update({"epoch": epoch, "train_loss": float(loss.detach())})
        print(json.dumps(report), flush=True)
        if report["selected_evidence_mass"] > best_selected_mass:
            best_selected_mass = report["selected_evidence_mass"]
            router.save(args.output)
    print(json.dumps({
        "result": "done",
        "output": args.output,
        "train_features": train_x.shape[0],
        "valid_features": valid_x.shape[0],
        "best_selected_evidence_mass": best_selected_mass,
        "final": metrics(router, valid_x, valid_y, valid_masses),
    }, indent=2))


if __name__ == "__main__":
    main()
