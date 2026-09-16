#!/usr/bin/env python3
"""Calibrate predecessor existence while preserving the deployed pair encoder."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from export_typed_pair_parity_sample import load_typed_pair_model
from export_typed_pair_encoder import export_typed_pair_encoder_binary
from export_typed_link_parity_sample import export_typed_link_parity_sample
from train_typed_memory_writer import load_writer_examples, load_token_embedding_table
from train_typed_pair_verifier import build_active_pairs, score_pairs, TypedPairVerifier, export_typed_link_binary
from typed_memory_training import file_fingerprint, clone_state_dict


@torch.inference_mode()
def compile_sets(model, examples, device):
    pairs = build_active_pairs(examples)
    _, _, scores = score_pairs(model, examples, pairs, device, 64)
    groups = collections.defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[pair["left"]].append(i)
    features, labels = [], []
    def add(values, positive):
        if values.numel():
            features.append(TypedPairVerifier.set_link_features(values, 7))
            labels.append(float(positive))
    for indices in groups.values():
        values = scores[indices]
        correct = torch.tensor([pairs[i]["joint_same"] for i in indices], dtype=torch.bool)
        add(values, bool(correct.any()))
        # Candidate-count one must distinguish a match from an unrelated item.
        add(values[correct], True)
        add(values[~correct], False)
        for value in values[~correct]:
            add(value.reshape(1), False)
    return torch.stack(features), torch.tensor(labels)


@torch.inference_mode()
def evaluate(head, x, y):
    predicted = head(x).squeeze(-1) > 0
    positive = (predicted[y > 0.5]).float().mean().item()
    negative = (~predicted[y < 0.5]).float().mean().item()
    return {"accept": positive, "reject": negative, "balanced": (positive + negative) / 2}


def main():
    p = argparse.ArgumentParser()
    for name in ("pair_checkpoint", "pair_binary", "gguf", "train_features", "train_jsonl",
                 "valid_features", "valid_jsonl", "output"):
        p.add_argument(name)
    p.add_argument("--lib", required=True)
    p.add_argument("--tok-probe", required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.manual_seed(2720916)
    checkpoint, model = load_typed_pair_model(args.pair_checkpoint)
    model.requires_grad_(False).eval().to(args.device)
    embeddings = load_token_embedding_table(args.gguf, args.lib, checkpoint["backbone_sha256"], checkpoint["hidden"])
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    train = load_writer_examples(args.train_features, args.train_jsonl, checkpoint["backbone_sha256"], tokenizer, embeddings)
    valid = load_writer_examples(args.valid_features, args.valid_jsonl, checkpoint["backbone_sha256"], tokenizer, embeddings)
    if {e["world_id"] for e in train} & {e["world_id"] for e in valid}:
        raise ValueError("overlapping train and validation worlds")
    tx, ty = compile_sets(model, train, args.device)
    vx, vy = compile_sets(model, valid, args.device)
    del model, train, valid, embeddings
    tokenizer._proc.terminate()
    tokenizer._proc.wait()
    tx, ty, vx, vy = [t.clone().to(args.device) for t in (tx, ty, vx, vy)]
    rank = checkpoint["rank"]
    head = nn.Sequential(nn.Linear(7, rank), nn.GELU(), nn.Linear(rank, 1)).to(args.device)
    old = checkpoint["verifier_state_dict"]
    with torch.no_grad():
        head[0].weight.zero_()
        previous_width = old["predecessor_exists_head.0.weight"].shape[1]
        head[0].weight[:, :previous_width].copy_(old["predecessor_exists_head.0.weight"])
        head[0].bias.copy_(old["predecessor_exists_head.0.bias"])
        head[2].weight.copy_(old["predecessor_exists_head.2.weight"])
        head[2].bias.copy_(old["predecessor_exists_head.2.bias"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    positives = (ty > 0.5).nonzero().flatten()
    negatives = (ty < 0.5).nonzero().flatten()
    best = evaluate(head, vx, vy)
    best_state = clone_state_dict(head)
    selected = 0
    print(json.dumps({"phase": "absolute_link_baseline", **best}), flush=True)
    for step in range(1, args.steps + 1):
        indices = torch.cat([pool[torch.randint(len(pool), (64,), device=args.device)] for pool in (positives, negatives)])
        loss = F.binary_cross_entropy_with_logits(head(tx[indices]).squeeze(-1), ty[indices])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0:
            metrics = evaluate(head, vx, vy)
            if (min(metrics["accept"], metrics["reject"]), metrics["balanced"]) > (min(best["accept"], best["reject"]), best["balanced"]):
                best, best_state, selected = metrics, clone_state_dict(head), step
            print(json.dumps({"phase": "absolute_link_valid", "step": step, **metrics}), flush=True)
    for name, value in best_state.items():
        checkpoint["verifier_state_dict"]["predecessor_exists_head." + name] = value
    checkpoint.update(set_link_head_version=3, valid_metrics=best, selected_step=selected,
                      training_config=vars(args), parent_checkpoint_sha256=file_fingerprint(args.pair_checkpoint))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "link.pt")
    export_typed_pair_encoder_binary(output / "link.pt", output / "pair.bntpair")
    if file_fingerprint(output / "pair.bntpair") != file_fingerprint(args.pair_binary):
        raise ValueError("pair encoder changed; cannot reuse the deployed query activator")
    export_typed_link_binary(output / "link.pt", output / "link.bntlink")
    for count in (1, 2, 17):
        export_typed_link_parity_sample(output / "link.pt", output / f"link-{count}.parity", count)
    print(json.dumps({"phase": "absolute_link_done", "selected_step": selected, **best,
                      "pair_encoder_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
