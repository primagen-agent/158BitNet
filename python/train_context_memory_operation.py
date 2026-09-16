#!/usr/bin/env python3
"""Learn create/update from the complete frozen-backbone message, not value slots."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from train_typed_memory_writer import load_raw_worlds
from typed_memory_training import file_fingerprint, clone_state_dict

FORMAT = "TYPED_CONTEXT_OPERATION_V1"


def operation_features(hidden):
    mixed = hidden.float().mean(dim=1)
    content = mixed[1:] if len(mixed) > 1 else mixed
    return torch.cat((F.normalize(content.mean(dim=0), dim=0),
                      F.normalize(mixed[-1], dim=0)))


def load_features(cache_path, raw_path, writer):
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if (cache["backbone_sha256"] != writer["backbone_sha256"] or
        cache["source_fingerprint"] != file_fingerprint(raw_path) or
        cache["hidden_layer_bands"] != writer["hidden_layer_bands"]):
        raise ValueError("operation feature provenance mismatch")
    worlds = load_raw_worlds(raw_path)
    banks = {row["world_id"]: row["episode_hidden"] for row in cache["rows"]}
    if set(worlds) != set(banks):
        raise ValueError("operation feature worlds mismatch")
    x, y, domains = [], [], []
    for name, world in worlds.items():
        if len(banks[name]) != len(world["events"]):
            raise ValueError("operation event count mismatch")
        for hidden, event in zip(banks[name], world["events"]):
            x.append(operation_features(hidden))
            y.append(int(event["operation"] == "update"))
            domains.append(world["domain"])
    return torch.stack(x), torch.tensor(y), set(worlds), domains


@torch.inference_mode()
def evaluate(head, x, y, domains=None):
    prediction = head(x).argmax(dim=-1)
    result = {name: (prediction[y == value] == value).float().mean().item()
            for value, name in enumerate(("create", "update"))}
    if domains is not None and len(set(domains)) > 1:
        result["domains"] = {}
        for domain in sorted(set(domains)):
            mask = torch.tensor([d == domain for d in domains], device=y.device)
            result["domains"][domain] = {name: (prediction[mask & (y == value)] == value).float().mean().item()
                for value, name in enumerate(("create", "update"))}
    return result


def operation_selection_key(metrics):
    groups = list(metrics.get("domains", {}).values()) or [metrics]
    values = [group[name] for group in groups for name in ("create", "update")]
    return min(values), sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    for name in ("writer", "train_features", "train_jsonl", "valid_features", "valid_jsonl", "output"):
        parser.add_argument(name)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--writer-binary", required=True)
    args = parser.parse_args()
    torch.manual_seed(2730916)
    writer = torch.load(args.writer, map_location="cpu", weights_only=True)
    tx, ty, train_ids, train_domains = load_features(args.train_features, args.train_jsonl, writer)
    vx, vy, valid_ids, valid_domains = load_features(args.valid_features, args.valid_jsonl, writer)
    if train_ids & valid_ids:
        raise ValueError("training and development worlds overlap")
    tx, ty, vx, vy = [t.to(args.device) for t in (tx, ty, vx, vy)]
    head = nn.Sequential(nn.Linear(writer["hidden"] * 2, writer["rank"]), nn.GELU(),
                         nn.Linear(writer["rank"], 2)).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    pools = [(torch.tensor([d == domain for d in train_domains], device=ty.device) & (ty == value)).nonzero().flatten()
             for domain in sorted(set(train_domains)) for value in (0, 1)]
    if any(len(pool) == 0 for pool in pools) or args.steps < 1:
        raise ValueError("every domain needs both operations and at least one step")
    best, selected, state = None, 0, None
    for step in range(1, args.steps + 1):
        indices = torch.cat([pool[torch.randint(len(pool), (max(1, 128 // len(pools)),), device=args.device)] for pool in pools])
        loss = F.cross_entropy(head(tx[indices]), ty[indices])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 50 == 0 or step == args.steps:
            metrics = evaluate(head, vx, vy, valid_domains)
            key = operation_selection_key(metrics)
            if best is None or key > operation_selection_key(best):
                best, selected, state = metrics, step, clone_state_dict(head)
            print(json.dumps({"phase": "context_operation_valid", "step": step, **metrics}), flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "state_dict": state, "selected_step": selected,
                "valid_metrics": best, "backbone_sha256": writer["backbone_sha256"],
                "writer_checkpoint_fingerprint": file_fingerprint(args.writer),
                "writer_binary_sha256": file_fingerprint(args.writer_binary),
                "training_config": vars(args), "train_source_sha256": file_fingerprint(args.train_jsonl),
                "valid_source_sha256": file_fingerprint(args.valid_jsonl)}, output)
    print(json.dumps({"phase": "context_operation_done", "selected_step": selected, **best}), flush=True)


if __name__ == "__main__":
    main()
