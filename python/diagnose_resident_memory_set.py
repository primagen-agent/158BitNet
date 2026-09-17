#!/usr/bin/env python3
"""Diagnose address separation and in-sample learning before scaling training."""
import argparse
import inspect
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from train_resident_memory_set import (create_model, collate, compile_states,
                                      evaluate, load_worlds, read_batch, training_loss,
                                      FactorizedResidentMemorySet, select_sets,
                                      initialize_training_model, DEFAULT_TRAINING_SEED)
from typed_memory_training import file_fingerprint


@torch.inference_mode()
def address_stats(model, features, device):
    model.eval()
    states = compile_states(model, features, device)
    similarities, order_ratios = [], []
    for state in states:
        if state.ndim == 3:
            mask = state[..., -1] > .5
            state = (state[..., :-1] * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        x = F.normalize(state.float(), dim=-1)
        cosine = x @ x.T
        off_diagonal = ~torch.eye(len(x), dtype=torch.bool)
        similarities.extend(cosine[off_diagonal].tolist())
        pos = torch.linspace(0, 1, len(x), device=device)
        if hasattr(model, "order"):
            order = model.order(torch.stack((pos, 1 - pos), -1))
            order_ratios.extend(order.norm(dim=-1).cpu().tolist())
    return {"address_cosine_mean": sum(similarities) / len(similarities),
            "address_cosine_min": min(similarities), "address_cosine_max": max(similarities),
            "mean_order_to_address_norm_ratio": sum(order_ratios) / len(order_ratios) if order_ratios else None}


def row_error_metrics(scores, count_prediction, targets, version_pairs):
    """Oracle cardinality is diagnostic postprocessing, never a serving input."""
    k = len(targets)
    ranked = scores.argsort(descending=True).tolist()
    oracle = sorted(ranked[:k])
    target = sorted(targets)
    selected = sorted(ranked[:count_prediction])
    threshold_selected = (scores > 0).nonzero().flatten().tolist()
    previous_version_error = False
    if k == 1 and oracle != target:
        previous_version_error = any(set((target[0], oracle[0])) == pair for pair in version_pairs)
    return {"count_correct": count_prediction == k, "set_correct": selected == target,
            "threshold_set_correct": threshold_selected == target,
            "threshold_count_correct": len(threshold_selected) == k,
            "oracle_count_set_correct": oracle == target,
            "previous_version_error_with_oracle_count": previous_version_error,
            "other_binding_error_with_oracle_count": k == 1 and oracle != target and not previous_version_error,
            "top1": ranked[0], "selected": selected, "oracle_count_selected": oracle,
            "threshold_selected": threshold_selected}


@torch.inference_mode()
def error_decomposition(model, worlds, features, device):
    model.eval(); states = compile_states(model, features, device)
    indices = [(wi, qi) for wi, w in enumerate(worlds) for qi in range(len(w["queries"]))]
    groups, details, current_tops = {}, [], {}
    scope_groups, seen_links = {}, set()
    link_metrics = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    for start in range(0, len(indices), 32):
        items = indices[start:start + 32]
        inputs = read_batch(features, states, items, device)
        components = None
        if isinstance(model, FactorizedResidentMemorySet) or getattr(model, "has_identity_channel", False):
            scores, counts, components = model.read_components(*inputs)
        else:
            scores, counts = model.read(*inputs)
        for i, (wi, qi) in enumerate(items):
            w = worlds[wi]; q = w["queries"][qi]
            predicted_count = min(int(counts[i].argmax()), len(w["events"]))
            version_pairs = [set(h["targets"]) for h in w["queries"] if h["kind"] == "history"]
            if components is not None:
                scope = scope_groups.setdefault(q["kind"], {"correct": 0, "total": 0})
                scope["total"] += 1
                scope["correct"] += int(bool(components["history"][i] > 0) == (q["kind"] == "history"))
                if wi not in seen_links:
                    seen_links.add(wi)
                    for first in range(len(w["events"])):
                        for second in range(first + 1, len(w["events"])):
                            gold = {first, second} in version_pairs
                            predicted = bool(components["links"][i, first, second] > 0)
                            key = ("true_positive" if gold else "false_positive") if predicted else ("false_negative" if gold else "true_negative")
                            link_metrics[key] += 1
            row = row_error_metrics(scores[i, :len(w["events"])].cpu(), predicted_count, q["targets"], version_pairs)
            g = groups.setdefault(q["kind"], {"total": 0, **{key: 0 for key in row if key.endswith("correct") or key.endswith("count")}})
            g["total"] += 1
            for key in g:
                if key != "total": g[key] += int(row[key])
            if q["kind"] == "current": current_tops.setdefault(wi, set()).add(row["top1"])
            details.append({"world_id": w["world_id"], "query": q["text"], "kind": q["kind"],
                            "target_count": len(q["targets"]), "predicted_count": predicted_count, **row})
    return {"diagnostic_only": True, "oracle_count_is_not_accuracy": True, "groups": groups,
            "threshold_readout": "fixed logit > 0; no gold cardinality; diagnostic alternative, not deployed",
            "mean_distinct_top1_for_four_current_queries": sum(map(len, current_tops.values())) / len(current_tops),
            "learned_scope": scope_groups, "learned_version_links": link_metrics if scope_groups else None,
            "cases": details}


@torch.inference_mode()
def memory_interventions(model, worlds, features, device):
    """Paired storage controls, not normal QA accuracy or inference gold hints.

    Annotations construct alternative stores before the query is read. The
    reader only sees physically retained resident slots and the unchanged query.
    Erase controls retain distractors; keep controls retain all versions of the
    requested facts. Always report both: an ignore-all reader passes erasure.
    """
    model.eval(); states = compile_states(model, features, device)
    indices = [(wi, qi) for wi, w in enumerate(worlds) for qi, q in enumerate(w["queries"]) if q["targets"]]
    results = {}
    for mode in ("erase_relevant", "keep_relevant"):
        cases, query_only, retained, mappings = [], [], [], []
        for wi, qi in indices:
            world = worlds[wi]; query = world["queries"][qi]
            relevant = set(query["targets"])
            for other in world["queries"]:
                if other["kind"] == "history" and relevant.intersection(other["targets"]):
                    relevant.update(other["targets"])
            keep = [i for i in range(len(world["events"])) if (i in relevant) == (mode == "keep_relevant")]
            if mode == "erase_relevant" and not keep: continue  # Not a nonempty-distractor control.
            cases.append(query)
            mappings.append(keep)
            retained.append(states[wi][keep])
            query_only.append({"queries": [features[wi]["queries"][qi]]})
        groups = {}
        for start in range(0, len(cases), 32):
            items = [(i, 0) for i in range(start, min(start + 32, len(cases)))]
            inputs = read_batch(query_only, retained, items, device)
            scores, counts = model.read(*inputs)
            predicted = select_sets(scores, counts, inputs[-2])
            for batch_row, (i, _) in enumerate(items):
                chosen = [mappings[i][j] for j in predicted[batch_row].nonzero().flatten().tolist()]
                expected = cases[i]["targets"] if mode == "keep_relevant" else []
                stat = groups.setdefault(cases[i]["kind"], {"correct": 0, "total": 0})
                stat["total"] += 1; stat["correct"] += chosen == sorted(expected)
        results[mode] = {"groups": groups, "correct": sum(x["correct"] for x in groups.values()),
                         "total": sum(x["total"] for x in groups.values())}
    return {"diagnostic_only": True, "oracle_constructed_alternative_stores": True,
            "gold_at_read": False, "query_features_unchanged": True, "controls": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run"); p.add_argument("output"); p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--memory-interventions", action="store_true")
    p.add_argument("--checkpoint", choices=("final", "selected"), default="final")
    a = p.parse_args()
    if a.steps < 0: p.error("steps must be nonnegative; zero skips the fit check")
    root = Path(a.run); output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    cache = torch.load(root / "model/features.pt", map_location="cpu", weights_only=True)
    checkpoint_path = root / "model" / (a.checkpoint + "_research.pt")
    trained = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("backbone_sha256", "train_sha256", "valid_sha256"):
        if trained[key] != cache[key]: raise ValueError("checkpoint/cache identity mismatch")
    for split in ("train", "valid"):
        if file_fingerprint(root / "data" / (split + ".jsonl")) != cache[split + "_sha256"]:
            raise ValueError("cache/source identity mismatch")
    worlds = load_worlds(root / "data/train.jsonl", "train")
    features = cache["train"]
    hidden = features[0]["events"][0].shape[-1]
    order_scale = trained["configuration"].get("initial_order_scale", 1.)
    architecture = trained["configuration"].get("architecture", "pooled")
    seed = trained["configuration"].get("seed", DEFAULT_TRAINING_SEED)
    model = initialize_training_model(hidden, architecture, order_scale, seed, a.device)
    initial = address_stats(model, features[:8], a.device)
    model.load_state_dict(trained["state_dict"])
    after = address_stats(model, features[:8], a.device)
    print(json.dumps({"phase": "address_separation", "initial": initial, "trained": after}), flush=True)
    decompositions = {}
    for split in ("train", "valid"):
        split_worlds = load_worlds(root / "data" / (split + ".jsonl"), split)
        decompositions[split] = error_decomposition(model, split_worlds, cache[split], a.device)
        print(json.dumps({"phase": "error_decomposition", "split": split,
                          **{k: v for k, v in decompositions[split].items() if k != "cases"}}), flush=True)
    expected = json.loads((root / "model/summary.json").read_text())[a.checkpoint]
    actual = decompositions["valid"]["groups"]
    if sum(group["set_correct"] for group in actual.values()) != expected["correct"]:
        raise AssertionError("checkpoint diagnostic does not reproduce its recorded development result")
    for kind, group in actual.items():
        if group["set_correct"] != expected["groups"][kind]["correct"]:
            raise AssertionError("checkpoint diagnostic changed a recorded group result")
    interventions = None
    if a.memory_interventions:
        interventions = memory_interventions(model, load_worlds(root / "data/valid.jsonl", "valid"), cache["valid"], a.device)
        print(json.dumps({"phase": "memory_interventions", **interventions}), flush=True)
    # Reset shared parameters; no per-world state is optimized as a parameter.
    model = initialize_training_model(hidden, architecture, order_scale, seed, a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    world, feature = worlds[:1], features[:1]
    indices = [(0, i) for i in range(len(world[0]["queries"]))]
    trajectory = []
    for step in range(1, a.steps + 1):
        model.train()
        data = collate(world, feature, indices, a.device)
        loss = training_loss(model, data, world, indices, trained["configuration"].get("ranking_loss_weight", 0.))
        if not torch.isfinite(loss): raise ValueError("nonfinite sanity loss")
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == a.steps:
            metrics, _ = evaluate(model, world, feature, a.device)
            row = {"step": step, "loss": float(loss.detach()), **metrics}
            trajectory.append(row)
            print(json.dumps({"phase": "single_world_fit", **row}), flush=True)
    report = {"diagnostic_only": True, "not_a_generalization_test": True, "deployment_changed": False,
              "checkpoint_sha256": file_fingerprint(checkpoint_path), "checkpoint": a.checkpoint,
              "diagnostic_source_sha256": file_fingerprint(__file__),
              "reader_source_sha256": file_fingerprint(inspect.getfile(create_model)),
              "backbone_sha256": trained["backbone_sha256"], "architecture": architecture,
              "address_diagnostic_view": "normalized_per_event_summary_not_token_separation", "initial_address_stats": initial,
              "trained_address_stats": after, "single_world_trajectory": trajectory,
              "error_decomposition": decompositions, "memory_interventions": interventions,
              "checkpoint_step": trained.get("step", trained.get("selected_step")), "training_seed": seed}
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
