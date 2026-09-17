#!/usr/bin/env python3
"""Frozen-checkpoint crossed controls; diagnostic data is never training input.

Known/development entities, source wording and question wording are crossed on
identical fresh worlds. Additional controls change values or fact ownership.
All inference still uses neural resident addresses, not annotations or rules.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import re

import torch

from prepare_memory_set_curriculum import make_world, NAMES, FORMS, UPDATES
from train_resident_memory_set import (CTokenizer, GGUFWeights, TorchBackbone, encode_worlds,
                                      evaluate, create_model, file_fingerprint)
from diagnose_resident_memory_set import error_decomposition

QUERY_FORMS = {
    "current": "Which {r0} does {n0} have now?",
    "history": "Give the {r0} of {n0} both before and after the change.",
    "two_relations": "Give both the {r0} and the {r1} that {n0} has now.",
    "two_people": "Give the {r0} now associated with {n0} and with {n1}.",
    "four_facts": "For {n0} and {n1}, give both people's {r0} and {r1} now.",
}


def substitute(text, mapping):
    """Only a synthetic corpus transformation, never a memory-read operation."""
    if not mapping: return text
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?!\w)")
    return pattern.sub(lambda match: mapping[match.group()], text)


def transform_world(world, annotation, condition, *, new_entities=False, source_wording=False,
                    query_wording=False, value_map=None, swap_bindings=False):
    result = copy.deepcopy(world)
    names = dict(zip(NAMES["train"], NAMES["valid"])) if new_entities else {}
    values = value_map or {}
    event_keys = copy.deepcopy(annotation["event_keys"])
    if swap_bindings:
        # Swap one complete updated relation stream, not all facts of a person.
        updated = [spec["keys"][0] for q, spec in zip(world["queries"], annotation["query_specs"]) if q["kind"] == "history"]
        if len(updated) != 2 or updated[0][1] != updated[1][1]: raise ValueError("unsupported binding control")
        swap = {updated[0][0]: updated[1][0], updated[1][0]: updated[0][0]}
        for i, key in enumerate(event_keys):
            if key[1] == updated[0][1] and key[0] in swap:
                event_keys[i] = [swap[key[0]], key[1]]
    previous = {}
    for i, (original, event, key) in enumerate(zip(world["events"], result["events"], event_keys)):
        entity, relation = key
        local_names = dict(names)
        local_names[annotation["event_keys"][i][0]] = names.get(entity, entity)
        value = values.get(original["value"], original["value"])
        text = substitute(original["text"], {**values, **local_names})
        if source_wording:
            template = UPDATES["valid"] if tuple(key) in previous else FORMS["valid"][0]
            text = template.format(name=names.get(entity, entity), relation=relation,
                                   value=value, old=previous.get(tuple(key), ""))
        previous[tuple(key)] = value
        start = len(text[:text.index(value)].encode())
        event.update(text=text, value=value, value_start=start, value_end=start + len(value.encode()))
    groups = {}
    for i, key in enumerate(event_keys): groups.setdefault(tuple(key), []).append(i)
    for original, query, spec in zip(world["queries"], result["queries"], annotation["query_specs"]):
        text = substitute(original["text"], names)
        if query_wording:
            entities = list(dict.fromkeys(key[0] for key in spec["keys"]))
            relations = list(dict.fromkeys(key[1] for key in spec["keys"]))
            fields = {"n" + str(i): names.get(n, n) for i, n in enumerate(entities)}
            fields.update({"r" + str(i): r for i, r in enumerate(relations)})
            text = QUERY_FORMS[spec["form"]].format(**fields)
        targets = []
        for key in spec["keys"]:
            available = groups.get(tuple(key), [])
            if available: targets.extend(available if query["kind"] == "history" else available[-1:])
        targets = sorted(set(targets))
        query.update(text=text, targets=targets, answers=[result["events"][i]["value"] for i in targets])
    result.update(world_id=condition + ":" + world["world_id"], split="diagnostic", evaluation_only=True,
                  diagnostic_only=True, condition=condition, oracle_write_boundaries=True)
    return result


def build_controls(count=24, fresh_seed=2910917):
    if not 1 <= count <= 128 or fresh_seed == 2810916: raise ValueError("fresh diagnostic worlds required; replay is limited to 128 training worlds")
    conditions = {"replay": [], "replay_new_values": [], "fresh_prefix_shift": [],
                  "fresh_binding_swap": [], "fresh_full_surface_shift": []}
    for factors in itertools.product((0, 1), repeat=3): conditions["fresh_e%d_s%d_q%d" % factors] = []
    all_train = [make_world(i, "train", 2810916, True) for i in range(128)]
    training_values = {e["value"] for w in all_train for e in w["events"]}
    fresh_values = set()
    for i in range(count):
        original_annotation, fresh_annotation = {}, {}
        original = make_world(i, "train", 2810916, True, original_annotation, True)
        fresh = make_world(i, "train", fresh_seed, True, fresh_annotation, True)
        values = {e["value"] for e in fresh["events"]}
        if values & (training_values | fresh_values): raise ValueError("fresh value collision")
        fresh_values.update(values)
        conditions["replay"].append(transform_world(original, original_annotation, "replay"))
        # Keep the trained value format; only nonce contents change.
        value_map = {e["value"]: f"train-{fresh_seed + i * 100 + j:08x}" for j, e in enumerate(original["events"])}
        if set(value_map.values()) & training_values: raise ValueError("replacement value collision")
        conditions["replay_new_values"].append(transform_world(original, original_annotation, "replay_new_values", value_map=value_map))
        for factors in itertools.product((0, 1), repeat=3):
            name = "fresh_e%d_s%d_q%d" % factors
            conditions[name].append(transform_world(fresh, fresh_annotation, name, new_entities=bool(factors[0]),
                                                     source_wording=bool(factors[1]), query_wording=bool(factors[2])))
        value_map = {e["value"]: e["value"].replace("train-", "valid-", 1) for e in fresh["events"]}
        conditions["fresh_prefix_shift"].append(transform_world(fresh, fresh_annotation, "fresh_prefix_shift", value_map=value_map))
        conditions["fresh_full_surface_shift"].append(transform_world(fresh, fresh_annotation, "fresh_full_surface_shift", value_map=value_map,
                                                                        new_entities=True, source_wording=True, query_wording=True))
        conditions["fresh_binding_swap"].append(transform_world(fresh, fresh_annotation, "fresh_binding_swap", swap_bindings=True))
    return conditions


def load_diagnostic_features(path, binding, worlds):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in binding.items():
        if saved.get(key) != value: raise ValueError("diagnostic feature identity mismatch: " + key)
    features = saved["features"]
    if len(features) != len(worlds): raise ValueError("diagnostic cache world count mismatch")
    for world, feature in zip(worlds, features):
        for key in ("events", "queries"):
            if len(world[key]) != len(feature[key]): raise ValueError("diagnostic cache geometry mismatch")
            if any(not isinstance(x, torch.Tensor) or x.ndim != 2 or not 1 < len(x) <= 128 for x in feature[key]):
                raise ValueError("invalid diagnostic token features")
    return features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf"); p.add_argument("output"); p.add_argument("--checkpoint", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    p.add_argument("--lib", required=True); p.add_argument("--tok-probe", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--worlds", type=int, default=24)
    p.add_argument("--feature-cache", help="only the identity-bound evaluation-only diagnostic cache")
    p.add_argument("--components", action="store_true", help="add count/ranking/scope/link error decomposition")
    a = p.parse_args()
    if len({name for name, _ in a.checkpoint}) != len(a.checkpoint): p.error("duplicate checkpoint name")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name, _ in a.checkpoint): p.error("unsafe checkpoint label")
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    backbone_sha = file_fingerprint(a.gguf)
    checkpoints = []
    for name, path in a.checkpoint:
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["backbone_sha256"] != backbone_sha: raise ValueError("checkpoint/backbone identity mismatch")
        if item["configuration"]["architecture"] not in ("factorized", "identity"): raise ValueError("unsupported diagnostic checkpoint")
        checkpoints.append((name, path, item))
    controls = build_controls(a.worlds)
    flat = [world for worlds in controls.values() for world in worlds]
    corpus = root / "diagnostic_worlds.jsonl"
    corpus.write_text("".join(json.dumps(w) + "\n" for w in flat))
    binding = {"format": "RESIDENT_FACTORIAL_DIAGNOSTIC_FEATURES_V1", "backbone_sha256": backbone_sha,
               "diagnostic_corpus_sha256": file_fingerprint(corpus), "evaluation_only": True}
    if a.feature_cache:
        features = load_diagnostic_features(a.feature_cache, binding, flat)
    else:
        tokenizer = CTokenizer(a.tok_probe, a.gguf); weights = GGUFWeights(a.gguf, a.lib)
        backbone = TorchBackbone(weights, device=a.device, dtype=torch.float32).eval()
        try:
            features = encode_worlds(flat, backbone, tokenizer)  # no attention KV reuse
        finally:
            del backbone; weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
        torch.save({**binding, "features": features}, root / "diagnostic_features.pt")
    if a.device == "cuda": torch.cuda.empty_cache()
    results = {}; identity_features = None
    for name, path, checkpoint in checkpoints:
        config = checkpoint["configuration"]
        model_features = features
        if config["architecture"] == "identity":
            if identity_features is None:
                from train_resident_identity import append_lexical_features
                tokenizer = CTokenizer(a.tok_probe, a.gguf); weights = GGUFWeights(a.gguf, a.lib)
                try: identity_features = append_lexical_features(flat, features, weights, tokenizer)
                finally:
                    weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
            model_features = identity_features
        model = create_model(model_features[0]["events"][0].shape[-1], initial_order_scale=config["initial_order_scale"], architecture=config["architecture"]).to(a.device)
        model.load_state_dict(checkpoint["state_dict"]); model.eval().requires_grad_(False)
        rows = {}; offset = 0
        for condition, worlds in controls.items():
            metrics, cases = evaluate(model, worlds, model_features[offset:offset + len(worlds)], a.device)
            rows[condition] = metrics
            if a.components:
                components = error_decomposition(model, worlds, model_features[offset:offset + len(worlds)], a.device)
                rows[condition]["components"] = {key: value for key, value in components.items() if key != "cases"}
            (root / (name + "_" + condition + ".json")).write_text(json.dumps(cases) + "\n")
            offset += len(worlds)
            positive = sum(metrics["groups"][kind]["correct"] for kind in ("current", "multi", "history"))
            print(json.dumps({"phase": "factorial", "checkpoint": name, "condition": condition, "positive_correct": positive,
                              "positive_total": len(worlds) * 10, "null_correct": metrics["groups"]["null"]["correct"]}), flush=True)
        if any(not torch.equal(value.cpu(), checkpoint["state_dict"][key]) for key, value in model.state_dict().items()):
            raise AssertionError("frozen checkpoint changed")
        results[name] = {"path": path, "sha256": file_fingerprint(path), "training_seed": config.get("seed", 2810916),
                         "distractor_pairs": config.get("distractor_pairs", False),
                         "selected_step": checkpoint["selected_step"], "weights_unchanged": True, "conditions": rows}
        del model
    report = {**binding, "worlds_per_condition": a.worlds, "results": results, "trained": False,
              "oracle_write_boundaries": True, "automatic_memory_accuracy": False, "test_opened": False,
              "locomo_used": False, "kv_reuse": False, "deployment_changed": False,
              "component_decomposition": a.components, "feature_cache": a.feature_cache,
              "source_sha256": file_fingerprint(__file__),
              "caveats": ["Development entities/wording are diagnostic factors, not a new independent test set.",
                          "Fresh worlds change factual configurations and nonce values; they are not guaranteed to contain unseen atomic entity-relation pairs.",
                          "Binding swap preserves question text and cardinality but changes ownership of one complete relation stream."]}
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FROZEN_FACTORIAL_DONE", flush=True)


if __name__ == "__main__": main()
