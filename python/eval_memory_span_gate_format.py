#!/usr/bin/env python3
"""Diagnostic-only, meaning-preserving quote-format probes of frozen gates."""
import argparse
import json
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_memory_gate import NEGATIVES
from train_memory_span_gate import SpanGate, batch, encode
from typed_memory_training import file_fingerprint


def add_material_quotes(row):
    if row["label"] != 0 or row["kind"] != "non_assertion":
        raise ValueError("quote probe must remain a task-material negative")
    for template in NEGATIVES["valid"]:
        prefix, suffix = template.split("{text}")
        text = row["text"]
        if text.startswith(prefix) and text.endswith(suffix):
            end = len(text) - len(suffix) if suffix else len(text)
            content = text[len(prefix):end]
            return {**row, "text": prefix + '"' + content + '"' + suffix,
                    "segments": [{**s, "start": s["start"] + 1, "end": s["end"] + 1}
                                 for s in row["segments"]]}
    raise ValueError("unknown development wrapper; refusing heuristic rewrite")


def remove_material_quotes(row):
    if row["label"] != 0 or row["kind"] != "non_assertion" or row["text"].count('"') != 2:
        raise ValueError("expected one quoted task-material span")
    encoded = row["text"].encode("utf-8")
    quotes = [i for i, byte in enumerate(encoded) if byte == ord('"')]
    return {**row, "text": row["text"].replace('"', ''),
            "segments": [{**s, "start": s["start"] - sum(q < s["start"] for q in quotes),
                          "end": s["end"] - sum(q < s["end"] for q in quotes)} for s in row["segments"]]}


@torch.inference_mode()
def predict(model, features, rows, device):
    model.eval()
    result = []
    for start in range(0, len(rows), 64):
        stop = min(start + 64, len(rows))
        x, valid, mask, _ = batch(features, range(start, stop), device)
        margins, _ = model(x, valid, mask)
        result.extend({**row, "predicted": int(margin > 0), "write_margin": float(margin)}
                      for row, margin in zip(rows[start:stop], margins))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "run", "output"): p.add_argument(name)
    p.add_argument("--lib", required=True); p.add_argument("--tok-probe", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run = Path(args.run); output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    saved = torch.load(run / "features.pt", map_location="cpu", weights_only=True)
    if saved["backbone_sha256"] != file_fingerprint(args.gguf): raise ValueError("backbone identity mismatch")
    datasets = {s: json.loads((run / (s + ".json")).read_text()) for s in ("train", "valid")}
    for split in datasets:
        if saved[split + "_sha256"] != file_fingerprint(run / (split + ".json")):
            raise ValueError("feature/source identity mismatch")
    probes = {}
    vi = [i for i, r in enumerate(datasets["valid"]) if not r["label"] and r["kind"] == "non_assertion"]
    probes["development_add_quotes"] = ("valid", vi, add_material_quotes)
    ti = []
    for domain in ("replay", "dialogue"):
        indices = [i for i, r in enumerate(datasets["train"]) if r["domain"] == domain and not r["label"]
                   and r["kind"] == "non_assertion" and r["text"].count('"') == 2]
        ti += indices[::max(1, len(indices) // 32)][:32]
    probes["training_remove_quotes"] = ("train", ti, remove_material_quotes)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(weights, device=args.device, dtype=torch.float32).eval()
    prepared = {}
    try:
        for name, (split, indices, transform) in probes.items():
            rows = [datasets[split][i] for i in indices]
            transformed = [transform(r) for r in rows]
            prepared[name] = (rows, [saved[split][i] for i in indices], transformed, encode(transformed, backbone, tokenizer))
    finally:
        del backbone; weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
    results = {}
    for arm in ("classification_only", "joint_roles"):
        path = run / arm / "final_research.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("backbone_sha256", "train_sha256", "valid_sha256"):
            if checkpoint[key] != saved[key]: raise ValueError("checkpoint identity mismatch")
        model = SpanGate(checkpoint["state_dict"]["project.weight"].shape[1]).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        results[arm] = {"checkpoint_sha256": file_fingerprint(path), "probes": {}}
        for name, (rows, features, transformed, variants) in prepared.items():
            before = predict(model, features, rows, args.device)
            after = predict(model, variants, transformed, args.device)
            summary = {"total": len(rows), "false_writes_before": sum(r["predicted"] for r in before),
                       "false_writes_after": sum(r["predicted"] for r in after),
                       "write_to_ignore": sum(a["predicted"] == 1 and b["predicted"] == 0 for a, b in zip(before, after)),
                       "ignore_to_write": sum(a["predicted"] == 0 and b["predicted"] == 1 for a, b in zip(before, after))}
            print(json.dumps({"arm": arm, "probe": name, **summary}), flush=True)
            results[arm]["probes"][name] = {**summary, "cases": [{"before": a, "after": b} for a, b in zip(before, after)]}
    report = {"diagnostic_only": True, "deployment_changed": False, "locomo_used": False,
              "backbone_sha256": saved["backbone_sha256"], "results": results}
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
