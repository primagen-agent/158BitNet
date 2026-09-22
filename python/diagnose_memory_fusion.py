"""Cross name and question changes without updating the fusion checkpoint."""
import argparse
import collections
import copy
import json
from pathlib import Path
import re

import torch

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from memory_fusion import FORMAT, GatedMemoryFusion
from prepare_memory_fusion_curriculum import NAMES, TASKS, TRAIN_ENTITY_VARIANTS
from torch_backbone import TorchBackbone
from train_memory_fusion import (load_rows, compile_rows, encode_sources,
                                 generation_evaluate, row_memory, BACKBONE_SHA256)
from typed_memory_training import file_fingerprint


def change_questions(rows, use_training):
    result = copy.deepcopy(rows)
    for row in result:
        names = [name for name in set(NAMES[row["split"]] + TRAIN_ENTITY_VARIANTS) if re.search(r"\b" + name + r"\b", row["query"])]
        relation = next(task for task in TASKS if task[0] in row["memory"][0])
        if len(names) != 1: raise ValueError("diagnostic query subject mismatch")
        row["query"] = relation[2 if use_training else 3].format(name=names[0])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf"); p.add_argument("checkpoint"); p.add_argument("data"); p.add_argument("output")
    p.add_argument("--lib", required=True); p.add_argument("--tok-probe", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--decision-only", action="store_true", help="measure the first response token without teacher-forced answer context")
    a = p.parse_args()
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    before = file_fingerprint(a.checkpoint)
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["format"] != FORMAT or checkpoint["backbone_sha256"] != file_fingerprint(a.gguf) or checkpoint["backbone_sha256"] != BACKBONE_SHA256:
        raise ValueError("checkpoint binding mismatch")
    data = Path(a.data)
    for split in ("train", "valid"):
        if file_fingerprint(data / f"{split}.jsonl") != checkpoint[f"{split}_sha256"]:
            raise ValueError("curriculum mismatch")
    train, valid = load_rows(data / "train.jsonl", "train")[:32], load_rows(data / "valid.jsonl", "valid")
    conditions = {"training_replay": train, "new_names_only": change_questions(valid, True),
                  "new_question_only": change_questions(train, False)}
    weights = GGUFWeights(a.gguf, a.lib); tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        backbone = TorchBackbone(weights, device=a.device, dtype=torch.bfloat16 if a.device.startswith("cuda") else torch.float32)
        backbone.requires_grad_(False).eval(); weights.close()
        fusion = GatedMemoryFusion(checkpoint["hidden"], checkpoint["layers"], checkpoint["heads"],
                                   evidence_gate=checkpoint["configuration"].get("evidence_gate", False)).to(a.device)
        fusion.load_state_dict(checkpoint["state_dict"]); fusion.requires_grad_(False).eval()
        bank = encode_sources(train + valid, backbone, tokenizer)
        result = {}; decisions = collections.defaultdict(list)
        with torch.no_grad():
            for row in compile_rows(valid, tokenizer, a.device):
                memory = row_memory(row, bank, fusion.hidden, a.device)
                logits = backbone(torch.tensor(row["prefix"], device=a.device), memory_fusion=fusion.bind(memory)).float()
                target = row["tokens"][row["prefix_length"]]
                decisions[row["condition"]].append({"correct": int(logits.argmax()) == int(target),
                    "nll": float(torch.nn.functional.cross_entropy(logits[None], target[None]))})
        decision_metrics = {kind: {"correct": sum(x["correct"] for x in items), "total": len(items),
                                   "mean_nll": sum(x["nll"] for x in items) / len(items)} for kind, items in decisions.items()}
        for name, rows in (() if a.decision_only else conditions.items()):
            result[name] = generation_evaluate(compile_rows(rows, tokenizer, a.device), backbone, fusion, bank, tokenizer, 32)
            print(json.dumps({"phase": "diagnostic", "condition": name, "groups": result[name]["groups"],
                              "paired_swaps": result[name]["paired_swaps"]}), flush=True)
        if file_fingerprint(a.checkpoint) != before: raise AssertionError("diagnostic changed checkpoint")
        report = {"checkpoint_sha256": before, "evaluation_only": True, "weights_updated": False,
                  "training_replay_is_not_generalization": True, "conditions": result,
                  "first_response_token": decision_metrics}
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"phase": "first_response_token", "groups": decision_metrics}), flush=True)
    finally:
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()


if __name__ == "__main__": main()
