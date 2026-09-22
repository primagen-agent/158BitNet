"""Train/evaluate the first architectural gate: memory-conditioned generation.

Correct episode boundaries are supplied. This is NOT autonomous memory, a
LoCoMo evaluation, or a deployable C model. No source text enters the prompt;
every decode step recomputes the prefix, without KV caching.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from memory_fusion import FORMAT, GatedMemoryFusion
from torch_backbone import TorchBackbone
from typed_memory_training import clone_state_dict, file_fingerprint, reject_evaluation_row

BACKBONE_SHA256 = "44cb4e0db8374d4247bba391b3d3b7c0b3ee815c36cf2e20d0d1a3570677bd95"


def prompt(query):
    return ("<|im_start|>system\nReply briefly in plain text. Do not use tools.<|im_end|>\n"
            "<|im_start|>user\n" + query + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n")


def load_rows(path, split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    for row in rows:
        reject_evaluation_row(row)
        if row.get("locomo_used") or row["split"] != split or not row.get("oracle_episode_boundaries"):
            raise ValueError("wrong data provenance")
        if row.get("automatic_memory") or not isinstance(row["memory"], list):
            raise ValueError("oracle-boundary diagnostic required")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty/duplicate curriculum")
    return rows


def compile_rows(rows, tokenizer, device):
    result = []
    for row in rows:
        prefix = tokenizer.encode(prompt(row["query"]), True)
        full = tokenizer.encode(prompt(row["query"]) + row["answer"], True) + [tokenizer.eos()]
        if full[:len(prefix)] != prefix or len(full) <= len(prefix) or len(full) > 256:
            raise ValueError("token boundary or sequence length mismatch")
        result.append({**row, "prefix": prefix, "tokens": torch.tensor(full, device=device),
                       "prefix_length": len(prefix)})
    return result


@torch.no_grad()
def encode_sources(rows, backbone, tokenizer):
    bank = {}
    for row in rows:
        for text in row["memory"]:
            if text in bank:
                continue
            ids = tokenizer.encode(text, True)
            if not 1 < len(ids) <= 128:
                raise ValueError("source length outside diagnostic limit")
            tokens = torch.tensor(ids, device=backbone.device)
            hidden = F.normalize(backbone(tokens, return_hidden=True).float(), dim=-1)
            lexical = F.normalize(backbone.token_embd[tokens].float(), dim=-1)
            bank[text] = torch.cat((hidden[1:], lexical[1:]), dim=-1).half().cpu()
        if len(bank) and len(bank) % 32 == 0:
            print(json.dumps({"phase": "memory_features", "sources": len(bank)}), flush=True)
    return bank


def row_memory(row, bank, hidden, device):
    if not row["memory"]:
        return torch.empty(0, 2 * hidden, device=device)
    return torch.cat([bank[text] for text in row["memory"]]).to(device=device, dtype=torch.float32)


def loss_for(row, backbone, fusion, bank, gate_loss_weight=0., decision_loss_weight=1.):
    memory = row_memory(row, bank, fusion.hidden, backbone.device)
    observations = [] if gate_loss_weight else None
    logits = backbone(row["tokens"][:-1], logits_all=True, memory_fusion=fusion.bind(memory, observations))
    start = row["prefix_length"] - 1
    errors = F.cross_entropy(logits[start:].float(), row["tokens"][start + 1:], reduction="none")
    importance = torch.ones_like(errors)
    importance[:3] = decision_loss_weight
    loss = (errors * importance).sum() / importance.sum()
    if gate_loss_weight:
        # Only the query-prefix position: teacher-forced answer tokens cannot
        # reveal the target to this auxiliary memory-usability decision.
        gate_logits = torch.stack([gate[start, 0] for gate in observations])
        target = torch.full_like(gate_logits, float(row["condition"] != "removed"))
        loss = loss + gate_loss_weight * F.binary_cross_entropy_with_logits(gate_logits, target)
    return loss


@torch.no_grad()
def teacher_evaluate(rows, backbone, fusion, bank):
    fusion.eval()
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(float(loss_for(row, backbone, fusion, bank)))
    return {kind: sum(values) / len(values) for kind, values in groups.items()}


@torch.no_grad()
def generate(row, backbone, fusion, bank, tokenizer, limit):
    ids = list(row["prefix"])
    memory = row_memory(row, bank, fusion.hidden, backbone.device)
    fused = fusion.bind(memory)
    stop = {tokenizer.eos()}
    generated = []
    for _ in range(limit):
        logits = backbone(torch.tensor(ids, device=backbone.device), memory_fusion=fused)
        token = int(logits.argmax())
        if token in stop:
            break
        ids.append(token); generated.append(token)
    return b"".join(tokenizer.decode_pieces(generated)).decode("utf-8", errors="replace").strip()


def judge(row, answer):
    values = [v for v in row["value_vocabulary"] if re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", answer, re.I)]
    expected = row["expected_value"]
    if expected is None:
        grounded = not values and any(cue in answer.lower() for cue in ("don't know", "do not know", "not know", "not sure", "not provided", "no information"))
    else:
        grounded = values == [expected]
    natural = not any(cue in answer.lower() for cue in ("<tool", "<think", "<answer", "memory", "remember", "recall"))
    # These diagnostic targets explicitly name the subject. A correct value
    # attributed to a misspelled/different person is not a passing answer.
    subject = (row["answer"][len("I don't know "):].split("'s", 1)[0]
               if expected is None else row["answer"].split(" ", 1)[0])
    subject_exact = bool(re.search(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", answer))
    return {"value_grounded": grounded, "natural_surface": natural,
            "subject_exact": subject_exact, "exact_sentence": answer == row["answer"],
            "passed": grounded and natural and subject_exact}


@torch.no_grad()
def generation_evaluate(rows, backbone, fusion, bank, tokenizer, limit):
    fusion.eval(); cases = []; groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        text = generate(row, backbone, fusion, bank, tokenizer, limit)
        result = {"id": row["id"], "group": row["group"], "condition": row["condition"],
                  "query": row["query"], "memory": row["memory"], "expected": row["answer"],
                  "actual": text, **judge(row, text)}
        cases.append(result); groups[row["condition"]].append(result)
        print(json.dumps({"phase": "generate", "case": i + 1, "total": len(rows),
                          "condition": row["condition"], "passed": result["passed"], "actual": text}), flush=True)
    metrics = {kind: {"total": len(items), **{key: sum(x[key] for x in items) for key in
                ("passed", "value_grounded", "natural_surface", "subject_exact", "exact_sentence")}} for kind, items in groups.items()}
    pair_groups = collections.defaultdict(list)
    for row in cases:
        if row["condition"] in ("original", "swapped"):
            pair_groups[row["group"]].append(row)
    pairs = sum(len(items) == 2 and all(x["passed"] for x in items) for items in pair_groups.values())
    return {"groups": metrics, "paired_swaps": {"passed": pairs, "total": len(pair_groups)}, "cases": cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf"); p.add_argument("data"); p.add_argument("output")
    p.add_argument("--lib", required=True); p.add_argument("--tok-probe", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--steps", type=int, default=300)
    p.add_argument("--accumulation", type=int, default=4); p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=2e-4); p.add_argument("--seed", type=int, default=3180918)
    p.add_argument("--generation-tokens", type=int, default=32)
    p.add_argument("--evidence-gate", action="store_true")
    p.add_argument("--gate-loss-weight", type=float, default=0.)
    p.add_argument("--decision-loss-weight", type=float, default=1.)
    p.add_argument("--select-by-generation", action="store_true")
    a = p.parse_args()
    if min(a.steps, a.accumulation, a.eval_every, a.generation_tokens) < 1 or a.learning_rate <= 0:
        p.error("positive training limits required")
    if a.gate_loss_weight < 0 or a.decision_loss_weight < 1 or (a.gate_loss_weight and not a.evidence_gate):
        p.error("auxiliary gate supervision requires evidence gate; nonnegative loss weights required")
    if file_fingerprint(a.gguf) != BACKBONE_SHA256:
        p.error("this experiment is bound to the exact 0.5B backbone")
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    source = Path(a.data)
    train, valid = load_rows(source / "train.jsonl", "train"), load_rows(source / "valid.jsonl", "valid")
    if {r["group"] for r in train} & {r["group"] for r in valid}:
        raise ValueError("world leakage")
    metadata = {"format": FORMAT, "configuration": vars(a), "backbone_sha256": BACKBONE_SHA256,
                "train_sha256": file_fingerprint(source / "train.jsonl"), "valid_sha256": file_fingerprint(source / "valid.jsonl"),
                "kv_cache_used": False, "lora_used": False, "locomo_used": False,
                "oracle_episode_boundaries": True, "automatic_memory": False, "deployable": False,
                "source_text_in_generation_prompt": False, "torch_version": str(torch.__version__),
                "implementation_sha256": {name: file_fingerprint(Path(__file__).with_name(name)) for name in
                    ("train_memory_fusion.py", "memory_fusion.py", "torch_backbone.py", "c_tokenizer.py")},
                "tokenizer_binary_sha256": file_fingerprint(a.tok_probe),
                "weight_shim_sha256": file_fingerprint(a.lib)}
    (root / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    random.seed(a.seed); torch.manual_seed(a.seed)
    weights = GGUFWeights(a.gguf, a.lib); tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        backbone = TorchBackbone(weights, device=a.device, dtype=torch.bfloat16 if a.device.startswith("cuda") else torch.float32)
        backbone.requires_grad_(False).eval()
        weights.close()
        fusion = GatedMemoryFusion(backbone.cfg.hidden, backbone.cfg.n_layers, evidence_gate=a.evidence_gate).to(a.device)
        bank = encode_sources(train + valid, backbone, tokenizer)
        torch.save({**metadata, "sources": bank}, root / "source_features.pt")
        restored = torch.load(root / "source_features.pt", map_location="cpu", weights_only=True)
        if restored["backbone_sha256"] != BACKBONE_SHA256 or any(not torch.equal(v, restored["sources"][k]) for k, v in bank.items()):
            raise AssertionError("persisted memory feature round trip failed")
        bank = restored["sources"]
        train, valid = compile_rows(train, tokenizer, a.device), compile_rows(valid, tokenizer, a.device)
        control = torch.tensor(valid[0]["prefix"], device=a.device)
        with torch.no_grad():
            baseline_logits = backbone(control)
            zero_logits = backbone(control, memory_fusion=fusion.bind(row_memory(valid[0], bank, fusion.hidden, a.device)))
        if not torch.equal(baseline_logits, zero_logits):
            raise AssertionError("zero initialized fusion changes baseline")
        metadata["initial_logits_exact"] = True
        initial = teacher_evaluate(valid, backbone, fusion, bank)
        print(json.dumps({"phase": "initial", "loss": initial, "trainable_parameters": sum(x.numel() for x in fusion.parameters())}), flush=True)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=a.learning_rate, weight_decay=.01)
        rng = random.Random(a.seed); best = float("inf"); best_selection = None; best_step = 0; started = time.monotonic()
        pools = {kind: [r for r in train if r["condition"] == kind] for kind in ("original", "swapped", "removed", "distractor")}
        for step in range(1, a.steps + 1):
            fusion.train(); optimizer.zero_grad(set_to_none=True); total = 0.
            for j in range(a.accumulation):
                row = rng.choice(pools[tuple(pools)[(step * a.accumulation + j) % len(pools)]])
                loss = loss_for(row, backbone, fusion, bank, a.gate_loss_weight, a.decision_loss_weight)
                if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
                (loss / a.accumulation).backward(); total += float(loss.detach()) / a.accumulation
            norm = torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.)
            if not torch.isfinite(norm): raise ValueError("nonfinite gradients")
            optimizer.step()
            if step == 1 or step % 10 == 0:
                print(json.dumps({"phase": "train", "step": step, "loss": total, "gradient_norm": float(norm),
                                  "seconds": time.monotonic() - started,
                                  "gpu_peak_bytes": torch.cuda.max_memory_allocated() if a.device.startswith("cuda") else 0}), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                metrics = teacher_evaluate(valid, backbone, fusion, bank); score = sum(metrics.values()) / len(metrics)
                print(json.dumps({"phase": "valid", "step": step, "loss": metrics}), flush=True)
                generated = None
                if a.select_by_generation:
                    generated = generation_evaluate(valid, backbone, fusion, bank, tokenizer, a.generation_tokens)
                    selection = (min(g["passed"] / g["total"] for g in generated["groups"].values()),
                                 generated["paired_swaps"]["passed"] / generated["paired_swaps"]["total"],
                                 sum(g["passed"] for g in generated["groups"].values()), -score)
                    improved = best_selection is None or selection > best_selection
                    if improved: best_selection = selection
                else:
                    improved = score < best
                if improved:
                    best, best_step = score, step
                    torch.save({**metadata, "hidden": fusion.hidden, "layers": fusion.layers, "heads": fusion.heads,
                                "selected_step": step, "valid_loss": metrics, "valid_generation": generated,
                                "state_dict": clone_state_dict(fusion)}, root / "selected.pt")
        selected = torch.load(root / "selected.pt", map_location="cpu", weights_only=True)
        fusion.load_state_dict(selected["state_dict"]); fusion.eval()
        with torch.no_grad():
            empty = fusion.bind(torch.empty(0, 2 * fusion.hidden, device=a.device))
            empty_equal = torch.equal(baseline_logits, backbone(control, memory_fusion=empty))
        if not empty_equal: raise AssertionError("empty memory changed frozen backbone")
        result = generation_evaluate(valid, backbone, fusion, bank, tokenizer, a.generation_tokens)
        group_pass = all(g["passed"] / g["total"] >= .9 for g in result["groups"].values())
        pair_pass = result["paired_swaps"]["passed"] / result["paired_swaps"]["total"] >= .9
        report = {**metadata, "selected_step": best_step, "selected_sha256": file_fingerprint(root / "selected.pt"),
                  "initial_valid_loss": initial, "selected_valid_loss": selected["valid_loss"],
                  "empty_memory_logits_exact": empty_equal, "state_roundtrip_exact": True,
                  "grader_version": "value_and_subject_v2",
                  "gate_one_passed": bool(group_pass and pair_pass and empty_equal),
                  "gate_threshold": "at least 90% value-and-subject correctness per group and both answers correct in at least 90% of paired swaps; empty-memory logits identical",
                  "limitations": ["Oracle episode boundaries; not automatic memory.", "Two synthetic relations and shared value vocabulary; development only.",
                                  "No C deployment or independent final-test claim."], **result}
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"phase": "done", "gate_one_passed": report["gate_one_passed"], "groups": result["groups"], "paired_swaps": result["paired_swaps"]}), flush=True)
    finally:
        weights.close()
        tokenizer._proc.terminate(); tokenizer._proc.wait()


if __name__ == "__main__": main()
