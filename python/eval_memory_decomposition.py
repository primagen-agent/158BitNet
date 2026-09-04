#!/usr/bin/env python3
"""Decompose memory QA failures into storage, interference, and decoding.

The evaluator deliberately avoids KV cache.  It compares:
  - memory_all: every supplied memory chunk is committed;
  - memory_evidence: only annotated evidence chunks are committed;
  - memory_distractors: only annotated distractors are committed;
  - no_memory: the same query is scored with the memory path disabled;
  - full_context: all supplied chunks are placed directly in the prompt.

For each available mode it reports teacher-forced likelihood and ranking of
the gold answer among answer distractors.  Free generation is measured only
for memory_all, because it is the expensive and least diagnostic stage.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from backbone_lora import load_lora_bundle  # noqa: E402
from eval_bnmem_torch_locomo import (  # noqa: E402
    CTokenDecoder,
    CTokenizer,
    GGUFWeights,
    TorchBackbone,
    forward_chunk,
    generate,
    load_memory,
)
from eval_locomo import f1_score, official_like_score  # noqa: E402
from eval_reference_protocol import (  # noqa: E402
    gen_implicit_sessions,
    gen_sessions,
)
from train_data import render_chunk  # noqa: E402


def read_jsonl_rows(path, max_samples):
    paths = sorted(Path(path).glob("*.jsonl")) if Path(path).is_dir() else [
        Path(path)]
    rows = []
    seen = set()
    for source in paths:
        for line_index, line in enumerate(
            source.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            locomo_id = str(row.get("locomo_id", ""))
            key = (
                ":".join(locomo_id.split(":")[:2])
                if locomo_id else f"{source}:{line_index}")
            if key in seen:
                continue
            seen.add(key)
            row["_diagnostic_id"] = key
            row["_diagnostic_group"] = str(
                row.get("category", row.get("stratum", "unknown")))
            rows.append(row)
            if max_samples and len(rows) >= max_samples:
                return rows
    return rows


def reference_rows(protocol, n_per_op):
    sessions = (
        gen_implicit_sessions(n_per_op)
        if protocol == "implicit" else gen_sessions(n_per_op))
    rows = []
    for index, session in enumerate(sessions):
        messages = []
        for key in ("mem", "mem2", "mem3"):
            if key not in session:
                continue
            messages.append([
                {"role": "user", "content": session[key]},
                {"role": "assistant", "content": "OK"},
            ])
        answer = (
            session["gold"][0]
            if session["gold"] is not None
            else "No information available")
        preferred_distractors = []
        if protocol == "explicit" and session["op"] in {"update", "forget"}:
            match = re.search(r"\bis\s+(.+?)\.\s*$", session["mem"])
            if match:
                preferred_distractors.append(match.group(1))
        messages.append([
            {"role": "user", "content": session["query"]},
            {"role": "assistant", "content": answer},
        ])
        rows.append({
            "messages": messages,
            "query_turn_id": len(messages) - 1,
            "evidence_message_indices": list(range(len(messages) - 1)),
            "distractor_message_indices": [],
            "_diagnostic_id": f"{protocol}:{index}",
            "_diagnostic_group": session["op"],
            "_preferred_distractors": preferred_distractors,
        })
    return rows


def answer_text(row):
    query_index = row.get("query_turn_id", len(row["messages"]) - 1)
    _prompt, target = render_chunk(row["messages"][query_index], True)
    return str(target)


def candidate_sets(rows, count, seed):
    rng = random.Random(seed)
    grouped = collections.defaultdict(list)
    for row in rows:
        answer = answer_text(row)
        if answer not in grouped[row["_diagnostic_group"]]:
            grouped[row["_diagnostic_group"]].append(answer)
    result = {}
    global_answers = list(dict.fromkeys(answer_text(row) for row in rows))
    for row in rows:
        gold = answer_text(row)
        pool = [
            value for value in grouped[row["_diagnostic_group"]]
            if value != gold]
        preferred = [
            value for value in row.get("_preferred_distractors", [])
            if value != gold]
        pool = preferred + [value for value in pool if value not in preferred]
        if len(pool) < count - 1:
            pool.extend(
                value for value in global_answers
                if value != gold and value not in pool)
        rng.shuffle(pool)
        result[row["_diagnostic_id"]] = [gold] + pool[:count - 1]
    return result


@torch.inference_mode()
def score_sequence(backbone, memory, prompt_ids, target_ids):
    full = prompt_ids + target_ids[:-1]
    tokens = torch.tensor(
        full, device=backbone.device, dtype=torch.long)
    logits = backbone(
        tokens, memory_v6=memory, logits_all=True)
    if memory is not None:
        memory.discard_captured()
    selected = logits[len(full) - len(target_ids): len(full)].float()
    target = torch.tensor(target_ids, device=backbone.device)
    loss = F.cross_entropy(selected, target, reduction="none")
    prediction = selected.argmax(dim=-1)
    return {
        "nll": float(loss.mean()),
        "token_acc": float((prediction == target).float().mean()),
        "exact": float(bool(torch.all(prediction == target))),
    }


@torch.inference_mode()
def score_candidates(backbone, memory, prompt_ids, candidates, tokenizer, eos,
                     gold_nll):
    scores = [gold_nll]
    for text in candidates[1:]:
        ids = tokenizer.encode(text, add_bos=False) + [eos]
        scores.append(score_sequence(
            backbone, memory, prompt_ids, ids)["nll"])
    if len(scores) <= 1:
        return {"candidate_acc": None, "candidate_margin": None}
    best_distractor = min(scores[1:])
    return {
        "candidate_acc": float(scores[0] <= best_distractor),
        "candidate_margin": best_distractor - scores[0],
    }


def commit_chunks(backbone, memory, row, tokenizer, indices):
    memory.reset_state()
    query_index = row.get("query_turn_id", len(row["messages"]) - 1)
    selected = set(indices)
    for index, chunk in enumerate(row["messages"][:query_index]):
        if index not in selected:
            continue
        text, _target = render_chunk(chunk, False)
        ids = tokenizer.encode(text, add_bos=True)
        forward_chunk(backbone, memory, ids)
        memory.commit_all()
    memory.discard_captured()


def query_material(row, tokenizer, eos):
    query_index = row.get("query_turn_id", len(row["messages"]) - 1)
    prompt, target = render_chunk(row["messages"][query_index], True)
    return (
        tokenizer.encode(prompt, add_bos=True),
        tokenizer.encode(target, add_bos=False) + [eos],
        target,
    )


def full_context_material(row, tokenizer, eos):
    query_index = row.get("query_turn_id", len(row["messages"]) - 1)
    messages = []
    for chunk in row["messages"][:query_index]:
        messages.extend(chunk)
    messages.extend(row["messages"][query_index])
    prompt, target = render_chunk(messages, True)
    return (
        tokenizer.encode(prompt, add_bos=True),
        tokenizer.encode(target, add_bos=False) + [eos],
    )


def aggregate(results):
    summary = {}
    for mode in (
        "memory_all", "memory_evidence", "memory_distractors",
        "no_memory", "full_context",
    ):
        rows = [row[mode] for row in results if row.get(mode)]
        if not rows:
            continue
        fields = (
            "nll", "token_acc", "exact",
            "candidate_acc", "candidate_margin",
        )
        summary[mode] = {}
        for field in fields:
            values = [
                value[field] for value in rows
                if value.get(field) is not None]
            summary[mode][field] = (
                sum(values) / len(values) if values else None)
        summary[mode]["samples"] = len(rows)
    generated = [row for row in results if "free_f1" in row]
    if generated:
        summary["free_generation"] = {
            "f1": sum(row["free_f1"] for row in generated) / len(generated),
            "samples": len(generated),
        }
    by_group = collections.defaultdict(list)
    for row in results:
        by_group[row["group"]].append(row)
    summary["by_group"] = {
        group: {
            "samples": len(group_rows),
            "memory_nll": sum(
                row["memory_all"]["nll"] for row in group_rows)
            / len(group_rows),
            "memory_candidate_acc": sum(
                row["memory_all"]["candidate_acc"] or 0.0
                for row in group_rows)
            / len(group_rows),
            "free_f1": (
                sum(row["free_f1"] for row in group_rows
                    if "free_f1" in row)
                / sum("free_f1" in row for row in group_rows)
                if any("free_f1" in row for row in group_rows)
                else None),
        }
        for group, group_rows in sorted(by_group.items())
    }
    single = [
        row for row in results
        if len(row.get("evidence_indices", [])) == 1]
    if single:
        summary["single_evidence"] = {
            "samples": len(single),
            "memory_nll": sum(
                row["memory_all"]["nll"] for row in single) / len(single),
            "candidate_acc": sum(
                row["memory_all"]["candidate_acc"] or 0.0 for row in single)
            / len(single),
            "free_f1": (
                sum(row["free_f1"] for row in single if "free_f1" in row)
                / sum("free_f1" in row for row in single)
                if any("free_f1" in row for row in single)
                else None),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("memory_model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jsonl")
    source.add_argument(
        "--reference-protocol", choices=("explicit", "implicit"))
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--n-per-op", type=int, default=30)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument(
        "--free-generation-samples", type=int, default=0,
        help="run expensive greedy generation only for the first N samples")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--layer-ablation-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--device", default="auto",
        choices=("auto", "cuda", "mps", "cpu"))
    args = parser.parse_args()

    rows = (
        read_jsonl_rows(args.jsonl, args.max_samples)
        if args.jsonl else
        reference_rows(args.reference_protocol, args.n_per_op))
    if args.max_samples:
        rows = rows[:args.max_samples]
    candidates = candidate_sets(
        rows, max(args.candidate_count, 1), args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(
        weights, device=device, dtype=torch.bfloat16)
    if args.lora:
        backbone_lora, bundled_output = load_lora_bundle(
            args.lora, backbone.cfg, device=device)
        backbone.backbone_lora = backbone_lora
        if bundled_output is not None:
            backbone.output_lora = bundled_output
    memory = load_memory(args.memory_model, backbone, device)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)
    eos = tokenizer.eos()
    im_end = tokenizer.encode("<|im_end|>", add_bos=False)
    stop_ids = {eos}
    if len(im_end) == 1:
        stop_ids.add(im_end[0])

    results = []
    started = time.time()
    for sample_index, row in enumerate(rows):
        query_index = row.get("query_turn_id", len(row["messages"]) - 1)
        all_indices = list(range(query_index))
        evidence_indices = row.get("evidence_message_indices")
        distractor_indices = row.get("distractor_message_indices")
        prompt_ids, target_ids, target_text = query_material(
            row, tokenizer, eos)
        answer_candidates = candidates[row["_diagnostic_id"]]

        result = {
            "id": row["_diagnostic_id"],
            "group": row["_diagnostic_group"],
            "target": target_text,
            "evidence_indices": evidence_indices or [],
            "distractor_indices": distractor_indices or [],
        }

        commit_chunks(
            backbone, memory, row, tokenizer, all_indices)
        result["memory_all"] = score_sequence(
            backbone, memory, prompt_ids, target_ids)
        result["memory_all"].update(score_candidates(
            backbone, memory, prompt_ids, answer_candidates,
            tokenizer, eos, result["memory_all"]["nll"]))

        result["no_memory"] = score_sequence(
            backbone, None, prompt_ids, target_ids)
        result["no_memory"].update(score_candidates(
            backbone, None, prompt_ids, answer_candidates,
            tokenizer, eos, result["no_memory"]["nll"]))

        if sample_index < args.free_generation_samples:
            saved_M, saved_S = memory.M.clone(), memory.S.clone()
            prediction = generate(
                backbone, memory, prompt_ids, decoder, stop_ids,
                args.max_answer_tokens)
            category = row.get("category")
            if category in (1, 2, 3, 4, 5):
                free_score = official_like_score(
                    prediction,
                    {"category": category, "answer": target_text})
            else:
                free_score = f1_score(prediction, target_text)
            result["prediction"] = prediction
            result["free_f1"] = free_score
            memory.M, memory.S = saved_M, saved_S

        if evidence_indices is not None and evidence_indices != all_indices:
            commit_chunks(
                backbone, memory, row, tokenizer, evidence_indices)
            result["memory_evidence"] = score_sequence(
                backbone, memory, prompt_ids, target_ids)
            result["memory_evidence"].update(score_candidates(
                backbone, memory, prompt_ids, answer_candidates,
                tokenizer, eos, result["memory_evidence"]["nll"]))
        if distractor_indices:
            commit_chunks(
                backbone, memory, row, tokenizer, distractor_indices)
            result["memory_distractors"] = score_sequence(
                backbone, memory, prompt_ids, target_ids)
            result["memory_distractors"].update(score_candidates(
                backbone, memory, prompt_ids, answer_candidates,
                tokenizer, eos, result["memory_distractors"]["nll"]))

        context_prompt, context_target = full_context_material(
            row, tokenizer, eos)
        if len(context_prompt) + len(context_target) <= args.max_context_tokens:
            result["full_context"] = score_sequence(
                backbone, None, context_prompt, context_target)
            result["full_context"].update(score_candidates(
                backbone, None, context_prompt, answer_candidates,
                tokenizer, eos, result["full_context"]["nll"]))

        if sample_index < args.layer_ablation_samples:
            commit_chunks(
                backbone, memory, row, tokenizer, all_indices)
            layer_delta = []
            for slot, layer_id in enumerate(memory.layer_ids):
                mask = torch.ones(
                    memory.n_layers, device=device, dtype=torch.float32)
                mask[slot] = 0.0
                memory.layer_gate_override = mask
                ablated = score_sequence(
                    backbone, memory, prompt_ids, target_ids)
                layer_delta.append({
                    "layer_id": layer_id,
                    "nll_increase": (
                        ablated["nll"] - result["memory_all"]["nll"]),
                })
            memory.layer_gate_override = None
            result["layer_ablation"] = layer_delta

        results.append(result)
        print(json.dumps({
            "sample": sample_index + 1,
            "total": len(rows),
            "id": result["id"],
            "memory_nll": result["memory_all"]["nll"],
            "no_memory_nll": result["no_memory"]["nll"],
            "candidate_acc": result["memory_all"]["candidate_acc"],
            "free_f1": result.get("free_f1"),
        }), flush=True)

    payload = {
        "summary": aggregate(results),
        "seconds": time.time() - started,
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
