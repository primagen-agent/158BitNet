#!/usr/bin/env python3
"""Evaluate the frozen backbone with evidence in full context and no KV cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from eval_locomo import f1_score, official_like_score  # noqa: E402
from ggw import GGUFWeights  # noqa: E402
from torch_backbone import TorchBackbone  # noqa: E402
from train_data import CTokenDecoder, CTokenizer, render_chunk  # noqa: E402
from backbone_lora import load_lora_bundle  # noqa: E402


@torch.inference_mode()
def generate(backbone, prompt_ids, decoder, stop_ids, max_tokens):
    generated = []
    for _ in range(max_tokens):
        tokens = torch.tensor(
            prompt_ids + generated, device="cuda", dtype=torch.long)
        logits = backbone(tokens)
        next_id = int(logits.argmax().item())
        if next_id in stop_ids:
            break
        generated.append(next_id)
    return decoder.decode(generated).strip()


@torch.inference_mode()
def teacher_metrics(backbone, prompt_ids, target_ids):
    full = prompt_ids + target_ids[:-1]
    logits = backbone(
        torch.tensor(full, device="cuda", dtype=torch.long),
        logits_all=True)
    selected = logits[len(full) - len(target_ids): len(full)]
    target = torch.tensor(target_ids, device="cuda")
    nll = torch.nn.functional.cross_entropy(
        selected.float(), target, reduction="mean").item()
    prediction = selected.argmax(dim=-1)
    token_acc = (prediction == target).float().mean().item()
    exact = float(bool(torch.all(prediction == target)))
    return nll, token_acc, exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("jsonl")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument(
        "--evidence-only", action="store_true",
        help="place only labelled evidence messages plus the query in context")
    parser.add_argument(
        "--flatten-context", action="store_true",
        help="flatten evidence records and the question into one user turn")
    args = parser.parse_args()

    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(weights, device="cuda", dtype=torch.bfloat16)
    if args.lora:
        backbone_lora, bundled_output = load_lora_bundle(
            args.lora, backbone.cfg, device="cuda")
        backbone.backbone_lora = backbone_lora
        if bundled_output is not None:
            backbone.output_lora = bundled_output
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)
    eos = tokenizer.eos()
    im_end = tokenizer.encode("<|im_end|>", add_bos=False)
    stop_ids = {eos}
    if len(im_end) == 1:
        stop_ids.add(im_end[0])

    rows = []
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
        if len(rows) >= args.max_samples:
            break

    results = []
    started = time.time()
    for index, row in enumerate(rows):
        query_index = row.get("query_turn_id", len(row["messages"]) - 1)
        evidence_indices = {
            int(value)
            for value in row.get("evidence_message_indices", [])
        }
        if args.flatten_context:
            records = []
            selected_indices = (
                sorted(evidence_indices)
                if args.evidence_only else range(query_index))
            for chunk_index in selected_indices:
                for message in row["messages"][chunk_index]:
                    if message.get("role") != "user":
                        continue
                    content = str(message.get("content", ""))
                    content = content.replace(
                        "\n\nStore this conversation in long-term memory. "
                        "Reply OK.", "")
                    records.append(content)
            query_chunk = row["messages"][query_index]
            question = str(query_chunk[0].get("content", ""))
            target_text = str(query_chunk[-1].get("content", ""))
            flattened = [{
                "role": "user",
                "content": (
                    "Relevant conversation records:\n\n"
                    + "\n\n".join(records)
                    + "\n\n"
                    + question),
            }, {
                "role": "assistant",
                "content": target_text,
            }]
            text, target_text = render_chunk(flattened, True)
            prompt_ids = tokenizer.encode(text, add_bos=True)
        else:
            prompt_ids = []
            first = True
            for chunk_index, chunk in enumerate(row["messages"]):
                if chunk_index > query_index:
                    break
                is_query = chunk_index == query_index
                if (
                    args.evidence_only
                    and not is_query
                    and chunk_index not in evidence_indices
                ):
                    continue
                text, target_text = render_chunk(chunk, is_query)
                ids = tokenizer.encode(text, add_bos=first)
                first = False
                prompt_ids.extend(ids)
                if is_query:
                    break
        target_ids = tokenizer.encode(
            target_text, add_bos=False) + [eos]
        nll, token_acc, exact = teacher_metrics(
            backbone, prompt_ids, target_ids)
        prediction = generate(
            backbone, prompt_ids, decoder, stop_ids,
            args.max_answer_tokens)
        category = row.get("category")
        if category in (1, 2, 3, 4, 5):
            score = official_like_score(
                prediction,
                {"category": int(category), "answer": target_text})
        else:
            score = f1_score(prediction, target_text)
        results.append({
            "locomo_id": row.get("locomo_id"),
            "category": category,
            "answer": target_text,
            "prediction": prediction,
            "f1": score,
            "teacher_nll": nll,
            "teacher_token_acc": token_acc,
            "teacher_exact": exact,
        })
        print(
            f"[{index + 1}/{len(rows)}] f1={score:.4f} "
            f"prediction={prediction!r}",
            flush=True)

    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8")
    count = max(len(results), 1)
    print(json.dumps({
        "samples": len(results),
        "free_f1": sum(row["f1"] for row in results) / count,
        "teacher_exact": sum(
            row["teacher_exact"] for row in results) / count,
        "teacher_token_acc": sum(
            row["teacher_token_acc"] for row in results) / count,
        "teacher_nll": sum(row["teacher_nll"] for row in results) / count,
        "seconds": time.time() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
