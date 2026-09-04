#!/usr/bin/env python3
"""Compare teacher-forced and free generation on memory-training JSONL."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from eval_bnmem_torch_locomo import (  # noqa: E402
    CTokenDecoder,
    CTokenizer,
    GGUFWeights,
    MetisMemory,
    TorchBackbone,
    forward_chunk,
    generate,
    load_memory,
)
from eval_locomo import f1_score, official_like_score  # noqa: E402
from train_data import render_chunk  # noqa: E402
from output_lora import load_output_lora  # noqa: E402
from backbone_lora import load_lora_bundle  # noqa: E402
from answer_decoder import MemoryAnswerDecoder  # noqa: E402
from train_memory_layer_router import (  # noqa: E402
    LayerRouter,
    attention_stats,
)


@torch.inference_mode()
def teacher_metrics(backbone, memory, prompt_ids, target_ids):
    full = prompt_ids + target_ids[:-1]
    logits = backbone(
        torch.tensor(full, device="cuda", dtype=torch.long),
        memory_v6=memory, logits_all=True)
    memory.discard_captured()
    selected = logits[len(full) - len(target_ids): len(full)]
    target = torch.tensor(target_ids, device="cuda")
    nll = torch.nn.functional.cross_entropy(
        selected.float(), target, reduction="mean").item()
    prediction = selected.argmax(dim=-1)
    token_acc = (prediction == target).float().mean().item()
    exact = float(bool(torch.all(prediction == target)))
    return nll, token_acc, exact


@torch.inference_mode()
def retrieve_memory_excerpts(backbone, memory, prompt_ids, decoder,
                             window_count, window_size, layer_mode,
                             layer_index=-1, layer_router=None,
                             exclude_texts=None):
    """Select non-overlapping source-token windows using memory attention.

    The selection uses only the model's own final-layer slot attention.  It
    does not inspect evidence labels or target answers.
    """
    hidden = backbone(
        torch.tensor(prompt_ids, device="cuda", dtype=torch.long),
        memory_v6=memory, return_hidden=layer_router is not None)
    memory.discard_captured()
    weights = getattr(memory, "last_pointer_weights", None)
    layer_weights = [
        item for item in getattr(
            memory, "last_pointer_weights_by_layer", [])
        if item is not None]
    all_layer_weights = getattr(
        memory, "last_pointer_weights_by_layer", [])
    if layer_router is not None:
        if not all_layer_weights or any(
            value is None for value in all_layer_weights
        ):
            return []
        stats = torch.cat([
            attention_stats(value) for value in all_layer_weights])
        feature = torch.cat((hidden[-1].float(), stats)).unsqueeze(0)
        with torch.no_grad():
            layer_index = int(
                layer_router(feature).argmax(dim=-1).item())
        weights = all_layer_weights[layer_index]
    elif layer_index >= 0:
        if layer_index >= len(all_layer_weights):
            raise ValueError(
                f"retrieval layer {layer_index} is outside "
                f"[0, {len(all_layer_weights) - 1}]")
        weights = all_layer_weights[layer_index]
    elif layer_mode != "last" and layer_weights:
        stacked = torch.stack(layer_weights)
        if layer_mode == "mean":
            weights = stacked.mean(dim=0)
        elif layer_mode == "max":
            weights = stacked.max(dim=0).values
        elif layer_mode == "rrf":
            final_scores = stacked[:, -1, :]
            order = final_scores.argsort(dim=-1, descending=True)
            ranks = order.argsort(dim=-1).float()
            weights = (
                1.0 / (60.0 + ranks + 1.0)
            ).sum(dim=0, keepdim=True)
    token_ids = getattr(memory, "pointer_token_ids", None)
    if (
        weights is None
        or token_ids is None
        or weights.shape[-1] != token_ids.numel()
        or token_ids.numel() == 0
    ):
        return []
    scores = weights[-1].float().clone()
    scores[token_ids < 0] = -float("inf")
    radius = max(window_size // 2, 1)
    windows = []
    excluded = set(exclude_texts or ())
    blocked = torch.zeros_like(scores, dtype=torch.bool)
    max_attempts = window_count + max(len(excluded) * 2, 4)
    for _ in range(max_attempts):
        if len(windows) >= window_count:
            break
        candidate_scores = scores.masked_fill(blocked, -float("inf"))
        center = int(candidate_scores.argmax().item())
        if not torch.isfinite(candidate_scores[center]):
            break
        start = max(0, center - radius)
        end = min(token_ids.numel(), start + window_size)
        start = max(0, end - window_size)
        ids = token_ids[start:end]
        ids = ids[ids >= 0].tolist()
        if ids:
            text = decoder.decode(ids)
            text = re.sub(r"<\|im_(?:start|end)\|>", " ", text)
            text = re.sub(
                r"Store this conversation in long-term memory\."
                r"\s*Reply OK\.?", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\b(?:assistant|user|OK)\b", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text and text not in excluded and text not in windows:
                windows.append(text)
        blocked[max(0, start - radius):min(
            token_ids.numel(), end + radius)] = True
    return windows


def add_retrieved_excerpts(prompt, excerpts):
    if not excerpts:
        return prompt
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    if marker not in prompt:
        raise ValueError("query prompt is missing the assistant marker")
    memory_text = "\nRetrieved memory excerpts:\n" + "\n".join(
        f"- {text}" for text in excerpts)
    return prompt.replace(
        marker, memory_text + "<|im_end|>\n<|im_start|>assistant\n", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("memory_model")
    parser.add_argument("jsonl")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--categories", default="1,2,3,4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-lora", default=None)
    parser.add_argument("--lora", default=None,
                        help="BNLORA1 backbone/output adapter bundle")
    parser.add_argument(
        "--answer-decoder", default=None,
        help="BNANSWER1 nonlinear memory answer decoder")
    parser.add_argument(
        "--state-mode", choices=("delta", "slots"), default="delta")
    parser.add_argument("--max-memory-slots", type=int, default=4096)
    parser.add_argument("--slot-temperature", type=float, default=0.07)
    parser.add_argument("--pointer-mix", type=float, default=0.0)
    parser.add_argument(
        "--retrieval-windows", type=int, default=0,
        help="number of model-selected token windows inserted into the query")
    parser.add_argument(
        "--retrieval-window-size", type=int, default=32,
        help="tokens per model-selected retrieval window")
    parser.add_argument(
        "--retrieval-rounds", type=int, default=1,
        help="iterative retrieval passes; later passes condition on excerpts "
             "selected by earlier passes")
    parser.add_argument(
        "--retrieval-layer-mode",
        choices=("last", "mean", "max", "rrf"), default="last",
        help="aggregate slot attention across memory layers")
    parser.add_argument(
        "--retrieval-layer-index", type=int, default=-1,
        help="use one zero-based memory layer instead of layer aggregation")
    parser.add_argument(
        "--layer-router", default=None,
        help="BNROUTER1 question-conditioned retrieval layer router")
    parser.add_argument(
        "--gamma-override", type=float, default=None,
        help="override checkpoint memory blend gamma for diagnostics")
    args = parser.parse_args()
    if not 0.0 <= args.pointer_mix <= 1.0:
        raise ValueError("--pointer-mix must be in [0, 1]")
    if args.retrieval_windows < 0:
        raise ValueError("--retrieval-windows must be non-negative")
    if args.retrieval_window_size < 1:
        raise ValueError("--retrieval-window-size must be positive")
    if args.retrieval_rounds < 1:
        raise ValueError("--retrieval-rounds must be positive")

    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(weights, device="cuda", dtype=torch.bfloat16)
    if args.lora:
        backbone_lora, bundled_output = load_lora_bundle(
            args.lora, backbone.cfg, device="cuda")
        backbone.backbone_lora = backbone_lora
        if bundled_output is not None:
            backbone.output_lora = bundled_output
    if args.output_lora:
        backbone.output_lora = load_output_lora(
            args.output_lora, backbone.cfg.hidden, backbone.cfg.vocab)
    if args.answer_decoder:
        backbone.answer_decoder = MemoryAnswerDecoder.load(
            args.answer_decoder, backbone.cfg.hidden, device="cuda")
    layer_router = (
        LayerRouter.load(
            args.layer_router, device="cuda", gguf_path=args.gguf)
        if args.layer_router else None)
    memory = load_memory(
        args.memory_model, backbone,
        state_mode=args.state_mode,
        max_memory_slots=args.max_memory_slots,
        slot_temperature=args.slot_temperature)
    if args.gamma_override is not None:
        if not 0.0 <= args.gamma_override <= 1.0:
            raise ValueError("--gamma-override must be in [0, 1]")
        memory.gamma = args.gamma_override
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)
    eos = tokenizer.eos()
    im_end = tokenizer.encode("<|im_end|>", add_bos=False)
    stop_ids = {eos}
    if len(im_end) == 1:
        stop_ids.add(im_end[0])

    wanted_categories = {
        int(value) for value in args.categories.split(",") if value.strip()}
    rows, seen = [], set()
    for line_index, line in enumerate(
        Path(args.jsonl).read_text(encoding="utf-8").splitlines()
    ):
        row = json.loads(line)
        has_category = "category" in row
        if has_category and int(row["category"]) not in wanted_categories:
            continue
        locomo_id = str(row.get("locomo_id", ""))
        if locomo_id:
            base_id = ("locomo", ":".join(locomo_id.split(":")[:2]))
        elif row.get("sample_id"):
            base_id = ("sample", str(row["sample_id"]))
        else:
            base_id = ("line", line_index)
        if base_id in seen:
            continue
        seen.add(base_id)
        rows.append(row)
        if len(rows) >= args.max_samples:
            break

    results = []
    started = time.time()
    for index, row in enumerate(rows):
        memory.reset_state()
        query_index = row.get("query_turn_id", len(row["messages"]) - 1)
        for chunk_index, chunk in enumerate(row["messages"]):
            if chunk_index == query_index:
                break
            text, _ = render_chunk(chunk, False)
            ids = tokenizer.encode(text, add_bos=True)
            forward_chunk(backbone, memory, ids)
            memory.commit_all()

        query_chunk = row["messages"][query_index]
        prompt, target_text = render_chunk(query_chunk, True)
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        excerpts = []
        if args.retrieval_windows:
            for _round in range(args.retrieval_rounds):
                round_prompt = add_retrieved_excerpts(prompt, excerpts)
                round_prompt_ids = tokenizer.encode(
                    round_prompt, add_bos=True)
                additions = retrieve_memory_excerpts(
                    backbone, memory, round_prompt_ids, decoder,
                    args.retrieval_windows, args.retrieval_window_size,
                    args.retrieval_layer_mode, args.retrieval_layer_index,
                    layer_router, exclude_texts=excerpts)
                if not additions:
                    break
                excerpts.extend(additions)
            prompt = add_retrieved_excerpts(prompt, excerpts)
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
        target_ids = tokenizer.encode(target_text, add_bos=False) + [eos]
        saved_state = memory.clone_runtime_state()

        nll, token_acc, exact = teacher_metrics(
            backbone, memory, prompt_ids, target_ids)
        memory.restore_runtime_state(saved_state)
        prediction = generate(
            backbone, memory, prompt_ids, decoder, stop_ids,
            args.max_answer_tokens, args.pointer_mix)
        category = int(row["category"]) if "category" in row else None
        if category in (1, 2, 3, 4, 5):
            score = official_like_score(
                prediction, {"category": category, "answer": target_text})
        else:
            score = f1_score(prediction, target_text)
        result = {
            "locomo_id": row.get("locomo_id"),
            "category": category,
            "answer": target_text,
            "prediction": prediction,
            "retrieved_excerpts": excerpts,
            "f1": score,
            "teacher_nll": nll,
            "teacher_token_acc": token_acc,
            "teacher_exact": exact,
        }
        results.append(result)
        print(
            f"[{index + 1}/{len(rows)}] f1={score:.4f} "
            f"teacher_exact={exact:.0f} prediction={prediction!r}",
            flush=True)

    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({
        "samples": len(results),
        "free_f1": sum(x["f1"] for x in results) / max(len(results), 1),
        "teacher_exact": sum(
            x["teacher_exact"] for x in results) / max(len(results), 1),
        "teacher_token_acc": sum(
            x["teacher_token_acc"] for x in results) / max(len(results), 1),
        "teacher_nll": sum(
            x["teacher_nll"] for x in results) / max(len(results), 1),
        "seconds": time.time() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
