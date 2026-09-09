#!/usr/bin/env python3
"""GPU LoCoMo generation evaluator for unified BNMEM1 checkpoints."""

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from eval_locomo import chunk_turns, official_like_score  # noqa: E402
from ggw import GGUFWeights  # noqa: E402
from torch_backbone import TorchBackbone  # noqa: E402
from train_data import CTokenDecoder, CTokenizer, render_chunk  # noqa: E402
from train_memory import MetisMemory  # noqa: E402
from backbone_lora import load_lora_bundle  # noqa: E402
from bnmem_export import load_bnmem_v1  # noqa: E402
from model_identity import require_matching_sha256  # noqa: E402


def load_memory(path, backbone, device=None, state_mode="delta",
                max_memory_slots=4096, slot_temperature=0.07):
    device = device or backbone.device
    checkpoint = load_bnmem_v1(path)
    require_matching_sha256(
        "memory model", checkpoint["backbone_sha256"],
        backbone.gguf_path)
    if (
        checkpoint["d_model"],
        checkpoint["kv_dim"],
        checkpoint["q_dim"],
        checkpoint["head_dim"],
    ) != (
        backbone.cfg.hidden,
        backbone.cfg.kv_dim,
        backbone.cfg.q_dim,
        backbone.cfg.head_dim,
    ):
        raise ValueError("checkpoint/backbone geometry mismatch")

    denom_name = (
        "abs_plus_one"
        if checkpoint["denom_mode"] == 2 else "signed_plus_one")
    query_mode = (
        "backbone_delta"
        if checkpoint["query_add_backbone"] else "independent")
    memory = MetisMemory(
        backbone, checkpoint["layer_ids"],
        gamma=checkpoint["gamma"], tau=checkpoint["tau"],
        rho=checkpoint["rho"], k_min=checkpoint["k_min"],
        alpha_max_tokens=checkpoint.get("alpha_max_tokens", 0),
        alpha_max_fraction=checkpoint.get("alpha_max_fraction", 0.0),
        beta_scale=checkpoint["beta_scale"],
        query_rank=checkpoint["query_rank"], query_mode=query_mode,
        kv_rank=checkpoint["kv_rank"], denom_mode=denom_name,
        query_gate_lambda=0.0, kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        fusion_mode=checkpoint.get("fusion_mode", "fixed"),
        state_mode=state_mode, max_memory_slots=max_memory_slots,
        slot_temperature=slot_temperature,
        device=device, dtype=torch.float32, seed=0)
    arrays = dict(checkpoint["tensors"])
    with torch.no_grad():
        for name, value in arrays.items():
            getattr(memory, name).copy_(
                value.to(device=device, dtype=getattr(memory, name).dtype))
    return memory


def memory_chunk(date, turns, part, total):
    transcript = "\n".join(
        f"{turn['speaker']}: {turn['text']}" for turn in turns)
    content = (
        f"Conversation session on {date} (part {part}/{total}):\n"
        f"{transcript}\n\n"
        "Store this conversation in long-term memory. Reply OK.")
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "OK"},
    ]


def query_prompt(question):
    content = (
        "Answer the question using only the stored conversation memory. "
        "Give only the shortest direct answer. If the conversation does not "
        "contain the answer, reply exactly: No information available.\n"
        f"Question: {question}")
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": ""},
    ]


@torch.inference_mode()
def forward_chunk(backbone, memory, token_ids):
    tokens = torch.tensor(
        token_ids, device=backbone.device, dtype=torch.long)
    backbone(tokens, memory_v6=memory)
    if memory.state_mode == "slots":
        previous = getattr(memory, "pointer_token_ids", None)
        current = torch.tensor(
            token_ids, device=backbone.device, dtype=torch.long)
        if current.numel() > 0:
            current = current.clone()
            current[0] = -1
        if previous is None:
            previous = current.new_empty((0,))
        memory.pointer_token_ids = torch.cat(
            (previous, current))[-memory.max_memory_slots:]


@torch.inference_mode()
def generate(backbone, memory, prompt_ids, decoder, stop_ids, max_tokens,
             pointer_mix=0.0):
    generated = []
    for _ in range(max_tokens):
        tokens = torch.tensor(
            prompt_ids + generated,
            device=backbone.device, dtype=torch.long)
        logits = backbone(tokens, memory_v6=memory)
        memory.discard_captured()
        scores = logits.float()
        pointer_weights = getattr(memory, "last_pointer_weights", None)
        pointer_ids = getattr(memory, "pointer_token_ids", None)
        if (
            pointer_mix > 0.0
            and pointer_weights is not None
            and pointer_ids is not None
            and pointer_weights.shape[-1] == pointer_ids.numel()
        ):
            pointer = torch.zeros_like(scores)
            valid = pointer_ids >= 0
            pointer.scatter_add_(
                0, pointer_ids[valid],
                pointer_weights[-1, valid].float())
            for stop_id in stop_ids:
                pointer[stop_id] = 0.0
            pointer_total = pointer.sum()
            if pointer_total > 0:
                pointer = pointer / pointer_total
                scores = (
                    (1.0 - pointer_mix) * scores.softmax(dim=-1)
                    + pointer_mix * pointer)
        token = int(scores.argmax().item())
        if token in stop_ids:
            break
        generated.append(token)
    return decoder.decode(generated).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("memory_model")
    parser.add_argument("dataset")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument(
        "--device", default="cuda",
        choices=("cuda", "mps", "cpu"))
    parser.add_argument(
        "--state-mode", choices=("delta", "slots"), default="delta")
    parser.add_argument("--max-memory-slots", type=int, default=4096)
    parser.add_argument("--slot-temperature", type=float, default=0.07)
    parser.add_argument("--pointer-mix", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.pointer_mix <= 1.0:
        raise ValueError("--pointer-mix must be in [0, 1]")

    t0 = time.time()
    weights = GGUFWeights(args.gguf, args.lib)
    backbone = TorchBackbone(
        weights, device=args.device, dtype=torch.bfloat16)
    if args.lora:
        backbone_lora, bundled_output = load_lora_bundle(
            args.lora, backbone.cfg, device=args.device)
        backbone.backbone_lora = backbone_lora
        if bundled_output is not None:
            backbone.output_lora = bundled_output
    memory = load_memory(
        args.memory_model, backbone, args.device,
        state_mode=args.state_mode,
        max_memory_slots=args.max_memory_slots,
        slot_temperature=args.slot_temperature)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    decoder = CTokenDecoder(args.tok_probe, args.gguf)
    eos = tokenizer.eos()
    im_end = tokenizer.encode("<|im_end|>", add_bos=False)
    stop_ids = {eos}
    if len(im_end) == 1:
        stop_ids.add(im_end[0])

    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    sample = samples[args.conversation]
    conversation = sample["conversation"]
    session_keys = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.split("_")[1]))

    memory.reset_state()
    for si, key in enumerate(session_keys):
        date = conversation.get(key + "_date_time", "unknown date")
        chunks = chunk_turns(conversation[key])
        for index, turns in enumerate(chunks):
            messages = memory_chunk(date, turns, index + 1, len(chunks))
            text, _ = render_chunk(messages, False)
            ids = tokenizer.encode(text, add_bos=True)
            forward_chunk(backbone, memory, ids)
            memory.commit_all()
        print(f"[session {si + 1}/{len(session_keys)}]", flush=True)

    saved_state = memory.clone_runtime_state()
    results = []
    qas = sample["qa"][:args.max_questions or None]
    for qi, qa in enumerate(qas):
        memory.restore_runtime_state(saved_state)
        messages = query_prompt(qa["question"])
        text, _ = render_chunk(messages, True)
        ids = tokenizer.encode(text, add_bos=True)
        prediction = generate(
            backbone, memory, ids, decoder, stop_ids,
            args.max_answer_tokens, args.pointer_mix)
        score = official_like_score(prediction, qa)
        results.append({
            "question": qa["question"],
            "answer": qa.get("answer"),
            "category": int(qa["category"]),
            "prediction": prediction,
            "f1": score,
        })
        print(
            f"[QA {qi + 1}/{len(qas)}] f1={score:.4f} "
            f"prediction={prediction!r}", flush=True)

    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8")
    by_category = collections.defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row["f1"])
    official = [
        row["f1"] for row in results if row["category"] in (1, 2, 3, 4)]
    summary = {
        "official_categories_1_4": sum(official) / max(len(official), 1),
        "questions": len(results),
        "seconds": time.time() - t0,
        "by_category": {
            str(key): sum(values) / len(values)
            for key, values in sorted(by_category.items())
        },
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
