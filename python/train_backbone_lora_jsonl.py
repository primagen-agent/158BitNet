#!/usr/bin/env python3
"""Train BNLORA1 Q/V/O adapters with all evidence in one no-KV context."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from backbone_lora import (  # noqa: E402
    BackboneLoRA,
    load_lora_bundle,
    save_lora_bundle,
)
from ggw import GGUFWeights  # noqa: E402
from torch_backbone import TorchBackbone  # noqa: E402
from train_data import (  # noqa: E402
    CTokenizer,
    dataset_order,
    load_dataset,
    render_chunk,
)
from output_lora import OutputLoRA  # noqa: E402


class FullContextLoRATrainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        self.device = "cuda"

        print('{"phase":"load_backbone"}', flush=True)
        started = time.time()
        weights = GGUFWeights(args.gguf, args.lib)
        self.backbone = TorchBackbone(
            weights, device="cuda", dtype=torch.bfloat16)
        if args.init_lora:
            self.lora, output_lora = load_lora_bundle(
                args.init_lora, self.backbone.cfg, device="cuda")
        else:
            if args.lora_blocks == "all":
                blocks = list(range(self.backbone.cfg.n_layers))
            elif args.lora_blocks.startswith("last:"):
                count = int(args.lora_blocks.split(":", 1)[1])
                blocks = list(range(self.backbone.cfg.n_layers - count,
                                    self.backbone.cfg.n_layers))
            else:
                blocks = [
                    int(value) for value in args.lora_blocks.split(",")
                    if value.strip()
                ]
            targets = tuple(
                value.strip() for value in args.lora_targets.split(",")
                if value.strip())
            self.lora = BackboneLoRA(
                self.backbone.cfg, blocks, targets,
                rank=args.lora_rank, alpha=args.lora_alpha, device="cuda")
            output_lora = None
        if output_lora is None and args.output_lora_rank > 0:
            output_lora = OutputLoRA(
                self.backbone.cfg.hidden,
                self.backbone.cfg.vocab,
                args.output_lora_rank,
                args.output_lora_alpha,
                device="cuda")
        self.output_lora = output_lora
        if self.output_lora is not None:
            self.backbone.output_lora = self.output_lora
        train_blocks = self._parse_blocks(
            args.train_lora_blocks, self.backbone.cfg.n_layers)
        for projection in self.lora.tensors():
            trainable = projection.block_index in train_blocks
            for parameter in projection.parameters():
                parameter.requires_grad_(trainable)
        self.trainable_parameters = [
            parameter for parameter in self.lora.parameters()
            if parameter.requires_grad
        ]
        if self.output_lora is not None:
            self.trainable_parameters.extend(
                self.output_lora.parameters())
        if not self.trainable_parameters:
            raise ValueError("no trainable LoRA parameters selected")
        self.backbone.backbone_lora = self.lora
        print(
            f'{{"phase":"backbone_loaded","sec":{time.time()-started:.1f},'
            f'"lora_tensors":{len(self.lora.tensors())},'
            f'"output_lora":{str(self.output_lora is not None).lower()},'
            f'"trainable_parameters":'
            f'{sum(p.numel() for p in self.trainable_parameters)}}}',
            flush=True)

        self.tokenizer = CTokenizer(args.tok_probe, args.gguf)
        self.eos = self.tokenizer.eos()
        self.train_strata = load_dataset(args.data)
        self.valid_strata = load_dataset(args.valid_data)
        train_order = dataset_order(self.train_strata, args.seed)
        valid_order = dataset_order(self.valid_strata, args.seed + 1)
        if len(train_order) < args.samples:
            raise ValueError(
                f"train data too small: {len(train_order)} < {args.samples}")
        self.train_order = train_order[:args.samples]
        self.valid_order = self._dedupe(valid_order)[:args.valid]
        if len(self.valid_order) < args.valid:
            raise ValueError(
                f"valid data too small: {len(self.valid_order)} < {args.valid}")
        print(
            f'{{"phase":"data_split","train":{len(self.train_order)},'
            f'"valid":{len(self.valid_order)}}}', flush=True)

        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters, lr=args.lr, weight_decay=args.wd)
        self.best_valid = float("inf")

    @staticmethod
    def _parse_blocks(spec, n_layers):
        if spec == "all":
            return set(range(n_layers))
        if spec.startswith("last:"):
            count = int(spec.split(":", 1)[1])
            if count < 1 or count > n_layers:
                raise ValueError("invalid last:N LoRA block selection")
            return set(range(n_layers - count, n_layers))
        blocks = {
            int(value) for value in spec.split(",") if value.strip()}
        if not blocks or min(blocks) < 0 or max(blocks) >= n_layers:
            raise ValueError("invalid LoRA block selection")
        return blocks

    def _dedupe(self, order):
        output, seen = [], set()
        for stratum, index in order:
            row = json.loads(self.valid_strata[stratum][1][index])
            locomo_id = str(row.get("locomo_id", ""))
            if locomo_id:
                key = ":".join(locomo_id.split(":")[:2])
            else:
                key = str(row.get("sample_id", (stratum, index)))
            if key in seen:
                continue
            seen.add(key)
            output.append((stratum, index))
        return output

    def encode(self, row, cache):
        query_index = row.get(
            "query_turn_id", len(row["messages"]) - 1)
        prompt_ids = []
        target = None
        first = True
        for chunk_index, chunk in enumerate(row["messages"]):
            if chunk_index > query_index:
                break
            is_query = chunk_index == query_index
            text, maybe_target = render_chunk(chunk, is_query)
            key = ("b" if first else "n", text)
            ids = cache.get(key)
            if ids is None:
                ids = self.tokenizer.encode(text, add_bos=first)
                cache[key] = ids
            prompt_ids.extend(ids)
            first = False
            if is_query:
                target = maybe_target
                break
        if target is None:
            raise ValueError("sample has no query target")
        target_key = ("target", target)
        target_ids = cache.get(target_key)
        if target_ids is None:
            target_ids = (
                self.tokenizer.encode(target, add_bos=False) + [self.eos])
            cache[target_key] = target_ids
        if len(prompt_ids) + len(target_ids) > self.args.max_tokens:
            raise ValueError("sample exceeds --max-tokens")
        return prompt_ids, target_ids

    def loss_for(self, row, cache):
        prompt_ids, target_ids = self.encode(row, cache)
        full = prompt_ids + target_ids[:-1]
        logits = self.backbone(
            torch.tensor(full, device="cuda", dtype=torch.long),
            logits_all=True)
        selected = logits[len(full) - len(target_ids): len(full)]
        target = torch.tensor(target_ids, device="cuda")
        loss = F.cross_entropy(selected.float(), target, reduction="mean")
        prediction = selected.argmax(dim=-1)
        correct = int((prediction == target).sum().item())
        exact = int(correct == len(target_ids))
        return loss, len(target_ids), correct, exact

    @torch.no_grad()
    def validate(self, cache):
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        exact = 0
        samples = 0
        for stratum, index in self.valid_order[:self.args.valid_subset]:
            row = json.loads(self.valid_strata[stratum][1][index])
            try:
                loss, tokens, sample_correct, sample_exact = self.loss_for(
                    row, cache)
            except (ValueError, RuntimeError):
                continue
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            correct += sample_correct
            exact += sample_exact
            samples += 1
        return {
            "loss": total_loss / max(total_tokens, 1),
            "token_acc": correct / max(total_tokens, 1),
            "exact_acc": exact / max(samples, 1),
            "tokens": total_tokens,
            "samples": samples,
        }

    def save(self, path):
        tensors = self.lora.tensors()
        if self.output_lora is not None:
            tensors.append(self.output_lora)
        save_lora_bundle(path, tensors)
        print(
            f'{{"exported_lora":"{path}",'
            f'"tensors":{len(tensors)}}}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("data")
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-output", default=None)
    parser.add_argument("--init-lora", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-targets", default="q,v,o")
    parser.add_argument("--lora-blocks", default="all")
    parser.add_argument("--output-lora-rank", type=int, default=0)
    parser.add_argument("--output-lora-alpha", type=float, default=16.0)
    parser.add_argument(
        "--train-lora-blocks", default="all",
        help="subset of loaded adapters to train, e.g. last:8")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--valid", type=int, default=256)
    parser.add_argument("--valid-subset", type=int, default=128)
    parser.add_argument("--valid-every", type=int, default=250)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trainer = FullContextLoRATrainer(args)
    cache = {}
    started = time.time()
    cursor = 0
    step = 0
    attempts = 0
    max_attempts = args.steps * 20
    while step < args.steps and attempts < max_attempts:
        attempts += 1
        trainer.optimizer.zero_grad(set_to_none=True)
        stratum, index = trainer.train_order[cursor]
        cursor = (cursor + 1) % len(trainer.train_order)
        row = json.loads(trainer.train_strata[stratum][1][index])
        try:
            loss, tokens, _correct, _exact = trainer.loss_for(row, cache)
        except ValueError as exc:
            print(
                f'{{"step":{step},"skip":"{type(exc).__name__}:'
                f'{str(exc)[:100]}"}}', flush=True)
            continue
        except torch.OutOfMemoryError as exc:
            trainer.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f'{{"step":{step},"skip":"OutOfMemoryError:'
                f'{str(exc)[:100]}"}}', flush=True)
            continue
        objective = loss
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            trainer.trainable_parameters, args.clip)
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
        trainer.optimizer.step()
        speed = (step + 1) / max(time.time() - started, 1e-6) * 3600
        print(
            f'{{"step":{step},"split":"train",'
            f'"loss":{float(loss.item()):.6f},"tokens":{tokens},'
            f'"steps_per_hour":{speed:.0f}}}', flush=True)

        if (step + 1) % args.valid_every == 0:
            metrics = trainer.validate(cache)
            print(
                f'{{"step":{step},"split":"valid",'
                f'"loss":{metrics["loss"]:.6f},'
                f'"token_acc":{metrics["token_acc"]:.6f},'
                f'"exact_acc":{metrics["exact_acc"]:.6f},'
                f'"tokens":{metrics["tokens"]},'
                f'"samples":{metrics["samples"]}}}', flush=True)
            if metrics["loss"] < trainer.best_valid:
                trainer.best_valid = metrics["loss"]
                trainer.save(args.output)
        step += 1

    if step < args.steps:
        raise RuntimeError(
            f"only completed {step}/{args.steps} updates after "
            f"{attempts} attempts")

    if trainer.best_valid == float("inf"):
        trainer.save(args.output)
    if args.final_output:
        trainer.save(args.final_output)
    print('{"result":"done"}', flush=True)


if __name__ == "__main__":
    main()
