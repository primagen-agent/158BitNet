#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-models/bitcpm4-0.5b-tq2_0.gguf}"
data_dir="${DATA_DIR:-build/memory_05b_training_data}"
output="${OUTPUT:-build/memory_05b.bnmem}"
final_output="${FINAL_OUTPUT:-build/memory_05b.final.bnmem}"
steps="${STEPS:-1792}"
init_memory="${INIT_MEMORY:-}"

if [[ ! -d "$data_dir/train" || ! -d "$data_dir/valid" ]]; then
  python3 python/prepare_memory_curriculum.py "$data_dir" \
    --train 1024 \
    --valid 128 \
    --stage2-train 1024 \
    --stage2-valid 128 \
    --stage3-train 768 \
    --stage3-valid 96 \
    --long-train 2048 \
    --long-valid 256 \
    --seed 20260905
fi

extra_args=()
if [[ -n "$init_memory" ]]; then
  extra_args+=(--init-memory "$init_memory")
fi

python3 python/train_memory.py \
  "$gguf" "$data_dir/train" \
  --valid-data "$data_dir/valid" \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --output "$output" \
  --final-output "$final_output" \
  --device auto \
  --layers all \
  --gamma 0.9 \
  --fusion-mode residual_gate \
  --tau 1.0 \
  --rho 0.9 \
  --alpha-max-tokens 16 \
  --alpha-max-fraction 0.2 \
  --beta-scale 0.9 \
  --gdu-alpha-init 0.99 \
  --gdu-beta-init 0.5 \
  --query-rank 0 \
  --kv-rank 0 \
  --query-mode independent \
  --denom-mode signed_plus_one \
  --state-mode delta \
  --retrieval-windows 0 \
  --samples 8448 \
  --valid 1056 \
  --steps "$steps" \
  --batch 32 \
  --lr 2e-4 \
  --wd 0.01 \
  --warmup 16 \
  --valid-every 16 \
  --valid-subset 128 \
  --self-prefix-prob 0.1 \
  --contrastive-lambda 0.5 \
  --contrastive-margin 1.0 \
  --contrastive-negatives 1 \
  --evidence-lambda 0 \
  --write-selection-lambda 0.5 \
  --fusion-gate-lambda 0.2 \
  --query-gate-lambda 0 \
  --kv-gate-lambda 0 \
  --layer-gate-lambda 0 \
  --task0-weight-start 0.25 \
  --task0-weight-end 0.10 \
  --task1-weight-start 0.35 \
  --task1-weight-end 0.25 \
  --task2-weight-start 0.20 \
  --task2-weight-end 0.30 \
  --task3-weight-start 0.10 \
  --task3-weight-end 0.20 \
  --task4-weight-start 0.10 \
  --task4-weight-end 0.15 \
  --output-lora-rank 0 \
  --backbone-lora-rank 0 \
  ${extra_args[@]+"${extra_args[@]}"} \
  --seed 20260908
