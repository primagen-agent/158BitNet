#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-models/bitcpm4-0.5b-tq2_0.gguf}"
data_dir="${DATA_DIR:-build/memory_v73_official_tasks_data}"
output="${OUTPUT:-build/memory_v73_05b_official_tasks.bnmem}"
final_output="${FINAL_OUTPUT:-build/memory_v73_05b_official_tasks.final.bnmem}"

if [[ ! -d "$data_dir/train" || ! -d "$data_dir/valid" ]]; then
  python3 python/prepare_memory_curriculum.py "$data_dir" \
    --train 1024 \
    --valid 128 \
    --stage2-train 1024 \
    --stage2-valid 128 \
    --stage3-train 1024 \
    --stage3-valid 128 \
    --seed 20260905
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  --tau 1.0 \
  --rho 0.9 \
  --beta-scale 0.9 \
  --query-rank 0 \
  --kv-rank 0 \
  --query-mode independent \
  --denom-mode signed_plus_one \
  --state-mode delta \
  --retrieval-windows 0 \
  --samples 7168 \
  --valid 896 \
  --steps 1792 \
  --batch 8 \
  --lr 2e-4 \
  --wd 0.01 \
  --warmup 25 \
  --valid-every 64 \
  --valid-subset 128 \
  --self-prefix-prob 0 \
  --contrastive-lambda 0 \
  --evidence-lambda 0 \
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
  --seed 20260905
