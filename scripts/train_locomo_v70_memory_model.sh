#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-models/bitcpm4-3b-tq2_0.gguf}"
train_data="${TRAIN_DATA:-build/locomo_retrieval_hardneg_v45/train}"
valid_data="${VALID_DATA:-build/locomo_retrieval_hardneg_v45/valid}"
output="${OUTPUT:-build/locomo_v70_memory_model_delta.bnmem}"
final_output="${FINAL_OUTPUT:-build/locomo_v70_memory_model_delta.final.bnmem}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python python/train_memory.py \
  "$gguf" "$train_data" \
  --valid-data "$valid_data" \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --output "$output" \
  --final-output "$final_output" \
  --layers all \
  --gamma 0.975 \
  --tau 1.0 \
  --rho 0.9 \
  --beta-scale 0.9 \
  --query-rank 64 \
  --kv-rank 64 \
  --query-mode backbone_delta \
  --denom-mode signed_plus_one \
  --state-mode delta \
  --retrieval-windows 0 \
  --samples 6852 \
  --valid 390 \
  --steps 12000 \
  --batch 1 \
  --lr 5e-5 \
  --wd 0.01 \
  --warmup 200 \
  --valid-every 500 \
  --valid-subset 64 \
  --self-prefix-prob 0 \
  --contrastive-lambda 1.0 \
  --contrastive-margin 1.0 \
  --contrastive-negatives 1 \
  --evidence-lambda 0 \
  --query-gate-lambda 1e-6 \
  --kv-gate-lambda 1e-6 \
  --layer-gate-lambda 1e-6 \
  --nas-lr 1e-3 \
  --nas-warmup 1000 \
  --nas-every 8 \
  --output-lora-rank 0 \
  --backbone-lora-rank 0 \
  --seed 20260904
