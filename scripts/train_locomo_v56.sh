#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-models/bitcpm4-3b-tq2_0.gguf}"
train_data="${TRAIN_DATA:-build/locomo_retrieval_hardneg_v45/train}"
valid_data="${VALID_DATA:-build/locomo_retrieval_hardneg_v45/valid}"
init_memory="${INIT_MEMORY:-build/locomo_v50_full_joint_hardneg.bnmem}"
output="${OUTPUT:-build/locomo_v56_structured_memory_decoder.bnmem}"
final_output="${FINAL_OUTPUT:-build/locomo_v56_structured_memory_decoder.final.bnmem}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python python/train_memory.py \
  "$gguf" "$train_data" \
  --valid-data "$valid_data" \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --init-memory "$init_memory" \
  --output "$output" \
  --final-output "$final_output" \
  --freeze-memory \
  --answer-decoder-width 256 \
  --structured-memory-answer-decoder \
  --layers all \
  --gamma 0.975 \
  --tau 1.0 \
  --rho 0.9 \
  --beta-scale 0.9 \
  --query-rank 64 \
  --kv-rank 64 \
  --query-mode backbone_delta \
  --denom-mode signed_plus_one \
  --state-mode slots \
  --max-memory-slots 2048 \
  --slot-temperature 0.07 \
  --retrieval-windows 4 \
  --retrieval-window-size 32 \
  --retrieval-layer-mode max \
  --samples 6200 \
  --valid 390 \
  --steps 700 \
  --batch 1 \
  --lr 1e-4 \
  --wd 0.01 \
  --warmup 100 \
  --valid-every 500 \
  --valid-subset 64 \
  --self-prefix-prob 0.10 \
  --contrastive-lambda 0 \
  --evidence-lambda 0 \
  --query-gate-lambda 0 \
  --kv-gate-lambda 0 \
  --layer-gate-lambda 0 \
  --output-lora-rank 0 \
  --backbone-lora-rank 0 \
  --seed 20260929
