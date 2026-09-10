#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-/home/ubuntu/models/gguf/bitcpm4-0.5b-tq2_0.gguf}"
output="${OUTPUT:-build/memory_v101_05b_address_controller.bnctrl}"
address_cache="${ADDRESS_CACHE:-build/memory_v101_address_features.npz}"
action_cache="${ACTION_CACHE:-build/memory_v91f_action_features.npz}"
action_root="${ACTION_ROOT:-build/memory_v91f_action_router}"

python3 python/train_addressed_memory_controller.py \
  "$gguf" "$output" \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --cache "$address_cache" \
  --action-train "$action_root/train.jsonl" \
  --action-valid "$action_root/valid.jsonl" \
  --action-cache "$action_cache" \
  --train-examples 280 \
  --valid-examples 60 \
  --rank 128 \
  --action-rank 64 \
  --steps 3000 \
  --batch 64 \
  --action-batch 256 \
  --lr 1e-3 \
  --temperature 0.07 \
  --action-weight 2.0 \
  --max-tokens 96 \
  --pooling mean_last \
  --device cuda \
  --seed 20260910
