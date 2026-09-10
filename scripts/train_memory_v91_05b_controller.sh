#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-/home/ubuntu/models/gguf/bitcpm4-0.5b-tq2_0.gguf}"
data_root="${DATA_ROOT:-build/memory_v91e_action_router}"
output="${OUTPUT:-build/memory_v91e_05b_controller.bnctrl}"
address_cache="${ADDRESS_CACHE:-build/memory_v91_address_features.npz}"
action_cache="${ACTION_CACHE:-build/memory_v91e_action_features.npz}"
steps="${STEPS:-2000}"
action_ignore_weight="${ACTION_IGNORE_WEIGHT:-1.0}"
action_update_weight="${ACTION_UPDATE_WEIGHT:-1.0}"

if [[ ! -f "$data_root/train.jsonl" ||
      ! -f "$data_root/valid.jsonl" ]]; then
  python3 python/prepare_memory_v91_action_router.py \
    "$data_root" \
    --train-per-action 2000 \
    --valid-per-action 400
fi

python3 python/train_addressed_memory_controller.py \
  "$gguf" "$output" \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --cache "$address_cache" \
  --action-train "$data_root/train.jsonl" \
  --action-valid "$data_root/valid.jsonl" \
  --action-cache "$action_cache" \
  --train-examples 110 \
  --valid-examples 28 \
  --rank 128 \
  --action-rank 64 \
  --steps "$steps" \
  --batch 64 \
  --action-batch 256 \
  --lr 1e-3 \
  --temperature 0.07 \
  --action-weight 2.0 \
  --action-ignore-weight "$action_ignore_weight" \
  --action-update-weight "$action_update_weight" \
  --max-tokens 96 \
  --pooling mean_last \
  --device cuda \
  --seed 20260909
