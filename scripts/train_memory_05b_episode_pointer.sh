#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-models/bitcpm4-0.5b-tq2_0.gguf}"
data_dir="${DATA_DIR:-build/memory_v87_paper_curriculum}"
output="${OUTPUT:-build/memory_05b_episode_pointer.pt}"
binary_output="${BINARY_OUTPUT:-build/memory_05b_episode_pointer.bneptr}"

if [[ ! -d "$data_dir/train" || ! -d "$data_dir/valid" ]]; then
  echo "missing five-task memory curriculum: $data_dir/{train,valid}" >&2
  echo "set DATA_DIR to the generated paper-five-task curriculum" >&2
  exit 1
fi

python3 python/train_episode_pointer.py \
  "$gguf" \
  "$data_dir/train" \
  "$data_dir/valid" \
  "$output" \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --train-cache build/memory_05b_episode_pointer_train_features.pt \
  --valid-cache build/memory_05b_episode_pointer_valid_features.pt \
  --train-limit 2000 \
  --valid-limit 1000 \
  --max-source-tokens 256 \
  --max-query-tokens 128 \
  --max-span 160 \
  --rank 128 \
  --steps 3000 \
  --batch 24 \
  --lr 1.5e-4 \
  --weight-decay 0.01 \
  --eval-every 50 \
  --patience 8 \
  --device cuda \
  --seed 20260913

PYTHONPATH=python python3 -c \
  "from train_episode_pointer import export_episode_pointer_binary; export_episode_pointer_binary('$output', '$binary_output')"

printf 'trained checkpoint: %s\n' "$output"
printf 'C runtime artifact: %s\n' "$binary_output"
