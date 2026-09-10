#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-/home/ubuntu/models/gguf/bitcpm4-0.5b-tq2_0.gguf}"
data_root="${DATA_ROOT:-build/memory_v103_payload_pointer}"
output="${OUTPUT:-build/memory_v103_05b_compound_pointer.bnptr5}"
cache_root="${CACHE_ROOT:-build/memory_v103_payload_features}"

if [[ ! -f "$data_root/train/len8/reconstruction.jsonl" ||
      ! -f "$data_root/valid_ood/len8/reconstruction.jsonl" ]]; then
  python3 python/prepare_memory_v92_payload_pointer.py \
    "$gguf" "$data_root" \
    --tok-probe build/tok_probe \
    --lengths 1,2,3,4,5,6,7,8 \
    --train-per-length 1800 \
    --valid-per-length 300 \
    --compound-fraction 0.5 \
    --reserved-ood-payload cobalt-cedar \
    --seed 20260910
fi

python3 python/train_memory_v89_copy_pointer.py \
  "$gguf" \
  "$data_root/train" \
  "$data_root/valid_id" \
  "$data_root/valid_ood" \
  "$output" \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --cache-dir "$cache_root" \
  --device cuda \
  --epochs 100 \
  --batch 32 \
  --lr 2e-4 \
  --max-span 16 \
  --max-copy-bytes 128 \
  --max-token-bytes 128 \
  --pointer-rank 64 \
  --pointer-task write \
  --target-id-exact 0.99 \
  --target-ood-exact 0.98 \
  --seed 20260910
