#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

gguf="${GGUF:-/home/ubuntu/models/gguf/bitcpm4-0.5b-tq2_0.gguf}"

GGUF="$gguf" \
DATA_ROOT=build/memory_v91f_action_router \
OUTPUT=build/memory_v91f_05b_controller.bnctrl \
ACTION_CACHE=build/memory_v91f_action_features.npz \
scripts/train_memory_v91_05b_controller.sh

GGUF="$gguf" scripts/train_memory_v103_05b_compound_pointer.sh
GGUF="$gguf" scripts/train_memory_v101_05b_address_controller.sh

PYTHONPATH=python python3 python/merge_memory_controller_heads.py \
  build/memory_v101_05b_address_controller.pt \
  build/memory_v91f_05b_controller.pt \
  "$gguf" \
  build/memory_v102_05b_combined_controller.bnctrl
