#!/usr/bin/env bash
set -euo pipefail
model_path=${1:?matching 0.5B GGUF required}
data_dir=${2:?mixed training/development directory required}
run_dir=${3:?new research run directory required}
extra=()
if [[ ${MEMORY_GATE_BALANCE_QUOTES:-0} == 1 ]]; then extra+=(--balance-quote-format); fi
if [[ -n ${MEMORY_GATE_FEATURE_CACHE:-} ]]; then extra+=(--feature-cache "$MEMORY_GATE_FEATURE_CACHE"); fi
cmake -S . -B build
cmake --build build --target tok_probe -j 8
cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so
python3 python/train_memory_span_gate.py "$model_path" "$data_dir" "$run_dir" \
  --lib build/libggwshim.so --tok-probe build/tok_probe --device cuda "${extra[@]}"
# Research .pt files are intentionally not exported or loaded by the C server.
