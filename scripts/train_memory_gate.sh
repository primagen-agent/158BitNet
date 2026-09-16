#!/usr/bin/env bash
set -euo pipefail
model_path=${1:?matching 0.5B GGUF required}
data_dir=${2:?mixed training/development directory required}
run_dir=${3:?new gate run directory required}
cmake -S . -B build
cmake --build build --target tok_probe test_memory_gate_parity openai_server -j 8
cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so
python3 python/train_memory_gate.py "$model_path" "$data_dir" "$run_dir" \
  --lib build/libggwshim.so --tok-probe build/tok_probe --device cuda \
  --pooling "${MEMORY_GATE_POOLING:-mean_last}" \
  --curriculum "${MEMORY_GATE_CURRICULUM:-basic}"
python3 - "$run_dir/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if not summary["eligible_for_c_validation"]:
    raise SystemExit("gate rejected: require zero development false writes and at least 80% recall in every positive group")
PY
BITNET_NUM_THREADS=4 build/test_memory_gate_parity "$run_dir/gate.bnctrl" "$model_path" \
  "$run_dir/valid.parity" > "$run_dir/c_validation.jsonl"
touch "$run_dir/training.done"
