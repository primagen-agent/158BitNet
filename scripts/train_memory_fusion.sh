#!/usr/bin/env bash
# First architecture gate only: oracle-boundary memory -> frozen LM generation.
set -euo pipefail
gguf=${1:?usage: train_memory_fusion.sh GGUF NEW_OUTPUT_DIRECTORY}
run_root=${2:?usage: train_memory_fusion.sh GGUF NEW_OUTPUT_DIRECTORY}
if [[ -e "$run_root" ]]; then
  printf 'Use a new output directory: %s\n' "$run_root" >&2
  exit 2
fi
cmake -S . -B build
cmake --build build --target tok_probe -j 8
cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so
data_options=()
model_options=()
if [[ "${FUSION_DIVERSE_ENTITIES:-0}" == "1" ]]; then
  data_options+=(--diverse-entities)
fi
if [[ "${FUSION_EVIDENCE_GATE:-0}" == "1" ]]; then
  model_options+=(--evidence-gate --gate-loss-weight 1 --decision-loss-weight 4 --select-by-generation)
fi
python3 python/prepare_memory_fusion_curriculum.py "$run_root/data" \
  --train-worlds "${FUSION_TRAIN_WORLDS:-32}" "${data_options[@]}"
python3 python/train_memory_fusion.py "$gguf" "$run_root/data" "$run_root/model" \
  --lib build/libggwshim.so --tok-probe build/tok_probe --device cuda \
  --steps "${FUSION_STEPS:-300}" --accumulation "${FUSION_ACCUMULATION:-4}" \
  --eval-every "${FUSION_EVAL_EVERY:-50}" "${model_options[@]}"
