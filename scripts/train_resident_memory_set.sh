#!/usr/bin/env bash
set -euo pipefail
model_path=${1:?matching 0.5B GGUF required}
run_dir=${2:?new research run directory required}
extra=()
if [[ ${RESIDENT_COUNTERFACTUAL_MEMORY:-0} == 1 ]]; then extra+=(--counterfactual-memory); fi
if [[ ${RESIDENT_DISTRACTOR_PAIRS:-0} == 1 ]]; then extra+=(--distractor-pairs); fi
curriculum=()
diagnostic=()
if [[ ${RESIDENT_MEMORY_INTERVENTIONS:-0} == 1 ]]; then diagnostic+=(--memory-interventions); fi
if [[ ${RESIDENT_DIVERSE_TRAIN:-0} == 1 ]]; then curriculum+=(--diverse-train); fi
if [[ -n ${RESIDENT_FEATURE_CACHE:-} ]]; then extra+=(--feature-cache "$RESIDENT_FEATURE_CACHE"); fi
if [[ -e "$run_dir" ]]; then printf 'Use a new run directory.\n' >&2; exit 2; fi
python3 python/prepare_memory_set_curriculum.py "$run_dir/data" "${curriculum[@]}"
cmake -S . -B build
cmake --build build --target tok_probe -j 8
cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so
python3 python/train_resident_memory_set.py "$model_path" "$run_dir/data" "$run_dir/model" \
  --lib build/libggwshim.so --tok-probe build/tok_probe --device cuda \
  --seed "${RESIDENT_TRAINING_SEED:-2810916}" \
  --initial-order-scale "${RESIDENT_INITIAL_ORDER_SCALE:-1}" \
  --architecture "${RESIDENT_ARCHITECTURE:-pooled}" \
  --invariance-weight "${RESIDENT_INVARIANCE_WEIGHT:-1}" \
  --ranking-loss-weight "${RESIDENT_RANKING_LOSS_WEIGHT:-0}" "${extra[@]}"
python3 python/diagnose_resident_memory_set.py "$run_dir" "$run_dir/binding_diagnostic.json" \
  --device cuda --steps 0 "${diagnostic[@]}"
# test.jsonl is sealed; no C export or serving-model replacement is performed.
