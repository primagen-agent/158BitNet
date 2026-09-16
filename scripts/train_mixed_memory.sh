#!/usr/bin/env bash
set -euo pipefail
model_path=${1:?matching 0.5B GGUF required}
initial_writer=${2:?compatible original writer checkpoint required}
replay_data=${3:?original training/development directory required}
dialogue_data=${4:?conversational training/development directory required}
run_dir=${5:?new experiment directory required}
if [[ -e "$run_dir" ]]; then
  printf 'Use a new experiment directory.\n' >&2
  exit 2
fi
python3 python/prepare_mixed_memory_curriculum.py "$replay_data" "$dialogue_data" "$run_dir/data"
MEMORY_RETAIN_DOMAIN=replay bash scripts/train_natural_memory.sh "$model_path" "$initial_writer" "$run_dir"
bash scripts/train_context_memory.sh "$model_path" "$run_dir" "$run_dir/context"
touch "$run_dir/experiment.done"
