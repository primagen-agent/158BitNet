#!/usr/bin/env bash
# Isolate conversational writer generalization before changing pair/query models.
set -euo pipefail
model_path=${1:?matching 0.5B GGUF required}
initial_writer=${2:?compatible writer checkpoint required}
run_dir=${3:?new experiment directory required}
if [[ -e "$run_dir" ]]; then
  printf 'Use a new directory; do not overwrite or silently resume an experiment.\n' >&2
  exit 2
fi
python3 python/prepare_dialogue_memory_curriculum.py "$run_dir/data"
python3 python/prepare_natural_memory_eval.py "$run_dir/data/valid.jsonl" \
  "$run_dir/development_chat.json" --role development
bash scripts/train_natural_memory.sh "$model_path" "$initial_writer" "$run_dir"
bash scripts/train_context_memory.sh "$model_path" "$run_dir" "$run_dir/context"
# test.jsonl stays sealed and is not encoded or used for checkpoint selection.
touch "$run_dir/experiment.done"
