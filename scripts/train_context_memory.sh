#!/usr/bin/env bash
set -euo pipefail
model_path=${1:?matching 0.5B GGUF required}
base_run=${2:?frozen natural-writer run directory required}
run_dir=${3:-build/context_operation}
mkdir -p "$run_dir"
python3 python/train_context_memory_operation.py "$base_run/writer.pt" \
  "$base_run/train_features.pt" "$base_run/data/train.jsonl" \
  "$base_run/valid_features.pt" "$base_run/data/valid.jsonl" "$run_dir/operation.pt" \
  --writer-binary "$base_run/writer.bntwrite" --device cuda
cmake -S . -B build
cmake --build build --target test_typed_writer_parity tok_probe -j 8
cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so
python3 python/export_typed_writer_model.py "$base_run/writer.pt" "$base_run/tagger.pt" \
  "$base_run/keys.pt" "$run_dir/writer.bntwrite" --predicate-anchor-weight 0 --value-anchor-weight 0 \
  --operation-model "$run_dir/operation.pt"
for index in 3 40; do
  python3 python/export_typed_writer_parity_sample.py "$base_run/writer.pt" "$base_run/tagger.pt" \
    "$base_run/keys.pt" "$base_run/valid_features.pt" "$base_run/data/valid.jsonl" \
    "$model_path" "$run_dir/writer-$index.parity" --lib build/libggwshim.so --tok-probe build/tok_probe \
    --example-index "$index" --predicate-anchor-weight 0 --value-anchor-weight 0 \
    --operation-model "$run_dir/operation.pt" --device cuda
  build/test_typed_writer_parity "$run_dir/writer.bntwrite" "$model_path" "$run_dir/writer-$index.parity"
done
sha256sum "$run_dir/writer.bntwrite"
touch "$run_dir/training.done"
