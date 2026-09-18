#!/usr/bin/env bash
# Re-run the training stages that calculate resident-mode writes and recalls.
# Outputs stay under a new build/ run directory and do not replace bundled models.
set -euo pipefail

gguf=${1:?usage: retrain_resident_and_writer.sh GGUF NEW_RUN_DIRECTORY}
run_root=${2:?usage: retrain_resident_and_writer.sh GGUF NEW_RUN_DIRECTORY}
if [[ -e "$run_root" ]]; then
  printf 'Use a new run directory: %s\n' "$run_root" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
package="$repo_root/training/memory/resident-0.5b"
cd "$repo_root"
python3 "$package/verify.py" "$gguf"

cmake -S . -B build
cmake --build build --target tok_probe test_typed_writer_parity -j 8
cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so

mkdir -p "$run_root/writer/data"
cp "$package"/corpora/writer/{train.jsonl,valid.jsonl,manifest.json} "$run_root/writer/data/"

# The baseline run computes the exact-C-tokenizer frozen-backbone feature
# cache used by identity training. Compare its new weights on development data;
# the retained selected baseline is used as the identity stage's initializer.
python3 python/train_resident_memory_set.py "$gguf" \
  "$package/corpora/resident" "$run_root/baseline" \
  --lib build/libggwshim.so --tok-probe build/tok_probe --device cuda \
  --architecture factorized --initial-order-scale 0.025 \
  --seed 2810917 --steps 1000
python3 python/train_resident_identity.py "$gguf" \
  "$package/corpora/resident" "$run_root/baseline/features.pt" \
  "$package/checkpoints/baseline.pt" "$run_root/identity" \
  --lib build/libggwshim.so --tok-probe build/tok_probe --device cuda \
  --seed 2810917 --steps 1000
python3 python/export_resident_identity.py \
  "$run_root/identity/selected_research.pt" "$run_root/identity.bnresid"

bash scripts/train_natural_memory.sh "$gguf" \
  "$package/checkpoints/writer_init.pt" "$run_root/writer"
bash scripts/train_context_memory.sh "$gguf" \
  "$run_root/writer" "$run_root/context"

printf 'Retraining finished. Evaluate new artifacts before replacing the checked-in bundle.\n'
