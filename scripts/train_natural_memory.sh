#!/usr/bin/env bash
# Train only the memory writer; the exact 0.5B backbone stays frozen.
set -euo pipefail
model_path=${1:?matching GGUF path required}
initial_writer=${2:--}
run_dir=${3:-build/natural_memory_v271}
pair_checkpoint=${4:-}
pair_binary=${5:-}
if [[ -n "$pair_checkpoint" && -z "$pair_binary" ]] || [[ -z "$pair_checkpoint" && -n "$pair_binary" ]]; then
  printf 'Supply both the frozen pair checkpoint and its binary.\n' >&2
  exit 2
fi
mkdir -p "$run_dir"
writer_init=()
if [[ "$initial_writer" != "-" ]]; then
  writer_init=(--init-checkpoint "$initial_writer")
fi
cmake -S . -B build
cmake --build build --target openai_server test_typed_writer_parity test_typed_link_parity tok_probe -j 8
cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so
if [[ ! -f "$run_dir/data/manifest.json" ]]; then
  python3 python/prepare_natural_memory_curriculum.py "$run_dir/data"
fi
python3 python/prepare_typed_memory_features.py "$model_path" \
  "$run_dir/data/train.jsonl" "$run_dir/data/valid.jsonl" \
  --lib build/libggwshim.so --tok-probe build/tok_probe \
  --train-cache "$run_dir/train_features.pt" --valid-cache "$run_dir/valid_features.pt" \
  --hidden-layer-bands 0-5,6-11,12-17,18-23 --max-episode-tokens 128 --device cuda
python3 python/train_typed_memory_writer.py \
  "$run_dir/train_features.pt" "$run_dir/data/train.jsonl" \
  "$run_dir/valid_features.pt" "$run_dir/data/valid.jsonl" "$run_dir/writer.pt" \
  --gguf "$model_path" --lib build/libggwshim.so --tok-probe build/tok_probe \
  "${writer_init[@]}" --separate-address-localizer \
  --predict-span-boundaries --use-token-embeddings --writer-focus \
  --span-boundary-weight 2 --span-attention-weight 2 --hard-negative-weight 0.25 \
  --steps 800 --batch 64 --eval-every 100 --patience 6 --learning-rate 0.0004 --device cuda
python3 python/train_typed_span_tagger.py "$run_dir/writer.pt" \
  "$run_dir/train_features.pt" "$run_dir/data/train.jsonl" \
  "$run_dir/valid_features.pt" "$run_dir/data/valid.jsonl" "$run_dir/tagger.pt" \
  --gguf "$model_path" --lib build/libggwshim.so --tok-probe build/tok_probe \
  --steps 600 --batch 64 --max-span 16 --create-fraction 0.5 \
  --learning-rate 0.0004 --eval-every 100 --patience 4 --device cuda
# Reuse the supervised writer anchors. Fusion is disabled for this controlled
# writer-only run; no coefficients are tuned on the independent final test.
PYTHONPATH=python python3 - "$run_dir" <<'PY'
import sys, torch
from pathlib import Path
from train_typed_anchor_keys import TypedAnchorKeys, FORMAT
from typed_memory_training import clone_state_dict, file_fingerprint
p = Path(sys.argv[1])
c = torch.load(p / "writer.pt", map_location="cpu", weights_only=True)
keys = TypedAnchorKeys(c["state_dict"]["token_keys"])
torch.save({"format": FORMAT, "backbone_sha256": c["backbone_sha256"],
            "writer_checkpoint_fingerprint": file_fingerprint(p / "writer.pt"),
            "state_dict": clone_state_dict(keys)}, p / "keys.pt")
PY
python3 python/export_typed_writer_model.py "$run_dir/writer.pt" "$run_dir/tagger.pt" \
  "$run_dir/keys.pt" "$run_dir/writer.bntwrite" --predicate-anchor-weight 0 --value-anchor-weight 0
python3 python/export_typed_writer_parity_sample.py "$run_dir/writer.pt" "$run_dir/tagger.pt" \
  "$run_dir/keys.pt" "$run_dir/valid_features.pt" "$run_dir/data/valid.jsonl" "$model_path" "$run_dir/writer.parity" \
  --lib build/libggwshim.so --tok-probe build/tok_probe \
  --example-index 3 --predicate-anchor-weight 0 --value-anchor-weight 0 --device cuda
build/test_typed_writer_parity "$run_dir/writer.bntwrite" "$model_path" "$run_dir/writer.parity"
if [[ -n "$pair_checkpoint" ]]; then
  python3 python/train_natural_link_head.py "$pair_checkpoint" "$pair_binary" "$model_path" \
    "$run_dir/train_features.pt" "$run_dir/data/train.jsonl" \
    "$run_dir/valid_features.pt" "$run_dir/data/valid.jsonl" "$run_dir/link" \
    --lib build/libggwshim.so --tok-probe build/tok_probe
  for count in 1 2 17; do
    build/test_typed_link_parity "$run_dir/link/link.bntlink" "$model_path" "$run_dir/link/link-$count.parity"
  done
fi
touch "$run_dir/training.done"
