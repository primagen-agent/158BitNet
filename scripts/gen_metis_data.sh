#!/bin/sh
# scripts/gen_metis_data.sh — one-off offline data generation (Metis synthetic set).
# Usage: scripts/gen_metis_data.sh [METIS_REPO_DIR] [OUT_DIR]
set -e
METIS_DIR="${1:-$HOME/workdir/AI/Metis}"
OUT_DIR="${2:-data/synth_memory}"
if [ ! -f "$METIS_DIR/scripts/gen_synth_memory_data.py" ]; then
    echo "error: gen_synth_memory_data.py not found under $METIS_DIR" >&2; exit 1
fi
mkdir -p "$OUT_DIR"
python3 "$METIS_DIR/scripts/gen_synth_memory_data.py" "$OUT_DIR"
echo "done: $(ls "$OUT_DIR" | wc -l | tr -d ' ') files in $OUT_DIR"
