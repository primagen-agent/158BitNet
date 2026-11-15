#!/bin/sh
# Copy the runtime sources the unified server needs to the board tree.
#   scripts/sync_board_src.sh [adb-serial]
# Only files that changed since the last run cost network time (adb push of a
# directory skips nothing, so the list is explicit).
set -e
SERIAL=${1:-192.168.70.120:5555}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${DEST:-/data/bq}
FILES="tools/bonsai_server.c tools/ctx_probe.c scripts/build_server_board.sh
       scripts/serve_board.sh
       src/bonsai.c src/minicpm.c src/minicpm.h src/dspine.c src/quant_ptq1_0.c
       src/quant_ptq1_0.h src/quant_tq2_0.c src/quant_tq2_0.h src/gguf.c src/gguf.h
       src/output_head.h
       src/bitnet.c src/bitnet_dispatch.c src/ops.c src/sampler.c src/tensor.c
       include/bonsai.h include/dspine.h"
for f in $FILES; do
    adb -s "$SERIAL" push "$ROOT/$f" "$DEST/$f" >/dev/null
    echo "pushed $f"
done
