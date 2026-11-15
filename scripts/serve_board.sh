#!/bin/sh
# Start the unified server on the Cix board with the tested configuration.
#   scripts/serve_board.sh [extra bonsai_server args...]
# One binary serves every registered model; only one is held in memory at a
# time (--max-resident 1, the default), so the resident set follows the last
# request and an idle server uses under 1 GB.
#
# Tuning that matters on this board (see the measurements in the header of
# tools/bonsai_server.c):
#   BQ_P2=1        the 2-bit i8mm repack the 27B verify depends on (server default)
#   BQ_GPU_PCT=45  qwen35 single-token matvec split; the Mali earns nothing here
#   BQ_MM_GPU=0    keep the block verify on the i8mm cores (10 costs ~5%)
#   BQ_MC_HEAD=2   ternary head copy + exact top-K rescore (minicpm.c default;
#                  MCH=1 forces the old int8 copy, MCH=0 the native head)
set -e
ADB=${ADB:-192.168.70.120:5555}
DATA=${DATA:-/data/bq}
PORT=${PORT:-8080}
THREADS=${THREADS:-8}
CTX=${CTX:-2048}
GPUPCT=${GPUPCT:-45}
MMGPU=${MMGPU:-0}

adb -s "$ADB" shell "pkill -x bonsai_server; sleep 1"
adb -s "$ADB" shell "cd $DATA && BQ_MC_HEAD=${MCH:-2} BITNET_OUTPUT_THREADS=${OTH:-0} nohup ./bonsai_server \
  --host 0.0.0.0 --port $PORT --threads $THREADS --ctx $CTX \
  --gpu-pct $GPUPCT --mm-gpu $MMGPU \
  --model bonsai27b=$DATA/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --draft $DATA/dspine27.safetensors \
  --model cpm05=$DATA/models/bitcpm4-0.5b-tq2_0.gguf \
  --model cpm1b=$DATA/models/bitcpm4-1b-tq2_0.gguf \
  --model cpm3b=$DATA/models/bitcpm4-3b-tq2_0.gguf \
  --model cpm8b=$DATA/models/bitcpm4-8b-tq2_0.gguf \
  $* > $DATA/server.log 2>&1 </dev/null & sleep 1; echo started" >/dev/null 2>&1 &
# adb keeps the shell's pty open while the server runs, so the launch above is
# backgrounded locally and readiness is polled with curl instead of slept off.
curl -s --max-time 5 --retry 20 --retry-connrefused --retry-delay 1 \
     "http://${ADB%%:*}:$PORT/health"; echo
