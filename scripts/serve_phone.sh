#!/bin/sh
# Serve the models a phone already holds through the unified server.
#
#   scripts/build_server_host.sh                      # cross-build for bionic
#   adb -s PHONE push build-bonsai/bonsai_server /data/local/tmp/
#   scripts/serve_phone.sh -s PHONE:5555 [-d DIR] [-t 4] [-g 69] [-m 30] [-p 8080]
#
# Every model is optional: a file that is not in DIR is simply not registered.
# The knob defaults are the SM8550 (Adreno) optimum measured in
# tools/bonsai_server.c's header, where -t 4 matters most (8 threads put a
# worker on an A510 and every layer barrier waits for it).
set -e
SERIAL=
DIR=/data/local/tmp
THREADS=4
GPUPCT=69
MMGPU=30
PORT=8080
while getopts "s:d:t:g:m:p:" o; do
    case $o in
        s) SERIAL=$OPTARG ;;
        d) DIR=$OPTARG ;;
        t) THREADS=$OPTARG ;;
        g) GPUPCT=$OPTARG ;;
        m) MMGPU=$OPTARG ;;
        p) PORT=$OPTARG ;;
        *) echo "usage: $0 -s SERIAL [-d DIR] [-t n] [-g pct] [-m pct] [-p port]" >&2; exit 2 ;;
    esac
done
[ -n "$SERIAL" ] || { echo "-s SERIAL required" >&2; exit 2; }
MODELS=
add() { # id filename
    adb -s "$SERIAL" shell "[ -f $DIR/$2 ]" >/dev/null 2>&1 &&
        MODELS="$MODELS --model $1=$DIR/$2"
}
add cpm05 bitcpm05-tq2_0.gguf
add cpm1b  bitcpm4-1b-tq2_0.gguf
add cpm3b  bitcpm4-3b-tq2_0.gguf
add cpm8b  bitcpm4-8b-tq2_0.gguf
add bonsai27b Ternary-Bonsai-2-27B-PTQ1_0.gguf
DRAFT=
adb -s "$SERIAL" shell "[ -f $DIR/dspine27.safetensors ]" >/dev/null 2>&1 &&
    DRAFT="--draft $DIR/dspine27.safetensors"
[ -n "$MODELS" ] || { echo "no models in $DIR on $SERIAL" >&2; exit 1; }
adb -s "$SERIAL" shell "pkill -x bonsai_server; sleep 1" >/dev/null 2>&1
adb -s "$SERIAL" shell "cd $DIR && BQ_GPU_PCT=$GPUPCT BQ_MM_GPU=$MMGPU nohup ./bonsai_server \
    --host 0.0.0.0 --port $PORT --threads $THREADS --ctx 2048 $MODELS $DRAFT \
    > $DIR/server.log 2>&1 </dev/null & sleep 1; echo started" >/dev/null 2>&1 &
IP=${SERIAL%%:*}
curl -s --max-time 5 --retry 30 --retry-connrefused --retry-delay 1 \
     "http://$IP:$PORT/health"; echo
