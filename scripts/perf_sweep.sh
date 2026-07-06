#!/usr/bin/env bash
# Usage: scripts/perf_sweep.sh <device> <model> <threads> [tier] <test-binary>
# Device ∈ {android,mac,x86-linux}
set -euo pipefail

DEVICE="${1:-}"
MODEL="${2:-}"
THREADS="${3:-}"
TIER="${4:-}"
TEST_BIN="${5:-}"

if [[ -z "$DEVICE" || -z "$MODEL" || -z "$THREADS" || -z "$TEST_BIN" ]]; then
  echo "usage: $0 <device> <model> <threads> [tier] <test-binary>" >&2
  exit 64
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/build-strict/perf-logs/$DEVICE"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"

# Write summary header if file is new
if [[ ! -f "$SUMMARY" ]]; then
  printf "model\tthreads\ttier\ttest_bin\trc\telapsed_s\ttok_per_s\n" > "$SUMMARY"
fi

KEY="${MODEL}|${THREADS}|${TIER:-default}|${TEST_BIN}"
if grep -qF "$KEY" "$SUMMARY"; then
  echo "[skip] $KEY"
  exit 0
fi

case "$DEVICE" in
  android)
    ADB="adb -s 192.168.210.10:5555"
    ENV="BITNET_NUM_THREADS=$THREADS${TIER:+ BITNET_CPU_TIER=$TIER}"
    CMD="$ADB shell 'cd /data/local/tmp/bitnet-test && $ENV ./$TEST_BIN /data/models/$MODEL'"
    ;;
  mac)
    ENV="BITNET_NUM_THREADS=$THREADS${TIER:+ BITNET_CPU_TIER=$TIER}"
    CMD="$ENV $REPO_ROOT/build-arm64/$TEST_BIN $REPO_ROOT/models/$MODEL"
    ;;
  x86-linux)
    SSH="sshpass -p cck@110119 ssh -o StrictHostKeyChecking=no cuick@192.168.210.24"
    ENV="BITNET_NUM_THREADS=$THREADS${TIER:+ BITNET_CPU_TIER=$TIER}"
    CMD="$SSH 'cd ~/bitnet-test/repo && $ENV ./build/$TEST_BIN ~/bitnet-test/models/$MODEL'"
    ;;
  *)
    echo "unknown device: $DEVICE" >&2
    exit 64
    ;;
esac

LOG_FILE="$LOG_DIR/${MODEL}_t${THREADS}${TIER:+_$TIER}_${TEST_BIN}.log"
START=$(date +%s)
RC=0
# shellcheck disable=SC2086
eval "$CMD" > "$LOG_FILE" 2>&1 || RC=$?
END=$(date +%s)
ELAPSED=$((END - START))

TOK_PER_S=""
if grep -qE 'decode_tok_s' "$LOG_FILE"; then
  TOK_PER_S=$(grep -E 'decode_tok_s' "$LOG_FILE" | tail -1 | awk -F= '{print $NF}')
fi

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$MODEL" "$THREADS" "${TIER:-default}" "$TEST_BIN" "$RC" "$ELAPSED" "${TOK_PER_S:-}" \
  >> "$SUMMARY"

echo "[done] $KEY rc=$RC elapsed=${ELAPSED}s tok/s=${TOK_PER_S:-n/a}"
exit $RC
