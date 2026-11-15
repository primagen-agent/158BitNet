#!/bin/sh
# Validate the Vulkan matvec kernels numerically and measure them.
#
#   scripts/validate_vk.sh /data/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf
#
# Stage 1 checks that the CPU path still produces the expected sentence, then
# the following stages run each GPU kernel and each device-side stage with
# BQ_VERIFY=1, which recomputes every GPU-produced row on the CPU and reports
# the worst relative difference.  The last stage measures the split, and the
# summary prints the hybrid speedup against the CPU-only run.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:?usage: validate_vk.sh MODEL.gguf}
BIN=$ROOT/build-bonsai/bonsai_generate
LOG=${TMPDIR:-/tmp}/bq_vk_validate
PROMPT="What is the capital of France? Answer in one sentence."
mkdir -p "$LOG"

"$ROOT/scripts/build_bonsai.sh" >/dev/null

# run <tag> <env assignments...>
run() {
    tag=$1; shift
    echo "--- $tag"
    # no -t: the tool picks the performance-core count for a hybrid run and
    # every core for a CPU-only one, which is what each of them measures best
    env "$@" "$BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOK" >"$LOG/$tag.log" 2>&1 || {
        echo "  FAIL: exit $?"; tail -5 "$LOG/$tag.log"; return 1; }
    sed -n 's/^\[tok [0-9]*: [0-9.]*s\] //p' "$LOG/$tag.log" | tr -d '\n' | sed 's/^/  text: /'
    echo
    grep -h "\[verify\]" "$LOG/$tag.log" | tail -1 | sed 's/^/  /'
    grep -h "\[prof\] matvec" "$LOG/$tag.log" | tail -1 | sed 's/^/  /'
}

echo "=== stage 1: CPU reference ==="
N_TOK=6 run cpu BQ_GPU=0
grep -q "Paris" "$LOG/cpu.log" || { echo "FAIL: CPU output lost 'Paris'"; exit 1; }

echo "=== stage 2: GPU arithmetic kernel ==="
N_TOK=2 run gpu-arith BQ_GPU_PCT=50 BQ_VERIFY=1

echo "=== stage 3: GPU lookup-table kernel ==="
N_TOK=2 run gpu-lut BQ_GPU_PCT=50 BQ_VK_LUT=1 BQ_VERIFY=1

echo "=== stage 4: GPU shared-memory staged kernel ==="
# The staged kernel carries a barrier, which only the vendored Qualcomm driver
# is trusted with: the backend gates it on driverID and on window width, and
# BQ_VK_SMEM forces it on for this check.
N_TOK=2 run gpu-smem BQ_GPU_PCT=50 BQ_VK_SMEM=1 BQ_VERIFY=1
grep -q "staged matvec enabled" "$LOG/gpu-smem.log" ||
    echo "  note: the staged kernel was not created on this driver"

echo "=== stage 4b: batched submission vs one submit per matvec ==="
N_TOK=2 run gpu-batch-off BQ_GPU_PCT=50 BQ_VK_BATCH=0 BQ_VERIFY=1

echo "=== stage 5: every per-layer stage on the device ==="
# Norms, conv, the GDN recurrence, attention and sampling all live on the GPU
# in this configuration; the text must still come out identical.
N_TOK=6 run gpu-stages BQ_GPU_PCT=50 BQ_GPU_STAGES=15 BQ_PROFILE=1
grep -q "Paris" "$LOG/gpu-stages.log" || { echo "FAIL: GPU stages changed the output"; exit 1; }

echo "=== stage 6: speed ==="
N_TOK=6 run speed-cpu   BQ_GPU=0 BQ_PROFILE=1
# The CPU kernel is half the work in a hybrid run, so time the paired-block
# variant against it on its own first.
N_TOK=6 run speed-cpu2  BQ_GPU=0 BQ_CPU_UNROLL2=1 BQ_PROFILE=1
# A faster kernel moves the best split, so sweep every kernel rather than
# assuming the GPU should take half the rows.  The share that measured best on
# an Adreno 740 is 69 (60 for narrow batches, which the tool picks itself).
for pct in 50 60 69 75; do
    N_TOK=6 run speed-arith$pct BQ_GPU_PCT=$pct BQ_PROFILE=1
done
for pct in 50 60 69 75; do
    N_TOK=6 run speed-lut$pct  BQ_GPU_PCT=$pct BQ_VK_LUT=1  BQ_PROFILE=1
done
for pct in 50 60 69 75; do
    N_TOK=6 run speed-smem$pct BQ_GPU_PCT=$pct BQ_VK_SMEM=1 BQ_PROFILE=1
done
# Batching turns a group of independent matvecs into one round trip instead of
# one each; measure both rather than assume the win survives here.
for pct in 60 69; do
    N_TOK=6 run speed-nobatch$pct BQ_GPU_PCT=$pct BQ_VK_BATCH=0 BQ_PROFILE=1
done

# seconds per eval of a run, from its profile line
seval() {
    awk '/forward [0-9.]+ s over [0-9]+ evals/ {
        for (i = 1; i <= NF; ++i) {
            if ($i == "forward") f = $(i + 1)
            if ($i == "evals;") n = $(i - 1)
        }
        printf "%.3f", f / n
        exit
    }' "$1"
}

echo "=== summary: seconds per eval (generated tokens only) ==="
cpu=$(seval "$LOG/speed-cpu.log")
printf '  %-22s %s\n' "cpu-only" "$cpu"
best=""; besttag=""
for tag in speed-cpu2 speed-arith50 speed-arith60 speed-arith69 speed-arith75 \
           speed-lut50 speed-lut69 \
           speed-smem50 speed-smem69 \
           speed-nobatch60 speed-nobatch69; do
    [ -f "$LOG/$tag.log" ] || continue
    v=$(seval "$LOG/$tag.log")
    [ -n "$v" ] || continue
    printf '  %-22s %s\n' "$tag" "$v"
    if [ -z "$best" ] || awk -v a="$v" -v b="$best" 'BEGIN{exit !(a < b)}'; then
        best=$v; besttag=$tag
    fi
done
if [ -n "$best" ]; then
    echo "  best: $besttag at $best s/eval"
    awk -v c="$cpu" -v b="$best" -v t="$besttag" \
        'BEGIN { printf "  hybrid speedup: %.2fx against cpu-only\n", c / b }'
fi

echo "logs in $LOG"
