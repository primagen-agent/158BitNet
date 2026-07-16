# Three-Device Adaptation + Perf Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 158BitNet runtime on three devices (Android 865, Apple Silicon Mac arm64, x86 Linux dell-precision-5810), transfer all four BitCPM4 GGUF models to each, run `ctest` + `test_profile_decode` (4 models × 6 thread configs) on each, plus a `BITNET_CPU_TIER` sweep on x86 Linux, and commit a single markdown report at `docs/superpowers/reports/2026-07-03-three-device-perf.md`.

**Architecture:** Operational, not feature-code. A driver script (`scripts/perf_sweep.sh`) orchestrates every run, captures logs to `build-strict/perf-logs/<device>/`, and writes one summary row per run to `summary.tsv` (idempotent — re-runs skip already-recorded runs). Three hosts run in parallel where wall-time matters; the script abstracts the host-specific command shape (adb shell, local ./, ssh).

**Tech Stack:** bash, CMake (host + NDK cross-compile), ADB, sshpass/scp, Python (optional — only for summary aggregation), `test_profile_decode` + `ctest` from the existing build tree.

## Global Constraints

- Source of truth for models: `/data/models/` on Android (already has all four GGUF files).
- Mac builds **arm64 native** (NOT x86_64 under Rosetta). Use `build-arm64/` as a fresh build dir; do NOT touch the existing `build/` (which is x86_64 Rosetta).
- x86 Linux = dell-precision-5810 at `192.168.210.24`, user `cuick`, password `cck@110119` (via sshpass). Workspace: `~/bitnet-test/` on dell-precision-5810.
- All artifacts (build dirs, models, logs) land under `build*/`, `models/`, or `build-strict/perf-logs/` — already gitignored. No tracked file should grow beyond the report markdown.
- Decode benchmark: prompt `"The capital of France is"`, `max_tokens=64`, `BITNET_NUM_THREADS` from {1,2,3,4,6}.
- Tier sweep (x86 Linux only): `BITNET_CPU_TIER` from {scalar, avx2, avx_vnni, avx512_vnni}.
- Models: `bitcpm4-{0.5b,1b,3b,8b}-tq2_0.gguf`. Total ~4 GB on each host.
- LoRA artifact for `test_lora_loader`: pre-generated `build/test_lora_loader.bnlora` copied to each host.
- Don't fix bugs found during the sweep — record them in "Issues found" of the report.
- All commits on branch `feature/x86-optimization`. Use `git -c commit.gpgsign=false` to bypass signing in this sandbox.

## File Structure

**Created:**
- `scripts/perf_sweep.sh` — driver. Args: `<device> <model> <threads> [tier] <test-binary>`. Writes to `build-strict/perf-logs/<device>/`.
- `scripts/perf_summarize.py` — aggregator. Reads `summary.tsv` from each device, emits report markdown skeleton.
- `docs/superpowers/reports/2026-07-03-three-device-perf.md` — final report (filled by `perf_summarize.py` + manual notes).

**Created at runtime, not tracked:**
- `models/*.gguf` — pulled from Android.
- `build-arm64/` — Mac arm64 build.
- `~/bitnet-test/` on dell-precision-5810 — x86 Linux build + models + lora.
- `build-strict/perf-logs/{android,mac,x86-linux}/` — logs and summary.tsv.

**Not modified:** no source-code changes.

---

### Task 1: Local log directory + driver script scaffold

**Files:**
- Create: `scripts/perf_sweep.sh`
- Create: `scripts/perf_summarize.py`

**Produces:** `perf_sweep.sh` accepts device + test args, writes logs to `build-strict/perf-logs/<device>/`, appends a row to `summary.tsv`, skips runs whose row is already present. `perf_summarize.py` reads all summary.tsv files and emits a markdown skeleton.

- [ ] **Step 1: Create `scripts/perf_sweep.sh`**

```bash
#!/usr/bin/env bash
# Usage: scripts/perf_sweep.sh <device> <model> <threads> [tier] <test-binary>
# Device ∈ {android,mac,x86-linux}
set -u

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

KEY="${MODEL}|${THREADS}|${TIER}|${TEST_BIN}"
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
    CMD="$SSH 'cd ~/bitnet-test && $ENV ./build/$TEST_BIN ~/bitnet-test/models/$MODEL'"
    ;;
  *)
    echo "unknown device: $DEVICE" >&2
    exit 64
    ;;
esac

LOG_FILE="$LOG_DIR/${MODEL}_t${THREADS}${TIER:+_$TIER}_${TEST_BIN}.log"
START=$(date +%s)
# shellcheck disable=SC2086
eval "$CMD" > "$LOG_FILE" 2>&1
RC=$?
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
```

- [ ] **Step 2: Make it executable + smoke test**

```bash
chmod +x scripts/perf_sweep.sh
scripts/perf_sweep.sh nonexistent-device foo 1 test_xx
```
Expected: stderr `usage: ...`, exit code 64.

- [ ] **Step 3: Create `scripts/perf_summarize.py`**

```python
#!/usr/bin/env python3
"""Aggregate build-strict/perf-logs/<device>/summary.tsv into a markdown report skeleton."""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / "build-strict" / "perf-logs"
OUT = REPO / "docs" / "superpowers" / "reports" / "2026-07-03-three-device-perf.md"

DEVICES = ["android", "mac", "x86-linux"]

def load(device):
    f = LOG_ROOT / device / "summary.tsv"
    if not f.exists():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Three-Device Adaptation + Perf Sweep Report",
             "",
             "**Date**: 2026-07-03  **Branch**: feature/x86-optimization",
             "",
             "## Summary",
             ""]

    # Per-device summary
    summary_rows = []
    for d in DEVICES:
        rows = load(d)
        decode = [r for r in rows if r["test_bin"] == "test_profile_decode"]
        by_model = defaultdict(list)
        for r in decode:
            if r["tok_per_s"]:
                by_model[r["model"]].append(float(r["tok_per_s"]))
        best = {m: (max(v) if v else "—") for m, v in by_model.items()}
        ctest_fail = sum(1 for r in rows if r["test_bin"] == "ctest" and r["rc"] != "0")
        ctest_total = sum(1 for r in rows if r["test_bin"] == "ctest")
        summary_rows.append([
            d, f"{ctest_total - ctest_fail}/{ctest_total}",
            best.get("bitcpm4-0.5b-tq2_0.gguf", "—"),
            best.get("bitcpm4-1b-tq2_0.gguf", "—"),
            best.get("bitcpm4-3b-tq2_0.gguf", "—"),
            best.get("bitcpm4-8b-tq2_0.gguf", "—"),
        ])
    lines.append(md_table(
        ["Device", "ctest", "best 0.5B tok/s", "best 1B tok/s",
         "best 3B tok/s", "best 8B tok/s"], summary_rows))
    lines.append("")

    # Per-device detail
    for d in DEVICES:
        rows = load(d)
        lines += [f"## {d}", ""]
        ctest = [r for r in rows if r["test_bin"] == "ctest"]
        decode = [r for r in rows if r["test_bin"] == "test_profile_decode"]
        lines.append("### ctest")
        if ctest:
            lines.append(md_table(
                ["model", "threads", "tier", "rc", "elapsed_s"],
                [[r["model"], r["threads"], r["tier"], r["rc"], r["elapsed_s"]]
                 for r in ctest]))
        else:
            lines.append("_(no ctest runs recorded)_")
        lines += ["", "### decode (test_profile_decode)"]
        if decode:
            lines.append(md_table(
                ["model", "threads", "tier", "tok/s", "elapsed_s"],
                [[r["model"], r["threads"], r["tier"],
                  r["tok_per_s"] or "—", r["elapsed_s"]] for r in decode]))
        else:
            lines.append("_(no decode runs recorded)_")
        lines.append("")

    # Tier sweep
    tier_rows = [r for r in load("x86-linux")
                 if r["test_bin"] == "test_profile_decode" and r["tier"] != "default"]
    lines += ["## x86 Linux tier sweep", ""]
    if tier_rows:
        lines.append(md_table(
            ["model", "tier", "threads", "tok/s", "elapsed_s"],
            [[r["model"], r["tier"], r["threads"],
              r["tok_per_s"] or "—", r["elapsed_s"]] for r in tier_rows]))
    else:
        lines.append("_(no tier sweep runs recorded)_")
    lines += ["", "## Issues found", "", "_TBD — fill in after run._",
              "", "## Recommendations", "", "_TBD — fill in after run._"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the script compiles**

```bash
chmod +x scripts/perf_summarize.py
python3 -c "import ast; ast.parse(open('scripts/perf_summarize.py').read())"
```
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/perf_sweep.sh scripts/perf_summarize.py
git -c commit.gpgsign=false commit -m "feat(tools): perf_sweep driver + summary aggregator"
```

---

### Task 2: Transfer all four GGUF models from Android to local Mac

**Files:** none created locally (models land in `models/`, gitignored).

- [ ] **Step 1: Verify ADB device is connected**

```bash
adb devices
```
Expected: `192.168.210.10:5555    device` in output.

- [ ] **Step 2: Pull all four models**

```bash
mkdir -p /Users/chersu/workdir/AI/158BitNet/models
for f in bitcpm4-0.5b-tq2_0.gguf bitcpm4-1b-tq2_0.gguf bitcpm4-3b-tq2_0.gguf bitcpm4-8b-tq2_0.gguf; do
  adb -s 192.168.210.10:5555 pull "/data/models/$f" "/Users/chersu/workdir/AI/158BitNet/models/$f"
done
ls -lh /Users/chersu/workdir/AI/158BitNet/models/
```
Expected: four .gguf files, total ~4 GB.

- [ ] **Step 3: Verify checksums against Android's files**

```bash
for f in bitcpm4-0.5b-tq2_0.gguf bitcpm4-1b-tq2_0.gguf bitcpm4-3b-tq2_0.gguf bitcpm4-8b-tq2_0.gguf; do
  L=$(shasum -a 256 "/Users/chersu/workdir/AI/158BitNet/models/$f" | awk '{print $1}')
  R=$(adb -s 192.168.210.10:5555 shell "sha256sum /data/models/$f" | awk '{print $1}')
  echo "$f: local=$L remote=$R $([ "$L" = "$R" ] && echo OK || echo MISMATCH)"
done
```
Expected: all four lines end with `OK`.

- [ ] **Step 4: Confirm `models/` is gitignored**

```bash
git check-ignore -v /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf
```
Expected: prints `.gitignore:1:build/    models/` (or similar — any path that includes `models/`).

No commit (models are gitignored).

---

### Task 3: Transfer models + LoRA to x86 Linux dell-precision-5810

**Files:** none.

- [ ] **Step 1: Create remote workspace**

```bash
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 \
  'mkdir -p ~/bitnet-test/models ~/bitnet-test/lora && ls -la ~/bitnet-test/'
```
Expected: `models/` and `lora/` directories listed.

- [ ] **Step 2: scp the four GGUF files**

```bash
scp -o StrictHostKeyChecking=no \
  /Users/chersu/workdir/AI/158BitNet/models/*.gguf \
  cuick@192.168.210.24:~/bitnet-test/models/
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 \
  'ls -lh ~/bitnet-test/models/'
```
Expected: four files listed, total ~4 GB.

- [ ] **Step 3: scp the LoRA artifact**

```bash
scp -o StrictHostKeyChecking=no \
  /Users/chersu/workdir/AI/158BitNet/build/test_lora_loader.bnlora \
  cuick@192.168.210.24:~/bitnet-test/lora/xiaoli.bnlora
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 \
  'ls -la ~/bitnet-test/lora/'
```
Expected: `xiaoli.bnlora` present.

- [ ] **Step 4: Verify checksums on dell-precision-5810**

```bash
for f in bitcpm4-0.5b-tq2_0.gguf bitcpm4-1b-tq2_0.gguf bitcpm4-3b-tq2_0.gguf bitcpm4-8b-tq2_0.gguf; do
  L=$(shasum -a 256 "/Users/chersu/workdir/AI/158BitNet/models/$f" | awk '{print $1}')
  R=$(sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 \
      "sha256sum ~/bitnet-test/models/$f" | awk '{print $1}')
  echo "$f: local=$L remote=$R $([ "$L" = "$R" ] && echo OK || echo MISMATCH)"
done
```
Expected: all `OK`.

No commit.

---

### Task 4: Sanity-check tier override on existing local x86_64 build

**Files:** none.

Purpose: confirm `BITNET_CPU_TIER` is honored by the existing build before relying on it remotely. The existing local `build/` is x86_64 under Rosetta and stays on scalar; this confirms the override path works at all.

- [ ] **Step 1: Run test_cpu_detect under each tier**

```bash
cd /Users/chersu/workdir/AI/158BitNet
for tier in scalar avx2 avx_vnni avx512_vnni; do
  echo "--- tier=$tier ---"
  BITNET_CPU_TIER=$tier BITNET_QUIET=1 ./build/test_cpu_detect
done
```
Expected: all four report `tier=scalar` (because the local CPU is actually Apple M4 under Rosetta which doesn't expose AVX). If you see warnings about unsupported tier overrides, that's expected and the script falls back to scalar — that's the test passing.

- [ ] **Step 2: Confirm test_dispatch_init still passes**

```bash
cd build && ctest -R test_dispatch_init --output-on-failure
```
Expected: `1 test passed`.

No commit.

---

### Task 5: Build on x86 Linux dell-precision-5810

**Files:** none created locally (build lives at `~/bitnet-test/build/` on dell-precision-5810).

- [ ] **Step 1: Configure + build on dell-precision-5810**

```bash
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 << 'EOF'
cd ~/bitnet-test
# pull source: rsync the repo from local
EOF

# Sync source tree to dell-precision-5810 (rsync, excludes build*/ + models/ + .git/ for speed)
rsync -a --delete \
  --exclude='build/' --exclude='build-arm64/' --exclude='build-strict/' \
  --exclude='models/' --exclude='.worktrees/' --exclude='.superpowers/' \
  /Users/chersu/workdir/AI/158BitNet/ \
  cuick@192.168.210.24:~/bitnet-test/repo/

sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 << 'EOF'
cd ~/bitnet-test/repo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
cmake --build build -j 16 2>&1 | tail -20
EOF
```
Expected: configure + build succeed; final lines report `[100%]` and `Built target ...`.

- [ ] **Step 2: Verify CPU tier auto-detect on dell-precision-5810**

```bash
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 \
  'cd ~/bitnet-test/repo && ./build/test_cpu_detect'
```
Expected: prints detected features + a tier. The exact tier depends on dell-precision-5810's CPU — note it for the report. It MUST NOT be `scalar` (otherwise the dispatch work is broken); ideally `avx2` or higher.

- [ ] **Step 3: Quick correctness smoke on dell-precision-5810 (ctest)**

```bash
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 \
  'cd ~/bitnet-test/repo && ctest --test-dir build -R "test_cpu_detect|test_dispatch_init|test_ops|test_i2s_correctness|test_quant_tq2_0|test_q6k_layout" --output-on-failure'
```
Expected: all six model-free tests pass.

- [ ] **Step 4: Generate the LoRA artifact on dell-precision-5810**

```bash
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 << 'EOF'
cd ~/bitnet-test/repo
cmake --build build --target train_xiaoli_lora -j 16 2>&1 | tail -5
./build/train_xiaoli_lora models/bitcpm4-1b-tq2_0.gguf ~/bitnet-test/lora/xiaoli.bnlora 2>&1 | tail -5
ls -la ~/bitnet-test/lora/
EOF
```
Expected: `xiaoli.bnlora` present (newly generated; may differ in bytes from local copy but will work for `test_lora_loader`).

No commit.

---

### Task 6: Build on local Mac (arm64 native)

**Files:** none created at repo root (build lives at `build-arm64/`, gitignored).

- [ ] **Step 1: Configure + build arm64 native**

```bash
cd /Users/chersu/workdir/AI/158BitNet
rm -rf build-arm64
cmake -S . -B build-arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
cmake --build build-arm64 -j 8 2>&1 | tail -20
```
Expected: configure + build succeed; built target list ends with `test_i2s_bench` or similar.

- [ ] **Step 2: Verify arch + tier**

```bash
file build-arm64/test_cpu_detect
./build-arm64/test_cpu_detect
```
Expected: `Mach-O 64-bit executable arm64` from `file`; tier detected should be ARM NEON (printed as `scalar` since the dispatch doesn't expose a NEON tier name in the public enum, but `bitnet_cpu_features_t` should show ARM features). Check the output for NEON/dotprod lines.

- [ ] **Step 3: Quick ctest on Mac arm64 (model-free tests)**

```bash
ctest --test-dir build-arm64 -R "test_cpu_detect|test_dispatch_init|test_ops|test_i2s_correctness|test_quant_tq2_0|test_q6k_layout" --output-on-failure
```
Expected: all pass.

- [ ] **Step 4: Stage the LoRA into build-arm64/**

```bash
cp /Users/chersu/workdir/AI/158BitNet/build/test_lora_loader.bnlora /Users/chersu/workdir/AI/158BitNet/build-arm64/
ls -la build-arm64/test_lora_loader.bnlora
```
Expected: file present.

No commit (build-arm64 is gitignored).

---

### Task 7: Build on Android via NDK cross-compile

**Files:** none.

- [ ] **Step 1: Verify NDK is present**

```bash
ls ~/Library/Android/sdk/ndk/ 2>&1 | head
echo "---"
ls ~/Library/Android/sdk/ndk/*/source.properties 2>&1 | head -1
```
Expected: at least one NDK version listed. Note the path for the env var.

- [ ] **Step 2: Run the Android build script**

```bash
cd /Users/chersu/workdir/AI/158BitNet
export ANDROID_NDK=$(ls -d ~/Library/Android/sdk/ndk/*/ | sort -V | tail -1 | sed 's:/$::')
echo "ANDROID_NDK=$ANDROID_NDK"
./scripts/build_android.sh openai_server minimal_generate gguf_inspect test_cpu_detect test_dispatch_init test_ops test_i2s_correctness test_quant_tq2_0 test_q6k_layout test_kv_cache test_tokenizer test_gguf test_lora_loader test_api_smoke test_generation_controls test_profile_decode test_i2s_bench 2>&1 | tail -30
```
Expected: each target reports `[100%] Built target <name>`. The list above includes everything we'll run on-device plus a couple of profiling bins.

- [ ] **Step 3: Verify the Android build dir**

```bash
ls -lh /Users/chersu/workdir/AI/158BitNet/build/android-arm64-v8a/test_profile_decode
file /Users/chersu/workdir/AI/158BitNet/build/android-arm64-v8a/test_profile_decode
```
Expected: file exists; `file` reports ELF 64-bit LSB shared object, ARM aarch64.

No commit (build/ is gitignored).

---

### Task 8: Push Android binaries + LoRA to device

**Files:** none.

- [ ] **Step 1: Create on-device workspace**

```bash
adb -s 192.168.210.10:5555 shell 'mkdir -p /data/local/tmp/bitnet-test /data/local/tmp/bitnet-test/lora && ls -la /data/local/tmp/bitnet-test'
```
Expected: empty dirs exist.

- [ ] **Step 2: Push binaries**

```bash
cd /Users/chersu/workdir/AI/158BitNet/build/android-arm64-v8a
for bin in openai_server minimal_generate gguf_inspect test_cpu_detect test_dispatch_init test_ops test_i2s_correctness test_quant_tq2_0 test_q6k_layout test_kv_cache test_tokenizer test_gguf test_lora_loader test_api_smoke test_generation_controls test_profile_decode; do
  adb -s 192.168.210.10:5555 push "$bin" "/data/local/tmp/bitnet-test/$bin" 2>&1 | tail -1
done
```
Expected: each line says `... KB/s (...)` — pushing succeeded.

- [ ] **Step 3: Push LoRA artifact**

```bash
adb -s 192.168.210.10:5555 push /Users/chersu/workdir/AI/158BitNet/build/test_lora_loader.bnlora /data/local/tmp/bitnet-test/lora/xiaoli.bnlora 2>&1 | tail -1
```
Expected: pushed.

- [ ] **Step 4: chmod +x on device**

```bash
adb -s 192.168.210.10:5555 shell 'chmod 755 /data/local/tmp/bitnet-test/* /data/local/tmp/bitnet-test/lora/xiaoli.bnlora'
```
Expected: silent success.

- [ ] **Step 5: Verify by running test_cpu_detect on device**

```bash
adb -s 192.168.210.10:5555 shell 'cd /data/local/tmp/bitnet-test && ./test_cpu_detect'
```
Expected: prints detected features (Cortex-A77 has NEON, no AVX).

No commit.

---

### Task 9: Run ctest on x86 Linux dell-precision-5810

**Files:** none.

- [ ] **Step 1: Configure summary for ctest rows**

Run ctest via a wrapper that records the test-binary as "ctest" so the summary script picks it up. The ctest invocation lives in `~/bitnet-test/repo/build/` — we use a one-off SSH that records the result.

```bash
sshpass -p 'cck@110119' ssh -o StrictHostKeyChecking=no cuick@192.168.210.24 << 'EOF' > build-strict/perf-logs/x86-linux/ctest.log 2>&1
cd ~/bitnet-test/repo
ctest --test-dir build --output-on-failure 2>&1
EOF
echo "ctest rc=$?"
tail -30 build-strict/perf-logs/x86-linux/ctest.log
```
Expected: tail shows `100% tests passed, 0 tests failed` (or similar).

- [ ] **Step 2: Manually append a summary row**

The driver script handles single runs; ctest is a wrapper. Append one row representing the ctest result.

```bash
SUMMARY=build-strict/perf-logs/x86-linux/summary.tsv
[[ -f $SUMMARY ]] || printf "model\tthreads\ttier\ttest_bin\trc\telapsed_s\ttok_per_s\n" > $SUMMARY
if ! grep -qF "bitcpm4-1b-tq2_0.gguf|0|default|ctest" $SUMMARY; then
  RC=$(grep -cE 'Test #.*Passed' build-strict/perf-logs/x86-linux/ctest.log || echo 0)
  FAIL=$(grep -cE 'Test #.*\*\*\*Failed' build-strict/perf-logs/x86-linux/ctest.log || echo 0)
  TOTAL=$((RC + FAIL))
  printf "bitcpm4-1b-tq2_0.gguf\t0\tdefault\tctest\t%s\t0\t\n" "$FAIL" >> $SUMMARY
fi
cat $SUMMARY
```
Expected: one row, `rc=<fail_count>`. If any test failed, fill it in and continue; if all passed, rc=0.

No commit.

---

### Task 10: Run ctest on local Mac arm64

**Files:** none.

- [ ] **Step 1: Run ctest**

```bash
cd /Users/chersu/workdir/AI/158BitNet
mkdir -p build-strict/perf-logs/mac
ctest --test-dir build-arm64 --output-on-failure > build-strict/perf-logs/mac/ctest.log 2>&1
echo "rc=$?"
tail -30 build-strict/perf-logs/mac/ctest.log
```
Expected: tail shows `100% tests passed, 0 tests failed`.

- [ ] **Step 2: Append ctest summary row**

```bash
SUMMARY=build-strict/perf-logs/mac/summary.tsv
[[ -f $SUMMARY ]] || printf "model\tthreads\ttier\ttest_bin\trc\telapsed_s\ttok_per_s\n" > $SUMMARY
if ! grep -qF "bitcpm4-1b-tq2_0.gguf|0|default|ctest" $SUMMARY; then
  FAIL=$(grep -cE 'Test #.*\*\*\*Failed' build-strict/perf-logs/mac/ctest.log || echo 0)
  printf "bitcpm4-1b-tq2_0.gguf\t0\tdefault\tctest\t%s\t0\t\n" "$FAIL" >> $SUMMARY
fi
cat $SUMMARY
```
Expected: one row with `rc=<fail_count>`.

No commit.

---

### Task 11: Run ctest on Android

**Files:** none.

The ctest-registered Android tests use `BITNET_SOURCE_DIR` baked at configure time to point at the host repo. They can't run directly on-device without source files. Run only the model-free tests via direct binary invocation (they don't need a model).

- [ ] **Step 1: Run model-free tests on device**

```bash
mkdir -p build-strict/perf-logs/android
adb -s 192.168.210.10:5555 shell 'cd /data/local/tmp/bitnet-test
for t in test_cpu_detect test_dispatch_init test_ops test_i2s_correctness test_quant_tq2_0 test_q6k_layout; do
  echo "=== $t ==="
  ./$t
done
' > build-strict/perf-logs/android/ctest.log 2>&1
echo "rc=$?"
tail -50 build-strict/perf-logs/android/ctest.log
```
Expected: every test ends with `OK` (test_cpu_detect prints `OK (tier=...)`).

- [ ] **Step 2: Run model-dependent tests on device (they need /data/models/bitcpm4-1b-tq2_0.gguf)**

The Android ctest binaries were built with `BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}"` so they look at the host source path. They WILL fail to load models when run on-device because the host path doesn't exist there. Run them with a tweak: cd into the source dir on host first. **Wait — they're already on device.** On-device they need `/data/local/tmp/bitnet-test/<model>` paths. Since the test paths are hard-coded at compile time, the model-dependent tests are effectively not runnable on-device with the current test setup.

Record this as an Issue:

```bash
SUMMARY=build-strict/perf-logs/android/summary.tsv
[[ -f $SUMMARY ]] || printf "model\tthreads\ttier\ttest_bin\trc\telapsed_s\ttok_per_s\n" > $SUMMARY
if ! grep -qF "bitcpm4-1b-tq2_0.gguf|0|default|ctest" $SUMMARY; then
  printf "bitcpm4-1b-tq2_0.gguf\t0\tdefault\tctest\tskipped_model_path\t0\t\n" >> $SUMMARY
fi
echo "model-dependent ctest tests skipped on Android — hard-coded BITNET_SOURCE_DIR path" >> build-strict/perf-logs/android/issues.txt
cat $SUMMARY
```
Expected: row with `rc=skipped_model_path`.

No commit.

---

### Task 12: Decode microbench on x86 Linux dell-precision-5810 (24 runs)

**Files:** none.

- [ ] **Step 1: Run the full decode matrix (4 models × 6 threads = 24 runs)**

```bash
cd /Users/chersu/workdir/AI/158BitNet
for model in bitcpm4-0.5b-tq2_0.gguf bitcpm4-1b-tq2_0.gguf bitcpm4-3b-tq2_0.gguf bitcpm4-8b-tq2_0.gguf; do
  for t in 1 2 3 4 6; do
    scripts/perf_sweep.sh x86-linux "$model" "$t" "" test_profile_decode
  done
done
tail -30 build-strict/perf-logs/x86-linux/summary.tsv
```
Expected: 24 new rows appended (or skipped if already present from a re-run). Each row has a `tok_per_s` value.

- [ ] **Step 2: Confirm at least one row has a non-empty tok/s**

```bash
awk -F'\t' 'NR>1 && $7 != "" {c++} END {print "rows with tok/s:", c+0}' build-strict/perf-logs/x86-linux/summary.tsv
```
Expected: `rows with tok/s: 24` (assuming first run). If 0, check logs under `build-strict/perf-logs/x86-linux/` for parse errors.

No commit.

---

### Task 13: Decode microbench on local Mac arm64 (24 runs)

**Files:** none.

- [ ] **Step 1: Run the full decode matrix**

```bash
cd /Users/chersu/workdir/AI/158BitNet
for model in bitcpm4-0.5b-tq2_0.gguf bitcpm4-1b-tq2_0.gguf bitcpm4-3b-tq2_0.gguf bitcpm4-8b-tq2_0.gguf; do
  for t in 1 2 3 4 6; do
    scripts/perf_sweep.sh mac "$model" "$t" "" test_profile_decode
  done
done
tail -30 build-strict/perf-logs/mac/summary.tsv
```
Expected: 24 rows added.

- [ ] **Step 2: Verify**

```bash
awk -F'\t' 'NR>1 && $7 != "" {c++} END {print "rows with tok/s:", c+0}' build-strict/perf-logs/mac/summary.tsv
```
Expected: `rows with tok/s: 24`.

No commit.

---

### Task 14: Decode microbench on Android (24 runs)

**Files:** none.

- [ ] **Step 1: Run the full decode matrix**

```bash
cd /Users/chersu/workdir/AI/158BitNet
for model in bitcpm4-0.5b-tq2_0.gguf bitcpm4-1b-tq2_0.gguf bitcpm4-3b-tq2_0.gguf bitcpm4-8b-tq2_0.gguf; do
  for t in 1 2 3 4 6; do
    scripts/perf_sweep.sh android "$model" "$t" "" test_profile_decode
  done
done
tail -30 build-strict/perf-logs/android/summary.tsv
```
Expected: 24 rows added. Note that on Android the 8B runs will be the slowest (per README ~8 tok/s baseline).

- [ ] **Step 2: Verify**

```bash
awk -F'\t' 'NR>1 && $7 != "" {c++} END {print "rows with tok/s:", c+0}' build-strict/perf-logs/android/summary.tsv
```
Expected: `rows with tok/s: 24`.

No commit.

---

### Task 15: x86 Linux tier sweep (96 runs)

**Files:** none.

- [ ] **Step 1: Run the tier sweep (4 models × 4 tiers × 6 threads = 96 runs)**

```bash
cd /Users/chersu/workdir/AI/158BitNet
for model in bitcpm4-0.5b-tq2_0.gguf bitcpm4-1b-tq2_0.gguf bitcpm4-3b-tq2_0.gguf bitcpm4-8b-tq2_0.gguf; do
  for tier in scalar avx2 avx_vnni avx512_vnni; do
    for t in 1 2 3 4 6; do
      scripts/perf_sweep.sh x86-linux "$model" "$t" "$tier" test_profile_decode
    done
  done
done
```
Expected: 96 rows appended. This is the longest single task in the plan (~45 min wall time on dell-precision-5810).

- [ ] **Step 2: Sanity-check that tier overrides took effect**

```bash
# For each tier, sample one log and confirm the startup banner reports the right tier
for tier in scalar avx2 avx_vnni avx512_vnni; do
  LOG=$(ls build-strict/perf-logs/x86-linux/*_${tier}_test_profile_decode.log 2>/dev/null | head -1)
  if [[ -n $LOG ]]; then
    echo "=== $tier (from $LOG) ==="
    head -5 "$LOG"
  fi
done
```
Expected: each tier's startup banner shows that tier. If `avx512_vnni` shows `avx2` (or similar fallback) it means dell-precision-5810 doesn't have AVX512-VNNI — record this as an Issue.

- [ ] **Step 3: Confirm full count**

```bash
awk -F'\t' 'NR>1 && $7 != "" {c++} END {print "rows with tok/s:", c+0}' build-strict/perf-logs/x86-linux/summary.tsv
```
Expected: 24 (decode) + 96 (tier sweep) = 120 rows with tok/s plus the 1 ctest row.

No commit.

---

### Task 16: Generate the report

**Files:**
- Create: `docs/superpowers/reports/2026-07-03-three-device-perf.md`

- [ ] **Step 1: Run the aggregator**

```bash
cd /Users/chersu/workdir/AI/158BitNet
python3 scripts/perf_summarize.py
```
Expected: `wrote docs/superpowers/reports/2026-07-03-three-device-perf.md`.

- [ ] **Step 2: Read the generated report and fill in Issues + Recommendations**

```bash
$EDITOR docs/superpowers/reports/2026-07-03-three-device-perf.md
```

Look at:
- "Issues found" — add anything anomalous from `build-strict/perf-logs/*/issues.txt`, plus anything that stood out in the logs (test failures, fallback warnings, surprising tok/s numbers).
- "Recommendations" — for each device, fill in:
  - Recommended default `BITNET_NUM_THREADS` (from the decode sweep, peak tok/s).
  - For Android: any `BITNET_OUTPUT_CHUNK_ROWS` observation if 0.5B shows chunk-related issues.
  - For x86 Linux: which tier to leave as auto-detect default.

- [ ] **Step 3: Verify the report renders**

```bash
head -80 docs/superpowers/reports/2026-07-03-three-device-perf.md
wc -l docs/superpowers/reports/2026-07-03-three-device-perf.md
```
Expected: report header + Summary table + three per-device sections + tier sweep + Issues + Recommendations. No `TBD` markers remain.

- [ ] **Step 4: Commit the report**

```bash
git add docs/superpowers/reports/2026-07-03-three-device-perf.md
git -c commit.gpgsign=false commit -m "docs: three-device adaptation + perf sweep report"
```

---

### Task 17: Final verification

**Files:** none.

- [ ] **Step 1: Confirm all summary.tsv files are populated**

```bash
for d in android mac x86-linux; do
  ROWS=$(wc -l < build-strict/perf-logs/$d/summary.tsv)
  echo "$d: $ROWS rows (incl header)"
done
```
Expected: android 25, mac 25, x86-linux 121 (header + 24 decode + 96 tier sweep + 1 ctest = 122; mac/android header + 24 decode + 1 ctest = 26).

- [ ] **Step 2: Confirm report is committed**

```bash
git log --oneline -5
git status --short
```
Expected: top commit is the report commit. No untracked files outside `build*/`, `models/`, `.superpowers/`, `.worktrees/`.

- [ ] **Step 3: Summary to user**

Print a one-paragraph summary to the user:
- Which devices built and ran successfully.
- Total decode runs captured.
- Whether the x86 tier sweep found any fallback warnings.
- Where the report lives and what's in "Issues found" / "Recommendations".