# Per-Device Adaptation & Perf Tuning — Three-Device Sweep

**Date**: 2026-07-03
**Status**: Approved (brainstorm)
**Owner**: TBD
**Branch**: `feature/x86-optimization`

## Summary

Run a full per-device adaptation + perf sweep across the three documented test devices (Android Snapdragon 865, local Apple Silicon Mac, x86 Linux server dell-precision-5810) using all four BitCPM4 model sizes (0.5B / 1B / 3B / 8B). Capture correctness (`ctest`) and decode performance (`test_profile_decode`) on every device, plus a `BITNET_CPU_TIER` sweep on x86 Linux to validate the new runtime dispatch work that is the reason this branch exists. Produce a single markdown report.

## Goals

1. Build the runtime successfully on each of the three devices, using the appropriate build path per arch (native CMake on Mac/Linux, NDK cross-compile on Android).
2. Run the full `ctest` suite on each device and capture pass/fail per test.
3. Run `test_profile_decode` with `BITNET_NUM_THREADS ∈ {1,2,3,4,6}` for all four model sizes on each device.
4. On x86 Linux only: run the decode workload under all four `BITNET_CPU_TIER` values (`scalar` / `avx2` / `avx_vnni` / `avx512_vnni`) to validate runtime dispatch selection.
5. Commit a single report at `docs/superpowers/reports/2026-07-03-three-device-perf.md` with summary tables and per-device detail.

## Non-Goals

- No source-code changes. The spec assumes the current `feature/x86-optimization` tree builds and runs on all three devices; if a real bug surfaces, we record it under "Issues found" but do not fix it in this pass.
- No HTTP server end-to-end tests, no LoRA API tests, no Q8-KV-cache profiling. The user-selected workload scope is `ctest` + `test_profile_decode` + x86 tier sweep only.
- No kernel-level microbenches (no `test_i2s_bench`, `test_tq2_kernel_bench`, `test_dequant_bench`). Those exist for kernel-development work and are out of scope here.
- No comparison against the README's Snapdragon 865 historical best-known numbers beyond reporting what we observe. The README numbers are already-baselined; we are not retuning to them.

## Devices

| Device | Address | Effective arch | Tiers reachable | Build path |
|---|---|---|---|---|
| Android 865 | `192.168.210.10` (ADB) | aarch64 (Cortex-A77) | scalar + ARM NEON (no AVX) | NDK cross-compile via `scripts/build_android.sh` → `build/android-arm64-v8a/` |
| Local Mac (Apple M4) | this host | arm64 native | scalar + ARM NEON | `cmake -S . -B build-arm64` + `cmake --build build-arm64` |
| x86 Linux (dell-precision-5810) | `192.168.210.24` (SSH, user `cuick`) | x86_64 (40 cores) | scalar / AVX2 / AVX-VNNI / AVX512-VNNI | `cmake -S . -B build` + `cmake --build build` |

The Mac builds **arm64 native** (not x86_64 under Rosetta 2). Rosetta 2 does not expose AVX2+ to x86 guests, so a Rosetta build would always sit on the scalar fallback — useless for tuning. The arm64 native path gives us the real Apple Silicon ARM NEON number that matches an M-series Mac's actual capability.

## Workload matrix

```
                 ctest    decode × models × threads                x86 tier sweep
Android          yes      {0.5B,1B,3B,8B} × {1,2,3,4,6}              n/a
Mac (arm64)      yes      {0.5B,1B,3B,8B} × {1,2,3,4,6}              n/a
x86 Linux        yes      {0.5B,1B,3B,8B} × {1,2,3,4,6}              {0.5B,1B,3B,8B} × {scalar,avx2,avx_vnni,avx512_vnni} × {1,2,3,4,6}
```

- `ctest` is internally pinned to 1B (`tests/*` hard-code `bitcpm4-1b-tq2_0.gguf`). One `ctest --output-on-failure` per device.
- `decode` is `./build/test_profile_decode <model>` with `BITNET_NUM_THREADS=N`. Each invocation uses the prompt `"The capital of France is"` and `max_tokens=64` (matches the README's microbench pattern).
- `tier sweep` runs the same `test_profile_decode` workload under `BITNET_CPU_TIER=<tier>` on x86 Linux only. Purpose: validate that `bitnet_dispatch_init()` correctly selects each tier when forced via the env override.

Per-device run count:

- Android: 1 + 4×6 = **25 runs**.
- Mac: 1 + 4×6 = **25 runs**.
- x86 Linux: 1 + 4×6 + 4×4×6 = **121 runs**.

Wall-time estimate (rough, based on README's 865 baseline + extrapolation):

- Android: ctest ~1 min, decode 8B ~10 min, others < 4 min ⇒ ~15 min.
- Mac (arm64): should be comparable to Android or faster on 1B/0.5B; 8B can be slow. ~20 min.
- x86 Linux tier sweep: 4× the basic decode count at ~1B-scale throughput. ~45 min.

## Model transfer

Source of truth: `/data/models/` on Android (already contains all four GGUF files).

- **To local Mac**: `adb -s 192.168.210.10:5555 pull /data/models/<file> /Users/chersu/workdir/AI/158BitNet/models/` for each file. ~4 GB total.
- **To x86 Linux (dell-precision-5810)**: `scp models/*.gguf cuick@192.168.210.24:~/bitnet-test/models/`. ~4 GB total over LAN.

Local Mac keeps models at `/Users/chersu/workdir/AI/158BitNet/models/`. x86 Linux keeps them at `~/bitnet-test/models/`. Android keeps them at the existing `/data/models/`.

## Build artifacts

- **Android**: `build/android-arm64-v8a/` (driven by `scripts/build_android.sh`). Binaries copied to `/data/local/tmp/bitnet-test/` on device.
- **Mac**: `build-arm64/` (native arm64 build). Binaries run from the host.
- **x86 Linux**: `~/bitnet-test/build/` (CMake build on the remote host). Binaries run via `ssh`.

LoRA artifact for `test_lora_loader`: the existing local `build/test_lora_loader.bnlora` was generated from a prior `train_xiaoli_lora` run. On Mac we copy it from `build/` into `build-arm64/` (it's a small file, copy is enough). On x86 Linux we copy it to `~/bitnet-test/lora/xiaoli.bnlora` and re-run `train_xiaoli_lora` once after build to keep it reproducible from a fresh tree. On Android we push the same file to `/data/local/tmp/bitnet-test/lora/`.

## Test execution driver

A single bash driver script `scripts/perf_sweep.sh` orchestrates the sweep. Each run writes:
- stdout/stderr to `build-strict/perf-logs/<device>/<run-id>.log`
- a one-line summary (model, threads, tier, tok/s, pass/fail) to `build-strict/perf-logs/<device>/summary.tsv`

The script is idempotent — re-running it skips runs whose `summary.tsv` row is already populated. This makes the sweep recoverable across long pauses.

## Report

`docs/superpowers/reports/2026-07-03-three-device-perf.md` with these sections:

1. **Summary** — one row per device, columns: ctest result, best tok/s for each of 0.5B/1B/3B/8B, observed tier on x86 Linux, any caveats.
2. **Per-device detail** — three subsections (Android, Mac, x86 Linux). Each contains:
   - ctest table (test name, result, notes).
   - decode table (model × thread × tok/s).
3. **x86 Linux tier sweep** — model × tier × thread × tok/s grid. This validates the dispatch work: `scalar` should match the no-override row in §2; `avx2` should be ≥ `scalar`; `avx512_vnni` should be the fastest tier where supported.
4. **Issues found** — anything anomalous: build failures, tests that fail or were skipped, decode numbers that look wrong (e.g. tier override didn't take effect). Empty if nothing found.
5. **Recommendations** — concrete tuning knobs (suggested default `BITNET_NUM_THREADS` per device, suggested `BITNET_OUTPUT_CHUNK_ROWS` for Android if relevant, any tier that should be the default per device).

## Risks

- **8B decode on Android is slow** (~8 tok/s baseline per README; 64 tokens ≈ 8 s per run × 6 thread configs = ~50 s for 8B on Android alone). Tolerable but the slowest part of the Android sweep.
- **8B on x86 Linux tier sweep is 4×** that ⇒ ~3 min for 8B tier sweep alone. Total x86 sweep ~45 min, fits inside a single shell session.
- **macOS arm64 build**: the existing `build/` is x86_64 (Rosetta). We'll create a fresh `build-arm64/` to avoid touching the working tree; both can coexist because the build dirs are separate.
- **Build flag divergence**: NDK cross-compile uses `-march=armv8.2-a+dotprod`; native Mac arm64 build inherits the default `APPLE` branch which checks `-mcpu=native / apple-m2 / apple-m1` flags. We do not override these — let CMake pick what works.
- **x86 Linux tier sweep may hit CPU feature detection surprises**. `BITNET_CPU_TIER=avx512_vnni` on a host without AVX512-VNNI will fall back to auto-detect with a warning. We record the warning but treat the run as that tier for reporting.

## Out-of-band cleanup

- Model files land in `models/` on local Mac (already gitignored).
- Log files land in `build-strict/perf-logs/` (already gitignored via `build-strict/`).
- No tracked file should grow beyond the report markdown.