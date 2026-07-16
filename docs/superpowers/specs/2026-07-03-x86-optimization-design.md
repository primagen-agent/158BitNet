# x86 SIMD Optimization with Runtime CPU Dispatch — Design

**Date**: 2026-07-03
**Status**: Approved (brainstorm)
**Owner**: TBD

## Summary

158BitNet's runtime currently ships ARM-only SIMD optimization. Every NEON /
NEON-dotprod code path is gated at **compile time** by `#ifdef __ARM_NEON` /
`__ARM_FEATURE_DOTPROD`, and the CMake build only sets ARM march flags. On x86,
the project compiles to scalar fallbacks — correct, but slow.

This design adds full x86 SIMD parity with **runtime CPU detection** at startup,
auto-selecting among four tiers:

- **Scalar** — baseline fallback
- **AVX2 + FMA** — Haswell+ / Zen 2+
- **AVX-VNNI** — Ice Lake+ / Zen 4 (`_mm256_dpbusd_epi32`)
- **AVX512-VNNI** — Ice Lake+ / Zen 4 (`_mm512_dpbusd_epi32`)

Three OSes are targeted: **Linux, macOS, Windows** (GCC/Clang + MSVC). ARM
behaviour is unchanged.

## Goals

1. Every ARM NEON kernel path in `quant_tq2_0.c`, `quant_q6k.c`, `ops.c`, and
   the bitnet.c hot-path NEON blocks has an x86 equivalent at every tier.
2. A single binary runs on any x86-64 CPU and silently picks the fastest safe
   tier at first kernel invocation via `pthread_once`.
3. The existing public API (`quant_tq2_0.h`, `quant_q6k.h`, `ops.h`) is
   unchanged. Existing call sites in `bitnet.c` are unchanged.
4. ARM builds compile byte-for-byte the same as today.
5. CI exercises x86 builds on Linux, macOS, and Windows for every push.

## Non-Goals

- No change to ARM compile-time selection (`#ifdef __ARM_NEON`).
- No change to GGUF on-disk format.
- No change to public API signatures.
- No GPU work — the Metal path stays ARM-targeted.
- No ARM-x86 universal-binary work beyond the optional macOS universal flag.

## Architecture

### Module Layout

```
src/
  bitnet.c                ← unchanged; calls public API (now trampolined)
  quant_tq2_0.c           ← ARM + scalar paths unchanged
  quant_q6k.c             ← ARM + scalar paths unchanged
  ops.c                   ← ARM + scalar paths unchanged
  bitnet_internal.h       ← unchanged

  cpu_detect.h            ← NEW: bitnet_cpu_features_t, bitnet_cpu_pick_tier()
  cpu_detect.c            ← NEW: __builtin_cpu_supports / __cpuid impls
  bitnet_dispatch.h       ← NEW: struct of function pointers (one per kernel)
  bitnet_dispatch.c       ← NEW: pthread_once init that picks tier & fills table

  x86/
    quant_tq2_0_x86.c     ← AVX2 / AVX-VNNI / AVX512-VNNI impls of TQ2_0 kernels
    quant_q6k_x86.c       ← same for Q6K
    ops_x86.c             ← same for RMSNorm / softmax / etc.
    bitnet_hotpath_x86.c  ← same for the bitnet.c-internal NEON blocks
    kernel_registry.c     ← NEW: 4 function-pointer tables (one per tier)
    pack_x86.h            ← NEW: per-tier pack/unpack helpers + layout docs
    pack_x86.c            ← NEW: bitnet_tq2_0_reorder_to_i2s_x8 (8-row),
                                bitnet_tq2_0_reorder_to_i2s_x16 (16-row)
```

### Dispatch Flow

1. `bitnet_cpu_init()` runs under `pthread_once` on the first call to any
   public kernel.
2. It queries CPU features via `cpu_detect.c` and picks one of
   `BITNET_TIER_SCALAR / AVX2 / AVX_VNNI / AVX512_VNNI`.
3. It sets the global `g_bitnet_dispatch = &g_dispatch_<tier>`.
4. Every public kernel (`bitnet_tq2_0_matmul_i2s_parallel`,
   `bitnet_q6k_matmul_*`, `bitnet_rms_norm_eps`, …) becomes a 3-line
   trampoline:

   ```c
   return g_bitnet_dispatch->tq2_matmul_i2s_parallel(...);
   ```

5. Model-load code in `bitnet.c` calls
   `g_bitnet_dispatch->pick_tq2_pack_format()` to choose I2S layout (4-row ARM,
   8-row AVX2, 16-row AVX512); the loaded format is stored on the model so the
   matching decode kernel is invoked.

### API Stability

Every existing function in `quant_tq2_0.h`, `quant_q6k.h`, `ops.h` keeps its
signature and behaviour. ARM builds compile exactly as today; the trampolines
just resolve to the existing implementations.

## CPU Detection

### Public Interface

```c
typedef enum {
    BITNET_TIER_SCALAR,
    BITNET_TIER_AVX2,
    BITNET_TIER_AVX_VNNI,
    BITNET_TIER_AVX512_VNNI,
} bitnet_cpu_tier_t;

typedef struct {
    int has_sse3;        // _mm_shuffle_epi8 — baseline on x86-64
    int has_avx2;
    int has_fma;
    int has_avx_vnni;    // AVX-VNNI (Ice Lake+, Zen 4)
    int has_avx512f;
    int has_avx512bw;
    int has_avx512vnni;  // Ice Lake+, Zen 4 (via AVX512)
} bitnet_cpu_features_t;

bitnet_cpu_tier_t bitnet_cpu_pick_tier(void);
const char *bitnet_cpu_tier_name(bitnet_cpu_tier_t);
```

### Per-Platform Implementation

- **Linux/macOS GCC/Clang**: `__builtin_cpu_init()` +
  `__builtin_cpu_supports("avx2" / "avx512vnni" / "avxvnni" / "fma")`. The
  builtin already handles OS-level XGETBV checks.
- **Windows MSVC**: `__cpuid` (leaf 1 ECX for SSE/AVX baseline),
  `__cpuidex(., 7, 0)` (EBX for AVX2, ECX bit 11 for AVX512-VNNI), leaf `7.1`
  (EAX bit 4 for AVX-VNNI), plus `_xgetbv(_XCR_XFEATURE_ENABLED_MASK_MASK)` to
  confirm the OS saves YMM/ZMM state. Silently downgrades if the OS doesn't
  support a feature.

### Tier Selection

Priority order:

| Condition | Tier |
|---|---|
| `has_avx512f && has_avx512bw && has_avx512vnni` | AVX512_VNNI |
| `has_avx2 && has_avx_vnni && has_fma` | AVX_VNNI |
| `has_avx2 && has_fma` | AVX2 |
| else | SCALAR |

Every tier is gated on `has_sse3` for shuffle-based table lookups (always true
on x86-64 in practice).

### Startup Logging

On first call, emit one line to `stderr`:

```
[bitnet] cpu tier: avx2_vnni (avx2+fma+avxvnni)
```

Bypassed if `BITNET_QUIET=1` is set.

### Manual Override

`BITNET_CPU_TIER=scalar|avx2|avx_vnni|avx512_vnni` forces a tier for benchmarking
/ A-B comparison. Validated against detected features; falls back with a warning
if the CPU can't actually run the requested tier.

## Per-Tier Weight Packing

The packed format is chosen once at model load based on the active tier, then
that tier's decode kernels consume it.

### Layout Definitions

| Pack format | Used by tier | Width | Alignment |
|---|---|---|---|
| `BITNET_TQ2_PACK_TQ2_0` | SCALAR | baseline | — |
| `BITNET_TQ2_PACK_I2S_ARM` | ARM (existing) | 4 rows | 16 B |
| `BITNET_TQ2_PACK_I2S_X8` | AVX2, AVX_VNNI | 8 rows | 32 B |
| `BITNET_TQ2_PACK_I2S_X16` | AVX512_VNNI | 16 rows | 64 B |

### Why These Widths

- AVX2 `dpbusd_epi32` processes 32 int8 lanes per instruction → 8 rows × 4-element
  block groups hit the lanes cleanly.
- AVX512-VNNI `dpbusd_epi32` processes 64 int8 lanes → 16 rows fit naturally.
- 8/16-row groups improve activation-vector reuse: each quantized activation
  vector is loaded once and dotted against many weight rows before eviction
  from L1.

### Layout Details

Per I2S-X8 block group (256-element TQ2_0 block × 8 rows):

```
[ scales:    8 × fp16  ]    (32 bytes, 32-byte aligned)
[ packed qs: 8 × 64 bytes ] (512 bytes, 32-byte aligned)
[ bsums:     8 × int32  ]   (32 bytes, optional — VNNI paths only)
```

I2S-X16 is the same pattern doubled, 64-byte aligned.

### API Additions

```c
typedef enum {
    BITNET_TQ2_PACK_TQ2_0,
    BITNET_TQ2_PACK_I2S_ARM,
    BITNET_TQ2_PACK_I2S_X8,
    BITNET_TQ2_PACK_I2S_X16,
} bitnet_tq2_pack_format_t;

size_t bitnet_tq2_0_i2s_x8_packed_size(int out_dim, int in_dim);
int    bitnet_tq2_0_reorder_to_i2s_x8(const void *weight, int out_dim, int in_dim,
                                       uint8_t *packed, float *packed_scales,
                                       int32_t *packed_bsums);
/* similarly for i2s_x16 */
```

### Model-Load Flow

Today `bitnet.c` calls `bitnet_tq2_0_reorder_to_i2s(...)` unconditionally at
load. After this work it calls
`g_bitnet_dispatch->pick_tq2_pack_format()` (returns one of the four enum
values), then the matching reorder function. The chosen format is stored on
the loaded model (`bitnet_model_t->tq2_pack_format`).

### Decode Flow

The existing decode already routes through the dispatch table, so the right
kernel for the loaded format gets called automatically. Each kernel asserts
its expected format on first call (debug builds only) to catch load/decode
format mismatches.

### Backward Compatibility

GGUF files on disk are unchanged. The packed-format choice happens entirely
in memory at load time.

## Kernel Intrinsic Mapping

### Translation Table

| ARM (NEON / dotprod) | x86 — AVX2 | x86 — AVX-VNNI | x86 — AVX512-VNNI |
|---|---|---|---|
| `vdotq_s32(c, a, b)` | `_mm256_maddubs_epi16` + `_mm256_madd_epi16` (2-step) | `_mm256_dpbusd_epi32` | `_mm512_dpbusd_epi32` |
| `vqtbl1q_s8(table, idx)` | `_mm_shuffle_epi8` (SSE3) | `_mm_shuffle_epi8` | two `_mm512_shuffle_epi8` halves (AVX512BW) |
| `vld1q_s8/p` | `_mm_loadu_si128` / `_mm256_loadu_si256` | same | `_mm512_loadu_si512` |
| `vld1q_f32` | `_mm_loadu_ps` / `_mm256_loadu_ps` | same | `_mm512_loadu_ps` |
| `vfmaq_f32(a,b,c)` | `_mm256_fmadd_ps` (FMA3) | same | `_mm512_fmadd_ps` |
| `vaddvq_f32` | `_mm256_hadd_ps` reduce + extract | same | `_mm512_reduce_add_ps` |
| `vmaxvq_f32` | extract + manual max | same | `_mm512_reduce_max_ps` |
| `vabsq_f32`, `vmaxq_f32`, `vmulq_n_f32` | `_mm256_and_ps` (sign mask), `_mm256_max_ps`, `_mm256_mul_ps` | same | `_mm512_*` |
| `vdupq_n_f32(x)` | `_mm256_set1_ps(x)` | same | `_mm512_set1_ps` |

### Semantic Notes

1. **`vqtbl1q_s8` vs `_mm_shuffle_epi8`**: ARM returns 0 for indices ≥ 16;
   SSSE3 shuffle zeroes bytes with the high index bit set. Behaviour matches
   exactly for the `{0,1,2,3}` ternary codes used in TQ2_0.

2. **`vdotq_s32` signedness**: ARM's `vdotq_s32` is signed×signed→int32.
   AVX-VNNI's `_mm256_dpbusd_epi32` is **unsigned×signed** (uint8 × int8). The
   TQ2_0 "bsums trick" decodes activations as `{0,1,2}` (non-negative), so
   they fit `uint8` cleanly. The bsums subtraction still happens once per
   block; results are bit-identical to ARM.

3. **AVX2 (no VNNI) emulation**: `_mm256_maddubs_epi16` (uint8×int8→int16) +
   `_mm256_madd_epi16` (int16×int16→int32, horizontal). Real-world throughput
   is roughly 40–60% of AVX-VNNI for these kernels — the reason the AVX-VNNI
   tier exists.

4. **Dual-accumulator ILP**: ARM code uses two `vdotq_s32` accumulators to
   break dependency chains. The x86 ports mirror this: two
   `_mm256_dpbusd_epi32` accumulators on AVX-VNNI; four accumulators on
   AVX512-VNNI since 512-bit ports are fewer.

### Per-Platform File Organization

- **GCC/Clang**: single `quant_tq2_0_x86.c` file, three function variants
  tagged with `__attribute__((target("avx2,fma")))`,
  `__attribute__((target("avx2,fma,avxvnni")))`,
  `__attribute__((target("avx512f,avx512bw,avx512vnni")))`.

- **MSVC**: same source files compiled three times into separate OBJECT
  libraries with different `/arch:` flags and a `BITNET_X86_TIER=` macro that
  suffixes every public symbol (`tq2_matmul_i2s_avx2`,
  `tq2_matmul_i2s_avxvnni`, `tq2_matmul_i2s_avx512vnni`). The MSVC
  `kernel_registry.c` references all three suffixed sets.

## Build System

### Architecture Detection

```cmake
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" BITNET_ARCH)
```

Possible values: `x86_64` / `amd64` (Linux/macOS/Windows Intel),
`arm64` / `aarch64` (existing), `i686` (32-bit, falls back to scalar).

### x86 Source Set

```cmake
if(BITNET_ARCH STREQUAL "x86_64" OR BITNET_ARCH STREQUAL "amd64")
    target_sources(bitnet PRIVATE
        src/cpu_detect.c
        src/bitnet_dispatch.c
        src/x86/quant_tq2_0_x86.c
        src/x86/quant_q6k_x86.c
        src/x86/ops_x86.c
        src/x86/bitnet_hotpath_x86.c
        src/x86/pack_x86.c
        src/x86/kernel_registry.c)
    target_compile_definitions(bitnet PUBLIC BITNET_HAS_X86=1)
endif()
```

### Per-Tier Compilation

- **GCC/Clang**: file compiled once with `-msse3` baseline; per-function
  `__attribute__((target(...)))` handles SIMD tiers. No extra flags needed.

- **MSVC**: three OBJECT libraries on Windows, each compiling the same
  `*_x86.c` files with a different `/arch:` flag and a different
  `BITNET_X86_TIER=` macro:

  ```cmake
  if(MSVC)
      add_library(bitnet_x86_avx2 OBJECT ${BITNET_X86_SOURCES})
      target_compile_definitions(bitnet_x86_avx2 PRIVATE BITNET_X86_TIER=2)
      target_compile_options(bitnet_x86_avx2 PRIVATE /arch:AVX2)

      add_library(bitnet_x86_avxvnni OBJECT ${BITNET_X86_SOURCES})
      target_compile_definitions(bitnet_x86_avxvnni PRIVATE BITNET_X86_TIER=3)
      # /arch:AVX512 makes AVX-VNNI intrinsics available; the source gates
      # which intrinsics are actually emitted via __AVX_VNNI__ feature macros
      # so no AVX512-only instructions land in this object.
      target_compile_options(bitnet_x86_avxvnni PRIVATE /arch:AVX512)

      add_library(bitnet_x86_avx512 OBJECT ${BITNET_X86_SOURCES})
      target_compile_definitions(bitnet_x86_avx512 PRIVATE BITNET_X86_TIER=4)
      target_compile_options(bitnet_x86_avx512 PRIVATE /arch:AVX512)

      target_link_libraries(bitnet PRIVATE
          bitnet_x86_avx2 bitnet_x86_avxvnni bitnet_x86_avx512)
  endif()
  ```

  The `BITNET_X86_TIER` macro drives a `#define` that suffixes every public
  symbol so the three object libraries don't collide at link time.

### macOS Universal Binary

Drop `set(CMAKE_OSX_ARCHITECTURES arm64)` when targeting x86 macOS. Optional
`BITNET_BUILD_UNIVERSAL=ON` adds `-arch arm64 -arch x86_64` on macOS so one
binary runs on both Apple Silicon and Intel with proper per-arch dispatch.

## CI Matrix

New `.github/workflows/ci-x86.yml`:

| Job | OS | Compiler | Notes |
|---|---|---|---|
| `linux-x86-gcc` | ubuntu-22.04 | gcc 11 | AVX2 hardware; runs ctest |
| `linux-x86-clang` | ubuntu-22.04 | clang 14 | AVX2 hardware; runs ctest |
| `linux-x86-newer` | ubuntu-24.04 | gcc 13 | often AVX-VNNI; runs ctest |
| `macos-intel` | macos-13 | clang (Xcode) | AVX2; runs ctest |
| `windows-msvc` | windows-2022 | MSVC 19.39 | AVX2/AVX-VNNI; runs ctest |
| `qemu-avx512` | ubuntu-22.04 + QEMU | gcc | AVX512-tier tests via `qemu-x86_64 -cpu max` |

Every job: `cmake -S . -B build && cmake --build build -j && ctest --test-dir
build --output-on-failure`. Existing tests cover correctness across all
dispatch tiers via the `BITNET_CPU_TIER` env override.

### Local Benchmarking

New `test_x86_tier_bench.c` compares all four tiers on the host's hardware so
the speedup is visible against the actual model files used.

## Phasing

Seven phases, each a self-contained, mergeable branch with full tests passing.
The ARM-shipping path stays safe at every step.

| # | Phase | Deliverable | Risk |
|---|---|---|---|
| 1 | Scaffolding | `cpu_detect.{c,h}`, `bitnet_dispatch.{c,h}`, `kernel_registry.c` wired into CMake for x86_64 + Windows MSVC. All four tiers resolve to scalar trampolines. ARM builds bit-identical. | Low |
| 2 | ops.c parity | AVX2 / AVX-VNNI / AVX512-VNNI of `rms_norm_eps`, `rms_norm_inplace_eps`, softmax, and the other `ops.c` NEON blocks. | Low |
| 3 | TQ2_0 baseline kernels | AVX2 variants of `matmul_vector_lut*`, `matmul_vector_i8_neon*` (using `_mm_shuffle_epi8` for VTBL, `_mm256_maddubs/madd` for dotprod emulation). No weight-layout changes yet. | Medium |
| 4 | Q6K output projection | AVX2 + AVX-VNNI Q6K kernels. Output projection dominates decode cost for small models — where AVX-VNNI starts to pay off. | Medium |
| 5 | AVX-VNNI tier | Replace AVX2 emulation in TQ2_0/Q6K with `_mm256_dpbusd_epi32`. Significant throughput jump on Ice Lake+ / Zen 4. | Low |
| 6 | Wider packing (I2S-X8, I2S-X16) | Per-tier reorder at model load + new AVX2 (8-row) and AVX512-VNNI (16-row) kernels. | Higher |
| 7 | AVX512-VNNI tier | 512-bit kernels (`_mm512_dpbusd_epi32`), 4-accumulator ILP, I2S-X16 packing. ~1.5–2× over AVX-VNNI on supported CPUs. | Medium |

**Phases 1–5 deliver ~85% of the achievable x86 speedup**; phases 6–7 are the
tuning layers that close the gap to ARM-class decode.

## Estimated Effort

- ~3500 lines of new x86 intrinsic code (TQ2_0: ~2000, Q6K: ~600, ops: ~400,
  bitnet hotpath: ~500).
- ~600 lines of dispatch / detect / pack scaffolding.
- ~600 lines of CMake / CI / script changes.
- ~7 reviewable PRs across the seven phases.

## Validation

- Existing ctest suite runs unchanged across every tier via
  `BITNET_CPU_TIER=...` env override.
- `test_x86_tier_bench.c` produces a per-tier tok/s report for each model.
- CI matrix covers Linux / macOS / Windows / QEMU-AVX512.
- ARM builds compile byte-for-byte the same as today (verified via CI).
