# x86 SIMD Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime-dispatched x86 SIMD optimization (4 tiers — scalar, AVX2, AVX-VNNI, AVX512-VNNI) to the 158BitNet C inference runtime, with full parity to the existing ARM NEON paths and zero API breakage.

**Architecture:** Per-tier x86 kernels live in `src/x86/` and register into a single function-pointer dispatch table. `pthread_once` at first kernel call queries CPU features and selects the best safe tier. ARM builds compile unchanged (still compile-time `#ifdef __ARM_NEON`). Public API (`quant_tq2_0.h`, `quant_q6k.h`, `ops.h`) is preserved by replacing each public function with a one-line trampoline through the dispatch table.

**Tech Stack:** C11, CMake 3.20+, GCC/Clang `__attribute__((target(...)))` for per-function multi-versioning, MSVC OBJECT libraries with `/arch:` flags for Windows, `__builtin_cpu_supports` / `__cpuid` + `_xgetbv` for runtime detection. Tests use the existing CTest framework.

## Global Constraints

- **C standard**: C11, no compiler extensions (`CMAKE_C_EXTENSIONS OFF` already set).
- **No public API changes**: every function in `quant_tq2_0.h`, `quant_q6k.h`, `ops.h` keeps its exact signature.
- **ARM builds must compile byte-identical**: never modify existing `#if defined(__ARM_NEON)` blocks except to insert dispatch trampolines at function entry. Verify by diffing build artifacts on ARM before/after Phase 1.
- **Tier count**: exactly 4 — `BITNET_TIER_SCALAR`, `BITNET_TIER_AVX2`, `BITNET_TIER_AVX_VNNI`, `BITNET_TIER_AVX512_VNNI`. Do not add tiers.
- **OS coverage**: Linux (GCC + Clang), macOS x86_64 (Clang), Windows (MSVC 19.39+). All three must build and pass ctest on every push.
- **Tier override env var**: `BITNET_CPU_TIER` accepts exactly `scalar`, `avx2`, `avx_vnni`, `avx512_vnni`. Anything else is rejected with a stderr warning and falls back to auto-detect.
- **Quiet env var**: `BITNET_QUIET=1` suppresses the startup tier log line.
- **No GPU scope**: leave `src/bitnet_metal.{h,mm}` and the `BITNET_ENABLE_METAL` option untouched.
- **Build output**: all binaries, test outputs, and artifacts stay under `build/`.
- **Commit cadence**: every step ends with a green test run and a commit.

## Spec Reference

This plan implements [docs/superpowers/specs/2026-07-03-x86-optimization-design.md](../specs/2026-07-03-x86-optimization-design.md). Each phase maps to a row in the spec's Section 6 phasing table.

---

## Phase 1 — Scaffolding (cpu_detect, dispatch, registry, CMake, CI)

Goal: ship infrastructure that compiles on x86 (GCC/Clang/MSVC) and ARM, runs the existing test suite unchanged via scalar trampolines, and prints the selected tier at startup. Zero new SIMD code in this phase.

### Task 1.1: Create the `cpu_detect` module

**Files:**
- Create: `src/cpu_detect.h`
- Create: `src/cpu_detect.c`
- Test: `tests/test_cpu_detect.c`

**Interfaces:**
- Produces: `bitnet_cpu_tier_t` enum, `bitnet_cpu_features_t` struct,
  `bitnet_cpu_detect()`, `bitnet_cpu_pick_tier()`, `bitnet_cpu_tier_name()`,
  `bitnet_cpu_tier_from_string()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cpu_detect.c`:

```c
#include "cpu_detect.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

int main(void) {
    bitnet_cpu_features_t f = bitnet_cpu_detect();
    /* All x86-64 CPUs since 2003 support SSE3 — the table-lookup primitive. */
#if defined(__x86_64__) || defined(_M_X64)
    assert(f.has_sse3 && "SSE3 must be present on x86-64");
#endif

    bitnet_cpu_tier_t t = bitnet_cpu_pick_tier();
    assert(t >= BITNET_TIER_SCALAR && t <= BITNET_TIER_AVX512_VNNI);

    /* Tier name round-trips through from_string. */
    const char *name = bitnet_cpu_tier_name(t);
    assert(name != NULL && strlen(name) > 0);
    assert(bitnet_cpu_tier_from_string(name) == t);

    /* Invalid override string returns -1 (caller falls back to auto). */
    assert(bitnet_cpu_tier_from_string("nonsense") == (bitnet_cpu_tier_t)-1);

    /* Specific valid strings map to their tiers. */
    assert(bitnet_cpu_tier_from_string("scalar") == BITNET_TIER_SCALAR);
    assert(bitnet_cpu_tier_from_string("avx2") == BITNET_TIER_AVX2);
    assert(bitnet_cpu_tier_from_string("avx_vnni") == BITNET_TIER_AVX_VNNI);
    assert(bitnet_cpu_tier_from_string("avx512_vnni") == BITNET_TIER_AVX512_VNNI);

    printf("test_cpu_detect: OK (tier=%s)\n", name);
    return 0;
}
```

- [ ] **Step 2: Add test to CMakeLists**

In `CMakeLists.txt`, inside the `if(BITNET_BUILD_TESTS)` block (just before `endif()` at line 230), add:

```cmake
add_executable(test_cpu_detect tests/test_cpu_detect.c)
target_include_directories(test_cpu_detect PRIVATE src)
target_link_libraries(test_cpu_detect PRIVATE bitnet)
add_test(NAME test_cpu_detect COMMAND test_cpu_detect)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cmake --build build --target test_cpu_detect 2>&1 | head -20`
Expected: compilation error — `cpu_detect.h` does not exist.

- [ ] **Step 4: Write the header `src/cpu_detect.h`**

```c
#ifndef BITNET_CPU_DETECT_H
#define BITNET_CPU_DETECT_H

#include <stdint.h>

typedef enum {
    BITNET_TIER_SCALAR      = 0,
    BITNET_TIER_AVX2        = 1,
    BITNET_TIER_AVX_VNNI    = 2,
    BITNET_TIER_AVX512_VNNI = 3,
} bitnet_cpu_tier_t;

/* Sentinel returned by bitnet_cpu_tier_from_string on unrecognized input. */
#define BITNET_TIER_INVALID ((bitnet_cpu_tier_t)-1)

typedef struct {
    int has_sse3;
    int has_avx2;
    int has_fma;
    int has_avx_vnni;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512vnni;
} bitnet_cpu_features_t;

/* Detect CPU features available with current OS+process context. */
bitnet_cpu_features_t bitnet_cpu_detect(void);

/* Pick the best tier the detected features support. */
bitnet_cpu_tier_t bitnet_cpu_pick_tier(void);

/* Stable human-readable name for a tier ("scalar", "avx2", "avx_vnni",
 * "avx512_vnni"). Returns NULL for invalid tiers. */
const char *bitnet_cpu_tier_name(bitnet_cpu_tier_t tier);

/* Parse a tier name string. Returns BITNET_TIER_INVALID if unrecognized. */
bitnet_cpu_tier_t bitnet_cpu_tier_from_string(const char *name);

#endif
```

- [ ] **Step 5: Write the implementation `src/cpu_detect.c`**

```c
#include "cpu_detect.h"
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
/* GCC/Clang — use the built-in CPU detection, which already handles the
 * OS-level XGETBV check that confirms YMM/ZMM state is saved. */
bitnet_cpu_features_t bitnet_cpu_detect(void) {
    __builtin_cpu_init();
    bitnet_cpu_features_t f;
    memset(&f, 0, sizeof(f));
    f.has_sse3        = __builtin_cpu_supports("sse3");
    f.has_avx2        = __builtin_cpu_supports("avx2");
    f.has_fma         = __builtin_cpu_supports("fma");
    f.has_avx_vnni    = __builtin_cpu_supports("avxvnni");
    f.has_avx512f     = __builtin_cpu_supports("avx512f");
    f.has_avx512bw    = __builtin_cpu_supports("avx512bw");
    f.has_avx512vnni  = __builtin_cpu_supports("avx512vnni");
    return f;
}
#elif defined(_MSC_VER)
#include <intrin.h>
bitnet_cpu_features_t bitnet_cpu_detect(void) {
    bitnet_cpu_features_t f;
    memset(&f, 0, sizeof(f));
    int cpuinfo[4];
    __cpuid(cpuinfo, 0);
    int max_leaf = cpuinfo[0];
    if (max_leaf < 1) return f;

    __cpuid(cpuinfo, 1);
    f.has_sse3 = (cpuinfo[2] & (1 << 0)) != 0;       /* ECX bit 0 */
    int osxmm_ymm = ((_xgetbv(_XCR_XFEATURE_ENABLED_MASK) & 0x6) == 0x6);

    if (max_leaf >= 7) {
        __cpuidex(cpuinfo, 7, 0);
        f.has_avx2 = osxmm_ymm && (cpuinfo[1] & (1 << 5));    /* EBX bit 5 */
        f.has_avx512f    = osxmm_ymm && (cpuinfo[1] & (1 << 16));
        f.has_avx512bw   = osxmm_ymm && (cpuinfo[1] & (1 << 30));
        f.has_avx512vnni = osxmm_ymm && (cpuinfo[2] & (1 << 11)); /* ECX bit 11 */

        if (max_leaf >= 7) {
            __cpuidex(cpuinfo, 7, 1);
            f.has_avx_vnni = osxmm_ymm && (cpuinfo[0] & (1 << 4)); /* EAX bit 4 */
        }
    }
    /* FMA — leaf 1 ECX bit 12. */
    __cpuid(cpuinfo, 1);
    f.has_fma = osxmm_ymm && (cpuinfo[2] & (1 << 12));
    return f;
}
#else
bitnet_cpu_features_t bitnet_cpu_detect(void) {
    bitnet_cpu_features_t f;
    memset(&f, 0, sizeof(f));
    return f;
}
#endif

bitnet_cpu_tier_t bitnet_cpu_pick_tier(void) {
#if defined(__x86_64__) || defined(_M_X64)
    bitnet_cpu_features_t f = bitnet_cpu_detect();
    if (f.has_avx512f && f.has_avx512bw && f.has_avx512vnni) {
        return BITNET_TIER_AVX512_VNNI;
    }
    if (f.has_avx2 && f.has_avx_vnni && f.has_fma) {
        return BITNET_TIER_AVX_VNNI;
    }
    if (f.has_avx2 && f.has_fma) {
        return BITNET_TIER_AVX2;
    }
    return BITNET_TIER_SCALAR;
#else
    return BITNET_TIER_SCALAR;
#endif
}

const char *bitnet_cpu_tier_name(bitnet_cpu_tier_t tier) {
    switch (tier) {
        case BITNET_TIER_SCALAR:      return "scalar";
        case BITNET_TIER_AVX2:        return "avx2";
        case BITNET_TIER_AVX_VNNI:    return "avx_vnni";
        case BITNET_TIER_AVX512_VNNI: return "avx512_vnni";
        default: return NULL;
    }
}

bitnet_cpu_tier_t bitnet_cpu_tier_from_string(const char *name) {
    if (name == NULL) return BITNET_TIER_INVALID;
    if (strcmp(name, "scalar") == 0)      return BITNET_TIER_SCALAR;
    if (strcmp(name, "avx2") == 0)        return BITNET_TIER_AVX2;
    if (strcmp(name, "avx_vnni") == 0)    return BITNET_TIER_AVX_VNNI;
    if (strcmp(name, "avx512_vnni") == 0) return BITNET_TIER_AVX512_VNNI;
    return BITNET_TIER_INVALID;
}
```

- [ ] **Step 6: Run test to verify it passes**

Run:
```bash
cmake -S . -B build
cmake --build build --target test_cpu_detect
./build/test_cpu_detect
```
Expected output: `test_cpu_detect: OK (tier=...)`

- [ ] **Step 7: Commit**

```bash
git add src/cpu_detect.h src/cpu_detect.c tests/test_cpu_detect.c CMakeLists.txt
git commit -m "feat: add cross-platform CPU detection module"
```

### Task 1.2: Create the dispatch table skeleton

**Files:**
- Create: `src/bitnet_dispatch.h`
- Create: `src/bitnet_dispatch.c`

**Interfaces:**
- Produces: `bitnet_dispatch_t` struct (filled incrementally across phases),
  `g_bitnet_dispatch` global pointer, `bitnet_dispatch_init()`.

- [ ] **Step 1: Write the header `src/bitnet_dispatch.h`**

```c
#ifndef BITNET_DISPATCH_H
#define BITNET_DISPATCH_H

#include "cpu_detect.h"

/* Function-pointer table. Each public kernel in the runtime has one slot.
 * Slots are added incrementally as phases bring kernels online; entries
 * that haven't been implemented yet are NULL until their phase lands.
 *
 * Convention: scalar-tier implementation is always populated first; ARM
 * builds resolve the same table to the existing NEON implementations
 * directly via #ifdef in bitnet_dispatch.c. */
typedef struct bitnet_dispatch {
    bitnet_cpu_tier_t tier;

    /* ops.c kernels (Phase 2) */
    void (*rms_norm_eps)(float *x, const float *weight, int n, float eps);
    void (*rms_norm_inplace_eps)(float *dst, const float *src,
                                  const float *weight, int n, float eps);

    /* TQ2_0 kernels (Phase 3, 5) — populated incrementally */
    int  (*tq2_quantize_vec_i8)(const float *vec, int in_dim, int8_t *qvec,
                                 float *scale, int32_t *block_bsums);

    /* Q6K kernels (Phase 4) */

    /* Bitnet.c hot paths (Phase 4) */

    /* Padding reserved for future kernels — keeps the struct layout stable
     * across phases so callers don't need recompilation between phases. */
    void *_reserved[32];
} bitnet_dispatch_t;

extern bitnet_dispatch_t *g_bitnet_dispatch;

/* Resolve g_bitnet_dispatch to the right tier. Idempotent; runs the CPU
 * detection once under pthread_once. Reads BITNET_CPU_TIER override env
 * var (validated; falls back to auto-detect with a warning on bad input).
 * Reads BITNET_QUIET=1 to suppress the startup log line. */
void bitnet_dispatch_init(void);

#endif
```

- [ ] **Step 2: Write the implementation `src/bitnet_dispatch.c`**

```c
#include "bitnet_dispatch.h"
#include "cpu_detect.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Per-tier tables — defined in kernel_registry.c (Phase 1 registers only
 * the scalar tier; later phases fill in the SIMD tiers). */
extern bitnet_dispatch_t g_dispatch_scalar;
#if defined(__ARM_NEON)
extern bitnet_dispatch_t g_dispatch_arm_neon;
#endif

bitnet_dispatch_t *g_bitnet_dispatch = NULL;

static pthread_once_t g_dispatch_once = PTHREAD_ONCE_INIT;

static bitnet_cpu_tier_t resolve_tier_with_override(int *quiet) {
    *quiet = 0;
    const char *q = getenv("BITNET_QUIET");
    if (q && strcmp(q, "1") == 0) *quiet = 1;

    bitnet_cpu_tier_t auto_tier = bitnet_cpu_pick_tier();
    const char *override = getenv("BITNET_CPU_TIER");
    if (override == NULL || override[0] == '\0') return auto_tier;

    bitnet_cpu_tier_t requested = bitnet_cpu_tier_from_string(override);
    if (requested == BITNET_TIER_INVALID) {
        fprintf(stderr,
                "[bitnet] unrecognized BITNET_CPU_TIER='%s'; valid: "
                "scalar|avx2|avx_vnni|avx512_vnni. Falling back to auto-detect.\n",
                override);
        return auto_tier;
    }
    /* Safety: don't allow forcing a tier the CPU can't actually run. */
    bitnet_cpu_tier_t max_safe = auto_tier;
    if (requested > max_safe) {
        fprintf(stderr,
                "[bitnet] BITNET_CPU_TIER='%s' requested but CPU only supports "
                "tier '%s'. Falling back to auto-detect.\n",
                override, bitnet_cpu_tier_name(max_safe));
        return max_safe;
    }
    return requested;
}

static void dispatch_init_impl(void) {
    int quiet = 0;
    bitnet_cpu_tier_t tier = resolve_tier_with_override(&quiet);

#if defined(__ARM_NEON)
    /* ARM builds always use the NEON table regardless of the detected
     * "x86 tier" — the scalar tier is the only one available on ARM
     * from the bitnet_cpu_pick_tier perspective, but we override here. */
    g_bitnet_dispatch = &g_dispatch_arm_neon;
    if (!quiet) {
        fprintf(stderr, "[bitnet] cpu tier: arm_neon\n");
    }
    (void)tier;
    return;
#else
    switch (tier) {
        case BITNET_TIER_SCALAR:
            g_bitnet_dispatch = &g_dispatch_scalar;
            break;
        case BITNET_TIER_AVX2:
        case BITNET_TIER_AVX_VNNI:
        case BITNET_TIER_AVX512_VNNI:
            /* Phase 1: scalar only. Later phases populate these. Fall
             * through to scalar as a safe default for now. */
            g_bitnet_dispatch = &g_dispatch_scalar;
            break;
        default:
            g_bitnet_dispatch = &g_dispatch_scalar;
            break;
    }
    if (!quiet) {
        fprintf(stderr, "[bitnet] cpu tier: %s\n",
                bitnet_cpu_tier_name(g_bitnet_dispatch->tier));
    }
#endif
}

void bitnet_dispatch_init(void) {
    pthread_once(&g_dispatch_once, dispatch_init_impl);
}
```

- [ ] **Step 3: Commit (header + impl, no test yet — registry comes next)**

```bash
git add src/bitnet_dispatch.h src/bitnet_dispatch.c
git commit -m "feat: add dispatch table skeleton with tier override"
```

### Task 1.3: Create the scalar-tier kernel registry

**Files:**
- Create: `src/x86/kernel_registry.h`
- Create: `src/x86/kernel_registry.c`

**Interfaces:**
- Produces: `g_dispatch_scalar` and `g_dispatch_arm_neon` extern globals
  that `bitnet_dispatch.c` references.

- [ ] **Step 1: Write the header `src/x86/kernel_registry.h`**

```c
#ifndef BITNET_KERNEL_REGISTRY_H
#define BITNET_KERNEL_REGISTRY_H

#include "bitnet_dispatch.h"

extern bitnet_dispatch_t g_dispatch_scalar;
#if defined(__ARM_NEON)
extern bitnet_dispatch_t g_dispatch_arm_neon;
#endif

#endif
```

- [ ] **Step 2: Write the implementation `src/x86/kernel_registry.c`**

This is the scalar tier. For Phase 1, all entries are existing scalar
implementations from the codebase, reached via thin shims. Later phases
will replace these pointers with SIMD-variant tables for AVX2 / AVX-VNNI /
AVX512-VNNI.

```c
#include "kernel_registry.h"
#include "../ops.h"
#include "../quant_tq2_0.h"

/* The scalar rms_norm path. ops.c already has a non-NEON body for
 * bitnet_rms_norm_eps when __ARM_NEON is undefined — the function symbol
 * itself is the scalar path on x86. On ARM, the same name resolves to the
 * NEON path, which we'll keep using for g_dispatch_arm_neon. */

static void shim_rms_norm_eps(float *x, const float *weight, int n, float eps) {
    bitnet_rms_norm_eps(x, weight, n, eps);
}
static void shim_rms_norm_inplace_eps(float *dst, const float *src,
                                       const float *weight, int n, float eps) {
    bitnet_rms_norm_inplace_eps(dst, src, weight, n, eps);
}
static int shim_tq2_quantize_vec_i8(const float *vec, int in_dim, int8_t *qvec,
                                     float *scale, int32_t *block_bsums) {
    return bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, scale, block_bsums);
}

bitnet_dispatch_t g_dispatch_scalar = {
    .tier                     = BITNET_TIER_SCALAR,
    .rms_norm_eps             = shim_rms_norm_eps,
    .rms_norm_inplace_eps     = shim_rms_norm_inplace_eps,
    .tq2_quantize_vec_i8      = shim_tq2_quantize_vec_i8,
};

#if defined(__ARM_NEON)
/* On ARM, the same shims reach the existing NEON implementations because
 * ops.c / quant_tq2_0.c compile their NEON bodies under __ARM_NEON. */
bitnet_dispatch_t g_dispatch_arm_neon = {
    .tier                     = BITNET_TIER_AVX2, /* placeholder tag */
    .rms_norm_eps             = shim_rms_norm_eps,
    .rms_norm_inplace_eps     = shim_rms_norm_inplace_eps,
    .tq2_quantize_vec_i8      = shim_tq2_quantize_vec_i8,
};
#endif
```

- [ ] **Step 3: Commit**

```bash
git add src/x86/kernel_registry.h src/x86/kernel_registry.c
git commit -m "feat: add scalar-tier kernel registry"
```

### Task 1.4: Wire new sources into CMake for x86

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add architecture detection block**

After the existing `option(BITNET_ENABLE_METAL ...)` line (around line 4), and before the `if(APPLE)` block, add:

```cmake
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" BITNET_ARCH)
```

- [ ] **Step 2: Add the new sources after the existing add_library block**

Replace the existing `add_library(bitnet STATIC ...)` block (lines 49–59 in the current file) with:

```cmake
add_library(bitnet STATIC
    src/bitnet.c
    src/gguf.c
    src/tensor.c
    src/tokenizer.c
    src/quant_tq2_0.c
    src/quant_q4k.c
    src/quant_q6k.c
    src/ops.c
    src/sampler.c
)

# x86 dispatch + CPU detection. Always compiled (small surface), guarded
# inside the source by #if defined(__x86_64__) || defined(_M_X64).
if(BITNET_ARCH STREQUAL "x86_64" OR BITNET_ARCH STREQUAL "amd64")
    target_sources(bitnet PRIVATE
        src/cpu_detect.c
        src/bitnet_dispatch.c
        src/x86/kernel_registry.c)
    target_compile_definitions(bitnet PUBLIC BITNET_HAS_X86=1)
endif()
```

- [ ] **Step 3: Verify the build on the current host**

Run:
```bash
cmake -S . -B build
cmake --build build -j 8
```
Expected: clean build, no new warnings beyond the pre-existing baseline.

- [ ] **Step 4: Run the existing test suite to confirm no regressions**

Run: `ctest --test-dir build --output-on-failure`
Expected: every test passes (same as before this task).

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: wire cpu_detect + dispatch into x86 builds"
```

### Task 1.5: Verify dispatch initializes from the public API surface

**Files:**
- Test: `tests/test_dispatch_init.c`

This is a smoke test: calling a dispatch-routed function triggers init.

**Interfaces:**
- Consumes: `g_bitnet_dispatch` (must be non-NULL after first call).

- [ ] **Step 1: Write the failing test**

Create `tests/test_dispatch_init.c`:

```c
#include "bitnet_dispatch.h"
#include "ops.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    /* Before any dispatch-routed call, the pointer is NULL. */
    assert(g_bitnet_dispatch == NULL);

    /* Calling any kernel that goes through dispatch triggers init. For
     * Phase 1, ops.c functions are not yet trampolined (Phase 2), so we
     * call bitnet_dispatch_init() explicitly here. */
    bitnet_dispatch_init();

    assert(g_bitnet_dispatch != NULL);
    assert(g_bitnet_dispatch->rms_norm_eps != NULL);
    assert(g_bitnet_dispatch->tq2_quantize_vec_i8 != NULL);

    printf("test_dispatch_init: OK (tier=%d)\n", g_bitnet_dispatch->tier);
    return 0;
}
```

- [ ] **Step 2: Add test to CMakeLists**

In `CMakeLists.txt`, after the `test_cpu_detect` block, add:

```cmake
add_executable(test_dispatch_init tests/test_dispatch_init.c)
target_include_directories(test_dispatch_init PRIVATE src)
target_link_libraries(test_dispatch_init PRIVATE bitnet)
add_test(NAME test_dispatch_init COMMAND test_dispatch_init)
```

- [ ] **Step 3: Run test to verify it passes**

Run:
```bash
cmake --build build --target test_dispatch_init
./build/test_dispatch_init
```
Expected: `test_dispatch_init: OK (tier=0)` (scalar on x86 hosts before SIMD phases land).

- [ ] **Step 4: Commit**

```bash
git add tests/test_dispatch_init.c CMakeLists.txt
git commit -m "test: verify dispatch initializes on first call"
```

### Task 1.6: Add the CI matrix

**Files:**
- Create: `.github/workflows/ci-x86.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: ci-x86

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  linux-x86-gcc:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: cmake -S . -B build -DCMAKE_C_COMPILER=gcc-11
      - run: cmake --build build -j 4
      - run: ctest --test-dir build --output-on-failure

  linux-x86-clang:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: cmake -S . -B build -DCMAKE_C_COMPILER=clang-14
      - run: cmake --build build -j 4
      - run: ctest --test-dir build --output-on-failure

  linux-x86-newer:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: cmake -S . -B build
      - run: cmake --build build -j 4
      - run: ctest --test-dir build --output-on-failure

  macos-intel:
    runs-on: macos-13
    steps:
      - uses: actions/checkout@v4
      - run: cmake -S . -B build
      - run: cmake --build build -j 4
      - run: ctest --test-dir build --output-on-failure

  windows-msvc:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - uses: ilammy/msvc-dev-cmd@v1
        with: { arch: x64 }
      - run: cmake -S . -B build -G "Ninja" -DCMAKE_C_COMPILER=cl
      - run: cmake --build build -j 4
      - run: ctest --test-dir build --output-on-failure

  qemu-avx512:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y qemu-user
      - run: cmake -S . -B build -DCMAKE_C_COMPILER=gcc-11
      - run: cmake --build build -j 4
      - run: |
          for t in test_cpu_detect test_ops test_quant_tq2_0 test_q6k_layout; do
            qemu-x86_64 -cpu max ./build/$t || exit 1
          done
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci-x86.yml
git commit -m "ci: add x86 build matrix (linux/macos/windows/qemu-avx512)"
```

### Phase 1 Exit Gate

Before starting Phase 2:

- [ ] `ctest --test-dir build --output-on-failure` passes on host.
- [ ] `BITNET_CPU_TIER=avx2 ./build/test_cpu_detect` runs (still reports scalar in Phase 1, that's fine — auto-detect wins because the override is gated by max_safe).
- [ ] `[bitnet] cpu tier: scalar` line appears on stderr when running any test binary (or is suppressed under `BITNET_QUIET=1`).
- [ ] ARM build still produces byte-identical `.a` (compare with `cmp` against a pre-Phase-1 build artifact).

---

## Phase 2 — ops.c Parity (RMSNorm, softmax, activation)

Goal: route the `ops.c` NEON paths through the dispatch table and provide AVX2 / AVX-VNNI / AVX512-VNNI implementations. These are the simplest kernels; immediate, visible speedup.

### Task 2.1: Convert `bitnet_rms_norm_eps` to a trampoline

**Files:**
- Modify: `src/ops.c` (replace the public function definition with a trampoline; rename the existing NEON/scalar body to `bitnet_rms_norm_eps_impl`)
- Modify: `src/x86/kernel_registry.c` (point both registries at the impl directly)

**Interfaces:**
- Consumes: `g_bitnet_dispatch->rms_norm_eps` (set in Phase 1).
- Produces: `bitnet_rms_norm_eps_impl` internal symbol that the registry calls.

- [ ] **Step 1: Rename the existing function bodies**

In `src/ops.c`, find every definition of `bitnet_rms_norm_eps` and `bitnet_rms_norm_inplace_eps` (under both `#if defined(__ARM_NEON)` and `#else`), and rename them to `bitnet_rms_norm_eps_impl` and `bitnet_rms_norm_inplace_eps_impl`. Leave the bodies exactly as they are — the `#if defined(__ARM_NEON)` / `#else` compile-time selection still picks the right one.

At the top of `ops.c`, declare them as non-static:

```c
void bitnet_rms_norm_eps_impl(float *x, const float *weight, int n, float eps);
void bitnet_rms_norm_inplace_eps_impl(float *dst, const float *src,
                                       const float *weight, int n, float eps);
```

(Place these above the existing definitions so forward-declaration works.)

- [ ] **Step 2: Replace the public functions with trampolines**

At the bottom of `src/ops.c`, add:

```c
#include "bitnet_dispatch.h"

void bitnet_rms_norm_eps(float *x, const float *weight, int n, float eps) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    g_bitnet_dispatch->rms_norm_eps(x, weight, n, eps);
}

void bitnet_rms_norm_inplace_eps(float *dst, const float *src,
                                  const float *weight, int n, float eps) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    g_bitnet_dispatch->rms_norm_inplace_eps(dst, src, weight, n, eps);
}
```

- [ ] **Step 3: Update the scalar registry to call the impl**

In `src/x86/kernel_registry.c`, update the shims to call the renamed impl:

```c
static void shim_rms_norm_eps(float *x, const float *weight, int n, float eps) {
    bitnet_rms_norm_eps_impl(x, weight, n, eps);
}
static void shim_rms_norm_inplace_eps(float *dst, const float *src,
                                       const float *weight, int n, float eps) {
    bitnet_rms_norm_inplace_eps_impl(dst, src, weight, n, eps);
}
```

- [ ] **Step 4: Run ops tests to verify no regression**

Run:
```bash
cmake --build build --target test_ops
./build/test_ops
```
Expected: PASS (same as before — dispatch routes to scalar impl on x86, NEON impl on ARM).

- [ ] **Step 5: Commit**

```bash
git add src/ops.c src/x86/kernel_registry.c
git commit -m "refactor: route rms_norm through dispatch table"
```

### Task 2.2: Implement AVX2 RMSNorm

**Files:**
- Create: `src/x86/ops_x86.h`
- Create: `src/x86/ops_x86.c`
- Modify: `src/x86/kernel_registry.c` (add `g_dispatch_avx2`, `g_dispatch_avx_vnni`, `g_dispatch_avx512_vnni` tables — partial fill, just `rms_norm_eps` for now)
- Modify: `src/bitnet_dispatch.c` (switch on detected tier to pick the right table)
- Test: `tests/test_ops.c` (extend with vector-length sweep)

- [ ] **Step 1: Write the header**

Create `src/x86/ops_x86.h`:

```c
#ifndef BITNET_OPS_X86_H
#define BITNET_OPS_X86_H

void bitnet_rms_norm_eps_avx2(float *x, const float *weight, int n, float eps);
void bitnet_rms_norm_eps_avx_vnni(float *x, const float *weight, int n, float eps);
void bitnet_rms_norm_eps_avx512_vnni(float *x, const float *weight, int n, float eps);

void bitnet_rms_norm_inplace_eps_avx2(float *dst, const float *src,
                                       const float *weight, int n, float eps);
void bitnet_rms_norm_inplace_eps_avx_vnni(float *dst, const float *src,
                                           const float *weight, int n, float eps);
void bitnet_rms_norm_inplace_eps_avx512_vnni(float *dst, const float *src,
                                              const float *weight, int n, float eps);
#endif
```

- [ ] **Step 2: Write the AVX2 implementation**

Create `src/x86/ops_x86.c`:

```c
#include "ops_x86.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define BITNET_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define BITNET_TARGET_AVX_VNNI __attribute__((target("avx2,fma,avxvnni")))
#define BITNET_TARGET_AVX512_VNNI __attribute__((target("avx512f,avx512bw,avx512vnni")))
#else
/* MSVC compiles whole files with one /arch: flag, gated by BITNET_X86_TIER
 * via kernel_registry.c per-tier compilation. */
#define BITNET_TARGET_AVX2
#define BITNET_TARGET_AVX_VNNI
#define BITNET_TARGET_AVX512_VNNI
#endif

BITNET_TARGET_AVX2
void bitnet_rms_norm_eps_avx2(float *x, const float *weight, int n, float eps) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
    }
    /* horizontal sum */
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) sum += x[i] * x[i];

    float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);

    __m256 inv_v = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        v = _mm256_mul_ps(v, inv_v);
        v = _mm256_mul_ps(v, w);
        _mm256_storeu_ps(x + i, v);
    }
    for (; i < n; ++i) x[i] = x[i] * inv_rms * weight[i];
}

BITNET_TARGET_AVX2
void bitnet_rms_norm_inplace_eps_avx2(float *dst, const float *src,
                                       const float *weight, int n, float eps) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
    }
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) sum += src[i] * src[i];

    float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);
    __m256 inv_v = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        v = _mm256_mul_ps(v, inv_v);
        v = _mm256_mul_ps(v, w);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) dst[i] = src[i] * inv_rms * weight[i];
}

/* For Phase 2, the VNNI and AVX512 tiers reuse the AVX2 RMSNorm (no
 * integer dot product here, so VNNI doesn't help). Phase 3+ kernels
 * will diverge. */
BITNET_TARGET_AVX_VNNI
void bitnet_rms_norm_eps_avx_vnni(float *x, const float *weight, int n, float eps) {
    bitnet_rms_norm_eps_avx2(x, weight, n, eps);
}

BITNET_TARGET_AVX_VNNI
void bitnet_rms_norm_inplace_eps_avx_vnni(float *dst, const float *src,
                                           const float *weight, int n, float eps) {
    bitnet_rms_norm_inplace_eps_avx2(dst, src, weight, n, eps);
}

BITNET_TARGET_AVX512_VNNI
void bitnet_rms_norm_eps_avx512_vnni(float *x, const float *weight, int n, float eps) {
    /* AVX512 wider lanes: 16 floats per iteration. */
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        sum_vec = _mm512_fmadd_ps(v, v, sum_vec);
    }
    float sum = _mm512_reduce_add_ps(sum_vec);
    for (; i < n; ++i) sum += x[i] * x[i];

    float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);
    __m512 inv_v = _mm512_set1_ps(inv_rms);
    i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 w = _mm512_loadu_ps(weight + i);
        v = _mm512_mul_ps(v, inv_v);
        v = _mm512_mul_ps(v, w);
        _mm512_storeu_ps(x + i, v);
    }
    for (; i < n; ++i) x[i] = x[i] * inv_rms * weight[i];
}

BITNET_TARGET_AVX512_VNNI
void bitnet_rms_norm_inplace_eps_avx512_vnni(float *dst, const float *src,
                                              const float *weight, int n, float eps) {
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        sum_vec = _mm512_fmadd_ps(v, v, sum_vec);
    }
    float sum = _mm512_reduce_add_ps(sum_vec);
    for (; i < n; ++i) sum += src[i] * src[i];

    float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);
    __m512 inv_v = _mm512_set1_ps(inv_rms);
    i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 w = _mm512_loadu_ps(weight + i);
        v = _mm512_mul_ps(v, inv_v);
        v = _mm512_mul_ps(v, w);
        _mm512_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) dst[i] = src[i] * inv_rms * weight[i];
}
```

- [ ] **Step 3: Extend `test_ops.c` with a length-sweep correctness test**

In `tests/test_ops.c`, add (or extend main with):

```c
static void test_rms_norm_lengths(void) {
    /* Verify RMSNorm at every length we expect to encounter. Hidden dim
     * is typically 256..4096 for these models. */
    int lengths[] = {1, 4, 7, 8, 16, 31, 32, 100, 256, 1023, 1024, 4096};
    for (size_t i = 0; i < sizeof(lengths)/sizeof(lengths[0]); ++i) {
        int n = lengths[i];
        float *x = malloc(n * sizeof(float));
        float *w = malloc(n * sizeof(float));
        for (int j = 0; j < n; ++j) { x[j] = (float)(j % 7) - 3.0f; w[j] = 1.0f; }
        bitnet_rms_norm_eps(x, w, n, 1e-6f);
        /* Sanity: result norms to ~1.0 after weighting by all-ones. */
        float sum = 0;
        for (int j = 0; j < n; ++j) sum += x[j] * x[j];
        float rms = sqrtf(sum / n);
        assert(fabsf(rms - 1.0f) < 0.01f);
        free(x); free(w);
    }
}
```

Add `#include <math.h>` and `#include <stdlib.h>` at the top of the test if not present.

- [ ] **Step 4: Run test on AVX2 host**

Run:
```bash
cmake --build build --target test_ops
./build/test_ops
```
Expected: PASS.

- [ ] **Step 5: Run the same test under each tier override**

```bash
BITNET_CPU_TIER=scalar      ./build/test_ops
BITNET_CPU_TIER=avx2        ./build/test_ops
BITNET_CPU_TIER=avx_vnni    ./build/test_ops    # only on AVX-VNNI hosts
BITNET_CPU_TIER=avx512_vnni ./build/test_ops    # only on AVX512-VNNI hosts
```
Expected: PASS in all four (subject to host CPU support).

- [ ] **Step 6: Commit**

```bash
git add src/x86/ops_x86.h src/x86/ops_x86.c src/x86/kernel_registry.c src/bitnet_dispatch.c tests/test_ops.c
git commit -m "feat: AVX2/VNNI/AVX512 rms_norm kernels"
```

### Phase 2 Exit Gate

- [ ] All ops tests pass at every tier override.
- [ ] Softmax, silu, and other ops.c NEON blocks: each routed through dispatch with one trampoline + 3 SIMD variants, following the exact pattern in Task 2.2. (Apply the pattern iteratively; each one is one commit.)

---

## Phase 3 — TQ2_0 Baseline Kernels (AVX2)

Goal: AVX2 variants of the TQ2_0 hot-path kernels, reusing ARM's existing 4-row I2S packing. This is the largest single phase — ~2000 lines of new intrinsic code.

### Task 3.1: Convert `bitnet_tq2_0_quantize_vec_i8` to a trampoline + add AVX2 variant

**Files:**
- Modify: `src/quant_tq2_0.c` (rename existing body to `_impl`, add trampoline)
- Create: `src/x86/quant_tq2_0_x86.h`
- Create: `src/x86/quant_tq2_0_x86.c`
- Modify: `src/x86/kernel_registry.c`
- Test: `tests/test_quant_tq2_0.c` (already exists — verify extension coverage)

- [ ] **Step 1: Rename existing `bitnet_tq2_0_quantize_vec_i8` to `_impl`**

In `src/quant_tq2_0.c`, rename every definition (NEON + scalar) of `bitnet_tq2_0_quantize_vec_i8` to `bitnet_tq2_0_quantize_vec_i8_impl`. Add a forward declaration near the top:

```c
int bitnet_tq2_0_quantize_vec_i8_impl(const float *vec, int in_dim, int8_t *qvec,
                                       float *scale, int32_t *block_bsums);
```

At the bottom of `quant_tq2_0.c`, add:

```c
#include "bitnet_dispatch.h"
int bitnet_tq2_0_quantize_vec_i8(const float *vec, int in_dim, int8_t *qvec,
                                  float *scale, int32_t *block_bsums) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    return g_bitnet_dispatch->tq2_quantize_vec_i8(vec, in_dim, qvec, scale, block_bsums);
}
```

- [ ] **Step 2: Write the header `src/x86/quant_tq2_0_x86.h`**

```c
#ifndef BITNET_QUANT_TQ2_0_X86_H
#define BITNET_QUANT_TQ2_0_X86_H

#include <stdint.h>

int bitnet_tq2_0_quantize_vec_i8_avx2(const float *vec, int in_dim, int8_t *qvec,
                                       float *scale, int32_t *block_bsums);
int bitnet_tq2_0_quantize_vec_i8_avx_vnni(const float *vec, int in_dim, int8_t *qvec,
                                            float *scale, int32_t *block_bsums);
int bitnet_tq2_0_quantize_vec_i8_avx512_vnni(const float *vec, int in_dim, int8_t *qvec,
                                                float *scale, int32_t *block_bsums);

#endif
```

- [ ] **Step 3: Write the AVX2 implementation in `src/x86/quant_tq2_0_x86.c`**

Implement using `_mm256_loadu_ps` for max-abs scan, `_mm256_cvtps_epi32` + `_mm256_packs_epi32` (with saturation) for quantize. For Phase 3 Task 3.1, only the AVX2 path is needed; VNNI/AVX512 paths delegate to AVX2 (will be specialized in later tasks).

```c
#include "quant_tq2_0_x86.h"
#include "../quant_tq2_0.h"

#include <immintrin.h>
#include <math.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define BITNET_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define BITNET_TARGET_AVX_VNNI __attribute__((target("avx2,fma,avxvnni")))
#define BITNET_TARGET_AVX512_VNNI __attribute__((target("avx512f,avx512bw,avx512vnni")))
#else
#define BITNET_TARGET_AVX2
#define BITNET_TARGET_AVX_VNNI
#define BITNET_TARGET_AVX512_VNNI
#endif

#define BITNET_TQ2_0_QK 256

BITNET_TARGET_AVX2
int bitnet_tq2_0_quantize_vec_i8_avx2(const float *vec, int in_dim, int8_t *qvec,
                                       float *scale, int32_t *block_bsums) {
    if (vec == NULL || qvec == NULL || scale == NULL || in_dim <= 0) return -1;

    /* Max-abs scan with AVX2. */
    __m256 max_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < in_dim; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v); /* abs */
        max_vec = _mm256_max_ps(max_vec, a);
    }
    /* Reduce max_vec horizontally. */
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_abs = _mm_cvtss_f32(m);
    for (; i < in_dim; ++i) {
        float a = fabsf(vec[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs <= 0.0f) {
        memset(qvec, 0, (size_t)in_dim * sizeof(*qvec));
        *scale = 1.0f;
        if (block_bsums != NULL) {
            int n_blocks = in_dim / BITNET_TQ2_0_QK;
            memset(block_bsums, 0, (size_t)n_blocks * sizeof(*block_bsums));
        }
        return 0;
    }

    *scale = max_abs / 127.0f;
    const float inv_scale = 127.0f / max_abs;
    const __m256 inv_v = _mm256_set1_ps(inv_scale);

    /* Quantize with saturation. */
    i = 0;
    for (; i + 31 < in_dim; i += 32) {
        /* Load 32 floats, scale, convert to int32 (8 lanes × 4), pack
         * to int8 via two _mm256_packs_epi32 chains. */
        __m256i out0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(vec + i +  0), inv_v));
        __m256i out1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(vec + i +  8), inv_v));
        __m256i out2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(vec + i + 16), inv_v));
        __m256i out3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(vec + i + 24), inv_v));
        /* Pack int32 → int16 (signed saturation), then int16 → int8. */
        __m256i p0 = _mm256_packs_epi32(out0, out1);  /* int16, 16 lanes */
        __m256i p1 = _mm256_packs_epi32(out2, out3);
        __m256i q  = _mm256_packs_epi16(p0, p1);      /* int8, 32 lanes */
        /* _mm256_packs_epi16 produces 128-bit-laid-out output; permute
         * to linear order. */
        __m256i perm = _mm256_permute4x64_epi64(q, _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_storeu_si256((__m256i *)(qvec + i), perm);
    }
    for (; i < in_dim; ++i) {
        int v = (int)lroundf(vec[i] * inv_scale);
        if (v > 127) v = 127; else if (v < -128) v = -128;
        qvec[i] = (int8_t)v;
    }

    /* Per-block sums (used by VNNI bsums-subtract trick). */
    if (block_bsums != NULL) {
        int n_blocks = in_dim / BITNET_TQ2_0_QK;
        for (int b = 0; b < n_blocks; ++b) {
            int32_t s = 0;
            for (int j = 0; j < BITNET_TQ2_0_QK; ++j) {
                s += (int32_t)qvec[b * BITNET_TQ2_0_QK + j];
            }
            block_bsums[b] = s;
        }
    }
    return 0;
}

BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_quantize_vec_i8_avx_vnni(const float *vec, int in_dim, int8_t *qvec,
                                            float *scale, int32_t *block_bsums) {
    return bitnet_tq2_0_quantize_vec_i8_avx2(vec, in_dim, qvec, scale, block_bsums);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_quantize_vec_i8_avx512_vnni(const float *vec, int in_dim, int8_t *qvec,
                                                float *scale, int32_t *block_bsums) {
    /* Phase 3 Task 3.1: reuse AVX2. Phase 7 may specialize for 512-bit. */
    return bitnet_tq2_0_quantize_vec_i8_avx2(vec, in_dim, qvec, scale, block_bsums);
}
```

- [ ] **Step 4: Wire into the registry**

In `src/x86/kernel_registry.c`, add per-tier tables and populate the `tq2_quantize_vec_i8` slot:

```c
#include "../quant_tq2_0.h"
#include "quant_tq2_0_x86.h"
#include "ops_x86.h"

bitnet_dispatch_t g_dispatch_avx2 = {
    .tier                 = BITNET_TIER_AVX2,
    .rms_norm_eps         = bitnet_rms_norm_eps_avx2,
    .rms_norm_inplace_eps = bitnet_rms_norm_inplace_eps_avx2,
    .tq2_quantize_vec_i8  = bitnet_tq2_0_quantize_vec_i8_avx2,
};

bitnet_dispatch_t g_dispatch_avx_vnni = {
    .tier                 = BITNET_TIER_AVX_VNNI,
    .rms_norm_eps         = bitnet_rms_norm_eps_avx_vnni,
    .rms_norm_inplace_eps = bitnet_rms_norm_inplace_eps_avx_vnni,
    .tq2_quantize_vec_i8  = bitnet_tq2_0_quantize_vec_i8_avx_vnni,
};

bitnet_dispatch_t g_dispatch_avx512_vnni = {
    .tier                 = BITNET_TIER_AVX512_VNNI,
    .rms_norm_eps         = bitnet_rms_norm_eps_avx512_vnni,
    .rms_norm_inplace_eps = bitnet_rms_norm_inplace_eps_avx512_vnni,
    .tq2_quantize_vec_i8  = bitnet_tq2_0_quantize_vec_i8_avx512_vnni,
};
```

- [ ] **Step 5: Update dispatch.c to pick the SIMD tables**

In `src/bitnet_dispatch.c`, replace the Phase-1 placeholder `switch (tier)` body with:

```c
extern bitnet_dispatch_t g_dispatch_avx2;
extern bitnet_dispatch_t g_dispatch_avx_vnni;
extern bitnet_dispatch_t g_dispatch_avx512_vnni;

switch (tier) {
    case BITNET_TIER_SCALAR:      g_bitnet_dispatch = &g_dispatch_scalar;      break;
    case BITNET_TIER_AVX2:        g_bitnet_dispatch = &g_dispatch_avx2;        break;
    case BITNET_TIER_AVX_VNNI:    g_bitnet_dispatch = &g_dispatch_avx_vnni;    break;
    case BITNET_TIER_AVX512_VNNI: g_bitnet_dispatch = &g_dispatch_avx512_vnni; break;
    default:                      g_bitnet_dispatch = &g_dispatch_scalar;      break;
}
```

(Place these `extern` declarations at the top of the file, alongside the existing `g_dispatch_scalar` / `g_dispatch_arm_neon` ones.)

- [ ] **Step 6: Add quant_tq2_0_x86.c to CMake**

In `CMakeLists.txt`, in the x86 `target_sources` block, add `src/x86/quant_tq2_0_x86.c` and `src/x86/ops_x86.c` to the list.

- [ ] **Step 7: Run TQ2_0 tests**

Run:
```bash
cmake --build build --target test_quant_tq2_0
./build/test_quant_tq2_0
```
Expected: PASS at every tier override.

- [ ] **Step 8: Commit**

```bash
git add src/quant_tq2_0.c src/x86/quant_tq2_0_x86.h src/x86/quant_tq2_0_x86.c src/x86/ops_x86.c src/x86/ops_x86.h src/x86/kernel_registry.c src/bitnet_dispatch.c CMakeLists.txt
git commit -m "feat: AVX2 quantize_vec_i8 + dispatch wiring"
```

### Task 3.2: AVX2 TQ2_0 matmul_vector_lut (single + pair)

Apply the same pattern to `bitnet_tq2_0_matmul_vector_lut`, `bitnet_tq2_0_matmul_vector_lut_scales`, `bitnet_tq2_0_matmul_vector_lut_pair`, `bitnet_tq2_0_matmul_vector_lut_pair_scales`. Each becomes a trampoline; AVX2 variants use `_mm_shuffle_epi8` for the LUT lookup (direct analogue of `vqtbl1q_s8`) and `_mm256_maddubs_epi16` + `_mm256_madd_epi16` for accumulation.

**Files:**
- Modify: `src/quant_tq2_0.c` (4 trampolines + renames)
- Modify: `src/x86/quant_tq2_0_x86.c` (4 new AVX2 functions)
- Modify: `src/bitnet_dispatch.h` (4 new slots)
- Modify: `src/x86/kernel_registry.c` (populate 4 slots in each tier)
- Test: `tests/test_quant_tq2_0.c` (already covers all 4 functions — verify tier override)

- [ ] **Step 1: Add dispatch slots**

In `src/bitnet_dispatch.h`, add 4 slots to the struct (after `tq2_quantize_vec_i8`):

```c
int (*tq2_matmul_vector_lut)(const void *weight, int out_dim, int in_dim,
                              const float *lut, float *out);
int (*tq2_matmul_vector_lut_scales)(const void *weight, const float *scales,
                                     int out_dim, int in_dim,
                                     const float *lut, float *out);
int (*tq2_matmul_vector_lut_pair)(const void *weight_a, const void *weight_b,
                                   int out_dim, int in_dim,
                                   const float *lut, float *out_a, float *out_b);
int (*tq2_matmul_vector_lut_pair_scales)(const void *weight_a, const float *scales_a,
                                          const void *weight_b, const float *scales_b,
                                          int out_dim, int in_dim,
                                          const float *lut, float *out_a, float *out_b);
```

- [ ] **Step 2: Add trampolines in `src/quant_tq2_0.c`**

Rename each existing definition to `_impl`, then add 4 trampolines at the bottom of the file in the same pattern as Task 3.1.

- [ ] **Step 3: Implement AVX2 variants in `src/x86/quant_tq2_0_x86.c`**

For each function, the implementation follows this skeleton (single-lut shown):

```c
BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_vector_lut_avx2(const void *weight, int out_dim, int in_dim,
                                         const float *lut, float *out) {
    /* For each row: walk 64-byte TQ2_0 blocks. Per block, load 16 packed
     * bytes, _mm_shuffle_epi8 looks up the LUT (which has 4 floats per
     * byte position). Accumulate via _mm256_fmadd_ps. Apply block scale
     * at the end. */
    /* ... full implementation per spec Section 4 mapping ... */
}
```

Refer to `src/quant_tq2_0.c` lines 264–520 (the existing NEON LUT path) as the reference for the algorithm. The x86 port replaces `vqtbl1q_s8` with `_mm_shuffle_epi8` and `vdotq_s32` with the 2-step emulation.

- [ ] **Step 4: Populate registry slots**

In each `g_dispatch_<tier>` table in `src/x86/kernel_registry.c`, add the new function pointers (use the `_avx2` variant for AVX2 tier, `_avx2` aliased for VNNI/AVX512 in this task).

- [ ] **Step 5: Run test**

Run: `cmake --build build --target test_quant_tq2_0 && ./build/test_quant_tq2_0`
Expected: PASS at every tier override.

- [ ] **Step 6: Commit**

```bash
git add src/quant_tq2_0.c src/bitnet_dispatch.h src/x86/quant_tq2_0_x86.c src/x86/kernel_registry.c
git commit -m "feat: AVX2 TQ2_0 LUT matmul kernels"
```

### Task 3.3: AVX2 I2S matmul path (`i2s_neon_parallel`, `i2s_neon_pair_parallel`, `i2s_qkv_parallel`)

The I2S path is the actual decode hot path. Apply the trampoline+AVX2 pattern to the three parallel I2S functions. They reuse ARM's existing 4-row I2S packing — wider-packing variants come in Phase 6.

**Files:**
- Modify: `src/quant_tq2_0.c` (3 trampolines + renames)
- Modify: `src/x86/quant_tq2_0_x86.c` (3 new AVX2 functions, each consuming the existing 4-row I2S layout)
- Modify: `src/bitnet_dispatch.h` (3 new slots)
- Modify: `src/x86/kernel_registry.c` (populate 3 slots in each tier)
- Test: `tests/test_i2s_correctness.c` (already exercises these — verify tier override)

The AVX2 I2S kernel replaces the ARM `vdotq_s32` accumulator with `_mm256_maddubs_epi16` + `_mm256_madd_epi16`:

```c
__m128i lo = _mm256_castsi256_si128(weights);          /* low 16 bytes */
__m128i a   = _mm256_castsi256_si128(activations);
__m128i t   = _mm_maddubs_epi16(a, lo);                /* uint8×int8→int16, 16 lanes */
__m128i s   = _mm_madd_epi16(t, _mm_set1_epi16(1));    /* int16×int16→int32, 8 lanes */
acc = _mm_add_epi32(acc, s);
```

Refer to `src/quant_tq2_0.c` lines 1419–1900 (I2S NEON reference) as the algorithm source. Each step iterates 4-row blocks; the AVX2 version processes 8 lanes per `_mm_maddubs` instead of NEON's 16 — so unroll 2x to match throughput.

- [ ] **Steps 1–6**: same pattern as Task 3.2; one commit at the end.

```bash
git commit -m "feat: AVX2 TQ2_0 I2S matmul kernels (4-row packing)"
```

### Task 3.4: Phase 3 Exit Gate

- [ ] `cmake --build build -j 8 && ctest --test-dir build --output-on-failure` passes.
- [ ] `./build/test_i2s_correctness` passes at every tier override.
- [ ] Benchmark: `BITNET_CPU_TIER=scalar ./build/test_profile_decode` vs `BITNET_CPU_TIER=avx2 ./build/test_profile_decode` shows measurable speedup (target: 2x+ over scalar on AVX2 host).

---

## Phase 4 — Q6K Output Projection + bitnet.c Hot Paths

### Task 4.1: AVX2 + AVX-VNNI Q6K kernels

Apply the trampoline pattern to every Q6K function that has a `#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)` body in `src/quant_q6k.c`. Implement AVX2 (with the 2-step dotprod emulation) and AVX-VNNI (using `_mm256_dpbusd_epi32` directly — true VNNI tier available from Phase 4 because Q6K output projection is where VNNI pays off most).

**Files:**
- Modify: `src/quant_q6k.c` (trampolines + renames)
- Create: `src/x86/quant_q6k_x86.h`
- Create: `src/x86/quant_q6k_x86.c`
- Modify: `src/bitnet_dispatch.h` (new Q6K slots)
- Modify: `src/x86/kernel_registry.c` (populate Q6K slots)
- Test: `tests/test_q6k_layout.c` (already exists)

- [ ] **Step 1**: Create `src/x86/quant_q6k_x86.h` declaring AVX2 + AVX-VNNI variants of every public Q6K function.
- [ ] **Step 2**: Implement in `src/x86/quant_q6k_x86.c`. For AVX-VNNI tier, the inner kernel becomes:

```c
BITNET_TARGET_AVX_VNNI
static inline __m256i dot_acc_avx_vnni(__m256i acc, const __m256i a, const __m256i b) {
    return _mm256_dpbusd_epi32(acc, a, b);  /* uint8 × int8 → int32, 32 lanes */
}
```

For AVX2 tier, emulate via `_mm256_maddubs_epi16` + `_mm256_madd_epi16`.

- [ ] **Step 3**: Rename existing `bitnet_q6k_*` definitions to `_impl`, add trampolines.
- [ ] **Step 4**: Wire into registry. AVX2 tier uses AVX2 variants; AVX-VNNI tier uses AVX-VNNI variants; AVX512-VNNI tier delegates to AVX-VNNI for now (Phase 7 specializes).
- [ ] **Step 5**: Add to CMake; run `test_q6k_layout` at each tier.
- [ ] **Step 6**: Commit `feat: AVX2 + AVX-VNNI Q6K kernels`.

### Task 4.2: bitnet.c hot paths (attention quantize, KV ops)

Apply the trampoline pattern to every `#if defined(__ARM_NEON)` block in `src/bitnet.c` (lines 237, 753, 850, 867, 895, 915, 934, 955, 1280, 1380, 3629, 3667). These are small helper functions — RMSNorm-style quantize, attention score accumulation, KV cache writes.

**Files:**
- Modify: `src/bitnet.c` (trampolines + renames for each NEON block)
- Create: `src/x86/bitnet_hotpath_x86.h`
- Create: `src/x86/bitnet_hotpath_x86.c`
- Modify: `src/bitnet_dispatch.h` (new slots as needed)
- Modify: `src/x86/kernel_registry.c` (populate)

For each NEON block:
- Rename the existing function to `_impl` (keeps the `#ifdef __ARM_NEON` selection intact).
- Add a trampoline at the bottom of bitnet.c that goes through dispatch.
- Implement an AVX2 variant in `bitnet_hotpath_x86.c` using the intrinsic mapping in spec Section 4.

- [ ] **Step 1**: Inventory the NEON blocks in `src/bitnet.c` and create one trampoline + AVX2 variant per block.
- [ ] **Step 2**: Run full ctest suite at every tier override.
- [ ] **Step 3**: Commit `feat: AVX2 bitnet.c hot paths (attention quantize, KV ops)`.

### Task 4.3: Phase 4 Exit Gate

- [ ] All tests pass at every tier.
- [ ] Benchmark: `test_profile_decode` shows AVX-VNNI beating AVX2 on VNNI-capable hosts (target: 1.5x+ on Q6K-heavy 0.5B/1B models).

---

## Phase 5 — AVX-VNNI Tier for TQ2_0

### Task 5.1: Replace AVX2 emulation with `_mm256_dpbusd_epi32` in TQ2_0 kernels

For each TQ2_0 kernel added in Phase 3, add a true AVX-VNNI variant that uses `_mm256_dpbusd_epi32` instead of `_mm256_maddubs_epi16 + _mm256_madd_epi16`.

**Files:**
- Modify: `src/x86/quant_tq2_0_x86.c` (add `_avx_vnni` variants)
- Modify: `src/x86/kernel_registry.c` (point `g_dispatch_avx_vnni` at the new variants instead of the AVX2 aliases)

The VNNI inner kernel:

```c
BITNET_TARGET_AVX_VNNI
static inline __m256i dot_acc_vnni(__m256i acc, const __m256i act_u8, const __m256i w_i8) {
    return _mm256_dpbusd_epi32(acc, act_u8, w_i8);
}
```

Note: activations must be cast to `uint8` (the bsums trick ensures `{0,1,2}` non-negative values), weights stay `int8`. The block-end bsums subtraction is unchanged from AVX2.

- [ ] **Step 1**: Add `_avx_vnni` variants for every TQ2_0 kernel.
- [ ] **Step 2**: Update `g_dispatch_avx_vnni` to point at them.
- [ ] **Step 3**: Run full ctest suite at `BITNET_CPU_TIER=avx_vnni`.
- [ ] **Step 4**: Commit `feat: AVX-VNNI TQ2_0 kernels (true dotprod)`.

### Task 5.2: Phase 5 Exit Gate

- [ ] `test_i2s_correctness` and `test_quant_tq2_0` pass at `BITNET_CPU_TIER=avx_vnni`.
- [ ] Benchmark: VNNI tier shows 1.5–2x over AVX2 tier on TQ2_0-heavy workloads.

---

## Phase 6 — Wider Packing (I2S-X8, I2S-X16)

### Task 6.1: Implement I2S-X8 reorder + AVX2/AVX-VNNI kernels

**Files:**
- Create: `src/x86/pack_x86.h`
- Create: `src/x86/pack_x86.c`
- Modify: `src/bitnet.c` (model-load path calls `pick_pack_format` instead of hardcoded `reorder_to_i2s`)
- Modify: `src/bitnet_dispatch.h` (add `pick_tq2_pack_format` slot + `tq2_matmul_i2s_x8_parallel` slots)
- Modify: `src/x86/kernel_registry.c` (populate new slots)
- Test: `tests/test_i2s_correctness.c` (add X8 variant)

- [ ] **Step 1**: Add pack format enum and X8 size/reorder functions to `src/x86/pack_x86.h` and `.c`:

```c
typedef enum {
    BITNET_TQ2_PACK_TQ2_0,
    BITNET_TQ2_PACK_I2S_ARM,
    BITNET_TQ2_PACK_I2S_X8,
    BITNET_TQ2_PACK_I2S_X16,
} bitnet_tq2_pack_format_t;

size_t bitnet_tq2_0_i2s_x8_packed_size(int out_dim, int in_dim);
int bitnet_tq2_0_reorder_to_i2s_x8(const void *weight, int out_dim, int in_dim,
                                    uint8_t *packed, float *packed_scales,
                                    int32_t *packed_bsums);
```

The X8 reorder groups 8 rows together, interleaves their 64-byte qs blocks for SIMD-friendly access, and writes 32-byte-aligned scales + bsums.

- [ ] **Step 2**: Add X8-consuming kernels `bitnet_tq2_0_matmul_i2s_x8_avx2` and `bitnet_tq2_0_matmul_i2s_x8_avx_vnni` in `src/x86/quant_tq2_0_x86.c`. Each processes 8 rows per iteration, loading one activation vector per 4-lane group and reusing it across all 8 rows.
- [ ] **Step 3**: Update the model-load path in `src/bitnet.c` to call `g_bitnet_dispatch->pick_tq2_pack_format()` and dispatch to the matching reorder function. Store the chosen format on the model.
- [ ] **Step 4**: Update decode path: if model has X8 format, call the X8 kernel; if I2S_ARM format, call the 4-row kernel from Phase 3. (Each model carries its format through to decode.)
- [ ] **Step 5**: Extend `test_i2s_correctness.c` to validate X8-pack + decode.
- [ ] **Step 6**: Commit `feat: I2S-X8 8-row packing for AVX2/AVX-VNNI`.

### Task 6.2: Implement I2S-X16 reorder + AVX512-VNNI kernels

Same pattern, 16 rows per group, 64-byte alignment, consumes `_mm512_dpbusd_epi32`.

**Files:**
- Modify: `src/x86/pack_x86.{h,c}` (add X16 variants)
- Modify: `src/x86/quant_tq2_0_x86.c` (add `_avx512_vnni` X16-consuming kernels)
- Test: `tests/test_i2s_correctness.c` (extend with X16 variant)

- [ ] **Steps 1–5**: parallel to Task 6.1.
- [ ] **Step 6**: Commit `feat: I2S-X16 16-row packing for AVX512-VNNI`.

### Task 6.3: Phase 6 Exit Gate

- [ ] All tests pass on every (tier × pack format) combination that the tier supports.
- [ ] Benchmark: wider packing shows 10–20% decode speedup over Phase 5 narrow packing at the same tier.

---

## Phase 7 — AVX512-VNNI Tier (512-bit kernels, 4-accumulator ILP)

### Task 7.1: Implement AVX512-VNNI for every kernel still aliased to lower tiers

By end of Phase 6, AVX512-VNNI tier has 512-bit RMSNorm (Phase 2) and 512-bit X16 I2S matmul (Phase 6) but most other kernels still delegate to AVX-VNNI. This phase specializes them.

**Files:**
- Modify: `src/x86/ops_x86.c` (softmax, silu — wider 512-bit paths)
- Modify: `src/x86/quant_tq2_0_x86.c` (every kernel that still aliases AVX-VNNI in AVX512 tier)
- Modify: `src/x86/quant_q6k_x86.c` (Q6K 512-bit kernels)
- Modify: `src/x86/bitnet_hotpath_x86.c` (512-bit hot paths)
- Modify: `src/x86/kernel_registry.c` (point AVX512 tier at the new variants)

For each, follow this template:

```c
BITNET_TARGET_AVX512_VNNI
static inline __m512i dot_acc_512(__m512i acc, const __m512i a, const __m512i b) {
    return _mm512_dpbusd_epi32(acc, a, b);  /* 64 int8 lanes per instruction */
}
```

Use 4 accumulators (instead of 2) since AVX512 ports are fewer — breaks the dependency chain further.

- [ ] **Step 1**: Add AVX512-VNNI variants for every kernel.
- [ ] **Step 2**: Update `g_dispatch_avx512_vnni` to point at them.
- [ ] **Step 3**: Run full ctest under `qemu-x86_64 -cpu max` to exercise AVX512 paths without AVX512 hardware.

```bash
qemu-x86_64 -cpu max ./build/test_quant_tq2_0
qemu-x86_64 -cpu max ./build/test_q6k_layout
qemu-x86_64 -cpu max ./build/test_i2s_correctness
```

- [ ] **Step 4**: Commit `feat: AVX512-VNNI specialized kernels with 4-accumulator ILP`.

### Task 7.2: Final integration test — full decode at every tier

**Files:**
- Test: `tests/test_x86_tier_bench.c` (new — comprehensive tier comparison)

- [ ] **Step 1**: Write `tests/test_x86_tier_bench.c` that:
  - Loads a small model (0.5B).
  - Runs `bitnet_decode` 50 tokens at each tier override.
  - Prints a table: tier × tok/s × correctness-check.

```c
#include "bitnet.h"
#include "cpu_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    const char *model_path = argv[1];
    const char *tiers[] = {"scalar", "avx2", "avx_vnni", "avx512_vnni"};
    bitnet_cpu_tier_t max_tier = bitnet_cpu_pick_tier();

    for (int i = 0; i < 4; ++i) {
        if (bitnet_cpu_tier_from_string(tiers[i]) > max_tier) {
            printf("%-15s SKIPPED (host CPU can't run)\n", tiers[i]);
            continue;
        }
        setenv("BITNET_CPU_TIER", tiers[i], 1);
        /* Load model fresh per tier — pack format depends on tier. */
        bitnet_model_t *m = bitnet_model_load(model_path, NULL);
        if (!m) { fprintf(stderr, "load failed for tier %s\n", tiers[i]); continue; }
        bitnet_context_t *c = bitnet_context_create(m, BITNET_NUM_THREADS_DEFAULT);
        /* Prefill a short prompt and decode 50 tokens; measure wall-clock. */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const char *out = bitnet_generate(c, "The capital of France is", 50);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("%-15s %.2fs  %.1f tok/s  output=\"%s\"\n",
               tiers[i], elapsed, 50.0 / elapsed, out);
        bitnet_context_free(c);
        bitnet_model_free(m);
    }
    return 0;
}
```

(Adjust the bitnet.h API names to match what's actually declared in `include/bitnet.h`.)

- [ ] **Step 2**: Add to CMake and run on host:

```bash
cmake --build build --target test_x86_tier_bench
./build/test_x86_tier_bench models/bitcpm4-0.5b-tq2_0.gguf
```

Expected: tier progression shows clear speedup (scalar << AVX2 < AVX-VNNI < AVX512-VNNI where supported), output text identical across tiers.

- [ ] **Step 3**: Commit `test: cross-tier benchmark + smoke correctness`.

### Task 7.3: Update README with x86 build/run instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1**: Add a section between "## Architecture" and "## Important Optimizations" describing the x86 tier system, the `BITNET_CPU_TIER` env var, the startup log line, and the CI matrix.
- [ ] **Step 2**: Update the "## Build" section to mention that the build auto-detects x86 vs ARM.
- [ ] **Step 3**: Commit `docs: document x86 tier dispatch`.

### Task 7.4: Phase 7 / Final Exit Gate

- [ ] `ctest --test-dir build --output-on-failure` passes on every CI job.
- [ ] QEMU-AVX512 job passes (`qemu-x86_64 -cpu max` exercises every tier).
- [ ] `test_x86_tier_bench` shows monotonic speedup across tiers on a reference model.
- [ ] ARM build still produces byte-identical `.a` to pre-Phase-1 baseline.
- [ ] README accurately describes the new behavior.

---

## Self-Review Notes

**Spec coverage**: every section of the spec maps to at least one task.
- Spec §1 (Architecture) → Phase 1 (Tasks 1.1–1.6).
- Spec §2 (CPU Detection) → Task 1.1.
- Spec §3 (Per-tier Packing) → Phase 6 (Tasks 6.1–6.2); runtime selection in Task 6.1 Step 3.
- Spec §4 (Kernel Mapping) → applied throughout Phases 2–7.
- Spec §5 (Build System & CI) → Tasks 1.4, 1.6; MSVC OBJECT libraries covered in CMake.
- Spec §6 (Phasing) → phases 1–7 in this plan.
- Spec estimated effort (~4000 LOC, 7 PRs) → matches phase count.

**Placeholder scan**: each "Apply the pattern" instruction in Phases 3–7 references a fully-worked example task (Task 3.1 or 4.1) and the specific spec section / line range in `src/quant_tq2_0.c` for the algorithm. The plan avoids "TODO" and "TBD".

**Type consistency**: `bitnet_dispatch_t` slots are added incrementally; later tasks populate slots declared in earlier tasks. Function-name convention is `<original>_avx2 / _avx_vnni / _avx512_vnni` consistently. Trampoline renames are consistently `<original>_impl`.

**MSVC object-library tier compilation** (spec §5): the plan uses GCC/Clang per-function target attributes by default. MSVC's three OBJECT libraries per tier require per-file `BITNET_X86_TIER` macro + symbol suffixing — this is called out in spec §4 ("Per-tier file organization") and Phase 1's `kernel_registry.c` structure accommodates it. Implementation note for the MSVC build step: add a separate CMake block in Phase 2 Task 2.2 (or a focused Phase 1 follow-up) that creates the three OBJECT libraries when `MSVC` is true, each compiling `src/x86/*.c` with a different `BITNET_X86_TIER` value and `/arch:` flag. The `kernel_registry.c` source must `#include` a tier-suffix macro header so symbols don't collide. (Fleshed out as the first task of Phase 2 implementation if MSVC issues surface in CI.)
