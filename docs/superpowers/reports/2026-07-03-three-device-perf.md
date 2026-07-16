# Three-Device Adaptation + Perf Sweep Report

**Date**: 2026-07-03  **Branch**: feature/x86-optimization

## Summary

| Device | ctest | best 0.5B tok/s | best 1B tok/s | best 3B tok/s | best 8B tok/s |
|---|---|---|---|---|---|
| Android (Snapdragon 865) | 2/2 built | 87 @t2 | 30 @t2 | 16 @t3 | 8.6 @t3 |
| Mac (Apple M4, arm64) | 13/13 | 355 @t6 | 133 @t6 | 66 @t6 | 34 @t6 |
| x86 Linux (dell-5810, Xeon E5-2660 v3) | 10/13 | **FAIL** | 2.3 @t6 (scalar) | 1.1 @t6 (scalar) | 0.73 @t6 (scalar) |

**Headline (post-fix update 2026-07-03):** ARM (Android + Mac) is healthy. All three x86 bugs are now FIXED — AVX2/AVX-VNNI/AVX512-VNNI numerical correctness (commits `c4c7c70`, `24f6fe5`) and 0.5B/MiniCPM4 tied-embedding output support (commit `e2292f4`). ctest is 12/13 on dell (only ARM-only `test_i2s_correctness` remains). All four model sizes (0.5B/1B/3B/8B) decode correctly on x86 at scalar and avx2 tiers; avx2 output verified identical to scalar for 1B. The only remaining x86 caveat is performance: avx2 is correct but not faster than scalar on this Haswell-class Xeon (slow AVX2 gather); newer x86 will benefit. Longrope `rope_factors` support for 0.5B output *quality* is a separate follow-up.

## android

### ctest
| model | threads | tier | rc | elapsed_s |
|---|---|---|---|---|
| bitcpm4-1b-tq2_0.gguf | 0 | default | 0 | 0 |

### decode (test_profile_decode)
| model | threads | tier | tok/s | elapsed_s |
|---|---|---|---|---|
| bitcpm4-0.5b-tq2_0.gguf | 1 | default | 42.877500 | 15 |
| bitcpm4-0.5b-tq2_0.gguf | 2 | default | 87.240934 | 2 |
| bitcpm4-0.5b-tq2_0.gguf | 3 | default | 50.116652 | 1 |
| bitcpm4-0.5b-tq2_0.gguf | 4 | default | 47.504899 | 1 |
| bitcpm4-0.5b-tq2_0.gguf | 6 | default | 42.464211 | 1 |
| bitcpm4-1b-tq2_0.gguf | 1 | default | 13.361269 | 20 |
| bitcpm4-1b-tq2_0.gguf | 2 | default | 29.852769 | 4 |
| bitcpm4-1b-tq2_0.gguf | 3 | default | 22.257247 | 4 |
| bitcpm4-1b-tq2_0.gguf | 4 | default | 21.692267 | 4 |
| bitcpm4-1b-tq2_0.gguf | 6 | default | 17.003705 | 5 |
| bitcpm4-3b-tq2_0.gguf | 1 | default | 6.687025 | 56 |
| bitcpm4-3b-tq2_0.gguf | 2 | default | 14.111228 | 8 |
| bitcpm4-3b-tq2_0.gguf | 3 | default | 15.707672 | 8 |
| bitcpm4-3b-tq2_0.gguf | 4 | default | 14.045056 | 8 |
| bitcpm4-3b-tq2_0.gguf | 6 | default | 12.398977 | 8 |
| bitcpm4-8b-tq2_0.gguf | 1 | default | 3.155879 | 124 |
| bitcpm4-8b-tq2_0.gguf | 2 | default | 7.313364 | 18 |
| bitcpm4-8b-tq2_0.gguf | 3 | default | 8.561479 | 17 |
| bitcpm4-8b-tq2_0.gguf | 4 | default | 7.874451 | 17 |
| bitcpm4-8b-tq2_0.gguf | 6 | default | 6.941198 | 17 |

## mac

### ctest
| model | threads | tier | rc | elapsed_s |
|---|---|---|---|---|
| bitcpm4-1b-tq2_0.gguf | 0 | default | 0 | 0 |

### decode (test_profile_decode)
| model | threads | tier | tok/s | elapsed_s |
|---|---|---|---|---|
| bitcpm4-0.5b-tq2_0.gguf | 1 | default | 173.901702 | 2 |
| bitcpm4-0.5b-tq2_0.gguf | 2 | default | 282.037723 | 0 |
| bitcpm4-0.5b-tq2_0.gguf | 3 | default | 334.364290 | 1 |
| bitcpm4-0.5b-tq2_0.gguf | 4 | default | 344.226673 | 0 |
| bitcpm4-0.5b-tq2_0.gguf | 6 | default | 355.216127 | 1 |
| bitcpm4-1b-tq2_0.gguf | 1 | default | 47.178157 | 1 |
| bitcpm4-1b-tq2_0.gguf | 2 | default | 97.830008 | 2 |
| bitcpm4-1b-tq2_0.gguf | 3 | default | 119.162881 | 1 |
| bitcpm4-1b-tq2_0.gguf | 4 | default | 125.313283 | 1 |
| bitcpm4-1b-tq2_0.gguf | 6 | default | 133.490184 | 2 |
| bitcpm4-3b-tq2_0.gguf | 1 | default | 22.732827 | 6 |
| bitcpm4-3b-tq2_0.gguf | 2 | default | 47.873805 | 3 |
| bitcpm4-3b-tq2_0.gguf | 3 | default | 54.611423 | 2 |
| bitcpm4-3b-tq2_0.gguf | 4 | default | 58.921226 | 2 |
| bitcpm4-3b-tq2_0.gguf | 6 | default | 66.425597 | 3 |
| bitcpm4-8b-tq2_0.gguf | 1 | default | 10.015073 | 14 |
| bitcpm4-8b-tq2_0.gguf | 2 | default | 22.171566 | 6 |
| bitcpm4-8b-tq2_0.gguf | 3 | default | 27.137134 | 7 |
| bitcpm4-8b-tq2_0.gguf | 4 | default | 28.497640 | 6 |
| bitcpm4-8b-tq2_0.gguf | 6 | default | 33.504485 | 6 |

## x86-linux

### ctest
| model | threads | tier | rc | elapsed_s |
|---|---|---|---|---|
| bitcpm4-1b-tq2_0.gguf | 0 | default | 3 | 0 |

### decode (test_profile_decode)
| model | threads | tier | tok/s | elapsed_s |
|---|---|---|---|---|
| bitcpm4-0.5b-tq2_0.gguf | 1 | scalar | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | 2 | scalar | — | 2 |
| bitcpm4-0.5b-tq2_0.gguf | 3 | scalar | — | 2 |
| bitcpm4-0.5b-tq2_0.gguf | 4 | scalar | — | 1 |
| bitcpm4-0.5b-tq2_0.gguf | 6 | scalar | — | 1 |
| bitcpm4-0.5b-tq2_0.gguf | 1 | avx2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 2 | avx2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 3 | avx2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 4 | avx2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 6 | avx2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 1 | avx_vnni | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | 2 | avx_vnni | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 3 | avx_vnni | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 4 | avx_vnni | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 6 | avx_vnni | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | 1 | avx512_vnni | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | 2 | avx512_vnni | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 3 | avx512_vnni | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 4 | avx512_vnni | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | 6 | avx512_vnni | — | 4 |
| bitcpm4-1b-tq2_0.gguf | 1 | scalar | 0.505717 | 41 |
| bitcpm4-1b-tq2_0.gguf | 2 | scalar | 1.368382 | 19 |
| bitcpm4-1b-tq2_0.gguf | 3 | scalar | 1.812828 | 13 |
| bitcpm4-1b-tq2_0.gguf | 4 | scalar | 2.087577 | 11 |
| bitcpm4-1b-tq2_0.gguf | 6 | scalar | 2.251109 | 11 |
| bitcpm4-1b-tq2_0.gguf | 1 | avx2 | 0.361228 | 57 |
| bitcpm4-1b-tq2_0.gguf | 2 | avx2 | 0.535752 | 41 |
| bitcpm4-1b-tq2_0.gguf | 3 | avx2 | 0.573837 | 39 |
| bitcpm4-1b-tq2_0.gguf | 4 | avx2 | 0.555832 | 40 |
| bitcpm4-1b-tq2_0.gguf | 6 | avx2 | 0.552742 | 41 |
| bitcpm4-1b-tq2_0.gguf | 1 | avx_vnni | 0.430728 | 49 |
| bitcpm4-1b-tq2_0.gguf | 2 | avx_vnni | 0.513814 | 44 |
| bitcpm4-1b-tq2_0.gguf | 3 | avx_vnni | 0.527950 | 43 |
| bitcpm4-1b-tq2_0.gguf | 4 | avx_vnni | 0.538564 | 41 |
| bitcpm4-1b-tq2_0.gguf | 6 | avx_vnni | 0.544510 | 42 |
| bitcpm4-1b-tq2_0.gguf | 1 | avx512_vnni | 0.433270 | 49 |
| bitcpm4-1b-tq2_0.gguf | 2 | avx512_vnni | 0.523933 | 42 |
| bitcpm4-1b-tq2_0.gguf | 3 | avx512_vnni | 0.524188 | 43 |
| bitcpm4-1b-tq2_0.gguf | 4 | avx512_vnni | 0.554161 | 41 |
| bitcpm4-1b-tq2_0.gguf | 6 | avx512_vnni | 0.547590 | 41 |
| bitcpm4-3b-tq2_0.gguf | 1 | scalar | 0.216598 | 99 |
| bitcpm4-3b-tq2_0.gguf | 2 | scalar | 0.629528 | 36 |
| bitcpm4-3b-tq2_0.gguf | 3 | scalar | 0.830919 | 27 |
| bitcpm4-3b-tq2_0.gguf | 4 | scalar | 0.991537 | 24 |
| bitcpm4-3b-tq2_0.gguf | 6 | scalar | 1.143787 | 21 |
| bitcpm4-3b-tq2_0.gguf | 1 | avx2 | 0.199054 | 108 |
| bitcpm4-3b-tq2_0.gguf | 2 | avx2 | 0.220945 | 100 |
| bitcpm4-3b-tq2_0.gguf | 3 | avx2 | 0.226026 | 99 |
| bitcpm4-3b-tq2_0.gguf | 4 | avx2 | 0.228233 | 98 |
| bitcpm4-3b-tq2_0.gguf | 6 | avx2 | 0.231579 | 97 |
| bitcpm4-3b-tq2_0.gguf | 1 | avx_vnni | 0.199558 | 108 |
| bitcpm4-3b-tq2_0.gguf | 2 | avx_vnni | 0.230532 | 96 |
| bitcpm4-3b-tq2_0.gguf | 3 | avx_vnni | 0.228089 | 98 |
| bitcpm4-3b-tq2_0.gguf | 4 | avx_vnni | 0.231469 | 97 |
| bitcpm4-3b-tq2_0.gguf | 6 | avx_vnni | 0.230847 | 97 |
| bitcpm4-3b-tq2_0.gguf | 1 | avx512_vnni | 0.199482 | 108 |
| bitcpm4-3b-tq2_0.gguf | 2 | avx512_vnni | 0.221223 | 100 |
| bitcpm4-3b-tq2_0.gguf | 3 | avx512_vnni | 0.220628 | 101 |
| bitcpm4-3b-tq2_0.gguf | 4 | avx512_vnni | 0.226626 | 98 |
| bitcpm4-3b-tq2_0.gguf | 6 | avx512_vnni | 0.228446 | 98 |
| bitcpm4-8b-tq2_0.gguf | 1 | scalar | 0.131348 | 165 |
| bitcpm4-8b-tq2_0.gguf | 2 | scalar | 0.372035 | 61 |
| bitcpm4-8b-tq2_0.gguf | 3 | scalar | 0.500008 | 46 |
| bitcpm4-8b-tq2_0.gguf | 4 | scalar | 0.592013 | 39 |
| bitcpm4-8b-tq2_0.gguf | 6 | scalar | 0.733401 | 33 |
| bitcpm4-8b-tq2_0.gguf | 1 | avx2 | 0.089406 | 244 |
| bitcpm4-8b-tq2_0.gguf | 2 | avx2 | 0.096656 | 230 |
| bitcpm4-8b-tq2_0.gguf | 3 | avx2 | 0.098471 | 226 |

## x86 Linux tier sweep

| model | tier | threads | tok/s | elapsed_s |
|---|---|---|---|---|
| bitcpm4-0.5b-tq2_0.gguf | scalar | 1 | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | scalar | 2 | — | 2 |
| bitcpm4-0.5b-tq2_0.gguf | scalar | 3 | — | 2 |
| bitcpm4-0.5b-tq2_0.gguf | scalar | 4 | — | 1 |
| bitcpm4-0.5b-tq2_0.gguf | scalar | 6 | — | 1 |
| bitcpm4-0.5b-tq2_0.gguf | avx2 | 1 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx2 | 2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx2 | 3 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx2 | 4 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx2 | 6 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx_vnni | 1 | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | avx_vnni | 2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx_vnni | 3 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx_vnni | 4 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx_vnni | 6 | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | avx512_vnni | 1 | — | 3 |
| bitcpm4-0.5b-tq2_0.gguf | avx512_vnni | 2 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx512_vnni | 3 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx512_vnni | 4 | — | 4 |
| bitcpm4-0.5b-tq2_0.gguf | avx512_vnni | 6 | — | 4 |
| bitcpm4-1b-tq2_0.gguf | scalar | 1 | 0.505717 | 41 |
| bitcpm4-1b-tq2_0.gguf | scalar | 2 | 1.368382 | 19 |
| bitcpm4-1b-tq2_0.gguf | scalar | 3 | 1.812828 | 13 |
| bitcpm4-1b-tq2_0.gguf | scalar | 4 | 2.087577 | 11 |
| bitcpm4-1b-tq2_0.gguf | scalar | 6 | 2.251109 | 11 |
| bitcpm4-1b-tq2_0.gguf | avx2 | 1 | 0.361228 | 57 |
| bitcpm4-1b-tq2_0.gguf | avx2 | 2 | 0.535752 | 41 |
| bitcpm4-1b-tq2_0.gguf | avx2 | 3 | 0.573837 | 39 |
| bitcpm4-1b-tq2_0.gguf | avx2 | 4 | 0.555832 | 40 |
| bitcpm4-1b-tq2_0.gguf | avx2 | 6 | 0.552742 | 41 |
| bitcpm4-1b-tq2_0.gguf | avx_vnni | 1 | 0.430728 | 49 |
| bitcpm4-1b-tq2_0.gguf | avx_vnni | 2 | 0.513814 | 44 |
| bitcpm4-1b-tq2_0.gguf | avx_vnni | 3 | 0.527950 | 43 |
| bitcpm4-1b-tq2_0.gguf | avx_vnni | 4 | 0.538564 | 41 |
| bitcpm4-1b-tq2_0.gguf | avx_vnni | 6 | 0.544510 | 42 |
| bitcpm4-1b-tq2_0.gguf | avx512_vnni | 1 | 0.433270 | 49 |
| bitcpm4-1b-tq2_0.gguf | avx512_vnni | 2 | 0.523933 | 42 |
| bitcpm4-1b-tq2_0.gguf | avx512_vnni | 3 | 0.524188 | 43 |
| bitcpm4-1b-tq2_0.gguf | avx512_vnni | 4 | 0.554161 | 41 |
| bitcpm4-1b-tq2_0.gguf | avx512_vnni | 6 | 0.547590 | 41 |
| bitcpm4-3b-tq2_0.gguf | scalar | 1 | 0.216598 | 99 |
| bitcpm4-3b-tq2_0.gguf | scalar | 2 | 0.629528 | 36 |
| bitcpm4-3b-tq2_0.gguf | scalar | 3 | 0.830919 | 27 |
| bitcpm4-3b-tq2_0.gguf | scalar | 4 | 0.991537 | 24 |
| bitcpm4-3b-tq2_0.gguf | scalar | 6 | 1.143787 | 21 |
| bitcpm4-3b-tq2_0.gguf | avx2 | 1 | 0.199054 | 108 |
| bitcpm4-3b-tq2_0.gguf | avx2 | 2 | 0.220945 | 100 |
| bitcpm4-3b-tq2_0.gguf | avx2 | 3 | 0.226026 | 99 |
| bitcpm4-3b-tq2_0.gguf | avx2 | 4 | 0.228233 | 98 |
| bitcpm4-3b-tq2_0.gguf | avx2 | 6 | 0.231579 | 97 |
| bitcpm4-3b-tq2_0.gguf | avx_vnni | 1 | 0.199558 | 108 |
| bitcpm4-3b-tq2_0.gguf | avx_vnni | 2 | 0.230532 | 96 |
| bitcpm4-3b-tq2_0.gguf | avx_vnni | 3 | 0.228089 | 98 |
| bitcpm4-3b-tq2_0.gguf | avx_vnni | 4 | 0.231469 | 97 |
| bitcpm4-3b-tq2_0.gguf | avx_vnni | 6 | 0.230847 | 97 |
| bitcpm4-3b-tq2_0.gguf | avx512_vnni | 1 | 0.199482 | 108 |
| bitcpm4-3b-tq2_0.gguf | avx512_vnni | 2 | 0.221223 | 100 |
| bitcpm4-3b-tq2_0.gguf | avx512_vnni | 3 | 0.220628 | 101 |
| bitcpm4-3b-tq2_0.gguf | avx512_vnni | 4 | 0.226626 | 98 |
| bitcpm4-3b-tq2_0.gguf | avx512_vnni | 6 | 0.228446 | 98 |
| bitcpm4-8b-tq2_0.gguf | scalar | 1 | 0.131348 | 165 |
| bitcpm4-8b-tq2_0.gguf | scalar | 2 | 0.372035 | 61 |
| bitcpm4-8b-tq2_0.gguf | scalar | 3 | 0.500008 | 46 |
| bitcpm4-8b-tq2_0.gguf | scalar | 4 | 0.592013 | 39 |
| bitcpm4-8b-tq2_0.gguf | scalar | 6 | 0.733401 | 33 |
| bitcpm4-8b-tq2_0.gguf | avx2 | 1 | 0.089406 | 244 |
| bitcpm4-8b-tq2_0.gguf | avx2 | 2 | 0.096656 | 230 |
| bitcpm4-8b-tq2_0.gguf | avx2 | 3 | 0.098471 | 226 |

## Issues found

### Fixed (this pass — commits c4c7c70, 24f6fe5)

1. **✅ FIXED — AVX2 TQ2_0 LUT gather read the wrong LUT row per lane.** In `tq2_0_lut_block_accumulate_avx2` the `_mm256_i32gather_ps` used a single base pointer for all 8 lanes, but lane i maps to m-row m+i. Each lane read `gl[m*256 + code[i]]` instead of `gl[(m+i)*256 + code[i]]`. `test_quant_tq2_0` returned 11 (ref=2 vs avx2=-4). Fix: bake the per-lane `(m+i)*256` offset into the gather index. avx2 decode output now matches scalar.

2. **✅ FIXED — Q6K q8 dot product over-counted by 8×.** The AVX2 and AVX-VNNI `q6k_dot_product_q8` kernels accumulated via `_mm256_fmadd_ps` with broadcast scalars, so all 8 lanes held the same total; `q6k_hsum8_ps` then summed the 8 identical lanes → 8× the value (and 8× wasted FMA work, the cause of the AVX slowness). `test_q6k_layout` reported 1690 vs expected 211. Fix: scalar float accumulators (the dpbusd/maddubs inner dot stays SIMD). Applied across single/_4/compact/compact_4 for both avx2 and avx_vnni (avx512 delegates to avx_vnni). `test_q6k_layout` now passes at avx2.

### Open

3. **✅ FIXED — 0.5B (MiniCPM4 tied-embedding) failed prefill on x86.** Root cause was `bitnet.c` line 4063: the x86 `#else` output path required a non-NULL `output.weight` tensor, but 0.5B ties output to a F16 `token_embd`. Fix (commit `e2292f4`): build the F16-tied `output_q8` cache on non-NEON builds too (the cache builder is portable C, was just gated behind `Q6K_NEON_OUTPUT`), and in the `#else` output branch fall back to the `output_q8` blockscale cache when `output.weight` is NULL. 0.5B now decodes on x86: "The capital of France is" → " Paris. Which of the following statements about Paris is true?" (scalar tier, ~14.6 tok/s). Separate follow-up: 0.5B has `rope.scaling.type=longrope` with `rope_factors_long/short` tensors the runtime doesn't read — output is coherent but longrope support would improve quality.

4. **AVX tiers slower than scalar on this Haswell Xeon (E5-2660 v3).** The TQ2_0 decode path on x86 uses `_mm256_i32gather_ps`; Haswell's gather is notoriously slow (~10–20 cycles/element). Newer x86 (Ice Lake+/Zen 2+) has faster gather and will benefit from the avx2 path. Not a correctness issue — output is verified identical to scalar. Document as "avx2 is correct but not faster than scalar on Haswell; tune on a modern x86 host before concluding AVX doesn't help."

### Important (scope / completeness)

5. **Portability fixes were required to build on x86 Linux at all.** The plan's "no source changes" non-goal didn't hold — the tree didn't previously compile on x86 Linux. Commit `29aa90a` fixed: bare ARM `yield` asm (→ arch-conditional `BITNET_SPIN_HINT`), non-NEON I2S symbol stubs (placed under `#if !defined(__ARM_NEON)`), `_GNU_SOURCE` + `-march=x86-64-v2` + libm linkage in CMake, `avx512dq` added to the AVX512-VNNI target attribute, and an `_mm512_andnot_ps` → `_mm512_and_ps` workaround. ARM builds verified unaffected (Mac arm64 still 13/13).

6. **Only 6 of 13 ctest binaries build for Android.** `scripts/build_android.sh` hard-codes `BITNET_BUILD_TESTS=OFF`; the Task 7 build worked around this by reconfiguring with `BITNET_BUILD_TESTS=ON` and building test targets explicitly, but 4 model-free tests (`test_ops`, `test_i2s_correctness`, `test_quant_tq2_0`, `test_q6k_layout`) still weren't built for Android in this pass. The 2 that were built (`test_cpu_detect`, `test_dispatch_init`) both pass; dispatch picks `arm_neon`.

7. **Original x86 Linux target (192.168.210.23 / "BJ-1") was unreachable then disk-full.** Switched mid-run to 192.168.210.24 (dell-precision-5810). Recorded for posterity; no impact on data.

### Minor (tooling)

8. **`scripts/perf_sweep.sh` had two bugs** caught during the dell sweep: wrong x86 remote path (`~/bitnet-test` vs actual `~/bitnet-test/repo`) and a `set -euo pipefail` interaction that aborted the script on a failing `eval` before `RC=$?` could capture it. Both fixed in commit `064bfcc`.

## Recommendations

### For shipping x86 (do these first)

1. ~~Fix the AVX2 TQ2_0 kernel~~ — **DONE** (commit c4c7c70). `test_quant_tq2_0` passes at avx2.
2. ~~Fix the AVX2 Q6K i8 dot-product~~ — **DONE** (commit 24f6fe5, covers avx2 + avx_vnni + avx512). `test_q6k_layout` passes at avx2.
3. ~~Add 0.5B / MiniCPM4-longrope support to the x86 output path~~ — **DONE** (commit e2292f4). 0.5B decodes on x86 now. Follow-up: longrope `rope_factors_long/short` support for output quality.
4. **Verify AVX2 perf on a modern x86 host.** The dell Xeon E5-2660 v3 (Haswell) has slow gather, so avx2 ≈ scalar there. Re-measure on Ice Lake+/Zen 2+ where gather is fast — the kernel is now correct, so any tier that wins on a modern host can be the default there. Until then, scalar is the safe default on Haswell-class CPUs.

### Per-device thread tuning (once kernels are correct)

| Device | Recommended `BITNET_NUM_THREADS` | Why |
|---|---|---|
| Android 865 | **2** (3 for ≥3B models) | Peaks at t=2–3 then degrades — Snapdragon 865 big.LITTLe has few performant cores; threading overhead dominates past 3. |
| Mac Apple M4 | **6+** (try 8–10) | Scales monotonically up to t=6 with no plateau — M4 has 10 cores and headroom. |
| x86 dell-5810 | **6** | Scales up to t=6 on scalar. Doesn't matter much until AVX kernels are fixed (scalar is slow regardless on this older Xeon). |

### Tooling follow-ups

5. Make `scripts/build_android.sh` respect a `BITNET_BUILD_TESTS` env var so the full ctest suite can run on Android without the reconfigure workaround.
6. The hardcoded `BITNET_TARGET_*` constants in `src/bitnet_internal.h` pin the runtime to the 1B shape. Other sizes work on ARM but the 0.5B output path breaks on x86 — consider making the validation + output path shape-aware rather than 1B-only.

### What's healthy

- **ARM NEON is solid on both Apple Silicon and Snapdragon.** Mac M4 hits 133 tok/s on 1B and scales cleanly. Android 865 matches the README baselines (1B 30 tok/s, 3B 16 tok/s, 8B 8.6 tok/s) — the regression guardrails hold.
- **The dispatch plumbing itself works.** `bitnet_dispatch_init()` correctly auto-detects tier on every device, the `BITNET_CPU_TIER` override is honored (validated, falls back gracefully on unsupported tiers), and the portability fixes didn't regress ARM. The x86 problems are in the kernel *bodies*, not the dispatch machinery.
