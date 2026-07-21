#include "quant_tq2_0_x86.h"
#include "../quant_tq2_0.h"
#include "../thread_config.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#if defined(__GNUC__) || defined(__clang__)
#define BITNET_TARGET_AVX2 __attribute__((target("avx2,fma")))
#ifdef BITNET_AVX_VNNI_AS_AVX512
#define BITNET_TARGET_AVX_VNNI __attribute__((target("avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")))
#else
#define BITNET_TARGET_AVX_VNNI __attribute__((target("avx2,fma,avxvnni")))
#endif
#define BITNET_TARGET_AVX512_VNNI __attribute__((target("avx512f,avx512bw,avx512vnni,avx512dq")))
#else
#define BITNET_TARGET_AVX2
#define BITNET_TARGET_AVX_VNNI
#define BITNET_TARGET_AVX512_VNNI
#endif

#ifndef BITNET_TQ2_0_QK
#define BITNET_TQ2_0_QK 256
#endif

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
    /* Horizontal max-reduce max_vec across 8 lanes. */
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 m = _mm_max_ps(hi, lo);                       /* [m0,m1,m2,m3] */
    __m128 sh = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2));  /* [m2,m3,m0,m1] */
    m = _mm_max_ps(m, sh);                               /* [max(m0,m2), max(m1,m3), ...] */
    sh = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));  /* swap pairs */
    m = _mm_max_ps(m, sh);                               /* all 4 lanes = global max */
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

    /* Quantize with saturation. Process 16 floats per iteration using
     * 128-bit SSE packs, which (unlike the 256-bit packs) do not cross
     * 128-bit lanes and so produce sequentially-laid-out int8 directly —
     * no permute needed. The earlier 256-bit packs + permute4x64 variant
     * corrupted bytes 4..7 of every 16-byte half because 64-bit lane
     * permutation cannot undo the within-lane interleaving the packs
     * introduce. */
    i = 0;
    for (; i + 15 < in_dim; i += 16) {
        __m256i t0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(vec + i + 0), inv_v));
        __m256i t1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(vec + i + 8), inv_v));
        /* 256-bit -> two 128-bit halves of int32. */
        __m128i lo0 = _mm256_castsi256_si128(t0);
        __m128i hi0 = _mm256_extracti128_si256(t0, 1);
        __m128i lo1 = _mm256_castsi256_si128(t1);
        __m128i hi1 = _mm256_extracti128_si256(t1, 1);
        /* packs_epi16/packs_epi32 are 128-bit ops: no lane crossing. */
        __m128i p0 = _mm_packs_epi32(lo0, hi0);   /* 8 int16: vec[i+0..7] */
        __m128i p1 = _mm_packs_epi32(lo1, hi1);   /* 8 int16: vec[i+8..15] */
        __m128i q  = _mm_packs_epi16(p0, p1);     /* 16 int8, sequential */
        _mm_storeu_si128((__m128i *)(qvec + i), q);
    }
    for (; i < in_dim; ++i) {
        int v = (int)lrintf(vec[i] * inv_scale);
        if (v > 127) v = 127; else if (v < -127) v = -127;
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
    /* Phase 3: reuse AVX2. Phase 5 may specialise for AVX-VNNI. */
    return bitnet_tq2_0_quantize_vec_i8_avx2(vec, in_dim, qvec, scale, block_bsums);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_quantize_vec_i8_avx512_vnni(const float *vec, int in_dim, int8_t *qvec,
                                                float *scale, int32_t *block_bsums) {
    /* Phase 3: reuse AVX2. Phase 7 may specialise for 512-bit. */
    return bitnet_tq2_0_quantize_vec_i8_avx2(vec, in_dim, qvec, scale, block_bsums);
}

/* =========================================================================
 * TQ2_0 LUT matmul (float-LUT path)
 *
 * Algorithm reference: src/quant_tq2_0.c::tq2_0_matmul_row_lut*.
 *
 * The LUT is a flat float array indexed by:
 *   entry = lut + ((block*2 + group) * 32 + m) * 256
 * Each "entry" is 256 floats — one per possible byte value (0..255) for
 * the weight code at position m within the group. The contribution of
 * weight byte qs[m] is entry[qs[m]].
 *
 * Per weight block (2 groups × 32 positions = 64 byte codes), the kernel
 * gathers 64 float contributions and sums them to form a per-block scalar
 * accumulator, which is then multiplied by the block scale (read from the
 * block header OR provided externally via the _scales variant).
 *
 * AVX2 strategy: for each m-step of 8, load 8 weight bytes, zero-extend to
 * int32, scale by 4 (sizeof(float)), and use `_mm256_i32gather_ps` against
 * the entry base pointer to fetch 8 contributions in one instruction.
 * Horizontally reduce 8→1 with hadd chains and accumulate into the block
 * scalar. Two groups per block, then scale and add to the row accumulator.
 *
 * The pair variant shares the LUT between two weight matrices (gate + up),
 * amortising the LUT build cost.
 * ========================================================================= */

#ifndef BITNET_TQ2_0_QK
#define BITNET_TQ2_0_QK 256
#endif
#ifndef BITNET_TQ2_0_QS_SIZE
#define BITNET_TQ2_0_QS_SIZE 64
#endif
#ifndef BITNET_TQ2_0_BLOCK_SIZE
#define BITNET_TQ2_0_BLOCK_SIZE 66
#endif

/* fp16 -> fp32 conversion is shared with the scalar reference path via
 * bitnet_fp16_to_fp32() declared in quant_tq2_0.h. */

/* Reduce 8 floats (__m256) to a single scalar. */
BITNET_TARGET_AVX2
static inline float tq2_x86_hadd8_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);            /* 4 lanes */
    __m128 sh = _mm_movehdup_ps(s);           /* [s1,s1,s3,s3] */
    s = _mm_add_ps(s, sh);                    /* [s0+s1, _, s2+s3, _] */
    sh = _mm_movehl_ps(sh, s);                /* [s2+s3, s3, s2, s3] (high half) */
    s = _mm_add_ss(s, sh);                    /* s0 + s1 + s2 + s3 in lane 0 */
    return _mm_cvtss_f32(s);
}

/* Inner accumulation kernel for one weight block.
 *
 * Walks 2 groups × 32 positions = 64 byte codes from `qs_base`, gathers each
 * byte's float contribution from the per-group sub-LUT, and returns the total
 * contribution as a float. */
BITNET_TARGET_AVX2
static inline float tq2_0_lut_block_accumulate_avx2(const uint8_t *qs_base,
                                                       const float *group_lut_base) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    /* Per-lane m-row index: lane i within a gather maps to m-row (m+i), so the
     * gather index must carry both the code AND the m-row offset. iota gives
     * the i=0..7 lane identity; stride256 scales a row index to float offsets. */
    const __m256i iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i stride256 = _mm256_set1_epi32(256);

    /* group 0: m=0..7 and m=8..15, then m=16..23 and m=24..31 in parallel.
     * We process 16 codes per iteration to expose ILP across two accumulators. */
    for (int g = 0; g < 2; ++g) {
        const uint8_t *qs = qs_base + (size_t)g * 32u;
        const float *gl = group_lut_base + (size_t)g * 32u * 256u;
        for (int m = 0; m < 32; m += 16) {
            /* Load 16 bytes: 8 lo + 8 hi. */
            __m128i raw_lo = _mm_loadl_epi64((const __m128i *)(qs + m + 0));
            __m128i raw_hi = _mm_loadl_epi64((const __m128i *)(qs + m + 8));
            __m256i codes_lo = _mm256_cvtepu8_epi32(raw_lo);   /* AVX2: 8 × int32 */
            __m256i codes_hi = _mm256_cvtepu8_epi32(raw_hi);

            /* Per-lane m-row offset: lane i reads gl[(m+i)*256 + code[i]].
             * lo covers rows m+0..m+7, hi covers rows m+8..m+15. */
            __m256i moff_lo = _mm256_mullo_epi32(
                _mm256_add_epi32(_mm256_set1_epi32(m), iota), stride256);
            __m256i moff_hi = _mm256_mullo_epi32(
                _mm256_add_epi32(_mm256_set1_epi32(m + 8), iota), stride256);

            /* Gather with scale=4 (sizeof float); idx is in float units. */
            __m256 g0 = _mm256_i32gather_ps(gl,
                _mm256_add_epi32(codes_lo, moff_lo), 4);
            __m256 g1 = _mm256_i32gather_ps(gl,
                _mm256_add_epi32(codes_hi, moff_hi), 4);
            acc0 = _mm256_add_ps(acc0, g0);
            acc1 = _mm256_add_ps(acc1, g1);
        }
    }

    return tq2_x86_hadd8_ps(_mm256_add_ps(acc0, acc1));
}

BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_vector_lut_avx2(const void *weight, int out_dim, int in_dim,
                                          const float *lut, float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row;

    if (weight == NULL || lut == NULL || out == NULL || out_dim <= 0 || in_dim <= 0) {
        return -1;
    }
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    const size_t row_stride = (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
    for (int row = 0; row < out_dim; ++row) {
        const uint8_t *row_w = bytes + (size_t)row * row_stride;
        float sum = 0.0f;
        for (int k = 0; k < blocks_per_row; ++k) {
            const bitnet_tq2_0_block_t *blk =
                (const bitnet_tq2_0_block_t *)(row_w + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            const float d = bitnet_fp16_to_fp32(blk->d);
            const float *group_lut = lut + ((size_t)k * 2u) * 32u * 256u;
            float block_sum = tq2_0_lut_block_accumulate_avx2(blk->qs, group_lut);
            sum += block_sum * d;
        }
        out[row] = sum;
    }
    return 0;
}

BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_vector_lut_scales_avx2(const void *weight, const float *scales,
                                                  int out_dim, int in_dim,
                                                  const float *lut, float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row;

    if (weight == NULL || scales == NULL || lut == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0) {
        return -1;
    }
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    const size_t row_stride = (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
    for (int row = 0; row < out_dim; ++row) {
        const uint8_t *row_w = bytes + (size_t)row * row_stride;
        const float *row_scales = scales + (size_t)row * (size_t)blocks_per_row;
        float sum = 0.0f;
        for (int k = 0; k < blocks_per_row; ++k) {
            const bitnet_tq2_0_block_t *blk =
                (const bitnet_tq2_0_block_t *)(row_w + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            const float d = row_scales[k];
            const float *group_lut = lut + ((size_t)k * 2u) * 32u * 256u;
            float block_sum = tq2_0_lut_block_accumulate_avx2(blk->qs, group_lut);
            sum += block_sum * d;
        }
        out[row] = sum;
    }
    return 0;
}

BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_vector_lut_pair_avx2(const void *weight_a, const void *weight_b,
                                                int out_dim, int in_dim, const float *lut,
                                                float *out_a, float *out_b) {
    const uint8_t *bytes_a = (const uint8_t *)weight_a;
    const uint8_t *bytes_b = (const uint8_t *)weight_b;
    int blocks_per_row;

    if (weight_a == NULL || weight_b == NULL || lut == NULL ||
        out_a == NULL || out_b == NULL || out_dim <= 0 || in_dim <= 0) {
        return -1;
    }
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    const size_t row_stride = (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
    for (int row = 0; row < out_dim; ++row) {
        const uint8_t *row_a = bytes_a + (size_t)row * row_stride;
        const uint8_t *row_b = bytes_b + (size_t)row * row_stride;
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int k = 0; k < blocks_per_row; ++k) {
            const bitnet_tq2_0_block_t *blk_a =
                (const bitnet_tq2_0_block_t *)(row_a + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            const bitnet_tq2_0_block_t *blk_b =
                (const bitnet_tq2_0_block_t *)(row_b + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            const float d_a = bitnet_fp16_to_fp32(blk_a->d);
            const float d_b = bitnet_fp16_to_fp32(blk_b->d);
            const float *group_lut = lut + ((size_t)k * 2u) * 32u * 256u;
            float sa = tq2_0_lut_block_accumulate_avx2(blk_a->qs, group_lut);
            float sb = tq2_0_lut_block_accumulate_avx2(blk_b->qs, group_lut);
            sum_a += sa * d_a;
            sum_b += sb * d_b;
        }
        out_a[row] = sum_a;
        out_b[row] = sum_b;
    }
    return 0;
}

BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx2(const void *weight_a, const float *scales_a,
                                                       const void *weight_b, const float *scales_b,
                                                       int out_dim, int in_dim, const float *lut,
                                                       float *out_a, float *out_b) {
    const uint8_t *bytes_a = (const uint8_t *)weight_a;
    const uint8_t *bytes_b = (const uint8_t *)weight_b;
    int blocks_per_row;

    if (weight_a == NULL || weight_b == NULL || scales_a == NULL || scales_b == NULL ||
        lut == NULL || out_a == NULL || out_b == NULL || out_dim <= 0 || in_dim <= 0) {
        return -1;
    }
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    const size_t row_stride = (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
    for (int row = 0; row < out_dim; ++row) {
        const uint8_t *row_a = bytes_a + (size_t)row * row_stride;
        const uint8_t *row_b = bytes_b + (size_t)row * row_stride;
        const float *rsa = scales_a + (size_t)row * (size_t)blocks_per_row;
        const float *rsb = scales_b + (size_t)row * (size_t)blocks_per_row;
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int k = 0; k < blocks_per_row; ++k) {
            const bitnet_tq2_0_block_t *blk_a =
                (const bitnet_tq2_0_block_t *)(row_a + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            const bitnet_tq2_0_block_t *blk_b =
                (const bitnet_tq2_0_block_t *)(row_b + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            const float *group_lut = lut + ((size_t)k * 2u) * 32u * 256u;
            float sa = tq2_0_lut_block_accumulate_avx2(blk_a->qs, group_lut);
            float sb = tq2_0_lut_block_accumulate_avx2(blk_b->qs, group_lut);
            sum_a += sa * rsa[k];
            sum_b += sb * rsb[k];
        }
        out_a[row] = sum_a;
        out_b[row] = sum_b;
    }
    return 0;
}

/* VNNI / AVX512 delegate to AVX2 in Phase 3. */
BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_matmul_vector_lut_avx_vnni(const void *weight, int out_dim, int in_dim,
                                              const float *lut, float *out) {
    return bitnet_tq2_0_matmul_vector_lut_avx2(weight, out_dim, in_dim, lut, out);
}
BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_matmul_vector_lut_scales_avx_vnni(const void *weight, const float *scales,
                                                     int out_dim, int in_dim,
                                                     const float *lut, float *out) {
    return bitnet_tq2_0_matmul_vector_lut_scales_avx2(weight, scales, out_dim, in_dim, lut, out);
}
BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_matmul_vector_lut_pair_avx_vnni(const void *weight_a, const void *weight_b,
                                                   int out_dim, int in_dim, const float *lut,
                                                   float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_vector_lut_pair_avx2(weight_a, weight_b, out_dim, in_dim, lut, out_a, out_b);
}
BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx_vnni(const void *weight_a, const float *scales_a,
                                                          const void *weight_b, const float *scales_b,
                                                          int out_dim, int in_dim, const float *lut,
                                                          float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_vector_lut_pair_scales_avx2(weight_a, scales_a, weight_b, scales_b,
                                                              out_dim, in_dim, lut, out_a, out_b);
}
BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_matmul_vector_lut_avx512_vnni(const void *weight, int out_dim, int in_dim,
                                                  const float *lut, float *out) {
    return bitnet_tq2_0_matmul_vector_lut_avx2(weight, out_dim, in_dim, lut, out);
}
BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_matmul_vector_lut_scales_avx512_vnni(const void *weight, const float *scales,
                                                         int out_dim, int in_dim,
                                                         const float *lut, float *out) {
    return bitnet_tq2_0_matmul_vector_lut_scales_avx2(weight, scales, out_dim, in_dim, lut, out);
}
BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_matmul_vector_lut_pair_avx512_vnni(const void *weight_a, const void *weight_b,
                                                       int out_dim, int in_dim, const float *lut,
                                                       float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_vector_lut_pair_avx2(weight_a, weight_b, out_dim, in_dim, lut, out_a, out_b);
}
BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx512_vnni(const void *weight_a, const float *scales_a,
                                                              const void *weight_b, const float *scales_b,
                                                              int out_dim, int in_dim, const float *lut,
                                                              float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_vector_lut_pair_scales_avx2(weight_a, scales_a, weight_b, scales_b,
                                                              out_dim, in_dim, lut, out_a, out_b);
}

/* =========================================================================
 * TQ2_0 I2S matmul (decode hot path)
 *
 * Algorithm reference: src/quant_tq2_0.c::i2s_matmul_4rows_neon.
 *
 * I2S packed layout (per group of 4 rows):
 *   - Packed bytes: each byte holds 4 row codes at one column position:
 *       byte = (row0 << 6) | (row1 << 4) | (row2 << 2) | row3
 *     where each row code is 2 bits (0..3). The packed buffer for one group
 *     spans `sub_blocks_per_group * QK_I2S` bytes.
 *   - Scales: `blocks_per_row * 4` floats (one per row per TQ2_0 block).
 *   - The activation qvec is shared across all 4 rows.
 *
 * Per TQ2_0 block (4 sub-blocks of QK_I2S=64 bytes each):
 *   For each sub-block, walk 64 bytes of packed data and 64 bytes of qvec,
 *   extract 4 row codes from each packed byte via SSSE3 byte shuffles, and
 *   accumulate 4 per-row dot products.
 *
 * Unsigned-LUT path (mandatory on x86 because `_mm_maddubs_epi16` requires
 * uint8 × int8): row codes stay in {0,1,2,3}, dot product accumulates
 * `Σ code * qvec`. To recover the mathematically-correct `Σ (code-1) * qvec`,
 * subtract `Σ qvec` per block (the per-block activation bsum) at the end.
 *
 * AVX2 width: `_mm256_maddubs_epi16` processes 32 bytes per instruction
 * (vs NEON's 16-byte `vdotq_s32`). Dual 256-bit accumulators per row
 * provide ILP, matching the NEON reference's dual-accumulator structure.
 *
 * On x86 there is no thread pool in Phase 3, so the `_parallel` variants
 * process the whole matrix single-threaded.
 * ========================================================================= */

#ifndef QK_I2S
#define QK_I2S 64
#endif

/* SSSE3 byte-shuffle tables for I2S code extraction (unsigned path).
 *
 * Indexed by 4-bit nibble (high or low). Returns the 2-bit code in bits 6-7
 * (lut_hi2) or bits 0-1 (lut_lo2) of the original byte, replicated to all 8
 * bits of the output byte so `_mm_maddubs_epi16` sees {0,1,2,3} per lane.
 *
 * `_mm_shuffle_epi8(lut, idx)` returns `lut[idx[i] & 0x0F]` per byte, or 0
 * when bit 7 of idx[i] is set. Since our indices are nibbles (0..15), this
 * matches ARM `vqtbl1q_u8` exactly.
 */
static const uint8_t i2s_lut_hi2_unsigned_x86[16] = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
};
static const uint8_t i2s_lut_lo2_unsigned_x86[16] = {
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
};

/* Per-row accumulator block-of-4 kernel.
 *
 * Computes 4 output rows for one group (rows row_base..row_base+3) over all
 * blocks_per_row TQ2_0 blocks. Writes 4 floats to out[0..3].
 *
 * Mirrors i2s_matmul_4rows_neon but with 256-bit AVX2 intrinsics. Uses the
 * unsigned-LUT path; if `bsums == NULL`, computes the per-block activation
 * sum inline so we can still apply the bsums correction.
 */
BITNET_TARGET_AVX2
static void i2s_matmul_4rows_avx2(const uint8_t *packed_grp, const float *scales_grp,
                                    int blocks_per_row, const int8_t *qvec,
                                    const int32_t *bsums, float vec_scale,
                                    float *out_0, float *out_1, float *out_2, float *out_3) {
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i lut_hi2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)i2s_lut_hi2_unsigned_x86));
    const __m256i lut_lo2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)i2s_lut_lo2_unsigned_x86));
    const __m256i ones_16 = _mm256_set1_epi16(1);

    float sum_0 = 0.0f, sum_1 = 0.0f, sum_2 = 0.0f, sum_3 = 0.0f;
    int32_t *local_bsums = NULL;

    if (bsums == NULL) {
        /* Caller asked for signed-LUT semantics; emulate by computing the
         * activation block sums locally so we can stay on the unsigned path. */
        local_bsums = (int32_t *)_mm_malloc((size_t)blocks_per_row * sizeof(int32_t), 32);
        if (local_bsums != NULL) {
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                int32_t s = 0;
                const int8_t *qv = qvec + (size_t)blk * BITNET_TQ2_0_QK;
                for (int j = 0; j < BITNET_TQ2_0_QK; ++j) s += (int32_t)qv[j];
                local_bsums[blk] = s;
            }
            bsums = local_bsums;
        }
    }

    for (int blk = 0; blk < blocks_per_row; ++blk) {
        /* Four per-row accumulators (256-bit = 8 int32 lanes each). Dual
         * lo/hi per row to break the dependency chain between iterations:
         * odd iterations add to *_lo, even iterations add to *_hi. */
        __m256i acc_0_lo = _mm256_setzero_si256(), acc_0_hi = _mm256_setzero_si256();
        __m256i acc_1_lo = _mm256_setzero_si256(), acc_1_hi = _mm256_setzero_si256();
        __m256i acc_2_lo = _mm256_setzero_si256(), acc_2_hi = _mm256_setzero_si256();
        __m256i acc_3_lo = _mm256_setzero_si256(), acc_3_hi = _mm256_setzero_si256();

        /* Each TQ2_0 block = 4 I2S sub-blocks of QK_I2S=64 bytes each. */
        int iter = 0;
        for (int sub = 0; sub < 4; ++sub) {
            int sb = blk * 4 + sub;
            const uint8_t *pb = packed_grp + (size_t)sb * (size_t)QK_I2S;
            const int8_t *qv = qvec + (size_t)sb * (size_t)QK_I2S;

            /* Process 64 bytes in 2 iterations of 32 bytes. Each iteration
             * covers 16 byte positions per __m128i half of the __m256i. */
            for (int i = 0; i < QK_I2S; i += 32, ++iter) {
                /* Load 32 packed bytes (split into two 16-byte halves for
                 * shuffle) and 32 qvec bytes (one full 256-bit vector). */
                __m256i pk = _mm256_loadu_si256((const __m256i *)(pb + i));
                __m256i v  = _mm256_loadu_si256((const __m256i *)(qv + i));

                /* High/low nibble of each byte (per-lane); the LUTs are
                 * broadcast into both 128-bit lanes so four vpshufb
                 * lookups produce the four 32-byte row-code vectors. */
                __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(pk, 4), mask_0f);
                __m256i lo_nib = _mm256_and_si256(pk, mask_0f);

                __m256i c0 = _mm256_shuffle_epi8(lut_hi2, hi_nib);
                __m256i c1 = _mm256_shuffle_epi8(lut_lo2, hi_nib);
                __m256i c2 = _mm256_shuffle_epi8(lut_hi2, lo_nib);
                __m256i c3 = _mm256_shuffle_epi8(lut_lo2, lo_nib);

                /* uint8 × int8 -> int16 (horizontally paired), then
                 * int16 × 1 -> int32 (horizontally paired again).
                 * _mm256_maddubs_epi16 operates as two independent 128-bit
                 * lanes, which matches the per-half layout above. */
                __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(c0, v), ones_16);
                __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(c1, v), ones_16);
                __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(c2, v), ones_16);
                __m256i p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(c3, v), ones_16);

                /* Alternate lo/hi accumulators per iteration for ILP. */
                if ((iter & 1) == 0) {
                    acc_0_lo = _mm256_add_epi32(acc_0_lo, p0);
                    acc_1_lo = _mm256_add_epi32(acc_1_lo, p1);
                    acc_2_lo = _mm256_add_epi32(acc_2_lo, p2);
                    acc_3_lo = _mm256_add_epi32(acc_3_lo, p3);
                } else {
                    acc_0_hi = _mm256_add_epi32(acc_0_hi, p0);
                    acc_1_hi = _mm256_add_epi32(acc_1_hi, p1);
                    acc_2_hi = _mm256_add_epi32(acc_2_hi, p2);
                    acc_3_hi = _mm256_add_epi32(acc_3_hi, p3);
                }
            }
        }

        /* Horizontal reduce 8 int32 lanes per accumulator pair to a scalar. */
        __m256i acc0 = _mm256_add_epi32(acc_0_lo, acc_0_hi);
        __m256i acc1 = _mm256_add_epi32(acc_1_lo, acc_1_hi);
        __m256i acc2 = _mm256_add_epi32(acc_2_lo, acc_2_hi);
        __m256i acc3 = _mm256_add_epi32(acc_3_lo, acc_3_hi);

        __m128i a0 = _mm_add_epi32(_mm256_castsi256_si128(acc0), _mm256_extracti128_si256(acc0, 1));
        __m128i a1 = _mm_add_epi32(_mm256_castsi256_si128(acc1), _mm256_extracti128_si256(acc1, 1));
        __m128i a2 = _mm_add_epi32(_mm256_castsi256_si128(acc2), _mm256_extracti128_si256(acc2, 1));
        __m128i a3 = _mm_add_epi32(_mm256_castsi256_si128(acc3), _mm256_extracti128_si256(acc3, 1));
        /* Reduce 4 int32 lanes -> 1: sum pairs (0+1, 2+3), then those halves. */
        a0 = _mm_add_epi32(a0, _mm_shuffle_epi32(a0, _MM_SHUFFLE(0, 1, 2, 3)));
        a1 = _mm_add_epi32(a1, _mm_shuffle_epi32(a1, _MM_SHUFFLE(0, 1, 2, 3)));
        a2 = _mm_add_epi32(a2, _mm_shuffle_epi32(a2, _MM_SHUFFLE(0, 1, 2, 3)));
        a3 = _mm_add_epi32(a3, _mm_shuffle_epi32(a3, _MM_SHUFFLE(0, 1, 2, 3)));
        a0 = _mm_add_epi32(a0, _mm_shuffle_epi32(a0, _MM_SHUFFLE(2, 3, 0, 1)));
        a1 = _mm_add_epi32(a1, _mm_shuffle_epi32(a1, _MM_SHUFFLE(2, 3, 0, 1)));
        a2 = _mm_add_epi32(a2, _mm_shuffle_epi32(a2, _MM_SHUFFLE(2, 3, 0, 1)));
        a3 = _mm_add_epi32(a3, _mm_shuffle_epi32(a3, _MM_SHUFFLE(2, 3, 0, 1)));

        int32_t dot_0 = _mm_cvtsi128_si32(a0);
        int32_t dot_1 = _mm_cvtsi128_si32(a1);
        int32_t dot_2 = _mm_cvtsi128_si32(a2);
        int32_t dot_3 = _mm_cvtsi128_si32(a3);

        /* Apply per-block activation bsums correction (unsigned-LUT path). */
        if (bsums != NULL) {
            const int32_t bsum = bsums[blk];
            dot_0 -= bsum;
            dot_1 -= bsum;
            dot_2 -= bsum;
            dot_3 -= bsum;
        }

        /* Per-row scale (4 floats per block, one per row). */
        const float *sb_scales = scales_grp + (size_t)blk * 4u;
        sum_0 += (float)dot_0 * sb_scales[0];
        sum_1 += (float)dot_1 * sb_scales[1];
        sum_2 += (float)dot_2 * sb_scales[2];
        sum_3 += (float)dot_3 * sb_scales[3];
    }

    *out_0 = sum_0 * vec_scale;
    *out_1 = sum_1 * vec_scale;
    *out_2 = sum_2 * vec_scale;
    *out_3 = sum_3 * vec_scale;

    if (local_bsums != NULL) _mm_free(local_bsums);
}

/* Single-matrix I2S matmul (no thread pool on x86 in Phase 3). */
BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx2(const uint8_t *packed, const float *scales,
                                                  const int32_t *bsums, int out_dim, int in_dim,
                                                  const int8_t *qvec, float vec_scale, float *out) {
    int blocks_per_row, out_dim_padded, n_groups, sub_blocks_per_group;

    if (packed == NULL || scales == NULL || qvec == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) return -1;

    out_dim_padded = (out_dim + 3) & ~3;
    n_groups = out_dim_padded / 4;
    sub_blocks_per_group = in_dim / QK_I2S;

    #pragma omp parallel for schedule(static) num_threads(bitnet_thread_count(16))
    for (int grp = 0; grp < n_groups; ++grp) {
        int row_base = grp * 4;
        const uint8_t *packed_grp = packed + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const float *scales_grp = scales + (size_t)grp * (size_t)blocks_per_row * 4u;

        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        i2s_matmul_4rows_avx2(packed_grp, scales_grp, blocks_per_row, qvec,
                                bsums, vec_scale, &r0, &r1, &r2, &r3);

        if (row_base + 0 < out_dim) out[row_base + 0] = r0;
        if (row_base + 1 < out_dim) out[row_base + 1] = r1;
        if (row_base + 2 < out_dim) out[row_base + 2] = r2;
        if (row_base + 3 < out_dim) out[row_base + 3] = r3;
    }
    return 0;
}

/* Paired I2S matmul: two matrices sharing the same qvec. */
BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx2(const uint8_t *packed_a, const float *scales_a,
                                                       const uint8_t *packed_b, const float *scales_b,
                                                       const int32_t *bsums, int out_dim, int in_dim,
                                                       const int8_t *qvec, float vec_scale,
                                                       float *out_a, float *out_b) {
    int blocks_per_row, out_dim_padded, n_groups, sub_blocks_per_group;

    if (packed_a == NULL || scales_a == NULL || packed_b == NULL || scales_b == NULL ||
        qvec == NULL || out_a == NULL || out_b == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) return -1;

    out_dim_padded = (out_dim + 3) & ~3;
    n_groups = out_dim_padded / 4;
    sub_blocks_per_group = in_dim / QK_I2S;

    #pragma omp parallel for schedule(static) num_threads(bitnet_thread_count(16))
    for (int grp = 0; grp < n_groups; ++grp) {
        int row_base = grp * 4;
        const uint8_t *pg_a = packed_a + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const uint8_t *pg_b = packed_b + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const float *sc_a = scales_a + (size_t)grp * (size_t)blocks_per_row * 4u;
        const float *sc_b = scales_b + (size_t)grp * (size_t)blocks_per_row * 4u;

        float a0, a1, a2, a3, b0, b1, b2, b3;
        i2s_matmul_4rows_avx2(pg_a, sc_a, blocks_per_row, qvec, bsums, vec_scale,
                                &a0, &a1, &a2, &a3);
        i2s_matmul_4rows_avx2(pg_b, sc_b, blocks_per_row, qvec, bsums, vec_scale,
                                &b0, &b1, &b2, &b3);

        if (row_base + 0 < out_dim) { out_a[row_base + 0] = a0; out_b[row_base + 0] = b0; }
        if (row_base + 1 < out_dim) { out_a[row_base + 1] = a1; out_b[row_base + 1] = b1; }
        if (row_base + 2 < out_dim) { out_a[row_base + 2] = a2; out_b[row_base + 2] = b2; }
        if (row_base + 3 < out_dim) { out_a[row_base + 3] = a3; out_b[row_base + 3] = b3; }
    }
    return 0;
}

/* Fused Q/K/V I2S matmul: Q is single-matrix, K+V are paired. */
BITNET_TARGET_AVX2
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx2(const uint8_t *packed_q, const float *scales_q,
                                                 const uint8_t *packed_k, const float *scales_k,
                                                 const uint8_t *packed_v, const float *scales_v,
                                                 const int32_t *bsums,
                                                 int q_dim, int kv_dim, int in_dim,
                                                 const int8_t *qvec, float vec_scale,
                                                 float *out_q, float *out_k, float *out_v) {
    int blocks_per_row, sub_blocks_per_group, q_groups, kv_groups;
    if (packed_q == NULL || scales_q == NULL || packed_k == NULL || scales_k == NULL ||
        packed_v == NULL || scales_v == NULL || qvec == NULL ||
        out_q == NULL || out_k == NULL || out_v == NULL ||
        q_dim <= 0 || kv_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    sub_blocks_per_group = in_dim / QK_I2S;
    q_groups = ((q_dim + 3) & ~3) / 4;
    kv_groups = ((kv_dim + 3) & ~3) / 4;

    /* One team handles all three projections.  K and V share a task so the
     * activation vector remains hot and no second OpenMP region is started. */
    #pragma omp parallel for schedule(static) num_threads(bitnet_thread_count(16))
    for (int task = 0; task < q_groups + kv_groups; ++task) {
        if (task < q_groups) {
            int row = task * 4;
            const uint8_t *pg = packed_q + (size_t)task * (size_t)sub_blocks_per_group * QK_I2S;
            const float *sc = scales_q + (size_t)task * (size_t)blocks_per_row * 4u;
            float r0, r1, r2, r3;
            i2s_matmul_4rows_avx2(pg, sc, blocks_per_row, qvec, bsums, vec_scale,
                                  &r0, &r1, &r2, &r3);
            if (row + 0 < q_dim) out_q[row + 0] = r0;
            if (row + 1 < q_dim) out_q[row + 1] = r1;
            if (row + 2 < q_dim) out_q[row + 2] = r2;
            if (row + 3 < q_dim) out_q[row + 3] = r3;
        } else {
            int grp = task - q_groups;
            int row = grp * 4;
            const uint8_t *pg_k = packed_k + (size_t)grp * (size_t)sub_blocks_per_group * QK_I2S;
            const uint8_t *pg_v = packed_v + (size_t)grp * (size_t)sub_blocks_per_group * QK_I2S;
            const float *sc_k = scales_k + (size_t)grp * (size_t)blocks_per_row * 4u;
            const float *sc_v = scales_v + (size_t)grp * (size_t)blocks_per_row * 4u;
            float k0, k1, k2, k3, v0, v1, v2, v3;
            i2s_matmul_4rows_avx2(pg_k, sc_k, blocks_per_row, qvec, bsums, vec_scale,
                                  &k0, &k1, &k2, &k3);
            i2s_matmul_4rows_avx2(pg_v, sc_v, blocks_per_row, qvec, bsums, vec_scale,
                                  &v0, &v1, &v2, &v3);
            if (row + 0 < kv_dim) { out_k[row + 0] = k0; out_v[row + 0] = v0; }
            if (row + 1 < kv_dim) { out_k[row + 1] = k1; out_v[row + 1] = v1; }
            if (row + 2 < kv_dim) { out_k[row + 2] = k2; out_v[row + 2] = v2; }
            if (row + 3 < kv_dim) { out_k[row + 3] = k3; out_v[row + 3] = v3; }
        }
    }
    return 0;
}

/* VNNI / AVX512 delegate to AVX2 in Phase 3. */
BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx_vnni(const uint8_t *packed, const float *scales,
                                                      const int32_t *bsums, int out_dim, int in_dim,
                                                      const int8_t *qvec, float vec_scale, float *out) {
    return bitnet_tq2_0_matmul_i2s_neon_parallel_avx2(packed, scales, bsums,
                                                         out_dim, in_dim, qvec, vec_scale, out);
}
BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx_vnni(const uint8_t *packed_a, const float *scales_a,
                                                           const uint8_t *packed_b, const float *scales_b,
                                                           const int32_t *bsums, int out_dim, int in_dim,
                                                           const int8_t *qvec, float vec_scale,
                                                           float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx2(packed_a, scales_a, packed_b, scales_b,
                                                              bsums, out_dim, in_dim, qvec, vec_scale,
                                                              out_a, out_b);
}
BITNET_TARGET_AVX_VNNI
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx_vnni(const uint8_t *packed_q, const float *scales_q,
                                                     const uint8_t *packed_k, const float *scales_k,
                                                     const uint8_t *packed_v, const float *scales_v,
                                                     const int32_t *bsums,
                                                     int q_dim, int kv_dim, int in_dim,
                                                     const int8_t *qvec, float vec_scale,
                                                     float *out_q, float *out_k, float *out_v) {
    return bitnet_tq2_0_matmul_i2s_qkv_parallel_avx2(packed_q, scales_q, packed_k, scales_k,
                                                        packed_v, scales_v, bsums,
                                                        q_dim, kv_dim, in_dim,
                                                        qvec, vec_scale, out_q, out_k, out_v);
}
/* =====================================================================
 * AVX512-VNNI I2S kernel — Cascade Lake+ (Ice Lake, Zen 4 with AVX512-VNNI).
 * Replaces the AVX2 maddubs+madd dot product with _mm512_dpbusd_epi32
 * (64 unsigned uint8 codes × 64 signed int8 activations -> 16 int32 partial
 * sums). Code extraction from the I2S packed layout still uses AVX2 shuffle
 * (Cascade Lake does not have AVX512-VBMI). Substantially higher instruction
 * throughput per byte: dpbusd processes 64×64 elements per instruction vs.
 * maddubs 16×16 + madd 8×8 per instruction in AVX2 mode.
 * ===================================================================== */
BITNET_TARGET_AVX512_VNNI
static void i2s_matmul_4rows_avx512_vnni(const uint8_t *packed_grp, const float *scales_grp,
                                           int blocks_per_row, const int8_t *qvec,
                                           const int32_t *bsums, float vec_scale,
                                           float *out_0, float *out_1, float *out_2, float *out_3) {
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i lut_hi2 = _mm_loadu_si128((const __m128i *)i2s_lut_hi2_unsigned_x86);
    const __m128i lut_lo2 = _mm_loadu_si128((const __m128i *)i2s_lut_lo2_unsigned_x86);

    float sum_0 = 0.0f, sum_1 = 0.0f, sum_2 = 0.0f, sum_3 = 0.0f;

    for (int blk = 0; blk < blocks_per_row; ++blk) {
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();

        /* 4 I2S sub-blocks per TQ2 block, each QK_I2S=64 bytes */
        for (int sub = 0; sub < 4; ++sub) {
            int sb = blk * 4 + sub;
            const uint8_t *pb = packed_grp + (size_t)sb * (size_t)QK_I2S;
            const int8_t *qv = qvec + (size_t)sb * (size_t)QK_I2S;

            /* Process 64 packed bytes in two 32-byte iter amounts.  Per 32-byte
             * half: extract 4 rows' 32 codes via 128-bit shuffle (same pattern
             * as the AVX2 kernel), then combine halves into __m512i for dpbusd. */
            __m256i row0_lo, row0_hi, row1_lo, row1_hi, row2_lo, row2_hi, row3_lo, row3_hi;

            for (int half = 0; half < 2; ++half) {
                int off = half * 32;
                __m128i pk_lo = _mm_loadu_si128((const __m128i *)(pb + off));
                __m128i pk_hi = _mm_loadu_si128((const __m128i *)(pb + off + 16));

                __m128i hi_nib_lo = _mm_srli_epi16(pk_lo, 4);
                __m128i hi_nib_hi = _mm_srli_epi16(pk_hi, 4);
                hi_nib_lo = _mm_and_si128(hi_nib_lo, mask_0f);
                hi_nib_hi = _mm_and_si128(hi_nib_hi, mask_0f);
                __m128i lo_nib_lo = _mm_and_si128(pk_lo, mask_0f);
                __m128i lo_nib_hi = _mm_and_si128(pk_hi, mask_0f);

                __m128i c0_lo = _mm_shuffle_epi8(lut_hi2, hi_nib_lo);
                __m128i c1_lo = _mm_shuffle_epi8(lut_lo2, hi_nib_lo);
                __m128i c2_lo = _mm_shuffle_epi8(lut_hi2, lo_nib_lo);
                __m128i c3_lo = _mm_shuffle_epi8(lut_lo2, lo_nib_lo);
                __m128i c0_hi = _mm_shuffle_epi8(lut_hi2, hi_nib_hi);
                __m128i c1_hi = _mm_shuffle_epi8(lut_lo2, hi_nib_hi);
                __m128i c2_hi = _mm_shuffle_epi8(lut_hi2, lo_nib_hi);
                __m128i c3_hi = _mm_shuffle_epi8(lut_lo2, lo_nib_hi);

                /* Combine 2×16 bytes into __m256i (32 codes) */
                if (half == 0) {
                    row0_lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c0_lo), c0_hi, 1);
                    row1_lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c1_lo), c1_hi, 1);
                    row2_lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c2_lo), c2_hi, 1);
                    row3_lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c3_lo), c3_hi, 1);
                } else {
                    row0_hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c0_lo), c0_hi, 1);
                    row1_hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c1_lo), c1_hi, 1);
                    row2_hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c2_lo), c2_hi, 1);
                    row3_hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c3_lo), c3_hi, 1);
                }
            }

            /* Combine two 32-byte halves -> 64-byte rows for dpbusd.
             * Load 64 qvec bytes -> __m512i. */
            __m512i r0 = _mm512_inserti64x4(_mm512_castsi256_si512(row0_lo), row0_hi, 1);
            __m512i r1 = _mm512_inserti64x4(_mm512_castsi256_si512(row1_lo), row1_hi, 1);
            __m512i r2 = _mm512_inserti64x4(_mm512_castsi256_si512(row2_lo), row2_hi, 1);
            __m512i r3 = _mm512_inserti64x4(_mm512_castsi256_si512(row3_lo), row3_hi, 1);

            __m256i q_lo = _mm256_loadu_si256((const __m256i *)(qv));
            __m256i q_hi = _mm256_loadu_si256((const __m256i *)(qv + 32));
            __m512i q512 = _mm512_inserti64x4(_mm512_castsi256_si512(q_lo), q_hi, 1);

            /* dpbusd(unsigned codes, signed qvec): 64 × 64 products -> 16 i32 */
            acc0 = _mm512_dpbusd_epi32(acc0, r0, q512);
            acc1 = _mm512_dpbusd_epi32(acc1, r1, q512);
            acc2 = _mm512_dpbusd_epi32(acc2, r2, q512);
            acc3 = _mm512_dpbusd_epi32(acc3, r3, q512);
        }

        /* Horizontal reduce: 16 i32 lanes -> 1 scalar per row */
        int32_t dot_0 = _mm512_reduce_add_epi32(acc0);
        int32_t dot_1 = _mm512_reduce_add_epi32(acc1);
        int32_t dot_2 = _mm512_reduce_add_epi32(acc2);
        int32_t dot_3 = _mm512_reduce_add_epi32(acc3);

        /* Per-block activation bsums correction */
        if (bsums != NULL) {
            const int32_t bsum = bsums[blk];
            dot_0 -= bsum;  dot_1 -= bsum;
            dot_2 -= bsum;  dot_3 -= bsum;
        }

        /* Per-row scale application */
        const float *sb_scales = scales_grp + (size_t)blk * 4u;
        sum_0 += (float)dot_0 * sb_scales[0];
        sum_1 += (float)dot_1 * sb_scales[1];
        sum_2 += (float)dot_2 * sb_scales[2];
        sum_3 += (float)dot_3 * sb_scales[3];
    }

    *out_0 = sum_0 * vec_scale;
    *out_1 = sum_1 * vec_scale;
    *out_2 = sum_2 * vec_scale;
    *out_3 = sum_3 * vec_scale;
}

BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx512_vnni(const uint8_t *packed, const float *scales,
                                                         const int32_t *bsums, int out_dim, int in_dim,
                                                         const int8_t *qvec, float vec_scale, float *out) {
    int blocks_per_row, out_dim_padded, n_groups, sub_blocks_per_group;
    if (packed == NULL || scales == NULL || qvec == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) return -1;
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) return -1;
    out_dim_padded = (out_dim + 3) & ~3;
    n_groups = out_dim_padded / 4;
    sub_blocks_per_group = in_dim / QK_I2S;
    #pragma omp parallel for schedule(static) num_threads(bitnet_thread_count(16))
    for (int grp = 0; grp < n_groups; ++grp) {
        int row_base = grp * 4;
        const uint8_t *packed_grp = packed + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const float *scales_grp = scales + (size_t)grp * (size_t)blocks_per_row * 4u;
        float r0,r1,r2,r3;
        i2s_matmul_4rows_avx512_vnni(packed_grp,scales_grp,blocks_per_row,qvec,bsums,vec_scale,&r0,&r1,&r2,&r3);
        if (row_base+0<out_dim) out[row_base+0]=r0; if (row_base+1<out_dim) out[row_base+1]=r1;
        if (row_base+2<out_dim) out[row_base+2]=r2; if (row_base+3<out_dim) out[row_base+3]=r3;
    }
    return 0;
}
BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx512_vnni(const uint8_t *packed_a, const float *scales_a,
                                                              const uint8_t *packed_b, const float *scales_b,
                                                              const int32_t *bsums, int out_dim, int in_dim,
                                                              const int8_t *qvec, float vec_scale,
                                                              float *out_a, float *out_b) {
    int blocks_per_row, sub_blocks_per_group, n_groups;
    if (packed_a == NULL || scales_a == NULL || packed_b == NULL || scales_b == NULL ||
        qvec == NULL || out_a == NULL || out_b == NULL || out_dim <= 0 || in_dim <= 0 ||
        in_dim % BITNET_TQ2_0_QK != 0) return -1;
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    sub_blocks_per_group = in_dim / QK_I2S;
    n_groups = ((out_dim + 3) & ~3) / 4;
    #pragma omp parallel for schedule(static) num_threads(bitnet_thread_count(16))
    for (int grp = 0; grp < n_groups; ++grp) {
        int row = grp * 4;
        const uint8_t *pg_a = packed_a + (size_t)grp * (size_t)sub_blocks_per_group * QK_I2S;
        const uint8_t *pg_b = packed_b + (size_t)grp * (size_t)sub_blocks_per_group * QK_I2S;
        const float *sc_a = scales_a + (size_t)grp * (size_t)blocks_per_row * 4u;
        const float *sc_b = scales_b + (size_t)grp * (size_t)blocks_per_row * 4u;
        float a0, a1, a2, a3, b0, b1, b2, b3;
        i2s_matmul_4rows_avx512_vnni(pg_a, sc_a, blocks_per_row, qvec, bsums, vec_scale,
                                     &a0, &a1, &a2, &a3);
        i2s_matmul_4rows_avx512_vnni(pg_b, sc_b, blocks_per_row, qvec, bsums, vec_scale,
                                     &b0, &b1, &b2, &b3);
        if (row + 0 < out_dim) { out_a[row + 0] = a0; out_b[row + 0] = b0; }
        if (row + 1 < out_dim) { out_a[row + 1] = a1; out_b[row + 1] = b1; }
        if (row + 2 < out_dim) { out_a[row + 2] = a2; out_b[row + 2] = b2; }
        if (row + 3 < out_dim) { out_a[row + 3] = a3; out_b[row + 3] = b3; }
    }
    return 0;
}
BITNET_TARGET_AVX512_VNNI
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx512_vnni(const uint8_t *packed_q, const float *scales_q,
                                                        const uint8_t *packed_k, const float *scales_k,
                                                        const uint8_t *packed_v, const float *scales_v,
                                                        const int32_t *bsums,
                                                        int q_dim, int kv_dim, int in_dim,
                                                        const int8_t *qvec, float vec_scale,
                                                        float *out_q, float *out_k, float *out_v) {
    int blocks_per_row, sub_blocks_per_group, q_groups, kv_groups;
    if (packed_q == NULL || scales_q == NULL || packed_k == NULL || scales_k == NULL ||
        packed_v == NULL || scales_v == NULL || qvec == NULL || out_q == NULL ||
        out_k == NULL || out_v == NULL || q_dim <= 0 || kv_dim <= 0 || in_dim <= 0 ||
        in_dim % BITNET_TQ2_0_QK != 0) return -1;
    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    sub_blocks_per_group = in_dim / QK_I2S;
    q_groups = ((q_dim + 3) & ~3) / 4;
    kv_groups = ((kv_dim + 3) & ~3) / 4;
    #pragma omp parallel for schedule(static) num_threads(bitnet_thread_count(16))
    for (int task = 0; task < q_groups + kv_groups; ++task) {
        if (task < q_groups) {
            int row = task * 4;
            const uint8_t *pg = packed_q + (size_t)task * (size_t)sub_blocks_per_group * QK_I2S;
            const float *sc = scales_q + (size_t)task * (size_t)blocks_per_row * 4u;
            float r0, r1, r2, r3;
            i2s_matmul_4rows_avx512_vnni(pg, sc, blocks_per_row, qvec, bsums, vec_scale,
                                         &r0, &r1, &r2, &r3);
            if (row + 0 < q_dim) out_q[row + 0] = r0;
            if (row + 1 < q_dim) out_q[row + 1] = r1;
            if (row + 2 < q_dim) out_q[row + 2] = r2;
            if (row + 3 < q_dim) out_q[row + 3] = r3;
        } else {
            int grp = task - q_groups;
            int row = grp * 4;
            const uint8_t *pg_k = packed_k + (size_t)grp * (size_t)sub_blocks_per_group * QK_I2S;
            const uint8_t *pg_v = packed_v + (size_t)grp * (size_t)sub_blocks_per_group * QK_I2S;
            const float *sc_k = scales_k + (size_t)grp * (size_t)blocks_per_row * 4u;
            const float *sc_v = scales_v + (size_t)grp * (size_t)blocks_per_row * 4u;
            float k0, k1, k2, k3, v0, v1, v2, v3;
            i2s_matmul_4rows_avx512_vnni(pg_k, sc_k, blocks_per_row, qvec, bsums, vec_scale,
                                         &k0, &k1, &k2, &k3);
            i2s_matmul_4rows_avx512_vnni(pg_v, sc_v, blocks_per_row, qvec, bsums, vec_scale,
                                         &v0, &v1, &v2, &v3);
            if (row + 0 < kv_dim) { out_k[row + 0] = k0; out_v[row + 0] = v0; }
            if (row + 1 < kv_dim) { out_k[row + 1] = k1; out_v[row + 1] = v1; }
            if (row + 2 < kv_dim) { out_k[row + 2] = k2; out_v[row + 2] = v2; }
            if (row + 3 < kv_dim) { out_k[row + 3] = k3; out_v[row + 3] = v3; }
        }
    }
    return 0;
}

#else  /* !defined(__x86_64__) && !defined(_M_X64) */

/* Non-x86 build: the per-tier symbols still need to exist so kernel_registry.c
 * can reference them under #if defined(__x86_64__). This translation unit is
 * only compiled into the library once; on ARM the kernel_registry.c gated
 * block is excluded too, so these stubs never get linked. They are present
 * purely to satisfy compilation if this file is ever built on non-x86. */
int bitnet_tq2_0_quantize_vec_i8_avx2(const float *vec, int in_dim, int8_t *qvec,
                                       float *scale, int32_t *block_bsums) {
    (void)vec; (void)in_dim; (void)qvec; (void)scale; (void)block_bsums;
    return -1;
}
int bitnet_tq2_0_quantize_vec_i8_avx_vnni(const float *vec, int in_dim, int8_t *qvec,
                                            float *scale, int32_t *block_bsums) {
    (void)vec; (void)in_dim; (void)qvec; (void)scale; (void)block_bsums;
    return -1;
}
int bitnet_tq2_0_quantize_vec_i8_avx512_vnni(const float *vec, int in_dim, int8_t *qvec,
                                                float *scale, int32_t *block_bsums) {
    (void)vec; (void)in_dim; (void)qvec; (void)scale; (void)block_bsums;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_avx2(const void *weight, int out_dim, int in_dim,
                                          const float *lut, float *out) {
    (void)weight; (void)out_dim; (void)in_dim; (void)lut; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_scales_avx2(const void *weight, const float *scales,
                                                  int out_dim, int in_dim,
                                                  const float *lut, float *out) {
    (void)weight; (void)scales; (void)out_dim; (void)in_dim; (void)lut; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_pair_avx2(const void *weight_a, const void *weight_b,
                                                int out_dim, int in_dim, const float *lut,
                                                float *out_a, float *out_b) {
    (void)weight_a; (void)weight_b; (void)out_dim; (void)in_dim; (void)lut;
    (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx2(const void *weight_a, const float *scales_a,
                                                       const void *weight_b, const float *scales_b,
                                                       int out_dim, int in_dim, const float *lut,
                                                       float *out_a, float *out_b) {
    (void)weight_a; (void)scales_a; (void)weight_b; (void)scales_b; (void)out_dim;
    (void)in_dim; (void)lut; (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_avx_vnni(const void *weight, int out_dim, int in_dim,
                                              const float *lut, float *out) {
    (void)weight; (void)out_dim; (void)in_dim; (void)lut; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_scales_avx_vnni(const void *weight, const float *scales,
                                                     int out_dim, int in_dim,
                                                     const float *lut, float *out) {
    (void)weight; (void)scales; (void)out_dim; (void)in_dim; (void)lut; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_pair_avx_vnni(const void *weight_a, const void *weight_b,
                                                   int out_dim, int in_dim, const float *lut,
                                                   float *out_a, float *out_b) {
    (void)weight_a; (void)weight_b; (void)out_dim; (void)in_dim; (void)lut;
    (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx_vnni(const void *weight_a, const float *scales_a,
                                                          const void *weight_b, const float *scales_b,
                                                          int out_dim, int in_dim, const float *lut,
                                                          float *out_a, float *out_b) {
    (void)weight_a; (void)scales_a; (void)weight_b; (void)scales_b; (void)out_dim;
    (void)in_dim; (void)lut; (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_avx512_vnni(const void *weight, int out_dim, int in_dim,
                                                  const float *lut, float *out) {
    (void)weight; (void)out_dim; (void)in_dim; (void)lut; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_scales_avx512_vnni(const void *weight, const float *scales,
                                                         int out_dim, int in_dim,
                                                         const float *lut, float *out) {
    (void)weight; (void)scales; (void)out_dim; (void)in_dim; (void)lut; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_pair_avx512_vnni(const void *weight_a, const void *weight_b,
                                                       int out_dim, int in_dim, const float *lut,
                                                       float *out_a, float *out_b) {
    (void)weight_a; (void)weight_b; (void)out_dim; (void)in_dim; (void)lut;
    (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx512_vnni(const void *weight_a, const float *scales_a,
                                                              const void *weight_b, const float *scales_b,
                                                              int out_dim, int in_dim, const float *lut,
                                                              float *out_a, float *out_b) {
    (void)weight_a; (void)scales_a; (void)weight_b; (void)scales_b; (void)out_dim;
    (void)in_dim; (void)lut; (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx2(const uint8_t *packed, const float *scales,
                                                  const int32_t *bsums, int out_dim, int in_dim,
                                                  const int8_t *qvec, float vec_scale, float *out) {
    (void)packed; (void)scales; (void)bsums; (void)out_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx_vnni(const uint8_t *packed, const float *scales,
                                                      const int32_t *bsums, int out_dim, int in_dim,
                                                      const int8_t *qvec, float vec_scale, float *out) {
    (void)packed; (void)scales; (void)bsums; (void)out_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx512_vnni(const uint8_t *packed, const float *scales,
                                                         const int32_t *bsums, int out_dim, int in_dim,
                                                         const int8_t *qvec, float vec_scale, float *out) {
    (void)packed; (void)scales; (void)bsums; (void)out_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx2(const uint8_t *packed_a, const float *scales_a,
                                                       const uint8_t *packed_b, const float *scales_b,
                                                       const int32_t *bsums, int out_dim, int in_dim,
                                                       const int8_t *qvec, float vec_scale,
                                                       float *out_a, float *out_b) {
    (void)packed_a; (void)scales_a; (void)packed_b; (void)scales_b; (void)bsums;
    (void)out_dim; (void)in_dim; (void)qvec; (void)vec_scale; (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx_vnni(const uint8_t *packed_a, const float *scales_a,
                                                           const uint8_t *packed_b, const float *scales_b,
                                                           const int32_t *bsums, int out_dim, int in_dim,
                                                           const int8_t *qvec, float vec_scale,
                                                           float *out_a, float *out_b) {
    (void)packed_a; (void)scales_a; (void)packed_b; (void)scales_b; (void)bsums;
    (void)out_dim; (void)in_dim; (void)qvec; (void)vec_scale; (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx512_vnni(const uint8_t *packed_a, const float *scales_a,
                                                              const uint8_t *packed_b, const float *scales_b,
                                                              const int32_t *bsums, int out_dim, int in_dim,
                                                              const int8_t *qvec, float vec_scale,
                                                              float *out_a, float *out_b) {
    (void)packed_a; (void)scales_a; (void)packed_b; (void)scales_b; (void)bsums;
    (void)out_dim; (void)in_dim; (void)qvec; (void)vec_scale; (void)out_a; (void)out_b;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx2(const uint8_t *packed_q, const float *scales_q,
                                                 const uint8_t *packed_k, const float *scales_k,
                                                 const uint8_t *packed_v, const float *scales_v,
                                                 const int32_t *bsums,
                                                 int q_dim, int kv_dim, int in_dim,
                                                 const int8_t *qvec, float vec_scale,
                                                 float *out_q, float *out_k, float *out_v) {
    (void)packed_q; (void)scales_q; (void)packed_k; (void)scales_k; (void)packed_v;
    (void)scales_v; (void)bsums; (void)q_dim; (void)kv_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out_q; (void)out_k; (void)out_v;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx_vnni(const uint8_t *packed_q, const float *scales_q,
                                                     const uint8_t *packed_k, const float *scales_k,
                                                     const uint8_t *packed_v, const float *scales_v,
                                                     const int32_t *bsums,
                                                     int q_dim, int kv_dim, int in_dim,
                                                     const int8_t *qvec, float vec_scale,
                                                     float *out_q, float *out_k, float *out_v) {
    (void)packed_q; (void)scales_q; (void)packed_k; (void)scales_k; (void)packed_v;
    (void)scales_v; (void)bsums; (void)q_dim; (void)kv_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out_q; (void)out_k; (void)out_v;
    return -1;
}
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx512_vnni(const uint8_t *packed_q, const float *scales_q,
                                                        const uint8_t *packed_k, const float *scales_k,
                                                        const uint8_t *packed_v, const float *scales_v,
                                                        const int32_t *bsums,
                                                        int q_dim, int kv_dim, int in_dim,
                                                        const int8_t *qvec, float vec_scale,
                                                        float *out_q, float *out_k, float *out_v) {
    (void)packed_q; (void)scales_q; (void)packed_k; (void)scales_k; (void)packed_v;
    (void)scales_v; (void)bsums; (void)q_dim; (void)kv_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out_q; (void)out_k; (void)out_v;
    return -1;
}

#endif
