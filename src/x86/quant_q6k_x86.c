#include "quant_q6k_x86.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>
#include <math.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define BITNET_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define BITNET_TARGET_AVX_VNNI __attribute__((target("avx2,fma,avxvnni")))
#define BITNET_TARGET_AVX512_VNNI __attribute__((target("avx512f,avx512bw,avx512vnni,avx512dq")))
#else
#define BITNET_TARGET_AVX2
#define BITNET_TARGET_AVX_VNNI
#define BITNET_TARGET_AVX512_VNNI
#endif

/* =====================================================================
 * Q6K algorithm summary
 * ---------------------------------------------------------------------
 * Each Q6K block packs 256 6-bit weights. After expand_to_q8, weights are
 * signed int8 in [-32, 31]; activations are signed int8 in [-128, 127].
 *
 * x86 dotprod emulation:
 *   - VNNI:  _mm256_dpbusd_epi32(acc, a_u8, b_i8) -> 8 int32 lanes
 *   - AVX2:  _mm256_maddubs_epi16(a_u8, b_i8) then _mm256_madd_epi16
 *
 * Since Q6K weights are signed, we add 32 to make them uint8 (0..63),
 * use the unsigned×signed hardware path, then subtract 32*sum(activations)
 * per group. The bias correction is computed once per 16-element group.
 *
 * To fill a 32-byte VNNI operand, we pair two adjacent 16-element groups
 * (32 bytes of weights + 32 bytes of activations). Each dpbusd produces
 * 8 int32 lanes (4-wide groups); we horizontally sum them into 2 ints,
 * one per source group. The bias correction is applied per-group.
 * ===================================================================== */

/* fp16 -> fp32 (matches the helper in quant_q6k.c). */
static inline float q6k_fp16_to_fp32_x86(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t raw;

    if (exp == 0) {
        if (mant == 0) {
            raw = sign << 31;
            float f;
            memcpy(&f, &raw, sizeof(f));
            return f;
        }
        float value = ldexpf((float)mant / 1024.0f, -14);
        return sign ? -value : value;
    }
    if (exp == 31) {
        raw = (sign << 31) | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &raw, sizeof(f));
        return f;
    }
    exp = exp - 15u + 127u;
    mant <<= 13;
    raw = (sign << 31) | (exp << 23) | mant;
    float f;
    memcpy(&f, &raw, sizeof(f));
    return f;
}

/* Reference scalar implementation of one 16-element group dot product.
 * Kept as documentation of the bias-correction invariant; the AVX2 and
 * AVX-VNNI tiers now use vectorized helpers (q6k_dot16_avx2 and
 * q6k_dot32_pair_avx_vnni[_int] respectively) that produce bit-identical
 * results. Uncomment + recompile to spot-check parity. */
#if 0
static inline int32_t q6k_dot16_scalar(const int8_t *q, const int8_t *v) {
    int32_t acc = 0;
    for (int i = 0; i < 16; ++i) {
        acc += (int32_t)q[i] * (int32_t)v[i];
    }
    return acc;
}
#endif

/* Sum 4 int32 lanes of __m128i horizontally. */
static inline int32_t q6k_hsum4_epi32(__m128i v) {
    __m128i sh = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm_add_epi32(v, sh);
    sh = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm_add_epi32(v, sh);
    return _mm_cvtsi128_si32(v);
}

/* Sum 8 int32 lanes of __m256i horizontally. */
static inline int32_t q6k_hsum8_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    return q6k_hsum4_epi32(s);
}

/* Sum 8 float lanes of __m256 horizontally. */
BITNET_TARGET_AVX2
static inline float q6k_hsum8_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(hi, lo);
    __m128 sh = _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1));
    s = _mm_add_ps(s, sh);
    sh = _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2));
    s = _mm_add_ps(s, sh);
    return _mm_cvtss_f32(s);
}

/* =====================================================================
 * AVX2 tier
 *
 * We process 32 elements (two adjacent 16-element groups) at a time using
 * a single 256-bit maddubs + madd. Each 32-byte block produces 8 int32
 * lanes that we horizontally sum; bias correction uses sum_of_32_acts.
 * ===================================================================== */

BITNET_TARGET_AVX2
int bitnet_q6k_dot_product_i8_neon_avx2(const bitnet_q6k_block_t *block,
                                          const int8_t *qvec, float vec_scale,
                                          size_t len, float *out) {
    if (block == NULL || qvec == NULL || out == NULL || len < BITNET_Q6K_QK) {
        return -1;
    }

    const float d = q6k_fp16_to_fp32_x86(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;
    float sum = 0.0f;

    /* The inline i8 path requires Q6K-specific unpacking that is intricate
     * to vectorize; we use the scalar reference here. The dominant cost is
     * in the q8 expanded path (used by output projection), which is fully
     * SIMD-ized below. Phase 7 can specialize this if profiling warrants. */
    for (size_t n = 0; n < BITNET_Q6K_QK; n += 128) {
        const int8_t *v = qvec + n;
        int32_t acc[8] = {0};

        for (int l = 0; l < 16; ++l) {
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc[0] += q1 * (int)v[l + 0];
            acc[2] += q2 * (int)v[l + 32];
            acc[4] += q3 * (int)v[l + 64];
            acc[6] += q4 * (int)v[l + 96];
        }
        for (int l = 16; l < 32; ++l) {
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc[1] += q1 * (int)v[l + 0];
            acc[3] += q2 * (int)v[l + 32];
            acc[5] += q3 * (int)v[l + 64];
            acc[7] += q4 * (int)v[l + 96];
        }

        sum += d * vec_scale * (
            (float)sc[0] * (float)acc[0] +
            (float)sc[1] * (float)acc[1] +
            (float)sc[2] * (float)acc[2] +
            (float)sc[3] * (float)acc[3] +
            (float)sc[4] * (float)acc[4] +
            (float)sc[5] * (float)acc[5] +
            (float)sc[6] * (float)acc[6] +
            (float)sc[7] * (float)acc[7]);

        ql += 64;
        qh += 32;
        sc += 8;
    }

    *out = sum;
    return 0;
}

/* Per-16-element dot product using AVX2 maddubs+madd with bias correction.
 * Q6K weights are signed int8 in [-32, 31]; activations are signed int8.
 * _mm256_maddubs_epi16 wants uint8×int8, so we add 32 to the weights to
 * make them unsigned (0..63), then subtract 32*sum(v) for bias correction.
 *
 * We use 128-bit ops so a single 16-element group fits exactly: maddubs
 * turns 16 u8×i8 pairs into 8 int16 lanes; madd turns those into 4 int32
 * lanes that we horizontally sum. This preserves the per-group scale
 * granularity cleanly. */
BITNET_TARGET_AVX2
static inline int32_t q6k_dot16_avx2(const __m128i w16_i8, const __m128i v16_i8) {
    const __m128i bias16 = _mm_set1_epi8(32);
    const __m128i ones_16 = _mm_set1_epi16(1);
    const __m128i w16_u8 = _mm_add_epi8(w16_i8, bias16);
    const __m128i prod16 = _mm_maddubs_epi16(w16_u8, v16_i8);
    const __m128i prod32 = _mm_madd_epi16(prod16, ones_16);
    int32_t raw = q6k_hsum4_epi32(prod32);
    /* Bias correction: 32 * sum(16 activations). */
    const __m128i vsum16 = _mm_madd_epi16(_mm_maddubs_epi16(_mm_set1_epi8(1), v16_i8), ones_16);
    int32_t sum_v = q6k_hsum4_epi32(vsum16);
    return raw - 32 * sum_v;
}

BITNET_TARGET_AVX2
int bitnet_q6k_dot_product_q8_avx2(const int8_t *q8, const float *scales,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float *out) {
    if (q8 == NULL || scales == NULL || qvec == NULL || out == NULL ||
        blocks_per_row <= 0) {
        return -1;
    }

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const float *sc = scales + (size_t)b * 16u;

        /* 16 groups of 16 elements. Each scale applies to one 16-element
         * group. AVX2 maddubs+madd does the inner dot (q6k_dot16_avx2);
         * the per-group contribution is a scalar, accumulated in fp32. */
        for (int g = 0; g < 16; ++g) {
            __m128i w16 = _mm_loadu_si128((const __m128i *)(q + g * 16));
            __m128i v16 = _mm_loadu_si128((const __m128i *)(v + g * 16));
            int32_t d0 = q6k_dot16_avx2(w16, v16);

            acc += (float)d0 * sc[g];
        }
    }

    *out = acc * vec_scale;
    return 0;
}

BITNET_TARGET_AVX2
int bitnet_q6k_dot_product_q8_4_avx2(const int8_t *q8, const float *scales,
                                       int row_stride, int scale_stride,
                                       int blocks_per_row, const int8_t *qvec,
                                       float vec_scale, float out[4]) {
    if (q8 == NULL || scales == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q0 = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *q1 = q0 + (size_t)row_stride;
        const int8_t *q2 = q1 + (size_t)row_stride;
        const int8_t *q3 = q2 + (size_t)row_stride;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const float *s0 = scales + (size_t)b * 16u;
        const float *s1 = s0 + (size_t)scale_stride;
        const float *s2 = s1 + (size_t)scale_stride;
        const float *s3 = s2 + (size_t)scale_stride;

        for (int g = 0; g < 16; ++g) {
            __m128i vp = _mm_loadu_si128((const __m128i *)(v + g * 16));
            __m128i w0 = _mm_loadu_si128((const __m128i *)(q0 + g * 16));
            __m128i w1 = _mm_loadu_si128((const __m128i *)(q1 + g * 16));
            __m128i w2 = _mm_loadu_si128((const __m128i *)(q2 + g * 16));
            __m128i w3 = _mm_loadu_si128((const __m128i *)(q3 + g * 16));
            int32_t d0 = q6k_dot16_avx2(w0, vp);
            int32_t d1 = q6k_dot16_avx2(w1, vp);
            int32_t d2 = q6k_dot16_avx2(w2, vp);
            int32_t d3 = q6k_dot16_avx2(w3, vp);

            acc0 += (float)d0 * s0[g];
            acc1 += (float)d1 * s1[g];
            acc2 += (float)d2 * s2[g];
            acc3 += (float)d3 * s3[g];
        }
    }

    out[0] = acc0 * vec_scale;
    out[1] = acc1 * vec_scale;
    out[2] = acc2 * vec_scale;
    out[3] = acc3 * vec_scale;
    return 0;
}

BITNET_TARGET_AVX2
int bitnet_q6k_dot_product_q8_8_avx2(const int8_t *q8, const float *scales,
                                       int row_stride, int scale_stride,
                                       int blocks_per_row, const int8_t *qvec,
                                       float vec_scale, float out[8]) {
    if (bitnet_q6k_dot_product_q8_4_avx2(q8, scales, row_stride, scale_stride,
                                           blocks_per_row, qvec, vec_scale, out) != 0) {
        return -1;
    }
    return bitnet_q6k_dot_product_q8_4_avx2(q8 + 4 * (size_t)row_stride,
                                              scales + 4 * (size_t)scale_stride,
                                              row_stride, scale_stride,
                                              blocks_per_row, qvec, vec_scale, out + 4);
}

BITNET_TARGET_AVX2
int bitnet_q6k_dot_product_q8_compact_avx2(const int8_t *q8, const int8_t *scales,
                                             const float *d, int blocks_per_row,
                                             const int8_t *qvec, float vec_scale, float *out) {
    if (q8 == NULL || scales == NULL || d == NULL || qvec == NULL || out == NULL ||
        blocks_per_row <= 0) {
        return -1;
    }

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const int8_t *sc = scales + (size_t)b * 16u;
        int32_t block_sum = 0;

        for (int g = 0; g < 16; ++g) {
            __m128i w16 = _mm_loadu_si128((const __m128i *)(q + g * 16));
            __m128i v16 = _mm_loadu_si128((const __m128i *)(v + g * 16));
            int32_t dot = q6k_dot16_avx2(w16, v16);
            block_sum += dot * (int32_t)sc[g];
        }

        acc += (float)block_sum * d[b];
    }

    *out = acc * vec_scale;
    return 0;
}

BITNET_TARGET_AVX2
int bitnet_q6k_dot_product_q8_compact_4_avx2(const int8_t *q8, const int8_t *scales,
                                                const float *d, int row_stride,
                                                int scale_stride, int d_stride,
                                                int blocks_per_row, const int8_t *qvec,
                                                float vec_scale, float out[4]) {
    if (q8 == NULL || scales == NULL || d == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || d_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q0 = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *q1 = q0 + (size_t)row_stride;
        const int8_t *q2 = q1 + (size_t)row_stride;
        const int8_t *q3 = q2 + (size_t)row_stride;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const int8_t *s0 = scales + (size_t)b * 16u;
        const int8_t *s1 = s0 + (size_t)scale_stride;
        const int8_t *s2 = s1 + (size_t)scale_stride;
        const int8_t *s3 = s2 + (size_t)scale_stride;
        int32_t bs0 = 0, bs1 = 0, bs2 = 0, bs3 = 0;

        for (int g = 0; g < 16; ++g) {
            __m128i vp = _mm_loadu_si128((const __m128i *)(v + g * 16));
            __m128i w0 = _mm_loadu_si128((const __m128i *)(q0 + g * 16));
            __m128i w1 = _mm_loadu_si128((const __m128i *)(q1 + g * 16));
            __m128i w2 = _mm_loadu_si128((const __m128i *)(q2 + g * 16));
            __m128i w3 = _mm_loadu_si128((const __m128i *)(q3 + g * 16));
            int32_t d0 = q6k_dot16_avx2(w0, vp);
            int32_t d1 = q6k_dot16_avx2(w1, vp);
            int32_t d2 = q6k_dot16_avx2(w2, vp);
            int32_t d3 = q6k_dot16_avx2(w3, vp);
            bs0 += d0 * (int32_t)s0[g];
            bs1 += d1 * (int32_t)s1[g];
            bs2 += d2 * (int32_t)s2[g];
            bs3 += d3 * (int32_t)s3[g];
        }

        acc0 += (float)bs0 * d[b];
        acc1 += (float)bs1 * d[(size_t)d_stride + b];
        acc2 += (float)bs2 * d[(size_t)d_stride * 2u + b];
        acc3 += (float)bs3 * d[(size_t)d_stride * 3u + b];
    }

    out[0] = acc0 * vec_scale;
    out[1] = acc1 * vec_scale;
    out[2] = acc2 * vec_scale;
    out[3] = acc3 * vec_scale;
    return 0;
}

BITNET_TARGET_AVX2
int bitnet_q6k_dot_product_q8_compact_8_avx2(const int8_t *q8, const int8_t *scales,
                                                const float *d, int row_stride,
                                                int scale_stride, int d_stride,
                                                int blocks_per_row, const int8_t *qvec,
                                                float vec_scale, float out[8]) {
    if (bitnet_q6k_dot_product_q8_compact_4_avx2(q8, scales, d, row_stride, scale_stride,
                                                   d_stride, blocks_per_row, qvec, vec_scale, out) != 0) {
        return -1;
    }
    return bitnet_q6k_dot_product_q8_compact_4_avx2(q8 + 4 * (size_t)row_stride,
                                                       scales + 4 * (size_t)scale_stride,
                                                       d + 4 * (size_t)d_stride,
                                                       row_stride, scale_stride, d_stride,
                                                       blocks_per_row, qvec, vec_scale, out + 4);
}

/* =====================================================================
 * AVX-VNNI tier
 *
 * Uses _mm256_dpbusd_epi32 directly. The 32-byte operand fits two adjacent
 * 16-element groups. Each dpbusd produces 8 int32 lanes (4 products each);
 * we sum them all for the 32-element raw dot, then subtract 32*sum(v) for
 * the bias correction.
 *
 * Since the per-group scale granularity is 16 elements, we still need to
 * extract per-16-element dot products. We do this by computing the dot
 * over 16 elements at a time using a 32-byte dpbusd with the upper 16
 * bytes zeroed (the high half contributes 0 to the result).
 *
 * Alternative: pair adjacent groups (which often share scales in practice)
 * and emit one 32-element dot covering both. We don't do this because the
 * ARM reference applies per-group scales; matching that granularity keeps
 * the result bit-identical.
 * ===================================================================== */

/* Pack two adjacent 16-element groups (with different scales) into a
 * single 32-byte dpbusd. The 32-byte dot produces 8 int32 lanes — low 4
 * are partial sums for group N, high 4 for group N+1. We horizontally sum
 * each half separately, apply per-group bias correction and per-group
 * scale. Returns the two scaled contributions via out0/out1. */
BITNET_TARGET_AVX_VNNI
static inline void q6k_dot32_pair_avx_vnni(const __m256i w32_i8, const __m256i v32_i8,
                                            float scale0, float scale1,
                                            __m256 *acc) {
    const __m256i bias32 = _mm256_set1_epi8(32);
    const __m256i w32_u8 = _mm256_add_epi8(w32_i8, bias32);

    /* Single 32-byte VNNI dot. Low 4 int32 lanes = group N partial sums;
     * high 4 lanes = group N+1 partial sums. */
    const __m256i raw = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w32_u8, v32_i8);

    /* Per-half horizontal sum. */
    const __m128i lo = _mm256_castsi256_si128(raw);
    const __m128i hi = _mm256_extracti128_si256(raw, 1);
    int32_t raw0 = q6k_hsum4_epi32(lo);
    int32_t raw1 = q6k_hsum4_epi32(hi);

    /* Per-group bias correction: subtract 32 * sum(16 activations).
     * _mm_maddubs_epi16(ones, v_low)  -> 8 int16 partial sums of v_low
     * _mm_madd_epi16(_, ones_16)      -> 4 int32 lanes summing to sum(v_low)
     * Same for v_high. */
    const __m128i ones_8 = _mm_set1_epi8(1);
    const __m128i ones_16 = _mm_set1_epi16(1);
    const __m128i v_low = _mm256_castsi256_si128(v32_i8);
    const __m128i v_high = _mm256_extracti128_si256(v32_i8, 1);
    int32_t sumv0 = q6k_hsum4_epi32(_mm_madd_epi16(_mm_maddubs_epi16(ones_8, v_low), ones_16));
    int32_t sumv1 = q6k_hsum4_epi32(_mm_madd_epi16(_mm_maddubs_epi16(ones_8, v_high), ones_16));

    int32_t dot0 = raw0 - 32 * sumv0;
    int32_t dot1 = raw1 - 32 * sumv1;

    /* FMA-accumulate each contribution into the wide accumulator, weighted
     * by its own per-group scale. */
    *acc = _mm256_fmadd_ps(_mm256_set1_ps((float)dot0),
                           _mm256_set1_ps(scale0), *acc);
    *acc = _mm256_fmadd_ps(_mm256_set1_ps((float)dot1),
                           _mm256_set1_ps(scale1), *acc);
}

/* Pack two adjacent 16-element groups (with different scales) into a
 * single 32-byte dpbusd and return the two per-group dot products
 * (after bias correction) via *dot0 / *dot1. Used by compact paths
 * that accumulate per-block integer sums (scales are int8_t). */
BITNET_TARGET_AVX_VNNI
static inline void q6k_dot32_pair_avx_vnni_int(const __m256i w32_i8, const __m256i v32_i8,
                                                int32_t *dot0, int32_t *dot1) {
    const __m256i bias32 = _mm256_set1_epi8(32);
    const __m256i w32_u8 = _mm256_add_epi8(w32_i8, bias32);

    const __m256i raw = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w32_u8, v32_i8);
    const __m128i lo = _mm256_castsi256_si128(raw);
    const __m128i hi = _mm256_extracti128_si256(raw, 1);
    int32_t raw0 = q6k_hsum4_epi32(lo);
    int32_t raw1 = q6k_hsum4_epi32(hi);

    const __m128i ones_8 = _mm_set1_epi8(1);
    const __m128i ones_16 = _mm_set1_epi16(1);
    const __m128i v_low = _mm256_castsi256_si128(v32_i8);
    const __m128i v_high = _mm256_extracti128_si256(v32_i8, 1);
    int32_t sumv0 = q6k_hsum4_epi32(_mm_madd_epi16(_mm_maddubs_epi16(ones_8, v_low), ones_16));
    int32_t sumv1 = q6k_hsum4_epi32(_mm_madd_epi16(_mm_maddubs_epi16(ones_8, v_high), ones_16));

    *dot0 = raw0 - 32 * sumv0;
    *dot1 = raw1 - 32 * sumv1;
}

/* 16-element dot product using one 32-byte dpbusd (upper 16 bytes zeroed).
 * Retained for code paths that cannot pair adjacent groups (e.g. compact
 * kernels with int8 scales, where block_sum reduces per-block). Currently
 * unused after the AVX-VNNI pair-packing refactor; kept as a building
 * block for future specializations. */
#if 0
static inline int32_t q6k_dot16_avx_vnni(const __m128i w16_i8, const __m128i v16_i8) {
    const __m128i bias = _mm_set1_epi8(32);
    const __m128i w16_u8 = _mm_add_epi8(w16_i8, bias);

    /* Place 16 bytes in low half of 32-byte reg; high half is undefined
     * (cast does not zero it). Load activations into a separate reg. To
     * make the high half contribute 0, we use _mm256_zextsi128_si256 via
     * insert into a zeroed register. */
    __m256i w_full = _mm256_setzero_si256();
    __m256i v_full = _mm256_setzero_si256();
    w_full = _mm256_inserti128_si256(w_full, w16_u8, 0);
    v_full = _mm256_inserti128_si256(v_full, v16_i8, 0);

    __m256i acc32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_full, v_full);
    /* High 4 lanes are 0 (zeroed inputs); low 4 lanes sum 4 products each
     * from the 16-byte input. Horizontal sum gives the 16-element dot. */
    int32_t raw = q6k_hsum8_epi32(acc32);

    /* Bias correction: subtract 32 * sum(16 activations). */
    const __m128i ones_16 = _mm_set1_epi16(1);
    /* _mm_maddubs_epi16(ones, v16): treats ones as uint8 (1) * v as int8 per
     * pair, summed to int16 lanes (8 of them). */
    const __m128i vsum16 = _mm_madd_epi16(_mm_maddubs_epi16(_mm_set1_epi8(1), v16_i8), ones_16);
    int32_t sum_v = q6k_hsum4_epi32(vsum16);

    return raw - 32 * sum_v;
}
#endif

BITNET_TARGET_AVX_VNNI
int bitnet_q6k_dot_product_i8_neon_avx_vnni(const bitnet_q6k_block_t *block,
                                                const int8_t *qvec, float vec_scale,
                                                size_t len, float *out) {
    /* Delegate to AVX2 — the inline i8 path's Q6K-specific bit unpacking
     * does not benefit from VNNI without significant restructuring, and
     * the q8 expanded path is the dominant cost in real workloads. */
    return bitnet_q6k_dot_product_i8_neon_avx2(block, qvec, vec_scale, len, out);
}

BITNET_TARGET_AVX_VNNI
int bitnet_q6k_dot_product_q8_avx_vnni(const int8_t *q8, const float *scales,
                                          int blocks_per_row, const int8_t *qvec,
                                          float vec_scale, float *out) {
    if (q8 == NULL || scales == NULL || qvec == NULL || out == NULL ||
        blocks_per_row <= 0) {
        return -1;
    }

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const float *sc = scales + (size_t)b * 16u;

        /* Process two 16-element groups per dpbusd: doubles VNNI throughput.
         * 16 groups per block -> 8 paired iterations. */
        for (int g = 0; g < 16; g += 2) {
            __m256i w32 = _mm256_loadu_si256((const __m256i *)(q + g * 16));
            __m256i v32 = _mm256_loadu_si256((const __m256i *)(v + g * 16));
            int32_t d0, d1;
            q6k_dot32_pair_avx_vnni_int(w32, v32, &d0, &d1);
            acc += (float)d0 * sc[g] + (float)d1 * sc[g + 1];
        }
    }

    *out = acc * vec_scale;
    return 0;
}

BITNET_TARGET_AVX_VNNI
int bitnet_q6k_dot_product_q8_4_avx_vnni(const int8_t *q8, const float *scales,
                                            int row_stride, int scale_stride,
                                            int blocks_per_row, const int8_t *qvec,
                                            float vec_scale, float out[4]) {
    if (q8 == NULL || scales == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q0 = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *q1 = q0 + (size_t)row_stride;
        const int8_t *q2 = q1 + (size_t)row_stride;
        const int8_t *q3 = q2 + (size_t)row_stride;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const float *s0 = scales + (size_t)b * 16u;
        const float *s1 = s0 + (size_t)scale_stride;
        const float *s2 = s1 + (size_t)scale_stride;
        const float *s3 = s2 + (size_t)scale_stride;

        for (int g = 0; g < 16; g += 2) {
            __m256i v32 = _mm256_loadu_si256((const __m256i *)(v + g * 16));
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(q0 + g * 16));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(q1 + g * 16));
            __m256i w2 = _mm256_loadu_si256((const __m256i *)(q2 + g * 16));
            __m256i w3 = _mm256_loadu_si256((const __m256i *)(q3 + g * 16));
            int32_t d00, d01, d10, d11, d20, d21, d30, d31;
            q6k_dot32_pair_avx_vnni_int(w0, v32, &d00, &d01);
            q6k_dot32_pair_avx_vnni_int(w1, v32, &d10, &d11);
            q6k_dot32_pair_avx_vnni_int(w2, v32, &d20, &d21);
            q6k_dot32_pair_avx_vnni_int(w3, v32, &d30, &d31);
            acc0 += (float)d00 * s0[g] + (float)d01 * s0[g + 1];
            acc1 += (float)d10 * s1[g] + (float)d11 * s1[g + 1];
            acc2 += (float)d20 * s2[g] + (float)d21 * s2[g + 1];
            acc3 += (float)d30 * s3[g] + (float)d31 * s3[g + 1];
        }
    }

    out[0] = acc0 * vec_scale;
    out[1] = acc1 * vec_scale;
    out[2] = acc2 * vec_scale;
    out[3] = acc3 * vec_scale;
    return 0;
}

BITNET_TARGET_AVX_VNNI
int bitnet_q6k_dot_product_q8_8_avx_vnni(const int8_t *q8, const float *scales,
                                            int row_stride, int scale_stride,
                                            int blocks_per_row, const int8_t *qvec,
                                            float vec_scale, float out[8]) {
    if (bitnet_q6k_dot_product_q8_4_avx_vnni(q8, scales, row_stride, scale_stride,
                                                blocks_per_row, qvec, vec_scale, out) != 0) {
        return -1;
    }
    return bitnet_q6k_dot_product_q8_4_avx_vnni(q8 + 4 * (size_t)row_stride,
                                                  scales + 4 * (size_t)scale_stride,
                                                  row_stride, scale_stride,
                                                  blocks_per_row, qvec, vec_scale, out + 4);
}

BITNET_TARGET_AVX_VNNI
int bitnet_q6k_dot_product_q8_compact_avx_vnni(const int8_t *q8, const int8_t *scales,
                                                  const float *d, int blocks_per_row,
                                                  const int8_t *qvec, float vec_scale, float *out) {
    if (q8 == NULL || scales == NULL || d == NULL || qvec == NULL || out == NULL ||
        blocks_per_row <= 0) {
        return -1;
    }

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const int8_t *sc = scales + (size_t)b * 16u;
        int32_t block_sum = 0;

        /* Two 16-element groups per dpbusd. */
        for (int g = 0; g < 16; g += 2) {
            __m256i w32 = _mm256_loadu_si256((const __m256i *)(q + g * 16));
            __m256i v32 = _mm256_loadu_si256((const __m256i *)(v + g * 16));
            int32_t d0, d1;
            q6k_dot32_pair_avx_vnni_int(w32, v32, &d0, &d1);
            block_sum += d0 * (int32_t)sc[g] + d1 * (int32_t)sc[g + 1];
        }

        acc += (float)block_sum * d[b];
    }

    *out = acc * vec_scale;
    return 0;
}

BITNET_TARGET_AVX_VNNI
int bitnet_q6k_dot_product_q8_compact_4_avx_vnni(const int8_t *q8, const int8_t *scales,
                                                     const float *d, int row_stride,
                                                     int scale_stride, int d_stride,
                                                     int blocks_per_row, const int8_t *qvec,
                                                     float vec_scale, float out[4]) {
    if (q8 == NULL || scales == NULL || d == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || d_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q0 = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *q1 = q0 + (size_t)row_stride;
        const int8_t *q2 = q1 + (size_t)row_stride;
        const int8_t *q3 = q2 + (size_t)row_stride;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const int8_t *s0 = scales + (size_t)b * 16u;
        const int8_t *s1 = s0 + (size_t)scale_stride;
        const int8_t *s2 = s1 + (size_t)scale_stride;
        const int8_t *s3 = s2 + (size_t)scale_stride;
        int32_t bs0 = 0, bs1 = 0, bs2 = 0, bs3 = 0;

        for (int g = 0; g < 16; g += 2) {
            __m256i v32 = _mm256_loadu_si256((const __m256i *)(v + g * 16));
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(q0 + g * 16));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(q1 + g * 16));
            __m256i w2 = _mm256_loadu_si256((const __m256i *)(q2 + g * 16));
            __m256i w3 = _mm256_loadu_si256((const __m256i *)(q3 + g * 16));
            int32_t d00, d01;
            int32_t d10, d11;
            int32_t d20, d21;
            int32_t d30, d31;
            q6k_dot32_pair_avx_vnni_int(w0, v32, &d00, &d01);
            q6k_dot32_pair_avx_vnni_int(w1, v32, &d10, &d11);
            q6k_dot32_pair_avx_vnni_int(w2, v32, &d20, &d21);
            q6k_dot32_pair_avx_vnni_int(w3, v32, &d30, &d31);
            bs0 += d00 * (int32_t)s0[g] + d01 * (int32_t)s0[g + 1];
            bs1 += d10 * (int32_t)s1[g] + d11 * (int32_t)s1[g + 1];
            bs2 += d20 * (int32_t)s2[g] + d21 * (int32_t)s2[g + 1];
            bs3 += d30 * (int32_t)s3[g] + d31 * (int32_t)s3[g + 1];
        }

        acc0 += (float)bs0 * d[b];
        acc1 += (float)bs1 * d[(size_t)d_stride + b];
        acc2 += (float)bs2 * d[(size_t)d_stride * 2u + b];
        acc3 += (float)bs3 * d[(size_t)d_stride * 3u + b];
    }

    out[0] = acc0 * vec_scale;
    out[1] = acc1 * vec_scale;
    out[2] = acc2 * vec_scale;
    out[3] = acc3 * vec_scale;
    return 0;
}

BITNET_TARGET_AVX_VNNI
int bitnet_q6k_dot_product_q8_compact_8_avx_vnni(const int8_t *q8, const int8_t *scales,
                                                     const float *d, int row_stride,
                                                     int scale_stride, int d_stride,
                                                     int blocks_per_row, const int8_t *qvec,
                                                     float vec_scale, float out[8]) {
    if (bitnet_q6k_dot_product_q8_compact_4_avx_vnni(q8, scales, d, row_stride, scale_stride,
                                                        d_stride, blocks_per_row, qvec, vec_scale, out) != 0) {
        return -1;
    }
    return bitnet_q6k_dot_product_q8_compact_4_avx_vnni(q8 + 4 * (size_t)row_stride,
                                                          scales + 4 * (size_t)scale_stride,
                                                          d + 4 * (size_t)d_stride,
                                                          row_stride, scale_stride, d_stride,
                                                          blocks_per_row, qvec, vec_scale, out + 4);
}

/* =====================================================================
 * AVX512-VNNI tier — delegates to AVX-VNNI in Phase 4. Phase 7 will
 * specialize using _mm512_dpbusd_epi32 (64-byte operands, 16 int32 lanes).
 * ===================================================================== */

BITNET_TARGET_AVX512_VNNI
int bitnet_q6k_dot_product_i8_neon_avx512_vnni(const bitnet_q6k_block_t *block,
                                                    const int8_t *qvec, float vec_scale,
                                                    size_t len, float *out) {
    return bitnet_q6k_dot_product_i8_neon_avx_vnni(block, qvec, vec_scale, len, out);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_q6k_dot_product_q8_avx512_vnni(const int8_t *q8, const float *scales,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float *out) {
    return bitnet_q6k_dot_product_q8_avx_vnni(q8, scales, blocks_per_row, qvec, vec_scale, out);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_q6k_dot_product_q8_4_avx512_vnni(const int8_t *q8, const float *scales,
                                               int row_stride, int scale_stride,
                                               int blocks_per_row, const int8_t *qvec,
                                               float vec_scale, float out[4]) {
    return bitnet_q6k_dot_product_q8_4_avx_vnni(q8, scales, row_stride, scale_stride,
                                                   blocks_per_row, qvec, vec_scale, out);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_q6k_dot_product_q8_8_avx512_vnni(const int8_t *q8, const float *scales,
                                               int row_stride, int scale_stride,
                                               int blocks_per_row, const int8_t *qvec,
                                               float vec_scale, float out[8]) {
    return bitnet_q6k_dot_product_q8_8_avx_vnni(q8, scales, row_stride, scale_stride,
                                                   blocks_per_row, qvec, vec_scale, out);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_q6k_dot_product_q8_compact_avx512_vnni(const int8_t *q8, const int8_t *scales,
                                                     const float *d, int blocks_per_row,
                                                     const int8_t *qvec, float vec_scale, float *out) {
    return bitnet_q6k_dot_product_q8_compact_avx_vnni(q8, scales, d, blocks_per_row,
                                                         qvec, vec_scale, out);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_q6k_dot_product_q8_compact_4_avx512_vnni(const int8_t *q8, const int8_t *scales,
                                                        const float *d, int row_stride,
                                                        int scale_stride, int d_stride,
                                                        int blocks_per_row, const int8_t *qvec,
                                                        float vec_scale, float out[4]) {
    return bitnet_q6k_dot_product_q8_compact_4_avx_vnni(q8, scales, d, row_stride,
                                                           scale_stride, d_stride,
                                                           blocks_per_row, qvec, vec_scale, out);
}

BITNET_TARGET_AVX512_VNNI
int bitnet_q6k_dot_product_q8_compact_8_avx512_vnni(const int8_t *q8, const int8_t *scales,
                                                        const float *d, int row_stride,
                                                        int scale_stride, int d_stride,
                                                        int blocks_per_row, const int8_t *qvec,
                                                        float vec_scale, float out[8]) {
    return bitnet_q6k_dot_product_q8_compact_8_avx_vnni(q8, scales, d, row_stride,
                                                           scale_stride, d_stride,
                                                           blocks_per_row, qvec, vec_scale, out);
}

#else  /* !defined(__x86_64__) && !defined(_M_X64) */

/* Non-x86 stubs: required so kernel_registry.c can reference these symbols
 * under #if defined(__x86_64__). On ARM the registry block is gated out
 * too, so these stubs never get linked. */

/* Stub parameters are intentionally unused. Suppress -Wunused-parameter
 * locally so the macros stay simple without per-parameter (void) casts
 * (which would otherwise trip -Wunused-value on comma-separated args). */
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define Q6K_STUB(NAME, SIG)                              \
    int NAME SIG {                                       \
        return -1;                                       \
    }

Q6K_STUB(bitnet_q6k_dot_product_i8_neon_avx2,
         (const bitnet_q6k_block_t *block, const int8_t *qvec, float vec_scale,
          size_t len, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_avx2,
         (const int8_t *q8, const float *scales, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_4_avx2,
         (const int8_t *q8, const float *scales, int row_stride, int scale_stride,
          int blocks_per_row, const int8_t *qvec, float vec_scale, float out[4]))
Q6K_STUB(bitnet_q6k_dot_product_q8_8_avx2,
         (const int8_t *q8, const float *scales, int row_stride, int scale_stride,
          int blocks_per_row, const int8_t *qvec, float vec_scale, float out[8]))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_avx2,
         (const int8_t *q8, const int8_t *scales, const float *d, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_4_avx2,
         (const int8_t *q8, const int8_t *scales, const float *d, int row_stride,
          int scale_stride, int d_stride, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float out[4]))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_8_avx2,
         (const int8_t *q8, const int8_t *scales, const float *d, int row_stride,
          int scale_stride, int d_stride, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float out[8]))

Q6K_STUB(bitnet_q6k_dot_product_i8_neon_avx_vnni,
         (const bitnet_q6k_block_t *block, const int8_t *qvec, float vec_scale,
          size_t len, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_avx_vnni,
         (const int8_t *q8, const float *scales, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_4_avx_vnni,
         (const int8_t *q8, const float *scales, int row_stride, int scale_stride,
          int blocks_per_row, const int8_t *qvec, float vec_scale, float out[4]))
Q6K_STUB(bitnet_q6k_dot_product_q8_8_avx_vnni,
         (const int8_t *q8, const float *scales, int row_stride, int scale_stride,
          int blocks_per_row, const int8_t *qvec, float vec_scale, float out[8]))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_avx_vnni,
         (const int8_t *q8, const int8_t *scales, const float *d, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_4_avx_vnni,
         (const int8_t *q8, const int8_t *scales, const float *d, int row_stride,
          int scale_stride, int d_stride, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float out[4]))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_8_avx_vnni,
         (const int8_t *q8, const int8_t *scales, const float *d, int row_stride,
          int scale_stride, int d_stride, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float out[8]))

Q6K_STUB(bitnet_q6k_dot_product_i8_neon_avx512_vnni,
         (const bitnet_q6k_block_t *block, const int8_t *qvec, float vec_scale,
          size_t len, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_avx512_vnni,
         (const int8_t *q8, const float *scales, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_4_avx512_vnni,
         (const int8_t *q8, const float *scales, int row_stride, int scale_stride,
          int blocks_per_row, const int8_t *qvec, float vec_scale, float out[4]))
Q6K_STUB(bitnet_q6k_dot_product_q8_8_avx512_vnni,
         (const int8_t *q8, const float *scales, int row_stride, int scale_stride,
          int blocks_per_row, const int8_t *qvec, float vec_scale, float out[8]))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_avx512_vnni,
         (const int8_t *q8, const int8_t *scales, const float *d, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float *out))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_4_avx512_vnni,
         (const int8_t *q8, const int8_t *scales, const float *d, int row_stride,
          int scale_stride, int d_stride, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float out[4]))
Q6K_STUB(bitnet_q6k_dot_product_q8_compact_8_avx512_vnni,
         (const int8_t *q8, const int8_t *scales, const float *d, int row_stride,
          int scale_stride, int d_stride, int blocks_per_row,
          const int8_t *qvec, float vec_scale, float out[8]))

#undef Q6K_STUB

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif  /* defined(__x86_64__) || defined(_M_X64) */
