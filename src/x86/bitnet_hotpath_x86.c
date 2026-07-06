#include "bitnet_hotpath_x86.h"

#include "../ops.h"
#include "../quant_tq2_0.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
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

/* ------------------------------------------------------------------ */
/* quantize_f32_to_i8                                                 */
/* ------------------------------------------------------------------ */
BITNET_TARGET_AVX2
static float bitnet_quantize_f32_to_i8_avx2_inner(const float *src, int n, int8_t *dst) {
    if (n <= 0) return 0.0f;

    __m256 max_vec = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 a = _mm256_andnot_ps(sign_mask, v);
        max_vec = _mm256_max_ps(max_vec, a);
    }
    /* Horizontal max of the 8 lanes. */
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_abs = _mm_cvtss_f32(m);
    for (; i < n; ++i) {
        float v = fabsf(src[i]);
        if (v > max_abs) max_abs = v;
    }

    if (max_abs == 0.0f) {
        memset(dst, 0, (size_t)n * sizeof(*dst));
        return 0.0f;
    }

    float inv_scale = 127.0f / max_abs;
    __m256 inv_v = _mm256_set1_ps(inv_scale);
    __m256 clamp_lo = _mm256_set1_ps(-127.5f);
    __m256 clamp_hi = _mm256_set1_ps(127.5f);
    /* Round-half-away-from-zero: copysignf(roundf(x), x). Use the trick
     * add 0.5 then truncate toward zero for positive; subtract 0.5 then
     * ceil toward zero for negative. Easier: cvtt rounds toward zero, so
     * add 0.5 for positive and -0.5 for negative. Use blend via sign bit
     * emulation. */
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 scaled = _mm256_mul_ps(v, inv_v);
        /* signed round-half-away: sign = copysign(0.5, scaled); rounded = trunc(scaled + sign) */
        __m256 sign = _mm256_or_ps(
            _mm256_and_ps(scaled, sign_mask),
            _mm256_set1_ps(0.5f));
        __m256 biased = _mm256_add_ps(scaled, sign);
        /* Truncate toward zero (cvtt). */
        __m256i q = _mm256_cvttps_epi32(biased);
        q = _mm256_max_epi32(q, _mm256_set1_epi32(-127));
        q = _mm256_min_epi32(q, _mm256_set1_epi32(127));
        /* Pack int32 -> int8 via two int16 passes. */
        __m128i lo128 = _mm256_castsi256_si128(q);
        __m128i hi128 = _mm256_extracti128_si256(q, 1);
        __m128i packed16 = _mm_packs_epi32(lo128, hi128);
        __m128i packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(dst + i), packed8);
    }
    for (; i < n; ++i) {
        float scaled = src[i] * inv_scale;
        int q = (int)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        dst[i] = (int8_t)q;
    }
    return max_abs / 127.0f;
}

BITNET_TARGET_AVX2
float bitnet_quantize_f32_to_i8_avx2(const float *src, int n, int8_t *dst) {
    return bitnet_quantize_f32_to_i8_avx2_inner(src, n, dst);
}
BITNET_TARGET_AVX_VNNI
float bitnet_quantize_f32_to_i8_avx_vnni(const float *src, int n, int8_t *dst) {
    return bitnet_quantize_f32_to_i8_avx2_inner(src, n, dst);
}
BITNET_TARGET_AVX512_VNNI
float bitnet_quantize_f32_to_i8_avx512_vnni(const float *src, int n, int8_t *dst) {
    return bitnet_quantize_f32_to_i8_avx2_inner(src, n, dst);
}

/* ------------------------------------------------------------------ */
/* rms_norm_quant_tq2_i8                                              */
/* ------------------------------------------------------------------ */
BITNET_TARGET_AVX2
static int bitnet_rms_norm_quant_tq2_i8_avx2_inner(float *dst, const float *src,
                                                    const float *weight, int n,
                                                    int8_t *qvec, float *scale,
                                                    int32_t *block_bsums, float eps) {
    if (dst == NULL || src == NULL || weight == NULL || qvec == NULL ||
        scale == NULL || n <= 0) {
        return -1;
    }

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

    const float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);

    __m256 inv_v = _mm256_set1_ps(inv_rms);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 max_vec = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        __m256 out = _mm256_mul_ps(_mm256_mul_ps(v, inv_v), w);
        _mm256_storeu_ps(dst + i, out);
        __m256 a = _mm256_andnot_ps(sign_mask, out);
        max_vec = _mm256_max_ps(max_vec, a);
    }
    __m128 mhi = _mm256_extractf128_ps(max_vec, 1);
    __m128 mlo = _mm256_castps256_ps128(max_vec);
    __m128 m = _mm_max_ps(mhi, mlo);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_abs = _mm_cvtss_f32(m);
    for (; i < n; ++i) {
        float out = src[i] * inv_rms * weight[i];
        float a = fabsf(out);
        dst[i] = out;
        if (a > max_abs) max_abs = a;
    }

    return bitnet_tq2_0_quantize_vec_i8_known_max(dst, n, qvec, scale,
                                                   block_bsums, max_abs);
}

BITNET_TARGET_AVX2
int bitnet_rms_norm_quant_tq2_i8_avx2(float *dst, const float *src, const float *weight,
                                       int n, int8_t *qvec, float *scale,
                                       int32_t *block_bsums, float eps) {
    return bitnet_rms_norm_quant_tq2_i8_avx2_inner(dst, src, weight, n, qvec,
                                                    scale, block_bsums, eps);
}
BITNET_TARGET_AVX_VNNI
int bitnet_rms_norm_quant_tq2_i8_avx_vnni(float *dst, const float *src, const float *weight,
                                          int n, int8_t *qvec, float *scale,
                                          int32_t *block_bsums, float eps) {
    return bitnet_rms_norm_quant_tq2_i8_avx2_inner(dst, src, weight, n, qvec,
                                                    scale, block_bsums, eps);
}
BITNET_TARGET_AVX512_VNNI
int bitnet_rms_norm_quant_tq2_i8_avx512_vnni(float *dst, const float *src, const float *weight,
                                             int n, int8_t *qvec, float *scale,
                                             int32_t *block_bsums, float eps) {
    return bitnet_rms_norm_quant_tq2_i8_avx2_inner(dst, src, weight, n, qvec,
                                                    scale, block_bsums, eps);
}

/* ------------------------------------------------------------------ */
/* residual_add_scaled                                                */
/* ------------------------------------------------------------------ */
BITNET_TARGET_AVX2
static void bitnet_residual_add_scaled_avx2_inner(float *out, const float *a,
                                                   const float *b, float b_scale, int n) {
    if (b_scale == 1.0f) {
        bitnet_residual_add(out, a, b, n);
        return;
    }
    __m256 sv = _mm256_set1_ps(b_scale);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(bv, sv, av));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i] * b_scale;
    }
}

BITNET_TARGET_AVX2
void bitnet_residual_add_scaled_avx2(float *out, const float *a, const float *b,
                                     float b_scale, int n) {
    bitnet_residual_add_scaled_avx2_inner(out, a, b, b_scale, n);
}
BITNET_TARGET_AVX_VNNI
void bitnet_residual_add_scaled_avx_vnni(float *out, const float *a, const float *b,
                                         float b_scale, int n) {
    bitnet_residual_add_scaled_avx2_inner(out, a, b, b_scale, n);
}
BITNET_TARGET_AVX512_VNNI
void bitnet_residual_add_scaled_avx512_vnni(float *out, const float *a, const float *b,
                                            float b_scale, int n) {
    bitnet_residual_add_scaled_avx2_inner(out, a, b, b_scale, n);
}

/* ------------------------------------------------------------------ */
/* dot_i8  (signed int8 x signed int8 dot product)                   */
/*                                                                    */
/* AVX2 has no unsigned/unsigned-byte SDOT, and maddubs is            */
/* uint8xint8. For signed x signed, sign-extend each 128-bit half     */
/* to int16 with vpmovsxbw, then madd_epi16 to int32. 32 bytes per    */
/* iteration -> 8 int32 lanes, h-summed at the end.                   */
/* ------------------------------------------------------------------ */
BITNET_TARGET_AVX2
static int bitnet_dot_i8_avx2_inner(const int8_t *a, const int8_t *b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m128i a_lo = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i b_lo = _mm_loadu_si128((const __m128i *)(b + i));
        __m256i a0 = _mm256_cvtepi8_epi16(a_lo);
        __m256i b0 = _mm256_cvtepi8_epi16(b_lo);
        __m256i p0 = _mm256_madd_epi16(a0, b0);
        acc = _mm256_add_epi32(acc, p0);
    }
    /* Horizontal sum across the 8 int32 lanes. Two-pass: extract high
     * 128, add to low 128, then hadd twice. See project commit 39795b5
     * for why we don't use the broken single-pass hadd pattern. */
    __m128i hi128 = _mm256_extracti128_si256(acc, 1);
    __m128i lo128 = _mm256_castsi256_si128(acc);
    __m128i s = _mm_add_epi32(hi128, lo128);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int sum = _mm_cvtsi128_si32(s);
    for (; i < n; ++i) sum += (int)a[i] * (int)b[i];
    return sum;
}

BITNET_TARGET_AVX2
int bitnet_dot_i8_avx2(const int8_t *a, const int8_t *b, int n) {
    return bitnet_dot_i8_avx2_inner(a, b, n);
}
BITNET_TARGET_AVX_VNNI
int bitnet_dot_i8_avx_vnni(const int8_t *a, const int8_t *b, int n) {
    return bitnet_dot_i8_avx2_inner(a, b, n);
}
BITNET_TARGET_AVX512_VNNI
int bitnet_dot_i8_avx512_vnni(const int8_t *a, const int8_t *b, int n) {
    return bitnet_dot_i8_avx2_inner(a, b, n);
}

/* ------------------------------------------------------------------ */
/* accum_i8_scaled  (dst[i] += scale * (float)src[i])                 */
/*                                                                    */
/* Widen int8 -> int32 in 8-lane groups, cvt to fp32, FMA into dst.    */
/* ------------------------------------------------------------------ */
BITNET_TARGET_AVX2
static void bitnet_accum_i8_scaled_avx2_inner(float *dst, const int8_t *src,
                                               float scale, int n) {
    __m256 scale_v = _mm256_set1_ps(scale);
    int i = 0;
    /* Process 32 int8 per iteration: four 8-lane int32 -> fp32 -> FMA.
     * `_mm256_cvtepi8_epi32` reads the low 8 bytes of a 128-bit source
     * and sign-extends into 8 int32 lanes (a full 256-bit register). */
    for (; i + 31 < n; i += 32) {
        __m128i s0 = _mm_loadu_si128((const __m128i *)(src + i));
        __m128i s1 = _mm_loadu_si128((const __m128i *)(src + i + 16));

        __m128i g0 = s0;                                       /* bytes  0..7  */
        __m128i g1 = _mm_srli_si128(s0, 8);                    /* bytes  8..15 */
        __m128i g2 = s1;                                       /* bytes 16..23 */
        __m128i g3 = _mm_srli_si128(s1, 8);                    /* bytes 24..31 */

        __m256 i0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g0));
        __m256 i1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g1));
        __m256 i2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g2));
        __m256 i3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g3));

        __m256 d0 = _mm256_loadu_ps(dst + i);
        __m256 d1 = _mm256_loadu_ps(dst + i + 8);
        __m256 d2 = _mm256_loadu_ps(dst + i + 16);
        __m256 d3 = _mm256_loadu_ps(dst + i + 24);

        _mm256_storeu_ps(dst + i,      _mm256_fmadd_ps(i0, scale_v, d0));
        _mm256_storeu_ps(dst + i + 8,  _mm256_fmadd_ps(i1, scale_v, d1));
        _mm256_storeu_ps(dst + i + 16, _mm256_fmadd_ps(i2, scale_v, d2));
        _mm256_storeu_ps(dst + i + 24, _mm256_fmadd_ps(i3, scale_v, d3));
    }
    for (; i < n; ++i) {
        dst[i] += scale * (float)src[i];
    }
}

BITNET_TARGET_AVX2
void bitnet_accum_i8_scaled_avx2(float *dst, const int8_t *src, float scale, int n) {
    bitnet_accum_i8_scaled_avx2_inner(dst, src, scale, n);
}
BITNET_TARGET_AVX_VNNI
void bitnet_accum_i8_scaled_avx_vnni(float *dst, const int8_t *src, float scale, int n) {
    bitnet_accum_i8_scaled_avx2_inner(dst, src, scale, n);
}
BITNET_TARGET_AVX512_VNNI
void bitnet_accum_i8_scaled_avx512_vnni(float *dst, const int8_t *src, float scale, int n) {
    bitnet_accum_i8_scaled_avx2_inner(dst, src, scale, n);
}

#endif /* __x86_64__ */
