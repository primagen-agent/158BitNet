#include "ops_x86.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>
#include <math.h>
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define BITNET_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define BITNET_TARGET_AVX_VNNI __attribute__((target("avx2,fma,avxvnni")))
#define BITNET_TARGET_AVX512_VNNI __attribute__((target("avx512f,avx512bw,avx512vnni,avx512dq")))
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

/* ========== SiLU family (elementwise) ==========
 * Pattern: load 8 (AVX2) or 16 (AVX512) floats, compute
 *   sig(x) = 1 / (1 + exp(-x)); result = x * sig(x)         (silu)
 *   result = x * sig(x) * up                                (silu_mul)
 *   + track max |result| across lanes                       (silu_mul_max_abs)
 * The NEON polynomial exp approximation gives no accuracy advantage over
 * libm `expf` for x86, and softmax already uses libm for accuracy. We do
 * the same here: spill to a tmp array, call expf per lane, reload. */

BITNET_TARGET_AVX2
void bitnet_silu_avx2(float *x, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        for (int j = 0; j < 8; ++j) {
            float sig = 1.0f / (1.0f + expf(-tmp[j]));
            tmp[j] = tmp[j] * sig;
        }
        _mm256_storeu_ps(x + i, _mm256_loadu_ps(tmp));
    }
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        x[i] = x[i] * sig;
    }
}

BITNET_TARGET_AVX2
void bitnet_silu_mul_avx2(float *gate, const float *up, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 gv = _mm256_loadu_ps(gate + i);
        float gtmp[8];
        _mm256_storeu_ps(gtmp, gv);
        __m256 uv = _mm256_loadu_ps(up + i);
        float utmp[8];
        _mm256_storeu_ps(utmp, uv);
        float out[8];
        for (int j = 0; j < 8; ++j) {
            float sig = 1.0f / (1.0f + expf(-gtmp[j]));
            out[j] = gtmp[j] * sig * utmp[j];
        }
        _mm256_storeu_ps(gate + i, _mm256_loadu_ps(out));
    }
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        gate[i] = gate[i] * sig * up[i];
    }
}

BITNET_TARGET_AVX2
float bitnet_silu_mul_max_abs_avx2(float *gate, const float *up, int n) {
    int i = 0;
    __m256 max_vec = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 gv = _mm256_loadu_ps(gate + i);
        float gtmp[8];
        _mm256_storeu_ps(gtmp, gv);
        __m256 uv = _mm256_loadu_ps(up + i);
        float utmp[8];
        _mm256_storeu_ps(utmp, uv);
        float out[8];
        for (int j = 0; j < 8; ++j) {
            float sig = 1.0f / (1.0f + expf(-gtmp[j]));
            out[j] = gtmp[j] * sig * utmp[j];
        }
        __m256 result = _mm256_loadu_ps(out);
        _mm256_storeu_ps(gate + i, result);
        __m256 abs_result = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), result);
        max_vec = _mm256_max_ps(max_vec, abs_result);
    }
    /* horizontal max */
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 s = _mm_max_ps(hi, lo);
    s = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
    s = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_abs = _mm_cvtss_f32(s);
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        float result = gate[i] * sig * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

BITNET_TARGET_AVX2
float bitnet_relu2_mul_max_abs_avx2(float *gate, const float *up, int n) {
    int i = 0;
    __m256 max_vec = _mm256_setzero_ps();
    __m256 zero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 gv = _mm256_max_ps(_mm256_loadu_ps(gate + i), zero);
        __m256 uv = _mm256_loadu_ps(up + i);
        __m256 sq = _mm256_mul_ps(gv, gv);
        __m256 result = _mm256_mul_ps(sq, uv);
        _mm256_storeu_ps(gate + i, result);
        __m256 abs_result = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), result);
        max_vec = _mm256_max_ps(max_vec, abs_result);
    }
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 s = _mm_max_ps(hi, lo);
    s = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
    s = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_abs = _mm_cvtss_f32(s);
    for (; i < n; ++i) {
        float g = gate[i] > 0.0f ? gate[i] : 0.0f;
        float result = g * g * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

BITNET_TARGET_AVX2
void bitnet_residual_add_avx2(float *out, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

/* Elementwise float ops have no int dot-product, so AVX-VNNI delegates to AVX2. */
BITNET_TARGET_AVX_VNNI
void bitnet_silu_avx_vnni(float *x, int n) {
    bitnet_silu_avx2(x, n);
}

BITNET_TARGET_AVX_VNNI
void bitnet_silu_mul_avx_vnni(float *gate, const float *up, int n) {
    bitnet_silu_mul_avx2(gate, up, n);
}

BITNET_TARGET_AVX_VNNI
float bitnet_silu_mul_max_abs_avx_vnni(float *gate, const float *up, int n) {
    return bitnet_silu_mul_max_abs_avx2(gate, up, n);
}

BITNET_TARGET_AVX_VNNI
float bitnet_relu2_mul_max_abs_avx_vnni(float *gate, const float *up, int n) {
    return bitnet_relu2_mul_max_abs_avx2(gate, up, n);
}

BITNET_TARGET_AVX_VNNI
void bitnet_residual_add_avx_vnni(float *out, const float *a, const float *b, int n) {
    bitnet_residual_add_avx2(out, a, b, n);
}

/* AVX512: 16-lane versions. SiLU family still spills to tmp for libm expf. */
BITNET_TARGET_AVX512_VNNI
void bitnet_silu_avx512_vnni(float *x, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        float tmp[16];
        _mm512_storeu_ps(tmp, v);
        for (int j = 0; j < 16; ++j) {
            float sig = 1.0f / (1.0f + expf(-tmp[j]));
            tmp[j] = tmp[j] * sig;
        }
        _mm512_storeu_ps(x + i, _mm512_loadu_ps(tmp));
    }
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        x[i] = x[i] * sig;
    }
}

BITNET_TARGET_AVX512_VNNI
void bitnet_silu_mul_avx512_vnni(float *gate, const float *up, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 gv = _mm512_loadu_ps(gate + i);
        __m512 uv = _mm512_loadu_ps(up + i);
        float gtmp[16], utmp[16], out[16];
        _mm512_storeu_ps(gtmp, gv);
        _mm512_storeu_ps(utmp, uv);
        for (int j = 0; j < 16; ++j) {
            float sig = 1.0f / (1.0f + expf(-gtmp[j]));
            out[j] = gtmp[j] * sig * utmp[j];
        }
        _mm512_storeu_ps(gate + i, _mm512_loadu_ps(out));
    }
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        gate[i] = gate[i] * sig * up[i];
    }
}

BITNET_TARGET_AVX512_VNNI
float bitnet_silu_mul_max_abs_avx512_vnni(float *gate, const float *up, int n) {
    int i = 0;
    __m512 max_vec = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 gv = _mm512_loadu_ps(gate + i);
        __m512 uv = _mm512_loadu_ps(up + i);
        float gtmp[16], utmp[16], out[16];
        _mm512_storeu_ps(gtmp, gv);
        _mm512_storeu_ps(utmp, uv);
        for (int j = 0; j < 16; ++j) {
            float sig = 1.0f / (1.0f + expf(-gtmp[j]));
            out[j] = gtmp[j] * sig * utmp[j];
        }
        __m512 result = _mm512_loadu_ps(out);
        _mm512_storeu_ps(gate + i, result);
        __m512 abs_result = _mm512_and_ps(_mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), result);
        max_vec = _mm512_max_ps(max_vec, abs_result);
    }
    float max_abs = _mm512_reduce_max_ps(max_vec);
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        float result = gate[i] * sig * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

BITNET_TARGET_AVX512_VNNI
float bitnet_relu2_mul_max_abs_avx512_vnni(float *gate, const float *up, int n) {
    int i = 0;
    __m512 max_vec = _mm512_setzero_ps();
    __m512 zero = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 gv = _mm512_max_ps(_mm512_loadu_ps(gate + i), zero);
        __m512 uv = _mm512_loadu_ps(up + i);
        __m512 sq = _mm512_mul_ps(gv, gv);
        __m512 result = _mm512_mul_ps(sq, uv);
        _mm512_storeu_ps(gate + i, result);
        __m512 abs_result = _mm512_and_ps(_mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), result);
        max_vec = _mm512_max_ps(max_vec, abs_result);
    }
    float max_abs = _mm512_reduce_max_ps(max_vec);
    for (; i < n; ++i) {
        float g = gate[i] > 0.0f ? gate[i] : 0.0f;
        float result = g * g * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

BITNET_TARGET_AVX512_VNNI
void bitnet_residual_add_avx512_vnni(float *out, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_add_ps(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

/* ========== Softmax ==========
 * Three-pass algorithm: max-scan, exp+sum, divide. Matches the NEON softmax
 * approach (ops.c): spill to a tmp array and call libm expf per lane for
 * accuracy — softmax is precision-sensitive. */

BITNET_TARGET_AVX2
void bitnet_softmax_avx2(float *x, int n) {
    if (n <= 0) return;

    /* Pass 1: max scan */
    __m256 max_vec = _mm256_set1_ps(-1e30f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        max_vec = _mm256_max_ps(max_vec, v);
    }
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(m);
    for (; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Pass 2: exp(x - max) + sum */
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 mbroadcast = _mm256_set1_ps(max_val);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), mbroadcast);
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        for (int j = 0; j < 8; ++j) tmp[j] = expf(tmp[j]);
        __m256 ev = _mm256_loadu_ps(tmp);
        _mm256_storeu_ps(x + i, ev);
        sum_vec = _mm256_add_ps(sum_vec, ev);
    }
    hi = _mm256_extractf128_ps(sum_vec, 1);
    lo = _mm256_castps256_ps128(sum_vec);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Pass 3: divide by sum */
    float inv_sum = 1.0f / sum;
    __m256 inv_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_mul_ps(v, inv_v);
        _mm256_storeu_ps(x + i, v);
    }
    for (; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

/* Softmax has no int dot-product; AVX-VNNI delegates to AVX2. */
BITNET_TARGET_AVX_VNNI
void bitnet_softmax_avx_vnni(float *x, int n) {
    bitnet_softmax_avx2(x, n);
}

BITNET_TARGET_AVX512_VNNI
void bitnet_softmax_avx512_vnni(float *x, int n) {
    if (n <= 0) return;

    /* Pass 1: max scan */
    __m512 max_vec = _mm512_set1_ps(-1e30f);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        max_vec = _mm512_max_ps(max_vec, v);
    }
    float max_val = _mm512_reduce_max_ps(max_vec);
    for (; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Pass 2: exp(x - max) + sum */
    __m512 sum_vec = _mm512_setzero_ps();
    __m512 mbroadcast = _mm512_set1_ps(max_val);
    i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x + i), mbroadcast);
        float tmp[16];
        _mm512_storeu_ps(tmp, v);
        for (int j = 0; j < 16; ++j) tmp[j] = expf(tmp[j]);
        __m512 ev = _mm512_loadu_ps(tmp);
        _mm512_storeu_ps(x + i, ev);
        sum_vec = _mm512_add_ps(sum_vec, ev);
    }
    float sum = _mm512_reduce_add_ps(sum_vec);
    for (; i < n; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Pass 3: divide by sum */
    float inv_sum = 1.0f / sum;
    __m512 inv_v = _mm512_set1_ps(inv_sum);
    i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        v = _mm512_mul_ps(v, inv_v);
        _mm512_storeu_ps(x + i, v);
    }
    for (; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

/* ========== RoPE (rotary position embedding) ==========
 * Processes pairs of floats (idx0, idx1) with shared cos/sin per pair as a
 * 2D rotation: (x0', x1') = (x0*c - x1*s, x0*s + x1*c).
 *
 * A true SIMD version needs a permutation to separate pair elements into
 * their own lanes so the multiply-add pattern lines up. Rope is rarely the
 * decode bottleneck (small dim count per token), so for the first cut we
 * fall back to scalar-in-AVX2-context: correct, and vectorization can be
 * revisited if benchmarks show it matters. */

BITNET_TARGET_AVX2
void bitnet_rope_apply_avx2(float *x, int n_heads, int head_dim, int rope_dim,
                             const float *rope_cos, const float *rope_sin) {
    if (x == NULL || rope_cos == NULL || rope_sin == NULL ||
        n_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
        return;
    }
    for (int h = 0; h < n_heads; ++h) {
        for (int j = 0; j < rope_dim / 2; ++j) {
            int idx0 = h * head_dim + 2 * j;
            int idx1 = h * head_dim + 2 * j + 1;
            float c = rope_cos[j];
            float s = rope_sin[j];
            float x0 = x[idx0];
            float x1 = x[idx1];
            x[idx0] = x0 * c - x1 * s;
            x[idx1] = x0 * s + x1 * c;
        }
    }
}

BITNET_TARGET_AVX_VNNI
void bitnet_rope_apply_avx_vnni(float *x, int n_heads, int head_dim, int rope_dim,
                                 const float *rope_cos, const float *rope_sin) {
    bitnet_rope_apply_avx2(x, n_heads, head_dim, rope_dim, rope_cos, rope_sin);
}

BITNET_TARGET_AVX512_VNNI
void bitnet_rope_apply_avx512_vnni(float *x, int n_heads, int head_dim, int rope_dim,
                                    const float *rope_cos, const float *rope_sin) {
    /* Same scalar-fallback rationale as the AVX2 variant. */
    if (x == NULL || rope_cos == NULL || rope_sin == NULL ||
        n_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
        return;
    }
    for (int h = 0; h < n_heads; ++h) {
        for (int j = 0; j < rope_dim / 2; ++j) {
            int idx0 = h * head_dim + 2 * j;
            int idx1 = h * head_dim + 2 * j + 1;
            float c = rope_cos[j];
            float s = rope_sin[j];
            float x0 = x[idx0];
            float x1 = x[idx1];
            x[idx0] = x0 * c - x1 * s;
            x[idx1] = x0 * s + x1 * c;
        }
    }
}

#else /* !__x86_64__ && !_M_X64 */

/* Non-x86 builds (e.g. Apple ARM64 host) compile this translation unit to an
 * empty object: the symbols live in the x86 dispatch tables, which are also
 * excluded on non-x86. See kernel_registry.c. */
typedef int bitnet_make_ops_x86_nonempty_t;

#endif
