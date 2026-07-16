#include "ops.h"

#include <math.h>
#include <stddef.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ========== RMSNorm ========== */

#if defined(__ARM_NEON)
void bitnet_rms_norm_eps_impl(float *x, const float *weight, int n, float eps) {
    /* Pass 1: sum of squares using NEON */
    float sum = 0.0f;
    int i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        sum_vec = vfmaq_f32(sum_vec, v, v);
    }
    sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }

    float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);

    /* Pass 2: normalize and scale using NEON */
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        v = vmulq_n_f32(v, inv_rms);
        v = vmulq_f32(v, w);
        vst1q_f32(x + i, v);
    }
    for (; i < n; ++i) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}
#else
void bitnet_rms_norm_eps_impl(float *x, const float *weight, int n, float eps) {
    int i = 0;
    float sum = 0.0f;
    float inv_rms = 0.0f;

    for (i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }

    inv_rms = 1.0f / sqrtf(sum / (float)n + eps);

    for (i = 0; i < n; ++i) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}
#endif

void bitnet_rms_norm(float *x, const float *weight, int n) {
    bitnet_rms_norm_eps(x, weight, n, 1e-6f);
}

/* ========== RMSNorm (src/dst separated) ========== */

#if defined(__ARM_NEON)
void bitnet_rms_norm_inplace_eps_impl(float *dst, const float *src, const float *weight, int n, float eps) {
    /* Pass 1: sum of squares using NEON */
    float sum = 0.0f;
    int i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        sum_vec = vfmaq_f32(sum_vec, v, v);
    }
    sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        sum += src[i] * src[i];
    }

    float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);

    /* Pass 2: normalize and scale using NEON, write to dst */
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        float32x4_t w = vld1q_f32(weight + i);
        v = vmulq_n_f32(v, inv_rms);
        v = vmulq_f32(v, w);
        vst1q_f32(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = src[i] * inv_rms * weight[i];
    }
}
#else
void bitnet_rms_norm_inplace_eps_impl(float *dst, const float *src, const float *weight, int n, float eps) {
    int i = 0;
    float sum = 0.0f;
    float inv_rms = 0.0f;

    for (i = 0; i < n; ++i) {
        sum += src[i] * src[i];
    }

    inv_rms = 1.0f / sqrtf(sum / (float)n + eps);

    for (i = 0; i < n; ++i) {
        dst[i] = src[i] * inv_rms * weight[i];
    }
}
#endif

void bitnet_rms_norm_inplace(float *dst, const float *src, const float *weight, int n) {
    bitnet_rms_norm_inplace_eps(dst, src, weight, n, 1e-6f);
}

/* ========== MatMul F32 (reference, not performance-critical) ========== */

void bitnet_matmul_f32(const float *a, const float *b, float *c, int m, int k, int n) {
    int i = 0;
    int j = 0;
    int l = 0;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/* ========== SiLU ========== */

#if defined(__ARM_NEON)
/* NEON polynomial approximation of expf for SiLU sigmoid.
 * Uses minimax Pade-like approximation on [-inf, +inf] range.
 * For sigmoid: sig(x) = 1/(1+exp(-x))
 * We compute exp(-x) using the standard log2e decomposition:
 *   exp(a) = 2^a = 2^(int(a) + frac(a)) = 2^int(a) * 2^frac(a)
 * Then 2^frac(a) is approximated by a degree-3 polynomial. */

static inline float32x4_t neon_exp_approx_f32(float32x4_t x) {
    /* Clamp to [-88, 88] to avoid overflow */
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    /* exp(x) = 2^(x/ln2) = 2^(i + f) where i = floor(x/ln2), f = frac */
    const float inv_ln2 = 1.44269504089f;
    float32x4_t t = vmulq_n_f32(x, inv_ln2);
    /* Round toward -inf for floor */
    int32x4_t ti = vcvtq_s32_f32(vsubq_f32(t, vdupq_n_f32(0.5f)));
    ti = vaddq_s32(ti, vdupq_n_s32(1)); /* correct rounding */
    float32x4_t tf = vsubq_f32(t, vcvtq_f32_s32(ti));

    /* Polynomial for 2^f on [0, 1]: P(f) = 1 + f*(a1 + f*(a2 + f*a3)) */
    const float a1 = 0.69314718056f;
    const float a2 = 0.24022650067f;
    const float a3 = 0.05549526761f;
    float32x4_t pf = vaddq_f32(vmulq_n_f32(tf, a3), vdupq_n_f32(a2));
    pf = vaddq_f32(vmulq_f32(tf, pf), vdupq_n_f32(a1));
    pf = vaddq_f32(vmulq_f32(tf, pf), vdupq_n_f32(1.0f));

    /* 2^i via bit manipulation: (i + 127) << 23 */
    int32x4_t bi = vaddq_s32(ti, vdupq_n_s32(127));
    bi = vshlq_n_s32(bi, 23);
    float32x4_t scale = vreinterpretq_f32_s32(bi);

    return vmulq_f32(pf, scale);
}

void bitnet_silu_impl(float *x, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        /* sig(x) = 1 / (1 + exp(-x)) */
        float32x4_t neg_v = vnegq_f32(v);
        float32x4_t exp_neg = neon_exp_approx_f32(neg_v);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, exp_neg));
        float32x4_t result = vmulq_f32(v, sig);
        vst1q_f32(x + i, result);
    }
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        x[i] = x[i] * sig;
    }
}

void bitnet_silu_mul_impl(float *gate, const float *up, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t gv = vld1q_f32(gate + i);
        float32x4_t uv = vld1q_f32(up + i);
        float32x4_t exp_neg = neon_exp_approx_f32(vnegq_f32(gv));
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, exp_neg));
        vst1q_f32(gate + i, vmulq_f32(vmulq_f32(gv, sig), uv));
    }
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        gate[i] = gate[i] * sig * up[i];
    }
}

float bitnet_silu_mul_max_abs_impl(float *gate, const float *up, int n) {
    int i = 0;
    float max_abs = 0.0f;
    float32x4_t max_vec = vdupq_n_f32(0.0f);

    for (; i + 3 < n; i += 4) {
        float32x4_t gv = vld1q_f32(gate + i);
        float32x4_t uv = vld1q_f32(up + i);
        float32x4_t exp_neg = neon_exp_approx_f32(vnegq_f32(gv));
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, exp_neg));
        float32x4_t result = vmulq_f32(vmulq_f32(gv, sig), uv);
        vst1q_f32(gate + i, result);
        max_vec = vmaxq_f32(max_vec, vabsq_f32(result));
    }
    max_abs = vmaxvq_f32(max_vec);
    for (; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        float result = gate[i] * sig * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

float bitnet_relu2_mul_max_abs_impl(float *gate, const float *up, int n) {
    int i = 0;
    float max_abs = 0.0f;
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    float32x4_t zero = vdupq_n_f32(0.0f);

    for (; i + 3 < n; i += 4) {
        float32x4_t gv = vmaxq_f32(vld1q_f32(gate + i), zero);
        float32x4_t uv = vld1q_f32(up + i);
        float32x4_t result = vmulq_f32(vmulq_f32(gv, gv), uv);
        vst1q_f32(gate + i, result);
        max_vec = vmaxq_f32(max_vec, vabsq_f32(result));
    }
    max_abs = vmaxvq_f32(max_vec);
    for (; i < n; ++i) {
        float g = gate[i] > 0.0f ? gate[i] : 0.0f;
        float result = g * g * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

void bitnet_residual_add_impl(float *out, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
#else
void bitnet_silu_impl(float *x, int n) {
    int i = 0;
    for (i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        x[i] = x[i] * sig;
    }
}

void bitnet_silu_mul_impl(float *gate, const float *up, int n) {
    int i = 0;
    for (i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        gate[i] = gate[i] * sig * up[i];
    }
}

float bitnet_silu_mul_max_abs_impl(float *gate, const float *up, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        float result = gate[i] * sig * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

float bitnet_relu2_mul_max_abs_impl(float *gate, const float *up, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float g = gate[i] > 0.0f ? gate[i] : 0.0f;
        float result = g * g * up[i];
        float a = fabsf(result);
        gate[i] = result;
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

void bitnet_residual_add_impl(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
#endif

/* ========== Softmax ========== */

#if defined(__ARM_NEON)
void bitnet_softmax_impl(float *x, int n) {
    /* Find max using NEON */
    int i = 0;
    float32x4_t max_vec = vdupq_n_f32(-1e30f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        max_vec = vmaxq_f32(max_vec, v);
    }
    float max_val = vmaxvq_f32(max_vec);
    /* Handle remaining elements for max */
    for (; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    /* Also check first few elements if n < 4 */
    if (n < 4) {
        for (i = 0; i < n; ++i) {
            if (x[i] > max_val) max_val = x[i];
        }
    }

    /* exp(x - max) and sum using NEON */
    i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    float32x4_t max_broadcast = vdupq_n_f32(max_val);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v = vsubq_f32(v, max_broadcast);
        /* Use standard expf for accuracy (softmax needs precision) */
        float tmp[4];
        vst1q_f32(tmp, v);
        tmp[0] = expf(tmp[0]);
        tmp[1] = expf(tmp[1]);
        tmp[2] = expf(tmp[2]);
        tmp[3] = expf(tmp[3]);
        float32x4_t ev = vld1q_f32(tmp);
        vst1q_f32(x + i, ev);
        sum_vec = vaddq_f32(sum_vec, ev);
    }
    float sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Divide by sum */
    float inv_sum = 1.0f / sum;
    i = 0;
    float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v = vmulq_f32(v, inv_sum_vec);
        vst1q_f32(x + i, v);
    }
    for (; i < n; ++i) {
        x[i] *= inv_sum;
    }
}
#else
void bitnet_softmax_impl(float *x, int n) {
    int i = 0;
    float max_val = x[0];
    float sum = 0.0f;

    for (i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    sum = 0.0f;
    for (i = 0; i < n; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (i = 0; i < n; ++i) {
        x[i] /= sum;
    }
}
#endif

/* ========== Residual Add ========== */
/* NEON + scalar residual_add definitions live with the SiLU family above
 * (renamed to _impl, dispatched via trampoline at the bottom of this file). */

/* ========== RoPE ========== */

#if defined(__ARM_NEON)
void bitnet_rope_apply_impl(float *x, int n_heads, int head_dim, int rope_dim,
                            const float *rope_cos, const float *rope_sin) {
    if (x == NULL || rope_cos == NULL || rope_sin == NULL ||
        n_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
        return;
    }

    for (int h = 0; h < n_heads; ++h) {
        int j = 0;
        for (; j + 1 < rope_dim / 2; j += 2) {
            /* Load 2 rotation pairs [x0,y0,x1,y1] and rotate each with its
             * (cos,sin). vtrn duplicates x into even lanes, y into odd lanes;
             * out = x*[c,s,c,s] + y*[-s,c,-s,c] per pair. */
            float c0 = rope_cos[j],     s0 = rope_sin[j];
            float c1 = rope_cos[j + 1], s1 = rope_sin[j + 1];
            float32x4_t v = vld1q_f32(x + h * head_dim + 2 * j);
            float32x4x2_t tr = vtrnq_f32(v, v);
            float32x4_t x_dup = tr.val[0];            /* [x0,x0,x1,x1] */
            float32x4_t y_dup = tr.val[1];            /* [y0,y0,y1,y1] */
            float32x4_t cs     = { c0, s0, c1, s1 };
            float32x4_t neg_sc = { -s0, c0, -s1, c1 };
            float32x4_t out = vfmaq_f32(vmulq_f32(x_dup, cs), y_dup, neg_sc);
            vst1q_f32(x + h * head_dim + 2 * j, out);
        }
        /* Handle remaining pair */
        for (; j < rope_dim / 2; ++j) {
            int idx0 = h * head_dim + 2 * j;
            int idx1 = idx0 + 1;
            float c = rope_cos[j];
            float s = rope_sin[j];
            float x0 = x[idx0];
            float x1 = x[idx1];
            x[idx0] = x0 * c - x1 * s;
            x[idx1] = x0 * s + x1 * c;
        }
    }
}
#else
void bitnet_rope_apply_impl(float *x, int n_heads, int head_dim, int rope_dim,
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
#endif

/* ========== Dispatch trampolines ========== */

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

void bitnet_silu(float *x, int n) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    g_bitnet_dispatch->silu(x, n);
}

void bitnet_silu_mul(float *gate, const float *up, int n) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    g_bitnet_dispatch->silu_mul(gate, up, n);
}

float bitnet_silu_mul_max_abs(float *gate, const float *up, int n) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    return g_bitnet_dispatch->silu_mul_max_abs(gate, up, n);
}

float bitnet_relu2_mul_max_abs(float *gate, const float *up, int n) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    return g_bitnet_dispatch->relu2_mul_max_abs(gate, up, n);
}

void bitnet_residual_add(float *out, const float *a, const float *b, int n) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    g_bitnet_dispatch->residual_add(out, a, b, n);
}

void bitnet_softmax(float *x, int n) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    g_bitnet_dispatch->softmax(x, n);
}

void bitnet_rope_apply(float *x, int n_heads, int head_dim, int rope_dim,
                       const float *rope_cos, const float *rope_sin) {
    if (g_bitnet_dispatch == NULL) bitnet_dispatch_init();
    g_bitnet_dispatch->rope_apply(x, n_heads, head_dim, rope_dim, rope_cos, rope_sin);
}
