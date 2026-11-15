#include "quant_ptq1_0.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__aarch64__)
#include <sys/auxv.h>
#endif

/* Trit value tables: trit_tab[n][byte] is the trit carried by digit position
 * n (0 = most significant) of a base-3 packed byte, as float -1/0/+1.
 * Derived exactly as the reference dequantizer:
 *     q    = (uint8_t)(byte * 3^n)
 *     digit= ((uint16_t)q * 3) >> 8
 *     trit = digit - 1 */
static float g_trit_tab[5][256];
static int g_trit_tab_ready = 0;

static const uint8_t g_pow3[5] = {1, 3, 9, 27, 81};

/* One trit of a base-3 packed byte, as 0/1/2 (caller subtracts one). */
static inline uint8_t ptq_trit_scalar(uint8_t b, unsigned pow3) {
    unsigned q = ((unsigned)b * pow3) & 0xFFu;
    return (uint8_t)((q * 3u) >> 8);
}

static void trit_tab_init(void) {
    for (int n = 0; n < 5; ++n) {
        for (int b = 0; b < 256; ++b) {
            uint8_t q = (uint8_t)((unsigned)b * g_pow3[n]);
            int digit = ((unsigned)q * 3u) >> 8;
            g_trit_tab[n][b] = (float)(digit - 1);
        }
    }
    g_trit_tab_ready = 1;
}

float bitnet_ptq1_0_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    float out;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* subnormal: normalize */
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = sign | (uint32_t)(127 - 15 - e) << 23 | mant << 13;
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | mant << 13;
    } else {
        bits = sign | (exp + 112u) << 23 | mant << 13;
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

size_t bitnet_ptq1_0_row_size(int n) {
    return ((size_t)(n / BITNET_PTQ1_0_QK)) * BITNET_PTQ1_0_BLOCK_SIZE;
}

/* Emit the 128 trit values of one block in element order.
 * Reference stages: qs[0..15] stride 16 for 80 values, qs[16..23] stride 8
 * for 40 values, qh[0..1] stride 2 for the final 8 values. */
static void block_decode(const bitnet_ptq1_0_block_t *blk, const float **tab, float *out) {
    for (int n = 0; n < 5; ++n) {
        const float *tn = tab[n];
        float *o = out + n * 16;
        for (int m = 0; m < 16; ++m) o[m] = tn[blk->qs[m]];
    }
    for (int n = 0; n < 5; ++n) {
        const float *tn = tab[n];
        float *o = out + 80 + n * 8;
        for (int m = 0; m < 8; ++m) o[m] = tn[blk->qs[16 + m]];
    }
    for (int n = 0; n < 4; ++n) {
        const float *tn = tab[n];
        float *o = out + 120 + n * 2;
        for (int h = 0; h < 2; ++h) o[h] = tn[blk->qh[h]];
    }
}

void bitnet_ptq1_0_dequantize_row(const void *src, float *out, int n) {
    const bitnet_ptq1_0_block_t *blocks = (const bitnet_ptq1_0_block_t *)src;
    float trits[BITNET_PTQ1_0_QK];
    const float *tab[5];
    int nb = n / BITNET_PTQ1_0_QK;

    if (!g_trit_tab_ready) trit_tab_init();
    for (int i = 0; i < 5; ++i) tab[i] = g_trit_tab[i];

    for (int b = 0; b < nb; ++b) {
        float d = bitnet_ptq1_0_fp16_to_fp32(blocks[b].d);
        block_decode(&blocks[b], tab, trits);
        for (int i = 0; i < BITNET_PTQ1_0_QK; ++i) out[b * BITNET_PTQ1_0_QK + i] = trits[i] * d;
    }
}

float bitnet_ptq1_0_dot_row_scalar(const void *src, const float *x, int n) {
    const bitnet_ptq1_0_block_t *blocks = (const bitnet_ptq1_0_block_t *)src;
    const float *tab[5];
    float sum = 0.0f;
    int nb = n / BITNET_PTQ1_0_QK;

    if (!g_trit_tab_ready) trit_tab_init();
    for (int i = 0; i < 5; ++i) tab[i] = g_trit_tab[i];

    for (int b = 0; b < nb; ++b) {
        const bitnet_ptq1_0_block_t *blk = &blocks[b];
        const float *xb = x + b * BITNET_PTQ1_0_QK;
        float acc = 0.0f;
        const float *t0 = tab[0], *t1 = tab[1], *t2 = tab[2], *t3 = tab[3], *t4 = tab[4];
        /* five independent accumulators: one per packed digit position, so the
         * FMA latency of one chain is hidden behind the others */
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f;

        for (int m = 0; m < 16; ++m) {
            const uint8_t b = blk->qs[m];
            a0 += t0[b] * xb[m];
            a1 += t1[b] * xb[16 + m];
            a2 += t2[b] * xb[32 + m];
            a3 += t3[b] * xb[48 + m];
            a4 += t4[b] * xb[64 + m];
        }
        for (int m = 0; m < 8; ++m) {
            const uint8_t b = blk->qs[16 + m];
            a0 += t0[b] * xb[80 + m];
            a1 += t1[b] * xb[88 + m];
            a2 += t2[b] * xb[96 + m];
            a3 += t3[b] * xb[104 + m];
            a4 += t4[b] * xb[112 + m];
        }
        for (int h = 0; h < 2; ++h) {
            const uint8_t b = blk->qh[h];
            a0 += t0[b] * xb[120 + h];
            a1 += t1[b] * xb[122 + h];
            a2 += t2[b] * xb[124 + h];
            a3 += t3[b] * xb[126 + h];
        }
        acc = ((a0 + a1) + (a2 + a3)) + a4;
        sum += acc * bitnet_ptq1_0_fp16_to_fp32(blk->d);
    }
    return sum;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

/* Same arithmetic as the scalar table lookup, but done on eight lanes at a
 * time:  q = (uint8_t)(b * 3^n);  digit = (q * 3) >> 8;  trit = digit - 1.
 * The multiply is done in 16-bit lanes (max 255 * 81 = 20655) and masked back
 * to a byte to reproduce the truncation the reference decoder relies on. */
static inline void ptq_trit8(uint8x8_t b, unsigned pow3, float32x4_t *lo, float32x4_t *hi) {
    uint16x8_t q = vmulq_u16(vmovl_u8(b), vdupq_n_u16((uint16_t)pow3));
    int16x8_t t;
    q = vandq_u16(q, vdupq_n_u16(0xFFu));
    q = vshrq_n_u16(vmulq_n_u16(q, 3u), 8);
    t = vsubq_s16(vreinterpretq_s16_u16(q), vdupq_n_s16(1));
    *lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(t)));
    *hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(t)));
}

float bitnet_ptq1_0_dot_row_neon(const void *src, const float *x, int n) {
    const bitnet_ptq1_0_block_t *blocks = (const bitnet_ptq1_0_block_t *)src;
    float sum = 0.0f;
    int nb = n / BITNET_PTQ1_0_QK;

    if (!g_trit_tab_ready) trit_tab_init();

    for (int b = 0; b < nb; ++b) {
        const bitnet_ptq1_0_block_t *blk = &blocks[b];
        const float *xb = x + b * BITNET_PTQ1_0_QK;
        const uint8x16_t q0 = vld1q_u8(blk->qs);
        const uint8x8_t q1 = vld1_u8(blk->qs + 16);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float tail = 0.0f;

        /* qs[0..15]: five digits, stride 16 over x[0..79] */
        for (int d = 0; d < 5; ++d) {
            const float *xp = xb + d * 16;
            float32x4_t lo, hi, lo2, hi2;
            ptq_trit8(vget_low_u8(q0), g_pow3[d], &lo, &hi);
            ptq_trit8(vget_high_u8(q0), g_pow3[d], &lo2, &hi2);
            acc0 = vfmaq_f32(acc0, lo, vld1q_f32(xp));
            acc1 = vfmaq_f32(acc1, hi, vld1q_f32(xp + 4));
            acc0 = vfmaq_f32(acc0, lo2, vld1q_f32(xp + 8));
            acc1 = vfmaq_f32(acc1, hi2, vld1q_f32(xp + 12));
        }
        /* qs[16..23]: five digits, stride 8 over x[80..119] */
        for (int d = 0; d < 5; ++d) {
            const float *xp = xb + 80 + d * 8;
            float32x4_t lo, hi;
            ptq_trit8(q1, g_pow3[d], &lo, &hi);
            acc0 = vfmaq_f32(acc0, lo, vld1q_f32(xp));
            acc1 = vfmaq_f32(acc1, hi, vld1q_f32(xp + 4));
        }
        /* qh[0..1]: four digits, stride 2 over x[120..127] */
        for (int d = 0; d < 4; ++d) {
            const float t0 = (float)ptq_trit_scalar(blk->qh[0], g_pow3[d]) - 1.0f;
            const float t1 = (float)ptq_trit_scalar(blk->qh[1], g_pow3[d]) - 1.0f;
            tail += t0 * xb[120 + 2 * d] + t1 * xb[121 + 2 * d];
        }

        {
            float acc = vaddvq_f32(vaddq_f32(acc0, acc1)) + tail;
            sum += acc * bitnet_ptq1_0_fp16_to_fp32(blk->d);
        }
    }
    return sum;
}
#endif

float bitnet_ptq1_0_dot_row(const void *src, const float *x, int n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return bitnet_ptq1_0_dot_row_neon(src, x, n);
#else
    return bitnet_ptq1_0_dot_row_scalar(src, x, n);
#endif
}

/* ------------------------------------------------------------------ */
/* q8_0 activations + integer PTQ1_0 dot product                       */
/* ------------------------------------------------------------------ */

size_t bitnet_q8_0_row_size(int n) {
    return ((size_t)(n / BITNET_Q8_0_QK)) * sizeof(bitnet_q8_0_block_t);
}

/* round-to-nearest-even float -> fp16, the conversion ggml uses for scales */
static uint16_t bq_f32_to_f16(float f) {
    uint32_t x;
    uint32_t sign, exp, mant;
    memcpy(&x, &f, sizeof(x));
    sign = (x >> 16) & 0x8000u;
    exp = (x >> 23) & 0xFFu;
    mant = x & 0x7FFFFFu;
    if (exp == 0xFFu) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp == 0) return (uint16_t)sign;
    {
        int e = (int)exp - 127 + 15;
        if (e >= 31) return (uint16_t)(sign | 0x7C00u);
        if (e <= 0) {
            /* subnormal half */
            uint32_t m = mant | 0x800000u;
            int shift = 14 - e;
            uint32_t q = m >> shift;
            uint32_t rem = m & ((1u << shift) - 1u);
            uint32_t half = 1u << (shift - 1);
            if (rem > half || (rem == half && (q & 1u))) ++q;
            return (uint16_t)(sign | q);
        }
        {
            uint32_t q = mant >> 13;
            uint32_t rem = mant & 0x1FFFu;
            if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) {
                ++q;
                if (q == 0x400u) { q = 0; ++e; if (e >= 31) return (uint16_t)(sign | 0x7C00u); }
            }
            return (uint16_t)(sign | ((uint32_t)e << 10) | q);
        }
    }
}

void bitnet_quantize_row_q8_0(const float *x, void *dst, int n) {
    bitnet_q8_0_block_t *y = (bitnet_q8_0_block_t *)dst;
    int nb = n / BITNET_Q8_0_QK;

    for (int i = 0; i < nb; ++i) {
        const float *xp = x + (size_t)i * BITNET_Q8_0_QK;
        float amax = 0.0f;
        float d, id;
        for (int j = 0; j < BITNET_Q8_0_QK; ++j) {
            float a = fabsf(xp[j]);
            if (a > amax) amax = a;
        }
        d = amax / 127.0f;
        id = d > 0.0f ? 1.0f / d : 0.0f;
        y[i].d = bq_f32_to_f16(d);
        for (int j = 0; j < BITNET_Q8_0_QK; ++j) {
            y[i].qs[j] = (int8_t)roundf(xp[j] * id);
        }
    }
}

#if !(defined(__aarch64__) && defined(__ARM_NEON))
/* Emit the 128 trits as int8 (values -1/0/+1) in element order. */
static void ptq_decode_scalar(const bitnet_ptq1_0_block_t *blk, int8_t out[128]) {
    for (int n = 0; n < 5; ++n) {
        int8_t *o = out + n * 16;
        for (int m = 0; m < 16; ++m) o[m] = (int8_t)(ptq_trit_scalar(blk->qs[m], g_pow3[n]) - 1);
    }
    for (int n = 0; n < 5; ++n) {
        int8_t *o = out + 80 + n * 8;
        for (int m = 0; m < 8; ++m) o[m] = (int8_t)(ptq_trit_scalar(blk->qs[16 + m], g_pow3[n]) - 1);
    }
    for (int n = 0; n < 4; ++n) {
        int8_t *o = out + 120 + n * 2;
        for (int h = 0; h < 2; ++h) o[h] = (int8_t)(ptq_trit_scalar(blk->qh[h], g_pow3[n]) - 1);
    }
}

static float ptq_dot_q8_scalar(const void *src, const void *xq, int n) {
    const bitnet_ptq1_0_block_t *blocks = (const bitnet_ptq1_0_block_t *)src;
    const bitnet_q8_0_block_t *y = (const bitnet_q8_0_block_t *)xq;
    int nb = n / BITNET_PTQ1_0_QK;
    float sum = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const float dx = bitnet_ptq1_0_fp16_to_fp32(blocks[b].d);
        int8_t trits[BITNET_PTQ1_0_QK];
        ptq_decode_scalar(&blocks[b], trits);
        for (int k = 0; k < 4; ++k) {
            const bitnet_q8_0_block_t *yb = &y[(size_t)b * 4 + k];
            int acc = 0;
            for (int j = 0; j < BITNET_Q8_0_QK; ++j) acc += (int)trits[k * 32 + j] * (int)yb->qs[j];
            sum += (float)acc * (dx * bitnet_ptq1_0_fp16_to_fp32(yb->d));
        }
    }
    return sum;
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

/* Decode one block into eight int8x16 vectors of -1/0/+1, element order.
 * The trit arithmetic is the reference identity
 *   ((uint8)(q * 3^n) * 3) >> 8  ==  ((q + (q >> 1)) >> 1) >> 6
 * evaluated on 8/16 u8 lanes, so no widening is needed. */
static inline void ptq_decode_neon(const bitnet_ptq1_0_block_t *b, int8x16_t e[8]) {
    static const uint8_t k_qh_pow3[8] = {1, 1, 3, 3, 9, 9, 27, 27};
    const int8x16_t one = vdupq_n_s8(1);
    const uint8x16_t a = vld1q_u8(b->qs);
    const uint8x8_t c = vld1_u8(b->qs + 16);
    const uint16_t qh16 = (uint16_t)b->qh[0] | ((uint16_t)b->qh[1] << 8);
    const uint8x8_t h = vreinterpret_u8_u16(vdup_n_u16(qh16));
    uint8x8_t u0, u1, u2, u3, u4, uh;

#define PTQ1_TRIT16(v) vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8((v), vshrq_n_u8((v), 1)), 6)), one)
#define PTQ1_TRIT8(v)  vshr_n_u8(vhadd_u8((v), vshr_n_u8((v), 1)), 6)

    e[0] = PTQ1_TRIT16(a);
    e[1] = PTQ1_TRIT16(vmulq_u8(a, vdupq_n_u8(3)));
    e[2] = PTQ1_TRIT16(vmulq_u8(a, vdupq_n_u8(9)));
    e[3] = PTQ1_TRIT16(vmulq_u8(a, vdupq_n_u8(27)));
    e[4] = PTQ1_TRIT16(vmulq_u8(a, vdupq_n_u8(81)));

    u0 = PTQ1_TRIT8(c);
    u1 = PTQ1_TRIT8(vmul_u8(c, vdup_n_u8(3)));
    u2 = PTQ1_TRIT8(vmul_u8(c, vdup_n_u8(9)));
    u3 = PTQ1_TRIT8(vmul_u8(c, vdup_n_u8(27)));
    u4 = PTQ1_TRIT8(vmul_u8(c, vdup_n_u8(81)));
    uh = PTQ1_TRIT8(vmul_u8(h, vld1_u8(k_qh_pow3)));

    e[5] = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(u0, u1)), one);
    e[6] = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(u2, u3)), one);
    e[7] = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(u4, uh)), one);

#undef PTQ1_TRIT16
#undef PTQ1_TRIT8
}

static float ptq_dot_q8_neon(const bitnet_ptq1_0_block_t *x, const bitnet_q8_0_block_t *y, int nb) {
    float32x4_t sumv = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; ++i) {
        int8x16_t e[8];
        const float dx = bitnet_ptq1_0_fp16_to_fp32(x[i].d);
        ptq_decode_neon(&x[i], e);
        for (int k = 0; k < 4; ++k) {
            const bitnet_q8_0_block_t *yb = &y[(size_t)i * 4 + k];
            const int8x16_t ya = vld1q_s8(yb->qs);
            const int8x16_t yc = vld1q_s8(yb->qs + 16);
            const int32x4_t p = vaddq_s32(
                vaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(e[2 * k]), vget_low_s8(ya))),
                          vpaddlq_s16(vmull_s8(vget_high_s8(e[2 * k]), vget_high_s8(ya)))),
                vaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(e[2 * k + 1]), vget_low_s8(yc))),
                          vpaddlq_s16(vmull_s8(vget_high_s8(e[2 * k + 1]), vget_high_s8(yc)))));
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), dx * bitnet_ptq1_0_fp16_to_fp32(yb->d));
        }
    }
    return vaddvq_f32(sumv);
}

#if defined(__GNUC__) || defined(__clang__)
/* ARMv8.2 dot product: four int8 lanes per SDOT, so 128 trits cost eight
 * instructions instead of the widening sequence above. */
__attribute__((target("arch=armv8.2-a+dotprod"), noinline))
static float ptq_dot_q8_dotprod(const bitnet_ptq1_0_block_t *x, const bitnet_q8_0_block_t *y, int nb) {
    float32x4_t sumv = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; ++i) {
        int8x16_t e[8];
        const float dx = bitnet_ptq1_0_fp16_to_fp32(x[i].d);
        ptq_decode_neon(&x[i], e);
        for (int k = 0; k < 4; ++k) {
            const bitnet_q8_0_block_t *yb = &y[(size_t)i * 4 + k];
            const int32x4_t p = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), e[2 * k], vld1q_s8(yb->qs)),
                e[2 * k + 1], vld1q_s8(yb->qs + 16));
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), dx * bitnet_ptq1_0_fp16_to_fp32(yb->d));
        }
    }
    return vaddvq_f32(sumv);
}

/* Two blocks at a time.  Each decode is a four-deep dependency chain, so
 * decoding the next block while the current one's dots issue keeps both NEON
 * pipes fed.  Selected with BQ_CPU_UNROLL2=1 so the two variants can be timed
 * against each other; the results differ only in summation order. */
__attribute__((target("arch=armv8.2-a+dotprod"), noinline))
static float ptq_dot_q8_dotprod2(const bitnet_ptq1_0_block_t *x, const bitnet_q8_0_block_t *y, int nb) {
    float32x4_t sumv = vdupq_n_f32(0.0f);
    float32x4_t sumw = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 1 < nb; i += 2) {
        int8x16_t e[8], f[8];
        const float dx = bitnet_ptq1_0_fp16_to_fp32(x[i].d);
        const float dy = bitnet_ptq1_0_fp16_to_fp32(x[i + 1].d);
        ptq_decode_neon(&x[i], e);
        ptq_decode_neon(&x[i + 1], f);
        for (int k = 0; k < 4; ++k) {
            const bitnet_q8_0_block_t *yb = &y[(size_t)i * 4 + k];
            const bitnet_q8_0_block_t *zb = &y[(size_t)(i + 1) * 4 + k];
            const int32x4_t p = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), e[2 * k], vld1q_s8(yb->qs)),
                e[2 * k + 1], vld1q_s8(yb->qs + 16));
            const int32x4_t q = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), f[2 * k], vld1q_s8(zb->qs)),
                f[2 * k + 1], vld1q_s8(zb->qs + 16));
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), dx * bitnet_ptq1_0_fp16_to_fp32(yb->d));
            sumw = vmlaq_n_f32(sumw, vcvtq_f32_s32(q), dy * bitnet_ptq1_0_fp16_to_fp32(zb->d));
        }
    }
    for (; i < nb; ++i) {
        int8x16_t e[8];
        const float dx = bitnet_ptq1_0_fp16_to_fp32(x[i].d);
        ptq_decode_neon(&x[i], e);
        for (int k = 0; k < 4; ++k) {
            const bitnet_q8_0_block_t *yb = &y[(size_t)i * 4 + k];
            const int32x4_t p = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), e[2 * k], vld1q_s8(yb->qs)),
                e[2 * k + 1], vld1q_s8(yb->qs + 16));
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), dx * bitnet_ptq1_0_fp16_to_fp32(yb->d));
        }
    }
    return vaddvq_f32(vaddq_f32(sumv, sumw));
}

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif

static int ptq_have_dotprod(void) {
    static int cached = -1;
    if (cached < 0) cached = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ? 1 : 0;
    return cached;
}

static int ptq_unroll2(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("BQ_CPU_UNROLL2");
        cached = (e != NULL && e[0] == '1') ? 1 : 0;
    }
    return cached;
}
#endif
#endif


/* ------------------------------------------------------------------ */
/* batched form: one packed row against B int8 activation rows        */
/* ------------------------------------------------------------------ */

/* Quantize a row to int8 with a single scale, the layout the batched
 * kernel wants (n must be a multiple of 128). */
void bitnet_quantize_row_i8(const float *x, int8_t *dst, float *scale, int n) {
    float amax = 0.0f;
    int i;
    for (i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    {
        float s = amax / 127.0f;
        float inv;
        if (!(s > 1e-8f)) s = 1e-8f;
        *scale = s;
        inv = 1.0f / s;
        for (i = 0; i < n; ++i) {
            int v = (int)lrintf(x[i] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            dst[i] = (int8_t)v;
        }
    }
}

/* Element index of group `g`, lane `d`, inside one 128-value block. */
static int ptq_group_index(int g, int d) {
    if (g < 16) return d * 16 + g;
    if (g < 24) return 80 + d * 8 + (g - 16);
    if (g < 28) return 64 + 4 * (g - 24) + d;
    if (g < 30) return 112 + 4 * (g - 28) + d;
    return 120 + 2 * d + (g - 30);
}

void bitnet_quantize_row_i8_groups(const float *x, int8_t *dst, float *scale, int n) {
    float amax = 0.0f;
    int i, nb = n / BITNET_PTQ1_0_QK;
    for (i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    {
        float s = amax / 127.0f;
        float inv;
        if (!(s > 1e-8f)) s = 1e-8f;
        *scale = s;
        inv = 1.0f / s;
        for (i = 0; i < nb; ++i) {
            const float *x0 = x + (size_t)i * BITNET_PTQ1_0_QK;
            int8_t *o = dst + (size_t)i * BITNET_PTQ1_0_QK;
            int g, d;
            for (g = 0; g < 32; ++g) {
                for (d = 0; d < 4; ++d) {
                    int v = (int)lrintf(x0[ptq_group_index(g, d)] * inv);
                    if (v > 127) v = 127;
                    if (v < -127) v = -127;
                    o[g * 4 + d] = (int8_t)v;
                }
            }
        }
    }
}

void bitnet_unpack_row_i8_groups(const int8_t *src, float *out, float scale, int n) {
    int i, nb = n / BITNET_PTQ1_0_QK;
    for (i = 0; i < nb; ++i) {
        const int8_t *s = src + (size_t)i * BITNET_PTQ1_0_QK;
        float *o = out + (size_t)i * BITNET_PTQ1_0_QK;
        int g, d;
        for (g = 0; g < 32; ++g) {
            for (d = 0; d < 4; ++d) {
                o[ptq_group_index(g, d)] = (float)s[g * 4 + d] * scale;
            }
        }
    }
}

#define PTQ_MM_MAX_B 16

static void ptq_trits_block(const bitnet_ptq1_0_block_t *blk, int8_t out[128]) {
    int n, m;
    for (n = 0; n < 5; ++n) {
        int8_t *o = out + n * 16;
        for (m = 0; m < 16; ++m) o[m] = (int8_t)(ptq_trit_scalar(blk->qs[m], g_pow3[n]) - 1);
    }
    for (n = 0; n < 5; ++n) {
        int8_t *o = out + 80 + n * 8;
        for (m = 0; m < 8; ++m) o[m] = (int8_t)(ptq_trit_scalar(blk->qs[16 + m], g_pow3[n]) - 1);
    }
    for (n = 0; n < 4; ++n) {
        int8_t *o = out + 120 + n * 2;
        for (m = 0; m < 2; ++m) o[m] = (int8_t)(ptq_trit_scalar(blk->qh[m], g_pow3[n]) - 1);
    }
}

static void ptq_matmat_scalar(const bitnet_ptq1_0_block_t *x, const int8_t *xq,
                              const float *xs, int nb, int B, float *out) {
    float acc[PTQ_MM_MAX_B];
    int i, b;
    for (b = 0; b < B; ++b) acc[b] = 0.0f;
    for (i = 0; i < nb; ++i) {
        int8_t t[128];
        float d = bitnet_ptq1_0_fp16_to_fp32(x[i].d);
        int j;
        ptq_trits_block(&x[i], t);
        for (b = 0; b < B; ++b) {
            const int8_t *xb = xq + (size_t)b * (size_t)nb * 128 + (size_t)i * 128;
            int32_t s = 0;
            for (j = 0; j < 128; ++j) s += (int)t[j] * (int)xb[j];
            acc[b] += (float)s * d;
        }
    }
    for (b = 0; b < B; ++b) out[b] = acc[b] * xs[b];
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("arch=armv8.2-a+dotprod"), noinline))
static void ptq_matmat_dotprod(const bitnet_ptq1_0_block_t *x, const int8_t *xq,
                               const float *xs, int nb, int B, float *out) {
    const size_t bstride = (size_t)nb * 128;

    /* The row count is a compile-time constant in each instantiation: with a
     * runtime B the accumulator array is indexed dynamically, the compiler
     * keeps it in memory and the sdot chains stall on it.  Measured on a
     * Cortex-X3/A715 mix this is worth 2-3x over the generic loop. */
#define PTQ_MM_BODY(BB) {                                                     \
        float32x4_t facc[BB];                                                 \
        int i, b;                                                             \
        for (b = 0; b < BB; ++b) facc[b] = vdupq_n_f32(0.0f);                 \
        for (i = 0; i < nb; ++i) {                                            \
            int8x16_t e[8];                                                   \
            const float32x4_t dv = vdupq_n_f32(bitnet_ptq1_0_fp16_to_fp32(x[i].d)); \
            ptq_decode_neon(&x[i], e);                                        \
            for (b = 0; b < BB; ++b) {                                        \
                const int8_t *xb = xq + (size_t)b * bstride + (size_t)i * 128; \
                int32x4_t t = vdupq_n_s32(0);                                  \
                t = vdotq_s32(t, e[0], vld1q_s8(xb + 0));                     \
                t = vdotq_s32(t, e[1], vld1q_s8(xb + 16));                    \
                t = vdotq_s32(t, e[2], vld1q_s8(xb + 32));                    \
                t = vdotq_s32(t, e[3], vld1q_s8(xb + 48));                    \
                t = vdotq_s32(t, e[4], vld1q_s8(xb + 64));                    \
                t = vdotq_s32(t, e[5], vld1q_s8(xb + 80));                    \
                t = vdotq_s32(t, e[6], vld1q_s8(xb + 96));                    \
                t = vdotq_s32(t, e[7], vld1q_s8(xb + 112));                   \
                facc[b] = vmlaq_f32(facc[b], vcvtq_f32_s32(t), dv);           \
            }                                                                 \
        }                                                                     \
        for (b = 0; b < BB; ++b) out[b] = vaddvq_f32(facc[b]) * xs[b];        \
    }

    switch (B) {
    case 1: PTQ_MM_BODY(1) break;
    case 2: PTQ_MM_BODY(2) break;
    case 3: PTQ_MM_BODY(3) break;
    case 4: PTQ_MM_BODY(4) break;
    case 5: PTQ_MM_BODY(5) break;
    case 6: PTQ_MM_BODY(6) break;
    case 8: PTQ_MM_BODY(8) break;
    default: {
        float acc[PTQ_MM_MAX_B];
        int i2, b2;
        for (b2 = 0; b2 < B; ++b2) acc[b2] = 0.0f;
        for (i2 = 0; i2 < nb; ++i2) {
            int8x16_t e[8];
            const float d = bitnet_ptq1_0_fp16_to_fp32(x[i2].d);
            ptq_decode_neon(&x[i2], e);
            for (b2 = 0; b2 < B; ++b2) {
                const int8_t *xb = xq + (size_t)b2 * bstride + (size_t)i2 * 128;
                int32x4_t t = vdupq_n_s32(0);
                t = vdotq_s32(t, e[0], vld1q_s8(xb + 0));
                t = vdotq_s32(t, e[1], vld1q_s8(xb + 16));
                t = vdotq_s32(t, e[2], vld1q_s8(xb + 32));
                t = vdotq_s32(t, e[3], vld1q_s8(xb + 48));
                t = vdotq_s32(t, e[4], vld1q_s8(xb + 64));
                t = vdotq_s32(t, e[5], vld1q_s8(xb + 80));
                t = vdotq_s32(t, e[6], vld1q_s8(xb + 96));
                t = vdotq_s32(t, e[7], vld1q_s8(xb + 112));
                acc[b2] += (float)vaddvq_s32(t) * d;
            }
        }
        for (b2 = 0; b2 < B; ++b2) out[b2] = acc[b2] * xs[b2];
        break;
    }
    }
#undef PTQ_MM_BODY
}
#endif

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("arch=armv8.2-a+dotprod"), noinline))
static void ptq_matmat_dotprod_x(const bitnet_ptq1_0_block_t *x, const int8_t *xq,
                                 const float *xs, int nb, int B, float *out) {
    const size_t bstride = (size_t)nb * 128;


/* Variant that splits each row over two integer accumulators and walks two
 * blocks at a time, on the theory that the 44 cycles/block measured at B=2
 * were a vdot latency chain.  Measured on the Cix P1 A720 it is neutral to
 * worse, so it is off by default (BQ_MM_X=1 selects it); the real fix for the
 * decode share is the i8mm path below. */
#define PTQ_MMX_BODY(BB) {                                                    \
        float32x4_t fa[BB], fb[BB];                                           \
        int i = 0, b;                                                         \
        for (b = 0; b < BB; ++b) { fa[b] = vdupq_n_f32(0.0f); fb[b] = fa[b]; }\
        for (; i + 1 < nb; i += 2) {                                          \
            int8x16_t e[8], g[8];                                             \
            const int8x16_t *ec = e;                                          \
            const int8x16_t *gc = g;                                          \
            const float32x4_t dv = vdupq_n_f32(                                \
                bitnet_ptq1_0_fp16_to_fp32(x[i].d));                          \
            const float32x4_t dw = vdupq_n_f32(                                \
                bitnet_ptq1_0_fp16_to_fp32(x[i + 1].d));                      \
            ptq_decode_neon(&x[i], e);                                        \
            ptq_decode_neon(&x[i + 1], g);                                    \
            for (b = 0; b < BB; ++b) {                                        \
                const int8_t *xb =                                             \
                    xq + (size_t)b * bstride + (size_t)i * 128;               \
                int32x4_t t0 = vdupq_n_s32(0), t1 = t0;                       \
                int32x4_t u0 = t0, u1 = t0;                                   \
                t0 = vdotq_s32(t0, ec[0], vld1q_s8(xb +   0));            \
                t1 = vdotq_s32(t1, ec[1], vld1q_s8(xb +  16));            \
                t0 = vdotq_s32(t0, ec[2], vld1q_s8(xb +  32));            \
                t1 = vdotq_s32(t1, ec[3], vld1q_s8(xb +  48));            \
                t0 = vdotq_s32(t0, ec[4], vld1q_s8(xb +  64));            \
                t1 = vdotq_s32(t1, ec[5], vld1q_s8(xb +  80));            \
                t0 = vdotq_s32(t0, ec[6], vld1q_s8(xb +  96));            \
                t1 = vdotq_s32(t1, ec[7], vld1q_s8(xb + 112));            \
                u0 = vdotq_s32(u0, gc[0], vld1q_s8(xb + 128));           \
                u1 = vdotq_s32(u1, gc[1], vld1q_s8(xb + 144));           \
                u0 = vdotq_s32(u0, gc[2], vld1q_s8(xb + 160));           \
                u1 = vdotq_s32(u1, gc[3], vld1q_s8(xb + 176));           \
                u0 = vdotq_s32(u0, gc[4], vld1q_s8(xb + 192));           \
                u1 = vdotq_s32(u1, gc[5], vld1q_s8(xb + 208));           \
                u0 = vdotq_s32(u0, gc[6], vld1q_s8(xb + 224));           \
                u1 = vdotq_s32(u1, gc[7], vld1q_s8(xb + 240));           \
                fa[b] = vmlaq_f32(fa[b],                                 \
                    vcvtq_f32_s32(vaddq_s32(t0, t1)), dv);                 \
                fb[b] = vmlaq_f32(fb[b],                                 \
                    vcvtq_f32_s32(vaddq_s32(u0, u1)), dw);                 \
            }                                                             \
        }                                                                 \
        for (; i < nb; ++i) {                                             \
            int8x16_t e[8];                                              \
            const float32x4_t dv = vdupq_n_f32(                           \
                bitnet_ptq1_0_fp16_to_fp32(x[i].d));                     \
            ptq_decode_neon(&x[i], e);                                   \
            for (b = 0; b < BB; ++b) {                                  \
                const int8_t *xb =                                      \
                    xq + (size_t)b * bstride + (size_t)i * 128;          \
                int32x4_t t0 = vdupq_n_s32(0), t1 = t0;                  \
                t0 = vdotq_s32(t0, e[0], vld1q_s8(xb +   0));            \
                t1 = vdotq_s32(t1, e[1], vld1q_s8(xb +  16));            \
                t0 = vdotq_s32(t0, e[2], vld1q_s8(xb +  32));            \
                t1 = vdotq_s32(t1, e[3], vld1q_s8(xb +  48));            \
                t0 = vdotq_s32(t0, e[4], vld1q_s8(xb +  64));            \
                t1 = vdotq_s32(t1, e[5], vld1q_s8(xb +  80));            \
                t0 = vdotq_s32(t0, e[6], vld1q_s8(xb +  96));            \
                t1 = vdotq_s32(t1, e[7], vld1q_s8(xb + 112));            \
                fa[b] = vmlaq_f32(fa[b],                                \
                    vcvtq_f32_s32(vaddq_s32(t0, t1)), dv);              \
            }                                                            \
        }                                                                \
        for (b = 0; b < BB; ++b)                                         \
            out[b] = vaddvq_f32(vaddq_f32(fa[b], fb[b])) * xs[b];       \
    }

    switch (B) {
    case 1: PTQ_MMX_BODY(1) break;
    case 2: PTQ_MMX_BODY(2) break;
    case 3: PTQ_MMX_BODY(3) break;
    case 4: PTQ_MMX_BODY(4) break;
    default: ptq_matmat_dotprod(x, xq, xs, nb, B, out); break;
    }
#undef PTQ_MMX_BODY
}

static int ptq_mm_x(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("BQ_MM_X");
        cached = e ? atoi(e) : 0;
    }
    return cached;
}
#endif

void bitnet_ptq1_0_matmat_row(const void *src, const int8_t *xq, const float *xscale,
                              int n, int B, float *out) {
    const int nb = n / 128;
#if defined(__GNUC__) || defined(__clang__)
    if (ptq_have_dotprod()) {
        if (ptq_mm_x())
            ptq_matmat_dotprod_x((const bitnet_ptq1_0_block_t *)src, xq, xscale, nb, B, out);
        else
            ptq_matmat_dotprod((const bitnet_ptq1_0_block_t *)src, xq, xscale, nb, B, out);
        return;
    }
#endif
    ptq_matmat_scalar((const bitnet_ptq1_0_block_t *)src, xq, xscale, nb, B, out);
}


/* ------------------------------------------------------------------ */
/* 2-bit repacked rows for the i8mm kernels                           */
/* ------------------------------------------------------------------ */

/* Two output rows share the packed stream, so one 16-byte SMMLA operand is
 * [row a, 8 weights | row a+1, 8 weights] over the same eight columns, and the
 * activation operand is [token t, 8 | token t+1, 8] over those columns.  A
 * block of 128 columns per row pair is 256 trits, stored as 64 bytes: byte b
 * carries stream values b, b+64, b+128, b+192, so a shift and an and give 16
 * decoded weights and the four multiply chains of the base-3 decoder are gone.
 *
 * The trits are stored biased (trit + 1, values 0..2) because that is what a
 * 2-bit field can hold; sum((trit+1)*x) - sum(x) is the wanted dot product, so
 * the preparation step also stores the per-block activation sum. */

typedef struct {
    void *at;        /* [nb][16][pairs] int8x16 activation tiles */
    int32_t *rsum;   /* [nb][B] block sums, for the trit bias */
    int nb, B, pairs;
} ptq_p2prep_t;

int bitnet_ptq1_0_p2_available(void) {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    static int cached = -1;
    if (cached < 0) {
#if defined(AT_HWCAP2)
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
        cached = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) ? 1 : 0;
#else
        cached = 0;
#endif
    }
    return cached;
#else
    return 0;
#endif
}

size_t bitnet_ptq1_0_p2_pair_bytes(int n) {
    return (size_t)(n / 128) * 64u;
}

size_t bitnet_ptq1_0_p2_prep_bytes(int n, int B) {
    const int nb = n / 128;
    const int pairs = (B + 1) / 2;
    return sizeof(ptq_p2prep_t) + (size_t)nb * (size_t)pairs * 16u * 16u +
           (size_t)nb * (size_t)B * 4u;
}

/* row_a and row_b are the packed 28-byte-block rows; scale gets 2*nb floats,
 * the first nb for row_a and the rest for row_b. */
void bitnet_ptq1_0_p2_pack(const void *row_a, const void *row_b, int n,
                           uint8_t *dst, float *scale) {
    const bitnet_ptq1_0_block_t *a = (const bitnet_ptq1_0_block_t *)row_a;
    const bitnet_ptq1_0_block_t *b = (const bitnet_ptq1_0_block_t *)row_b;
    const int nb = n / 128;
    int i, c, j, k, q;

    for (i = 0; i < nb; ++i) {
        int8_t ta[BITNET_PTQ1_0_QK], tb[BITNET_PTQ1_0_QK], s[2 * BITNET_PTQ1_0_QK];
        ptq_trits_block(&a[i], ta);
        ptq_trits_block(&b[i], tb);
        for (c = 0; c < 16; ++c)
            for (j = 0; j < 8; ++j) {
                s[16 * c + j] = ta[8 * c + j];
                s[16 * c + 8 + j] = tb[8 * c + j];
            }
        for (k = 0; k < 64; ++k) {
            unsigned v = 0;
            for (q = 0; q < 4; ++q) v |= ((unsigned)(s[k + 64 * q] + 1) & 3u) << (2 * q);
            dst[(size_t)i * 64 + (size_t)k] = (uint8_t)v;
        }
        scale[i] = bitnet_ptq1_0_fp16_to_fp32(a[i].d);
        scale[nb + i] = bitnet_ptq1_0_fp16_to_fp32(b[i].d);
    }
}

void bitnet_ptq1_0_p2_prep(void *prepv, const int8_t *xq, int n, int B) {
    ptq_p2prep_t *p = (ptq_p2prep_t *)prepv;
    const int nb = n / 128;
    const int pairs = (B + 1) / 2;
    const size_t bstride = (size_t)nb * 128;
    int i, t, b, j;

    p->nb = nb;
    p->B = B;
    p->pairs = pairs;
    p->at = (void *)((uint8_t *)prepv + sizeof(*p));
    p->rsum = (int32_t *)((uint8_t *)p->at + (size_t)nb * 16u * (size_t)pairs * 16u);
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (i = 0; i < nb; ++i)
        for (b = 0; b < B; ++b) {
            const int8_t *xb = xq + (size_t)b * bstride + (size_t)i * 128;
            int32_t s = 0;
            for (j = 0; j < 128; j += 16) s += vaddvq_s32(vpaddlq_s16(vpaddlq_s8(vld1q_s8(xb + j))));
            p->rsum[(size_t)i * B + b] = s;
        }
    for (i = 0; i < nb; ++i)
        for (t = 0; t < pairs; ++t) {
            const int t0 = 2 * t;
            const int t1 = (2 * t + 1 < B) ? 2 * t + 1 : B - 1;
            for (b = 0; b < 16; ++b) {
                const int8_t *lo = xq + (size_t)t0 * bstride + (size_t)i * 128 + 8 * b;
                const int8_t *hi = xq + (size_t)t1 * bstride + (size_t)i * 128 + 8 * b;
                ((int8x16_t *)p->at)[((size_t)i * 16 + b) * pairs + t] =
                    vcombine_s8(vld1_s8(lo), vld1_s8(hi));
            }
        }
#else
    (void)bstride; (void)j;
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON) && (defined(__GNUC__) || defined(__clang__))

/* Weight-stream prefetch distance in 64-byte blocks, 0 = off.  The verify is
 * the sum of a 7.3 GB weight stream and its compute, not the larger of the
 * two, so this is a probe for whether an explicit prefetch can overlap them. */
static int p2_pf_dist(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("PTQ_P2_PF");
        cached = e != NULL ? atoi(e) : 0;
    }
    return cached;
}

__attribute__((target("arch=armv8.6-a+i8mm"), noinline))
static void p2_decode(const uint8_t *p, int8x16_t e[16]) {
    const uint8x16_t m = vdupq_n_u8(0x03);
    int k;
    /* Four trits per byte, so the shifts are 0/2/4/6.  Written out instead of
     * looping over the field index: vshrq_n_u8 takes an immediate and clang
     * refuses the loop's non-constant argument (NDK builds). */
    for (k = 0; k < 4; ++k) {
        uint8x16_t v = vld1q_u8(p + 16 * k);
        e[0 * 4 + k] = vreinterpretq_s8_u8(vandq_u8(v, m));
        e[1 * 4 + k] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), m));
        e[2 * 4 + k] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), m));
        e[3 * 4 + k] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 6), m));
    }
}

__attribute__((target("arch=armv8.6-a+i8mm"), noinline))
static void p2_pair(const uint8_t *W, const float *d, const ptq_p2prep_t *p,
                    const float *xs, int n_out, int row0, float *yb) {
    const int nb = p->nb, B = p->B, pairs = p->pairs;
    float32x4_t acc[8];
    int i, t, b;
    const int pf = p2_pf_dist();

    /* Lane order of an SMMLA result is [row a,token 0 | row a,token 1 |
     * row a+1,token 0 | row a+1,token 1], which is exactly the output order
     * once the trit bias and the two block scales are folded in, so the whole
     * block reduces to a vector subtract, convert, scale and add. */
    for (t = 0; t < pairs; ++t) acc[t] = vdupq_n_f32(0.0f);
    for (i = 0; i < nb; ++i) {
        int8x16_t e[16];
        const float32x4_t dv = vcombine_f32(vdup_n_f32(d[i]), vdup_n_f32(d[nb + i]));
        if (pf > 0 && i + pf < nb)
            __builtin_prefetch(W + (size_t)(i + pf) * 64, 0, 3);
        p2_decode(W + (size_t)i * 64, e);
        for (t = 0; t < pairs; ++t) {
            int32x4_t s0 = vdupq_n_s32(0), s1 = s0, s2 = s0, s3 = s0, v, rv;
            const int8x16_t *at = (const int8x16_t *)p->at +
                                  ((size_t)i * 16) * (size_t)pairs + t;
            /* The trits are stored biased, so each lane is dot + sum(x) over
             * that token's 128 activations: subtract the token's sum once per
             * lane, i.e. [S(t0) S(t1) S(t0) S(t1)] in lane order, not one sum
             * per row. */
            const int32_t rs0 = p->rsum[(size_t)i * B + 2 * t];
            const int32_t rs1 = p->rsum[(size_t)i * B +
                                        ((2 * t + 1 < B) ? 2 * t + 1 : 2 * t)];
            for (b = 0; b < 16; b += 4) {
                s0 = vmmlaq_s32(s0, e[b + 0], at[(size_t)(b + 0) * (size_t)pairs]);
                s1 = vmmlaq_s32(s1, e[b + 1], at[(size_t)(b + 1) * (size_t)pairs]);
                s2 = vmmlaq_s32(s2, e[b + 2], at[(size_t)(b + 2) * (size_t)pairs]);
                s3 = vmmlaq_s32(s3, e[b + 3], at[(size_t)(b + 3) * (size_t)pairs]);
            }
            v = vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3));
            rv = vcombine_s32(vset_lane_s32(rs1, vdup_n_s32(rs0), 1),
                              vset_lane_s32(rs1, vdup_n_s32(rs0), 1));
            acc[t] = vmlaq_f32(acc[t], vcvtq_f32_s32(vsubq_s32(v, rv)), dv);
        }
    }
    for (t = 0; t < pairs; ++t) {
        float l[4];
        vst1q_f32(l, acc[t]);
        yb[(size_t)(2 * t) * (size_t)n_out + row0] = l[0] * xs[2 * t];
        yb[(size_t)(2 * t) * (size_t)n_out + row0 + 1] = l[2] * xs[2 * t];
        if (2 * t + 1 < B) {
            yb[(size_t)(2 * t + 1) * (size_t)n_out + row0] = l[1] * xs[2 * t + 1];
            yb[(size_t)(2 * t + 1) * (size_t)n_out + row0 + 1] = l[3] * xs[2 * t + 1];
        }
    }
}


/* The B <= 2 case (pairs == 1), which is the speculative verify's shape.
 *
 * The loop above materialises all sixteen decoded operands in e[16] before
 * running its SMMLAs.  That is the whole NEON register file, so the
 * accumulators, the tile pointers and half the operands spill - the
 * disassembly shows 61 sp accesses per 128-weight block.  Decoding one 16-byte
 * chunk into four operands and folding them straight into four independent
 * accumulator chains keeps four operands live instead of sixteen.  Byte k of
 * a 64-byte block carries column group c = k + 4f in field f, whose activation
 * tile is at[c], so the folds stay in tile order. */
__attribute__((target("arch=armv8.6-a+i8mm"), noinline))
static void p2_pair1(const uint8_t *W, const float *d, const ptq_p2prep_t *p,
                     const float *xs, int n_out, int row0, float *yb) {
    const int nb = p->nb, B = p->B;
    const int8x16_t *at = (const int8x16_t *)p->at;
    const uint8x16_t m = vdupq_n_u8(0x03);
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i, k;

    for (i = 0; i < nb; ++i) {
        int32x4_t s0 = vdupq_n_s32(0), s1 = s0, s2 = s0, s3 = s0, v, rv;
        const uint8_t *pb = W + (size_t)i * 64;
        const int8x16_t *tile = at + (size_t)i * 16;
        const float32x4_t dv = vcombine_f32(vdup_n_f32(d[i]), vdup_n_f32(d[nb + i]));
        const int32_t rs0 = p->rsum[(size_t)i * B];
        const int32_t rs1 = p->rsum[(size_t)i * B + ((B > 1) ? 1 : 0)];
        const int32x4_t bias = vcombine_s32(vset_lane_s32(rs1, vdup_n_s32(rs0), 1),
                                            vset_lane_s32(rs1, vdup_n_s32(rs0), 1));
        for (k = 0; k < 4; ++k) {
            const uint8x16_t w = vld1q_u8(pb + 16 * k);
            const int8x16_t e0 = vreinterpretq_s8_u8(vandq_u8(w, m));
            const int8x16_t e1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w, 2), m));
            const int8x16_t e2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w, 4), m));
            const int8x16_t e3 = vreinterpretq_s8_u8(vshrq_n_u8(w, 6));
            s0 = vmmlaq_s32(s0, e0, tile[k + 0]);
            s1 = vmmlaq_s32(s1, e1, tile[k + 4]);
            s2 = vmmlaq_s32(s2, e2, tile[k + 8]);
            s3 = vmmlaq_s32(s3, e3, tile[k + 12]);
        }
        v = vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3));
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vsubq_s32(v, bias)), dv);
    }
    {
        float l[4];
        vst1q_f32(l, acc);
        yb[row0] = l[0] * xs[0];
        yb[row0 + 1] = l[2] * xs[0];
        if (B > 1) {
            yb[(size_t)n_out + row0] = l[1] * xs[1];
            yb[(size_t)n_out + row0 + 1] = l[3] * xs[1];
        }
    }
}


/* Two row pairs at once.  The activation tiles are shared by every output row,
 * so a second pair reuses the sixteen tile loads of the first; what it adds is
 * four more accumulator chains and its own decode.  Per SMMLA that is 3.2
 * instructions against 4.0 for the single-pair loop, which matters because the
 * verify is instruction-issue bound at ~1.4 SMMLA/cycle. */
__attribute__((target("arch=armv8.6-a+i8mm"), noinline))
static void p2_pair2(const uint8_t *W, const float *d, const ptq_p2prep_t *p,
                     const float *xs, int n_out, int row0, float *yb) {
    const int nb = p->nb, B = p->B;
    const size_t pb = (size_t)nb * 64u;
    const int8x16_t *at = (const int8x16_t *)p->at;
    const uint8x16_t m = vdupq_n_u8(0x03);
    float32x4_t accA = vdupq_n_f32(0.0f), accB = accA;
    int i, k;

    for (i = 0; i < nb; ++i) {
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
        int32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        const uint8_t *pa = W + (size_t)i * 64;
        const uint8_t *pc = pa + pb;
        const int8x16_t *tile = at + (size_t)i * 16;
        const float32x4_t dvA = vcombine_f32(vdup_n_f32(d[i]), vdup_n_f32(d[nb + i]));
        const float32x4_t dvB = vcombine_f32(vdup_n_f32(d[2 * nb + i]),
                                             vdup_n_f32(d[3 * nb + i]));
        const int32x4_t bI = vcombine_s32(
            vset_lane_s32(p->rsum[(size_t)i * B + ((B > 1) ? 1 : 0)],
                          vdup_n_s32(p->rsum[(size_t)i * B]), 1),
            vset_lane_s32(p->rsum[(size_t)i * B + ((B > 1) ? 1 : 0)],
                          vdup_n_s32(p->rsum[(size_t)i * B]), 1));
        for (k = 0; k < 4; ++k) {
            const uint8x16_t wa = vld1q_u8(pa + 16 * k);
            const uint8x16_t wc = vld1q_u8(pc + 16 * k);
            a0 = vmmlaq_s32(a0, vreinterpretq_s8_u8(vandq_u8(wa, m)), tile[k]);
            b0 = vmmlaq_s32(b0, vreinterpretq_s8_u8(vandq_u8(wc, m)), tile[k]);
            a1 = vmmlaq_s32(a1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(wa, 2), m)), tile[k + 4]);
            b1 = vmmlaq_s32(b1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(wc, 2), m)), tile[k + 4]);
            a2 = vmmlaq_s32(a2, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(wa, 4), m)), tile[k + 8]);
            b2 = vmmlaq_s32(b2, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(wc, 4), m)), tile[k + 8]);
            a3 = vmmlaq_s32(a3, vreinterpretq_s8_u8(vshrq_n_u8(wa, 6)), tile[k + 12]);
            b3 = vmmlaq_s32(b3, vreinterpretq_s8_u8(vshrq_n_u8(wc, 6)), tile[k + 12]);
        }
        {
            const int32x4_t vA = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
            const int32x4_t vB = vaddq_s32(vaddq_s32(b0, b1), vaddq_s32(b2, b3));
            accA = vmlaq_f32(accA, vcvtq_f32_s32(vsubq_s32(vA, bI)), dvA);
            accB = vmlaq_f32(accB, vcvtq_f32_s32(vsubq_s32(vB, bI)), dvB);
        }
    }
    {
        float la[4], lb[4];
        vst1q_f32(la, accA);
        vst1q_f32(lb, accB);
        yb[row0] = la[0] * xs[0];
        yb[row0 + 1] = la[2] * xs[0];
        yb[row0 + 2] = lb[0] * xs[0];
        yb[row0 + 3] = lb[2] * xs[0];
        if (B > 1) {
            yb[(size_t)n_out + row0] = la[1] * xs[1];
            yb[(size_t)n_out + row0 + 1] = la[3] * xs[1];
            yb[(size_t)n_out + row0 + 2] = lb[1] * xs[1];
            yb[(size_t)n_out + row0 + 3] = lb[3] * xs[1];
        }
    }
}
#endif

void bitnet_ptq1_0_matmat_p2(const void *w2, const float *d, const void *prep,
                             const float *xscale, int B, int n_out, int row0, float *yb) {
#if defined(__aarch64__) && defined(__ARM_NEON) && (defined(__GNUC__) || defined(__clang__))
    const ptq_p2prep_t *pp = (const ptq_p2prep_t *)prep;
    if (pp->pairs == 1)
        p2_pair1((const uint8_t *)w2, d, pp, xscale, n_out, row0, yb);
    else
        p2_pair((const uint8_t *)w2, d, pp, xscale, n_out, row0, yb);
#else
    (void)w2; (void)d; (void)prep; (void)xscale; (void)B; (void)n_out; (void)row0; (void)yb;
#endif
}

void bitnet_ptq1_0_matmat_p2q(const void *w2, const float *d, const void *prep,
                              const float *xscale, int B, int n_out, int row0, float *yb) {
#if defined(__aarch64__) && defined(__ARM_NEON) && (defined(__GNUC__) || defined(__clang__))
    p2_pair2((const uint8_t *)w2, d, (const ptq_p2prep_t *)prep, xscale, n_out, row0, yb);
#else
    (void)w2; (void)d; (void)prep; (void)xscale; (void)B; (void)n_out; (void)row0; (void)yb;
#endif
}

float bitnet_ptq1_0_dot_row_q8(const void *src, const void *xq, int n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    const int nb = n / BITNET_PTQ1_0_QK;
#if defined(__GNUC__) || defined(__clang__)
    if (ptq_have_dotprod()) {
        if (ptq_unroll2())
            return ptq_dot_q8_dotprod2((const bitnet_ptq1_0_block_t *)src,
                                       (const bitnet_q8_0_block_t *)xq, nb);
        return ptq_dot_q8_dotprod((const bitnet_ptq1_0_block_t *)src,
                                  (const bitnet_q8_0_block_t *)xq, nb);
    }
#endif
    return ptq_dot_q8_neon((const bitnet_ptq1_0_block_t *)src,
                           (const bitnet_q8_0_block_t *)xq, nb);
#else
    return ptq_dot_q8_scalar(src, xq, n);
#endif
}

static const unsigned enc_pow3[5] = {1, 3, 9, 27, 81};

static inline float enc_bf16(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* One base-3 digit of a packed byte; the exact inverse of the decoder's
 * ptq_trit_scalar(): digit = ((byte * 3^n mod 256) * 3) >> 8. */
static unsigned enc_digit(unsigned b, unsigned p3) {
    unsigned q = ((b * p3) & 0xFFu);
    return (q * 3u) >> 8u;
}

static uint8_t enc_tab[243];
static int enc_ok = -1;

/* accumulator (five digits, 0..242) -> the byte that decodes to it */
static int enc_init(void) {
    int acc, b;
    if (enc_ok >= 0) return enc_ok;
    enc_ok = 0;
    for (acc = 0; acc < 243; ++acc) {
        for (b = 0; b < 256; ++b) {
            unsigned a2 = 0;
            int n;
            for (n = 0; n < 5; ++n) a2 += enc_digit((unsigned)b, enc_pow3[n]) * enc_pow3[n];
            if ((int)a2 == acc) { enc_tab[acc] = (uint8_t)b; break; }
        }
        if (b == 256) {
            return 0;
            return 0;
        }
    }
    enc_ok = 1;
    return 1;
}

static uint16_t enc_f16(float f) {
    uint32_t bits, s, m;
    int e;
    memcpy(&bits, &f, sizeof(bits));
    s = (bits >> 16) & 0x8000u;
    e = (int)((bits >> 23) & 0xFFu) - 127 + 15;
    m = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) return (uint16_t)(s | 0x7C00u | (m ? 0x200u : 0u));
    if (e <= 0) {
        int sh;
        uint32_t hm, rem, half;
        if (e < -10) return (uint16_t)s;
        m |= 0x800000u;
        sh = 14 - e;
        hm = m >> sh;
        rem = m & ((1u << sh) - 1u);
        half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1u))) ++hm;
        return (uint16_t)(s | hm);
    }
    if (e >= 31) return (uint16_t)(s | 0x7C00u);
    {
        uint32_t hm = m >> 13, rem = m & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
            if (++hm == 0x400u) { hm = 0; if (++e >= 31) return (uint16_t)(s | 0x7C00u); }
        }
        return (uint16_t)(s | ((uint32_t)e << 10) | hm);
    }
}

/* One row of bf16 weights to PTQ1_0: per 128-block absmean scale, round to
 * nearest trit, then pack the digits in the order block_decode() reads them. */
static void enc_row_bf16(const uint16_t *src, uint8_t *dst, int n) {
    int nb = n / BITNET_PTQ1_0_QK, blk, e;
    for (blk = 0; blk < nb; ++blk) {
        const uint16_t *sp = src + (size_t)blk * BITNET_PTQ1_0_QK;
        bitnet_ptq1_0_block_t *bp =
            (bitnet_ptq1_0_block_t *)(dst + (size_t)blk * BITNET_PTQ1_0_BLOCK_SIZE);
        unsigned acc[BITNET_PTQ1_0_QS_SIZE + BITNET_PTQ1_0_QH_SIZE];
        float v[BITNET_PTQ1_0_QK];
        float sum = 0.0f, d;
        int m;
        for (e = 0; e < BITNET_PTQ1_0_QK; ++e) {
            v[e] = enc_bf16(sp[e]);
            sum += fabsf(v[e]);
        }
        d = sum / (float)BITNET_PTQ1_0_QK;
        if (!(d > 1e-8f)) d = 1e-8f;
        /* round against the fp16 scale that is actually stored, so the
         * decoder reproduces d * clip(round(w / d)) element for element */
        d = bitnet_ptq1_0_fp16_to_fp32(enc_f16(d));
        if (!(d > 1e-8f)) d = 1e-8f;
        memset(acc, 0, sizeof(acc));
        for (e = 0; e < BITNET_PTQ1_0_QK; ++e) {
            int r = (int)lrintf(v[e] / d);
            unsigned dig, p3;
            if (r < -1) r = -1;
            if (r > 1) r = 1;
            dig = (unsigned)(r + 1);
            if (e < 80) {                 /* qs[0..15], digit e/16 */
                m = e & 15;
                p3 = enc_pow3[e >> 4];
            } else if (e < 120) {         /* qs[16..23], digit (e-80)/8 */
                m = 16 + ((e - 80) & 7);
                p3 = enc_pow3[(e - 80) >> 3];
            } else {                      /* qh[0..1], digit (e-120)/2 */
                m = BITNET_PTQ1_0_QS_SIZE + ((e - 120) & 1);
                p3 = enc_pow3[(e - 120) >> 1];
            }
            acc[m] += dig * p3;
        }
        for (m = 0; m < BITNET_PTQ1_0_QS_SIZE; ++m) bp->qs[m] = enc_tab[acc[m]];
        for (m = 0; m < BITNET_PTQ1_0_QH_SIZE; ++m)
            bp->qh[m] = enc_tab[acc[BITNET_PTQ1_0_QS_SIZE + m]];
        bp->d = enc_f16(d);
    }
}

/* Encode one row of bf16 weights (n a multiple of 128) into PTQ1_0 blocks.
 * Scale is the block absmean and each weight rounds to the nearest trit, the
 * pair the decoder inverts; used by the DSpine drafter, whose 4.05 GB of bf16
 * weights become 0.5 GB this way.  Returns 0, or -1 if n is not a multiple of
 * the block or the byte table cannot represent every digit tuple. */
int bitnet_ptq1_0_quantize_row(const void *src, void *dst, int n) {
    if (n <= 0 || n % BITNET_PTQ1_0_QK != 0) return -1;
    if (!enc_init()) return -1;
    enc_row_bf16((const uint16_t *)src, (uint8_t *)dst, n);
    return 0;
}
