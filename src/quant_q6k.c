#include "quant_q6k.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t raw = sign << 31;
            float f;
            memcpy(&f, &raw, sizeof(f));
            return f;
        }
        /* Denormalized: value = (-1)^sign * 2^(-14) * (mant/1024) */
        float value = ldexpf((float)mant / 1024.0f, -14);
        return sign ? -value : value;
    }

    if (exp == 31) {
        uint32_t raw = (sign << 31) | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &raw, sizeof(f));
        return f;
    }

    exp = exp - 15u + 127u;
    mant <<= 13;

    uint32_t raw = (sign << 31) | (exp << 23) | mant;
    float f;
    memcpy(&f, &raw, sizeof(f));
    return f;
}

int bitnet_q6k_dequantize_block(const bitnet_q6k_block_t *block, float *out, size_t out_len) {
    int n = 0;
    int l = 0;

    if (block == NULL || out == NULL || out_len < BITNET_Q6K_QK) {
        return -1;
    }

    const float d = fp16_to_fp32(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;

    for (n = 0; n < BITNET_Q6K_QK; n += 128) {
        for (l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l + 0] = d * (float)sc[is + 0] * (float)q1;
            out[l + 32] = d * (float)sc[is + 2] * (float)q2;
            out[l + 64] = d * (float)sc[is + 4] * (float)q3;
            out[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        out += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }

    return 0;
}

int bitnet_q6k_dot_product(const bitnet_q6k_block_t *block, const float *vec, size_t len, float *out) {
    int n = 0;
    float sum = 0.0f;

    if (block == NULL || vec == NULL || out == NULL || len < BITNET_Q6K_QK) {
        return -1;
    }

    const float d = fp16_to_fp32(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;

    for (n = 0; n < BITNET_Q6K_QK; n += 128) {
        const float *v = vec + n;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        float acc4 = 0.0f;
        float acc5 = 0.0f;
        float acc6 = 0.0f;
        float acc7 = 0.0f;

        for (int l = 0; l < 16; ++l) {
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc0 += (float)q1 * v[l + 0];
            acc2 += (float)q2 * v[l + 32];
            acc4 += (float)q3 * v[l + 64];
            acc6 += (float)q4 * v[l + 96];
        }

        for (int l = 16; l < 32; ++l) {
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc1 += (float)q1 * v[l + 0];
            acc3 += (float)q2 * v[l + 32];
            acc5 += (float)q3 * v[l + 64];
            acc7 += (float)q4 * v[l + 96];
        }

        sum += d * (
            (float)sc[0] * acc0 +
            (float)sc[1] * acc1 +
            (float)sc[2] * acc2 +
            (float)sc[3] * acc3 +
            (float)sc[4] * acc4 +
            (float)sc[5] * acc5 +
            (float)sc[6] * acc6 +
            (float)sc[7] * acc7);

        ql += 64;
        qh += 32;
        sc += 8;
    }

    *out = sum;
    return 0;
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int32_t q6k_neon_dot_i8_16(int8x16_t coeff, int8x16_t values) {
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, coeff, values);
    return vaddvq_s32(acc);
}

#define Q6K_NEON_UNPACK_LOW(ql_vec, qh_vec, shift) \
    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8( \
                  vandq_u8((ql_vec), vdupq_n_u8(0x0f)), \
                  vshlq_n_u8(vandq_u8(vshrq_n_u8((qh_vec), (shift)), vdupq_n_u8(3)), 4))), \
              vdupq_n_s8(32))

#define Q6K_NEON_UNPACK_LOW0(ql_vec, qh_vec) \
    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8( \
                  vandq_u8((ql_vec), vdupq_n_u8(0x0f)), \
                  vshlq_n_u8(vandq_u8((qh_vec), vdupq_n_u8(3)), 4))), \
              vdupq_n_s8(32))

#define Q6K_NEON_UNPACK_HIGH(ql_vec, qh_vec, shift) \
    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8( \
                  vshrq_n_u8((ql_vec), 4), \
                  vshlq_n_u8(vandq_u8(vshrq_n_u8((qh_vec), (shift)), vdupq_n_u8(3)), 4))), \
              vdupq_n_s8(32))
#endif

int bitnet_q6k_dot_product_i8_neon(const bitnet_q6k_block_t *block, const int8_t *qvec,
                                   float vec_scale, size_t len, float *out) {
    int n = 0;
    float sum = 0.0f;

    if (block == NULL || qvec == NULL || out == NULL || len < BITNET_Q6K_QK) {
        return -1;
    }

    const float d = fp16_to_fp32(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;

    for (n = 0; n < BITNET_Q6K_QK; n += 128) {
        const int8_t *v = qvec + n;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const uint8x16_t ql0 = vld1q_u8(ql + 0);
        const uint8x16_t ql1 = vld1q_u8(ql + 16);
        const uint8x16_t ql2 = vld1q_u8(ql + 32);
        const uint8x16_t ql3 = vld1q_u8(ql + 48);
        const uint8x16_t qh0 = vld1q_u8(qh + 0);
        const uint8x16_t qh1 = vld1q_u8(qh + 16);
        int32_t acc0 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_LOW0(ql0, qh0), vld1q_s8(v + 0));
        int32_t acc1 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_LOW0(ql1, qh1), vld1q_s8(v + 16));
        int32_t acc2 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_LOW(ql2, qh0, 2), vld1q_s8(v + 32));
        int32_t acc3 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_LOW(ql3, qh1, 2), vld1q_s8(v + 48));
        int32_t acc4 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_HIGH(ql0, qh0, 4), vld1q_s8(v + 64));
        int32_t acc5 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_HIGH(ql1, qh1, 4), vld1q_s8(v + 80));
        int32_t acc6 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_HIGH(ql2, qh0, 6), vld1q_s8(v + 96));
        int32_t acc7 = q6k_neon_dot_i8_16(Q6K_NEON_UNPACK_HIGH(ql3, qh1, 6), vld1q_s8(v + 112));
#else
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;
        int32_t acc4 = 0;
        int32_t acc5 = 0;
        int32_t acc6 = 0;
        int32_t acc7 = 0;

        for (int l = 0; l < 16; ++l) {
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc0 += q1 * (int)v[l + 0];
            acc2 += q2 * (int)v[l + 32];
            acc4 += q3 * (int)v[l + 64];
            acc6 += q4 * (int)v[l + 96];
        }

        for (int l = 16; l < 32; ++l) {
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc1 += q1 * (int)v[l + 0];
            acc3 += q2 * (int)v[l + 32];
            acc5 += q3 * (int)v[l + 64];
            acc7 += q4 * (int)v[l + 96];
        }
#endif

        sum += d * vec_scale * (
            (float)sc[0] * (float)acc0 +
            (float)sc[1] * (float)acc1 +
            (float)sc[2] * (float)acc2 +
            (float)sc[3] * (float)acc3 +
            (float)sc[4] * (float)acc4 +
            (float)sc[5] * (float)acc5 +
            (float)sc[6] * (float)acc6 +
            (float)sc[7] * (float)acc7);

        ql += 64;
        qh += 32;
        sc += 8;
    }

    *out = sum;
    return 0;
}

static inline int8_t q6k_unpack_low(uint8_t ql, uint8_t qh, int shift) {
    return (int8_t)((ql & 0x0fu) | (((qh >> shift) & 3u) << 4)) - 32;
}

static inline int8_t q6k_unpack_high(uint8_t ql, uint8_t qh, int shift) {
    return (int8_t)((ql >> 4) | (((qh >> shift) & 3u) << 4)) - 32;
}

int bitnet_q6k_expand_to_q8(const void *weight, int rows, int cols,
                            int8_t *q8, float *scales) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || q8 == NULL || scales == NULL ||
        rows <= 0 || cols <= 0 || cols % BITNET_Q6K_QK != 0) {
        return -1;
    }

    blocks_per_row = cols / BITNET_Q6K_QK;
    for (int row = 0; row < rows; ++row) {
        int8_t *row_q8 = q8 + (size_t)row * (size_t)cols;
        float *row_scales = scales + (size_t)row * (size_t)blocks_per_row * 16u;
        const uint8_t *row_bytes = bytes + (size_t)row * (size_t)blocks_per_row * BITNET_Q6K_BLOCK_SIZE;

        for (int b = 0; b < blocks_per_row; ++b) {
            const bitnet_q6k_block_t *block =
                (const bitnet_q6k_block_t *)(row_bytes + (size_t)b * BITNET_Q6K_BLOCK_SIZE);
            const float d = fp16_to_fp32(block->d);
            int8_t *dst = row_q8 + (size_t)b * BITNET_Q6K_QK;
            float *sc_dst = row_scales + (size_t)b * 16u;

            for (int half = 0; half < 2; ++half) {
                const uint8_t *ql = block->ql + (size_t)half * 64u;
                const uint8_t *qh = block->qh + (size_t)half * 32u;
                const int8_t *sc = block->scales + (size_t)half * 8u;
                int8_t *dst_h = dst + half * 128;

                for (int s = 0; s < 8; ++s) {
                    sc_dst[half * 8 + s] = d * (float)sc[s];
                }

                for (int l = 0; l < 32; ++l) {
                    dst_h[l + 0] = q6k_unpack_low(ql[l + 0], qh[l], 0);
                    dst_h[l + 32] = q6k_unpack_low(ql[l + 32], qh[l], 2);
                    dst_h[l + 64] = q6k_unpack_high(ql[l + 0], qh[l], 4);
                    dst_h[l + 96] = q6k_unpack_high(ql[l + 32], qh[l], 6);
                }
            }
        }
    }

    return 0;
}

int bitnet_q6k_expand_to_q8_compact(const void *weight, int rows, int cols,
                                    int8_t *q8, int8_t *scales, float *d) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || q8 == NULL || scales == NULL || d == NULL ||
        rows <= 0 || cols <= 0 || cols % BITNET_Q6K_QK != 0) {
        return -1;
    }

    blocks_per_row = cols / BITNET_Q6K_QK;
    for (int row = 0; row < rows; ++row) {
        int8_t *row_q8 = q8 + (size_t)row * (size_t)cols;
        int8_t *row_scales = scales + (size_t)row * (size_t)blocks_per_row * 16u;
        float *row_d = d + (size_t)row * (size_t)blocks_per_row;
        const uint8_t *row_bytes = bytes + (size_t)row * (size_t)blocks_per_row * BITNET_Q6K_BLOCK_SIZE;

        for (int b = 0; b < blocks_per_row; ++b) {
            const bitnet_q6k_block_t *block =
                (const bitnet_q6k_block_t *)(row_bytes + (size_t)b * BITNET_Q6K_BLOCK_SIZE);
            int8_t *dst = row_q8 + (size_t)b * BITNET_Q6K_QK;
            int8_t *sc_dst = row_scales + (size_t)b * 16u;

            row_d[b] = fp16_to_fp32(block->d);
            memcpy(sc_dst, block->scales, 16u * sizeof(*sc_dst));

            for (int half = 0; half < 2; ++half) {
                const uint8_t *ql = block->ql + (size_t)half * 64u;
                const uint8_t *qh = block->qh + (size_t)half * 32u;
                int8_t *dst_h = dst + half * 128;

                for (int l = 0; l < 32; ++l) {
                    dst_h[l + 0] = q6k_unpack_low(ql[l + 0], qh[l], 0);
                    dst_h[l + 32] = q6k_unpack_low(ql[l + 32], qh[l], 2);
                    dst_h[l + 64] = q6k_unpack_high(ql[l + 0], qh[l], 4);
                    dst_h[l + 96] = q6k_unpack_high(ql[l + 32], qh[l], 6);
                }
            }
        }
    }

    return 0;
}

int bitnet_q6k_dot_product_q8_neon(const int8_t *q8, const float *scales,
                                   int blocks_per_row, const int8_t *qvec,
                                   float vec_scale, float *out) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t sum_v = vdupq_n_f32(0.0f);
#else
    float sum = 0.0f;
#endif

    if (q8 == NULL || scales == NULL || qvec == NULL || out == NULL ||
        blocks_per_row <= 0) {
        return -1;
    }

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const float *sc = scales + (size_t)b * 16u;

        for (int g = 0; g < 16; ++g) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, vld1q_s8(q + g * 16), vld1q_s8(v + g * 16));
            sum_v = vfmaq_n_f32(sum_v, vcvtq_f32_s32(acc), sc[g]);
#else
            int32_t acc = 0;
            for (int i = 0; i < 16; ++i) {
                acc += (int32_t)q[g * 16 + i] * (int32_t)v[g * 16 + i];
            }
            sum += (float)acc * sc[g];
#endif
        }
    }

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    *out = vaddvq_f32(sum_v) * vec_scale;
#else
    *out = sum * vec_scale;
#endif
    return 0;
}

int bitnet_q6k_dot_product_q8_4_neon(const int8_t *q8, const float *scales,
                                     int row_stride, int scale_stride,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float out[4]) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t sum0_v = vdupq_n_f32(0.0f);
    float32x4_t sum1_v = vdupq_n_f32(0.0f);
    float32x4_t sum2_v = vdupq_n_f32(0.0f);
    float32x4_t sum3_v = vdupq_n_f32(0.0f);
#else
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
#endif

    if (q8 == NULL || scales == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

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
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            const int8x16_t vv = vld1q_s8(v + g * 16);
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            acc0 = vdotq_s32(acc0, vld1q_s8(q0 + g * 16), vv);
            acc1 = vdotq_s32(acc1, vld1q_s8(q1 + g * 16), vv);
            acc2 = vdotq_s32(acc2, vld1q_s8(q2 + g * 16), vv);
            acc3 = vdotq_s32(acc3, vld1q_s8(q3 + g * 16), vv);
            sum0_v = vfmaq_n_f32(sum0_v, vcvtq_f32_s32(acc0), s0[g]);
            sum1_v = vfmaq_n_f32(sum1_v, vcvtq_f32_s32(acc1), s1[g]);
            sum2_v = vfmaq_n_f32(sum2_v, vcvtq_f32_s32(acc2), s2[g]);
            sum3_v = vfmaq_n_f32(sum3_v, vcvtq_f32_s32(acc3), s3[g]);
#else
            int32_t acc0 = 0;
            int32_t acc1 = 0;
            int32_t acc2 = 0;
            int32_t acc3 = 0;
            for (int i = 0; i < 16; ++i) {
                const int vi = (int)v[g * 16 + i];
                acc0 += (int)q0[g * 16 + i] * vi;
                acc1 += (int)q1[g * 16 + i] * vi;
                acc2 += (int)q2[g * 16 + i] * vi;
                acc3 += (int)q3[g * 16 + i] * vi;
            }
            sum0 += (float)acc0 * s0[g];
            sum1 += (float)acc1 * s1[g];
            sum2 += (float)acc2 * s2[g];
            sum3 += (float)acc3 * s3[g];
#endif
        }
    }

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    out[0] = vaddvq_f32(sum0_v) * vec_scale;
    out[1] = vaddvq_f32(sum1_v) * vec_scale;
    out[2] = vaddvq_f32(sum2_v) * vec_scale;
    out[3] = vaddvq_f32(sum3_v) * vec_scale;
#else
    out[0] = sum0 * vec_scale;
    out[1] = sum1 * vec_scale;
    out[2] = sum2 * vec_scale;
    out[3] = sum3 * vec_scale;
#endif
    return 0;
}

int bitnet_q6k_dot_product_q8_8_neon(const int8_t *q8, const float *scales,
                                     int row_stride, int scale_stride,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float out[8]) {
    if (q8 == NULL || scales == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t sum0_v = vdupq_n_f32(0.0f);
    float32x4_t sum1_v = vdupq_n_f32(0.0f);
    float32x4_t sum2_v = vdupq_n_f32(0.0f);
    float32x4_t sum3_v = vdupq_n_f32(0.0f);
    float32x4_t sum4_v = vdupq_n_f32(0.0f);
    float32x4_t sum5_v = vdupq_n_f32(0.0f);
    float32x4_t sum6_v = vdupq_n_f32(0.0f);
    float32x4_t sum7_v = vdupq_n_f32(0.0f);

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q0 = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *q1 = q0 + (size_t)row_stride;
        const int8_t *q2 = q1 + (size_t)row_stride;
        const int8_t *q3 = q2 + (size_t)row_stride;
        const int8_t *q4 = q3 + (size_t)row_stride;
        const int8_t *q5 = q4 + (size_t)row_stride;
        const int8_t *q6 = q5 + (size_t)row_stride;
        const int8_t *q7 = q6 + (size_t)row_stride;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const float *s0 = scales + (size_t)b * 16u;
        const float *s1 = s0 + (size_t)scale_stride;
        const float *s2 = s1 + (size_t)scale_stride;
        const float *s3 = s2 + (size_t)scale_stride;
        const float *s4 = s3 + (size_t)scale_stride;
        const float *s5 = s4 + (size_t)scale_stride;
        const float *s6 = s5 + (size_t)scale_stride;
        const float *s7 = s6 + (size_t)scale_stride;

        for (int g = 0; g < 16; ++g) {
            const int8x16_t vv = vld1q_s8(v + g * 16);
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0);
            int32x4_t acc7 = vdupq_n_s32(0);
            acc0 = vdotq_s32(acc0, vld1q_s8(q0 + g * 16), vv);
            acc1 = vdotq_s32(acc1, vld1q_s8(q1 + g * 16), vv);
            acc2 = vdotq_s32(acc2, vld1q_s8(q2 + g * 16), vv);
            acc3 = vdotq_s32(acc3, vld1q_s8(q3 + g * 16), vv);
            acc4 = vdotq_s32(acc4, vld1q_s8(q4 + g * 16), vv);
            acc5 = vdotq_s32(acc5, vld1q_s8(q5 + g * 16), vv);
            acc6 = vdotq_s32(acc6, vld1q_s8(q6 + g * 16), vv);
            acc7 = vdotq_s32(acc7, vld1q_s8(q7 + g * 16), vv);
            sum0_v = vfmaq_n_f32(sum0_v, vcvtq_f32_s32(acc0), s0[g]);
            sum1_v = vfmaq_n_f32(sum1_v, vcvtq_f32_s32(acc1), s1[g]);
            sum2_v = vfmaq_n_f32(sum2_v, vcvtq_f32_s32(acc2), s2[g]);
            sum3_v = vfmaq_n_f32(sum3_v, vcvtq_f32_s32(acc3), s3[g]);
            sum4_v = vfmaq_n_f32(sum4_v, vcvtq_f32_s32(acc4), s4[g]);
            sum5_v = vfmaq_n_f32(sum5_v, vcvtq_f32_s32(acc5), s5[g]);
            sum6_v = vfmaq_n_f32(sum6_v, vcvtq_f32_s32(acc6), s6[g]);
            sum7_v = vfmaq_n_f32(sum7_v, vcvtq_f32_s32(acc7), s7[g]);
        }
    }

    out[0] = vaddvq_f32(sum0_v) * vec_scale;
    out[1] = vaddvq_f32(sum1_v) * vec_scale;
    out[2] = vaddvq_f32(sum2_v) * vec_scale;
    out[3] = vaddvq_f32(sum3_v) * vec_scale;
    out[4] = vaddvq_f32(sum4_v) * vec_scale;
    out[5] = vaddvq_f32(sum5_v) * vec_scale;
    out[6] = vaddvq_f32(sum6_v) * vec_scale;
    out[7] = vaddvq_f32(sum7_v) * vec_scale;
#else
    if (bitnet_q6k_dot_product_q8_4_neon(q8, scales, row_stride, scale_stride,
                                         blocks_per_row, qvec, vec_scale, out) != 0) {
        return -1;
    }
    if (bitnet_q6k_dot_product_q8_4_neon(q8 + 4 * (size_t)row_stride,
                                         scales + 4 * (size_t)scale_stride,
                                         row_stride, scale_stride,
                                         blocks_per_row, qvec, vec_scale, out + 4) != 0) {
        return -1;
    }
#endif

    return 0;
}

int bitnet_q6k_dot_product_q8_compact_neon(const int8_t *q8, const int8_t *scales,
                                           const float *d, int blocks_per_row,
                                           const int8_t *qvec, float vec_scale, float *out) {
    if (q8 == NULL || scales == NULL || d == NULL || qvec == NULL || out == NULL ||
        blocks_per_row <= 0) {
        return -1;
    }

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t sum_v = vdupq_n_f32(0.0f);

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const int8_t *sc = scales + (size_t)b * 16u;
        int32x4_t block = vdupq_n_s32(0);

        for (int g = 0; g < 16; ++g) {
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, vld1q_s8(q + g * 16), vld1q_s8(v + g * 16));
            block = vmlaq_n_s32(block, acc, (int32_t)sc[g]);
        }
        sum_v = vfmaq_n_f32(sum_v, vcvtq_f32_s32(block), d[b]);
    }

    *out = vaddvq_f32(sum_v) * vec_scale;
#else
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        int32_t block_sum = 0;
        for (int g = 0; g < 16; ++g) {
            int32_t acc = 0;
            for (int i = 0; i < 16; ++i) {
                int idx = b * BITNET_Q6K_QK + g * 16 + i;
                acc += (int32_t)q8[idx] * (int32_t)qvec[idx];
            }
            block_sum += acc * (int32_t)scales[b * 16 + g];
        }
        sum += (float)block_sum * d[b];
    }
    *out = sum * vec_scale;
#endif
    return 0;
}

int bitnet_q6k_dot_product_q8_compact_4_neon(const int8_t *q8, const int8_t *scales,
                                             const float *d, int row_stride,
                                             int scale_stride, int d_stride,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float out[4]) {
    if (q8 == NULL || scales == NULL || d == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || d_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t sum0_v = vdupq_n_f32(0.0f);
    float32x4_t sum1_v = vdupq_n_f32(0.0f);
    float32x4_t sum2_v = vdupq_n_f32(0.0f);
    float32x4_t sum3_v = vdupq_n_f32(0.0f);

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
        const float d0 = d[b];
        const float d1 = d[(size_t)d_stride + (size_t)b];
        const float d2 = d[(size_t)d_stride * 2u + (size_t)b];
        const float d3 = d[(size_t)d_stride * 3u + (size_t)b];
        int32x4_t block0 = vdupq_n_s32(0);
        int32x4_t block1 = vdupq_n_s32(0);
        int32x4_t block2 = vdupq_n_s32(0);
        int32x4_t block3 = vdupq_n_s32(0);

        for (int g = 0; g < 16; ++g) {
            const int8x16_t vv = vld1q_s8(v + g * 16);
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            acc0 = vdotq_s32(acc0, vld1q_s8(q0 + g * 16), vv);
            acc1 = vdotq_s32(acc1, vld1q_s8(q1 + g * 16), vv);
            acc2 = vdotq_s32(acc2, vld1q_s8(q2 + g * 16), vv);
            acc3 = vdotq_s32(acc3, vld1q_s8(q3 + g * 16), vv);
            block0 = vmlaq_n_s32(block0, acc0, (int32_t)s0[g]);
            block1 = vmlaq_n_s32(block1, acc1, (int32_t)s1[g]);
            block2 = vmlaq_n_s32(block2, acc2, (int32_t)s2[g]);
            block3 = vmlaq_n_s32(block3, acc3, (int32_t)s3[g]);
        }

        sum0_v = vfmaq_n_f32(sum0_v, vcvtq_f32_s32(block0), d0);
        sum1_v = vfmaq_n_f32(sum1_v, vcvtq_f32_s32(block1), d1);
        sum2_v = vfmaq_n_f32(sum2_v, vcvtq_f32_s32(block2), d2);
        sum3_v = vfmaq_n_f32(sum3_v, vcvtq_f32_s32(block3), d3);
    }

    out[0] = vaddvq_f32(sum0_v) * vec_scale;
    out[1] = vaddvq_f32(sum1_v) * vec_scale;
    out[2] = vaddvq_f32(sum2_v) * vec_scale;
    out[3] = vaddvq_f32(sum3_v) * vec_scale;
#else
    for (int row = 0; row < 4; ++row) {
        float sum = 0.0f;
        const int8_t *row_q8 = q8 + (size_t)row * (size_t)row_stride;
        const int8_t *row_scales = scales + (size_t)row * (size_t)scale_stride;
        const float *row_d = d + (size_t)row * (size_t)d_stride;
        for (int b = 0; b < blocks_per_row; ++b) {
            int32_t block_sum = 0;
            for (int g = 0; g < 16; ++g) {
                int32_t acc = 0;
                for (int i = 0; i < 16; ++i) {
                    int idx = b * BITNET_Q6K_QK + g * 16 + i;
                    acc += (int32_t)row_q8[idx] * (int32_t)qvec[idx];
                }
                block_sum += acc * (int32_t)row_scales[b * 16 + g];
            }
            sum += (float)block_sum * row_d[b];
        }
        out[row] = sum * vec_scale;
    }
#endif
    return 0;
}

int bitnet_q6k_dot_product_q8_compact_8_neon(const int8_t *q8, const int8_t *scales,
                                             const float *d, int row_stride,
                                             int scale_stride, int d_stride,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float out[8]) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (q8 == NULL || scales == NULL || d == NULL || qvec == NULL || out == NULL ||
        row_stride <= 0 || scale_stride <= 0 || d_stride <= 0 || blocks_per_row <= 0) {
        return -1;
    }

    float32x4_t sum0_v = vdupq_n_f32(0.0f);
    float32x4_t sum1_v = vdupq_n_f32(0.0f);
    float32x4_t sum2_v = vdupq_n_f32(0.0f);
    float32x4_t sum3_v = vdupq_n_f32(0.0f);
    float32x4_t sum4_v = vdupq_n_f32(0.0f);
    float32x4_t sum5_v = vdupq_n_f32(0.0f);
    float32x4_t sum6_v = vdupq_n_f32(0.0f);
    float32x4_t sum7_v = vdupq_n_f32(0.0f);

    for (int b = 0; b < blocks_per_row; ++b) {
        const int8_t *q0 = q8 + (size_t)b * BITNET_Q6K_QK;
        const int8_t *q1 = q0 + (size_t)row_stride;
        const int8_t *q2 = q1 + (size_t)row_stride;
        const int8_t *q3 = q2 + (size_t)row_stride;
        const int8_t *q4 = q3 + (size_t)row_stride;
        const int8_t *q5 = q4 + (size_t)row_stride;
        const int8_t *q6 = q5 + (size_t)row_stride;
        const int8_t *q7 = q6 + (size_t)row_stride;
        const int8_t *v = qvec + (size_t)b * BITNET_Q6K_QK;
        const int8_t *s0 = scales + (size_t)b * 16u;
        const int8_t *s1 = s0 + (size_t)scale_stride;
        const int8_t *s2 = s1 + (size_t)scale_stride;
        const int8_t *s3 = s2 + (size_t)scale_stride;
        const int8_t *s4 = s3 + (size_t)scale_stride;
        const int8_t *s5 = s4 + (size_t)scale_stride;
        const int8_t *s6 = s5 + (size_t)scale_stride;
        const int8_t *s7 = s6 + (size_t)scale_stride;
        const float d0 = d[b];
        const float d1 = d[(size_t)d_stride + (size_t)b];
        const float d2 = d[(size_t)d_stride * 2u + (size_t)b];
        const float d3 = d[(size_t)d_stride * 3u + (size_t)b];
        const float d4 = d[(size_t)d_stride * 4u + (size_t)b];
        const float d5 = d[(size_t)d_stride * 5u + (size_t)b];
        const float d6 = d[(size_t)d_stride * 6u + (size_t)b];
        const float d7 = d[(size_t)d_stride * 7u + (size_t)b];
        int32x4_t block0 = vdupq_n_s32(0);
        int32x4_t block1 = vdupq_n_s32(0);
        int32x4_t block2 = vdupq_n_s32(0);
        int32x4_t block3 = vdupq_n_s32(0);
        int32x4_t block4 = vdupq_n_s32(0);
        int32x4_t block5 = vdupq_n_s32(0);
        int32x4_t block6 = vdupq_n_s32(0);
        int32x4_t block7 = vdupq_n_s32(0);

        for (int g = 0; g < 16; ++g) {
            const int off = g * 16;
            const int8x16_t vv = vld1q_s8(v + off);
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0);
            int32x4_t acc7 = vdupq_n_s32(0);
            acc0 = vdotq_s32(acc0, vld1q_s8(q0 + off), vv);
            acc1 = vdotq_s32(acc1, vld1q_s8(q1 + off), vv);
            acc2 = vdotq_s32(acc2, vld1q_s8(q2 + off), vv);
            acc3 = vdotq_s32(acc3, vld1q_s8(q3 + off), vv);
            acc4 = vdotq_s32(acc4, vld1q_s8(q4 + off), vv);
            acc5 = vdotq_s32(acc5, vld1q_s8(q5 + off), vv);
            acc6 = vdotq_s32(acc6, vld1q_s8(q6 + off), vv);
            acc7 = vdotq_s32(acc7, vld1q_s8(q7 + off), vv);
            block0 = vmlaq_n_s32(block0, acc0, (int32_t)s0[g]);
            block1 = vmlaq_n_s32(block1, acc1, (int32_t)s1[g]);
            block2 = vmlaq_n_s32(block2, acc2, (int32_t)s2[g]);
            block3 = vmlaq_n_s32(block3, acc3, (int32_t)s3[g]);
            block4 = vmlaq_n_s32(block4, acc4, (int32_t)s4[g]);
            block5 = vmlaq_n_s32(block5, acc5, (int32_t)s5[g]);
            block6 = vmlaq_n_s32(block6, acc6, (int32_t)s6[g]);
            block7 = vmlaq_n_s32(block7, acc7, (int32_t)s7[g]);
        }

        sum0_v = vfmaq_n_f32(sum0_v, vcvtq_f32_s32(block0), d0);
        sum1_v = vfmaq_n_f32(sum1_v, vcvtq_f32_s32(block1), d1);
        sum2_v = vfmaq_n_f32(sum2_v, vcvtq_f32_s32(block2), d2);
        sum3_v = vfmaq_n_f32(sum3_v, vcvtq_f32_s32(block3), d3);
        sum4_v = vfmaq_n_f32(sum4_v, vcvtq_f32_s32(block4), d4);
        sum5_v = vfmaq_n_f32(sum5_v, vcvtq_f32_s32(block5), d5);
        sum6_v = vfmaq_n_f32(sum6_v, vcvtq_f32_s32(block6), d6);
        sum7_v = vfmaq_n_f32(sum7_v, vcvtq_f32_s32(block7), d7);
    }

    out[0] = vaddvq_f32(sum0_v) * vec_scale;
    out[1] = vaddvq_f32(sum1_v) * vec_scale;
    out[2] = vaddvq_f32(sum2_v) * vec_scale;
    out[3] = vaddvq_f32(sum3_v) * vec_scale;
    out[4] = vaddvq_f32(sum4_v) * vec_scale;
    out[5] = vaddvq_f32(sum5_v) * vec_scale;
    out[6] = vaddvq_f32(sum6_v) * vec_scale;
    out[7] = vaddvq_f32(sum7_v) * vec_scale;
    return 0;
#else
    if (bitnet_q6k_dot_product_q8_compact_4_neon(q8, scales, d,
                                                 row_stride, scale_stride, d_stride,
                                                 blocks_per_row, qvec, vec_scale, out) != 0) {
        return -1;
    }
    return bitnet_q6k_dot_product_q8_compact_4_neon(
        q8 + 4 * (size_t)row_stride,
        scales + 4 * (size_t)scale_stride,
        d + 4 * (size_t)d_stride,
        row_stride, scale_stride, d_stride,
        blocks_per_row, qvec, vec_scale, out + 4);
#endif
}
