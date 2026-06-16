#include "quant_tq2_0.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define BITNET_TQ2_HAS_NEON_DOTPROD 1
#else
#define BITNET_TQ2_HAS_NEON_DOTPROD 0
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

static pthread_once_t g_tq2_code_lut_once = PTHREAD_ONCE_INIT;
static float g_tq2_code_coeff[4][256];

static void init_tq2_code_lut_once(void) {
    for (int code = 0; code < 256; ++code) {
        g_tq2_code_coeff[0][code] = (float)((code & 3) - 1);
        g_tq2_code_coeff[1][code] = (float)(((code >> 2) & 3) - 1);
        g_tq2_code_coeff[2][code] = (float)(((code >> 4) & 3) - 1);
        g_tq2_code_coeff[3][code] = (float)(((code >> 6) & 3) - 1);
    }
}

/* TQ2_0 interleaved packing layout (matching llama.cpp exactly):
 *
 * Each block has qs[64] bytes encoding 256 ternary weights.
 * The packing uses 32-byte groups with 4 sub-values per byte
 * spanning 32-element strides (interleaved layout):
 *
 *   for j in {0, 32}:
 *     for l in {0, 1, 2, 3}:
 *       for m in {0..31}:
 *         q = (qs[j + m] >> (l*2)) & 3
 *         output position = (j/32)*128 + l*32 + m
 *         value = (q - 1) * d
 *
 * 2-bit code: 00=-1, 01=0, 10=+1, 11=reserved(0)
 * Decoding: value = (code - 1) * d
 */

int bitnet_tq2_0_dequantize_block(const bitnet_tq2_0_block_t *block, float *out, size_t out_len) {
    if (block == NULL || out == NULL || out_len < BITNET_TQ2_0_QK) {
        return -1;
    }

    float d = fp16_to_fp32(block->d);

    /* Interleaved layout matching llama.cpp dequantize_row_tq2_0 */
    for (int j = 0; j < BITNET_TQ2_0_QS_SIZE; j += 32) {
        for (int l = 0; l < 4; ++l) {
            for (int m = 0; m < 32; ++m) {
                int8_t q = (block->qs[j + m] >> (l * 2)) & 3;
                *out++ = (float)(q - 1) * d;
            }
        }
    }

    return 0;
}

int bitnet_tq2_0_dot_product(const bitnet_tq2_0_block_t *block, const float *vec, size_t len, float *out) {
    if (block == NULL || vec == NULL || out == NULL || len < BITNET_TQ2_0_QK) {
        return -1;
    }

    float d = fp16_to_fp32(block->d);
    float sum = 0.0f;

    /* Interleaved layout matching llama.cpp dequantize_row_tq2_0 */
    for (int j = 0; j < BITNET_TQ2_0_QS_SIZE; j += 32) {
        for (int l = 0; l < 4; ++l) {
            for (int m = 0; m < 32; ++m) {
                int8_t q = (block->qs[j + m] >> (l * 2)) & 3;
                int vec_idx = (j / 32) * 128 + l * 32 + m;
                sum += (float)(q - 1) * vec[vec_idx];
            }
        }
    }

    *out = sum * d;
    return 0;
}

/* Ternary matmul kernel with correct interleaved element ordering.
 *
 * For 1.58-bit weights {-1, 0, +1}, the multiply w*v is trivial:
 *   -1 * v = -v  (negate)
 *    0 * v =  0  (skip)
 *   +1 * v =  v  (add)
 *
 * Using (q-1) * vec[i] is branchless — the compiler can
 * vectorize this with -O3, and the values {-1, 0, 1} make
 * the multiply equivalent to conditional add/subtract.
 */
static void tq2_0_matmul_row(const uint8_t *weight, int blocks_per_row,
                              const float *vec, float *out_val) {
    float sum = 0.0f;

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block = (const bitnet_tq2_0_block_t *)(weight + block_offset);
        float d = fp16_to_fp32(block->d);
        float block_sum = 0.0f;
        int vec_base = k * BITNET_TQ2_0_QK;

        /* Interleaved layout matching llama.cpp dequantize_row_tq2_0 */
        for (int j = 0; j < BITNET_TQ2_0_QS_SIZE; j += 32) {
            for (int l = 0; l < 4; ++l) {
                for (int m = 0; m < 32; ++m) {
                    int8_t q = (block->qs[j + m] >> (l * 2)) & 3;
                    int vec_idx = vec_base + (j / 32) * 128 + l * 32 + m;
                    block_sum += (float)(q - 1) * vec[vec_idx];
                }
            }
        }

        sum += block_sum * d;
    }

    *out_val = sum;
}

static void build_tq2_0_vec_lut_blocks(const float *vec, int blocks_per_row, float *lut) {
    (void)pthread_once(&g_tq2_code_lut_once, init_tq2_code_lut_once);

    for (int k = 0; k < blocks_per_row; ++k) {
        const float *vec_block = vec + (size_t)k * BITNET_TQ2_0_QK;
        for (int g = 0; g < 2; ++g) {
            const float *v = vec_block + (size_t)g * 128;
            const float *c0 = g_tq2_code_coeff[0];
            const float *c1 = g_tq2_code_coeff[1];
            const float *c2 = g_tq2_code_coeff[2];
            const float *c3 = g_tq2_code_coeff[3];
            for (int m = 0; m < 32; ++m) {
                float *entry = lut + (((size_t)k * 2u + (size_t)g) * 32u + (size_t)m) * 256u;
                const float v0 = v[m];
                const float v1 = v[32 + m];
                const float v2 = v[64 + m];
                const float v3 = v[96 + m];

                for (int code = 0; code < 256; ++code) {
                    entry[code] = c0[code] * v0 + c1[code] * v1 +
                                  c2[code] * v2 + c3[code] * v3;
                }
            }
        }
    }
}

size_t bitnet_tq2_0_lut_float_count(int in_dim) {
    if (in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return 0;
    }
    return (size_t)(in_dim / BITNET_TQ2_0_QK) * 2u * 32u * 256u;
}

int bitnet_tq2_0_build_vec_lut(const float *vec, int in_dim, float *lut, size_t lut_count) {
    int blocks_per_row = 0;
    size_t required = 0;

    if (vec == NULL || lut == NULL || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    required = bitnet_tq2_0_lut_float_count(in_dim);
    if (lut_count < required || required == 0) {
        return -1;
    }

    build_tq2_0_vec_lut_blocks(vec, blocks_per_row, lut);
    return 0;
}

size_t bitnet_tq2_0_scale_float_count(int out_dim, int in_dim) {
    if (out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return 0;
    }
    return (size_t)out_dim * (size_t)(in_dim / BITNET_TQ2_0_QK);
}

int bitnet_tq2_0_build_scales(const void *weight, int out_dim, int in_dim,
                              float *scales, size_t scale_count) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;
    size_t required = 0;

    if (weight == NULL || scales == NULL || out_dim <= 0 ||
        in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    required = bitnet_tq2_0_scale_float_count(out_dim, in_dim);
    if (scale_count < required || required == 0) {
        return -1;
    }

    for (int row = 0; row < out_dim; ++row) {
        size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        for (int k = 0; k < blocks_per_row; ++k) {
            const bitnet_tq2_0_block_t *block =
                (const bitnet_tq2_0_block_t *)(bytes + row_offset +
                    (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            scales[(size_t)row * (size_t)blocks_per_row + (size_t)k] = fp16_to_fp32(block->d);
        }
    }

    return 0;
}

int bitnet_tq2_0_quantize_vec_i8(const float *vec, int in_dim, int8_t *qvec, float *scale, int32_t *block_bsums) {
    float max_abs = 0.0f;
    int bsums_computed = 0;

    if (vec == NULL || qvec == NULL || scale == NULL || in_dim <= 0) {
        return -1;
    }

#if defined(__ARM_NEON)
    /* NEON-accelerated max_abs */
    {
        float32x4_t max_vec = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 3 < in_dim; i += 4) {
            float32x4_t v = vld1q_f32(vec + i);
            max_vec = vmaxq_f32(max_vec, vabsq_f32(v));
        }
        max_abs = vmaxvq_f32(max_vec);
        for (; i < in_dim; ++i) {
            float a = fabsf(vec[i]);
            if (a > max_abs) max_abs = a;
        }
    }
#else
    for (int i = 0; i < in_dim; ++i) {
        float a = fabsf(vec[i]);
        if (a > max_abs) max_abs = a;
    }
#endif

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

#if defined(__ARM_NEON)
    /* NEON-accelerated quantize: float -> int8 with clamping */
    {
        float32x4_t inv_s = vdupq_n_f32(inv_scale);
        float32x4_t lo = vdupq_n_f32(-127.0f);
        float32x4_t hi = vdupq_n_f32(127.0f);
        int n_blocks = in_dim / BITNET_TQ2_0_QK;
        int i = 0;
        if (block_bsums != NULL && in_dim % BITNET_TQ2_0_QK == 0) {
            for (int block = 0; block < n_blocks; ++block) {
                int16x8_t bsum16 = vdupq_n_s16(0);
                const float *src = vec + (size_t)block * BITNET_TQ2_0_QK;
                int8_t *dst = qvec + (size_t)block * BITNET_TQ2_0_QK;

                for (int j = 0; j < BITNET_TQ2_0_QK; j += 16) {
                    float32x4_t v0 = vmulq_f32(vld1q_f32(src + j + 0), inv_s);
                    float32x4_t v1 = vmulq_f32(vld1q_f32(src + j + 4), inv_s);
                    float32x4_t v2 = vmulq_f32(vld1q_f32(src + j + 8), inv_s);
                    float32x4_t v3 = vmulq_f32(vld1q_f32(src + j + 12), inv_s);
                    int16x8_t lo16 = vcombine_s16(
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v0, lo), hi))),
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v1, lo), hi))));
                    int16x8_t hi16 = vcombine_s16(
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v2, lo), hi))),
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v3, lo), hi))));
                    int8x16_t i8 = vcombine_s8(vqmovn_s16(lo16), vqmovn_s16(hi16));
                    vst1q_s8(dst + j, i8);
                    bsum16 = vaddq_s16(bsum16, vpaddlq_s8(i8));
                }
                block_bsums[block] = (int32_t)vaddlvq_s16(bsum16);
            }
            i = n_blocks * BITNET_TQ2_0_QK;
        } else {
            if (block_bsums != NULL) {
                memset(block_bsums, 0, (size_t)n_blocks * sizeof(*block_bsums));
            }
            for (; i + 15 < in_dim; i += 16) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(vec + i + 0), inv_s);
                float32x4_t v1 = vmulq_f32(vld1q_f32(vec + i + 4), inv_s);
                float32x4_t v2 = vmulq_f32(vld1q_f32(vec + i + 8), inv_s);
                float32x4_t v3 = vmulq_f32(vld1q_f32(vec + i + 12), inv_s);
                int16x8_t lo16 = vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v0, lo), hi))),
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v1, lo), hi))));
                int16x8_t hi16 = vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v2, lo), hi))),
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v3, lo), hi))));
                int8x16_t i8 = vcombine_s8(vqmovn_s16(lo16), vqmovn_s16(hi16));
                vst1q_s8(qvec + i, i8);
                if (block_bsums != NULL) {
                    int block = i / BITNET_TQ2_0_QK;
                    if (block < n_blocks) {
                        block_bsums[block] += (int32_t)vaddlvq_s8(i8);
                    }
                }
            }
        }
        for (; i < in_dim; ++i) {
            int q = (int)lrintf(vec[i] * inv_scale);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            qvec[i] = (int8_t)q;
            if (block_bsums != NULL) {
                int block = i / BITNET_TQ2_0_QK;
                if (block < n_blocks) {
                    block_bsums[block] += q;
                }
            }
        }
        bsums_computed = (block_bsums != NULL);
    }
#else
    for (int i = 0; i < in_dim; ++i) {
        int q = (int)lrintf(vec[i] * inv_scale);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        qvec[i] = (int8_t)q;
    }
#endif

    /* Compute per-block bsums using NEON */
    if (block_bsums != NULL && !bsums_computed) {
        int n_blocks = in_dim / BITNET_TQ2_0_QK;
#if defined(__ARM_NEON)
        for (int k = 0; k < n_blocks; ++k) {
            const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
            int32_t bsum = 0;
            for (int i = 0; i < BITNET_TQ2_0_QK; i += 16) {
                bsum += (int32_t)vaddlvq_s8(vld1q_s8(qv + i));
            }
            block_bsums[k] = bsum;
        }
#else
        for (int k = 0; k < n_blocks; ++k) {
            int32_t bsum = 0;
            const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
            for (int i = 0; i < BITNET_TQ2_0_QK; ++i) {
                bsum += (int32_t)qv[i];
            }
            block_bsums[k] = bsum;
        }
#endif
    }
    return 0;
}

int bitnet_tq2_0_quantize_vec_i8_known_max(const float *vec, int in_dim, int8_t *qvec,
                                           float *scale, int32_t *block_bsums,
                                           float max_abs) {
    int bsums_computed = 0;

    if (vec == NULL || qvec == NULL || scale == NULL || in_dim <= 0) {
        return -1;
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

#if defined(__ARM_NEON)
    {
        float32x4_t inv_s = vdupq_n_f32(inv_scale);
        float32x4_t lo = vdupq_n_f32(-127.0f);
        float32x4_t hi = vdupq_n_f32(127.0f);
        int n_blocks = in_dim / BITNET_TQ2_0_QK;
        int i = 0;
        if (block_bsums != NULL && in_dim % BITNET_TQ2_0_QK == 0) {
            for (int block = 0; block < n_blocks; ++block) {
                int16x8_t bsum16 = vdupq_n_s16(0);
                const float *src = vec + (size_t)block * BITNET_TQ2_0_QK;
                int8_t *dst = qvec + (size_t)block * BITNET_TQ2_0_QK;

                for (int j = 0; j < BITNET_TQ2_0_QK; j += 16) {
                    float32x4_t v0 = vmulq_f32(vld1q_f32(src + j + 0), inv_s);
                    float32x4_t v1 = vmulq_f32(vld1q_f32(src + j + 4), inv_s);
                    float32x4_t v2 = vmulq_f32(vld1q_f32(src + j + 8), inv_s);
                    float32x4_t v3 = vmulq_f32(vld1q_f32(src + j + 12), inv_s);
                    int16x8_t lo16 = vcombine_s16(
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v0, lo), hi))),
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v1, lo), hi))));
                    int16x8_t hi16 = vcombine_s16(
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v2, lo), hi))),
                        vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v3, lo), hi))));
                    int8x16_t i8 = vcombine_s8(vqmovn_s16(lo16), vqmovn_s16(hi16));
                    vst1q_s8(dst + j, i8);
                    bsum16 = vaddq_s16(bsum16, vpaddlq_s8(i8));
                }
                block_bsums[block] = (int32_t)vaddlvq_s16(bsum16);
            }
            i = n_blocks * BITNET_TQ2_0_QK;
        } else {
            if (block_bsums != NULL) {
                memset(block_bsums, 0, (size_t)n_blocks * sizeof(*block_bsums));
            }
            for (; i + 15 < in_dim; i += 16) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(vec + i + 0), inv_s);
                float32x4_t v1 = vmulq_f32(vld1q_f32(vec + i + 4), inv_s);
                float32x4_t v2 = vmulq_f32(vld1q_f32(vec + i + 8), inv_s);
                float32x4_t v3 = vmulq_f32(vld1q_f32(vec + i + 12), inv_s);
                int16x8_t lo16 = vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v0, lo), hi))),
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v1, lo), hi))));
                int16x8_t hi16 = vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v2, lo), hi))),
                    vqmovn_s32(vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v3, lo), hi))));
                int8x16_t i8 = vcombine_s8(vqmovn_s16(lo16), vqmovn_s16(hi16));
                vst1q_s8(qvec + i, i8);
                if (block_bsums != NULL) {
                    int block = i / BITNET_TQ2_0_QK;
                    if (block < n_blocks) {
                        block_bsums[block] += (int32_t)vaddlvq_s8(i8);
                    }
                }
            }
        }
        for (; i < in_dim; ++i) {
            int q = (int)lrintf(vec[i] * inv_scale);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            qvec[i] = (int8_t)q;
            if (block_bsums != NULL) {
                int block = i / BITNET_TQ2_0_QK;
                if (block < n_blocks) {
                    block_bsums[block] += q;
                }
            }
        }
        bsums_computed = (block_bsums != NULL);
    }
#else
    for (int i = 0; i < in_dim; ++i) {
        int q = (int)lrintf(vec[i] * inv_scale);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        qvec[i] = (int8_t)q;
    }
#endif

    if (block_bsums != NULL && !bsums_computed) {
        int n_blocks = in_dim / BITNET_TQ2_0_QK;
#if defined(__ARM_NEON)
        for (int k = 0; k < n_blocks; ++k) {
            const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
            int32_t bsum = 0;
            for (int i = 0; i < BITNET_TQ2_0_QK; i += 16) {
                bsum += (int32_t)vaddlvq_s8(vld1q_s8(qv + i));
            }
            block_bsums[k] = bsum;
        }
#else
        for (int k = 0; k < n_blocks; ++k) {
            int32_t bsum = 0;
            const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
            for (int i = 0; i < BITNET_TQ2_0_QK; ++i) {
                bsum += (int32_t)qv[i];
            }
            block_bsums[k] = bsum;
        }
#endif
    }
    return 0;
}

size_t bitnet_tq2_0_i16_lut_count(int in_dim) {
    return bitnet_tq2_0_lut_float_count(in_dim);
}

int bitnet_tq2_0_build_i16_lut(const int8_t *qvec, int in_dim,
                               int16_t *lut, size_t lut_count) {
    int blocks_per_row = 0;
    size_t required = 0;

    if (qvec == NULL || lut == NULL || in_dim <= 0 ||
        in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    required = bitnet_tq2_0_i16_lut_count(in_dim);
    if (lut_count < required || required == 0) {
        return -1;
    }

    for (int k = 0; k < blocks_per_row; ++k) {
        const int8_t *vec_block = qvec + (size_t)k * BITNET_TQ2_0_QK;
        for (int g = 0; g < 2; ++g) {
            const int8_t *v = vec_block + (size_t)g * 128;
            for (int m = 0; m < 32; ++m) {
                int16_t *entry = lut + (((size_t)k * 2u + (size_t)g) * 32u + (size_t)m) * 256u;
                const int v0 = (int)v[m];
                const int v1 = (int)v[32 + m];
                const int v2 = (int)v[64 + m];
                const int v3 = (int)v[96 + m];
                for (int code = 0; code < 256; ++code) {
                    const int sum =
                        ((int)(code & 3) - 1) * v0 +
                        ((int)((code >> 2) & 3) - 1) * v1 +
                        ((int)((code >> 4) & 3) - 1) * v2 +
                        ((int)((code >> 6) & 3) - 1) * v3;
                    entry[code] = (int16_t)sum;
                }
            }
        }
    }

    return 0;
}

static void tq2_0_matmul_row_i16_lut_scales(const uint8_t *weight, const float *scales,
                                            int blocks_per_row, const int16_t *lut,
                                            float vec_scale, float *out_val) {
    float sum = 0.0f;

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block = (const bitnet_tq2_0_block_t *)(weight + block_offset);
        int block_sum = 0;

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs = block->qs + (size_t)g * 32u;
            const int16_t *group_lut = lut + (((size_t)k * 2u + (size_t)g) * 32u * 256u);
            for (int m = 0; m < 32; ++m) {
                block_sum += (int)group_lut[(size_t)m * 256u + qs[m]];
            }
        }

        sum += (float)block_sum * scales[k] * vec_scale;
    }

    *out_val = sum;
}

static int tq2_0_matmul_rows_i16_lut_parallel(const uint8_t *bytes, const float *scales,
                                              int blocks_per_row,
                                              int out_start, int out_count,
                                              const int16_t *lut, float vec_scale,
                                              float *out);

int bitnet_tq2_0_matmul_vector_i16_lut_scales(const void *weight, const float *scales,
                                              int out_dim, int in_dim,
                                              const int16_t *lut, float vec_scale,
                                              float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || scales == NULL || lut == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) {
        return -1;
    }

    return tq2_0_matmul_rows_i16_lut_parallel(bytes, scales, blocks_per_row,
                                              0, out_dim, lut, vec_scale, out);
}

static void tq2_0_matmul_row_i8_scales(const uint8_t *weight, const float *scales,
                                       int blocks_per_row, const int8_t *qvec,
                                       float vec_scale, float *out_val) {
    float sum = 0.0f;

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block = (const bitnet_tq2_0_block_t *)(weight + block_offset);
        const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
        int block_sum = 0;

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs = block->qs + (size_t)g * 32u;
            const int8_t *v = qv + (size_t)g * 128u;
            for (int m = 0; m < 32; ++m) {
                const uint8_t code = qs[m];
                block_sum += ((int)(code & 3u) - 1) * (int)v[m];
                block_sum += ((int)((code >> 2) & 3u) - 1) * (int)v[32 + m];
                block_sum += ((int)((code >> 4) & 3u) - 1) * (int)v[64 + m];
                block_sum += ((int)((code >> 6) & 3u) - 1) * (int)v[96 + m];
            }
        }

        sum += (float)block_sum * scales[k] * vec_scale;
    }

    *out_val = sum;
}

#if BITNET_TQ2_HAS_NEON_DOTPROD

/* VTBL nibble lookup tables for TQ2_0 dequantization (bsums variant).
 *
 * 2-bit code: 00=0, 01=1, 10=2, 11=0(reserved)
 * Using vqtbl1q_s8, one table lookup replaces AND+USHR+AND (4 instr -> 1-2 instr).
 * The bsums trick: instead of (code-1)*y, compute code*y then subtract sum(y).
 * This eliminates the subtraction from the unpack path.
 *
 * After vshrq_n_u8, the byte's low 4 bits are used as the VTBL index.
 * Bits [3:2] are "don't care" — the LUT must map all 16 indices correctly
 * based on bits [1:0] only, matching the code mapping:
 *   bits[1:0]=00 -> 0  (indices 0,4,8,12)
 *   bits[1:0]=01 -> 1  (indices 1,5,9,13)
 *   bits[1:0]=10 -> 2  (indices 2,6,10,14)
 *   bits[1:0]=11 -> 0  (indices 3,7,11,15)  reserved, maps to 0
 *
 * All 4 shift positions share the same LUT pattern. */
static const int8_t g_tq2_nibble_lut_data[16] = {
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
};

/* Macro: VTBL nibble unpack for TQ2_0 (bsums variant).
 * Uses vqtbl1q_s8 to look up the code value {0, 1, 2, 0}.
 * NOTE: vqtbl1q_s8 returns 0 for indices >= 16 (out of range).
 * After vshrq_n_u8 by >= 4 bits, the value is always 0-15 (safe for VTBL).
 * For shift < 4, we must mask with 0x0F to keep the index in range. */
#define TQ2_VTBL_NIBBLE_UNPACK(codes_vec, lut_vec, shift_val) \
    vqtbl1q_s8(lut_vec, vshrq_n_u8(codes_vec, shift_val))

/* shift=0: mask with 0x0F to keep index in VTBL range */
#define TQ2_VTBL_NIBBLE_UNPACK_0(codes_vec, lut_vec, mask_0f) \
    vqtbl1q_s8(lut_vec, vandq_u8(codes_vec, mask_0f))

/* shift=2: shift then mask with 0x0F */
#define TQ2_VTBL_NIBBLE_UNPACK_2(codes_vec, lut_vec, mask_0f) \
    vqtbl1q_s8(lut_vec, vandq_u8(vshrq_n_u8(codes_vec, 2), mask_0f))

/* Macro: unpack one 16-byte code vector into 4 code vectors and dot-product with 4 value vectors.
 * Bsums variant: codes are {0,1,2} not {-1,0,1}, so no subtraction in unpack.
 * The caller must subtract bsum * scales_sum from the final result. */
#define TQ2_UNPACK_AND_DOT_4(acc, codes, lut, m0f, v0, v1, v2, v3) \
    do { \
        int8x16_t _c0 = TQ2_VTBL_NIBBLE_UNPACK_0(codes, lut, m0f); \
        int8x16_t _c1 = TQ2_VTBL_NIBBLE_UNPACK_2(codes, lut, m0f); \
        int8x16_t _c2 = TQ2_VTBL_NIBBLE_UNPACK(codes, lut, 4); \
        int8x16_t _c3 = TQ2_VTBL_NIBBLE_UNPACK(codes, lut, 6); \
        acc = vdotq_s32(acc, _c0, v0); \
        acc = vdotq_s32(acc, _c1, v1); \
        acc = vdotq_s32(acc, _c2, v2); \
        acc = vdotq_s32(acc, _c3, v3); \
    } while (0)

/* Dual-accumulator variant: splits 4 dot products across 2 accumulators
 * to break the dependency chain and improve ILP.
 * acc_lo gets dot(c0,v0) + dot(c2,v2), acc_hi gets dot(c1,v1) + dot(c3,v3).
 * Caller must combine: acc = vaddq_s32(acc_lo, acc_hi) at the end. */
#define TQ2_UNPACK_AND_DOT_4_DUAL(acc_lo, acc_hi, codes, lut, m0f, v0, v1, v2, v3) \
    do { \
        int8x16_t _c0 = TQ2_VTBL_NIBBLE_UNPACK_0(codes, lut, m0f); \
        int8x16_t _c1 = TQ2_VTBL_NIBBLE_UNPACK_2(codes, lut, m0f); \
        int8x16_t _c2 = TQ2_VTBL_NIBBLE_UNPACK(codes, lut, 4); \
        int8x16_t _c3 = TQ2_VTBL_NIBBLE_UNPACK(codes, lut, 6); \
        acc_lo = vdotq_s32(acc_lo, _c0, v0); \
        acc_hi = vdotq_s32(acc_hi, _c1, v1); \
        acc_lo = vdotq_s32(acc_lo, _c2, v2); \
        acc_hi = vdotq_s32(acc_hi, _c3, v3); \
    } while (0)

static void tq2_0_matmul_row_i8_neon_scales(const uint8_t *weight, const float *scales,
                                            int blocks_per_row, const int8_t *qvec,
                                            float vec_scale, const int32_t *block_bsums, float *out_val) {
    float sum = 0.0f;
    const int8x16_t nibble_lut = vld1q_s8(g_tq2_nibble_lut_data);
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block = (const bitnet_tq2_0_block_t *)(weight + block_offset);
        const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
        int32x4_t acc_lo = vdupq_n_s32(0);
        int32x4_t acc_hi = vdupq_n_s32(0);

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs = block->qs + (size_t)g * 32u;
            const int8_t *v = qv + (size_t)g * 128u;

            uint8x16x2_t qs_pair = vld1q_u8_x2(qs);

            for (int p = 0; p < 2; ++p) {
                uint8x16_t codes = qs_pair.val[p];
                int off = p * 16;

                TQ2_UNPACK_AND_DOT_4_DUAL(acc_lo, acc_hi, codes, nibble_lut, mask_0f,
                    vld1q_s8(v + off), vld1q_s8(v + 32 + off),
                    vld1q_s8(v + 64 + off), vld1q_s8(v + 96 + off));
            }
        }

        /* Combine dual accumulators */
        int32x4_t block_acc = vaddq_s32(acc_lo, acc_hi);
        float block_result = (float)(vaddvq_s32(block_acc) - block_bsums[k]);
        sum += block_result * scales[k] * vec_scale;
    }

    *out_val = sum;
}

static void tq2_0_matmul_row_i8_neon_pair_scales(const uint8_t *weight_a, const float *scales_a,
                                                 const uint8_t *weight_b, const float *scales_b,
                                                 int blocks_per_row, const int8_t *qvec,
                                                 float vec_scale, const int32_t *block_bsums, float *out_a, float *out_b) {
    float sum_a = 0.0f, sum_b = 0.0f;
    const int8x16_t nibble_lut = vld1q_s8(g_tq2_nibble_lut_data);
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block_a = (const bitnet_tq2_0_block_t *)(weight_a + block_offset);
        const bitnet_tq2_0_block_t *block_b = (const bitnet_tq2_0_block_t *)(weight_b + block_offset);
        const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
        int32x4_t acc_a_lo = vdupq_n_s32(0), acc_a_hi = vdupq_n_s32(0);
        int32x4_t acc_b_lo = vdupq_n_s32(0), acc_b_hi = vdupq_n_s32(0);

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs_a = block_a->qs + (size_t)g * 32u;
            const uint8_t *qs_b = block_b->qs + (size_t)g * 32u;
            const int8_t *v = qv + (size_t)g * 128u;

            uint8x16x2_t qs_pair_a = vld1q_u8_x2(qs_a);
            uint8x16x2_t qs_pair_b = vld1q_u8_x2(qs_b);

            for (int p = 0; p < 2; ++p) {
                uint8x16_t codes_a = qs_pair_a.val[p];
                uint8x16_t codes_b = qs_pair_b.val[p];
                int off = p * 16;
                int8x16_t v0 = vld1q_s8(v + off);
                int8x16_t v1 = vld1q_s8(v + 32 + off);
                int8x16_t v2 = vld1q_s8(v + 64 + off);
                int8x16_t v3 = vld1q_s8(v + 96 + off);

                TQ2_UNPACK_AND_DOT_4_DUAL(acc_a_lo, acc_a_hi, codes_a, nibble_lut, mask_0f, v0, v1, v2, v3);
                TQ2_UNPACK_AND_DOT_4_DUAL(acc_b_lo, acc_b_hi, codes_b, nibble_lut, mask_0f, v0, v1, v2, v3);
            }
        }

        int32x4_t acc_a = vaddq_s32(acc_a_lo, acc_a_hi);
        int32x4_t acc_b = vaddq_s32(acc_b_lo, acc_b_hi);
        float block_result_a = (float)(vaddvq_s32(acc_a) - block_bsums[k]);
        float block_result_b = (float)(vaddvq_s32(acc_b) - block_bsums[k]);
        sum_a += block_result_a * scales_a[k] * vec_scale;
        sum_b += block_result_b * scales_b[k] * vec_scale;
    }

    *out_a = sum_a;
    *out_b = sum_b;
}

/* 4-row parallel kernel: compute dot products of 4 weight rows against the same qvec.
 * Maximizes data reuse of qvec (loaded once, used by 4 rows) and VTBL LUT tables.
 * Bsums variant: subtract block_bsum per block.
 * Uses dual accumulators to break vdotq dependency chains. */
static void tq2_0_matmul_row_i8_neon_quad_scales(
    const uint8_t *weight_0, const float *scales_0,
    const uint8_t *weight_1, const float *scales_1,
    const uint8_t *weight_2, const float *scales_2,
    const uint8_t *weight_3, const float *scales_3,
    int blocks_per_row, const int8_t *qvec,
    float vec_scale, const int32_t *block_bsums,
    float *out_0, float *out_1, float *out_2, float *out_3) {

    float sum_0 = 0.0f, sum_1 = 0.0f, sum_2 = 0.0f, sum_3 = 0.0f;
    const int8x16_t nibble_lut = vld1q_s8(g_tq2_nibble_lut_data);
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *blk_0 = (const bitnet_tq2_0_block_t *)(weight_0 + block_offset);
        const bitnet_tq2_0_block_t *blk_1 = (const bitnet_tq2_0_block_t *)(weight_1 + block_offset);
        const bitnet_tq2_0_block_t *blk_2 = (const bitnet_tq2_0_block_t *)(weight_2 + block_offset);
        const bitnet_tq2_0_block_t *blk_3 = (const bitnet_tq2_0_block_t *)(weight_3 + block_offset);
        const int8_t *qv = qvec + (size_t)k * BITNET_TQ2_0_QK;
        int32x4_t acc_0_lo = vdupq_n_s32(0), acc_0_hi = vdupq_n_s32(0);
        int32x4_t acc_1_lo = vdupq_n_s32(0), acc_1_hi = vdupq_n_s32(0);
        int32x4_t acc_2_lo = vdupq_n_s32(0), acc_2_hi = vdupq_n_s32(0);
        int32x4_t acc_3_lo = vdupq_n_s32(0), acc_3_hi = vdupq_n_s32(0);

        for (int g = 0; g < 2; ++g) {
            uint8x16x2_t qs_0 = vld1q_u8_x2(blk_0->qs + (size_t)g * 32u);
            uint8x16x2_t qs_1 = vld1q_u8_x2(blk_1->qs + (size_t)g * 32u);
            uint8x16x2_t qs_2 = vld1q_u8_x2(blk_2->qs + (size_t)g * 32u);
            uint8x16x2_t qs_3 = vld1q_u8_x2(blk_3->qs + (size_t)g * 32u);
            const int8_t *v = qv + (size_t)g * 128u;

            for (int p = 0; p < 2; ++p) {
                int off = p * 16;
                int8x16_t v0 = vld1q_s8(v + off);
                int8x16_t v1 = vld1q_s8(v + 32 + off);
                int8x16_t v2 = vld1q_s8(v + 64 + off);
                int8x16_t v3 = vld1q_s8(v + 96 + off);

                TQ2_UNPACK_AND_DOT_4_DUAL(acc_0_lo, acc_0_hi, qs_0.val[p], nibble_lut, mask_0f, v0, v1, v2, v3);
                TQ2_UNPACK_AND_DOT_4_DUAL(acc_1_lo, acc_1_hi, qs_1.val[p], nibble_lut, mask_0f, v0, v1, v2, v3);
                TQ2_UNPACK_AND_DOT_4_DUAL(acc_2_lo, acc_2_hi, qs_2.val[p], nibble_lut, mask_0f, v0, v1, v2, v3);
                TQ2_UNPACK_AND_DOT_4_DUAL(acc_3_lo, acc_3_hi, qs_3.val[p], nibble_lut, mask_0f, v0, v1, v2, v3);
            }
        }

        sum_0 += (float)(vaddvq_s32(vaddq_s32(acc_0_lo, acc_0_hi)) - block_bsums[k]) * scales_0[k] * vec_scale;
        sum_1 += (float)(vaddvq_s32(vaddq_s32(acc_1_lo, acc_1_hi)) - block_bsums[k]) * scales_1[k] * vec_scale;
        sum_2 += (float)(vaddvq_s32(vaddq_s32(acc_2_lo, acc_2_hi)) - block_bsums[k]) * scales_2[k] * vec_scale;
        sum_3 += (float)(vaddvq_s32(vaddq_s32(acc_3_lo, acc_3_hi)) - block_bsums[k]) * scales_3[k] * vec_scale;
    }

    *out_0 = sum_0;
    *out_1 = sum_1;
    *out_2 = sum_2;
    *out_3 = sum_3;
}
#else
static void tq2_0_matmul_row_i8_neon_scales(const uint8_t *weight, const float *scales,
                                            int blocks_per_row, const int8_t *qvec,
                                            float vec_scale, const int32_t *block_bsums, float *out_val) {
    (void)block_bsums;
    tq2_0_matmul_row_i8_scales(weight, scales, blocks_per_row, qvec, vec_scale, out_val);
}

static void tq2_0_matmul_row_i8_neon_pair_scales(const uint8_t *weight_a, const float *scales_a,
                                                 const uint8_t *weight_b, const float *scales_b,
                                                 int blocks_per_row, const int8_t *qvec,
                                                 float vec_scale, const int32_t *block_bsums, float *out_a, float *out_b) {
    (void)block_bsums;
    tq2_0_matmul_row_i8_scales(weight_a, scales_a, blocks_per_row, qvec, vec_scale, out_a);
    tq2_0_matmul_row_i8_scales(weight_b, scales_b, blocks_per_row, qvec, vec_scale, out_b);
}

static void tq2_0_matmul_row_i8_neon_quad_scales(
    const uint8_t *weight_0, const float *scales_0,
    const uint8_t *weight_1, const float *scales_1,
    const uint8_t *weight_2, const float *scales_2,
    const uint8_t *weight_3, const float *scales_3,
    int blocks_per_row, const int8_t *qvec,
    float vec_scale, const int32_t *block_bsums,
    float *out_0, float *out_1, float *out_2, float *out_3) {
    (void)block_bsums;
    tq2_0_matmul_row_i8_scales(weight_0, scales_0, blocks_per_row, qvec, vec_scale, out_0);
    tq2_0_matmul_row_i8_scales(weight_1, scales_1, blocks_per_row, qvec, vec_scale, out_1);
    tq2_0_matmul_row_i8_scales(weight_2, scales_2, blocks_per_row, qvec, vec_scale, out_2);
    tq2_0_matmul_row_i8_scales(weight_3, scales_3, blocks_per_row, qvec, vec_scale, out_3);
}
#endif

/* ========================================================================
 * TL1 vqtbl1q_s8 GEMM implementation
 *
 * Uses ARM NEON vqtbl1q_s8 table-lookup for high-performance ternary GEMM.
 * Follows llama.cpp/BitNet TL1 kernel design:
 *   1. Weight reordering: TQ2_0 interleaved → TL1 consecutive nibble format
 *   2. LUT preprocessor: builds vqtbl1q_s8 lookup table from float activations
 *   3. Blocked GEMM: BM rows × BK K-dim, LUT shared across rows
 *
 * TQ2_0 interleaved layout (stride-32):
 *   For byte at qs[g*32+m]:
 *     bits[1:0] = code for position g*128 + 0*32 + m
 *     bits[3:2] = code for position g*128 + 1*32 + m
 *     bits[5:4] = code for position g*128 + 2*32 + m
 *     bits[7:6] = code for position g*128 + 3*32 + m
 *
 * TL1 target layout (consecutive nibble pairs):
 *   For byte i (0..63):
 *     Low nibble bits[1:0]  = code for position 4i+0
 *     Low nibble bits[3:2]  = code for position 4i+1
 *     High nibble bits[5:4] = code for position 4i+2
 *     High nibble bits[7:6] = code for position 4i+3
 *
 * Weight reordering converts TQ2_0 interleaved to TL1 consecutive format.
 * This is a ONE-TIME cost at model load time.
 * ======================================================================== */

/* ---- Weight reordering: TQ2_0 interleaved → TL1 consecutive ---- */

int bitnet_tq2_0_reorder_to_tl1(const void *weight, int out_dim, int in_dim,
                                  uint8_t *reordered) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || reordered == NULL || out_dim <= 0 ||
        in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) return -1;

    for (int row = 0; row < out_dim; ++row) {
        size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        uint8_t *dst_row = reordered + (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_QS_SIZE;

        for (int k = 0; k < blocks_per_row; ++k) {
            const bitnet_tq2_0_block_t *block =
                (const bitnet_tq2_0_block_t *)(bytes + row_offset +
                    (size_t)k * BITNET_TQ2_0_BLOCK_SIZE);
            uint8_t *dst = dst_row + (size_t)k * BITNET_TQ2_0_QS_SIZE;

            /* Decode all 256 codes from TQ2_0 interleaved layout */
            uint8_t decoded[256];
            for (int g = 0; g < 2; ++g) {
                for (int l = 0; l < 4; ++l) {
                    for (int m = 0; m < 32; ++m) {
                        int pos = g * 128 + l * 32 + m;
                        decoded[pos] = (block->qs[g * 32 + m] >> (l * 2)) & 3;
                    }
                }
            }

            /* Re-encode in TL1 consecutive nibble format */
            for (int i = 0; i < 64; ++i) {
                uint8_t code0 = decoded[4 * i + 0];
                uint8_t code1 = decoded[4 * i + 1];
                uint8_t code2 = decoded[4 * i + 2];
                uint8_t code3 = decoded[4 * i + 3];
                dst[i] = (uint8_t)((code3 << 6) | (code2 << 4) | (code1 << 2) | code0);
            }
        }
    }

    return 0;
}

#if defined(__ARM_NEON)

/* ---- TL1 LUT preprocessor (following llama.cpp lut_ctor algorithm) ---- */

/* LUT size: for each group of 16 activations, we store 256 bytes of LUT.
 * Total: (in_dim / 16) * 256 bytes */
size_t bitnet_tq2_0_tl1_lut_size(int in_dim) {
    if (in_dim <= 0 || in_dim % 16 != 0) return 0;
    return (size_t)(in_dim / 16) * 256u;
}

/* Transpose 8x8 int16x8_t matrix in-place.
 * Follows llama.cpp's Transpose_8_8 exactly. */
static inline void tl1_transpose_8_8(
    int16x8_t *v0, int16x8_t *v1, int16x8_t *v2, int16x8_t *v3,
    int16x8_t *v4, int16x8_t *v5, int16x8_t *v6, int16x8_t *v7) {
    int16x8x2_t q04 = vzipq_s16(*v0, *v4);
    int16x8x2_t q15 = vzipq_s16(*v1, *v5);
    int16x8x2_t q26 = vzipq_s16(*v2, *v6);
    int16x8x2_t q37 = vzipq_s16(*v3, *v7);
    int16x8x2_t q0246_0 = vzipq_s16(q04.val[0], q26.val[0]);
    int16x8x2_t q0246_1 = vzipq_s16(q04.val[1], q26.val[1]);
    int16x8x2_t q1357_0 = vzipq_s16(q15.val[0], q37.val[0]);
    int16x8x2_t q1357_1 = vzipq_s16(q15.val[1], q37.val[1]);
    int16x8x2_t q_fin_0 = vzipq_s16(q0246_0.val[0], q1357_0.val[0]);
    int16x8x2_t q_fin_1 = vzipq_s16(q0246_0.val[1], q1357_0.val[1]);
    int16x8x2_t q_fin_2 = vzipq_s16(q0246_1.val[0], q1357_1.val[0]);
    int16x8x2_t q_fin_3 = vzipq_s16(q0246_1.val[1], q1357_1.val[1]);
    *v0 = q_fin_0.val[0];
    *v1 = q_fin_0.val[1];
    *v2 = q_fin_1.val[0];
    *v3 = q_fin_1.val[1];
    *v4 = q_fin_2.val[0];
    *v5 = q_fin_2.val[1];
    *v6 = q_fin_3.val[0];
    *v7 = q_fin_3.val[1];
}

/* Build TL1 vqtbl1q_s8 LUT from float activation vector.
 *
 * Follows llama.cpp's lut_ctor algorithm exactly:
 * 1. Quantize activations to int16 (per-tensor scale = 127/max_abs)
 * 2. For each group of 16 activations, build 16 LUT vectors using linear combinations
 * 3. Transpose 8x8 and rearrange with vqtbl1q_s8 + tbl_mask
 * 4. Store as int8
 *
 * The LUT encodes the relationship between weight nibble values and activation values.
 * For a 4-bit nibble n (0-15):
 *   code_lo = n & 3 (bits[1:0]): code for even-position activation
 *   code_hi = (n >> 2) & 3 (bits[3:2]): code for odd-position activation
 *   Code mapping: 0→-1, 1→0, 2→+1, 3→0(reserved)
 *   LUT[n] = code_lo_val * y_even + code_hi_val * y_odd
 *
 * LUT layout per 16 activations: 256 bytes
 *   Stored as 8 pairs of (high, low) int8x8 vectors after tbl_mask rearrangement.
 */
int bitnet_tq2_0_tl1_preprocessor(const float *vec, int in_dim,
                                    float *vec_scale, int8_t *lut, size_t lut_size) {
    size_t required = 0;
    float max_abs = 0.0f;
    float scale = 1.0f;

    if (vec == NULL || lut == NULL || vec_scale == NULL ||
        in_dim <= 0 || in_dim % 16 != 0) {
        return -1;
    }

    required = bitnet_tq2_0_tl1_lut_size(in_dim);
    if (lut_size < required || required == 0) {
        return -1;
    }

    /* Compute per-tensor scale: max_abs / 127 */
    {
        float32x4_t max_vec = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 3 < in_dim; i += 4) {
            float32x4_t v = vld1q_f32(vec + i);
            max_vec = vmaxq_f32(max_vec, vabsq_f32(v));
        }
        max_abs = vmaxvq_f32(max_vec);
        for (; i < in_dim; ++i) {
            float a = fabsf(vec[i]);
            if (a > max_abs) max_abs = a;
        }
    }

    if (max_abs <= 0.0f) {
        memset(lut, 0, lut_size);
        *vec_scale = 1.0f;
        return 0;
    }

    scale = 127.0f / max_abs;
    *vec_scale = max_abs / 127.0f;

    /* tbl_mask for rearranging after transpose.
     * Maps nibble index to correct byte position for vqtbl1q_s8:
     *   Even indices (0,2,4,...,14) → high bytes of each int16 pair
     *   Odd indices (1,3,5,...,15) → low bytes of each int16 pair */
    static const uint8_t tbl_mask_data[16] = {
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
    };
    const uint8x16_t tbl_mask_q = vld1q_u8(tbl_mask_data);

    /* Process 16 activations at a time */
    for (int k = 0; k < in_dim / 16; ++k) {
        const float *b = vec + k * 16;
        int16x8_t vec_lut[16];

        /* Deinterleave with vld2q_f32 (same as llama.cpp) */
        float32x4x2_t vec_bs_x0 = vld2q_f32(b);
        float32x4x2_t vec_bs_x1 = vld2q_f32(b + 8);

        /* Quantize to int16 */
        float32x4_t vec_f_0 = vmulq_n_f32(vec_bs_x0.val[0], scale);
        float32x4_t vec_f_1 = vmulq_n_f32(vec_bs_x0.val[1], scale);
        float32x4_t vec_f_2 = vmulq_n_f32(vec_bs_x1.val[0], scale);
        float32x4_t vec_f_3 = vmulq_n_f32(vec_bs_x1.val[1], scale);

        int32x4_t vec_b_0 = vcvtnq_s32_f32(vec_f_0);
        int32x4_t vec_b_1 = vcvtnq_s32_f32(vec_f_1);
        int32x4_t vec_b_2 = vcvtnq_s32_f32(vec_f_2);
        int32x4_t vec_b_3 = vcvtnq_s32_f32(vec_f_3);

        int16x4_t vec_b16_0 = vmovn_s32(vec_b_0);
        int16x4_t vec_b16_1 = vmovn_s32(vec_b_1);
        int16x4_t vec_b16_2 = vmovn_s32(vec_b_2);
        int16x4_t vec_b16_3 = vmovn_s32(vec_b_3);

        /* vec_bs_0 = even-position activations, vec_bs_1 = odd-position activations */
        int16x8_t vec_bs_0 = vcombine_s16(vec_b16_0, vec_b16_2);
        int16x8_t vec_bs_1 = vcombine_s16(vec_b16_1, vec_b16_3);

        /* Build 16 LUT vectors for all 16 possible nibble values.
         * Nibble n: code_lo = n&3 (even pos), code_hi = (n>>2)&3 (odd pos)
         * Code mapping: 0→-1, 1→0, 2→+1, 3→0
         * LUT[n] = code_lo_val * vec_bs_0 + code_hi_val * vec_bs_1 */
        vec_lut[0]  = vsubq_s16(vsubq_s16(vdupq_n_s16(0), vec_bs_0), vec_bs_1); /* -1,-1 */
        vec_lut[1]  = vsubq_s16(vdupq_n_s16(0), vec_bs_0);                       /*  0,-1 */
        vec_lut[2]  = vaddq_s16(vsubq_s16(vdupq_n_s16(0), vec_bs_0), vec_bs_1);  /* +1,-1 */
        vec_lut[3]  = vsubq_s16(vdupq_n_s16(0), vec_bs_1);                       /* -1, 0 (reserved→0) */
        vec_lut[4]  = vdupq_n_s16(0);                                             /*  0, 0 */
        vec_lut[5]  = vec_bs_1;                                                    /*  0,+1 */
        vec_lut[6]  = vsubq_s16(vec_bs_0, vec_bs_1);                              /* +1, 0 */
        vec_lut[7]  = vec_bs_0;                                                    /* -1,+1 (reserved→0) */
        vec_lut[8]  = vaddq_s16(vec_bs_0, vec_bs_1);                              /* +1,+1 */
        vec_lut[9]  = vec_bs_0;                                                    /*  0,+1 (reserved) */
        vec_lut[10] = vec_bs_1;                                                    /* +1, 0 (reserved) */
        vec_lut[11] = vdupq_n_s16(0);                                             /* reserved */
        vec_lut[12] = vdupq_n_s16(0);                                             /* reserved */
        vec_lut[13] = vdupq_n_s16(0);                                             /* reserved */
        vec_lut[14] = vdupq_n_s16(0);                                             /* reserved */
        vec_lut[15] = vdupq_n_s16(0);                                             /* reserved */

        /* Transpose 8x8 (same as llama.cpp) */
        tl1_transpose_8_8(&vec_lut[0], &vec_lut[1], &vec_lut[2], &vec_lut[3],
                          &vec_lut[4], &vec_lut[5], &vec_lut[6], &vec_lut[7]);
        tl1_transpose_8_8(&vec_lut[8], &vec_lut[9], &vec_lut[10], &vec_lut[11],
                          &vec_lut[12], &vec_lut[13], &vec_lut[14], &vec_lut[15]);

        /* Rearrange with vqtbl1q_s8 + tbl_mask and store as int8
         * (same as llama.cpp) */
        for (int idx = 0; idx < 8; idx++) {
            int8x16_t q0_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx]), tbl_mask_q);
            int8x8_t q0_low = vget_low_s8(q0_s);
            int8x8_t q0_high = vget_high_s8(q0_s);
            int8x16_t q1_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx + 8]), tbl_mask_q);
            int8x8_t q1_low = vget_low_s8(q1_s);
            int8x8_t q1_high = vget_high_s8(q1_s);
            vst1_s8(lut + k * 256 + idx * 32 + 0, q0_high);
            vst1_s8(lut + k * 256 + idx * 32 + 8, q1_high);
            vst1_s8(lut + k * 256 + idx * 32 + 16, q0_low);
            vst1_s8(lut + k * 256 + idx * 32 + 24, q1_low);
        }
    }

    return 0;
}

/* ---- TL1 Blocked GEMM kernel (following llama.cpp tbl_impl pattern) ---- */

/* Blocked GEMM with per-block scales.
 *
 * Processes BM rows at a time, sharing LUT across all rows.
 * For each TQ2_0 block (256 weights = 2 BK blocks of 128):
 *   1. Load LUT for this BK block
 *   2. For each group of 32 rows: vqtbl1q_s8 lookup, int16 accumulate
 *   3. Widen to int32, apply per-block scale and bsums correction
 *
 * Weight format: TL1 reordered (consecutive nibble pairs).
 * Each byte: low nibble = codes for positions 4i, 4i+1; high nibble = codes for 4i+2, 4i+3.
 */

#define TL1_BM 128
#define TL1_BK 128  /* = 64 weight bytes per row per BK block */

/* Process one BK=128 block for BM rows using vqtbl1q_s8.
 * Follows llama.cpp's tbl_impl pattern exactly.
 *
 * lut: LUT for this BK block (TL1_BK/2 * 16 = 1024 bytes)
 * weights: TL1 reordered weights for BM rows (BM * TL1_BK/2/2 = BM * 32 bytes)
 * c: int32 output accumulators for BM rows
 */
static inline void tl1_tbl_impl(int32_t *c, const int8_t *lut, const uint8_t *weights, int bm) {
    const int KK = TL1_BK / 2;  /* 64 LUT vectors */
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    const int16x8_t vec_zero = vdupq_n_s16(0);

    /* Load all LUT vectors into local array */
    int8x16_t vec_lut[KK];
    for (int k = 0; k < KK; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

    /* Process 32 rows at a time */
    for (int i = 0; i < bm; i += 32) {
        int16x8_t vec_c[8];
        for (int j = 0; j < 8; j++) {
            vec_c[j] = vec_zero;
        }

        /* KK/4 = 16 iterations of K loop */
        for (int k = 0; k < KK / 4; k++) {
            /* Row 0 */
            uint8x16_t vec_a_0 = vld1q_u8(weights + i * (KK / 2) + k * 32 + 0 * 16);
            uint8x16_t vec_a0_top = vshrq_n_u8(vec_a_0, 4);
            uint8x16_t vec_a0_bot = vandq_u8(vec_a_0, vec_mask);
            int8x16_t vec_v_0_left_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 0], vec_a0_top);
            int8x16_t vec_v_0_left_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 1], vec_a0_top);
            int8x16_t vec_v_0_right_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 2], vec_a0_bot);
            int8x16_t vec_v_0_right_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 3], vec_a0_bot);
            int8x16x2_t vec_v_left_0 = vzipq_s8(vec_v_0_left_tmp1, vec_v_0_left_tmp0);
            int8x16x2_t vec_v_right_0 = vzipq_s8(vec_v_0_right_tmp1, vec_v_0_right_tmp0);
            vec_c[0] = vaddq_s16(vec_c[0], vreinterpretq_s16_s8(vec_v_left_0.val[0]));
            vec_c[0] = vaddq_s16(vec_c[0], vreinterpretq_s16_s8(vec_v_right_0.val[0]));
            vec_c[1] = vaddq_s16(vec_c[1], vreinterpretq_s16_s8(vec_v_left_0.val[1]));
            vec_c[1] = vaddq_s16(vec_c[1], vreinterpretq_s16_s8(vec_v_right_0.val[1]));

            /* Row 1 */
            uint8x16_t vec_a_1 = vld1q_u8(weights + i * (KK / 2) + k * 32 + 1 * 16);
            uint8x16_t vec_a1_top = vshrq_n_u8(vec_a_1, 4);
            uint8x16_t vec_a1_bot = vandq_u8(vec_a_1, vec_mask);
            int8x16_t vec_v_1_left_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 0], vec_a1_top);
            int8x16_t vec_v_1_left_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 1], vec_a1_top);
            int8x16_t vec_v_1_right_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 2], vec_a1_bot);
            int8x16_t vec_v_1_right_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 3], vec_a1_bot);
            int8x16x2_t vec_v_left_1 = vzipq_s8(vec_v_1_left_tmp1, vec_v_1_left_tmp0);
            int8x16x2_t vec_v_right_1 = vzipq_s8(vec_v_1_right_tmp1, vec_v_1_right_tmp0);
            vec_c[2] = vaddq_s16(vec_c[2], vreinterpretq_s16_s8(vec_v_left_1.val[0]));
            vec_c[2] = vaddq_s16(vec_c[2], vreinterpretq_s16_s8(vec_v_right_1.val[0]));
            vec_c[3] = vaddq_s16(vec_c[3], vreinterpretq_s16_s8(vec_v_left_1.val[1]));
            vec_c[3] = vaddq_s16(vec_c[3], vreinterpretq_s16_s8(vec_v_right_1.val[1]));

            /* Row 2 */
            uint8x16_t vec_a_2 = vld1q_u8(weights + i * (KK / 2) + k * 32 + 2 * 16);
            uint8x16_t vec_a2_top = vshrq_n_u8(vec_a_2, 4);
            uint8x16_t vec_a2_bot = vandq_u8(vec_a_2, vec_mask);
            int8x16_t vec_v_2_left_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 0], vec_a2_top);
            int8x16_t vec_v_2_left_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 1], vec_a2_top);
            int8x16_t vec_v_2_right_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 2], vec_a2_bot);
            int8x16_t vec_v_2_right_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 3], vec_a2_bot);
            int8x16x2_t vec_v_left_2 = vzipq_s8(vec_v_2_left_tmp1, vec_v_2_left_tmp0);
            int8x16x2_t vec_v_right_2 = vzipq_s8(vec_v_2_right_tmp1, vec_v_2_right_tmp0);
            vec_c[4] = vaddq_s16(vec_c[4], vreinterpretq_s16_s8(vec_v_left_2.val[0]));
            vec_c[4] = vaddq_s16(vec_c[4], vreinterpretq_s16_s8(vec_v_right_2.val[0]));
            vec_c[5] = vaddq_s16(vec_c[5], vreinterpretq_s16_s8(vec_v_left_2.val[1]));
            vec_c[5] = vaddq_s16(vec_c[5], vreinterpretq_s16_s8(vec_v_right_2.val[1]));

            /* Row 3 */
            uint8x16_t vec_a_3 = vld1q_u8(weights + i * (KK / 2) + k * 32 + 3 * 16);
            uint8x16_t vec_a3_top = vshrq_n_u8(vec_a_3, 4);
            uint8x16_t vec_a3_bot = vandq_u8(vec_a_3, vec_mask);
            int8x16_t vec_v_3_left_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 0], vec_a3_top);
            int8x16_t vec_v_3_left_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 1], vec_a3_top);
            int8x16_t vec_v_3_right_tmp0 = vqtbl1q_s8(vec_lut[4 * k + 2], vec_a3_bot);
            int8x16_t vec_v_3_right_tmp1 = vqtbl1q_s8(vec_lut[4 * k + 3], vec_a3_bot);
            int8x16x2_t vec_v_left_3 = vzipq_s8(vec_v_3_left_tmp1, vec_v_3_left_tmp0);
            int8x16x2_t vec_v_right_3 = vzipq_s8(vec_v_3_right_tmp1, vec_v_3_right_tmp0);
            vec_c[6] = vaddq_s16(vec_c[6], vreinterpretq_s16_s8(vec_v_left_3.val[0]));
            vec_c[6] = vaddq_s16(vec_c[6], vreinterpretq_s16_s8(vec_v_right_3.val[0]));
            vec_c[7] = vaddq_s16(vec_c[7], vreinterpretq_s16_s8(vec_v_left_3.val[1]));
            vec_c[7] = vaddq_s16(vec_c[7], vreinterpretq_s16_s8(vec_v_right_3.val[1]));
        }

        /* Widen int16 to int32 and accumulate */
        int32x4_t v0_low = vmovl_s16(vget_low_s16(vec_c[0]));
        int32x4_t v0_high = vmovl_high_s16(vec_c[0]);
        vst1q_s32(c + i + 0, vaddq_s32(vld1q_s32(c + i + 0), v0_low));
        vst1q_s32(c + i + 4, vaddq_s32(vld1q_s32(c + i + 4), v0_high));
        int32x4_t v1_low = vmovl_s16(vget_low_s16(vec_c[1]));
        int32x4_t v1_high = vmovl_high_s16(vec_c[1]);
        vst1q_s32(c + i + 8, vaddq_s32(vld1q_s32(c + i + 8), v1_low));
        vst1q_s32(c + i + 12, vaddq_s32(vld1q_s32(c + i + 12), v1_high));
        int32x4_t v2_low = vmovl_s16(vget_low_s16(vec_c[2]));
        int32x4_t v2_high = vmovl_high_s16(vec_c[2]);
        vst1q_s32(c + i + 16, vaddq_s32(vld1q_s32(c + i + 16), v2_low));
        vst1q_s32(c + i + 20, vaddq_s32(vld1q_s32(c + i + 20), v2_high));
        int32x4_t v3_low = vmovl_s16(vget_low_s16(vec_c[3]));
        int32x4_t v3_high = vmovl_high_s16(vec_c[3]);
        vst1q_s32(c + i + 24, vaddq_s32(vld1q_s32(c + i + 24), v3_low));
        vst1q_s32(c + i + 28, vaddq_s32(vld1q_s32(c + i + 28), v3_high));
        int32x4_t v4_low = vmovl_s16(vget_low_s16(vec_c[4]));
        int32x4_t v4_high = vmovl_high_s16(vec_c[4]);
        vst1q_s32(c + i + 32, vaddq_s32(vld1q_s32(c + i + 32), v4_low));
        vst1q_s32(c + i + 36, vaddq_s32(vld1q_s32(c + i + 36), v4_high));
        int32x4_t v5_low = vmovl_s16(vget_low_s16(vec_c[5]));
        int32x4_t v5_high = vmovl_high_s16(vec_c[5]);
        vst1q_s32(c + i + 40, vaddq_s32(vld1q_s32(c + i + 40), v5_low));
        vst1q_s32(c + i + 44, vaddq_s32(vld1q_s32(c + i + 44), v5_high));
        int32x4_t v6_low = vmovl_s16(vget_low_s16(vec_c[6]));
        int32x4_t v6_high = vmovl_high_s16(vec_c[6]);
        vst1q_s32(c + i + 48, vaddq_s32(vld1q_s32(c + i + 48), v6_low));
        vst1q_s32(c + i + 52, vaddq_s32(vld1q_s32(c + i + 52), v6_high));
        int32x4_t v7_low = vmovl_s16(vget_low_s16(vec_c[7]));
        int32x4_t v7_high = vmovl_high_s16(vec_c[7]);
        vst1q_s32(c + i + 56, vaddq_s32(vld1q_s32(c + i + 56), v7_low));
        vst1q_s32(c + i + 60, vaddq_s32(vld1q_s32(c + i + 60), v7_high));
    }
}

/* TL1 GEMM: compute matmul using vqtbl1q_s8 blocked GEMM with per-block scales.
 *
 * weight_tl1: TL1 reordered weights (out_dim * blocks_per_row * QS_SIZE bytes)
 * scales: per-block float scales (out_dim * blocks_per_row)
 * bsums: per-block int32 bsums (blocks_per_row) - shared across rows for same activation
 * lut: pre-built TL1 LUT from tl1_preprocessor
 * vec_scale: activation quantization scale
 * out: float output (out_dim)
 */
int bitnet_tq2_0_tl1_qgemm(const uint8_t *weight_tl1, const float *scales,
                              const int32_t *bsums, int out_dim, int in_dim,
                              const int8_t *lut, float vec_scale, float *out) {
    int blocks_per_row = 0;

    if (weight_tl1 == NULL || scales == NULL || bsums == NULL ||
        lut == NULL || out == NULL || out_dim <= 0 ||
        in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) return -1;

    /* Process BM rows at a time */
    for (int i_start = 0; i_start < out_dim; i_start += TL1_BM) {
        int bm = TL1_BM;
        if (i_start + bm > out_dim) bm = out_dim - i_start;

        /* Initialize output */
        for (int i = 0; i < bm; i++) {
            out[i_start + i] = 0.0f;
        }

        /* For each TQ2_0 block, accumulate int32 results then apply scale */
        for (int blk = 0; blk < blocks_per_row; blk++) {
            /* int32 accumulators for this block, for bm rows */
            __attribute__((aligned(32))) int32_t cbits[TL1_BM];
            memset(cbits, 0, (size_t)bm * sizeof(int32_t));

            /* Process 2 BK blocks per TQ2_0 block */
            for (int h = 0; h < 2; h++) {
                int bk_idx = blk * 2 + h;
                /* LUT offset for this BK block:
                 * Each BK=128 block covers 128 activations = 8 groups of 16.
                 * Each group has 256 bytes of LUT. */
                const int8_t *bk_lut = lut + (size_t)bk_idx * (128 / 16) * 256;

                /* Weight offset for this BK block:
                 * Each row has blocks_per_row * QS_SIZE bytes.
                 * Within a block, BK=128 covers QS_SIZE/2 = 32 bytes.
                 * BK block h covers bytes [h*32 .. h*32+31] within the block. */
                /* Build weight pointer array for bm rows */
                __attribute__((aligned(32))) uint8_t w_buf[TL1_BM * 32];
                for (int i = 0; i < bm; i++) {
                    const uint8_t *row_weights = weight_tl1 +
                        (size_t)(i_start + i) * (size_t)blocks_per_row * BITNET_TQ2_0_QS_SIZE;
                    memcpy(w_buf + i * 32, row_weights + (size_t)blk * BITNET_TQ2_0_QS_SIZE + h * 32, 32);
                }

                tl1_tbl_impl(cbits, bk_lut, w_buf, bm);
            }

            /* Apply per-block scale and bsums correction.
             * The LUT uses bsums encoding {0,1,2} (not {-1,0,1}),
             * so we need to subtract bsums from the accumulated result.
             * result = (cbits[i] - bsums[blk]) * scales[blk] * vec_scale */
            for (int i = 0; i < bm; i++) {
                float row_scale = scales[(size_t)(i_start + i) * (size_t)blocks_per_row + (size_t)blk];
                out[i_start + i] += (float)(cbits[i] - bsums[blk]) * row_scale * vec_scale;
            }
        }
    }

    return 0;
}

/* TL1 paired GEMM: compute two matmuls (gate + up) sharing the same LUT */
int bitnet_tq2_0_tl1_qgemm_pair(const uint8_t *weight_a_tl1, const float *scales_a,
                                   const uint8_t *weight_b_tl1, const float *scales_b,
                                   const int32_t *bsums, int out_dim, int in_dim,
                                   const int8_t *lut, float vec_scale,
                                   float *out_a, float *out_b) {
    /* Simply call single GEMM twice - they share the same LUT */
    int rc_a = bitnet_tq2_0_tl1_qgemm(weight_a_tl1, scales_a, bsums,
                                         out_dim, in_dim, lut, vec_scale, out_a);
    if (rc_a != 0) return rc_a;
    return bitnet_tq2_0_tl1_qgemm(weight_b_tl1, scales_b, bsums,
                                     out_dim, in_dim, lut, vec_scale, out_b);
}

/* ========================================================================
 * I2_S interleaved format: 4-row packed 2-bit weights
 *
 * Design based on bitnet.cpp I2_S kernel:
 *   - 4 rows of weights interleaved into single bytes
 *   - Each byte: (row0_code << 6) | (row1_code << 4) | (row2_code << 2) | row3_code
 *   - I2S code: 0=-1, 1=0, 2=+1 (same as TQ2_0 code mapping)
 *   - ARM block size QK_I2S = 64
 *   - VTBL unpack maps packed 2-bit codes directly to signed int8 weights
 *   - vdotq_s32 for dot product accumulation
 *
 * Memory layout for packed data:
 *   For each group of 4 rows (out_dim padded to multiple of 4):
 *     For each 64-element sub-block:
 *       QK_I2S=64 bytes of interleaved data (1 byte per element position,
 *       each byte containing 4 rows' 2-bit codes)
 *   Total packed size = n_groups * sub_blocks_per_group * QK_I2S
 *
 * Scales are stored separately:
 *   packed_scales: per I2S sub-block, 4 floats (one per row)
 *   packed_bsums:  retained for API compatibility and diagnostics
 *   Note: 4 consecutive I2S sub-blocks share the same TQ2_0 scale.
 *
 * The matmul kernel processes per TQ2_0 block (4 I2S sub-blocks) to apply
 * per-block scale correctly. The bsums parameter is optional in this signed
 * I2S path because the lookup table emits code-1 directly.
 * ======================================================================== */

#define QK_I2S 64

size_t bitnet_tq2_0_i2s_packed_size(int out_dim, int in_dim) {
    if (out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return 0;
    }
    int out_dim_padded = (out_dim + 3) & ~3;
    int n_groups = out_dim_padded / 4;
    int sub_blocks_per_group = in_dim / QK_I2S;
    return (size_t)n_groups * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
}

/* Extract a single 2-bit code from TQ2_0 interleaved layout.
 * TQ2_0 layout: for byte at qs[g*32+m], bits[l*2+1:l*2] = code
 *   for position g*128 + l*32 + m
 * Returns the TQ2_0 code (0,1,2,3) where value = code - 1. */
static inline uint8_t tq2_0_extract_code(const uint8_t *qs, int pos) {
    int g = pos / 128;
    int l = (pos % 128) / 32;
    int m = pos % 32;
    return (qs[g * 32 + m] >> (l * 2)) & 3;
}

int bitnet_tq2_0_reorder_to_i2s(const void *weight, int out_dim, int in_dim,
                                  uint8_t *packed, float *packed_scales, int32_t *packed_bsums) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;
    int out_dim_padded = 0;
    int n_groups = 0;
    int sub_blocks_per_group = 0;

    if (weight == NULL || packed == NULL || packed_scales == NULL ||
        packed_bsums == NULL || out_dim <= 0 || in_dim <= 0 ||
        in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) return -1;

    out_dim_padded = (out_dim + 3) & ~3;
    n_groups = out_dim_padded / 4;
    sub_blocks_per_group = in_dim / QK_I2S;

    for (int grp = 0; grp < n_groups; ++grp) {
        int row_base = grp * 4;

        for (int sb = 0; sb < sub_blocks_per_group; ++sb) {
            int elem_base = sb * QK_I2S;

            /* Packed data: QK_I2S bytes for this sub-block */
            uint8_t *dst = packed + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S
                         + (size_t)sb * (size_t)QK_I2S;
            /* Scales: 4 floats for this sub-block (one per row) */
            float *dst_scales = packed_scales + ((size_t)grp * (size_t)sub_blocks_per_group + (size_t)sb) * 4u;
            /* Bsums: 4 int32 for this sub-block (one per row) */
            int32_t *dst_bsums = packed_bsums + ((size_t)grp * (size_t)sub_blocks_per_group + (size_t)sb) * 4u;

            /* Clear destination bytes (for padding rows) */
            memset(dst, 0, QK_I2S);

            for (int r = 0; r < 4; ++r) {
                int row = row_base + r;
                float row_scale = 0.0f;
                int32_t row_bsum = 0;

                if (row < out_dim) {
                    size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;

                    /* Determine which TQ2_0 block this sub-block starts in */
                    int tq2_blk = elem_base / BITNET_TQ2_0_QK;
                    const bitnet_tq2_0_block_t *block =
                        (const bitnet_tq2_0_block_t *)(bytes + row_offset +
                            (size_t)tq2_blk * BITNET_TQ2_0_BLOCK_SIZE);
                    row_scale = fp16_to_fp32(block->d);

                    /* Decode all 64 codes for this row in this sub-block and pack */
                    int32_t code_sum = 0;
                    for (int e = 0; e < QK_I2S; ++e) {
                        int global_pos = elem_base + e;
                        int blk = global_pos / BITNET_TQ2_0_QK;
                        int pos_in_blk = global_pos % BITNET_TQ2_0_QK;
                        const bitnet_tq2_0_block_t *blk_ptr =
                            (const bitnet_tq2_0_block_t *)(bytes + row_offset +
                                (size_t)blk * BITNET_TQ2_0_BLOCK_SIZE);
                        uint8_t code = tq2_0_extract_code(blk_ptr->qs, pos_in_blk);
                        /* Pack: row r's code goes into bits [(3-r)*2+1 : (3-r)*2] */
                        dst[e] |= (uint8_t)(code << ((3 - r) * 2));
                        code_sum += (int32_t)code;
                    }
                    /* bsum = sum(code - 1) = sum(code) - QK_I2S */
                    row_bsum = code_sum - QK_I2S;
                }

                dst_scales[r] = row_scale;
                dst_bsums[r] = row_bsum;
            }
        }
    }

    return 0;
}

#if BITNET_TQ2_HAS_NEON_DOTPROD

static const uint8_t i2s_lut_hi2_unsigned_data[16] = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
};

static const uint8_t i2s_lut_lo2_unsigned_data[16] = {
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
};

static const uint8_t i2s_lut_hi2_signed_data[16] = {
    255, 255, 255, 255, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2
};

static const uint8_t i2s_lut_lo2_signed_data[16] = {
    255, 0, 1, 2, 255, 0, 1, 2, 255, 0, 1, 2, 255, 0, 1, 2
};

#define I2S_UNPACK_CODES_VTBL(pk, c0, c1, c2, c3) do {       \
        uint8x16_t i2s_hi_nib = vshrq_n_u8((pk), 4);         \
        uint8x16_t i2s_lo_nib = vandq_u8((pk), mask_0f);     \
        (c0) = vreinterpretq_s8_u8(vqtbl1q_u8(lut_hi2, i2s_hi_nib)); \
        (c1) = vreinterpretq_s8_u8(vqtbl1q_u8(lut_lo2, i2s_hi_nib)); \
        (c2) = vreinterpretq_s8_u8(vqtbl1q_u8(lut_hi2, i2s_lo_nib)); \
        (c3) = vreinterpretq_s8_u8(vqtbl1q_u8(lut_lo2, i2s_lo_nib)); \
    } while (0)

/* I2_S NEON matmul kernel for a group of 4 rows.
 * Optimized version with dual accumulators and loop unrolling.
 *
 * Memory bandwidth advantage: 4 rows share 64 bytes of packed data
 * (vs TQ2_0's 4*66=264 bytes), giving ~4x bandwidth savings.
 *
 * The lookup table emits signed weights directly: code 0 -> -1, 1 -> 0,
 * 2 -> +1, 3 -> +2, matching the scalar TQ2_0 code-1 behavior. This avoids
 * per-block qvec bsum correction on the I2S path.
 */
static void i2s_matmul_4rows_neon(
    const uint8_t *packed_grp, const float *scales_grp,
    int blocks_per_row, const int8_t *qvec,
    const int32_t *bsums, float vec_scale,
    float *out_0, float *out_1, float *out_2, float *out_3) {

    const int use_signed_lut = (bsums == NULL);
    float sum_0 = 0.0f, sum_1 = 0.0f, sum_2 = 0.0f, sum_3 = 0.0f;
    const uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    const uint8x16_t lut_hi2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_hi2_signed_data :
                                        i2s_lut_hi2_unsigned_data);
    const uint8x16_t lut_lo2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_lo2_signed_data :
                                        i2s_lut_lo2_unsigned_data);

    /* Process per TQ2_0 block (4 I2S sub-blocks each) */
    for (int blk = 0; blk < blocks_per_row; ++blk) {
        /* Dual accumulators to break vdotq dependency chain */
        int32x4_t acc_0_lo = vdupq_n_s32(0), acc_0_hi = vdupq_n_s32(0);
        int32x4_t acc_1_lo = vdupq_n_s32(0), acc_1_hi = vdupq_n_s32(0);
        int32x4_t acc_2_lo = vdupq_n_s32(0), acc_2_hi = vdupq_n_s32(0);
        int32x4_t acc_3_lo = vdupq_n_s32(0), acc_3_hi = vdupq_n_s32(0);

        /* Each TQ2_0 block = 4 I2S sub-blocks of QK_I2S=64 bytes each */
        for (int sub = 0; sub < 4; ++sub) {
            int sb = blk * 4 + sub;
            const uint8_t *pb = packed_grp + (size_t)sb * (size_t)QK_I2S;
            const int8_t *qv = qvec + (size_t)sb * (size_t)QK_I2S;

            /* Process 64 bytes: 4 iterations of 16 bytes.
             * Unrolled with dual accumulators for ILP. */
            for (int i = 0; i < QK_I2S; i += 32) {
                /* First 16 bytes */
                {
                    uint8x16_t pk = vld1q_u8(pb + i);
                    int8x16_t v = vld1q_s8(qv + i);
                    int8x16_t c0, c1, c2, c3;
                    I2S_UNPACK_CODES_VTBL(pk, c0, c1, c2, c3);

                    acc_0_lo = vdotq_s32(acc_0_lo, c0, v);
                    acc_1_lo = vdotq_s32(acc_1_lo, c1, v);
                    acc_2_lo = vdotq_s32(acc_2_lo, c2, v);
                    acc_3_lo = vdotq_s32(acc_3_lo, c3, v);
                }
                /* Second 16 bytes */
                {
                    uint8x16_t pk = vld1q_u8(pb + i + 16);
                    int8x16_t v = vld1q_s8(qv + i + 16);
                    int8x16_t c0, c1, c2, c3;
                    I2S_UNPACK_CODES_VTBL(pk, c0, c1, c2, c3);

                    acc_0_hi = vdotq_s32(acc_0_hi, c0, v);
                    acc_1_hi = vdotq_s32(acc_1_hi, c1, v);
                    acc_2_hi = vdotq_s32(acc_2_hi, c2, v);
                    acc_3_hi = vdotq_s32(acc_3_hi, c3, v);
                }
            }
        }

        /* Get scale for this TQ2_0 block (same for all 4 sub-blocks within it) */
        const float *sb_scales = scales_grp + (size_t)(blk * 4) * 4;
        float s0 = sb_scales[0], s1 = sb_scales[1], s2 = sb_scales[2], s3 = sb_scales[3];

        /* Combine dual accumulators and apply the per-block row scales. */
        int32_t dot_0 = vaddvq_s32(vaddq_s32(acc_0_lo, acc_0_hi));
        int32_t dot_1 = vaddvq_s32(vaddq_s32(acc_1_lo, acc_1_hi));
        int32_t dot_2 = vaddvq_s32(vaddq_s32(acc_2_lo, acc_2_hi));
        int32_t dot_3 = vaddvq_s32(vaddq_s32(acc_3_lo, acc_3_hi));
        if (!use_signed_lut) {
            const int32_t bsum = bsums[blk];
            dot_0 -= bsum;
            dot_1 -= bsum;
            dot_2 -= bsum;
            dot_3 -= bsum;
        }
        sum_0 += (float)dot_0 * s0;
        sum_1 += (float)dot_1 * s1;
        sum_2 += (float)dot_2 * s2;
        sum_3 += (float)dot_3 * s3;
    }

    *out_0 = sum_0 * vec_scale;
    *out_1 = sum_1 * vec_scale;
    *out_2 = sum_2 * vec_scale;
    *out_3 = sum_3 * vec_scale;
}

static void i2s_matmul_8rows_neon(
    const uint8_t *packed_grp0, const uint8_t *packed_grp1,
    const float *scales_grp0, const float *scales_grp1,
    int blocks_per_row, const int8_t *qvec,
    const int32_t *bsums, float vec_scale,
    float out[8]) {

    const int use_signed_lut = (bsums == NULL);
    float sum_0 = 0.0f, sum_1 = 0.0f, sum_2 = 0.0f, sum_3 = 0.0f;
    float sum_4 = 0.0f, sum_5 = 0.0f, sum_6 = 0.0f, sum_7 = 0.0f;
    const uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    const uint8x16_t lut_hi2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_hi2_signed_data :
                                        i2s_lut_hi2_unsigned_data);
    const uint8x16_t lut_lo2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_lo2_signed_data :
                                        i2s_lut_lo2_unsigned_data);

    for (int blk = 0; blk < blocks_per_row; ++blk) {
        int32x4_t acc_0_lo = vdupq_n_s32(0), acc_0_hi = vdupq_n_s32(0);
        int32x4_t acc_1_lo = vdupq_n_s32(0), acc_1_hi = vdupq_n_s32(0);
        int32x4_t acc_2_lo = vdupq_n_s32(0), acc_2_hi = vdupq_n_s32(0);
        int32x4_t acc_3_lo = vdupq_n_s32(0), acc_3_hi = vdupq_n_s32(0);
        int32x4_t acc_4_lo = vdupq_n_s32(0), acc_4_hi = vdupq_n_s32(0);
        int32x4_t acc_5_lo = vdupq_n_s32(0), acc_5_hi = vdupq_n_s32(0);
        int32x4_t acc_6_lo = vdupq_n_s32(0), acc_6_hi = vdupq_n_s32(0);
        int32x4_t acc_7_lo = vdupq_n_s32(0), acc_7_hi = vdupq_n_s32(0);

        for (int sub = 0; sub < 4; ++sub) {
            int sb = blk * 4 + sub;
            const uint8_t *pb0 = packed_grp0 + (size_t)sb * (size_t)QK_I2S;
            const uint8_t *pb1 = packed_grp1 + (size_t)sb * (size_t)QK_I2S;
            const int8_t *qv = qvec + (size_t)sb * (size_t)QK_I2S;

            for (int i = 0; i < QK_I2S; i += 32) {
                int8x16_t v0 = vld1q_s8(qv + i);
                int8x16_t v1 = vld1q_s8(qv + i + 16);

                {
                    uint8x16_t pk0 = vld1q_u8(pb0 + i);
                    uint8x16_t pk1 = vld1q_u8(pb0 + i + 16);
                    int8x16_t c0_0, c1_0, c2_0, c3_0;
                    I2S_UNPACK_CODES_VTBL(pk0, c0_0, c1_0, c2_0, c3_0);
                    acc_0_lo = vdotq_s32(acc_0_lo, c0_0, v0);
                    acc_1_lo = vdotq_s32(acc_1_lo, c1_0, v0);
                    acc_2_lo = vdotq_s32(acc_2_lo, c2_0, v0);
                    acc_3_lo = vdotq_s32(acc_3_lo, c3_0, v0);
                    int8x16_t c0_1, c1_1, c2_1, c3_1;
                    I2S_UNPACK_CODES_VTBL(pk1, c0_1, c1_1, c2_1, c3_1);
                    acc_0_hi = vdotq_s32(acc_0_hi, c0_1, v1);
                    acc_1_hi = vdotq_s32(acc_1_hi, c1_1, v1);
                    acc_2_hi = vdotq_s32(acc_2_hi, c2_1, v1);
                    acc_3_hi = vdotq_s32(acc_3_hi, c3_1, v1);
                }
                {
                    uint8x16_t pk0 = vld1q_u8(pb1 + i);
                    uint8x16_t pk1 = vld1q_u8(pb1 + i + 16);
                    int8x16_t c0_0, c1_0, c2_0, c3_0;
                    I2S_UNPACK_CODES_VTBL(pk0, c0_0, c1_0, c2_0, c3_0);
                    acc_4_lo = vdotq_s32(acc_4_lo, c0_0, v0);
                    acc_5_lo = vdotq_s32(acc_5_lo, c1_0, v0);
                    acc_6_lo = vdotq_s32(acc_6_lo, c2_0, v0);
                    acc_7_lo = vdotq_s32(acc_7_lo, c3_0, v0);
                    int8x16_t c0_1, c1_1, c2_1, c3_1;
                    I2S_UNPACK_CODES_VTBL(pk1, c0_1, c1_1, c2_1, c3_1);
                    acc_4_hi = vdotq_s32(acc_4_hi, c0_1, v1);
                    acc_5_hi = vdotq_s32(acc_5_hi, c1_1, v1);
                    acc_6_hi = vdotq_s32(acc_6_hi, c2_1, v1);
                    acc_7_hi = vdotq_s32(acc_7_hi, c3_1, v1);
                }
            }
        }

        const float *sb_scales0 = scales_grp0 + (size_t)(blk * 4) * 4;
        const float *sb_scales1 = scales_grp1 + (size_t)(blk * 4) * 4;
        int32_t dot_0 = vaddvq_s32(vaddq_s32(acc_0_lo, acc_0_hi));
        int32_t dot_1 = vaddvq_s32(vaddq_s32(acc_1_lo, acc_1_hi));
        int32_t dot_2 = vaddvq_s32(vaddq_s32(acc_2_lo, acc_2_hi));
        int32_t dot_3 = vaddvq_s32(vaddq_s32(acc_3_lo, acc_3_hi));
        int32_t dot_4 = vaddvq_s32(vaddq_s32(acc_4_lo, acc_4_hi));
        int32_t dot_5 = vaddvq_s32(vaddq_s32(acc_5_lo, acc_5_hi));
        int32_t dot_6 = vaddvq_s32(vaddq_s32(acc_6_lo, acc_6_hi));
        int32_t dot_7 = vaddvq_s32(vaddq_s32(acc_7_lo, acc_7_hi));
        if (!use_signed_lut) {
            const int32_t bsum = bsums[blk];
            dot_0 -= bsum;
            dot_1 -= bsum;
            dot_2 -= bsum;
            dot_3 -= bsum;
            dot_4 -= bsum;
            dot_5 -= bsum;
            dot_6 -= bsum;
            dot_7 -= bsum;
        }
        sum_0 += (float)dot_0 * sb_scales0[0];
        sum_1 += (float)dot_1 * sb_scales0[1];
        sum_2 += (float)dot_2 * sb_scales0[2];
        sum_3 += (float)dot_3 * sb_scales0[3];
        sum_4 += (float)dot_4 * sb_scales1[0];
        sum_5 += (float)dot_5 * sb_scales1[1];
        sum_6 += (float)dot_6 * sb_scales1[2];
        sum_7 += (float)dot_7 * sb_scales1[3];
    }

    out[0] = sum_0 * vec_scale;
    out[1] = sum_1 * vec_scale;
    out[2] = sum_2 * vec_scale;
    out[3] = sum_3 * vec_scale;
    out[4] = sum_4 * vec_scale;
    out[5] = sum_5 * vec_scale;
    out[6] = sum_6 * vec_scale;
    out[7] = sum_7 * vec_scale;
}

int bitnet_tq2_0_matmul_i2s_neon(const uint8_t *packed, const float *scales,
                                   const int32_t *bsums, int out_dim, int in_dim,
                                   const int8_t *qvec, float vec_scale, float *out) {
    int blocks_per_row = 0;
    int out_dim_padded = 0;
    int n_groups = 0;
    int sub_blocks_per_group = 0;

    if (packed == NULL || scales == NULL ||
        qvec == NULL || out == NULL || out_dim <= 0 ||
        in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) return -1;

    out_dim_padded = (out_dim + 3) & ~3;
    n_groups = out_dim_padded / 4;
    sub_blocks_per_group = in_dim / QK_I2S;

    int grp = 0;
    for (; grp + 1 < n_groups; grp += 2) {
        int row_base = grp * 4;
        const uint8_t *packed_grp0 = packed + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const uint8_t *packed_grp1 = packed_grp0 + (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const float *scales_grp0 = scales + (size_t)grp * (size_t)sub_blocks_per_group * 4;
        const float *scales_grp1 = scales_grp0 + (size_t)sub_blocks_per_group * 4;
        float r[8];

        i2s_matmul_8rows_neon(packed_grp0, packed_grp1,
                              scales_grp0, scales_grp1,
                              blocks_per_row, qvec, bsums, vec_scale, r);

        for (int k = 0; k < 8 && row_base + k < out_dim; ++k) {
            out[row_base + k] = r[k];
        }
    }

    for (; grp < n_groups; ++grp) {
        int row_base = grp * 4;
        const uint8_t *packed_grp = packed + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const float *scales_grp = scales + (size_t)grp * (size_t)sub_blocks_per_group * 4;

        float r0, r1, r2, r3;
        i2s_matmul_4rows_neon(packed_grp, scales_grp,
                              blocks_per_row, qvec, bsums, vec_scale,
                              &r0, &r1, &r2, &r3);

        /* Only write valid rows */
        if (row_base < out_dim) out[row_base] = r0;
        if (row_base + 1 < out_dim) out[row_base + 1] = r1;
        if (row_base + 2 < out_dim) out[row_base + 2] = r2;
        if (row_base + 3 < out_dim) out[row_base + 3] = r3;
    }

    return 0;
}

int bitnet_tq2_0_matmul_i2s_neon_pair(const uint8_t *packed_a, const float *scales_a,
                                        const uint8_t *packed_b, const float *scales_b,
                                        const int32_t *bsums, int out_dim, int in_dim,
                                        const int8_t *qvec, float vec_scale,
                                        float *out_a, float *out_b) {
    /* Fused pair: process both matrices together, sharing qvec loads.
     * For each group of 4 rows, compute both A and B dot products
     * while qvec is in registers, avoiding redundant qvec loads. */
    int blocks_per_row = 0;
    int out_dim_padded = 0;
    int n_groups = 0;
    int sub_blocks_per_group = 0;

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

    const int use_signed_lut = (bsums == NULL);
    const uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    const uint8x16_t lut_hi2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_hi2_signed_data :
                                        i2s_lut_hi2_unsigned_data);
    const uint8x16_t lut_lo2 = vld1q_u8(use_signed_lut ?
                                        i2s_lut_lo2_signed_data :
                                        i2s_lut_lo2_unsigned_data);

    for (int grp = 0; grp < n_groups; ++grp) {
        int row_base = grp * 4;
        const uint8_t *pg_a = packed_a + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const uint8_t *pg_b = packed_b + (size_t)grp * (size_t)sub_blocks_per_group * (size_t)QK_I2S;
        const float *sc_a = scales_a + (size_t)grp * (size_t)sub_blocks_per_group * 4u;
        const float *sc_b = scales_b + (size_t)grp * (size_t)sub_blocks_per_group * 4u;

        float sa0 = 0.0f, sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
        float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;

        for (int blk = 0; blk < blocks_per_row; ++blk) {
            int32x4_t aa0_lo = vdupq_n_s32(0), aa0_hi = vdupq_n_s32(0);
            int32x4_t aa1_lo = vdupq_n_s32(0), aa1_hi = vdupq_n_s32(0);
            int32x4_t aa2_lo = vdupq_n_s32(0), aa2_hi = vdupq_n_s32(0);
            int32x4_t aa3_lo = vdupq_n_s32(0), aa3_hi = vdupq_n_s32(0);
            int32x4_t ab0_lo = vdupq_n_s32(0), ab0_hi = vdupq_n_s32(0);
            int32x4_t ab1_lo = vdupq_n_s32(0), ab1_hi = vdupq_n_s32(0);
            int32x4_t ab2_lo = vdupq_n_s32(0), ab2_hi = vdupq_n_s32(0);
            int32x4_t ab3_lo = vdupq_n_s32(0), ab3_hi = vdupq_n_s32(0);

            for (int sub = 0; sub < 4; ++sub) {
                int sb_idx = blk * 4 + sub;
                const uint8_t *pb_a = pg_a + (size_t)sb_idx * (size_t)QK_I2S;
                const uint8_t *pb_b = pg_b + (size_t)sb_idx * (size_t)QK_I2S;
                const int8_t *qv = qvec + (size_t)sb_idx * (size_t)QK_I2S;

                for (int i = 0; i < QK_I2S; i += 32) {
                    /* Load qvec once, use for both A and B */
                    int8x16_t v0 = vld1q_s8(qv + i);
                    int8x16_t v1 = vld1q_s8(qv + i + 16);

                    /* Matrix A */
                    {
                        uint8x16_t pk0 = vld1q_u8(pb_a + i);
                        uint8x16_t pk1 = vld1q_u8(pb_a + i + 16);

                        int8x16_t c0_0, c1_0, c2_0, c3_0;
                        I2S_UNPACK_CODES_VTBL(pk0, c0_0, c1_0, c2_0, c3_0);

                        aa0_lo = vdotq_s32(aa0_lo, c0_0, v0);
                        aa1_lo = vdotq_s32(aa1_lo, c1_0, v0);
                        aa2_lo = vdotq_s32(aa2_lo, c2_0, v0);
                        aa3_lo = vdotq_s32(aa3_lo, c3_0, v0);

                        int8x16_t c0_1, c1_1, c2_1, c3_1;
                        I2S_UNPACK_CODES_VTBL(pk1, c0_1, c1_1, c2_1, c3_1);

                        aa0_hi = vdotq_s32(aa0_hi, c0_1, v1);
                        aa1_hi = vdotq_s32(aa1_hi, c1_1, v1);
                        aa2_hi = vdotq_s32(aa2_hi, c2_1, v1);
                        aa3_hi = vdotq_s32(aa3_hi, c3_1, v1);
                    }

                    /* Matrix B (reuses v0, v1) */
                    {
                        uint8x16_t pk0 = vld1q_u8(pb_b + i);
                        uint8x16_t pk1 = vld1q_u8(pb_b + i + 16);

                        int8x16_t c0_0, c1_0, c2_0, c3_0;
                        I2S_UNPACK_CODES_VTBL(pk0, c0_0, c1_0, c2_0, c3_0);

                        ab0_lo = vdotq_s32(ab0_lo, c0_0, v0);
                        ab1_lo = vdotq_s32(ab1_lo, c1_0, v0);
                        ab2_lo = vdotq_s32(ab2_lo, c2_0, v0);
                        ab3_lo = vdotq_s32(ab3_lo, c3_0, v0);

                        int8x16_t c0_1, c1_1, c2_1, c3_1;
                        I2S_UNPACK_CODES_VTBL(pk1, c0_1, c1_1, c2_1, c3_1);

                        ab0_hi = vdotq_s32(ab0_hi, c0_1, v1);
                        ab1_hi = vdotq_s32(ab1_hi, c1_1, v1);
                        ab2_hi = vdotq_s32(ab2_hi, c2_1, v1);
                        ab3_hi = vdotq_s32(ab3_hi, c3_1, v1);
                    }
                }
            }

            const float *bsc_a = sc_a + (size_t)(blk * 4) * 4;
            const float *bsc_b = sc_b + (size_t)(blk * 4) * 4;

            int32_t da0 = vaddvq_s32(vaddq_s32(aa0_lo, aa0_hi));
            int32_t da1 = vaddvq_s32(vaddq_s32(aa1_lo, aa1_hi));
            int32_t da2 = vaddvq_s32(vaddq_s32(aa2_lo, aa2_hi));
            int32_t da3 = vaddvq_s32(vaddq_s32(aa3_lo, aa3_hi));
            int32_t db0 = vaddvq_s32(vaddq_s32(ab0_lo, ab0_hi));
            int32_t db1 = vaddvq_s32(vaddq_s32(ab1_lo, ab1_hi));
            int32_t db2 = vaddvq_s32(vaddq_s32(ab2_lo, ab2_hi));
            int32_t db3 = vaddvq_s32(vaddq_s32(ab3_lo, ab3_hi));
            if (!use_signed_lut) {
                const int32_t bsum = bsums[blk];
                da0 -= bsum;
                da1 -= bsum;
                da2 -= bsum;
                da3 -= bsum;
                db0 -= bsum;
                db1 -= bsum;
                db2 -= bsum;
                db3 -= bsum;
            }

            sa0 += (float)da0 * bsc_a[0];
            sa1 += (float)da1 * bsc_a[1];
            sa2 += (float)da2 * bsc_a[2];
            sa3 += (float)da3 * bsc_a[3];

            sb0 += (float)db0 * bsc_b[0];
            sb1 += (float)db1 * bsc_b[1];
            sb2 += (float)db2 * bsc_b[2];
            sb3 += (float)db3 * bsc_b[3];
        }

        if (row_base + 0 < out_dim) out_a[row_base + 0] = sa0 * vec_scale;
        if (row_base + 1 < out_dim) out_a[row_base + 1] = sa1 * vec_scale;
        if (row_base + 2 < out_dim) out_a[row_base + 2] = sa2 * vec_scale;
        if (row_base + 3 < out_dim) out_a[row_base + 3] = sa3 * vec_scale;
        if (row_base + 0 < out_dim) out_b[row_base + 0] = sb0 * vec_scale;
        if (row_base + 1 < out_dim) out_b[row_base + 1] = sb1 * vec_scale;
        if (row_base + 2 < out_dim) out_b[row_base + 2] = sb2 * vec_scale;
        if (row_base + 3 < out_dim) out_b[row_base + 3] = sb3 * vec_scale;
    }

    return 0;
}

#else /* !BITNET_TQ2_HAS_NEON_DOTPROD */

/* Scalar fallback: not supported without NEON dotprod */
int bitnet_tq2_0_matmul_i2s_neon(const uint8_t *packed, const float *scales,
                                   const int32_t *bsums, int out_dim, int in_dim,
                                   const int8_t *qvec, float vec_scale, float *out) {
    (void)packed; (void)scales; (void)bsums; (void)out_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out;
    return -1;
}

int bitnet_tq2_0_matmul_i2s_neon_pair(const uint8_t *packed_a, const float *scales_a,
                                        const uint8_t *packed_b, const float *scales_b,
                                        const int32_t *bsums, int out_dim, int in_dim,
                                        const int8_t *qvec, float vec_scale,
                                        float *out_a, float *out_b) {
    (void)packed_a; (void)scales_a; (void)packed_b; (void)scales_b;
    (void)bsums; (void)out_dim; (void)in_dim; (void)qvec; (void)vec_scale;
    (void)out_a; (void)out_b;
    return -1;
}

#endif /* BITNET_TQ2_HAS_NEON_DOTPROD */

/* Forward declarations for parallel dispatchers (defined after thread pool) */
#if BITNET_TQ2_HAS_NEON_DOTPROD
static int i2s_matmul_parallel(const uint8_t *packed, const float *scales,
                                const int32_t *bsums, int out_dim, int in_dim,
                                const int8_t *qvec, float vec_scale, float *out);
static int i2s_matmul_pair_parallel(const uint8_t *packed_a, const float *scales_a,
                                     const uint8_t *packed_b, const float *scales_b,
                                     const int32_t *bsums, int out_dim, int in_dim,
                                     const int8_t *qvec, float vec_scale,
                                     float *out_a, float *out_b);
static int i2s_matmul_qkv_parallel(const uint8_t *packed_q, const float *scales_q,
                                   const uint8_t *packed_k, const float *scales_k,
                                   const uint8_t *packed_v, const float *scales_v,
                                   const int32_t *bsums,
                                   int q_dim, int kv_dim, int in_dim,
                                   const int8_t *qvec, float vec_scale,
                                   float *out_q, float *out_k, float *out_v);
#endif

/* Public parallel I2S wrappers — dispatch to thread pool */
int bitnet_tq2_0_matmul_i2s_neon_parallel(const uint8_t *packed, const float *scales,
                                            const int32_t *bsums, int out_dim, int in_dim,
                                            const int8_t *qvec, float vec_scale, float *out) {
#if BITNET_TQ2_HAS_NEON_DOTPROD
    return i2s_matmul_parallel(packed, scales, bsums, out_dim, in_dim, qvec, vec_scale, out);
#else
    (void)packed; (void)scales; (void)bsums; (void)out_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)out;
    return -1;
#endif
}

int bitnet_tq2_0_matmul_i2s_neon_pair_parallel(const uint8_t *packed_a, const float *scales_a,
                                                  const uint8_t *packed_b, const float *scales_b,
                                                  const int32_t *bsums, int out_dim, int in_dim,
                                                  const int8_t *qvec, float vec_scale,
                                                  float *out_a, float *out_b) {
#if BITNET_TQ2_HAS_NEON_DOTPROD
    return i2s_matmul_pair_parallel(packed_a, scales_a, packed_b, scales_b,
                                     bsums, out_dim, in_dim, qvec, vec_scale, out_a, out_b);
#else
    (void)packed_a; (void)scales_a; (void)packed_b; (void)scales_b;
    (void)bsums; (void)out_dim; (void)in_dim; (void)qvec; (void)vec_scale;
    (void)out_a; (void)out_b;
    return -1;
#endif
}

int bitnet_tq2_0_matmul_i2s_qkv_parallel(const uint8_t *packed_q, const float *scales_q,
                                          const uint8_t *packed_k, const float *scales_k,
                                          const uint8_t *packed_v, const float *scales_v,
                                          const int32_t *bsums,
                                          int q_dim, int kv_dim, int in_dim,
                                          const int8_t *qvec, float vec_scale,
                                          float *out_q, float *out_k, float *out_v) {
#if BITNET_TQ2_HAS_NEON_DOTPROD
    return i2s_matmul_qkv_parallel(packed_q, scales_q,
                                   packed_k, scales_k,
                                   packed_v, scales_v,
                                   bsums, q_dim, kv_dim, in_dim,
                                   qvec, vec_scale, out_q, out_k, out_v);
#else
    (void)packed_q; (void)scales_q; (void)packed_k; (void)scales_k;
    (void)packed_v; (void)scales_v; (void)bsums; (void)q_dim; (void)kv_dim;
    (void)in_dim; (void)qvec; (void)vec_scale; (void)out_q; (void)out_k; (void)out_v;
    return -1;
#endif
}


size_t bitnet_tq2_0_vqtbl1q_lut_size(int in_dim) {
    return bitnet_tq2_0_tl1_lut_size(in_dim);
}

int bitnet_tq2_0_build_vqtbl1q_lut(const int8_t *qvec, int in_dim,
                                    int8_t *lut, size_t lut_size) {
    /* This old API is deprecated - use tl1_preprocessor instead */
    (void)qvec; (void)in_dim; (void)lut; (void)lut_size;
    return -1;
}

int bitnet_tq2_0_matmul_vqtbl1q_lut(const void *weight, const float *scales,
                                     int out_dim, int in_dim,
                                     const int8_t *qvec, float vec_scale,
                                     const int32_t *block_bsums,
                                     const int8_t *tl1_lut, float *out) {
    /* This old API is deprecated - use tl1_qgemm instead */
    (void)weight; (void)scales; (void)out_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)block_bsums; (void)tl1_lut; (void)out;
    return -1;
}

int bitnet_tq2_0_matmul_vqtbl1q_lut_pair(const void *weight_a, const float *scales_a,
                                          const void *weight_b, const float *scales_b,
                                          int out_dim, int in_dim,
                                          const int8_t *qvec, float vec_scale,
                                          const int32_t *block_bsums,
                                          const int8_t *tl1_lut,
                                          float *out_a, float *out_b) {
    /* This old API is deprecated - use tl1_qgemm_pair instead */
    (void)weight_a; (void)scales_a; (void)weight_b; (void)scales_b;
    (void)out_dim; (void)in_dim; (void)qvec; (void)vec_scale;
    (void)block_bsums; (void)tl1_lut; (void)out_a; (void)out_b;
    return -1;
}

#else /* !__ARM_NEON */

/* Scalar fallback stubs */
size_t bitnet_tq2_0_tl1_lut_size(int in_dim) {
    if (in_dim <= 0 || in_dim % 16 != 0) return 0;
    return (size_t)(in_dim / 16) * 256u;
}

int bitnet_tq2_0_tl1_preprocessor(const float *vec, int in_dim,
                                    float *vec_scale, int8_t *lut, size_t lut_size) {
    (void)vec; (void)in_dim; (void)vec_scale; (void)lut; (void)lut_size;
    return -1;
}

int bitnet_tq2_0_tl1_qgemm(const uint8_t *weight_tl1, const float *scales,
                              const int32_t *bsums, int out_dim, int in_dim,
                              const int8_t *lut, float vec_scale, float *out) {
    (void)weight_tl1; (void)scales; (void)bsums; (void)out_dim; (void)in_dim;
    (void)lut; (void)vec_scale; (void)out;
    return -1;
}

int bitnet_tq2_0_tl1_qgemm_pair(const uint8_t *weight_a_tl1, const float *scales_a,
                                   const uint8_t *weight_b_tl1, const float *scales_b,
                                   const int32_t *bsums, int out_dim, int in_dim,
                                   const int8_t *lut, float vec_scale,
                                   float *out_a, float *out_b) {
    (void)weight_a_tl1; (void)scales_a; (void)weight_b_tl1; (void)scales_b;
    (void)bsums; (void)out_dim; (void)in_dim; (void)lut; (void)vec_scale;
    (void)out_a; (void)out_b;
    return -1;
}

size_t bitnet_tq2_0_vqtbl1q_lut_size(int in_dim) {
    return bitnet_tq2_0_tl1_lut_size(in_dim);
}

int bitnet_tq2_0_build_vqtbl1q_lut(const int8_t *qvec, int in_dim,
                                    int8_t *lut, size_t lut_size) {
    (void)qvec; (void)in_dim; (void)lut; (void)lut_size;
    return -1;
}

int bitnet_tq2_0_matmul_vqtbl1q_lut(const void *weight, const float *scales,
                                     int out_dim, int in_dim,
                                     const int8_t *qvec, float vec_scale,
                                     const int32_t *block_bsums,
                                     const int8_t *tl1_lut, float *out) {
    (void)weight; (void)scales; (void)out_dim; (void)in_dim;
    (void)qvec; (void)vec_scale; (void)block_bsums; (void)tl1_lut; (void)out;
    return -1;
}

int bitnet_tq2_0_matmul_vqtbl1q_lut_pair(const void *weight_a, const float *scales_a,
                                          const void *weight_b, const float *scales_b,
                                          int out_dim, int in_dim,
                                          const int8_t *qvec, float vec_scale,
                                          const int32_t *block_bsums,
                                          const int8_t *tl1_lut,
                                          float *out_a, float *out_b) {
    (void)weight_a; (void)scales_a; (void)weight_b; (void)scales_b;
    (void)out_dim; (void)in_dim; (void)qvec; (void)vec_scale;
    (void)block_bsums; (void)tl1_lut; (void)out_a; (void)out_b;
    return -1;
}

#endif /* __ARM_NEON */

static int tq2_0_matmul_rows_i8_parallel(const uint8_t *bytes, const uint8_t *bytes_b,
                                         const float *scales, const float *scales_b,
                                         int blocks_per_row,
                                         int out_start, int out_count,
                                         const int8_t *qvec, float vec_scale,
                                         const int32_t *block_bsums, int use_neon,
                                         float *out, float *out_b);

int bitnet_tq2_0_matmul_vector_i8_scales(const void *weight, const float *scales,
                                         int out_dim, int in_dim,
                                         const int8_t *qvec, float vec_scale,
                                         float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || scales == NULL || qvec == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) {
        return -1;
    }

    return tq2_0_matmul_rows_i8_parallel(bytes, NULL, scales, NULL,
                                         blocks_per_row, 0, out_dim,
                                         qvec, vec_scale, NULL, 0, out, NULL);
}

int bitnet_tq2_0_matmul_vector_i8_neon_scales(const void *weight, const float *scales,
                                              int out_dim, int in_dim,
                                              const int8_t *qvec, float vec_scale,
                                              const int32_t *block_bsums, float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || scales == NULL || qvec == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) {
        return -1;
    }

    return tq2_0_matmul_rows_i8_parallel(bytes, NULL, scales, NULL,
                                         blocks_per_row, 0, out_dim,
                                         qvec, vec_scale, block_bsums, 1, out, NULL);
}

int bitnet_tq2_0_matmul_vector_i8_neon_scales_single(const void *weight, const float *scales,
                                                       int out_dim, int in_dim,
                                                       const int8_t *qvec, float vec_scale,
                                                       const int32_t *block_bsums, float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || scales == NULL || qvec == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) {
        return -1;
    }

#if BITNET_TQ2_HAS_NEON_DOTPROD
    /* Force single-threaded: use quad kernel for best single-thread performance */
    int j = 0;
    for (; j + 3 < out_dim; j += 4) {
        size_t off0 = (size_t)j * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        size_t off1 = off0 + (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        size_t off2 = off1 + (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        size_t off3 = off2 + (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        tq2_0_matmul_row_i8_neon_quad_scales(
            bytes + off0, scales + (size_t)j * (size_t)blocks_per_row,
            bytes + off1, scales + (size_t)(j + 1) * (size_t)blocks_per_row,
            bytes + off2, scales + (size_t)(j + 2) * (size_t)blocks_per_row,
            bytes + off3, scales + (size_t)(j + 3) * (size_t)blocks_per_row,
            blocks_per_row, qvec, vec_scale, block_bsums,
            &out[j], &out[j + 1], &out[j + 2], &out[j + 3]);
    }
    for (; j < out_dim; ++j) {
        size_t row_offset = (size_t)j * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        const float *row_scales = scales + (size_t)j * (size_t)blocks_per_row;
        tq2_0_matmul_row_i8_neon_scales(bytes + row_offset, row_scales,
                                        blocks_per_row, qvec, vec_scale, block_bsums, &out[j]);
    }
#else
    /* Fallback: scalar path */
    (void)block_bsums;
    for (int j = 0; j < out_dim; ++j) {
        size_t row_offset = (size_t)j * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        const float *row_scales = scales + (size_t)j * (size_t)blocks_per_row;
        tq2_0_matmul_row_i8_scales(bytes + row_offset, row_scales,
                                   blocks_per_row, qvec, vec_scale, &out[j]);
    }
#endif
    return 0;
}

int bitnet_tq2_0_matmul_vector_i8_neon_pair_scales(const void *weight_a, const float *scales_a,
                                                   const void *weight_b, const float *scales_b,
                                                   int out_dim, int in_dim,
                                                   const int8_t *qvec, float vec_scale,
                                                   const int32_t *block_bsums, float *out_a, float *out_b) {
    const uint8_t *bytes_a = (const uint8_t *)weight_a;
    const uint8_t *bytes_b = (const uint8_t *)weight_b;
    int blocks_per_row = 0;

    if (weight_a == NULL || scales_a == NULL || weight_b == NULL || scales_b == NULL ||
        qvec == NULL || out_a == NULL || out_b == NULL ||
        out_dim <= 0 || in_dim <= 0 || in_dim % BITNET_TQ2_0_QK != 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (blocks_per_row <= 0) {
        return -1;
    }

    return tq2_0_matmul_rows_i8_parallel(bytes_a, bytes_b, scales_a, scales_b,
                                         blocks_per_row, 0, out_dim,
                                         qvec, vec_scale, block_bsums, 1, out_a, out_b);
}

static void tq2_0_matmul_row_lut(const uint8_t *weight, int blocks_per_row,
                                  const float *lut, float *out_val) {
    float sum = 0.0f;

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block = (const bitnet_tq2_0_block_t *)(weight + block_offset);
        const float d = fp16_to_fp32(block->d);
        float block_sum = 0.0f;

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs = block->qs + (size_t)g * 32u;
            const float *group_lut = lut + (((size_t)k * 2u + (size_t)g) * 32u * 256u);
            for (int m = 0; m < 32; ++m) {
                block_sum += group_lut[(size_t)m * 256u + qs[m]];
            }
        }

        sum += block_sum * d;
    }

    *out_val = sum;
}

static void tq2_0_matmul_row_lut_pair(const uint8_t *weight_a, const uint8_t *weight_b,
                                      int blocks_per_row, const float *lut,
                                      float *out_a, float *out_b) {
    float sum_a = 0.0f;
    float sum_b = 0.0f;

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block_a = (const bitnet_tq2_0_block_t *)(weight_a + block_offset);
        const bitnet_tq2_0_block_t *block_b = (const bitnet_tq2_0_block_t *)(weight_b + block_offset);
        const float d_a = fp16_to_fp32(block_a->d);
        const float d_b = fp16_to_fp32(block_b->d);
        float block_sum_a = 0.0f;
        float block_sum_b = 0.0f;

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs_a = block_a->qs + (size_t)g * 32u;
            const uint8_t *qs_b = block_b->qs + (size_t)g * 32u;
            const float *group_lut = lut + (((size_t)k * 2u + (size_t)g) * 32u * 256u);
            for (int m = 0; m < 32; ++m) {
                size_t base = (size_t)m * 256u;
                block_sum_a += group_lut[base + qs_a[m]];
                block_sum_b += group_lut[base + qs_b[m]];
            }
        }

        sum_a += block_sum_a * d_a;
        sum_b += block_sum_b * d_b;
    }

    *out_a = sum_a;
    *out_b = sum_b;
}

static void tq2_0_matmul_row_lut_scales(const uint8_t *weight, const float *scales,
                                        int blocks_per_row, const float *lut,
                                        float *out_val) {
    float sum = 0.0f;

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block = (const bitnet_tq2_0_block_t *)(weight + block_offset);
        float block_sum = 0.0f;

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs = block->qs + (size_t)g * 32u;
            const float *group_lut = lut + (((size_t)k * 2u + (size_t)g) * 32u * 256u);
            for (int m = 0; m < 32; ++m) {
                block_sum += group_lut[(size_t)m * 256u + qs[m]];
            }
        }

        sum += block_sum * scales[k];
    }

    *out_val = sum;
}

static void tq2_0_matmul_row_lut_pair_scales(const uint8_t *weight_a, const float *scales_a,
                                             const uint8_t *weight_b, const float *scales_b,
                                             int blocks_per_row, const float *lut,
                                             float *out_a, float *out_b) {
    float sum_a = 0.0f;
    float sum_b = 0.0f;

    for (int k = 0; k < blocks_per_row; ++k) {
        size_t block_offset = (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
        const bitnet_tq2_0_block_t *block_a = (const bitnet_tq2_0_block_t *)(weight_a + block_offset);
        const bitnet_tq2_0_block_t *block_b = (const bitnet_tq2_0_block_t *)(weight_b + block_offset);
        float block_sum_a = 0.0f;
        float block_sum_b = 0.0f;

        for (int g = 0; g < 2; ++g) {
            const uint8_t *qs_a = block_a->qs + (size_t)g * 32u;
            const uint8_t *qs_b = block_b->qs + (size_t)g * 32u;
            const float *group_lut = lut + (((size_t)k * 2u + (size_t)g) * 32u * 256u);
            for (int m = 0; m < 32; ++m) {
                size_t base = (size_t)m * 256u;
                block_sum_a += group_lut[base + qs_a[m]];
                block_sum_b += group_lut[base + qs_b[m]];
            }
        }

        sum_a += block_sum_a * scales_a[k];
        sum_b += block_sum_b * scales_b[k];
    }

    *out_a = sum_a;
    *out_b = sum_b;
}

/* Lightweight thread pool using spin-wait + atomic generation counter.
 * Replaces mutex+cond_wait with much lower synchronization overhead.
 * Worker threads spin on generation counter to detect new work.
 * Completion is tracked via atomic completed counter. */
typedef struct {
    pthread_t threads[8];
    int n_threads;
    int initialized;

    /* Task parameters — written by dispatcher before incrementing generation */
    const uint8_t *bytes;
    const uint8_t *bytes_b;
    const float *scales;
    const float *scales_b;
    int blocks_per_row;
    int out_start;
    int out_count;
    const float *lut;
    const int8_t *qvec;
    const int16_t *i16_lut;
    float vec_scale;
    int use_neon_i8;
    const int32_t *block_bsums;
    float *out;
    float *out_b;

    /* I2S dispatch parameters */
    int dispatch_type;       /* 0 = TQ2_0, 1 = I2S single, 2 = I2S pair */
    const uint8_t *i2s_packed;
    const uint8_t *i2s_packed_b;
    const uint8_t *i2s_packed_c;
    const float *i2s_scales;
    const float *i2s_scales_b;
    const float *i2s_scales_c;
    const int32_t *i2s_bsums;
    int i2s_out_dim;
    int i2s_kv_out_dim;
    int i2s_in_dim;
    int i2s_blocks_per_row;
    int i2s_n_groups;        /* total groups of 4 rows (padded) */
    int i2s_q_groups;
    int i2s_kv_groups;
    int i2s_sub_blocks_per_group;
    int i2s_chunk_groups;
    float *i2s_out;
    float *i2s_out_b;
    float *i2s_out_c;

    /* Synchronization — all atomic, no mutex */
    atomic_uint generation;   /* incremented by dispatcher to signal new work */
    atomic_int next_row;      /* work-stealing row index */
    atomic_int completed;     /* number of workers finished */
    atomic_int stop;          /* 1 = shutdown */
    atomic_int park;          /* 1 = avoid busy-waiting while another pool runs */
} tq2_lut_thread_pool_t;

static tq2_lut_thread_pool_t g_lut_pool = {
    {0}, 0, 0,
    NULL, NULL, NULL, NULL,
    0, 0, 0,
    NULL, NULL, NULL,
    0.0f, 0, NULL, NULL, NULL,
    0,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    NULL, NULL, NULL,
    0, 0, 0, 0, 0
};
static pthread_once_t g_lut_pool_once = PTHREAD_ONCE_INIT;

#define BITNET_TQ2_SPIN_ITERS 50000
#if defined(__ANDROID__)
#define BITNET_TQ2_PARK_USLEEP 1
#else
#define BITNET_TQ2_PARK_USLEEP 100
#endif
#ifndef BITNET_DISPATCHER_RUNS_WORK
#define BITNET_DISPATCHER_RUNS_WORK 1
#endif

static void init_lut_pool_once(void);

static int choose_thread_count(void) {
    const char *env = getenv("BITNET_NUM_THREADS");
    long n = 0;

    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            n = parsed;
        }
    }

    if (n <= 0) {
        n = 3;
    }
    if (n < 1) n = 1;
    if (n > 8) n = 8;
    return (int)n;
}

static int choose_i2s_chunk_groups(void) {
    static int cached = 0;
    const char *env = NULL;
    long n = 0;

    if (cached > 0) {
        return cached;
    }

    env = getenv("BITNET_I2S_CHUNK_GROUPS");
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            n = parsed;
        }
    }

    if (n <= 0) n = 16;
    if (n < 1) n = 1;
    if (n > 128) n = 128;
    cached = (int)n;
    return cached;
}

static void tq2_lut_compute_rows(const uint8_t *bytes, const uint8_t *bytes_b,
                                 const float *scales, const float *scales_b,
                                 int blocks_per_row,
                                 int out_start, int row_begin, int row_end,
                                 const float *lut, float *out, float *out_b) {
    for (int j = row_begin; j < row_end; ++j) {
        int row = out_start + j;
        size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        const float *row_scales = scales == NULL ? NULL : scales + (size_t)row * (size_t)blocks_per_row;
        const float *row_scales_b = scales_b == NULL ? NULL : scales_b + (size_t)row * (size_t)blocks_per_row;
        if (bytes_b != NULL && out_b != NULL) {
            if (row_scales != NULL && row_scales_b != NULL) {
                tq2_0_matmul_row_lut_pair_scales(bytes + row_offset, row_scales,
                                                 bytes_b + row_offset, row_scales_b,
                                                 blocks_per_row, lut, &out[j], &out_b[j]);
            } else {
                tq2_0_matmul_row_lut_pair(bytes + row_offset, bytes_b + row_offset,
                                          blocks_per_row, lut, &out[j], &out_b[j]);
            }
        } else {
            if (row_scales != NULL) {
                tq2_0_matmul_row_lut_scales(bytes + row_offset, row_scales,
                                            blocks_per_row, lut, &out[j]);
            } else {
                tq2_0_matmul_row_lut(bytes + row_offset, blocks_per_row, lut, &out[j]);
            }
        }
    }
}

static void tq2_lut_compute_rows_tiled_single_scales(const uint8_t *bytes,
                                                     const float *scales,
                                                     int blocks_per_row,
                                                     int out_start, int row_begin, int row_end,
                                                     const float *lut, float *out) {
    const size_t row_stride = (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
    enum { tile_rows = 256 };
    float acc[tile_rows];

    for (int j = row_begin; j < row_end; ++j) {
        out[j] = 0.0f;
    }

    for (int tile_begin = row_begin; tile_begin < row_end; tile_begin += tile_rows) {
        int tile_end = tile_begin + tile_rows;
        if (tile_end > row_end) tile_end = row_end;
        const int tile_count = tile_end - tile_begin;

        for (int k = 0; k < blocks_per_row; ++k) {
            for (int j = 0; j < tile_count; ++j) {
                acc[j] = 0.0f;
            }

            for (int g = 0; g < 2; ++g) {
                const float *group_lut = lut + (((size_t)k * 2u + (size_t)g) * 32u * 256u);

                for (int m = 0; m < 32; ++m) {
                    const float *entry = group_lut + (size_t)m * 256u;

                    for (int j = 0; j < tile_count; ++j) {
                        const int row = out_start + tile_begin + j;
                        const uint8_t *block_bytes =
                            bytes + (size_t)row * row_stride + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
                        acc[j] += entry[block_bytes[(size_t)g * 32u + (size_t)m]];
                    }
                }
            }

            for (int j = 0; j < tile_count; ++j) {
                const int out_row = tile_begin + j;
                const int row = out_start + out_row;
                const size_t scale_idx = (size_t)row * (size_t)blocks_per_row + (size_t)k;
                out[out_row] += acc[j] * scales[scale_idx];
            }
        }
    }
}

static void tq2_lut_compute_rows_tiled_pair_scales(const uint8_t *bytes, const uint8_t *bytes_b,
                                                   const float *scales, const float *scales_b,
                                                   int blocks_per_row,
                                                   int out_start, int row_begin, int row_end,
                                                   const float *lut, float *out, float *out_b) {
    tq2_lut_compute_rows_tiled_single_scales(bytes, scales, blocks_per_row,
                                             out_start, row_begin, row_end,
                                             lut, out);
    tq2_lut_compute_rows_tiled_single_scales(bytes_b, scales_b, blocks_per_row,
                                             out_start, row_begin, row_end,
                                             lut, out_b);
}

static void tq2_lut_compute_rows_tiled_scales(const uint8_t *bytes, const uint8_t *bytes_b,
                                              const float *scales, const float *scales_b,
                                              int blocks_per_row,
                                              int out_start, int row_begin, int row_end,
                                              const float *lut, float *out, float *out_b) {
    if (bytes_b != NULL && out_b != NULL && scales_b != NULL) {
        tq2_lut_compute_rows_tiled_pair_scales(bytes, bytes_b, scales, scales_b,
                                               blocks_per_row, out_start,
                                               row_begin, row_end, lut, out, out_b);
    } else {
        tq2_lut_compute_rows_tiled_single_scales(bytes, scales, blocks_per_row,
                                                 out_start, row_begin, row_end,
                                                 lut, out);
    }
}

static void tq2_lut_pool_run_chunks(void) {
    for (;;) {
        /* I2S dispatch: work-stealing at group granularity (4 rows per group) */
        if (g_lut_pool.dispatch_type == 1 || g_lut_pool.dispatch_type == 2 ||
            g_lut_pool.dispatch_type == 3) {
#if BITNET_TQ2_HAS_NEON_DOTPROD
            const int chunk_groups = g_lut_pool.i2s_chunk_groups > 0 ?
                                     g_lut_pool.i2s_chunk_groups : 16;
            int grp_begin = atomic_fetch_add_explicit(&g_lut_pool.next_row,
                                                       chunk_groups,
                                                       memory_order_relaxed);
            int grp_end = grp_begin + chunk_groups;
            if (grp_begin >= g_lut_pool.i2s_n_groups) break;
            if (grp_end > g_lut_pool.i2s_n_groups) grp_end = g_lut_pool.i2s_n_groups;

            const int sbpg = g_lut_pool.i2s_sub_blocks_per_group;
            const int bpr = g_lut_pool.i2s_blocks_per_row;
            const int8_t *qvec = g_lut_pool.qvec;
            float vec_scale = g_lut_pool.vec_scale;
            const int32_t *bsums = g_lut_pool.i2s_bsums;

            if (g_lut_pool.dispatch_type == 3) {
                int grp = grp_begin;
                while (grp < grp_end) {
                    if (grp < g_lut_pool.i2s_q_groups) {
                        int row_base = grp * 4;
                        if (grp + 1 < grp_end && grp + 1 < g_lut_pool.i2s_q_groups) {
                            const uint8_t *packed_grp0 = g_lut_pool.i2s_packed + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                            const uint8_t *packed_grp1 = packed_grp0 + (size_t)sbpg * (size_t)QK_I2S;
                            const float *scales_grp0 = g_lut_pool.i2s_scales + (size_t)grp * (size_t)sbpg * 4;
                            const float *scales_grp1 = scales_grp0 + (size_t)sbpg * 4;
                            float r[8];

                            i2s_matmul_8rows_neon(packed_grp0, packed_grp1,
                                                  scales_grp0, scales_grp1,
                                                  bpr, qvec, bsums, vec_scale, r);

                            for (int k = 0; k < 8 && row_base + k < g_lut_pool.i2s_out_dim; ++k) {
                                g_lut_pool.i2s_out[row_base + k] = r[k];
                            }
                            grp += 2;
                        } else {
                            const uint8_t *packed_grp = g_lut_pool.i2s_packed + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                            const float *scales_grp = g_lut_pool.i2s_scales + (size_t)grp * (size_t)sbpg * 4;
                            float r0, r1, r2, r3;

                            i2s_matmul_4rows_neon(packed_grp, scales_grp,
                                                  bpr, qvec, bsums, vec_scale,
                                                  &r0, &r1, &r2, &r3);

                            if (row_base + 0 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 0] = r0;
                            if (row_base + 1 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 1] = r1;
                            if (row_base + 2 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 2] = r2;
                            if (row_base + 3 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 3] = r3;
                            ++grp;
                        }
                    } else {
                        const int kv_grp = grp - g_lut_pool.i2s_q_groups;
                        const int row_base = kv_grp * 4;
                        const uint8_t *packed_k = g_lut_pool.i2s_packed_b + (size_t)kv_grp * (size_t)sbpg * (size_t)QK_I2S;
                        const uint8_t *packed_v = g_lut_pool.i2s_packed_c + (size_t)kv_grp * (size_t)sbpg * (size_t)QK_I2S;
                        const float *scales_k = g_lut_pool.i2s_scales_b + (size_t)kv_grp * (size_t)sbpg * 4;
                        const float *scales_v = g_lut_pool.i2s_scales_c + (size_t)kv_grp * (size_t)sbpg * 4;
                        float k0, k1, k2, k3;
                        float v0, v1, v2, v3;

                        i2s_matmul_4rows_neon(packed_k, scales_k,
                                              bpr, qvec, bsums, vec_scale,
                                              &k0, &k1, &k2, &k3);
                        i2s_matmul_4rows_neon(packed_v, scales_v,
                                              bpr, qvec, bsums, vec_scale,
                                              &v0, &v1, &v2, &v3);

                        if (row_base + 0 < g_lut_pool.i2s_kv_out_dim) {
                            g_lut_pool.i2s_out_b[row_base + 0] = k0;
                            g_lut_pool.i2s_out_c[row_base + 0] = v0;
                        }
                        if (row_base + 1 < g_lut_pool.i2s_kv_out_dim) {
                            g_lut_pool.i2s_out_b[row_base + 1] = k1;
                            g_lut_pool.i2s_out_c[row_base + 1] = v1;
                        }
                        if (row_base + 2 < g_lut_pool.i2s_kv_out_dim) {
                            g_lut_pool.i2s_out_b[row_base + 2] = k2;
                            g_lut_pool.i2s_out_c[row_base + 2] = v2;
                        }
                        if (row_base + 3 < g_lut_pool.i2s_kv_out_dim) {
                            g_lut_pool.i2s_out_b[row_base + 3] = k3;
                            g_lut_pool.i2s_out_c[row_base + 3] = v3;
                        }
                        ++grp;
                    }
                }
            } else if (g_lut_pool.dispatch_type == 2) {
                /* I2S pair kernel */
                int grp = grp_begin;
                for (; grp + 1 < grp_end && grp + 1 < g_lut_pool.i2s_n_groups; grp += 2) {
                    int row_base = grp * 4;
                    const uint8_t *pg_a0 = g_lut_pool.i2s_packed + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                    const uint8_t *pg_a1 = pg_a0 + (size_t)sbpg * (size_t)QK_I2S;
                    const uint8_t *pg_b0 = g_lut_pool.i2s_packed_b + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                    const uint8_t *pg_b1 = pg_b0 + (size_t)sbpg * (size_t)QK_I2S;
                    const float *sc_a0 = g_lut_pool.i2s_scales + (size_t)grp * (size_t)sbpg * 4u;
                    const float *sc_a1 = sc_a0 + (size_t)sbpg * 4u;
                    const float *sc_b0 = g_lut_pool.i2s_scales_b + (size_t)grp * (size_t)sbpg * 4u;
                    const float *sc_b1 = sc_b0 + (size_t)sbpg * 4u;
                    float ra[8];
                    float rb[8];

                    i2s_matmul_8rows_neon(pg_a0, pg_a1, sc_a0, sc_a1,
                                          bpr, qvec, bsums, vec_scale, ra);
                    i2s_matmul_8rows_neon(pg_b0, pg_b1, sc_b0, sc_b1,
                                          bpr, qvec, bsums, vec_scale, rb);

                    for (int k = 0; k < 8 && row_base + k < g_lut_pool.i2s_out_dim; ++k) {
                        g_lut_pool.i2s_out[row_base + k] = ra[k];
                        g_lut_pool.i2s_out_b[row_base + k] = rb[k];
                    }
                }
                for (; grp < grp_end; ++grp) {
                    int row_base = grp * 4;
                    const uint8_t *pg_a = g_lut_pool.i2s_packed + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                    const uint8_t *pg_b = g_lut_pool.i2s_packed_b + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                    const float *sc_a = g_lut_pool.i2s_scales + (size_t)grp * (size_t)sbpg * 4u;
                    const float *sc_b = g_lut_pool.i2s_scales_b + (size_t)grp * (size_t)sbpg * 4u;
                    float a0, a1, a2, a3;
                    float b0, b1, b2, b3;

                    i2s_matmul_4rows_neon(pg_a, sc_a, bpr, qvec, bsums, vec_scale,
                                          &a0, &a1, &a2, &a3);
                    i2s_matmul_4rows_neon(pg_b, sc_b, bpr, qvec, bsums, vec_scale,
                                          &b0, &b1, &b2, &b3);

                    if (row_base + 0 < g_lut_pool.i2s_out_dim) {
                        g_lut_pool.i2s_out[row_base + 0] = a0;
                        g_lut_pool.i2s_out_b[row_base + 0] = b0;
                    }
                    if (row_base + 1 < g_lut_pool.i2s_out_dim) {
                        g_lut_pool.i2s_out[row_base + 1] = a1;
                        g_lut_pool.i2s_out_b[row_base + 1] = b1;
                    }
                    if (row_base + 2 < g_lut_pool.i2s_out_dim) {
                        g_lut_pool.i2s_out[row_base + 2] = a2;
                        g_lut_pool.i2s_out_b[row_base + 2] = b2;
                    }
                    if (row_base + 3 < g_lut_pool.i2s_out_dim) {
                        g_lut_pool.i2s_out[row_base + 3] = a3;
                        g_lut_pool.i2s_out_b[row_base + 3] = b3;
                    }
                }
            } else {
                /* I2S single kernel */
                int grp = grp_begin;
                for (; grp + 1 < grp_end && grp + 1 < g_lut_pool.i2s_n_groups; grp += 2) {
                    int row_base = grp * 4;
                    const uint8_t *packed_grp0 = g_lut_pool.i2s_packed + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                    const uint8_t *packed_grp1 = packed_grp0 + (size_t)sbpg * (size_t)QK_I2S;
                    const float *scales_grp0 = g_lut_pool.i2s_scales + (size_t)grp * (size_t)sbpg * 4;
                    const float *scales_grp1 = scales_grp0 + (size_t)sbpg * 4;
                    float r[8];

                    i2s_matmul_8rows_neon(packed_grp0, packed_grp1,
                                          scales_grp0, scales_grp1,
                                          bpr, qvec, bsums, vec_scale, r);

                    for (int k = 0; k < 8 && row_base + k < g_lut_pool.i2s_out_dim; ++k) {
                        g_lut_pool.i2s_out[row_base + k] = r[k];
                    }
                }
                for (; grp < grp_end; ++grp) {
                    int row_base = grp * 4;
                    const uint8_t *packed_grp = g_lut_pool.i2s_packed + (size_t)grp * (size_t)sbpg * (size_t)QK_I2S;
                    const float *scales_grp = g_lut_pool.i2s_scales + (size_t)grp * (size_t)sbpg * 4;

                    float r0, r1, r2, r3;
                    i2s_matmul_4rows_neon(packed_grp, scales_grp,
                                          bpr, qvec, bsums, vec_scale,
                                          &r0, &r1, &r2, &r3);

                    if (row_base + 0 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 0] = r0;
                    if (row_base + 1 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 1] = r1;
                    if (row_base + 2 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 2] = r2;
                    if (row_base + 3 < g_lut_pool.i2s_out_dim) g_lut_pool.i2s_out[row_base + 3] = r3;
                }
            }
#else
            /* Non-NEON: no I2S support, just break */
            break;
#endif
        } else if (g_lut_pool.i16_lut != NULL) {
            const int chunk_rows = 32;
            int row_begin = atomic_fetch_add_explicit(&g_lut_pool.next_row,
                                                      chunk_rows,
                                                      memory_order_relaxed);
            int row_end = row_begin + chunk_rows;
            if (row_begin >= g_lut_pool.out_count) break;
            if (row_end > g_lut_pool.out_count) row_end = g_lut_pool.out_count;

            for (int j = row_begin; j < row_end; ++j) {
                int row = g_lut_pool.out_start + j;
                size_t row_offset = (size_t)row * (size_t)g_lut_pool.blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                const float *row_scales =
                    g_lut_pool.scales + (size_t)row * (size_t)g_lut_pool.blocks_per_row;
                tq2_0_matmul_row_i16_lut_scales(g_lut_pool.bytes + row_offset,
                                                row_scales,
                                                g_lut_pool.blocks_per_row,
                                                g_lut_pool.i16_lut,
                                                g_lut_pool.vec_scale,
                                                &g_lut_pool.out[j]);
            }
        } else if (g_lut_pool.qvec != NULL) {
            const int chunk_rows = 32;
            int row_begin = atomic_fetch_add_explicit(&g_lut_pool.next_row,
                                                      chunk_rows,
                                                      memory_order_relaxed);
            int row_end = row_begin + chunk_rows;
            if (row_begin >= g_lut_pool.out_count) break;
            if (row_end > g_lut_pool.out_count) row_end = g_lut_pool.out_count;

            for (int j = row_begin; j < row_end; ++j) {
                int row = g_lut_pool.out_start + j;
                size_t row_offset = (size_t)row * (size_t)g_lut_pool.blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                const float *row_scales =
                    g_lut_pool.scales + (size_t)row * (size_t)g_lut_pool.blocks_per_row;
                const float *row_scales_b =
                    g_lut_pool.scales_b == NULL ? NULL :
                    g_lut_pool.scales_b + (size_t)row * (size_t)g_lut_pool.blocks_per_row;
                if (g_lut_pool.use_neon_i8) {
                    if (g_lut_pool.bytes_b != NULL && row_scales_b != NULL && g_lut_pool.out_b != NULL) {
                        tq2_0_matmul_row_i8_neon_pair_scales(g_lut_pool.bytes + row_offset,
                                                             row_scales,
                                                             g_lut_pool.bytes_b + row_offset,
                                                             row_scales_b,
                                                             g_lut_pool.blocks_per_row,
                                                             g_lut_pool.qvec,
                                                             g_lut_pool.vec_scale,
                                                             g_lut_pool.block_bsums,
                                                             &g_lut_pool.out[j],
                                                             &g_lut_pool.out_b[j]);
                    } else {
#if BITNET_TQ2_HAS_NEON_DOTPROD
                        /* Use quad kernel when 4+ rows remain (maximizes qvec reuse) */
                        if (row_end - j >= 4) {
                            int r0 = g_lut_pool.out_start + j;
                            int r1 = r0 + 1;
                            int r2 = r0 + 2;
                            int r3 = r0 + 3;
                            size_t off0 = (size_t)r0 * (size_t)g_lut_pool.blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                            size_t off1 = (size_t)r1 * (size_t)g_lut_pool.blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                            size_t off2 = (size_t)r2 * (size_t)g_lut_pool.blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                            size_t off3 = (size_t)r3 * (size_t)g_lut_pool.blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                            tq2_0_matmul_row_i8_neon_quad_scales(
                                g_lut_pool.bytes + off0,
                                g_lut_pool.scales + (size_t)r0 * (size_t)g_lut_pool.blocks_per_row,
                                g_lut_pool.bytes + off1,
                                g_lut_pool.scales + (size_t)r1 * (size_t)g_lut_pool.blocks_per_row,
                                g_lut_pool.bytes + off2,
                                g_lut_pool.scales + (size_t)r2 * (size_t)g_lut_pool.blocks_per_row,
                                g_lut_pool.bytes + off3,
                                g_lut_pool.scales + (size_t)r3 * (size_t)g_lut_pool.blocks_per_row,
                                g_lut_pool.blocks_per_row,
                                g_lut_pool.qvec,
                                g_lut_pool.vec_scale,
                                g_lut_pool.block_bsums,
                                &g_lut_pool.out[j],
                                &g_lut_pool.out[j + 1],
                                &g_lut_pool.out[j + 2],
                                &g_lut_pool.out[j + 3]);
                            j += 3; /* skip 3 extra rows (loop will ++j) */
                            continue;
                        }
#endif
                        tq2_0_matmul_row_i8_neon_scales(g_lut_pool.bytes + row_offset,
                                                        row_scales,
                                                        g_lut_pool.blocks_per_row,
                                                        g_lut_pool.qvec,
                                                        g_lut_pool.vec_scale,
                                                        g_lut_pool.block_bsums,
                                                        &g_lut_pool.out[j]);
                    }
                } else {
                    tq2_0_matmul_row_i8_scales(g_lut_pool.bytes + row_offset,
                                               row_scales,
                                               g_lut_pool.blocks_per_row,
                                               g_lut_pool.qvec,
                                               g_lut_pool.vec_scale,
                                               &g_lut_pool.out[j]);
                    if (g_lut_pool.bytes_b != NULL && row_scales_b != NULL && g_lut_pool.out_b != NULL) {
                        tq2_0_matmul_row_i8_scales(g_lut_pool.bytes_b + row_offset,
                                                   row_scales_b,
                                                   g_lut_pool.blocks_per_row,
                                                   g_lut_pool.qvec,
                                                   g_lut_pool.vec_scale,
                                                   &g_lut_pool.out_b[j]);
                    }
                }
            }
        } else if (g_lut_pool.scales != NULL &&
            (g_lut_pool.bytes_b == NULL || g_lut_pool.scales_b != NULL)) {
            const int chunk_rows = 32;
            int row_begin = atomic_fetch_add_explicit(&g_lut_pool.next_row,
                                                      chunk_rows,
                                                      memory_order_relaxed);
            int row_end = row_begin + chunk_rows;
            if (row_begin >= g_lut_pool.out_count) break;
            if (row_end > g_lut_pool.out_count) row_end = g_lut_pool.out_count;

            tq2_lut_compute_rows_tiled_scales(g_lut_pool.bytes,
                                              g_lut_pool.bytes_b,
                                              g_lut_pool.scales,
                                              g_lut_pool.scales_b,
                                              g_lut_pool.blocks_per_row,
                                              g_lut_pool.out_start,
                                              row_begin,
                                              row_end,
                                              g_lut_pool.lut,
                                              g_lut_pool.out,
                                              g_lut_pool.out_b);
        } else {
            int j = atomic_fetch_add_explicit(&g_lut_pool.next_row, 1, memory_order_relaxed);
            if (j >= g_lut_pool.out_count) break;

            int row = g_lut_pool.out_start + j;
            size_t row_offset = (size_t)row * (size_t)g_lut_pool.blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
            if (g_lut_pool.bytes_b != NULL && g_lut_pool.out_b != NULL) {
                tq2_0_matmul_row_lut_pair(g_lut_pool.bytes + row_offset,
                                          g_lut_pool.bytes_b + row_offset,
                                          g_lut_pool.blocks_per_row,
                                          g_lut_pool.lut,
                                          &g_lut_pool.out[j],
                                          &g_lut_pool.out_b[j]);
            } else {
                tq2_0_matmul_row_lut(g_lut_pool.bytes + row_offset,
                                     g_lut_pool.blocks_per_row,
                                     g_lut_pool.lut,
                                     &g_lut_pool.out[j]);
            }
        }
    }
}

static void *tq2_lut_pool_worker_main(void *ptr) {
    unsigned seen_generation = 0;
    (void)ptr;

#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    for (;;) {
        /* Hybrid wait: spin briefly for low latency, then yield to avoid wasting CPU */
        int spin_count = 0;
        while (atomic_load_explicit(&g_lut_pool.generation, memory_order_acquire) == seen_generation) {
            if (atomic_load_explicit(&g_lut_pool.stop, memory_order_acquire)) {
                return NULL;
            }
            if (atomic_load_explicit(&g_lut_pool.park, memory_order_acquire)) {
                usleep(BITNET_TQ2_PARK_USLEEP);
                spin_count = 0;
                continue;
            }
            if (++spin_count < BITNET_TQ2_SPIN_ITERS) {
                __asm__ __volatile__("yield" ::: "memory");
            } else {
                /* Back off to avoid burning CPU while idle */
                usleep(1);
                spin_count = 0;
            }
        }
        seen_generation = atomic_load_explicit(&g_lut_pool.generation, memory_order_acquire);

        tq2_lut_pool_run_chunks();

        /* Signal completion */
        atomic_fetch_add_explicit(&g_lut_pool.completed, 1, memory_order_release);
    }
}

void bitnet_tq2_0_park_workers(void) {
    (void)pthread_once(&g_lut_pool_once, init_lut_pool_once);
    if (g_lut_pool.initialized && g_lut_pool.n_threads > 1) {
        atomic_store_explicit(&g_lut_pool.park, 1, memory_order_release);
    }
}

static void init_lut_pool_once(void) {
    int n_threads = choose_thread_count();

    if (n_threads <= 1) {
        g_lut_pool.initialized = 1;
        return;
    }

    for (int i = 0; i < n_threads; ++i) {
        if (pthread_create(&g_lut_pool.threads[i], NULL, tq2_lut_pool_worker_main, NULL) != 0) {
            break;
        }
        ++g_lut_pool.n_threads;
    }

    /* Mark all workers as "completed" so first dispatch doesn't deadlock */
    atomic_store_explicit(&g_lut_pool.completed, g_lut_pool.n_threads, memory_order_release);
    g_lut_pool.initialized = 1;
}

/* Hybrid spin-wait: spin briefly for low latency, then back off to avoid wasting CPU */
#define BITNET_SPIN_WAIT(cond) do { \
    int _spin = 0; \
    while (!(cond)) { \
        if (++_spin < BITNET_TQ2_SPIN_ITERS) { \
            __asm__ __volatile__("yield" ::: "memory"); \
        } else { \
            usleep(1); \
            _spin = 0; \
        } \
    } \
} while (0)

static int tq2_0_matmul_rows_lut_parallel(const uint8_t *bytes, const uint8_t *bytes_b,
                                          const float *scales, const float *scales_b,
                                          int blocks_per_row,
                                          int out_start, int out_count,
                                          const float *lut, float *out, float *out_b) {
    (void)pthread_once(&g_lut_pool_once, init_lut_pool_once);

    if (!g_lut_pool.initialized || g_lut_pool.n_threads <= 1 || out_count < 16) {
        if (scales != NULL && (bytes_b == NULL || scales_b != NULL)) {
            tq2_lut_compute_rows_tiled_scales(bytes, bytes_b, scales, scales_b,
                                              blocks_per_row, out_start,
                                              0, out_count, lut, out, out_b);
        } else {
            tq2_lut_compute_rows(bytes, bytes_b, scales, scales_b, blocks_per_row,
                                 out_start, 0, out_count, lut, out, out_b);
        }
        return 0;
    }

    /* Wait for previous work to complete */
    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    /* Set task parameters */
    g_lut_pool.dispatch_type = 0; /* TQ2_0 dispatch */
    g_lut_pool.bytes = bytes;
    g_lut_pool.bytes_b = bytes_b;
    g_lut_pool.scales = scales;
    g_lut_pool.scales_b = scales_b;
    g_lut_pool.blocks_per_row = blocks_per_row;
    g_lut_pool.out_start = out_start;
    g_lut_pool.out_count = out_count;
    g_lut_pool.lut = lut;
    g_lut_pool.qvec = NULL;
    g_lut_pool.i16_lut = NULL;
    g_lut_pool.vec_scale = 0.0f;
    g_lut_pool.use_neon_i8 = 0;
    g_lut_pool.out = out;
    g_lut_pool.out_b = out_b;
    atomic_store_explicit(&g_lut_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_lut_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_lut_pool.completed, 0, memory_order_release);
    /* Increment generation to wake workers (release ensures params are visible) */
    atomic_fetch_add_explicit(&g_lut_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    tq2_lut_pool_run_chunks();
#endif

    /* Wait for all workers to complete */
    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    return 0;
}

static int tq2_0_matmul_rows_i8_parallel(const uint8_t *bytes, const uint8_t *bytes_b,
                                         const float *scales, const float *scales_b,
                                         int blocks_per_row,
                                         int out_start, int out_count,
                                         const int8_t *qvec, float vec_scale,
                                         const int32_t *block_bsums, int use_neon,
                                         float *out, float *out_b) {
    (void)pthread_once(&g_lut_pool_once, init_lut_pool_once);

    if (!g_lut_pool.initialized || g_lut_pool.n_threads <= 1 || out_count < 16) {
        if (use_neon && bytes_b == NULL) {
            /* Use quad kernel for better qvec data reuse and memory prefetch */
            int j = 0;
            for (; j + 3 < out_count; j += 4) {
                int r0 = out_start + j;
                size_t off0 = (size_t)r0 * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                size_t off1 = off0 + (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                size_t off2 = off1 + (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                size_t off3 = off2 + (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                tq2_0_matmul_row_i8_neon_quad_scales(
                    bytes + off0, scales + (size_t)r0 * (size_t)blocks_per_row,
                    bytes + off1, scales + (size_t)(r0 + 1) * (size_t)blocks_per_row,
                    bytes + off2, scales + (size_t)(r0 + 2) * (size_t)blocks_per_row,
                    bytes + off3, scales + (size_t)(r0 + 3) * (size_t)blocks_per_row,
                    blocks_per_row, qvec, vec_scale, block_bsums,
                    &out[j], &out[j + 1], &out[j + 2], &out[j + 3]);
            }
            for (; j < out_count; ++j) {
                int row = out_start + j;
                size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                const float *row_scales = scales + (size_t)row * (size_t)blocks_per_row;
                tq2_0_matmul_row_i8_neon_scales(bytes + row_offset, row_scales,
                                                blocks_per_row, qvec, vec_scale, block_bsums, &out[j]);
            }
        } else if (use_neon && bytes_b != NULL && out_b != NULL) {
            /* Pair kernel path */
            for (int j = 0; j < out_count; ++j) {
                int row = out_start + j;
                size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                const float *row_scales = scales + (size_t)row * (size_t)blocks_per_row;
                const float *row_scales_b = scales_b + (size_t)row * (size_t)blocks_per_row;
                tq2_0_matmul_row_i8_neon_pair_scales(bytes + row_offset, row_scales,
                                                     bytes_b + row_offset, row_scales_b,
                                                     blocks_per_row, qvec, vec_scale, block_bsums,
                                                     &out[j], &out_b[j]);
            }
        } else {
            for (int j = 0; j < out_count; ++j) {
                int row = out_start + j;
                size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
                const float *row_scales = scales + (size_t)row * (size_t)blocks_per_row;
                tq2_0_matmul_row_i8_scales(bytes + row_offset, row_scales,
                                           blocks_per_row, qvec, vec_scale, &out[j]);
                if (bytes_b != NULL && out_b != NULL) {
                    const float *row_scales_b = scales_b + (size_t)row * (size_t)blocks_per_row;
                    tq2_0_matmul_row_i8_scales(bytes_b + row_offset, row_scales_b,
                                               blocks_per_row, qvec, vec_scale, &out_b[j]);
                }
            }
        }
        return 0;
    }

    /* Wait for previous work to complete */
    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    /* Set task parameters */
    g_lut_pool.dispatch_type = 0; /* TQ2_0 dispatch */
    g_lut_pool.bytes = bytes;
    g_lut_pool.bytes_b = bytes_b;
    g_lut_pool.scales = scales;
    g_lut_pool.scales_b = scales_b;
    g_lut_pool.blocks_per_row = blocks_per_row;
    g_lut_pool.out_start = out_start;
    g_lut_pool.out_count = out_count;
    g_lut_pool.lut = NULL;
    g_lut_pool.qvec = qvec;
    g_lut_pool.i16_lut = NULL;
    g_lut_pool.vec_scale = vec_scale;
    g_lut_pool.use_neon_i8 = use_neon;
    g_lut_pool.block_bsums = block_bsums;
    g_lut_pool.out = out;
    g_lut_pool.out_b = out_b;
    atomic_store_explicit(&g_lut_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_lut_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_lut_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_lut_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    tq2_lut_pool_run_chunks();
#endif

    /* Wait for all workers to complete */
    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    return 0;
}

static int tq2_0_matmul_rows_i16_lut_parallel(const uint8_t *bytes, const float *scales,
                                              int blocks_per_row,
                                              int out_start, int out_count,
                                              const int16_t *lut, float vec_scale,
                                              float *out) {
    (void)pthread_once(&g_lut_pool_once, init_lut_pool_once);

    if (!g_lut_pool.initialized || g_lut_pool.n_threads <= 1 || out_count < 16) {
        for (int j = 0; j < out_count; ++j) {
            int row = out_start + j;
            size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
            const float *row_scales = scales + (size_t)row * (size_t)blocks_per_row;
            tq2_0_matmul_row_i16_lut_scales(bytes + row_offset, row_scales,
                                            blocks_per_row, lut, vec_scale, &out[j]);
        }
        return 0;
    }

    /* Wait for previous work to complete */
    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    g_lut_pool.dispatch_type = 0; /* TQ2_0 dispatch */
    g_lut_pool.bytes = bytes;
    g_lut_pool.bytes_b = NULL;
    g_lut_pool.scales = scales;
    g_lut_pool.scales_b = NULL;
    g_lut_pool.blocks_per_row = blocks_per_row;
    g_lut_pool.out_start = out_start;
    g_lut_pool.out_count = out_count;
    g_lut_pool.lut = NULL;
    g_lut_pool.qvec = NULL;
    g_lut_pool.i16_lut = lut;
    g_lut_pool.vec_scale = vec_scale;
    g_lut_pool.use_neon_i8 = 0;
    g_lut_pool.out = out;
    g_lut_pool.out_b = NULL;
    atomic_store_explicit(&g_lut_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_lut_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_lut_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_lut_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    tq2_lut_pool_run_chunks();
#endif

    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    return 0;
}

#if BITNET_TQ2_HAS_NEON_DOTPROD
/* Parallel I2S matmul dispatcher — single matrix */
static int i2s_matmul_parallel(const uint8_t *packed, const float *scales,
                                const int32_t *bsums, int out_dim, int in_dim,
                                const int8_t *qvec, float vec_scale, float *out) {
    int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    int out_dim_padded = (out_dim + 3) & ~3;
    int n_groups = out_dim_padded / 4;
    int sub_blocks_per_group = in_dim / QK_I2S;

    (void)pthread_once(&g_lut_pool_once, init_lut_pool_once);

    if (!g_lut_pool.initialized || g_lut_pool.n_threads <= 1 || n_groups < 4) {
        /* Fall back to single-threaded */
        return bitnet_tq2_0_matmul_i2s_neon(packed, scales, bsums, out_dim, in_dim,
                                              qvec, vec_scale, out);
    }

    /* Wait for previous work to complete */
    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    /* Set task parameters */
    g_lut_pool.dispatch_type = 1;
    g_lut_pool.i2s_packed = packed;
    g_lut_pool.i2s_packed_b = NULL;
    g_lut_pool.i2s_scales = scales;
    g_lut_pool.i2s_scales_b = NULL;
    g_lut_pool.i2s_bsums = bsums;
    g_lut_pool.i2s_out_dim = out_dim;
    g_lut_pool.i2s_in_dim = in_dim;
    g_lut_pool.i2s_blocks_per_row = blocks_per_row;
    g_lut_pool.i2s_n_groups = n_groups;
    g_lut_pool.i2s_sub_blocks_per_group = sub_blocks_per_group;
    g_lut_pool.i2s_chunk_groups = choose_i2s_chunk_groups();
    g_lut_pool.i2s_out = out;
    g_lut_pool.i2s_out_b = NULL;
    g_lut_pool.qvec = qvec;
    g_lut_pool.vec_scale = vec_scale;
    /* Clear TQ2_0 dispatch fields to avoid confusion */
    g_lut_pool.i16_lut = NULL;
    g_lut_pool.lut = NULL;
    atomic_store_explicit(&g_lut_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_lut_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_lut_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_lut_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    tq2_lut_pool_run_chunks();
#endif

    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    return 0;
}

/* Parallel I2S matmul dispatcher — pair (gate + up) */
static int i2s_matmul_pair_parallel(const uint8_t *packed_a, const float *scales_a,
                                     const uint8_t *packed_b, const float *scales_b,
                                     const int32_t *bsums, int out_dim, int in_dim,
                                     const int8_t *qvec, float vec_scale,
                                     float *out_a, float *out_b) {
    int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    int out_dim_padded = (out_dim + 3) & ~3;
    int n_groups = out_dim_padded / 4;
    int sub_blocks_per_group = in_dim / QK_I2S;

    (void)pthread_once(&g_lut_pool_once, init_lut_pool_once);

    if (!g_lut_pool.initialized || g_lut_pool.n_threads <= 1 || n_groups < 4) {
        /* Fall back to single-threaded */
        return bitnet_tq2_0_matmul_i2s_neon_pair(packed_a, scales_a, packed_b, scales_b,
                                                    bsums, out_dim, in_dim,
                                                    qvec, vec_scale, out_a, out_b);
    }

    /* Wait for previous work to complete */
    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    /* Set task parameters */
    g_lut_pool.dispatch_type = 2;
    g_lut_pool.i2s_packed = packed_a;
    g_lut_pool.i2s_packed_b = packed_b;
    g_lut_pool.i2s_scales = scales_a;
    g_lut_pool.i2s_scales_b = scales_b;
    g_lut_pool.i2s_bsums = bsums;
    g_lut_pool.i2s_out_dim = out_dim;
    g_lut_pool.i2s_in_dim = in_dim;
    g_lut_pool.i2s_blocks_per_row = blocks_per_row;
    g_lut_pool.i2s_n_groups = n_groups;
    g_lut_pool.i2s_sub_blocks_per_group = sub_blocks_per_group;
    g_lut_pool.i2s_chunk_groups = choose_i2s_chunk_groups();
    g_lut_pool.i2s_out = out_a;
    g_lut_pool.i2s_out_b = out_b;
    g_lut_pool.qvec = qvec;
    g_lut_pool.vec_scale = vec_scale;
    /* Clear TQ2_0 dispatch fields to avoid confusion */
    g_lut_pool.i16_lut = NULL;
    g_lut_pool.lut = NULL;
    atomic_store_explicit(&g_lut_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_lut_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_lut_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_lut_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    tq2_lut_pool_run_chunks();
#endif

    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    return 0;
}

/* Parallel I2S fused Q/K/V dispatcher. Q is a full matrix; K and V are
 * smaller matrices computed in the same worker wakeup. */
static int i2s_matmul_qkv_parallel(const uint8_t *packed_q, const float *scales_q,
                                   const uint8_t *packed_k, const float *scales_k,
                                   const uint8_t *packed_v, const float *scales_v,
                                   const int32_t *bsums,
                                   int q_dim, int kv_dim, int in_dim,
                                   const int8_t *qvec, float vec_scale,
                                   float *out_q, float *out_k, float *out_v) {
    int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    int q_dim_padded = (q_dim + 3) & ~3;
    int kv_dim_padded = (kv_dim + 3) & ~3;
    int q_groups = q_dim_padded / 4;
    int kv_groups = kv_dim_padded / 4;
    int sub_blocks_per_group = in_dim / QK_I2S;

    if (packed_q == NULL || scales_q == NULL || packed_k == NULL ||
        scales_k == NULL || packed_v == NULL || scales_v == NULL ||
        qvec == NULL || out_q == NULL || out_k == NULL ||
        out_v == NULL || q_dim <= 0 || kv_dim <= 0 || in_dim <= 0 ||
        in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row <= 0) {
        return -1;
    }

    (void)pthread_once(&g_lut_pool_once, init_lut_pool_once);

    if (!g_lut_pool.initialized || g_lut_pool.n_threads <= 1 || q_groups < 4) {
        if (bitnet_tq2_0_matmul_i2s_neon(packed_q, scales_q, bsums,
                                         q_dim, in_dim, qvec, vec_scale,
                                         out_q) != 0) {
            return -1;
        }
        return bitnet_tq2_0_matmul_i2s_neon_pair(packed_k, scales_k,
                                                 packed_v, scales_v,
                                                 bsums, kv_dim, in_dim,
                                                 qvec, vec_scale,
                                                 out_k, out_v);
    }

    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    g_lut_pool.dispatch_type = 3;
    g_lut_pool.i2s_packed = packed_q;
    g_lut_pool.i2s_packed_b = packed_k;
    g_lut_pool.i2s_packed_c = packed_v;
    g_lut_pool.i2s_scales = scales_q;
    g_lut_pool.i2s_scales_b = scales_k;
    g_lut_pool.i2s_scales_c = scales_v;
    g_lut_pool.i2s_bsums = bsums;
    g_lut_pool.i2s_out_dim = q_dim;
    g_lut_pool.i2s_kv_out_dim = kv_dim;
    g_lut_pool.i2s_in_dim = in_dim;
    g_lut_pool.i2s_blocks_per_row = blocks_per_row;
    g_lut_pool.i2s_q_groups = q_groups;
    g_lut_pool.i2s_kv_groups = kv_groups;
    g_lut_pool.i2s_n_groups = q_groups + kv_groups;
    g_lut_pool.i2s_sub_blocks_per_group = sub_blocks_per_group;
    g_lut_pool.i2s_chunk_groups = choose_i2s_chunk_groups();
    g_lut_pool.i2s_out = out_q;
    g_lut_pool.i2s_out_b = out_k;
    g_lut_pool.i2s_out_c = out_v;
    g_lut_pool.qvec = qvec;
    g_lut_pool.vec_scale = vec_scale;
    g_lut_pool.i16_lut = NULL;
    g_lut_pool.lut = NULL;
    atomic_store_explicit(&g_lut_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_lut_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_lut_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_lut_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    tq2_lut_pool_run_chunks();
#endif

    BITNET_SPIN_WAIT(atomic_load_explicit(&g_lut_pool.completed, memory_order_acquire) >= g_lut_pool.n_threads);

    return 0;
}
#endif

int bitnet_tq2_0_matmul_vector_lut(const void *weight, int out_dim, int in_dim, const float *lut, float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || lut == NULL || out == NULL || out_dim <= 0 || in_dim <= 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    return tq2_0_matmul_rows_lut_parallel(bytes, NULL, NULL, NULL,
                                          blocks_per_row, 0, out_dim, lut, out, NULL);
}

int bitnet_tq2_0_matmul_vector_lut_scales(const void *weight, const float *scales,
                                          int out_dim, int in_dim, const float *lut,
                                          float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;

    if (weight == NULL || scales == NULL || lut == NULL || out == NULL ||
        out_dim <= 0 || in_dim <= 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    return tq2_0_matmul_rows_lut_parallel(bytes, NULL, scales, NULL,
                                          blocks_per_row, 0, out_dim, lut, out, NULL);
}

int bitnet_tq2_0_matmul_vector_lut_pair(const void *weight_a, const void *weight_b,
                                        int out_dim, int in_dim, const float *lut,
                                        float *out_a, float *out_b) {
    const uint8_t *bytes_a = (const uint8_t *)weight_a;
    const uint8_t *bytes_b = (const uint8_t *)weight_b;
    int blocks_per_row = 0;

    if (weight_a == NULL || weight_b == NULL || lut == NULL ||
        out_a == NULL || out_b == NULL || out_dim <= 0 || in_dim <= 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    return tq2_0_matmul_rows_lut_parallel(bytes_a, bytes_b, NULL, NULL, blocks_per_row,
                                          0, out_dim, lut, out_a, out_b);
}

int bitnet_tq2_0_matmul_vector_lut_pair_scales(const void *weight_a, const float *scales_a,
                                               const void *weight_b, const float *scales_b,
                                               int out_dim, int in_dim, const float *lut,
                                               float *out_a, float *out_b) {
    const uint8_t *bytes_a = (const uint8_t *)weight_a;
    const uint8_t *bytes_b = (const uint8_t *)weight_b;
    int blocks_per_row = 0;

    if (weight_a == NULL || weight_b == NULL || scales_a == NULL || scales_b == NULL ||
        lut == NULL || out_a == NULL || out_b == NULL || out_dim <= 0 || in_dim <= 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    return tq2_0_matmul_rows_lut_parallel(bytes_a, bytes_b, scales_a, scales_b,
                                          blocks_per_row, 0, out_dim,
                                          lut, out_a, out_b);
}

int bitnet_tq2_0_matmul_vector(const void *weight, int out_dim, int in_dim, const float *vec, float *out) {
    return bitnet_tq2_0_matmul_vector_partial(weight, in_dim, 0, out_dim, vec, out);
}

int bitnet_tq2_0_matmul_vector_partial(const void *weight, int in_dim, int out_start, int out_count, const float *vec, float *out) {
    const uint8_t *bytes = (const uint8_t *)weight;
    int blocks_per_row = 0;
    int j = 0;

    if (weight == NULL || vec == NULL || out == NULL || in_dim <= 0 || out_count <= 0) {
        return -1;
    }

    blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    if (in_dim % BITNET_TQ2_0_QK != 0 || blocks_per_row == 0) {
        return -1;
    }

    if (out_count >= 512) {
        size_t lut_count = (size_t)blocks_per_row * 2u * 32u * 256u;
        float *lut = (float *)malloc(lut_count * sizeof(*lut));

        if (lut != NULL) {
            build_tq2_0_vec_lut_blocks(vec, blocks_per_row, lut);
            int rc = tq2_0_matmul_rows_lut_parallel(bytes, NULL, NULL, NULL, blocks_per_row,
                                                    out_start, out_count, lut, out, NULL);
            free(lut);
            if (rc == 0) return 0;
        }
    }

    for (j = 0; j < out_count; ++j) {
        int row = out_start + j;
        size_t row_offset = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
        tq2_0_matmul_row(bytes + row_offset, blocks_per_row, vec, &out[j]);
    }

    return 0;
}
