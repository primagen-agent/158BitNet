#include "quant_q8k.h"
#include "quant_tq2_0.h"
#include "quant_q6k.h"
#include "bitnet_dispatch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

int bitnet_quantize_q8k_impl(const float *input, int count, bitnet_q8k_block_t *blocks) {
    if (!input || !blocks || count <= 0 || count % 256) return -1;
    for (int b = 0; b < count / 256; ++b) {
        const float *x = input + b * 256;
        float maximum = 0.0f, signed_maximum = 0.0f;
        for (int i = 0; i < 256; ++i) {
            if (!isfinite(x[i])) return -1;
            if (fabsf(x[i]) > maximum) { maximum = fabsf(x[i]); signed_maximum = x[i]; }
        }
        if (!maximum) { memset(&blocks[b], 0, sizeof(blocks[b])); continue; }
        /* Keep the sign and choose the first absolute maximum, as Q8_K does.
         * This also fixes rounding differences at signed half-integer ties. */
        float inverse = -127.0f / signed_maximum;
        blocks[b].d = 1.0f / inverse;
        for (int i = 0; i < 256; ++i) {
            int value = (int)nearbyintf(inverse * x[i]);
            if (value > 127) value = 127;
            blocks[b].qs[i] = (int8_t)value;
        }
    }
    return 0;
}

static int dot16(const int8_t *a, const int8_t *b) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b)));
#else
    int sum = 0; for (int i = 0; i < 16; ++i) sum += a[i] * b[i]; return sum;
#endif
}

static float tq2_dot(const bitnet_tq2_0_block_t *w, const bitnet_q8k_block_t *x, int blocks) {
    float result = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        int sum = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int32x4_t acc = vdupq_n_s32(0);
        for (int group = 0; group < 2; ++group) {
            for (int shift = 0; shift < 4; ++shift) {
                for (int half = 0; half < 2; ++half) {
                    uint8x16_t packed = vld1q_u8(w[b].qs + group * 32 + half * 16);
                    uint8x16_t code = vandq_u8(vshlq_u8(packed, vdupq_n_s8((int8_t)(-2 * shift))), vdupq_n_u8(3));
                    int8x16_t signed_code = vsubq_s8(vreinterpretq_s8_u8(code), vdupq_n_s8(1));
                    acc = vdotq_s32(acc, signed_code, vld1q_s8(x[b].qs + group * 128 + shift * 32 + half * 16));
                }
            }
        }
        sum = vaddvq_s32(acc);
#else
        for (int group = 0; group < 2; ++group)
            for (int shift = 0; shift < 4; ++shift)
                for (int i = 0; i < 32; ++i)
                    sum += (((w[b].qs[group * 32 + i] >> (2 * shift)) & 3) - 1) *
                           x[b].qs[group * 128 + shift * 32 + i];
#endif
        float scale = bitnet_fp16_to_fp32(w[b].d) * x[b].d;
        result += scale * (float)sum;
    }
    return result;
}

static float q6_dot(const bitnet_q6k_block_t *w, const bitnet_q8k_block_t *x, int blocks) {
    float result = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        int sum = 0;
        for (int group = 0; group < 2; ++group) {
            const uint8_t *lo = w[b].ql + group * 64, *hi = w[b].qh + group * 32;
            for (int part = 0; part < 4; ++part) {
                for (int half = 0; half < 2; ++half) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                    /* Same integer (low|high)-32 values as the scalar path;
                     * NEON unpack and vdot are exact in the integer domain. */
                    uint8x16_t lo_vec = vld1q_u8(lo + (part & 1) * 32 + half * 16);
                    uint8x16_t lo_bits = part >= 2 ? vshrq_n_u8(lo_vec, 4) : lo_vec;
                    uint8x16_t low = vandq_u8(lo_bits, vdupq_n_u8(15));
                    uint8x16_t hi_vec = vld1q_u8(hi + half * 16);
                    uint8x16_t high = vshlq_u8(
                        vandq_u8(hi_vec, vdupq_n_u8((uint8_t)(3u << (2 * part)))),
                        vdupq_n_s8((int8_t)(4 - 2 * part)));
                    int8x16_t values = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(low, high)),
                                                vdupq_n_s8(32));
                    int dot = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), values,
                        vld1q_s8(x[b].qs + group * 128 + part * 32 + half * 16)));
                    sum += dot * w[b].scales[group * 8 + part * 2 + half];
#else
                    int8_t values[16];
                    for (int i = 0; i < 16; ++i) {
                        int j = half * 16 + i;
                        int low = (lo[(part & 1) * 32 + j] >> (part >= 2 ? 4 : 0)) & 15;
                        int high = ((hi[j] >> (2 * part)) & 3) << 4;
                        values[i] = (int8_t)((low | high) - 32);
                    }
                    int dot = dot16(values, x[b].qs + group * 128 + part * 32 + half * 16);
                    sum += dot * w[b].scales[group * 8 + part * 2 + half];
#endif
                }
            }
        }
        float scale = bitnet_fp16_to_fp32(w[b].d) * x[b].d;
        result += scale * (float)sum;
    }
    return result;
}

typedef struct q8k_job {
    const void *weights;
    const bitnet_q8k_block_t *input;
    float *output;
    int type, blocks;
} q8k_job_t;

static void compute_rows(void *opaque, int first, int last) {
    const q8k_job_t *job = (const q8k_job_t *)opaque;
    for (int row = first; row < last; ++row) {
        size_t offset = (size_t)row * (size_t)job->blocks;
        job->output[row] = job->type == 35 ?
            tq2_dot((const bitnet_tq2_0_block_t *)job->weights + offset, job->input, job->blocks) :
            q6_dot((const bitnet_q6k_block_t *)job->weights + offset, job->input, job->blocks);
    }
}

int bitnet_matmul_q8k_prepared(const void *weights, int type, int rows, int columns,
                               const bitnet_q8k_block_t *input, float *output) {
    if (!weights || !input || !output || rows <= 0 || columns <= 0 || columns % 256 ||
        (type != 35 && type != 14)) return -1;
    q8k_job_t job = {weights, input, output, type, columns / 256};
    return bitnet_quant_run_rows(compute_rows, &job, rows);
}

int bitnet_matmul_q8k_impl(const void *weights, int type, int rows, int columns,
                     const float *input, float *output) {
    if (!weights || !input || !output || rows <= 0 || columns <= 0 || columns % 256 ||
        (type != 35 && type != 14)) return -1;
    int blocks = columns / 256;
    /* Per-call heap allocation showed up as decode overhead; the largest
     * supported activation (16384 floats) fits a fixed stack scratch. */
    bitnet_q8k_block_t stack_quant[BITNET_Q8K_MAX_BLOCKS];
    bitnet_q8k_block_t *quant = blocks <= BITNET_Q8K_MAX_BLOCKS ?
        stack_quant : (bitnet_q8k_block_t *)malloc((size_t)blocks * sizeof(*quant));
    if (!quant) return -1;
    int result = bitnet_quantize_q8k(input, columns, quant);
    if (!result) result = bitnet_matmul_q8k_prepared(weights, type, rows, columns, quant, output);
    if (quant != stack_quant) free(quant);
    return result;
}

int bitnet_quantize_q8k(const float *input, int count, bitnet_q8k_block_t *blocks) {
    if (!g_bitnet_dispatch) bitnet_dispatch_init();
    return g_bitnet_dispatch->quantize_q8k(input, count, blocks);
}

int bitnet_matmul_q8k(const void *weights, int type, int rows, int columns,
                     const float *input, float *output) {
    if (!g_bitnet_dispatch) bitnet_dispatch_init();
    return g_bitnet_dispatch->matmul_q8k(weights, type, rows, columns, input, output);
}
