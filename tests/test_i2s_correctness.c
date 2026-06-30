/* I2_S kernel correctness test: compare results with existing VTBL kernel */
#include "quant_tq2_0.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int almost_equal(float a, float b, float tol) {
    float diff = a - b;
    if (diff < 0.0f) diff = -diff;
    return diff < tol;
}

int main(void) {
    /* Create test weights: 11 rows x 2048 cols (8-row path plus padded tail) */
    const int out_dim = 11;
    const int in_dim = BITNET_TQ2_0_QK * 8;  /* 2048 */
    const int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    const size_t weight_size = (size_t)out_dim * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;

    uint8_t *weight = (uint8_t *)calloc(weight_size, 1);
    if (weight == NULL) { fprintf(stderr, "alloc weight failed\n"); return 1; }

    /* Fill weights with varied patterns */
    srand(42);
    for (int row = 0; row < out_dim; ++row) {
        for (int k = 0; k < blocks_per_row; ++k) {
            size_t off = (size_t)row * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE
                       + (size_t)k * BITNET_TQ2_0_BLOCK_SIZE;
            bitnet_tq2_0_block_t *blk = (bitnet_tq2_0_block_t *)(weight + off);
            /* Fill qs with random codes */
            for (int i = 0; i < BITNET_TQ2_0_QS_SIZE; ++i) {
                blk->qs[i] = (uint8_t)(rand() & 0xFF);
            }
            /* Set scale to a reasonable value */
            float scale_val = 0.01f + (float)(rand() % 100) * 0.001f;
            /* Convert to fp16 */
            uint16_t fp16_bits;
            if (scale_val == 0.0f) {
                fp16_bits = 0;
            } else {
                uint32_t f32_bits;
                memcpy(&f32_bits, &scale_val, sizeof(f32_bits));
                uint32_t sign = (f32_bits >> 16) & 0x8000u;
                int32_t exp = (int32_t)((f32_bits >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (f32_bits >> 13) & 0x3FFu;
                if (exp <= 0) { fp16_bits = 0; }
                else if (exp >= 31) { fp16_bits = sign | 0x7BFFu; }
                else { fp16_bits = (uint16_t)(sign | ((uint32_t)exp << 10) | mant); }
            }
            blk->d = fp16_bits;
        }
    }

    /* Build scales array */
    size_t scale_count = bitnet_tq2_0_scale_float_count(out_dim, in_dim);
    float *scales = (float *)malloc(scale_count * sizeof(float));
    if (scales == NULL) { fprintf(stderr, "alloc scales failed\n"); return 1; }
    if (bitnet_tq2_0_build_scales(weight, out_dim, in_dim, scales, scale_count) != 0) {
        fprintf(stderr, "build_scales failed\n"); return 1;
    }

    /* Create test vector and quantize */
    float *vec = (float *)malloc((size_t)in_dim * sizeof(float));
    if (vec == NULL) { fprintf(stderr, "alloc vec failed\n"); return 1; }
    for (int i = 0; i < in_dim; ++i) {
        vec[i] = (float)(rand() % 200 - 100) / 100.0f;
    }

    int8_t *qvec = (int8_t *)malloc((size_t)in_dim * sizeof(int8_t));
    float vec_scale = 0.0f;
    int32_t *block_bsums = (int32_t *)malloc((size_t)blocks_per_row * sizeof(int32_t));
    if (qvec == NULL || block_bsums == NULL) { fprintf(stderr, "alloc qvec failed\n"); return 1; }
    if (bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, &vec_scale, block_bsums) != 0) {
        fprintf(stderr, "quantize_vec_i8 failed\n"); return 1;
    }

    /* Compute reference result using existing VTBL kernel */
    float *ref_out = (float *)calloc(out_dim, sizeof(float));
    if (ref_out == NULL) { fprintf(stderr, "alloc ref_out failed\n"); return 1; }
    if (bitnet_tq2_0_matmul_vector_i8_neon_scales(weight, scales, out_dim, in_dim,
                                                    qvec, vec_scale, block_bsums, ref_out) != 0) {
        fprintf(stderr, "reference matmul failed\n"); return 1;
    }

    /* Compute I2S result */
    size_t packed_size = bitnet_tq2_0_i2s_packed_size(out_dim, in_dim);
    if (packed_size == 0) { fprintf(stderr, "i2s_packed_size returned 0\n"); return 1; }
    printf("I2S packed size: %zu bytes\n", packed_size);

    int sub_blocks_per_group = in_dim / 64; /* QK_I2S = 64 */
    int out_dim_padded = (out_dim + 3) & ~3;
    int n_groups = out_dim_padded / 4;
    size_t scales_size = (size_t)n_groups * (size_t)sub_blocks_per_group * 4 * sizeof(float);
    size_t bsums_size = (size_t)n_groups * (size_t)sub_blocks_per_group * 4 * sizeof(int32_t);

    uint8_t *packed = (uint8_t *)calloc(packed_size, 1);
    float *packed_scales = (float *)calloc(scales_size / sizeof(float), sizeof(float));
    int32_t *packed_bsums = (int32_t *)calloc(bsums_size / sizeof(int32_t), sizeof(int32_t));
    float *i2s_out = (float *)calloc(out_dim, sizeof(float));

    if (packed == NULL || packed_scales == NULL || packed_bsums == NULL || i2s_out == NULL) {
        fprintf(stderr, "alloc i2s buffers failed\n"); return 1;
    }

    if (bitnet_tq2_0_reorder_to_i2s(weight, out_dim, in_dim,
                                      packed, packed_scales, packed_bsums) != 0) {
        fprintf(stderr, "reorder_to_i2s failed\n"); return 1;
    }

    if (bitnet_tq2_0_matmul_i2s_neon(packed, packed_scales, block_bsums,
                                       out_dim, in_dim, qvec, vec_scale, i2s_out) != 0) {
        fprintf(stderr, "matmul_i2s_neon failed\n"); return 1;
    }

    float *i2s_signed_out = (float *)calloc(out_dim, sizeof(float));
    if (i2s_signed_out == NULL) {
        fprintf(stderr, "alloc i2s_signed_out failed\n");
        return 1;
    }
    if (bitnet_tq2_0_matmul_i2s_neon(packed, packed_scales, NULL,
                                      out_dim, in_dim, qvec, vec_scale,
                                      i2s_signed_out) != 0) {
        fprintf(stderr, "matmul_i2s_neon signed path failed\n");
        return 1;
    }

    /* Compare results */
    int failures = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < out_dim; ++i) {
        float diff = fabsf(ref_out[i] - i2s_out[i]);
        if (diff > max_diff) max_diff = diff;
        if (!almost_equal(ref_out[i], i2s_out[i], 0.01f)) {
            printf("Row %d: ref=%.6f i2s=%.6f diff=%.6f\n", i, ref_out[i], i2s_out[i], diff);
            failures++;
        }
        {
            float signed_diff = fabsf(i2s_out[i] - i2s_signed_out[i]);
            if (!almost_equal(i2s_out[i], i2s_signed_out[i], 0.01f)) {
                printf("Row %d: i2s_bsum=%.6f i2s_signed=%.6f diff=%.6f\n",
                       i, i2s_out[i], i2s_signed_out[i], signed_diff);
                failures++;
            }
        }
    }

    printf("Max diff: %.6f\n", max_diff);
    if (failures > 0) {
        printf("FAIL: %d rows mismatched\n", failures);
    } else {
        printf("PASS: all %d rows match\n", out_dim);
    }

    free(weight);
    free(scales);
    free(vec);
    free(qvec);
    free(block_bsums);
    free(ref_out);
    free(packed);
    free(packed_scales);
    free(packed_bsums);
    free(i2s_out);
    free(i2s_signed_out);

    return failures > 0 ? 1 : 0;
}
