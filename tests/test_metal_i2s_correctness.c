#include "bitnet_metal.h"
#include "quant_tq2_0.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16_bits(float value) {
    uint32_t f32_bits = 0;
    uint32_t sign = 0;
    int32_t exp = 0;
    uint32_t mant = 0;

    memcpy(&f32_bits, &value, sizeof(f32_bits));
    sign = (f32_bits >> 16) & 0x8000u;
    exp = (int32_t)((f32_bits >> 23) & 0xffu) - 127 + 15;
    mant = (f32_bits >> 13) & 0x3ffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7bffu);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

int main(void) {
    const int out_dim = 37;
    const int in_dim = BITNET_TQ2_0_QK * 8;
    const int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    const size_t weight_size =
        (size_t)out_dim * (size_t)blocks_per_row * BITNET_TQ2_0_BLOCK_SIZE;
    uint8_t *weight = (uint8_t *)calloc(weight_size, 1);
    float *vec = (float *)malloc((size_t)in_dim * sizeof(*vec));
    int8_t *qvec = (int8_t *)malloc((size_t)in_dim * sizeof(*qvec));
    int32_t *block_bsums = (int32_t *)malloc((size_t)blocks_per_row * sizeof(*block_bsums));
    float vec_scale = 0.0f;
    float *cpu_out = (float *)calloc((size_t)out_dim, sizeof(*cpu_out));
    float *gpu_out = (float *)calloc((size_t)out_dim, sizeof(*gpu_out));
    uint8_t *packed = NULL;
    float *packed_scales = NULL;
    int32_t *packed_bsums = NULL;
    bitnet_metal_i2s_context_t *ctx = NULL;
    bitnet_metal_i2s_tensor_t *tensor = NULL;
    int failures = 0;
    float max_diff = 0.0f;

    if (!bitnet_metal_available()) {
        printf("SKIP: Metal device is not available in this process\n");
        return 0;
    }

    if (weight == NULL || vec == NULL || qvec == NULL || block_bsums == NULL ||
        cpu_out == NULL || gpu_out == NULL) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    srand(7);
    for (int row = 0; row < out_dim; ++row) {
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            bitnet_tq2_0_block_t *block = (bitnet_tq2_0_block_t *)
                (weight + ((size_t)row * (size_t)blocks_per_row + (size_t)blk) *
                              BITNET_TQ2_0_BLOCK_SIZE);
            for (int i = 0; i < BITNET_TQ2_0_QS_SIZE; ++i) {
                uint8_t packed_codes = 0;
                for (int shift = 0; shift < 8; shift += 2) {
                    packed_codes |= (uint8_t)((rand() % 3) << shift);
                }
                block->qs[i] = packed_codes;
            }
            block->d = f32_to_f16_bits(0.01f + (float)(rand() % 31) * 0.001f);
        }
    }

    for (int i = 0; i < in_dim; ++i) {
        vec[i] = (float)((rand() % 2001) - 1000) / 1000.0f;
    }
    if (bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, &vec_scale, block_bsums) != 0) {
        fprintf(stderr, "quantize failed\n");
        return 1;
    }

    {
        const size_t packed_size = bitnet_tq2_0_i2s_packed_size(out_dim, in_dim);
        const int out_dim_padded = (out_dim + 3) & ~3;
        const int n_groups = out_dim_padded / 4;
        const int sub_blocks_per_group = in_dim / 64;
        const size_t scales_count = (size_t)n_groups * (size_t)blocks_per_row * 4u;
        const size_t bsums_count = (size_t)n_groups * (size_t)sub_blocks_per_group * 4u;

        packed = (uint8_t *)calloc(packed_size, 1);
        packed_scales = (float *)calloc(scales_count, sizeof(*packed_scales));
        packed_bsums = (int32_t *)calloc(bsums_count, sizeof(*packed_bsums));
        if (packed == NULL || packed_scales == NULL || packed_bsums == NULL) {
            fprintf(stderr, "i2s alloc failed\n");
            return 1;
        }
    }

    if (bitnet_tq2_0_reorder_to_i2s(weight, out_dim, in_dim,
                                   packed, packed_scales, packed_bsums) != 0 ||
        bitnet_tq2_0_matmul_i2s_neon(packed, packed_scales, NULL,
                                     out_dim, in_dim, qvec, vec_scale, cpu_out) != 0) {
        fprintf(stderr, "cpu i2s failed\n");
        return 1;
    }

    if (bitnet_metal_i2s_create(&ctx) != 0) {
        fprintf(stderr, "metal i2s create failed\n");
        return 1;
    }
    if (bitnet_metal_i2s_tensor_create(ctx, packed, packed_scales,
                                       out_dim, in_dim, &tensor) != 0) {
        fprintf(stderr, "metal i2s tensor create failed\n");
        return 1;
    }
    if (bitnet_metal_i2s_compute(ctx, tensor, qvec, vec_scale, gpu_out) != 0) {
        fprintf(stderr, "metal i2s compute failed\n");
        return 1;
    }

    for (int i = 0; i < out_dim; ++i) {
        const float diff = fabsf(cpu_out[i] - gpu_out[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) {
            printf("row %d cpu=%.6f gpu=%.6f diff=%.6f\n",
                   i, cpu_out[i], gpu_out[i], diff);
            ++failures;
        }
    }

    printf("metal_i2s_max_diff=%.6f\n", max_diff);
    bitnet_metal_i2s_tensor_free(tensor);
    bitnet_metal_i2s_free(ctx);
    free(packed_bsums);
    free(packed_scales);
    free(packed);
    free(gpu_out);
    free(cpu_out);
    free(block_bsums);
    free(qvec);
    free(vec);
    free(weight);
    return failures == 0 ? 0 : 1;
}
