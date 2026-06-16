#include "quant_tq2_0.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static void fill_weight(bitnet_tq2_0_block_t *weight, int out_dim, int blocks_per_row) {
    for (int row = 0; row < out_dim; ++row) {
        for (int k = 0; k < blocks_per_row; ++k) {
            bitnet_tq2_0_block_t *block = weight + (size_t)row * (size_t)blocks_per_row + (size_t)k;
            for (int i = 0; i < BITNET_TQ2_0_QS_SIZE; ++i) {
                block->qs[i] = (uint8_t)((row * 17 + k * 31 + i * 13) & 0xff);
            }
            block->d = 0x3c00;
        }
    }
}

static void fill_vec(float *vec, int len) {
    for (int i = 0; i < len; ++i) {
        vec[i] = (float)((i % 23) - 11) / 11.0f;
    }
}

static int run_case(const char *name, int out_dim, int in_dim, int iters) {
    int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    size_t weight_count = (size_t)out_dim * (size_t)blocks_per_row;
    size_t lut_count = bitnet_tq2_0_lut_float_count(in_dim);
    size_t scale_count = bitnet_tq2_0_scale_float_count(out_dim, in_dim);
    bitnet_tq2_0_block_t *weight = NULL;
    float *scales = NULL;
    float *vec = NULL;
    float *lut = NULL;
    float *out = NULL;
    struct timespec t0;
    struct timespec t1;
    double sec = 0.0;

    weight = (bitnet_tq2_0_block_t *)malloc(weight_count * sizeof(*weight));
    scales = (float *)malloc(scale_count * sizeof(*scales));
    vec = (float *)malloc((size_t)in_dim * sizeof(*vec));
    lut = (float *)malloc(lut_count * sizeof(*lut));
    out = (float *)malloc((size_t)out_dim * sizeof(*out));
    if (weight == NULL || scales == NULL || vec == NULL || lut == NULL || out == NULL) {
        free(weight);
        free(scales);
        free(vec);
        free(lut);
        free(out);
        return 1;
    }

    fill_weight(weight, out_dim, blocks_per_row);
    fill_vec(vec, in_dim);
    if (bitnet_tq2_0_build_scales(weight, out_dim, in_dim, scales, scale_count) != 0) return 2;
    if (bitnet_tq2_0_build_vec_lut(vec, in_dim, lut, lut_count) != 0) return 3;
    if (bitnet_tq2_0_matmul_vector_lut_scales(weight, scales, out_dim, in_dim, lut, out) != 0) return 4;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_vector_lut_scales(weight, scales, out_dim, in_dim, lut, out) != 0) return 5;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    sec = elapsed_sec(t0, t1);
    printf("%s out=%d in=%d iters=%d total_sec=%.6f per_call_ms=%.6f calls_per_s=%.2f checksum=%.3f\n",
           name, out_dim, in_dim, iters, sec, sec * 1000.0 / (double)iters,
           (double)iters / sec, out[0] + out[out_dim / 2] + out[out_dim - 1]);

    free(weight);
    free(scales);
    free(vec);
    free(lut);
    free(out);
    return 0;
}

int main(void) {
    int rc = 0;
    rc = run_case("attn", 2048, 2048, 200);
    if (rc != 0) return rc;
    rc = run_case("ffn_up", 6144, 2048, 100);
    if (rc != 0) return rc;
    rc = run_case("ffn_down", 2048, 6144, 100);
    if (rc != 0) return rc;
    return 0;
}
