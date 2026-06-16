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

/* Bandwidth analysis for i8 NEON matmul */
static int run_i8_case(const char *name, int out_dim, int in_dim, int iters) {
    int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    size_t weight_count = (size_t)out_dim * (size_t)blocks_per_row;
    size_t scale_count = bitnet_tq2_0_scale_float_count(out_dim, in_dim);
    size_t weight_bytes = weight_count * BITNET_TQ2_0_BLOCK_SIZE;
    size_t qvec_bytes = (size_t)in_dim * sizeof(int8_t);
    size_t scales_bytes = scale_count * sizeof(float);
    /* Total data read per call: weight + qvec + scales */
    size_t total_read = weight_bytes + qvec_bytes + scales_bytes;

    bitnet_tq2_0_block_t *weight = NULL;
    float *scales = NULL;
    float *vec = NULL;
    int8_t *qvec = NULL;
    float *out = NULL;
    int32_t *block_bsums = NULL;
    float vec_scale = 0.0f;
    struct timespec t0, t1;
    double sec = 0.0;

    weight = (bitnet_tq2_0_block_t *)malloc(weight_bytes);
    scales = (float *)malloc(scale_count * sizeof(*scales));
    vec = (float *)malloc((size_t)in_dim * sizeof(*vec));
    qvec = (int8_t *)malloc((size_t)in_dim * sizeof(*qvec));
    out = (float *)malloc((size_t)out_dim * sizeof(*out));
    block_bsums = (int32_t *)malloc(((size_t)in_dim / BITNET_TQ2_0_QK) * sizeof(int32_t));
    if (!weight || !scales || !vec || !qvec || !out || !block_bsums) {
        free(weight); free(scales); free(vec); free(qvec); free(out); free(block_bsums);
        return 1;
    }

    fill_weight(weight, out_dim, blocks_per_row);
    fill_vec(vec, in_dim);
    if (bitnet_tq2_0_build_scales(weight, out_dim, in_dim, scales, scale_count) != 0) return 2;
    if (bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, &vec_scale, block_bsums) != 0) return 3;

    /* Warmup */
    if (bitnet_tq2_0_matmul_vector_i8_neon_scales(weight, scales, out_dim, in_dim,
                                                    qvec, vec_scale, block_bsums, out) != 0) return 4;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_vector_i8_neon_scales(weight, scales, out_dim, in_dim,
                                                        qvec, vec_scale, block_bsums, out) != 0) return 5;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    sec = elapsed_sec(t0, t1);
    double per_call_ms = sec * 1000.0 / (double)iters;
    double gb_read = (double)total_read / 1e9;
    double bandwidth_gbs = gb_read / (sec / (double)iters);
    double m1_bandwidth = 68.0; /* GB/s */
    double bw_util = bandwidth_gbs / m1_bandwidth * 100.0;

    printf("%s: out=%d in=%d weight=%.2fMB qvec=%.2fKB scales=%.2fKB total=%.2fMB\n",
           name, out_dim, in_dim,
           (double)weight_bytes / (1024.0 * 1024.0),
           (double)qvec_bytes / 1024.0,
           (double)scales_bytes / 1024.0,
           (double)total_read / (1024.0 * 1024.0));
    printf("  per_call=%.3fms bandwidth=%.1fGB/s utilization=%.1f%% checksum=%.3f\n",
           per_call_ms, bandwidth_gbs, bw_util,
           out[0] + out[out_dim / 2] + out[out_dim - 1]);

    free(weight); free(scales); free(vec); free(qvec); free(out); free(block_bsums);
    return 0;
}

/* Pair kernel bandwidth analysis */
static int run_i8_pair_case(const char *name, int out_dim, int in_dim, int iters) {
    int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    size_t weight_count = (size_t)out_dim * (size_t)blocks_per_row;
    size_t scale_count = bitnet_tq2_0_scale_float_count(out_dim, in_dim);
    size_t weight_bytes = weight_count * BITNET_TQ2_0_BLOCK_SIZE;
    size_t qvec_bytes = (size_t)in_dim * sizeof(int8_t);
    size_t scales_bytes = scale_count * sizeof(float);
    /* Pair: 2 weight matrices + qvec + 2 scales */
    size_t total_read = 2 * weight_bytes + qvec_bytes + 2 * scales_bytes;

    bitnet_tq2_0_block_t *weight_a = NULL;
    bitnet_tq2_0_block_t *weight_b = NULL;
    float *scales_a = NULL;
    float *scales_b = NULL;
    float *vec = NULL;
    int8_t *qvec = NULL;
    float *out_a = NULL;
    float *out_b = NULL;
    int32_t *block_bsums = NULL;
    float vec_scale = 0.0f;
    struct timespec t0, t1;

    weight_a = (bitnet_tq2_0_block_t *)malloc(weight_bytes);
    weight_b = (bitnet_tq2_0_block_t *)malloc(weight_bytes);
    scales_a = (float *)malloc(scale_count * sizeof(*scales_a));
    scales_b = (float *)malloc(scale_count * sizeof(*scales_b));
    vec = (float *)malloc((size_t)in_dim * sizeof(*vec));
    qvec = (int8_t *)malloc((size_t)in_dim * sizeof(*qvec));
    out_a = (float *)malloc((size_t)out_dim * sizeof(*out_a));
    out_b = (float *)malloc((size_t)out_dim * sizeof(*out_b));
    block_bsums = (int32_t *)malloc(((size_t)in_dim / BITNET_TQ2_0_QK) * sizeof(int32_t));

    if (!weight_a || !weight_b || !scales_a || !scales_b || !vec || !qvec || !out_a || !out_b || !block_bsums) {
        free(weight_a); free(weight_b); free(scales_a); free(scales_b);
        free(vec); free(qvec); free(out_a); free(out_b); free(block_bsums);
        return 1;
    }

    fill_weight(weight_a, out_dim, blocks_per_row);
    fill_weight(weight_b, out_dim, blocks_per_row);
    fill_vec(vec, in_dim);
    if (bitnet_tq2_0_build_scales(weight_a, out_dim, in_dim, scales_a, scale_count) != 0) return 2;
    if (bitnet_tq2_0_build_scales(weight_b, out_dim, in_dim, scales_b, scale_count) != 0) return 2;
    if (bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, &vec_scale, block_bsums) != 0) return 3;

    /* Warmup */
    if (bitnet_tq2_0_matmul_vector_i8_neon_pair_scales(weight_a, scales_a, weight_b, scales_b,
                                                        out_dim, in_dim, qvec, vec_scale,
                                                        block_bsums, out_a, out_b) != 0) return 4;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_vector_i8_neon_pair_scales(weight_a, scales_a, weight_b, scales_b,
                                                            out_dim, in_dim, qvec, vec_scale,
                                                            block_bsums, out_a, out_b) != 0) return 5;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double sec = elapsed_sec(t0, t1);
    double per_call_ms = sec * 1000.0 / (double)iters;
    double gb_read = (double)total_read / 1e9;
    double bandwidth_gbs = gb_read / (sec / (double)iters);
    double bw_util = bandwidth_gbs / 68.0 * 100.0;

    printf("%s_pair: out=%d in=%d 2xweight=%.2fMB total=%.2fMB\n",
           name, out_dim, in_dim,
           2.0 * (double)weight_bytes / (1024.0 * 1024.0),
           (double)total_read / (1024.0 * 1024.0));
    printf("  per_call=%.3fms bandwidth=%.1fGB/s utilization=%.1f%%\n",
           per_call_ms, bandwidth_gbs, bw_util);

    free(weight_a); free(weight_b); free(scales_a); free(scales_b);
    free(vec); free(qvec); free(out_a); free(out_b); free(block_bsums);
    return 0;
}

int main(void) {
    int rc = 0;

    printf("=== Single matrix i8 NEON bandwidth ===\n");
    rc = run_i8_case("qkv", 2048, 2048, 200);
    if (rc != 0) return rc;
    rc = run_i8_case("attn_out", 2048, 2048, 200);
    if (rc != 0) return rc;
    rc = run_i8_case("gate_up", 6144, 2048, 100);
    if (rc != 0) return rc;
    rc = run_i8_case("ffn_down", 2048, 6144, 100);
    if (rc != 0) return rc;

    printf("\n=== Pair matrix i8 NEON bandwidth ===\n");
    rc = run_i8_pair_case("gate_up", 6144, 2048, 50);
    if (rc != 0) return rc;

    return 0;
}
