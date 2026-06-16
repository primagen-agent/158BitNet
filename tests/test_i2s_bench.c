#include "quant_tq2_0.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static uint16_t f32_to_fp16(float value) {
    uint32_t raw = 0;
    memcpy(&raw, &value, sizeof(raw));

    uint32_t sign = (raw >> 16) & 0x8000u;
    int32_t exp = (int32_t)((raw >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (raw >> 13) & 0x03ffu;

    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7bffu);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static void fill_weight(bitnet_tq2_0_block_t *weight, int out_dim, int blocks_per_row, int seed) {
    for (int row = 0; row < out_dim; ++row) {
        for (int k = 0; k < blocks_per_row; ++k) {
            bitnet_tq2_0_block_t *block =
                weight + (size_t)row * (size_t)blocks_per_row + (size_t)k;
            for (int i = 0; i < BITNET_TQ2_0_QS_SIZE; ++i) {
                block->qs[i] = (uint8_t)((row * 17 + k * 31 + i * 13 + seed) & 0xff);
            }
            block->d = f32_to_fp16(0.01f + (float)((row + k + seed) % 17) * 0.001f);
        }
    }
}

static void fill_vec(float *vec, int len) {
    for (int i = 0; i < len; ++i) {
        vec[i] = (float)((i % 29) - 14) / 14.0f;
    }
}

static int build_i2s_buffers(const bitnet_tq2_0_block_t *weight, int out_dim, int in_dim,
                             uint8_t **packed_out, float **scales_out, int32_t **packed_bsums_out) {
    const int out_dim_padded = (out_dim + 3) & ~3;
    const int n_groups = out_dim_padded / 4;
    const int sub_blocks_per_group = in_dim / 64;
    const size_t packed_size = bitnet_tq2_0_i2s_packed_size(out_dim, in_dim);
    const size_t aux_count = (size_t)n_groups * (size_t)sub_blocks_per_group * 4u;

    uint8_t *packed = (uint8_t *)calloc(packed_size, 1);
    float *scales = (float *)calloc(aux_count, sizeof(*scales));
    int32_t *packed_bsums = (int32_t *)calloc(aux_count, sizeof(*packed_bsums));

    if (packed == NULL || scales == NULL || packed_bsums == NULL) {
        free(packed);
        free(scales);
        free(packed_bsums);
        return -1;
    }

    if (bitnet_tq2_0_reorder_to_i2s(weight, out_dim, in_dim, packed, scales, packed_bsums) != 0) {
        free(packed);
        free(scales);
        free(packed_bsums);
        return -1;
    }

    *packed_out = packed;
    *scales_out = scales;
    *packed_bsums_out = packed_bsums;
    return 0;
}

static int run_single_case(const char *name, int out_dim, int in_dim, int iters) {
    const int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    const size_t weight_count = (size_t)out_dim * (size_t)blocks_per_row;
    bitnet_tq2_0_block_t *weight = (bitnet_tq2_0_block_t *)malloc(weight_count * sizeof(*weight));
    float *vec = (float *)malloc((size_t)in_dim * sizeof(*vec));
    int8_t *qvec = (int8_t *)malloc((size_t)in_dim * sizeof(*qvec));
    int32_t *bsums = (int32_t *)malloc((size_t)blocks_per_row * sizeof(*bsums));
    float *out = (float *)malloc((size_t)out_dim * sizeof(*out));
    uint8_t *packed = NULL;
    float *scales = NULL;
    int32_t *packed_bsums = NULL;
    float vec_scale = 0.0f;
    struct timespec t0, t1;

    if (weight == NULL || vec == NULL || qvec == NULL || bsums == NULL || out == NULL) {
        return 1;
    }

    fill_weight(weight, out_dim, blocks_per_row, 3);
    fill_vec(vec, in_dim);
    if (bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, &vec_scale, bsums) != 0) return 2;
    if (build_i2s_buffers(weight, out_dim, in_dim, &packed, &scales, &packed_bsums) != 0) return 3;

    if (bitnet_tq2_0_matmul_i2s_neon(packed, scales, bsums, out_dim, in_dim,
                                     qvec, vec_scale, out) != 0) return 4;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_i2s_neon(packed, scales, bsums, out_dim, in_dim,
                                         qvec, vec_scale, out) != 0) return 5;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double single_sec = elapsed_sec(t0, t1);

    if (bitnet_tq2_0_matmul_i2s_neon_parallel(packed, scales, bsums, out_dim, in_dim,
                                              qvec, vec_scale, out) != 0) return 6;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_i2s_neon_parallel(packed, scales, bsums, out_dim, in_dim,
                                                  qvec, vec_scale, out) != 0) return 7;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double parallel_sec = elapsed_sec(t0, t1);

    printf("%s out=%d in=%d iters=%d single_ms=%.4f parallel_ms=%.4f checksum=%.3f\n",
           name, out_dim, in_dim, iters,
           single_sec * 1000.0 / (double)iters,
           parallel_sec * 1000.0 / (double)iters,
           out[0] + out[out_dim / 2] + out[out_dim - 1]);

    free(weight);
    free(vec);
    free(qvec);
    free(bsums);
    free(out);
    free(packed);
    free(scales);
    free(packed_bsums);
    return 0;
}

static int run_pair_case(const char *name, int out_dim, int in_dim, int iters) {
    const int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    const size_t weight_count = (size_t)out_dim * (size_t)blocks_per_row;
    bitnet_tq2_0_block_t *weight_a = (bitnet_tq2_0_block_t *)malloc(weight_count * sizeof(*weight_a));
    bitnet_tq2_0_block_t *weight_b = (bitnet_tq2_0_block_t *)malloc(weight_count * sizeof(*weight_b));
    float *vec = (float *)malloc((size_t)in_dim * sizeof(*vec));
    int8_t *qvec = (int8_t *)malloc((size_t)in_dim * sizeof(*qvec));
    int32_t *bsums = (int32_t *)malloc((size_t)blocks_per_row * sizeof(*bsums));
    float *out_a = (float *)malloc((size_t)out_dim * sizeof(*out_a));
    float *out_b = (float *)malloc((size_t)out_dim * sizeof(*out_b));
    uint8_t *packed_a = NULL;
    uint8_t *packed_b = NULL;
    float *scales_a = NULL;
    float *scales_b = NULL;
    int32_t *packed_bsums_a = NULL;
    int32_t *packed_bsums_b = NULL;
    float vec_scale = 0.0f;
    struct timespec t0, t1;

    if (weight_a == NULL || weight_b == NULL || vec == NULL || qvec == NULL ||
        bsums == NULL || out_a == NULL || out_b == NULL) {
        return 1;
    }

    fill_weight(weight_a, out_dim, blocks_per_row, 5);
    fill_weight(weight_b, out_dim, blocks_per_row, 11);
    fill_vec(vec, in_dim);
    if (bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, &vec_scale, bsums) != 0) return 2;
    if (build_i2s_buffers(weight_a, out_dim, in_dim, &packed_a, &scales_a, &packed_bsums_a) != 0) return 3;
    if (build_i2s_buffers(weight_b, out_dim, in_dim, &packed_b, &scales_b, &packed_bsums_b) != 0) return 4;

    if (bitnet_tq2_0_matmul_i2s_neon_pair(packed_a, scales_a, packed_b, scales_b,
                                          bsums, out_dim, in_dim, qvec, vec_scale,
                                          out_a, out_b) != 0) return 5;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_i2s_neon_pair(packed_a, scales_a, packed_b, scales_b,
                                              bsums, out_dim, in_dim, qvec, vec_scale,
                                              out_a, out_b) != 0) return 6;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double single_sec = elapsed_sec(t0, t1);

    if (bitnet_tq2_0_matmul_i2s_neon_pair_parallel(packed_a, scales_a, packed_b, scales_b,
                                                   bsums, out_dim, in_dim, qvec, vec_scale,
                                                   out_a, out_b) != 0) return 7;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_i2s_neon_pair_parallel(packed_a, scales_a, packed_b, scales_b,
                                                       bsums, out_dim, in_dim, qvec, vec_scale,
                                                       out_a, out_b) != 0) return 8;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double parallel_sec = elapsed_sec(t0, t1);

    printf("%s_pair out=%d in=%d iters=%d single_ms=%.4f parallel_ms=%.4f checksum=%.3f\n",
           name, out_dim, in_dim, iters,
           single_sec * 1000.0 / (double)iters,
           parallel_sec * 1000.0 / (double)iters,
           out_a[0] + out_a[out_dim / 2] + out_b[out_dim - 1]);

    free(weight_a);
    free(weight_b);
    free(vec);
    free(qvec);
    free(bsums);
    free(out_a);
    free(out_b);
    free(packed_a);
    free(packed_b);
    free(scales_a);
    free(scales_b);
    free(packed_bsums_a);
    free(packed_bsums_b);
    return 0;
}

static int run_pair_split_case(const char *name, int out_dim, int in_dim, int iters) {
    const int blocks_per_row = in_dim / BITNET_TQ2_0_QK;
    const size_t weight_count = (size_t)out_dim * (size_t)blocks_per_row;
    bitnet_tq2_0_block_t *weight_a = (bitnet_tq2_0_block_t *)malloc(weight_count * sizeof(*weight_a));
    bitnet_tq2_0_block_t *weight_b = (bitnet_tq2_0_block_t *)malloc(weight_count * sizeof(*weight_b));
    float *vec = (float *)malloc((size_t)in_dim * sizeof(*vec));
    int8_t *qvec = (int8_t *)malloc((size_t)in_dim * sizeof(*qvec));
    int32_t *bsums = (int32_t *)malloc((size_t)blocks_per_row * sizeof(*bsums));
    float *out_a = (float *)malloc((size_t)out_dim * sizeof(*out_a));
    float *out_b = (float *)malloc((size_t)out_dim * sizeof(*out_b));
    uint8_t *packed_a = NULL;
    uint8_t *packed_b = NULL;
    float *scales_a = NULL;
    float *scales_b = NULL;
    int32_t *packed_bsums_a = NULL;
    int32_t *packed_bsums_b = NULL;
    float vec_scale = 0.0f;
    struct timespec t0, t1;

    if (weight_a == NULL || weight_b == NULL || vec == NULL || qvec == NULL ||
        bsums == NULL || out_a == NULL || out_b == NULL) {
        return 1;
    }

    fill_weight(weight_a, out_dim, blocks_per_row, 5);
    fill_weight(weight_b, out_dim, blocks_per_row, 11);
    fill_vec(vec, in_dim);
    if (bitnet_tq2_0_quantize_vec_i8(vec, in_dim, qvec, &vec_scale, bsums) != 0) return 2;
    if (build_i2s_buffers(weight_a, out_dim, in_dim, &packed_a, &scales_a, &packed_bsums_a) != 0) return 3;
    if (build_i2s_buffers(weight_b, out_dim, in_dim, &packed_b, &scales_b, &packed_bsums_b) != 0) return 4;

    if (bitnet_tq2_0_matmul_i2s_neon_pair_parallel(packed_a, scales_a, packed_b, scales_b,
                                                   bsums, out_dim, in_dim, qvec, vec_scale,
                                                   out_a, out_b) != 0) return 5;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_i2s_neon_pair_parallel(packed_a, scales_a, packed_b, scales_b,
                                                       bsums, out_dim, in_dim, qvec, vec_scale,
                                                       out_a, out_b) != 0) return 6;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double pair_sec = elapsed_sec(t0, t1);

    if (bitnet_tq2_0_matmul_i2s_neon_parallel(packed_a, scales_a, bsums,
                                              out_dim, in_dim, qvec, vec_scale, out_a) != 0) return 7;
    if (bitnet_tq2_0_matmul_i2s_neon_parallel(packed_b, scales_b, bsums,
                                              out_dim, in_dim, qvec, vec_scale, out_b) != 0) return 8;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (bitnet_tq2_0_matmul_i2s_neon_parallel(packed_a, scales_a, bsums,
                                                  out_dim, in_dim, qvec, vec_scale, out_a) != 0) return 9;
        if (bitnet_tq2_0_matmul_i2s_neon_parallel(packed_b, scales_b, bsums,
                                                  out_dim, in_dim, qvec, vec_scale, out_b) != 0) return 10;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double split_sec = elapsed_sec(t0, t1);

    printf("%s_pair_vs_split out=%d in=%d iters=%d pair_parallel_ms=%.4f split_parallel_ms=%.4f checksum=%.3f\n",
           name, out_dim, in_dim, iters,
           pair_sec * 1000.0 / (double)iters,
           split_sec * 1000.0 / (double)iters,
           out_a[0] + out_a[out_dim / 2] + out_b[out_dim - 1]);

    free(weight_a);
    free(weight_b);
    free(vec);
    free(qvec);
    free(bsums);
    free(out_a);
    free(out_b);
    free(packed_a);
    free(packed_b);
    free(scales_a);
    free(scales_b);
    free(packed_bsums_a);
    free(packed_bsums_b);
    return 0;
}

static int parse_positive_int(const char *s) {
    char *end = NULL;
    long v = 0;
    if (s == NULL || s[0] == '\0') return -1;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > 1000000) return -1;
    return (int)v;
}

int main(int argc, char **argv) {
    int rc = 0;

    if (argc == 5) {
        const char *mode = argv[1];
        int out_dim = parse_positive_int(argv[2]);
        int in_dim = parse_positive_int(argv[3]);
        int iters = parse_positive_int(argv[4]);
        if (out_dim <= 0 || in_dim <= 0 || iters <= 0) {
            fprintf(stderr, "usage: %s [single|pair|pair_split out_dim in_dim iters]\n", argv[0]);
            return 64;
        }
        if (strcmp(mode, "single") == 0) {
            return run_single_case("custom", out_dim, in_dim, iters);
        }
        if (strcmp(mode, "pair") == 0) {
            return run_pair_case("custom", out_dim, in_dim, iters);
        }
        if (strcmp(mode, "pair_split") == 0) {
            return run_pair_split_case("custom", out_dim, in_dim, iters);
        }
        fprintf(stderr, "usage: %s [single|pair|pair_split out_dim in_dim iters]\n", argv[0]);
        return 64;
    }

    if (argc != 1) {
        fprintf(stderr, "usage: %s [single|pair|pair_split out_dim in_dim iters]\n", argv[0]);
        return 64;
    }

    rc = run_single_case("q", 2048, 2048, 500);
    if (rc != 0) return rc;
    rc = run_single_case("kv", 256, 2048, 1000);
    if (rc != 0) return rc;
    rc = run_pair_case("gate_up", 6144, 2048, 100);
    if (rc != 0) return rc;
    rc = run_pair_split_case("gate_up", 6144, 2048, 100);
    if (rc != 0) return rc;
    rc = run_single_case("ffn_down", 2048, 6144, 200);
    if (rc != 0) return rc;

    return 0;
}
