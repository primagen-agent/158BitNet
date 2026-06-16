#include "bitnet_internal.h"
#include "quant_tq2_0.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

typedef struct bench_layer_scales {
    float *gate;
    float *up;
    float *down;
} bench_layer_scales_t;

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static void fill_vec(float *vec, int len) {
    for (int i = 0; i < len; ++i) {
        vec[i] = (float)((i % 29) - 14) / 14.0f;
    }
}

static float *build_scales_for_tensor(const gguf_file_t *file, const gguf_tensor_t *tensor,
                                      int out_dim, int in_dim) {
    const void *weight = gguf_get_tensor_ptr(file, tensor);
    size_t count = bitnet_tq2_0_scale_float_count(out_dim, in_dim);
    float *scales = NULL;
    if (weight == NULL || count == 0) return NULL;
    scales = (float *)malloc(count * sizeof(*scales));
    if (scales == NULL) return NULL;
    if (bitnet_tq2_0_build_scales(weight, out_dim, in_dim, scales, count) != 0) {
        free(scales);
        return NULL;
    }
    return scales;
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    enum {
        emb_dim = BITNET_TARGET_EMBEDDING_LENGTH,
        ffn_dim = BITNET_TARGET_FEED_FORWARD_LENGTH,
        block_count = BITNET_TARGET_BLOCK_COUNT,
        iterations = 4
    };
    gguf_file_t file;
    bitnet_tensor_cache_t cache = {0};
    bench_layer_scales_t *scales = NULL;
    float *hidden = NULL;
    float *ffn = NULL;
    float *gate = NULL;
    float *up = NULL;
    float *down = NULL;
    float *i8_gate = NULL;
    float *i8_down = NULL;
    float *neon_gate = NULL;
    float *neon_down = NULL;
    float *lut = NULL;
    int16_t *i16_lut = NULL;
    int8_t *qhidden = NULL;
    int8_t *qffn = NULL;
    size_t emb_lut_count = bitnet_tq2_0_lut_float_count(emb_dim);
    size_t ffn_lut_count = bitnet_tq2_0_lut_float_count(ffn_dim);
    struct timespec t0;
    struct timespec t1;
    double pair_sec = 0.0;
    double separate_sec = 0.0;
    double down_sec = 0.0;
    double i8_pair_sec = 0.0;
    double i8_down_sec = 0.0;
    double neon_gate_sec = 0.0;
    double neon_down_sec = 0.0;
    double i16_gate_sec = 0.0;
    double i16_down_sec = 0.0;
    double all_sec = 0.0;
    float hidden_scale = 0.0f;
    float ffn_scale = 0.0f;
    float checksum = 0.0f;
    float max_i8_diff = 0.0f;
    float max_neon_diff = 0.0f;

    if (gguf_open(model_path, &file) != 0) return 1;
    if (bitnet_build_tensor_cache(&file, block_count, &cache) != 0) return 2;

    scales = (bench_layer_scales_t *)calloc(block_count, sizeof(*scales));
    hidden = (float *)malloc((size_t)emb_dim * sizeof(*hidden));
    ffn = (float *)malloc((size_t)ffn_dim * sizeof(*ffn));
    gate = (float *)malloc((size_t)ffn_dim * sizeof(*gate));
    up = (float *)malloc((size_t)ffn_dim * sizeof(*up));
    down = (float *)malloc((size_t)emb_dim * sizeof(*down));
    i8_gate = (float *)malloc((size_t)ffn_dim * sizeof(*i8_gate));
    i8_down = (float *)malloc((size_t)emb_dim * sizeof(*i8_down));
    neon_gate = (float *)malloc((size_t)ffn_dim * sizeof(*neon_gate));
    neon_down = (float *)malloc((size_t)emb_dim * sizeof(*neon_down));
    lut = (float *)malloc((emb_lut_count > ffn_lut_count ? emb_lut_count : ffn_lut_count) * sizeof(*lut));
    i16_lut = (int16_t *)malloc((emb_lut_count > ffn_lut_count ? emb_lut_count : ffn_lut_count) * sizeof(*i16_lut));
    qhidden = (int8_t *)malloc((size_t)emb_dim * sizeof(*qhidden));
    qffn = (int8_t *)malloc((size_t)ffn_dim * sizeof(*qffn));
    if (scales == NULL || hidden == NULL || ffn == NULL || gate == NULL ||
        up == NULL || down == NULL || i8_gate == NULL || i8_down == NULL ||
        neon_gate == NULL || neon_down == NULL ||
        lut == NULL || i16_lut == NULL || qhidden == NULL || qffn == NULL) {
        return 3;
    }

    for (int i = 0; i < block_count; ++i) {
        scales[i].gate = build_scales_for_tensor(&file, cache.blocks[i].ffn_gate, ffn_dim, emb_dim);
        scales[i].up = build_scales_for_tensor(&file, cache.blocks[i].ffn_up, ffn_dim, emb_dim);
        scales[i].down = build_scales_for_tensor(&file, cache.blocks[i].ffn_down, emb_dim, ffn_dim);
        if (scales[i].gate == NULL || scales[i].up == NULL || scales[i].down == NULL) {
            return 4;
        }
    }

    fill_vec(hidden, emb_dim);
    fill_vec(ffn, ffn_dim);
    int32_t hidden_block_bsums[32], ffn_block_bsums[32];
    if (bitnet_tq2_0_quantize_vec_i8(hidden, emb_dim, qhidden, &hidden_scale, hidden_block_bsums) != 0) return 11;
    if (bitnet_tq2_0_quantize_vec_i8(ffn, ffn_dim, qffn, &ffn_scale, ffn_block_bsums) != 0) return 12;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int iter = 0; iter < iterations; ++iter) {
        if (bitnet_tq2_0_build_vec_lut(hidden, emb_dim, lut, emb_lut_count) != 0) return 5;
        if (bitnet_tq2_0_build_i16_lut(qhidden, emb_dim, i16_lut, emb_lut_count) != 0) return 15;
        for (int i = 0; i < block_count; ++i) {
            const void *w_gate = gguf_get_tensor_ptr(&file, cache.blocks[i].ffn_gate);
            const void *w_up = gguf_get_tensor_ptr(&file, cache.blocks[i].ffn_up);
            struct timespec p0;
            struct timespec p1;
            clock_gettime(CLOCK_MONOTONIC, &p0);
            if (bitnet_tq2_0_matmul_vector_lut_pair_scales(w_gate, scales[i].gate,
                                                           w_up, scales[i].up,
                                                           ffn_dim, emb_dim, lut, gate, up) != 0) {
                return 6;
            }
            clock_gettime(CLOCK_MONOTONIC, &p1);
            pair_sec += elapsed_sec(p0, p1);
            checksum += gate[i % ffn_dim] + up[(i * 17) % ffn_dim];

            clock_gettime(CLOCK_MONOTONIC, &p0);
            if (bitnet_tq2_0_matmul_vector_lut_scales(w_gate, scales[i].gate,
                                                      ffn_dim, emb_dim, lut, gate) != 0) {
                return 9;
            }
            if (bitnet_tq2_0_matmul_vector_lut_scales(w_up, scales[i].up,
                                                      ffn_dim, emb_dim, lut, up) != 0) {
                return 10;
            }
            clock_gettime(CLOCK_MONOTONIC, &p1);
            separate_sec += elapsed_sec(p0, p1);
            checksum += gate[(i * 3) % ffn_dim] + up[(i * 19) % ffn_dim];

            clock_gettime(CLOCK_MONOTONIC, &p0);
            if (bitnet_tq2_0_matmul_vector_i8_scales(w_gate, scales[i].gate,
                                                     ffn_dim, emb_dim,
                                                     qhidden, hidden_scale, i8_gate) != 0) {
                return 13;
            }
            clock_gettime(CLOCK_MONOTONIC, &p1);
            i8_pair_sec += elapsed_sec(p0, p1);
            {
                float diff = gate[i % ffn_dim] - i8_gate[i % ffn_dim];
                if (diff < 0.0f) diff = -diff;
                if (diff > max_i8_diff) max_i8_diff = diff;
            }
            checksum += i8_gate[(i * 5) % ffn_dim];

            clock_gettime(CLOCK_MONOTONIC, &p0);
            if (bitnet_tq2_0_matmul_vector_i8_neon_scales(w_gate, scales[i].gate,
                                                          ffn_dim, emb_dim,
                                                          qhidden, hidden_scale, hidden_block_bsums, neon_gate) != 0) {
                return 19;
            }
            clock_gettime(CLOCK_MONOTONIC, &p1);
            neon_gate_sec += elapsed_sec(p0, p1);
            {
                float diff = i8_gate[i % ffn_dim] - neon_gate[i % ffn_dim];
                if (diff < 0.0f) diff = -diff;
                if (diff > max_neon_diff) max_neon_diff = diff;
            }
            checksum += neon_gate[(i * 23) % ffn_dim];

            clock_gettime(CLOCK_MONOTONIC, &p0);
            if (bitnet_tq2_0_matmul_vector_i16_lut_scales(w_gate, scales[i].gate,
                                                          ffn_dim, emb_dim,
                                                          i16_lut, hidden_scale, i8_gate) != 0) {
                return 16;
            }
            clock_gettime(CLOCK_MONOTONIC, &p1);
            i16_gate_sec += elapsed_sec(p0, p1);
            checksum += i8_gate[(i * 11) % ffn_dim];
        }

        if (bitnet_tq2_0_build_vec_lut(ffn, ffn_dim, lut, ffn_lut_count) != 0) return 7;
        if (bitnet_tq2_0_build_i16_lut(qffn, ffn_dim, i16_lut, ffn_lut_count) != 0) return 17;
        for (int i = 0; i < block_count; ++i) {
            const void *w_down = gguf_get_tensor_ptr(&file, cache.blocks[i].ffn_down);
            struct timespec d0;
            struct timespec d1;
            clock_gettime(CLOCK_MONOTONIC, &d0);
            if (bitnet_tq2_0_matmul_vector_lut_scales(w_down, scales[i].down,
                                                      emb_dim, ffn_dim, lut, down) != 0) {
                return 8;
            }
            clock_gettime(CLOCK_MONOTONIC, &d1);
            down_sec += elapsed_sec(d0, d1);
            checksum += down[i % emb_dim];

            clock_gettime(CLOCK_MONOTONIC, &d0);
            if (bitnet_tq2_0_matmul_vector_i8_scales(w_down, scales[i].down,
                                                     emb_dim, ffn_dim,
                                                     qffn, ffn_scale, i8_down) != 0) {
                return 14;
            }
            clock_gettime(CLOCK_MONOTONIC, &d1);
            i8_down_sec += elapsed_sec(d0, d1);
            {
                float diff = down[i % emb_dim] - i8_down[i % emb_dim];
                if (diff < 0.0f) diff = -diff;
                if (diff > max_i8_diff) max_i8_diff = diff;
            }
            checksum += i8_down[(i * 7) % emb_dim];

            clock_gettime(CLOCK_MONOTONIC, &d0);
            if (bitnet_tq2_0_matmul_vector_i8_neon_scales(w_down, scales[i].down,
                                                          emb_dim, ffn_dim,
                                                          qffn, ffn_scale, ffn_block_bsums, neon_down) != 0) {
                return 20;
            }
            clock_gettime(CLOCK_MONOTONIC, &d1);
            neon_down_sec += elapsed_sec(d0, d1);
            {
                float diff = i8_down[i % emb_dim] - neon_down[i % emb_dim];
                if (diff < 0.0f) diff = -diff;
                if (diff > max_neon_diff) max_neon_diff = diff;
            }
            checksum += neon_down[(i * 29) % emb_dim];

            clock_gettime(CLOCK_MONOTONIC, &d0);
            if (bitnet_tq2_0_matmul_vector_i16_lut_scales(w_down, scales[i].down,
                                                          emb_dim, ffn_dim,
                                                          i16_lut, ffn_scale, i8_down) != 0) {
                return 18;
            }
            clock_gettime(CLOCK_MONOTONIC, &d1);
            i16_down_sec += elapsed_sec(d0, d1);
            checksum += i8_down[(i * 13) % emb_dim];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    all_sec = elapsed_sec(t0, t1);

    printf("iterations=%d layers=%d\n", iterations, block_count);
    printf("ffn_pair_total_sec=%.6f per_layer_ms=%.6f\n",
           pair_sec, pair_sec * 1000.0 / (double)(iterations * block_count));
    printf("ffn_separate_total_sec=%.6f per_layer_ms=%.6f\n",
           separate_sec, separate_sec * 1000.0 / (double)(iterations * block_count));
    printf("ffn_down_total_sec=%.6f per_layer_ms=%.6f\n",
           down_sec, down_sec * 1000.0 / (double)(iterations * block_count));
    printf("i8_gate_total_sec=%.6f per_layer_ms=%.6f\n",
           i8_pair_sec, i8_pair_sec * 1000.0 / (double)(iterations * block_count));
    printf("i8_down_total_sec=%.6f per_layer_ms=%.6f max_sample_diff=%.6f\n",
           i8_down_sec, i8_down_sec * 1000.0 / (double)(iterations * block_count),
           (double)max_i8_diff);
    printf("neon_i8_gate_total_sec=%.6f per_layer_ms=%.6f\n",
           neon_gate_sec, neon_gate_sec * 1000.0 / (double)(iterations * block_count));
    printf("neon_i8_down_total_sec=%.6f per_layer_ms=%.6f max_sample_diff=%.6f\n",
           neon_down_sec, neon_down_sec * 1000.0 / (double)(iterations * block_count),
           (double)max_neon_diff);
    printf("i16_lut_gate_total_sec=%.6f per_layer_ms=%.6f\n",
           i16_gate_sec, i16_gate_sec * 1000.0 / (double)(iterations * block_count));
    printf("i16_lut_down_total_sec=%.6f per_layer_ms=%.6f\n",
           i16_down_sec, i16_down_sec * 1000.0 / (double)(iterations * block_count));
    printf("total_sec=%.6f ffn_stack_per_token_ms=%.6f checksum=%.3f\n",
           all_sec, all_sec * 1000.0 / (double)iterations, checksum);

    for (int i = 0; i < block_count; ++i) {
        free(scales[i].gate);
        free(scales[i].up);
        free(scales[i].down);
    }
    free(scales);
    free(hidden);
    free(ffn);
    free(gate);
    free(up);
    free(down);
    free(i8_gate);
    free(i8_down);
    free(neon_gate);
    free(neon_down);
    free(lut);
    free(i16_lut);
    free(qhidden);
    free(qffn);
    bitnet_free_tensor_cache(&cache);
    gguf_close(&file);
    return 0;
}
