#include "bitnet_internal.h"
#include "gguf.h"
#include "quant_q6k.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static void fill_qvec(int8_t *qvec, int len) {
    for (int i = 0; i < len; ++i) {
        qvec[i] = (int8_t)((i % 127) - 63);
    }
}

static int run_q6k_original(const uint8_t *output_data, int vocab_size,
                            int blocks_per_row, size_t row_size,
                            const int8_t *qvec, float scale, float *logits) {
    for (int row = 0; row < vocab_size; ++row) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            const bitnet_q6k_block_t *block =
                (const bitnet_q6k_block_t *)(output_data +
                    (size_t)row * row_size + (size_t)b * BITNET_Q6K_BLOCK_SIZE);
            float block_dot = 0.0f;
            if (bitnet_q6k_dot_product_i8_neon(block,
                                               qvec + (size_t)b * BITNET_Q6K_QK,
                                               scale, BITNET_Q6K_QK, &block_dot) != 0) {
                return -1;
            }
            sum += block_dot;
        }
        logits[row] = sum;
    }
    return 0;
}

static int run_q6k_expanded(const int8_t *q8, const float *scales, int vocab_size,
                            int emb_dim, int blocks_per_row,
                            const int8_t *qvec, float scale, float *logits) {
    for (int row = 0; row < vocab_size; ++row) {
        if (bitnet_q6k_dot_product_q8_neon(q8 + (size_t)row * (size_t)emb_dim,
                                           scales + (size_t)row * (size_t)blocks_per_row * 16u,
                                           blocks_per_row, qvec, scale, &logits[row]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int run_q6k_compact(const int8_t *q8, const int8_t *scales, const float *d,
                           int vocab_size, int emb_dim, int blocks_per_row,
                           const int8_t *qvec, float scale, float *logits) {
    const int scale_stride = blocks_per_row * 16;
    int row = 0;

    for (; row + 7 < vocab_size; row += 8) {
        if (bitnet_q6k_dot_product_q8_compact_8_neon(
                q8 + (size_t)row * (size_t)emb_dim,
                scales + (size_t)row * (size_t)scale_stride,
                d + (size_t)row * (size_t)blocks_per_row,
                emb_dim, scale_stride, blocks_per_row,
                blocks_per_row, qvec, scale, logits + row) != 0) {
            return -1;
        }
    }
    for (; row + 3 < vocab_size; row += 4) {
        if (bitnet_q6k_dot_product_q8_compact_4_neon(
                q8 + (size_t)row * (size_t)emb_dim,
                scales + (size_t)row * (size_t)scale_stride,
                d + (size_t)row * (size_t)blocks_per_row,
                emb_dim, scale_stride, blocks_per_row,
                blocks_per_row, qvec, scale, logits + row) != 0) {
            return -1;
        }
    }
    return row == vocab_size ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    enum { iters = 3 };
    int emb_dim = 0;
    int vocab_size = 0;
    int blocks_per_row = 0;
    size_t row_size = 0;
    gguf_file_t file;
    const gguf_tensor_t *output = NULL;
    const uint8_t *output_data = NULL;
    int8_t *qvec = NULL;
    float *logits = NULL;
    int8_t *expanded_q8 = NULL;
    float *expanded_scales = NULL;
    int8_t *compact_q8 = NULL;
    int8_t *compact_scales = NULL;
    float *compact_d = NULL;
    struct timespec t0, t1;
    double original_sec = 0.0;
    double expanded_sec = 0.0;
    double compact_sec = 0.0;
    double expand_sec = 0.0;
    double compact_expand_sec = 0.0;

    if (gguf_open(model_path, &file) != 0) return 1;
    output = gguf_find_tensor(&file, "output.weight");
    if (output == NULL || output->type != BITNET_TARGET_TENSOR_TYPE_Q6_K ||
        output->n_dims != 2 || output->dims[0] > INT32_MAX || output->dims[1] > INT32_MAX ||
        output->dims[0] == 0 || output->dims[0] % BITNET_Q6K_QK != 0) {
        gguf_close(&file);
        return 2;
    }
    emb_dim = (int)output->dims[0];
    vocab_size = (int)output->dims[1];
    blocks_per_row = emb_dim / BITNET_Q6K_QK;
    row_size = (size_t)blocks_per_row * BITNET_Q6K_BLOCK_SIZE;
    output_data = (const uint8_t *)gguf_get_tensor_ptr(&file, output);
    if (output_data == NULL) return 2;

    qvec = (int8_t *)malloc((size_t)emb_dim * sizeof(*qvec));
    logits = (float *)malloc((size_t)vocab_size * sizeof(*logits));
    expanded_q8 = (int8_t *)malloc((size_t)vocab_size * (size_t)emb_dim * sizeof(*expanded_q8));
    expanded_scales = (float *)malloc((size_t)vocab_size * (size_t)blocks_per_row * 16u * sizeof(*expanded_scales));
    compact_q8 = (int8_t *)malloc((size_t)vocab_size * (size_t)emb_dim * sizeof(*compact_q8));
    compact_scales = (int8_t *)malloc((size_t)vocab_size * (size_t)blocks_per_row * 16u * sizeof(*compact_scales));
    compact_d = (float *)malloc((size_t)vocab_size * (size_t)blocks_per_row * sizeof(*compact_d));
    if (qvec == NULL || logits == NULL || expanded_q8 == NULL || expanded_scales == NULL ||
        compact_q8 == NULL || compact_scales == NULL || compact_d == NULL) {
        return 3;
    }
    fill_qvec(qvec, emb_dim);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (bitnet_q6k_expand_to_q8(output_data, vocab_size, emb_dim,
                                expanded_q8, expanded_scales) != 0) {
        return 4;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    expand_sec = elapsed_sec(t0, t1);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (bitnet_q6k_expand_to_q8_compact(output_data, vocab_size, emb_dim,
                                        compact_q8, compact_scales, compact_d) != 0) {
        return 4;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    compact_expand_sec = elapsed_sec(t0, t1);

    if (run_q6k_original(output_data, vocab_size, blocks_per_row, row_size,
                         qvec, 1.0f / 63.0f, logits) != 0) {
        return 5;
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (run_q6k_original(output_data, vocab_size, blocks_per_row, row_size,
                             qvec, 1.0f / 63.0f, logits) != 0) {
            return 6;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    original_sec = elapsed_sec(t0, t1);

    if (run_q6k_expanded(expanded_q8, expanded_scales, vocab_size, emb_dim,
                         blocks_per_row, qvec, 1.0f / 63.0f, logits) != 0) {
        return 7;
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (run_q6k_expanded(expanded_q8, expanded_scales, vocab_size, emb_dim,
                             blocks_per_row, qvec, 1.0f / 63.0f, logits) != 0) {
            return 8;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    expanded_sec = elapsed_sec(t0, t1);

    if (run_q6k_compact(compact_q8, compact_scales, compact_d, vocab_size, emb_dim,
                        blocks_per_row, qvec, 1.0f / 63.0f, logits) != 0) {
        return 9;
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        if (run_q6k_compact(compact_q8, compact_scales, compact_d, vocab_size, emb_dim,
                            blocks_per_row, qvec, 1.0f / 63.0f, logits) != 0) {
            return 10;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    compact_sec = elapsed_sec(t0, t1);

    printf("expand_sec=%.6f\n", expand_sec);
    printf("compact_expand_sec=%.6f\n", compact_expand_sec);
    printf("original_ms=%.6f\n", original_sec * 1000.0 / (double)iters);
    printf("expanded_ms=%.6f\n", expanded_sec * 1000.0 / (double)iters);
    printf("compact_ms=%.6f\n", compact_sec * 1000.0 / (double)iters);
    printf("checksum=%.6f\n", logits[0] + logits[vocab_size / 2] + logits[vocab_size - 1]);

    free(qvec);
    free(logits);
    free(expanded_q8);
    free(expanded_scales);
    free(compact_q8);
    free(compact_scales);
    free(compact_d);
    gguf_close(&file);
    return 0;
}
