#ifndef QUANT_Q6K_H
#define QUANT_Q6K_H

#include <stddef.h>
#include <stdint.h>

#define BITNET_Q6K_QK 256
#define BITNET_Q6K_BLOCK_SIZE 210
#define BITNET_Q6K_QL_SIZE (BITNET_Q6K_QK / 2)
#define BITNET_Q6K_QH_SIZE (BITNET_Q6K_QK / 4)
#define BITNET_Q6K_SCALES_SIZE (BITNET_Q6K_QK / 16)

typedef struct bitnet_q6k_block {
    uint8_t ql[BITNET_Q6K_QL_SIZE];
    uint8_t qh[BITNET_Q6K_QH_SIZE];
    int8_t scales[BITNET_Q6K_SCALES_SIZE];
    uint16_t d;
} bitnet_q6k_block_t;

int bitnet_q6k_dequantize_block(const bitnet_q6k_block_t *block, float *out, size_t out_len);
int bitnet_q6k_dot_product(const bitnet_q6k_block_t *block, const float *vec, size_t len, float *out);
int bitnet_q6k_dot_product_i8_neon(const bitnet_q6k_block_t *block, const int8_t *qvec,
                                   float vec_scale, size_t len, float *out);
int bitnet_q6k_expand_to_q8(const void *weight, int rows, int cols,
                            int8_t *q8, float *scales);
int bitnet_q6k_expand_to_q8_compact(const void *weight, int rows, int cols,
                                    int8_t *q8, int8_t *scales, float *d);
int bitnet_q6k_dot_product_q8_neon(const int8_t *q8, const float *scales,
                                   int blocks_per_row, const int8_t *qvec,
                                   float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_4_neon(const int8_t *q8, const float *scales,
                                     int row_stride, int scale_stride,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_8_neon(const int8_t *q8, const float *scales,
                                     int row_stride, int scale_stride,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float out[8]);
int bitnet_q6k_dot_product_q8_compact_neon(const int8_t *q8, const int8_t *scales,
                                           const float *d, int blocks_per_row,
                                           const int8_t *qvec, float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_compact_4_neon(const int8_t *q8, const int8_t *scales,
                                             const float *d, int row_stride,
                                             int scale_stride, int d_stride,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_compact_8_neon(const int8_t *q8, const int8_t *scales,
                                             const float *d, int row_stride,
                                             int scale_stride, int d_stride,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float out[8]);

#endif
