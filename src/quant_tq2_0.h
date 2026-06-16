#ifndef QUANT_TQ2_0_H
#define QUANT_TQ2_0_H

#include <stddef.h>
#include <stdint.h>

#define BITNET_TQ2_0_QK 256
#define BITNET_TQ2_0_QS_SIZE 64
#define BITNET_TQ2_0_BLOCK_SIZE 66

typedef struct bitnet_tq2_0_block {
    uint8_t qs[BITNET_TQ2_0_QS_SIZE];
    uint16_t d;
} bitnet_tq2_0_block_t;

int bitnet_tq2_0_dequantize_block(const bitnet_tq2_0_block_t *block, float *out, size_t out_len);
int bitnet_tq2_0_dot_product(const bitnet_tq2_0_block_t *block, const float *vec, size_t len, float *out);
size_t bitnet_tq2_0_lut_float_count(int in_dim);
int bitnet_tq2_0_build_vec_lut(const float *vec, int in_dim, float *lut, size_t lut_count);
size_t bitnet_tq2_0_scale_float_count(int out_dim, int in_dim);
int bitnet_tq2_0_build_scales(const void *weight, int out_dim, int in_dim,
                              float *scales, size_t scale_count);
int bitnet_tq2_0_quantize_vec_i8(const float *vec, int in_dim, int8_t *qvec, float *scale, int32_t *block_bsums);
int bitnet_tq2_0_quantize_vec_i8_known_max(const float *vec, int in_dim, int8_t *qvec,
                                           float *scale, int32_t *block_bsums,
                                           float max_abs);
size_t bitnet_tq2_0_i16_lut_count(int in_dim);
int bitnet_tq2_0_build_i16_lut(const int8_t *qvec, int in_dim,
                               int16_t *lut, size_t lut_count);
int bitnet_tq2_0_matmul_vector_i8_scales(const void *weight, const float *scales,
                                         int out_dim, int in_dim,
                                         const int8_t *qvec, float vec_scale,
                                         float *out);
int bitnet_tq2_0_matmul_vector_i8_neon_scales(const void *weight, const float *scales,
                                              int out_dim, int in_dim,
                                              const int8_t *qvec, float vec_scale,
                                              const int32_t *block_bsums, float *out);
/* Single-threaded variant — avoids thread pool scheduling overhead.
 * Faster for small matrices where parallel overhead exceeds benefit. */
int bitnet_tq2_0_matmul_vector_i8_neon_scales_single(const void *weight, const float *scales,
                                                       int out_dim, int in_dim,
                                                       const int8_t *qvec, float vec_scale,
                                                       const int32_t *block_bsums, float *out);
int bitnet_tq2_0_matmul_vector_i8_neon_pair_scales(const void *weight_a, const float *scales_a,
                                                   const void *weight_b, const float *scales_b,
                                                   int out_dim, int in_dim,
                                                   const int8_t *qvec, float vec_scale,
                                                   const int32_t *block_bsums, float *out_a, float *out_b);

/* Fused QKV matmul: compute Q/K/V projections in a single thread pool dispatch.
 * The three weight matrices share the same quantized input vector (qvec).
 * out_q has emb_dim elements, out_k and out_v have kv_dim elements each.
 * This reduces thread pool synchronization overhead from 3 dispatches to 1. */
int bitnet_tq2_0_matmul_qkv_fused(const void *w_q, const float *scales_q,
                                    const void *w_k, const float *scales_k,
                                    const void *w_v, const float *scales_v,
                                    int emb_dim, int kv_dim, int in_dim,
                                    const int8_t *qvec, float vec_scale,
                                    const int32_t *block_bsums,
                                    float *out_q, float *out_k, float *out_v);
int bitnet_tq2_0_matmul_vector_i16_lut_scales(const void *weight, const float *scales,
                                              int out_dim, int in_dim,
                                              const int16_t *lut, float vec_scale,
                                              float *out);
int bitnet_tq2_0_matmul_vector_lut(const void *weight, int out_dim, int in_dim, const float *lut, float *out);
int bitnet_tq2_0_matmul_vector_lut_scales(const void *weight, const float *scales,
                                          int out_dim, int in_dim, const float *lut,
                                          float *out);
int bitnet_tq2_0_matmul_vector_lut_pair(const void *weight_a, const void *weight_b,
                                        int out_dim, int in_dim, const float *lut,
                                        float *out_a, float *out_b);
int bitnet_tq2_0_matmul_vector_lut_pair_scales(const void *weight_a, const float *scales_a,
                                               const void *weight_b, const float *scales_b,
                                               int out_dim, int in_dim, const float *lut,
                                               float *out_a, float *out_b);
int bitnet_tq2_0_matmul_vector(const void *weight, int out_dim, int in_dim, const float *vec, float *out);
int bitnet_tq2_0_matmul_vector_partial(const void *weight, int in_dim, int out_start, int out_count, const float *vec, float *out);

/* TL1 vqtbl1q_s8 GEMM functions */
int bitnet_tq2_0_reorder_to_tl1(const void *weight, int out_dim, int in_dim,
                                  uint8_t *reordered);
size_t bitnet_tq2_0_tl1_lut_size(int in_dim);
int bitnet_tq2_0_tl1_preprocessor(const float *vec, int in_dim,
                                    float *vec_scale, int8_t *lut, size_t lut_size);
int bitnet_tq2_0_tl1_qgemm(const uint8_t *weight_tl1, const float *scales,
                              const int32_t *bsums, int out_dim, int in_dim,
                              const int8_t *lut, float vec_scale, float *out);
int bitnet_tq2_0_tl1_qgemm_pair(const uint8_t *weight_a_tl1, const float *scales_a,
                                   const uint8_t *weight_b_tl1, const float *scales_b,
                                   const int32_t *bsums, int out_dim, int in_dim,
                                   const int8_t *lut, float vec_scale,
                                   float *out_a, float *out_b);

/* I2_S interleaved format: 4-row packed 2-bit weights.
 * The bsums arguments are optional for the signed NEON I2S path and are kept
 * in the API for compatibility with older callers/tests. */
size_t bitnet_tq2_0_i2s_packed_size(int out_dim, int in_dim);
int bitnet_tq2_0_reorder_to_i2s(const void *weight, int out_dim, int in_dim,
                                  uint8_t *packed, float *packed_scales, int32_t *packed_bsums);
int bitnet_tq2_0_matmul_i2s_neon(const uint8_t *packed, const float *scales,
                                   const int32_t *bsums, int out_dim, int in_dim,
                                   const int8_t *qvec, float vec_scale, float *out);
int bitnet_tq2_0_matmul_i2s_neon_pair(const uint8_t *packed_a, const float *scales_a,
                                        const uint8_t *packed_b, const float *scales_b,
                                        const int32_t *bsums, int out_dim, int in_dim,
                                        const int8_t *qvec, float vec_scale,
                                        float *out_a, float *out_b);
/* Multi-threaded I2S matmul — uses the global thread pool for parallelism */
int bitnet_tq2_0_matmul_i2s_neon_parallel(const uint8_t *packed, const float *scales,
                                            const int32_t *bsums, int out_dim, int in_dim,
                                            const int8_t *qvec, float vec_scale, float *out);
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel(const uint8_t *packed_a, const float *scales_a,
                                                  const uint8_t *packed_b, const float *scales_b,
                                                  const int32_t *bsums, int out_dim, int in_dim,
                                                  const int8_t *qvec, float vec_scale,
                                                  float *out_a, float *out_b);
int bitnet_tq2_0_matmul_i2s_qkv_parallel(const uint8_t *packed_q, const float *scales_q,
                                          const uint8_t *packed_k, const float *scales_k,
                                          const uint8_t *packed_v, const float *scales_v,
                                          const int32_t *bsums,
                                          int q_dim, int kv_dim, int in_dim,
                                          const int8_t *qvec, float vec_scale,
                                          float *out_q, float *out_k, float *out_v);
void bitnet_tq2_0_park_workers(void);

/* Deprecated TL1 API (kept for backward compatibility, returns -1) */
size_t bitnet_tq2_0_vqtbl1q_lut_size(int in_dim);
int bitnet_tq2_0_build_vqtbl1q_lut(const int8_t *qvec, int in_dim,
                                    int8_t *lut, size_t lut_size);
int bitnet_tq2_0_matmul_vqtbl1q_lut(const void *weight, const float *scales,
                                     int out_dim, int in_dim,
                                     const int8_t *qvec, float vec_scale,
                                     const int32_t *block_bsums,
                                     const int8_t *tl1_lut, float *out);
int bitnet_tq2_0_matmul_vqtbl1q_lut_pair(const void *weight_a, const float *scales_a,
                                          const void *weight_b, const float *scales_b,
                                          int out_dim, int in_dim,
                                          const int8_t *qvec, float vec_scale,
                                          const int32_t *block_bsums,
                                          const int8_t *tl1_lut,
                                          float *out_a, float *out_b);

#endif
