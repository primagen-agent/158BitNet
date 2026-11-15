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

/* fp16 -> fp32 conversion (matches scalar reference). Public so per-arch
 * translation units can share a single implementation. */
float bitnet_fp16_to_fp32(uint16_t h);
uint16_t bitnet_fp32_to_fp16(float v);

int bitnet_tq2_0_dequantize_block(const bitnet_tq2_0_block_t *block, float *out, size_t out_len);
int bitnet_tq2_0_dot_product(const bitnet_tq2_0_block_t *block, const float *vec, size_t len, float *out);
size_t bitnet_tq2_0_lut_float_count(int in_dim);
int bitnet_tq2_0_build_vec_lut(const float *vec, int in_dim, float *lut, size_t lut_count);
size_t bitnet_tq2_0_scale_float_count(int out_dim, int in_dim);
int bitnet_tq2_0_build_scales(const void *weight, int out_dim, int in_dim,
                              float *scales, size_t scale_count);
int bitnet_tq2_0_quantize_vec_i8(const float *vec, int in_dim, int8_t *qvec, float *scale, int32_t *block_bsums);
/* Internal _impl entry point — the per-tier dispatch trampolines in
 * kernel_registry.c (scalar / arm_neon shims) call this directly to avoid
 * infinite recursion through g_bitnet_dispatch. */
int bitnet_tq2_0_quantize_vec_i8_impl(const float *vec, int in_dim, int8_t *qvec,
                                      float *scale, int32_t *block_bsums);
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
/* Block-verify form: B activation rows share one weight pass.  Row b of the
 * activation is qvec_b + b*in_dim with scale vec_scale_b[b] and its own
 * blocks_per_row block sums (block_bsums_b + b*blocks_per_row).  The result
 * lands at out[b*out_dim + row].  B is 1..16. */
int bitnet_tq2_0_matmul_batched_i8(const void *weight, const float *scales,
                                   int out_dim, int in_dim,
                                   const int8_t *qvec_b, const float *vec_scale_b,
                                   const int32_t *block_bsums_b, int B, float *out);
/* Rows [row0, row1) only, written in place at the same row index of `out` with
 * row stride out_dim.  A hybrid caller hands the GPU the tail of the matrix and
 * runs this kernel over the head, both writing the same [B][out_dim] result. */
int bitnet_tq2_0_matmul_batched_i8_rows(const void *weight, const float *scales,
                                        int out_dim, int row0, int row1, int in_dim,
                                        const int8_t *qvec_b, const float *vec_scale_b,
                                        const int32_t *block_bsums_b, int B, float *out);
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
/* Internal _impl entry points — called by per-tier dispatch trampolines. */
int bitnet_tq2_0_matmul_vector_lut_impl(const void *weight, int out_dim, int in_dim,
                                         const float *lut, float *out);
int bitnet_tq2_0_matmul_vector_lut_scales_impl(const void *weight, const float *scales,
                                                int out_dim, int in_dim,
                                                const float *lut, float *out);
int bitnet_tq2_0_matmul_vector_lut_pair_impl(const void *weight_a, const void *weight_b,
                                              int out_dim, int in_dim, const float *lut,
                                              float *out_a, float *out_b);
int bitnet_tq2_0_matmul_vector_lut_pair_scales_impl(const void *weight_a, const float *scales_a,
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
 * The bsums input contains per-block quantized activation sums for unsigned
 * lookup correction. Signed-weight kernels do not use this input. */
size_t bitnet_tq2_0_i2s_packed_size(int out_dim, int in_dim);
int bitnet_tq2_0_reorder_to_i2s(const void *weight, int out_dim, int in_dim,
                                  uint8_t *packed, float *packed_scales);
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
/* Internal _impl entry points — called by per-tier dispatch trampolines. */
int bitnet_tq2_0_matmul_i2s_neon_parallel_impl(const uint8_t *packed, const float *scales,
                                                 const int32_t *bsums, int out_dim, int in_dim,
                                                 const int8_t *qvec, float vec_scale, float *out);
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_impl(const uint8_t *packed_a, const float *scales_a,
                                                      const uint8_t *packed_b, const float *scales_b,
                                                      const int32_t *bsums, int out_dim, int in_dim,
                                                      const int8_t *qvec, float vec_scale,
                                                      float *out_a, float *out_b);
int bitnet_tq2_0_matmul_i2s_qkv_parallel_impl(const uint8_t *packed_q, const float *scales_q,
                                                const uint8_t *packed_k, const float *scales_k,
                                                const uint8_t *packed_v, const float *scales_v,
                                                const int32_t *bsums,
                                                int q_dim, int kv_dim, int in_dim,
                                                const int8_t *qvec, float vec_scale,
                                                float *out_q, float *out_k, float *out_v);
void bitnet_tq2_0_park_workers(void);


#endif
