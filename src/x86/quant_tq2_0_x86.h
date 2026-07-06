#ifndef BITNET_QUANT_TQ2_0_X86_H
#define BITNET_QUANT_TQ2_0_X86_H

#include <stdint.h>

int bitnet_tq2_0_quantize_vec_i8_avx2(const float *vec, int in_dim, int8_t *qvec,
                                       float *scale, int32_t *block_bsums);
int bitnet_tq2_0_quantize_vec_i8_avx_vnni(const float *vec, int in_dim, int8_t *qvec,
                                            float *scale, int32_t *block_bsums);
int bitnet_tq2_0_quantize_vec_i8_avx512_vnni(const float *vec, int in_dim, int8_t *qvec,
                                                float *scale, int32_t *block_bsums);

int bitnet_tq2_0_matmul_vector_lut_avx2(const void *weight, int out_dim, int in_dim,
                                          const float *lut, float *out);
int bitnet_tq2_0_matmul_vector_lut_avx_vnni(const void *weight, int out_dim, int in_dim,
                                              const float *lut, float *out);
int bitnet_tq2_0_matmul_vector_lut_avx512_vnni(const void *weight, int out_dim, int in_dim,
                                                  const float *lut, float *out);

int bitnet_tq2_0_matmul_vector_lut_scales_avx2(const void *weight, const float *scales,
                                                  int out_dim, int in_dim,
                                                  const float *lut, float *out);
int bitnet_tq2_0_matmul_vector_lut_scales_avx_vnni(const void *weight, const float *scales,
                                                     int out_dim, int in_dim,
                                                     const float *lut, float *out);
int bitnet_tq2_0_matmul_vector_lut_scales_avx512_vnni(const void *weight, const float *scales,
                                                         int out_dim, int in_dim,
                                                         const float *lut, float *out);

int bitnet_tq2_0_matmul_vector_lut_pair_avx2(const void *weight_a, const void *weight_b,
                                                int out_dim, int in_dim, const float *lut,
                                                float *out_a, float *out_b);
int bitnet_tq2_0_matmul_vector_lut_pair_avx_vnni(const void *weight_a, const void *weight_b,
                                                   int out_dim, int in_dim, const float *lut,
                                                   float *out_a, float *out_b);
int bitnet_tq2_0_matmul_vector_lut_pair_avx512_vnni(const void *weight_a, const void *weight_b,
                                                       int out_dim, int in_dim, const float *lut,
                                                       float *out_a, float *out_b);

int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx2(const void *weight_a, const float *scales_a,
                                                       const void *weight_b, const float *scales_b,
                                                       int out_dim, int in_dim, const float *lut,
                                                       float *out_a, float *out_b);
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx_vnni(const void *weight_a, const float *scales_a,
                                                          const void *weight_b, const float *scales_b,
                                                          int out_dim, int in_dim, const float *lut,
                                                          float *out_a, float *out_b);
int bitnet_tq2_0_matmul_vector_lut_pair_scales_avx512_vnni(const void *weight_a, const float *scales_a,
                                                              const void *weight_b, const float *scales_b,
                                                              int out_dim, int in_dim, const float *lut,
                                                              float *out_a, float *out_b);

/* I2S matmul — single/pair/qkv parallel variants.
 * Phase 3: AVX2 implements these directly (no thread pool on x86 yet).
 * VNNI / AVX512 delegate to AVX2. */
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx2(const uint8_t *packed, const float *scales,
                                                  const int32_t *bsums, int out_dim, int in_dim,
                                                  const int8_t *qvec, float vec_scale, float *out);
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx_vnni(const uint8_t *packed, const float *scales,
                                                      const int32_t *bsums, int out_dim, int in_dim,
                                                      const int8_t *qvec, float vec_scale, float *out);
int bitnet_tq2_0_matmul_i2s_neon_parallel_avx512_vnni(const uint8_t *packed, const float *scales,
                                                         const int32_t *bsums, int out_dim, int in_dim,
                                                         const int8_t *qvec, float vec_scale, float *out);

int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx2(const uint8_t *packed_a, const float *scales_a,
                                                       const uint8_t *packed_b, const float *scales_b,
                                                       const int32_t *bsums, int out_dim, int in_dim,
                                                       const int8_t *qvec, float vec_scale,
                                                       float *out_a, float *out_b);
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx_vnni(const uint8_t *packed_a, const float *scales_a,
                                                           const uint8_t *packed_b, const float *scales_b,
                                                           const int32_t *bsums, int out_dim, int in_dim,
                                                           const int8_t *qvec, float vec_scale,
                                                           float *out_a, float *out_b);
int bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx512_vnni(const uint8_t *packed_a, const float *scales_a,
                                                              const uint8_t *packed_b, const float *scales_b,
                                                              const int32_t *bsums, int out_dim, int in_dim,
                                                              const int8_t *qvec, float vec_scale,
                                                              float *out_a, float *out_b);

int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx2(const uint8_t *packed_q, const float *scales_q,
                                                 const uint8_t *packed_k, const float *scales_k,
                                                 const uint8_t *packed_v, const float *scales_v,
                                                 const int32_t *bsums,
                                                 int q_dim, int kv_dim, int in_dim,
                                                 const int8_t *qvec, float vec_scale,
                                                 float *out_q, float *out_k, float *out_v);
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx_vnni(const uint8_t *packed_q, const float *scales_q,
                                                     const uint8_t *packed_k, const float *scales_k,
                                                     const uint8_t *packed_v, const float *scales_v,
                                                     const int32_t *bsums,
                                                     int q_dim, int kv_dim, int in_dim,
                                                     const int8_t *qvec, float vec_scale,
                                                     float *out_q, float *out_k, float *out_v);
int bitnet_tq2_0_matmul_i2s_qkv_parallel_avx512_vnni(const uint8_t *packed_q, const float *scales_q,
                                                        const uint8_t *packed_k, const float *scales_k,
                                                        const uint8_t *packed_v, const float *scales_v,
                                                        const int32_t *bsums,
                                                        int q_dim, int kv_dim, int in_dim,
                                                        const int8_t *qvec, float vec_scale,
                                                        float *out_q, float *out_k, float *out_v);

#endif
