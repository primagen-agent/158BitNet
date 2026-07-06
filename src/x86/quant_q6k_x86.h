#ifndef BITNET_QUANT_Q6K_X86_H
#define BITNET_QUANT_Q6K_X86_H

#include "../quant_q6k.h"
#include <stdint.h>

/* AVX2 tier — emulates dotprod via maddubs_epi16 + madd_epi16 with the
 * unsigned-bias trick (weights + 32 → uint8, subtract 32*sum(activations)
 * per lane). */
int bitnet_q6k_dot_product_i8_neon_avx2(const bitnet_q6k_block_t *block,
                                          const int8_t *qvec, float vec_scale,
                                          size_t len, float *out);
int bitnet_q6k_dot_product_q8_avx2(const int8_t *q8, const float *scales,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_4_avx2(const int8_t *q8, const float *scales,
                                       int row_stride, int scale_stride,
                                       int blocks_per_row, const int8_t *qvec,
                                       float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_8_avx2(const int8_t *q8, const float *scales,
                                       int row_stride, int scale_stride,
                                       int blocks_per_row, const int8_t *qvec,
                                       float vec_scale, float out[8]);
int bitnet_q6k_dot_product_q8_compact_avx2(const int8_t *q8, const int8_t *scales,
                                             const float *d, int blocks_per_row,
                                             const int8_t *qvec, float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_compact_4_avx2(const int8_t *q8, const int8_t *scales,
                                                const float *d, int row_stride,
                                                int scale_stride, int d_stride,
                                                int blocks_per_row, const int8_t *qvec,
                                                float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_compact_8_avx2(const int8_t *q8, const int8_t *scales,
                                                const float *d, int row_stride,
                                                int scale_stride, int d_stride,
                                                int blocks_per_row, const int8_t *qvec,
                                                float vec_scale, float out[8]);

/* AVX-VNNI tier — uses _mm256_dpbusd_epi32 directly (true unsigned×signed
 * dot product). The unsigned-bias trick is still required because Q6K
 * weights are signed; the +32 offset moves them into 0..63 uint8 range. */
int bitnet_q6k_dot_product_i8_neon_avx_vnni(const bitnet_q6k_block_t *block,
                                                const int8_t *qvec, float vec_scale,
                                                size_t len, float *out);
int bitnet_q6k_dot_product_q8_avx_vnni(const int8_t *q8, const float *scales,
                                          int blocks_per_row, const int8_t *qvec,
                                          float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_4_avx_vnni(const int8_t *q8, const float *scales,
                                            int row_stride, int scale_stride,
                                            int blocks_per_row, const int8_t *qvec,
                                            float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_8_avx_vnni(const int8_t *q8, const float *scales,
                                            int row_stride, int scale_stride,
                                            int blocks_per_row, const int8_t *qvec,
                                            float vec_scale, float out[8]);
int bitnet_q6k_dot_product_q8_compact_avx_vnni(const int8_t *q8, const int8_t *scales,
                                                  const float *d, int blocks_per_row,
                                                  const int8_t *qvec, float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_compact_4_avx_vnni(const int8_t *q8, const int8_t *scales,
                                                     const float *d, int row_stride,
                                                     int scale_stride, int d_stride,
                                                     int blocks_per_row, const int8_t *qvec,
                                                     float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_compact_8_avx_vnni(const int8_t *q8, const int8_t *scales,
                                                     const float *d, int row_stride,
                                                     int scale_stride, int d_stride,
                                                     int blocks_per_row, const int8_t *qvec,
                                                     float vec_scale, float out[8]);

/* AVX512-VNNI tier — delegates to AVX-VNNI in Phase 4. Phase 7 will
 * specialize using _mm512_dpbusd_epi32. */
int bitnet_q6k_dot_product_i8_neon_avx512_vnni(const bitnet_q6k_block_t *block,
                                                    const int8_t *qvec, float vec_scale,
                                                    size_t len, float *out);
int bitnet_q6k_dot_product_q8_avx512_vnni(const int8_t *q8, const float *scales,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_4_avx512_vnni(const int8_t *q8, const float *scales,
                                               int row_stride, int scale_stride,
                                               int blocks_per_row, const int8_t *qvec,
                                               float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_8_avx512_vnni(const int8_t *q8, const float *scales,
                                               int row_stride, int scale_stride,
                                               int blocks_per_row, const int8_t *qvec,
                                               float vec_scale, float out[8]);
int bitnet_q6k_dot_product_q8_compact_avx512_vnni(const int8_t *q8, const int8_t *scales,
                                                     const float *d, int blocks_per_row,
                                                     const int8_t *qvec, float vec_scale, float *out);
int bitnet_q6k_dot_product_q8_compact_4_avx512_vnni(const int8_t *q8, const int8_t *scales,
                                                        const float *d, int row_stride,
                                                        int scale_stride, int d_stride,
                                                        int blocks_per_row, const int8_t *qvec,
                                                        float vec_scale, float out[4]);
int bitnet_q6k_dot_product_q8_compact_8_avx512_vnni(const int8_t *q8, const int8_t *scales,
                                                        const float *d, int row_stride,
                                                        int scale_stride, int d_stride,
                                                        int blocks_per_row, const int8_t *qvec,
                                                        float vec_scale, float out[8]);

#endif
