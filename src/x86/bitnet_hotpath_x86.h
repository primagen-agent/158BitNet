#ifndef BITNET_HOTPATH_X86_H
#define BITNET_HOTPATH_X86_H

#include <stdint.h>

/* Phase 4.2 — promoted bitnet.c hot-path helpers, x86 SIMD variants.
 * Scalar and ARM NEON bodies live in bitnet.c as `_impl` symbols; the
 * dispatch table picks one of the three x86 variants below per tier.
 * VNNI and AVX512 variants currently alias the AVX2 implementation; they
 * will be specialised in a later phase (DPBUSD for VNNI, wide vectors for
 * AVX512). */

float bitnet_quantize_f32_to_i8_avx2(const float *src, int n, int8_t *dst);
float bitnet_quantize_f32_to_i8_avx_vnni(const float *src, int n, int8_t *dst);
float bitnet_quantize_f32_to_i8_avx512_vnni(const float *src, int n, int8_t *dst);

int bitnet_rms_norm_quant_tq2_i8_avx2(float *dst, const float *src, const float *weight,
                                       int n, int8_t *qvec, float *scale,
                                       int32_t *block_bsums, float eps);
int bitnet_rms_norm_quant_tq2_i8_avx_vnni(float *dst, const float *src, const float *weight,
                                          int n, int8_t *qvec, float *scale,
                                          int32_t *block_bsums, float eps);
int bitnet_rms_norm_quant_tq2_i8_avx512_vnni(float *dst, const float *src, const float *weight,
                                             int n, int8_t *qvec, float *scale,
                                             int32_t *block_bsums, float eps);

void bitnet_residual_add_scaled_avx2(float *out, const float *a, const float *b,
                                     float b_scale, int n);
void bitnet_residual_add_scaled_avx_vnni(float *out, const float *a, const float *b,
                                         float b_scale, int n);
void bitnet_residual_add_scaled_avx512_vnni(float *out, const float *a, const float *b,
                                            float b_scale, int n);

int bitnet_dot_i8_avx2(const int8_t *a, const int8_t *b, int n);
int bitnet_dot_i8_avx_vnni(const int8_t *a, const int8_t *b, int n);
int bitnet_dot_i8_avx512_vnni(const int8_t *a, const int8_t *b, int n);

void bitnet_accum_i8_scaled_avx2(float *dst, const int8_t *src, float scale, int n);
void bitnet_accum_i8_scaled_avx_vnni(float *dst, const int8_t *src, float scale, int n);
void bitnet_accum_i8_scaled_avx512_vnni(float *dst, const int8_t *src, float scale, int n);

#endif
