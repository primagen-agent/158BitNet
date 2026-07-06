#ifndef BITNET_DISPATCH_H
#define BITNET_DISPATCH_H

#include "cpu_detect.h"
#include "quant_q6k.h"

/* Function-pointer table. Each public kernel in the runtime has one slot.
 * Slots are added incrementally as phases bring kernels online; entries
 * that haven't been implemented yet are NULL until their phase lands.
 *
 * Convention: scalar-tier implementation is always populated first; ARM
 * builds resolve the same table to the existing NEON implementations
 * directly via #ifdef in bitnet_dispatch.c. */
typedef struct bitnet_dispatch {
    bitnet_cpu_tier_t tier;

    /* ops.c kernels (Phase 2) */
    void (*rms_norm_eps)(float *x, const float *weight, int n, float eps);
    void (*rms_norm_inplace_eps)(float *dst, const float *src,
                                  const float *weight, int n, float eps);
    void  (*silu)(float *x, int n);
    void  (*silu_mul)(float *gate, const float *up, int n);
    float (*silu_mul_max_abs)(float *gate, const float *up, int n);
    float (*relu2_mul_max_abs)(float *gate, const float *up, int n);
    void  (*residual_add)(float *out, const float *a, const float *b, int n);
    void  (*softmax)(float *x, int n);
    void  (*rope_apply)(float *x, int n_heads, int head_dim, int rope_dim,
                        const float *rope_cos, const float *rope_sin);

    /* TQ2_0 kernels (Phase 3, 5) — populated incrementally */
    int  (*tq2_quantize_vec_i8)(const float *vec, int in_dim, int8_t *qvec,
                                 float *scale, int32_t *block_bsums);
    int  (*tq2_matmul_vector_lut)(const void *weight, int out_dim, int in_dim,
                                   const float *lut, float *out);
    int  (*tq2_matmul_vector_lut_scales)(const void *weight, const float *scales,
                                          int out_dim, int in_dim,
                                          const float *lut, float *out);
    int  (*tq2_matmul_vector_lut_pair)(const void *weight_a, const void *weight_b,
                                        int out_dim, int in_dim,
                                        const float *lut, float *out_a, float *out_b);
    int  (*tq2_matmul_vector_lut_pair_scales)(const void *weight_a, const float *scales_a,
                                               const void *weight_b, const float *scales_b,
                                               int out_dim, int in_dim,
                                               const float *lut, float *out_a, float *out_b);

    /* TQ2_0 I2S matmul (Phase 3) — decode hot path. 4-row packed layout. */
    int  (*tq2_matmul_i2s_neon_parallel)(const uint8_t *packed, const float *scales,
                                          const int32_t *bsums, int out_dim, int in_dim,
                                          const int8_t *qvec, float vec_scale, float *out);
    int  (*tq2_matmul_i2s_neon_pair_parallel)(const uint8_t *packed_a, const float *scales_a,
                                                const uint8_t *packed_b, const float *scales_b,
                                                const int32_t *bsums, int out_dim, int in_dim,
                                                const int8_t *qvec, float vec_scale,
                                                float *out_a, float *out_b);
    int  (*tq2_matmul_i2s_qkv_parallel)(const uint8_t *packed_q, const float *scales_q,
                                         const uint8_t *packed_k, const float *scales_k,
                                         const uint8_t *packed_v, const float *scales_v,
                                         const int32_t *bsums,
                                         int q_dim, int kv_dim, int in_dim,
                                         const int8_t *qvec, float vec_scale,
                                         float *out_q, float *out_k, float *out_v);

    /* Q6K kernels (Phase 4) */
    int (*q6k_dot_product_i8)(const struct bitnet_q6k_block *block, const int8_t *qvec,
                              float vec_scale, size_t len, float *out);
    int (*q6k_dot_product_q8)(const int8_t *q8, const float *scales,
                              int blocks_per_row, const int8_t *qvec,
                              float vec_scale, float *out);
    int (*q6k_dot_product_q8_4)(const int8_t *q8, const float *scales,
                                int row_stride, int scale_stride,
                                int blocks_per_row, const int8_t *qvec,
                                float vec_scale, float out[4]);
    int (*q6k_dot_product_q8_8)(const int8_t *q8, const float *scales,
                                int row_stride, int scale_stride,
                                int blocks_per_row, const int8_t *qvec,
                                float vec_scale, float out[8]);
    int (*q6k_dot_product_q8_compact)(const int8_t *q8, const int8_t *scales,
                                      const float *d, int blocks_per_row,
                                      const int8_t *qvec, float vec_scale, float *out);
    int (*q6k_dot_product_q8_compact_4)(const int8_t *q8, const int8_t *scales,
                                        const float *d, int row_stride,
                                        int scale_stride, int d_stride,
                                        int blocks_per_row, const int8_t *qvec,
                                        float vec_scale, float out[4]);
    int (*q6k_dot_product_q8_compact_8)(const int8_t *q8, const int8_t *scales,
                                        const float *d, int row_stride,
                                        int scale_stride, int d_stride,
                                        int blocks_per_row, const int8_t *qvec,
                                        float vec_scale, float out[8]);

    /* Bitnet.c hot paths (Phase 4) — promoted static helpers serving the
     * Q8-KV-cache attention path and FFN residual. Wired through the
     * dispatch table so x86 tiers can supply SIMD variants; ARM and scalar
     * tiers route back to the existing `_impl` bodies in bitnet.c. */
    float (*quantize_f32_to_i8)(const float *src, int n, int8_t *dst);
    int   (*rms_norm_quant_tq2_i8)(float *dst, const float *src, const float *weight,
                                   int n, int8_t *qvec, float *scale,
                                   int32_t *block_bsums, float eps);
    void  (*residual_add_scaled)(float *out, const float *a, const float *b,
                                 float b_scale, int n);
    int   (*dot_i8)(const int8_t *a, const int8_t *b, int n);
    void  (*accum_i8_scaled)(float *dst, const int8_t *src, float scale, int n);

    /* Padding reserved for future kernels — keeps the struct layout stable
     * across phases so callers don't need recompilation between phases. */
    void *_reserved[6];
} bitnet_dispatch_t;

extern bitnet_dispatch_t *g_bitnet_dispatch;

/* Resolve g_bitnet_dispatch to the right tier. Idempotent; runs the CPU
 * detection once under pthread_once. Reads BITNET_CPU_TIER override env
 * var (validated; falls back to auto-detect with a warning on bad input).
 * Reads BITNET_QUIET=1 to suppress the startup log line. */
void bitnet_dispatch_init(void);

#endif
