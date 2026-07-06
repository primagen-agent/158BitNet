#include "kernel_registry.h"
#include "../bitnet_internal.h"
#include "../ops.h"
#include "../quant_q6k.h"
#include "../quant_tq2_0.h"
#include "bitnet_hotpath_x86.h"
#include "ops_x86.h"
#include "quant_q6k_x86.h"
#include "quant_tq2_0_x86.h"

/* The scalar rms_norm path. ops.c already has a non-NEON body for
 * bitnet_rms_norm_eps when __ARM_NEON is undefined — the function symbol
 * itself is the scalar path on x86. On ARM, the same name resolves to the
 * NEON path, which we'll keep using for g_dispatch_arm_neon. */

static void shim_rms_norm_eps(float *x, const float *weight, int n, float eps) {
    bitnet_rms_norm_eps_impl(x, weight, n, eps);
}
static void shim_rms_norm_inplace_eps(float *dst, const float *src,
                                       const float *weight, int n, float eps) {
    bitnet_rms_norm_inplace_eps_impl(dst, src, weight, n, eps);
}
static void shim_silu(float *x, int n) {
    bitnet_silu_impl(x, n);
}
static void shim_silu_mul(float *gate, const float *up, int n) {
    bitnet_silu_mul_impl(gate, up, n);
}
static float shim_silu_mul_max_abs(float *gate, const float *up, int n) {
    return bitnet_silu_mul_max_abs_impl(gate, up, n);
}
static float shim_relu2_mul_max_abs(float *gate, const float *up, int n) {
    return bitnet_relu2_mul_max_abs_impl(gate, up, n);
}
static void shim_residual_add(float *out, const float *a, const float *b, int n) {
    bitnet_residual_add_impl(out, a, b, n);
}
static void shim_softmax(float *x, int n) {
    bitnet_softmax_impl(x, n);
}
static void shim_rope_apply(float *x, int n_heads, int head_dim, int rope_dim,
                             const float *rope_cos, const float *rope_sin) {
    bitnet_rope_apply_impl(x, n_heads, head_dim, rope_dim, rope_cos, rope_sin);
}
static int shim_tq2_quantize_vec_i8(const float *vec, int in_dim, int8_t *qvec,
                                     float *scale, int32_t *block_bsums) {
    return bitnet_tq2_0_quantize_vec_i8_impl(vec, in_dim, qvec, scale, block_bsums);
}
static int shim_tq2_matmul_vector_lut(const void *weight, int out_dim, int in_dim,
                                       const float *lut, float *out) {
    return bitnet_tq2_0_matmul_vector_lut_impl(weight, out_dim, in_dim, lut, out);
}
static int shim_tq2_matmul_vector_lut_scales(const void *weight, const float *scales,
                                                int out_dim, int in_dim,
                                                const float *lut, float *out) {
    return bitnet_tq2_0_matmul_vector_lut_scales_impl(weight, scales, out_dim, in_dim, lut, out);
}
static int shim_tq2_matmul_vector_lut_pair(const void *weight_a, const void *weight_b,
                                             int out_dim, int in_dim,
                                             const float *lut, float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_vector_lut_pair_impl(weight_a, weight_b, out_dim, in_dim,
                                                      lut, out_a, out_b);
}
static int shim_tq2_matmul_vector_lut_pair_scales(const void *weight_a, const float *scales_a,
                                                    const void *weight_b, const float *scales_b,
                                                    int out_dim, int in_dim,
                                                    const float *lut, float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_vector_lut_pair_scales_impl(weight_a, scales_a, weight_b, scales_b,
                                                              out_dim, in_dim, lut, out_a, out_b);
}
static int shim_tq2_matmul_i2s_neon_parallel(const uint8_t *packed, const float *scales,
                                                const int32_t *bsums, int out_dim, int in_dim,
                                                const int8_t *qvec, float vec_scale, float *out) {
    return bitnet_tq2_0_matmul_i2s_neon_parallel_impl(packed, scales, bsums,
                                                         out_dim, in_dim, qvec, vec_scale, out);
}
static int shim_tq2_matmul_i2s_neon_pair_parallel(const uint8_t *packed_a, const float *scales_a,
                                                     const uint8_t *packed_b, const float *scales_b,
                                                     const int32_t *bsums, int out_dim, int in_dim,
                                                     const int8_t *qvec, float vec_scale,
                                                     float *out_a, float *out_b) {
    return bitnet_tq2_0_matmul_i2s_neon_pair_parallel_impl(packed_a, scales_a, packed_b, scales_b,
                                                              bsums, out_dim, in_dim, qvec, vec_scale,
                                                              out_a, out_b);
}
static int shim_tq2_matmul_i2s_qkv_parallel(const uint8_t *packed_q, const float *scales_q,
                                              const uint8_t *packed_k, const float *scales_k,
                                              const uint8_t *packed_v, const float *scales_v,
                                              const int32_t *bsums,
                                              int q_dim, int kv_dim, int in_dim,
                                              const int8_t *qvec, float vec_scale,
                                              float *out_q, float *out_k, float *out_v) {
    return bitnet_tq2_0_matmul_i2s_qkv_parallel_impl(packed_q, scales_q, packed_k, scales_k,
                                                        packed_v, scales_v, bsums,
                                                        q_dim, kv_dim, in_dim,
                                                        qvec, vec_scale, out_q, out_k, out_v);
}

static int shim_q6k_dot_product_i8(const bitnet_q6k_block_t *block, const int8_t *qvec,
                                   float vec_scale, size_t len, float *out) {
    return bitnet_q6k_dot_product_i8_neon_impl(block, qvec, vec_scale, len, out);
}
static int shim_q6k_dot_product_q8(const int8_t *q8, const float *scales,
                                   int blocks_per_row, const int8_t *qvec,
                                   float vec_scale, float *out) {
    return bitnet_q6k_dot_product_q8_neon_impl(q8, scales, blocks_per_row, qvec, vec_scale, out);
}
static int shim_q6k_dot_product_q8_4(const int8_t *q8, const float *scales,
                                     int row_stride, int scale_stride,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float out[4]) {
    return bitnet_q6k_dot_product_q8_4_neon_impl(q8, scales, row_stride, scale_stride,
                                                    blocks_per_row, qvec, vec_scale, out);
}
static int shim_q6k_dot_product_q8_8(const int8_t *q8, const float *scales,
                                     int row_stride, int scale_stride,
                                     int blocks_per_row, const int8_t *qvec,
                                     float vec_scale, float out[8]) {
    return bitnet_q6k_dot_product_q8_8_neon_impl(q8, scales, row_stride, scale_stride,
                                                    blocks_per_row, qvec, vec_scale, out);
}
static int shim_q6k_dot_product_q8_compact(const int8_t *q8, const int8_t *scales,
                                           const float *d, int blocks_per_row,
                                           const int8_t *qvec, float vec_scale, float *out) {
    return bitnet_q6k_dot_product_q8_compact_neon_impl(q8, scales, d, blocks_per_row,
                                                          qvec, vec_scale, out);
}
static int shim_q6k_dot_product_q8_compact_4(const int8_t *q8, const int8_t *scales,
                                             const float *d, int row_stride,
                                             int scale_stride, int d_stride,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float out[4]) {
    return bitnet_q6k_dot_product_q8_compact_4_neon_impl(q8, scales, d, row_stride,
                                                            scale_stride, d_stride,
                                                            blocks_per_row, qvec, vec_scale, out);
}
static int shim_q6k_dot_product_q8_compact_8(const int8_t *q8, const int8_t *scales,
                                             const float *d, int row_stride,
                                             int scale_stride, int d_stride,
                                             int blocks_per_row, const int8_t *qvec,
                                             float vec_scale, float out[8]) {
    return bitnet_q6k_dot_product_q8_compact_8_neon_impl(q8, scales, d, row_stride,
                                                            scale_stride, d_stride,
                                                            blocks_per_row, qvec, vec_scale, out);
}

/* Phase 4.2 — promoted bitnet.c hot-path shims. Scalar and ARM tiers route
 * back to the `_impl` bodies in bitnet.c; x86 tiers pick the SIMD variant. */
static float shim_quantize_f32_to_i8(const float *src, int n, int8_t *dst) {
    return bitnet_quantize_f32_to_i8_impl(src, n, dst);
}
static int shim_rms_norm_quant_tq2_i8(float *dst, const float *src, const float *weight,
                                      int n, int8_t *qvec, float *scale,
                                      int32_t *block_bsums, float eps) {
    return bitnet_rms_norm_quant_tq2_i8_impl(dst, src, weight, n, qvec,
                                              scale, block_bsums, eps);
}
static void shim_residual_add_scaled(float *out, const float *a, const float *b,
                                     float b_scale, int n) {
    bitnet_residual_add_scaled_impl(out, a, b, b_scale, n);
}
static int shim_dot_i8(const int8_t *a, const int8_t *b, int n) {
    return bitnet_dot_i8_impl(a, b, n);
}
static void shim_accum_i8_scaled(float *dst, const int8_t *src, float scale, int n) {
    bitnet_accum_i8_scaled_impl(dst, src, scale, n);
}

bitnet_dispatch_t g_dispatch_scalar = {
    .tier                     = BITNET_TIER_SCALAR,
    .rms_norm_eps             = shim_rms_norm_eps,
    .rms_norm_inplace_eps     = shim_rms_norm_inplace_eps,
    .silu                     = shim_silu,
    .silu_mul                 = shim_silu_mul,
    .silu_mul_max_abs         = shim_silu_mul_max_abs,
    .relu2_mul_max_abs        = shim_relu2_mul_max_abs,
    .residual_add             = shim_residual_add,
    .softmax                  = shim_softmax,
    .rope_apply               = shim_rope_apply,
    .tq2_quantize_vec_i8      = shim_tq2_quantize_vec_i8,
    .tq2_matmul_vector_lut              = shim_tq2_matmul_vector_lut,
    .tq2_matmul_vector_lut_scales       = shim_tq2_matmul_vector_lut_scales,
    .tq2_matmul_vector_lut_pair         = shim_tq2_matmul_vector_lut_pair,
    .tq2_matmul_vector_lut_pair_scales  = shim_tq2_matmul_vector_lut_pair_scales,
    .tq2_matmul_i2s_neon_parallel       = shim_tq2_matmul_i2s_neon_parallel,
    .tq2_matmul_i2s_neon_pair_parallel  = shim_tq2_matmul_i2s_neon_pair_parallel,
    .tq2_matmul_i2s_qkv_parallel        = shim_tq2_matmul_i2s_qkv_parallel,
    .q6k_dot_product_i8                 = shim_q6k_dot_product_i8,
    .q6k_dot_product_q8                 = shim_q6k_dot_product_q8,
    .q6k_dot_product_q8_4               = shim_q6k_dot_product_q8_4,
    .q6k_dot_product_q8_8               = shim_q6k_dot_product_q8_8,
    .q6k_dot_product_q8_compact         = shim_q6k_dot_product_q8_compact,
    .q6k_dot_product_q8_compact_4       = shim_q6k_dot_product_q8_compact_4,
    .q6k_dot_product_q8_compact_8       = shim_q6k_dot_product_q8_compact_8,
    .quantize_f32_to_i8                 = shim_quantize_f32_to_i8,
    .rms_norm_quant_tq2_i8              = shim_rms_norm_quant_tq2_i8,
    .residual_add_scaled                = shim_residual_add_scaled,
    .dot_i8                             = shim_dot_i8,
    .accum_i8_scaled                    = shim_accum_i8_scaled,
};

#if defined(__ARM_NEON)
/* On ARM, the same shims reach the existing NEON implementations because
 * ops.c / quant_tq2_0.c compile their NEON bodies under __ARM_NEON. */
bitnet_dispatch_t g_dispatch_arm_neon = {
    .tier                     = BITNET_TIER_AVX2, /* placeholder tag */
    .rms_norm_eps             = shim_rms_norm_eps,
    .rms_norm_inplace_eps     = shim_rms_norm_inplace_eps,
    .silu                     = shim_silu,
    .silu_mul                 = shim_silu_mul,
    .silu_mul_max_abs         = shim_silu_mul_max_abs,
    .relu2_mul_max_abs        = shim_relu2_mul_max_abs,
    .residual_add             = shim_residual_add,
    .softmax                  = shim_softmax,
    .rope_apply               = shim_rope_apply,
    .tq2_quantize_vec_i8      = shim_tq2_quantize_vec_i8,
    .tq2_matmul_vector_lut              = shim_tq2_matmul_vector_lut,
    .tq2_matmul_vector_lut_scales       = shim_tq2_matmul_vector_lut_scales,
    .tq2_matmul_vector_lut_pair         = shim_tq2_matmul_vector_lut_pair,
    .tq2_matmul_vector_lut_pair_scales  = shim_tq2_matmul_vector_lut_pair_scales,
    .tq2_matmul_i2s_neon_parallel       = shim_tq2_matmul_i2s_neon_parallel,
    .tq2_matmul_i2s_neon_pair_parallel  = shim_tq2_matmul_i2s_neon_pair_parallel,
    .tq2_matmul_i2s_qkv_parallel        = shim_tq2_matmul_i2s_qkv_parallel,
    .q6k_dot_product_i8                 = shim_q6k_dot_product_i8,
    .q6k_dot_product_q8                 = shim_q6k_dot_product_q8,
    .q6k_dot_product_q8_4               = shim_q6k_dot_product_q8_4,
    .q6k_dot_product_q8_8               = shim_q6k_dot_product_q8_8,
    .q6k_dot_product_q8_compact         = shim_q6k_dot_product_q8_compact,
    .q6k_dot_product_q8_compact_4       = shim_q6k_dot_product_q8_compact_4,
    .q6k_dot_product_q8_compact_8       = shim_q6k_dot_product_q8_compact_8,
    .quantize_f32_to_i8                 = shim_quantize_f32_to_i8,
    .rms_norm_quant_tq2_i8              = shim_rms_norm_quant_tq2_i8,
    .residual_add_scaled                = shim_residual_add_scaled,
    .dot_i8                             = shim_dot_i8,
    .accum_i8_scaled                    = shim_accum_i8_scaled,
};
#endif

#if defined(__x86_64__) || defined(_M_X64)
/* x86 per-tier tables. Populated incrementally per phase — entries not yet
 * implemented remain NULL. Phase 2 brings rms_norm online. */

bitnet_dispatch_t g_dispatch_avx2 = {
    .tier                     = BITNET_TIER_AVX2,
    .rms_norm_eps             = bitnet_rms_norm_eps_avx2,
    .rms_norm_inplace_eps     = bitnet_rms_norm_inplace_eps_avx2,
    .silu                     = bitnet_silu_avx2,
    .silu_mul                 = bitnet_silu_mul_avx2,
    .silu_mul_max_abs         = bitnet_silu_mul_max_abs_avx2,
    .relu2_mul_max_abs        = bitnet_relu2_mul_max_abs_avx2,
    .residual_add             = bitnet_residual_add_avx2,
    .softmax                  = bitnet_softmax_avx2,
    .rope_apply               = bitnet_rope_apply_avx2,
    .tq2_quantize_vec_i8      = bitnet_tq2_0_quantize_vec_i8_avx2,
    .tq2_matmul_vector_lut              = bitnet_tq2_0_matmul_vector_lut_avx2,
    .tq2_matmul_vector_lut_scales       = bitnet_tq2_0_matmul_vector_lut_scales_avx2,
    .tq2_matmul_vector_lut_pair         = bitnet_tq2_0_matmul_vector_lut_pair_avx2,
    .tq2_matmul_vector_lut_pair_scales  = bitnet_tq2_0_matmul_vector_lut_pair_scales_avx2,
    .tq2_matmul_i2s_neon_parallel       = bitnet_tq2_0_matmul_i2s_neon_parallel_avx2,
    .tq2_matmul_i2s_neon_pair_parallel  = bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx2,
    .tq2_matmul_i2s_qkv_parallel        = bitnet_tq2_0_matmul_i2s_qkv_parallel_avx2,
    .q6k_dot_product_i8                 = bitnet_q6k_dot_product_i8_neon_avx2,
    .q6k_dot_product_q8                 = bitnet_q6k_dot_product_q8_avx2,
    .q6k_dot_product_q8_4               = bitnet_q6k_dot_product_q8_4_avx2,
    .q6k_dot_product_q8_8               = bitnet_q6k_dot_product_q8_8_avx2,
    .q6k_dot_product_q8_compact         = bitnet_q6k_dot_product_q8_compact_avx2,
    .q6k_dot_product_q8_compact_4       = bitnet_q6k_dot_product_q8_compact_4_avx2,
    .q6k_dot_product_q8_compact_8       = bitnet_q6k_dot_product_q8_compact_8_avx2,
    .quantize_f32_to_i8                 = bitnet_quantize_f32_to_i8_avx2,
    .rms_norm_quant_tq2_i8              = bitnet_rms_norm_quant_tq2_i8_avx2,
    .residual_add_scaled                = bitnet_residual_add_scaled_avx2,
    .dot_i8                             = bitnet_dot_i8_avx2,
    .accum_i8_scaled                    = bitnet_accum_i8_scaled_avx2,
};

bitnet_dispatch_t g_dispatch_avx_vnni = {
    .tier                     = BITNET_TIER_AVX_VNNI,
    .rms_norm_eps             = bitnet_rms_norm_eps_avx_vnni,
    .rms_norm_inplace_eps     = bitnet_rms_norm_inplace_eps_avx_vnni,
    .silu                     = bitnet_silu_avx_vnni,
    .silu_mul                 = bitnet_silu_mul_avx_vnni,
    .silu_mul_max_abs         = bitnet_silu_mul_max_abs_avx_vnni,
    .relu2_mul_max_abs        = bitnet_relu2_mul_max_abs_avx_vnni,
    .residual_add             = bitnet_residual_add_avx_vnni,
    .softmax                  = bitnet_softmax_avx_vnni,
    .rope_apply               = bitnet_rope_apply_avx_vnni,
    .tq2_quantize_vec_i8      = bitnet_tq2_0_quantize_vec_i8_avx_vnni,
    .tq2_matmul_vector_lut              = bitnet_tq2_0_matmul_vector_lut_avx_vnni,
    .tq2_matmul_vector_lut_scales       = bitnet_tq2_0_matmul_vector_lut_scales_avx_vnni,
    .tq2_matmul_vector_lut_pair         = bitnet_tq2_0_matmul_vector_lut_pair_avx_vnni,
    .tq2_matmul_vector_lut_pair_scales  = bitnet_tq2_0_matmul_vector_lut_pair_scales_avx_vnni,
    .tq2_matmul_i2s_neon_parallel       = bitnet_tq2_0_matmul_i2s_neon_parallel_avx_vnni,
    .tq2_matmul_i2s_neon_pair_parallel  = bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx_vnni,
    .tq2_matmul_i2s_qkv_parallel        = bitnet_tq2_0_matmul_i2s_qkv_parallel_avx_vnni,
    .q6k_dot_product_i8                 = bitnet_q6k_dot_product_i8_neon_avx_vnni,
    .q6k_dot_product_q8                 = bitnet_q6k_dot_product_q8_avx_vnni,
    .q6k_dot_product_q8_4               = bitnet_q6k_dot_product_q8_4_avx_vnni,
    .q6k_dot_product_q8_8               = bitnet_q6k_dot_product_q8_8_avx_vnni,
    .q6k_dot_product_q8_compact         = bitnet_q6k_dot_product_q8_compact_avx_vnni,
    .q6k_dot_product_q8_compact_4       = bitnet_q6k_dot_product_q8_compact_4_avx_vnni,
    .q6k_dot_product_q8_compact_8       = bitnet_q6k_dot_product_q8_compact_8_avx_vnni,
    .quantize_f32_to_i8                 = bitnet_quantize_f32_to_i8_avx_vnni,
    .rms_norm_quant_tq2_i8              = bitnet_rms_norm_quant_tq2_i8_avx_vnni,
    .residual_add_scaled                = bitnet_residual_add_scaled_avx_vnni,
    .dot_i8                             = bitnet_dot_i8_avx_vnni,
    .accum_i8_scaled                    = bitnet_accum_i8_scaled_avx_vnni,
};

bitnet_dispatch_t g_dispatch_avx512_vnni = {
    .tier                     = BITNET_TIER_AVX512_VNNI,
    .rms_norm_eps             = bitnet_rms_norm_eps_avx512_vnni,
    .rms_norm_inplace_eps     = bitnet_rms_norm_inplace_eps_avx512_vnni,
    .silu                     = bitnet_silu_avx512_vnni,
    .silu_mul                 = bitnet_silu_mul_avx512_vnni,
    .silu_mul_max_abs         = bitnet_silu_mul_max_abs_avx512_vnni,
    .relu2_mul_max_abs        = bitnet_relu2_mul_max_abs_avx512_vnni,
    .residual_add             = bitnet_residual_add_avx512_vnni,
    .softmax                  = bitnet_softmax_avx512_vnni,
    .rope_apply               = bitnet_rope_apply_avx512_vnni,
    .tq2_quantize_vec_i8      = bitnet_tq2_0_quantize_vec_i8_avx512_vnni,
    .tq2_matmul_vector_lut              = bitnet_tq2_0_matmul_vector_lut_avx512_vnni,
    .tq2_matmul_vector_lut_scales       = bitnet_tq2_0_matmul_vector_lut_scales_avx512_vnni,
    .tq2_matmul_vector_lut_pair         = bitnet_tq2_0_matmul_vector_lut_pair_avx512_vnni,
    .tq2_matmul_vector_lut_pair_scales  = bitnet_tq2_0_matmul_vector_lut_pair_scales_avx512_vnni,
    .tq2_matmul_i2s_neon_parallel       = bitnet_tq2_0_matmul_i2s_neon_parallel_avx512_vnni,
    .tq2_matmul_i2s_neon_pair_parallel  = bitnet_tq2_0_matmul_i2s_neon_pair_parallel_avx512_vnni,
    .tq2_matmul_i2s_qkv_parallel        = bitnet_tq2_0_matmul_i2s_qkv_parallel_avx512_vnni,
    .q6k_dot_product_i8                 = bitnet_q6k_dot_product_i8_neon_avx512_vnni,
    .q6k_dot_product_q8                 = bitnet_q6k_dot_product_q8_avx512_vnni,
    .q6k_dot_product_q8_4               = bitnet_q6k_dot_product_q8_4_avx512_vnni,
    .q6k_dot_product_q8_8               = bitnet_q6k_dot_product_q8_8_avx512_vnni,
    .q6k_dot_product_q8_compact         = bitnet_q6k_dot_product_q8_compact_avx512_vnni,
    .q6k_dot_product_q8_compact_4       = bitnet_q6k_dot_product_q8_compact_4_avx512_vnni,
    .q6k_dot_product_q8_compact_8       = bitnet_q6k_dot_product_q8_compact_8_avx512_vnni,
    .quantize_f32_to_i8                 = bitnet_quantize_f32_to_i8_avx512_vnni,
    .rms_norm_quant_tq2_i8              = bitnet_rms_norm_quant_tq2_i8_avx512_vnni,
    .residual_add_scaled                = bitnet_residual_add_scaled_avx512_vnni,
    .dot_i8                             = bitnet_dot_i8_avx512_vnni,
    .accum_i8_scaled                    = bitnet_accum_i8_scaled_avx512_vnni,
};
#endif
