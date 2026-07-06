#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

void bitnet_rms_norm_eps(float *x, const float *weight, int n, float eps);
void bitnet_rms_norm_inplace_eps(float *dst, const float *src, const float *weight, int n, float eps);
/* Internal impl symbols (selected at compile time by __ARM_NEON). Dispatch
 * table shims call these directly to avoid infinite recursion through the
 * public trampolines. */
void bitnet_rms_norm_eps_impl(float *x, const float *weight, int n, float eps);
void bitnet_rms_norm_inplace_eps_impl(float *dst, const float *src, const float *weight, int n, float eps);
void bitnet_rms_norm(float *x, const float *weight, int n);
void bitnet_rms_norm_inplace(float *dst, const float *src, const float *weight, int n);
void bitnet_matmul_f32(const float *a, const float *b, float *c, int m, int k, int n);
void bitnet_silu(float *x, int n);
void bitnet_silu_mul(float *gate, const float *up, int n);
float bitnet_silu_mul_max_abs(float *gate, const float *up, int n);
float bitnet_relu2_mul_max_abs(float *gate, const float *up, int n);
void bitnet_softmax(float *x, int n);
void bitnet_residual_add(float *out, const float *a, const float *b, int n);
void bitnet_rope_apply(float *x, int n_heads, int head_dim, int rope_dim,
                       const float *rope_cos, const float *rope_sin);

/* Internal impl symbols for SiLU / residual / softmax / rope family
 * (selected at compile time by __ARM_NEON). Dispatch shims call these
 * directly to avoid infinite recursion through the public trampolines. */
void  bitnet_silu_impl(float *x, int n);
void  bitnet_silu_mul_impl(float *gate, const float *up, int n);
float bitnet_silu_mul_max_abs_impl(float *gate, const float *up, int n);
float bitnet_relu2_mul_max_abs_impl(float *gate, const float *up, int n);
void  bitnet_residual_add_impl(float *out, const float *a, const float *b, int n);
void  bitnet_softmax_impl(float *x, int n);
void  bitnet_rope_apply_impl(float *x, int n_heads, int head_dim, int rope_dim,
                              const float *rope_cos, const float *rope_sin);

#ifdef __cplusplus
}
#endif

#endif
