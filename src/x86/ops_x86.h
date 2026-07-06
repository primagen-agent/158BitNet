#ifndef BITNET_OPS_X86_H
#define BITNET_OPS_X86_H

void bitnet_rms_norm_eps_avx2(float *x, const float *weight, int n, float eps);
void bitnet_rms_norm_eps_avx_vnni(float *x, const float *weight, int n, float eps);
void bitnet_rms_norm_eps_avx512_vnni(float *x, const float *weight, int n, float eps);

void bitnet_rms_norm_inplace_eps_avx2(float *dst, const float *src,
                                       const float *weight, int n, float eps);
void bitnet_rms_norm_inplace_eps_avx_vnni(float *dst, const float *src,
                                           const float *weight, int n, float eps);
void bitnet_rms_norm_inplace_eps_avx512_vnni(float *dst, const float *src,
                                              const float *weight, int n, float eps);

void  bitnet_silu_avx2(float *x, int n);
void  bitnet_silu_avx_vnni(float *x, int n);
void  bitnet_silu_avx512_vnni(float *x, int n);

void  bitnet_silu_mul_avx2(float *gate, const float *up, int n);
void  bitnet_silu_mul_avx_vnni(float *gate, const float *up, int n);
void  bitnet_silu_mul_avx512_vnni(float *gate, const float *up, int n);

float bitnet_silu_mul_max_abs_avx2(float *gate, const float *up, int n);
float bitnet_silu_mul_max_abs_avx_vnni(float *gate, const float *up, int n);
float bitnet_silu_mul_max_abs_avx512_vnni(float *gate, const float *up, int n);

float bitnet_relu2_mul_max_abs_avx2(float *gate, const float *up, int n);
float bitnet_relu2_mul_max_abs_avx_vnni(float *gate, const float *up, int n);
float bitnet_relu2_mul_max_abs_avx512_vnni(float *gate, const float *up, int n);

void  bitnet_residual_add_avx2(float *out, const float *a, const float *b, int n);
void  bitnet_residual_add_avx_vnni(float *out, const float *a, const float *b, int n);
void  bitnet_residual_add_avx512_vnni(float *out, const float *a, const float *b, int n);

void  bitnet_softmax_avx2(float *x, int n);
void  bitnet_softmax_avx_vnni(float *x, int n);
void  bitnet_softmax_avx512_vnni(float *x, int n);

void  bitnet_rope_apply_avx2(float *x, int n_heads, int head_dim, int rope_dim,
                              const float *rope_cos, const float *rope_sin);
void  bitnet_rope_apply_avx_vnni(float *x, int n_heads, int head_dim, int rope_dim,
                                  const float *rope_cos, const float *rope_sin);
void  bitnet_rope_apply_avx512_vnni(float *x, int n_heads, int head_dim, int rope_dim,
                                     const float *rope_cos, const float *rope_sin);

#endif
