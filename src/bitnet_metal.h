#ifndef BITNET_METAL_H
#define BITNET_METAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitnet_metal_output bitnet_metal_output_t;
typedef struct bitnet_metal_i2s_context bitnet_metal_i2s_context_t;
typedef struct bitnet_metal_i2s_tensor bitnet_metal_i2s_tensor_t;

int bitnet_metal_available(void);

int bitnet_metal_output_create_q6k_compact(const int8_t *output_q8,
                                           const int8_t *output_scales,
                                           const float *output_d,
                                           int vocab_size,
                                           int emb_dim,
                                           bitnet_metal_output_t **out);

int bitnet_metal_output_compute_q6k_compact(bitnet_metal_output_t *state,
                                            const int8_t *qhidden,
                                            float hidden_scale,
                                            float *logits);

void bitnet_metal_output_free(bitnet_metal_output_t *state);

int bitnet_metal_i2s_create(bitnet_metal_i2s_context_t **out);

int bitnet_metal_i2s_tensor_create(bitnet_metal_i2s_context_t *ctx,
                                   const uint8_t *packed,
                                   const float *scales,
                                   int out_dim,
                                   int in_dim,
                                   bitnet_metal_i2s_tensor_t **out);

int bitnet_metal_i2s_compute(bitnet_metal_i2s_context_t *ctx,
                             bitnet_metal_i2s_tensor_t *tensor,
                             const int8_t *qvec,
                             float vec_scale,
                             float *out);

int bitnet_metal_i2s_compute_pair(bitnet_metal_i2s_context_t *ctx,
                                  bitnet_metal_i2s_tensor_t *tensor_a,
                                  bitnet_metal_i2s_tensor_t *tensor_b,
                                  const int8_t *qvec,
                                  float vec_scale,
                                  float *out_a,
                                  float *out_b);

void bitnet_metal_i2s_tensor_free(bitnet_metal_i2s_tensor_t *tensor);
void bitnet_metal_i2s_free(bitnet_metal_i2s_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
