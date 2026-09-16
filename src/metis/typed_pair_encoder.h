#ifndef BITNET_TYPED_PAIR_ENCODER_H
#define BITNET_TYPED_PAIR_ENCODER_H

#include <stddef.h>
#include <stdint.h>

typedef struct metis_typed_pair_encoder {
    int hidden_dim;
    int rank;
    int band_count;
    int layer_count;
    int pair_width;
    int head_width;
    uint32_t *band_start;
    uint32_t *band_end;
    uint8_t backbone_sha256[32];

    float *band_logits;
    float *projection[2];
    float *identity_projection[2];
    float *localizer_band_logits;
    float *localizer_projection[2];
    float *token_keys;
    float *fusion_hidden_weight[2];
    float *fusion_hidden_bias[2];
    float *fusion_output_weight[2];
    float fusion_output_bias[2];
    float *head_hidden_weight[2];
    float *head_hidden_bias[2];
    float *head_output_weight[2];
    float head_output_bias[2];
} metis_typed_pair_encoder_t;

metis_typed_pair_encoder_t *metis_typed_pair_encoder_load(
    const char *path, const char *backbone_path,
    int expected_hidden, char *error, size_t error_size);

void metis_typed_pair_encoder_free(
    metis_typed_pair_encoder_t *model);

/*
 * Hidden inputs use [token][band][hidden]. Identity inputs are the matching
 * token-embedding rows [token][hidden]. Outputs are the fused entity and
 * predicate pair features consumed by `.bntlink`, plus their verifier logits.
 */
int metis_typed_pair_encoder_score(
    const metis_typed_pair_encoder_t *model,
    const float *left_hidden, size_t left_count,
    const float *left_identity,
    const float *right_hidden, size_t right_count,
    const float *right_identity,
    float *entity_feature,
    float *predicate_feature,
    float *entity_logit,
    float *predicate_logit);

#endif
