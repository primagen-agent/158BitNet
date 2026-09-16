#ifndef BITNET_TYPED_LINK_MODEL_H
#define BITNET_TYPED_LINK_MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct metis_typed_link_model {
    int rank;
    int pair_feature_dim;
    int joint_hidden_dim;
    uint8_t backbone_sha256[32];

    float *joint_hidden_weight;
    float *joint_hidden_bias;
    float *joint_output_weight;
    float joint_output_bias;
    float *exists_hidden_weight;
    float *exists_hidden_bias;
    float *exists_output_weight;
    float exists_output_bias;
} metis_typed_link_model_t;

metis_typed_link_model_t *metis_typed_link_model_load(
    const char *path, const char *backbone_path,
    char *error, size_t error_size);

void metis_typed_link_model_free(
    metis_typed_link_model_t *model);

/*
 * Experimental V251 component boundary. Entity/predicate pair features and
 * their logits are produced by the frozen token-level verifier. This module
 * computes the learned residual joint score without reading episode text.
 */
int metis_typed_link_score_pairs(
    const metis_typed_link_model_t *model,
    const float *entity_features,
    const float *predicate_features,
    const float *entity_logits,
    const float *predicate_logits,
    size_t pair_count,
    float *joint_scores);

/*
 * Score whether a candidate set contains a predecessor. Candidate ordering
 * does not affect the result. A positive score accepts the highest joint
 * score; a non-positive score rejects the complete set.
 */
int metis_typed_link_predecessor_exists(
    const metis_typed_link_model_t *model,
    const float *joint_scores,
    size_t pair_count,
    float *exists_score);

int metis_typed_link_select_predecessor(
    const metis_typed_link_model_t *model,
    const float *joint_scores,
    size_t pair_count,
    size_t *selected_index,
    float *exists_score);

#endif
