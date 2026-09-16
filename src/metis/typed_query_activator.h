#ifndef BITNET_TYPED_QUERY_ACTIVATOR_H
#define BITNET_TYPED_QUERY_ACTIVATOR_H

#include <stddef.h>
#include <stdint.h>

typedef struct metis_typed_query_activator {
    int rank;
    int input_width;
    int candidate_hidden;
    int set_feature_count;
    int null_hidden;
    uint8_t backbone_sha256[32];
    uint8_t pair_model_sha256[32];

    float *candidate_hidden_weight;
    float *candidate_hidden_bias;
    float *candidate_output_weight;
    float candidate_output_bias;
    float *null_hidden_weight;
    float *null_hidden_bias;
    float *null_output_weight;
    float null_output_bias;
} metis_typed_query_activator_t;

metis_typed_query_activator_t *
metis_typed_query_activator_load(
    const char *path,
    const char *backbone_path,
    const char *pair_model_path,
    int expected_input_width,
    char *error,
    size_t error_size);

void metis_typed_query_activator_free(
    metis_typed_query_activator_t *model);

/*
 * Query/event pair features come from `.bntpair`. The dedicated candidate
 * head ranks active events, while the candidate-set-aware NULL head decides
 * whether any event should activate.
 */
int metis_typed_query_select(
    const metis_typed_query_activator_t *model,
    const float *pair_features,
    const float *entity_logits,
    const float *predicate_logits,
    size_t pair_count,
    size_t *selected_index,
    float *activation_score,
    float *null_score);

#endif
