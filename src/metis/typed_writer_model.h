#ifndef BITNET_TYPED_WRITER_MODEL_H
#define BITNET_TYPED_WRITER_MODEL_H

#include <stddef.h>
#include <stdint.h>

enum {
    METIS_TYPED_WRITER_FIELD_COUNT = 4,
    METIS_TYPED_WRITER_ENTITY = 0,
    METIS_TYPED_WRITER_PREDICATE = 1,
    METIS_TYPED_WRITER_VALUE = 2,
    METIS_TYPED_WRITER_TIME = 3
};

typedef struct metis_typed_writer_field {
    float *context_weight;
    float *context_bias;
    float *output_weight;
    float output_bias;
    float *length_logits;
    float residual_scale;
} metis_typed_writer_field_t;

typedef struct metis_typed_writer_model {
    int hidden_dim;
    int rank;
    int band_count;
    int layer_count;
    int max_span;
    uint32_t *band_start;
    uint32_t *band_end;
    uint8_t backbone_sha256[32];
    float predicate_anchor_weight;
    float value_anchor_weight;

    float *band_logits;
    float *projection[METIS_TYPED_WRITER_FIELD_COUNT];
    float *localizer_band_logits;
    float *localizer_projection[2];
    float *token_keys;
    float *span_boundary_keys;
    float *operation_hidden_weight;
    float *operation_hidden_bias;
    float *operation_output_weight;
    float *operation_output_bias;
    float *adapted_anchor_keys;
    metis_typed_writer_field_t
        field[METIS_TYPED_WRITER_FIELD_COUNT];
} metis_typed_writer_model_t;

typedef struct metis_typed_writer_result {
    int operation;
    float operation_logits[2];
    size_t span_start[METIS_TYPED_WRITER_FIELD_COUNT];
    size_t span_end[METIS_TYPED_WRITER_FIELD_COUNT];
    size_t anchor[METIS_TYPED_WRITER_FIELD_COUNT];
} metis_typed_writer_result_t;

metis_typed_writer_model_t *metis_typed_writer_model_load(
    const char *path, const char *backbone_path,
    int expected_hidden, char *error, size_t error_size);

void metis_typed_writer_model_free(
    metis_typed_writer_model_t *model);

/*
 * Hidden is [token][band][hidden], identity is [token][hidden], and output
 * spans are inclusive token indices. The caller maps them back to source
 * UTF-8 byte spans with the exact backbone tokenizer.
 */
int metis_typed_writer_predict(
    const metis_typed_writer_model_t *model,
    const float *hidden, const float *identity,
    size_t token_count, metis_typed_writer_result_t *output);

#endif
