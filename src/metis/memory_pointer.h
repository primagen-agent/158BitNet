#ifndef BITNET_MEMORY_POINTER_H
#define BITNET_MEMORY_POINTER_H

#include <stddef.h>
#include <stdint.h>

typedef struct metis_memory_pointer {
    int version;
    int hidden_dim;
    int max_span;
    float threshold;
    uint8_t backbone_sha256[32];
    float *start_weight;
    float start_bias;
    float *end_weight;
    float end_bias;
    int has_null;
    float *null_weight; /* [2][2 * hidden_dim] */
    float null_bias[2];
    int byte_mode;
    int rank;
    int max_token_bytes;
    float *token_weight;         /* [rank][hidden_dim] */
    float *byte_weight;          /* [257][rank] */
    float *previous_byte_weight; /* [257][rank] */
    float *next_byte_weight;     /* [257][rank] */
    float *offset_weight;        /* [max_token_bytes][rank] */
    float *inside_weight;        /* [rank] */
    float inside_bias;
} metis_memory_pointer_t;

metis_memory_pointer_t *metis_memory_pointer_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size);
void metis_memory_pointer_free(metis_memory_pointer_t *pointer);

/*
 * Selects the highest-scoring inclusive token span. Confidence is the score
 * margin over the second-best valid span. Returns 1 when it meets the model's
 * threshold, 0 when it is below threshold, and -1 on invalid input.
 */
int metis_memory_pointer_select(
    const metis_memory_pointer_t *pointer, const float *hidden,
    size_t token_count, size_t *start, size_t *end, float *confidence);

/*
 * Selects an inclusive byte span for a BNPTR5 write extractor. Each byte
 * references the record token whose hidden state produced it and its offset
 * inside that decoded token. The selected bytes are copied verbatim.
 */
int metis_memory_pointer_select_bytes(
    const metis_memory_pointer_t *pointer, const float *hidden,
    size_t token_count, const uint8_t *bytes,
    const size_t *byte_token_indices, const uint32_t *byte_token_offsets,
    size_t byte_count, size_t *start, size_t *end, float *confidence);

#endif
