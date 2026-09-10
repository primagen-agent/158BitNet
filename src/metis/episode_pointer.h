#ifndef BITNET_EPISODE_POINTER_H
#define BITNET_EPISODE_POINTER_H

#include <stddef.h>
#include <stdint.h>

typedef struct metis_episode_pointer {
    int hidden_dim;
    int rank;
    int max_span;
    uint8_t backbone_sha256[32];
    float *query_start;
    float *query_end;
    float *source_start;
    float *source_end;
    float *local_start_weight;
    float local_start_bias;
    float *local_end_weight;
    float local_end_bias;
    float local_scale[2];
} metis_episode_pointer_t;

metis_episode_pointer_t *metis_episode_pointer_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size);
void metis_episode_pointer_free(metis_episode_pointer_t *pointer);

/*
 * Select an inclusive answer span inside one already activated episode.
 * Source and query hidden states are encoded independently. This function
 * never searches other episodes and never injects source text into a prompt.
 */
int metis_episode_pointer_select(
    const metis_episode_pointer_t *pointer,
    const float *source_hidden, size_t source_count,
    const float *query_hidden, size_t query_count,
    size_t *start, size_t *end, float *score);

#endif
