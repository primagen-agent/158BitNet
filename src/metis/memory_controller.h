#ifndef BITNET_MEMORY_CONTROLLER_H
#define BITNET_MEMORY_CONTROLLER_H

#include <stddef.h>
#include <stdint.h>

typedef struct metis_memory_controller {
    int hidden_dim;
    int rank;
    int pooling; /* 1=last, 2=mean+last */
    float temperature;
    float address_threshold;
    uint8_t backbone_sha256[32];
    float *query_projection; /* [rank][hidden_dim] */
    float *entry_projection; /* [rank][hidden_dim] */
    /* BNCTRL3 operation router: IGNORE, WRITE, UPDATE, DELETE. */
    float *action_projection; /* [4][hidden_dim] */
    float action_bias[4];
} metis_memory_controller_t;

metis_memory_controller_t *metis_memory_controller_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size);
void metis_memory_controller_free(metis_memory_controller_t *controller);

/* Projects and L2-normalizes a pooled backbone hidden vector. */
int metis_memory_controller_project(
    const metis_memory_controller_t *controller, const float *hidden,
    int is_query, float *output);
float metis_memory_controller_score(
    const metis_memory_controller_t *controller,
    const float *query_key, const float *entry_key);
/* Returns action id 0=IGNORE, 1=WRITE, 2=UPDATE, 3=DELETE.
 * Legacy controllers without an action head return -1. */
int metis_memory_controller_classify(
    const metis_memory_controller_t *controller, const float *hidden,
    float *confidence);

#endif
