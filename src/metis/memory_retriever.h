#ifndef BITNET_MEMORY_RETRIEVER_H
#define BITNET_MEMORY_RETRIEVER_H

#include <stddef.h>
#include <stdint.h>

typedef struct metis_memory_retriever {
    int hidden_dim;
    int rank;
    float late_weight;
    float max_residual;
    uint8_t backbone_sha256[32];
    float *query_global;
    float *entry_global;
    float *query_token;
    float *entry_token;
} metis_memory_retriever_t;

metis_memory_retriever_t *metis_memory_retriever_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size);
void metis_memory_retriever_free(metis_memory_retriever_t *retriever);

/* hidden is row-major [tokens][hidden_dim]. Outputs are normalized. */
int metis_memory_retriever_encode(
    const metis_memory_retriever_t *retriever,
    const float *hidden, size_t tokens, int is_query,
    float *global_key, float *token_keys);
float metis_memory_retriever_score(
    const metis_memory_retriever_t *retriever,
    const float *query_global, const float *query_tokens, size_t query_count,
    const float *entry_global, const float *entry_tokens, size_t entry_count);
float metis_memory_retriever_residual(
    const metis_memory_retriever_t *retriever, float semantic_score);

#endif
