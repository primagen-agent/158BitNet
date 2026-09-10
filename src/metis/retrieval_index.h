#ifndef BITNET_RETRIEVAL_INDEX_H
#define BITNET_RETRIEVAL_INDEX_H

#include "memory_retriever.h"

#include <stddef.h>

typedef struct metis_retrieval_index {
    int rank;
    size_t count;
    size_t capacity;
    size_t *token_counts;
    float **global_keys;
    float **token_keys;
} metis_retrieval_index_t;

void metis_retrieval_index_init(metis_retrieval_index_t *index);
void metis_retrieval_index_clear(metis_retrieval_index_t *index);
int metis_retrieval_index_configure(
    metis_retrieval_index_t *index, int rank);
int metis_retrieval_index_set(
    metis_retrieval_index_t *index, size_t record_index,
    const float *global_key, const float *token_keys, size_t token_count);
int metis_retrieval_index_complete(
    const metis_retrieval_index_t *index, size_t record_count);
int metis_retrieval_index_rank(
    const metis_retrieval_index_t *index,
    const metis_memory_retriever_t *retriever,
    const char *const *records, const float *priorities,
    size_t record_count, const char *query, float priority_weight,
    const float *query_global, const float *query_tokens,
    size_t query_token_count, size_t *indices, int max_results);

#endif
