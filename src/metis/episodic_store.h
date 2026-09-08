#ifndef BITNET_EPISODIC_STORE_H
#define BITNET_EPISODIC_STORE_H

#include <stddef.h>

typedef struct metis_episodic_store {
    char **records;
    float **keys;
    size_t count;
    size_t capacity;
    int key_dim;
} metis_episodic_store_t;

void metis_episodic_init(metis_episodic_store_t *store);
void metis_episodic_clear(metis_episodic_store_t *store);
void metis_episodic_free(metis_episodic_store_t *store);
int metis_episodic_add(metis_episodic_store_t *store, const char *text);
int metis_episodic_configure_keys(
    metis_episodic_store_t *store, int key_dim);
int metis_episodic_add_with_key(
    metis_episodic_store_t *store, const char *text, const float *key);
size_t metis_episodic_count(const metis_episodic_store_t *store);

/* Returns a newly allocated, newline-delimited exact-record context ordered
 * by lexical relevance and recency. NULL means no record matched. */
char *metis_episodic_build_context(
    const metis_episodic_store_t *store, const char *query, int top_k);
char *metis_episodic_build_hybrid_context(
    const metis_episodic_store_t *store, const char *query,
    const float *query_key, float lexical_weight, int top_k);
int metis_episodic_rank_hybrid(
    const metis_episodic_store_t *store, const char *query,
    const float *query_key, float lexical_weight,
    size_t *indices, int max_results);

/* Atomic-at-object-level load: a malformed file leaves store unchanged. */
int metis_episodic_save(
    const metis_episodic_store_t *store, const char *path);
int metis_episodic_load(
    metis_episodic_store_t *store, const char *path);

#endif
