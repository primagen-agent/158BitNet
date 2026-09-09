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
/* Addressed long-term-memory operations. Keys are expected to be L2
 * normalized. UPDATE replaces the closest active record when its cosine
 * similarity reaches min_similarity, otherwise it appends a new record.
 * DELETE removes the closest matching record. replaced/deleted may be NULL. */
int metis_episodic_upsert_with_key(
    metis_episodic_store_t *store, const char *text, const float *key,
    float min_similarity, size_t *replaced);
int metis_episodic_delete_with_key(
    metis_episodic_store_t *store, const float *key,
    float min_similarity, size_t *deleted);
/* Replace the closest record with an explicit deletion instruction. Unlike
 * physical deletion, the tombstone remains retrievable so the generator can
 * distinguish "forgotten" from "never mentioned" and avoid hallucinating a
 * stale/default value. */
int metis_episodic_tombstone_with_key(
    metis_episodic_store_t *store, const char *text, const float *key,
    float min_similarity, size_t *replaced);
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
