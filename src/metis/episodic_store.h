#ifndef BITNET_EPISODIC_STORE_H
#define BITNET_EPISODIC_STORE_H

#include <stddef.h>

typedef struct metis_episodic_store {
    char **records;
    float **keys;
    float *priorities;
    size_t count;
    size_t capacity;
    int key_dim;
} metis_episodic_store_t;

void metis_episodic_init(metis_episodic_store_t *store);
void metis_episodic_clear(metis_episodic_store_t *store);
void metis_episodic_free(metis_episodic_store_t *store);
int metis_episodic_add_with_priority(
    metis_episodic_store_t *store, const char *text, const float *key,
    float priority);
/* Returns the actual source index, including when text is deduplicated. */
int metis_episodic_add_indexed(
    metis_episodic_store_t *store, const char *text, const float *key,
    float priority, size_t *record_index);
/* Roll back newly appended records after a failed event compilation. */
void metis_episodic_truncate(metis_episodic_store_t *store, size_t count);
size_t metis_episodic_count(const metis_episodic_store_t *store);

/* Atomic-at-object-level load: a malformed file leaves store unchanged. */
int metis_episodic_save(
    const metis_episodic_store_t *store, const char *path);
int metis_episodic_load(
    metis_episodic_store_t *store, const char *path);

#endif
