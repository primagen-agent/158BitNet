#ifndef BITNET_EVENT_STORE_H
#define BITNET_EVENT_STORE_H

#include <stddef.h>

#define METIS_EVENT_SPAN_UNKNOWN ((size_t)-1)

typedef enum metis_event_operation {
    METIS_EVENT_ASSERT = 0,
    METIS_EVENT_SUPERSEDE = 1,
    METIS_EVENT_RETRACT = 2
} metis_event_operation_t;

typedef enum metis_memory_kind {
    METIS_MEMORY_PROPERTY = 0,
    METIS_MEMORY_SET = 1,
    METIS_MEMORY_EVENT = 2
} metis_memory_kind_t;

typedef enum metis_event_polarity {
    METIS_EVENT_POSITIVE = 0,
    METIS_EVENT_NEGATIVE = 1
} metis_event_polarity_t;

typedef enum metis_event_modality {
    METIS_EVENT_ACTUAL = 0,
    METIS_EVENT_PLANNED = 1,
    METIS_EVENT_POSSIBLE = 2
} metis_event_modality_t;

typedef struct metis_event_record {
    char *event_id;
    char *episode_id;
    char *source_id;
    char *entity;
    char *predicate;
    char *value;
    char *valid_time;
    char *target_event_id;
    metis_event_operation_t operation;
    metis_memory_kind_t memory_kind;
    metis_event_polarity_t polarity;
    metis_event_modality_t modality;
    int active;
    size_t raw_record_index;
    size_t subject_start;
    size_t subject_end;
    size_t value_start;
    size_t value_end;
} metis_event_record_t;

typedef struct metis_event_store {
    metis_event_record_t *events;
    size_t count;
    size_t capacity;
} metis_event_store_t;

void metis_event_store_init(metis_event_store_t *store);
void metis_event_store_clear(metis_event_store_t *store);

/* Supersede/retract require an existing active target_event_id. The source
 * event remains immutable; only its derived active flag changes. */
int metis_event_store_apply(
    metis_event_store_t *store, const metis_event_record_t *event);
const metis_event_record_t *metis_event_store_find(
    const metis_event_store_t *store, const char *event_id);
const metis_event_record_t *metis_event_store_current(
    const metis_event_store_t *store,
    const char *entity, const char *predicate);
size_t metis_event_store_active_count(
    const metis_event_store_t *store,
    const char *entity, const char *predicate);
const metis_event_record_t *metis_event_store_active_at(
    const metis_event_store_t *store,
    const char *entity, const char *predicate, size_t ordinal);

int metis_event_store_save(
    const metis_event_store_t *store, const char *path);
int metis_event_store_load(
    metis_event_store_t *store, const char *path);

#endif
