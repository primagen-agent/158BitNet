#ifndef BITNET_MEMORY_SNAPSHOT_H
#define BITNET_MEMORY_SNAPSHOT_H
#include "episodic_store.h"
#include "event_store.h"

/* One writer per state directory. Immutable files plus an atomic manifest. */
int metis_memory_snapshot_save(const char *base,
    const metis_episodic_store_t *episodes, const metis_event_store_t *events);
/* Verifies whole-file hashes. Missing manifest falls back to legacy paths. */
int metis_memory_snapshot_paths(const char *base,
    char *episodes, size_t episodes_size, char *events, size_t events_size);
#endif
