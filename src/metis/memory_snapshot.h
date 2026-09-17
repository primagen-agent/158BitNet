#ifndef BITNET_MEMORY_SNAPSHOT_H
#define BITNET_MEMORY_SNAPSHOT_H
#include "episodic_store.h"
#include "event_store.h"
#include "resident_identity.h"

/* One writer per state directory. Immutable files plus an atomic manifest. */
int metis_memory_snapshot_save(const char *base,
    const metis_episodic_store_t *episodes, const metis_event_store_t *events);
/* Verifies whole-file hashes. Missing manifest falls back to legacy paths. */
int metis_memory_snapshot_paths(const char *base,
    char *episodes, size_t episodes_size, char *events, size_t events_size);
int metis_memory_snapshot_save_resident(const char *base,
    const metis_episodic_store_t *episodes, const metis_event_store_t *events,
    const resident_state_t *resident, const resident_model_t *model);
int metis_memory_snapshot_paths_resident(const char *base,
    char *episodes, size_t episodes_size, char *events, size_t events_size,
    char *resident, size_t resident_size);
#endif
