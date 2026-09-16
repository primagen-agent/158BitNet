#include "metis/memory_snapshot.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REQUIRE(test) do { if (!(test)) { fprintf(stderr, "snapshot failure at %d\n", __LINE__); return 1; } } while (0)

int main(void) {
    const char *base = "test-memory-snapshot";
    metis_episodic_store_t episodes, loaded;
    metis_event_store_t events;
    char ep[1200], ev[1200], original_ep[1200], original_ev[1200];
    metis_episodic_init(&episodes);
    metis_episodic_init(&loaded);
    metis_event_store_init(&events);
    metis_event_record_t item = {
        .event_id="one", .episode_id="one", .source_id="one",
        .entity="Alice", .predicate="lives_in", .value="Tokyo",
        .valid_time="", .target_event_id="", .operation=METIS_EVENT_ASSERT,
        .memory_kind=METIS_MEMORY_PROPERTY, .polarity=METIS_EVENT_POSITIVE,
        .modality=METIS_EVENT_ACTUAL, .raw_record_index=0,
        .subject_start=0, .subject_end=5, .value_start=15, .value_end=20,
    };
    REQUIRE(metis_episodic_add_with_priority(&episodes, "Alice lives in Tokyo", NULL, 0) == 0);
    REQUIRE(metis_event_store_apply(&events, &item) == 0);
    REQUIRE(metis_memory_snapshot_save(base, &episodes, &events) == 0);
    REQUIRE(metis_memory_snapshot_paths(base, original_ep, sizeof original_ep,
                                      original_ev, sizeof original_ev) == 0);
    REQUIRE(metis_episodic_add_with_priority(&episodes, "Another fact", NULL, 0) == 0);
    /* The second writer fails after the first file was successfully written. */
    events.events[0].raw_record_index = SIZE_MAX;
    REQUIRE(metis_memory_snapshot_save(base, &episodes, &events) != 0);
    REQUIRE(metis_memory_snapshot_paths(base, ep, sizeof ep, ev, sizeof ev) == 0);
    REQUIRE(strcmp(ep, original_ep) == 0 && strcmp(ev, original_ev) == 0);
    REQUIRE(metis_episodic_load(&loaded, ep) == 0 && loaded.count == 1);
    events.events[0].raw_record_index = 0;
    REQUIRE(metis_memory_snapshot_save(base, &episodes, &events) == 0);
    REQUIRE(metis_memory_snapshot_paths(base, ep, sizeof ep, ev, sizeof ev) == 0);
    REQUIRE(metis_episodic_load(&loaded, ep) == 0 && loaded.count == 2);
    /* Protect numeric metadata too, not just the existing per-string CRCs. */
    FILE *file = fopen(ev, "rb+");
    REQUIRE(file != NULL && fseek(file, 20, SEEK_SET) == 0);
    REQUIRE(fputc(99, file) != EOF && fclose(file) == 0);
    REQUIRE(metis_memory_snapshot_paths(base, ep, sizeof ep, ev, sizeof ev) != 0);
    remove(original_ep); remove(original_ev); remove(ep); remove(ev);
    remove("test-memory-snapshot.bnsnapshot");
    metis_episodic_free(&loaded);
    metis_episodic_free(&episodes);
    metis_event_store_clear(&events);
    puts("test_memory_snapshot: OK");
    return 0;
}
