#include "metis/episodic_store.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        failed = 1; \
        goto cleanup; \
    } \
} while (0)

int main(void) {
    const char *state_path = "/tmp/test-typed-memory.bnepisodic";
    const char *bad_path = "/tmp/test-typed-memory-bad.bnepisodic";
    metis_episodic_store_t store;
    metis_episodic_store_t loaded;
    FILE *bad_file = NULL;
    int failed = 0;

    metis_episodic_init(&store);
    metis_episodic_init(&loaded);
    CHECK(
        metis_episodic_add_with_priority(
            &store, "Briar uses a virtual ticket.", NULL, 0.4f) == 0,
        "add first source record");
    CHECK(
        metis_episodic_add_with_priority(
            &store, "Briar uses a virtual ticket.", NULL, 0.8f) == 0,
        "deduplicate source record");
    CHECK(store.count == 1, "duplicate record count");
    CHECK(fabsf(store.priorities[0] - 0.8f) < 1e-6f,
          "duplicate keeps higher priority");
    CHECK(
        metis_episodic_add_with_priority(
            &store, "Cleo follows radio bulletins.", NULL, 0.2f) == 0,
        "add second source record");
    CHECK(metis_episodic_count(&store) == 2, "source record count");
    size_t source_index = 99;
    CHECK(metis_episodic_add_indexed(&store,
          "Briar uses a virtual ticket.", NULL, 0.4f, &source_index) == 0,
          "repeat non-final source");
    CHECK(source_index == 0 && store.count == 2,
          "deduplication returns the original index");
    CHECK(metis_episodic_add_indexed(&store,
          "A rejected source.", NULL, 0.0f, &source_index) == 0,
          "append source before event compilation");
    metis_episodic_truncate(&store, 2);
    CHECK(store.count == 2, "rollback failed event source");
    CHECK(metis_episodic_save(&store, state_path) == 0, "save state");
    CHECK(metis_episodic_load(&loaded, state_path) == 0, "load state");
    CHECK(loaded.count == 2, "round-trip count");
    CHECK(strcmp(loaded.records[0], store.records[0]) == 0,
          "round-trip first record");
    CHECK(strcmp(loaded.records[1], store.records[1]) == 0,
          "round-trip second record");

    bad_file = fopen(bad_path, "wb");
    CHECK(bad_file != NULL, "create malformed state");
    CHECK(fwrite("bad", 1, 3, bad_file) == 3, "write malformed state");
    CHECK(fclose(bad_file) == 0, "close malformed state");
    bad_file = NULL;
    CHECK(metis_episodic_load(&loaded, bad_path) != 0,
          "reject malformed state");
    CHECK(loaded.count == 2, "failed import leaves state unchanged");

cleanup:
    if (bad_file != NULL) fclose(bad_file);
    remove(state_path);
    remove(bad_path);
    metis_episodic_free(&loaded);
    metis_episodic_free(&store);
    if (!failed) puts("test_episodic_store: OK");
    return failed;
}
