#include "metis/episodic_store.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        ++failures; \
    } \
} while (0)

int main(void) {
    metis_episodic_store_t store;
    metis_episodic_store_t loaded;
    char *context;
    metis_episodic_init(&store);
    CHECK(metis_episodic_configure_keys(&store, 2) == 0,
          "configure empty addressed store");
    {
        const float empty_delete_key[2] = {1.0f, 0.0f};
        size_t changed = SIZE_MAX;
        CHECK(metis_episodic_tombstone_with_key(
                  &store, "Forget an absent value.",
                  empty_delete_key, 0.9f, &changed) == 0,
              "empty addressed tombstone is storable");
        CHECK(metis_episodic_count(&store) == 1,
              "empty addressed tombstone creates one record");
        metis_episodic_clear(&store);
        CHECK(metis_episodic_configure_keys(&store, 2) == 0,
              "reconfigure addressed store after clear");
    }
    metis_episodic_init(&loaded);

    CHECK(metis_episodic_add(
              &store, "Mira's locker code is amber-4172.") == 0,
          "add first record");
    CHECK(metis_episodic_add(
              &store, "Jonah's travel date is Saturday morning.") == 0,
          "add second record");
    CHECK(metis_episodic_add(
              &store, "Mira's locker code is blue-9021 now.") == 0,
          "add update record");
    CHECK(metis_episodic_add(
              &store, "Mira's locker code is blue-9021 now.") == 0,
          "duplicate record accepted as no-op");
    CHECK(metis_episodic_count(&store) == 3, "duplicate not stored");
    CHECK(metis_episodic_configure_keys(&store, 2) == 0,
          "configure semantic key dimension");
    {
        const float old_key[2] = {0.8f, 0.2f};
        const float other_key[2] = {0.0f, 1.0f};
        const float new_key[2] = {1.0f, 0.0f};
        const float query_key[2] = {1.0f, 0.0f};
        const float update_key[2] = {0.99f, 0.01f};
        const float delete_key[2] = {0.0f, 1.0f};
        size_t changed = SIZE_MAX;
        CHECK(metis_episodic_add_with_key(
                  &store, "Mira's locker code is amber-4172.",
                  old_key) == 0,
              "attach old semantic key");
        CHECK(metis_episodic_add_with_key(
                  &store, "Jonah's travel date is Saturday morning.",
                  other_key) == 0,
              "attach unrelated semantic key");
        CHECK(metis_episodic_add_with_key(
                  &store, "Mira's locker code is blue-9021 now.",
                  new_key) == 0,
              "attach current semantic key");
        context = metis_episodic_build_hybrid_context(
            &store, "unseen query wording", query_key, 0.0f, 1);
        CHECK(context != NULL && strstr(context, "blue-9021") != NULL,
              "semantic-only retrieval");
        free(context);
        CHECK(metis_episodic_upsert_with_key(
                  &store, "Mira's locker code is violet-7710 now.",
                  update_key, 0.9f, &changed) == 0,
              "addressed update");
        CHECK(changed != SIZE_MAX, "addressed update replaced a record");
        CHECK(metis_episodic_count(&store) == 3,
              "addressed update preserves record count");
        context = metis_episodic_build_hybrid_context(
            &store, "unseen query wording", query_key, 0.0f, 1);
        CHECK(context != NULL && strstr(context, "violet-7710") != NULL &&
                  strstr(context, "blue-9021") == NULL,
              "addressed update removes stale value");
        free(context);
        CHECK(metis_episodic_delete_with_key(
                  &store, delete_key, 0.9f, &changed) == 1,
              "addressed delete");
        CHECK(changed != SIZE_MAX, "addressed delete reports record");
        CHECK(metis_episodic_count(&store) == 2,
              "addressed delete removes one record");
        CHECK(metis_episodic_tombstone_with_key(
                  &store, "Please forget Mira's locker code.",
                  update_key, 0.9f, &changed) == 0,
              "addressed tombstone");
        CHECK(changed != SIZE_MAX, "tombstone replaced current record");
        context = metis_episodic_build_hybrid_context(
            &store, "Mira locker code", query_key, 0.0f, 1);
        CHECK(context != NULL &&
                  strstr(context, "DELETED MEMORY") != NULL &&
                  strstr(context, "violet-7710") == NULL,
              "tombstone suppresses previous value");
        free(context);
    }

    context = metis_episodic_build_context(
        &store, "What is Mira's locker code?", 2);
    CHECK(context != NULL, "retrieval context");
    if (context != NULL) {
        const char *deleted = strstr(context, "DELETED MEMORY");
        const char *old_value = strstr(context, "amber-4172");
        CHECK(deleted != NULL && old_value != NULL,
              "retrieval contains tombstone and older unmatched version");
        free(context);
    }

    CHECK(metis_episodic_save(
              &store, "/tmp/test_episodic.bnepisodic") == 0,
          "save episodic records");
    CHECK(metis_episodic_load(
              &loaded, "/tmp/test_episodic.bnepisodic") == 0,
          "load episodic records");
    CHECK(metis_episodic_count(&loaded) == 2, "record count roundtrip");
    context = metis_episodic_build_context(
        &loaded, "DELETED MEMORY", 1);
    CHECK(context != NULL && strstr(context, "DELETED MEMORY") != NULL,
          "loaded retrieval");
    free(context);

    {
        FILE *file = fopen("/tmp/test_episodic.bnepisodic", "r+b");
        CHECK(file != NULL, "open episodic file for corruption");
        if (file != NULL) {
            CHECK(fseek(file, -1, SEEK_END) == 0,
                  "seek episodic payload");
            if (fseek(file, -1, SEEK_END) == 0) {
                int value = fgetc(file);
                CHECK(value != EOF, "read episodic payload");
                CHECK(fseek(file, -1, SEEK_CUR) == 0,
                      "rewind episodic payload");
                if (value != EOF) fputc(value ^ 1, file);
            }
            fclose(file);
        }
    }
    CHECK(metis_episodic_load(
              &loaded, "/tmp/test_episodic.bnepisodic") != 0,
          "payload corruption rejected");
    CHECK(metis_episodic_count(&loaded) == 2,
          "failed load leaves existing records intact");

    remove("/tmp/test_episodic.bnepisodic");
    metis_episodic_free(&loaded);
    metis_episodic_free(&store);
    if (failures != 0) return 1;
    printf("test_episodic_store: OK\n");
    return 0;
}
