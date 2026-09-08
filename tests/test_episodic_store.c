#include "metis/episodic_store.h"

#include <stdio.h>
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
    }

    context = metis_episodic_build_context(
        &store, "What is Mira's locker code?", 2);
    CHECK(context != NULL, "retrieval context");
    if (context != NULL) {
        const char *new_value = strstr(context, "blue-9021");
        const char *old_value = strstr(context, "amber-4172");
        CHECK(new_value != NULL && old_value != NULL,
              "retrieval contains matching versions");
        CHECK(new_value < old_value, "newer equal-score record first");
        free(context);
    }

    CHECK(metis_episodic_save(
              &store, "/tmp/test_episodic.bnepisodic") == 0,
          "save episodic records");
    CHECK(metis_episodic_load(
              &loaded, "/tmp/test_episodic.bnepisodic") == 0,
          "load episodic records");
    CHECK(metis_episodic_count(&loaded) == 3, "record count roundtrip");
    context = metis_episodic_build_context(
        &loaded, "When does Jonah travel?", 1);
    CHECK(context != NULL && strstr(context, "Saturday morning") != NULL,
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
    CHECK(metis_episodic_count(&loaded) == 3,
          "failed load leaves existing records intact");

    remove("/tmp/test_episodic.bnepisodic");
    metis_episodic_free(&loaded);
    metis_episodic_free(&store);
    if (failures != 0) return 1;
    printf("test_episodic_store: OK\n");
    return 0;
}
