#include "metis/event_store.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t crc32_bytes(const void *data, size_t size) {
    const unsigned char *bytes = data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t index = 0; index < size; ++index) {
        crc ^= bytes[index];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^
                  (0xEDB88320u &
                   (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

static int write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {
        value, value >> 8, value >> 16, value >> 24};
    return fwrite(bytes, 1, sizeof bytes, file) == sizeof bytes ? 0 : -1;
}

static int write_string(FILE *file, const char *value) {
    size_t length = strlen(value);
    return write_u32(file, (uint32_t)length) ||
           write_u32(file, crc32_bytes(value, length)) ||
           fwrite(value, 1, length, file) != length ? -1 : 0;
}

static int write_v1_fixture(const char *path) {
    static const unsigned char magic[8] =
        {'B','N','E','V','T','1',0,0};
    FILE *file = fopen(path, "wb");
    const char *strings[] = {
        "legacy-1", "episode-1", "D1:1", "Alice",
        "lives_in", "Seattle", "2025-01-01", "",
    };
    if (file == NULL) return -1;
    if (fwrite(magic, 1, sizeof magic, file) != sizeof magic ||
        write_u32(file, 1) || write_u32(file, 1) ||
        write_u32(file, METIS_EVENT_ASSERT) ||
        write_u32(file, 1) || write_u32(file, 0))
        goto fail;
    for (size_t index = 0;
         index < sizeof strings / sizeof strings[0]; ++index)
        if (write_string(file, strings[index]) != 0) goto fail;
    return fclose(file);
fail:
    fclose(file);
    remove(path);
    return -1;
}

static metis_event_record_t event(
    const char *id, const char *value,
    metis_event_operation_t operation, const char *target) {
    metis_event_record_t result = {
        .event_id = (char *)id,
        .episode_id = "episode-1",
        .source_id = "D1:1",
        .entity = "Alice",
        .predicate = "lives_in",
        .value = (char *)value,
        .valid_time = "2026-09-10",
        .target_event_id = (char *)(target == NULL ? "" : target),
        .operation = operation,
        .memory_kind = METIS_MEMORY_PROPERTY,
        .polarity = METIS_EVENT_POSITIVE,
        .modality = METIS_EVENT_ACTUAL,
        .active = 1,
        .raw_record_index = 3,
        .subject_start = 0,
        .subject_end = 5,
        .value_start = 15,
        .value_end = 15 + strlen(value),
    };
    return result;
}

static int corrupt_and_reject(
    metis_event_store_t *store, const char *path) {
    FILE *file = fopen(path, "r+b");
    int value;
    if (file == NULL || fseek(file, -1, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return -1;
    }
    value = fgetc(file);
    if (value == EOF || fseek(file, -1, SEEK_CUR) != 0) {
        fclose(file);
        return -1;
    }
    fputc(value ^ 1, file);
    fclose(file);
    if (metis_event_store_load(store, path) == 0) return -1;
    return 0;
}

int main(void) {
    metis_event_store_t store, loaded, legacy;
    metis_event_store_init(&store);
    metis_event_store_init(&loaded);
    metis_event_store_init(&legacy);
    metis_event_record_t first = event(
        "event-1", "Seattle", METIS_EVENT_ASSERT, NULL);
    metis_event_record_t second = event(
        "event-2", "Kyoto", METIS_EVENT_SUPERSEDE, "event-1");
    metis_event_record_t retract = event(
        "event-3", "Kyoto", METIS_EVENT_RETRACT, "event-2");
    metis_event_record_t wrong_target = event(
        "event-bad", "Lisbon", METIS_EVENT_SUPERSEDE, "missing");
    metis_event_record_t set_first = event(
        "event-set-1", "jazz", METIS_EVENT_ASSERT, NULL);
    metis_event_record_t set_second = event(
        "event-set-2", "folk", METIS_EVENT_ASSERT, NULL);
    metis_event_record_t invalid_set_update = event(
        "event-set-bad", "classical", METIS_EVENT_SUPERSEDE,
        "event-set-1");
    set_first.predicate = "likes";
    set_second.predicate = "likes";
    invalid_set_update.predicate = "likes";
    set_first.memory_kind = METIS_MEMORY_SET;
    set_second.memory_kind = METIS_MEMORY_SET;
    invalid_set_update.memory_kind = METIS_MEMORY_SET;
    if (metis_event_store_apply(&store, &first) ||
        metis_event_store_apply(&store, &second) ||
        metis_event_store_apply(&store, &set_first) ||
        metis_event_store_apply(&store, &set_second) ||
        metis_event_store_apply(&store, &invalid_set_update) == 0 ||
        metis_event_store_active_count(
            &store, "Alice", "likes") != 2 ||
        strcmp(metis_event_store_active_at(
            &store, "Alice", "likes", 0)->value, "folk") != 0 ||
        metis_event_store_apply(&store, &wrong_target) == 0 ||
        metis_event_store_find(&store, "event-1")->active ||
        strcmp(metis_event_store_current(
            &store, "Alice", "lives_in")->value, "Kyoto") != 0 ||
        metis_event_store_save(&store, "/tmp/test-events.bnevent") ||
        metis_event_store_load(&loaded, "/tmp/test-events.bnevent") ||
        strcmp(metis_event_store_current(
            &loaded, "Alice", "lives_in")->event_id, "event-2") != 0 ||
        metis_event_store_active_count(
            &loaded, "Alice", "likes") != 2 ||
        loaded.events[0].subject_start != 0 ||
        loaded.events[0].subject_end != 5 ||
        corrupt_and_reject(
            &loaded, "/tmp/test-events.bnevent") != 0 ||
        strcmp(metis_event_store_current(
            &loaded, "Alice", "lives_in")->event_id, "event-2") != 0 ||
        metis_event_store_apply(&loaded, &retract) ||
        metis_event_store_current(
            &loaded, "Alice", "lives_in") != NULL) {
        metis_event_store_clear(&legacy);
        metis_event_store_clear(&loaded);
        metis_event_store_clear(&store);
        return 1;
    }
    if (write_v1_fixture("/tmp/test-events-v1.bnevent") != 0 ||
        metis_event_store_load(
            &legacy, "/tmp/test-events-v1.bnevent") != 0 ||
        legacy.count != 1 ||
        legacy.events[0].memory_kind != METIS_MEMORY_PROPERTY ||
        legacy.events[0].polarity != METIS_EVENT_POSITIVE ||
        legacy.events[0].modality != METIS_EVENT_ACTUAL ||
        legacy.events[0].subject_start != METIS_EVENT_SPAN_UNKNOWN) {
        metis_event_store_clear(&legacy);
        metis_event_store_clear(&loaded);
        metis_event_store_clear(&store);
        return 1;
    }
    remove("/tmp/test-events.bnevent");
    remove("/tmp/test-events-v1.bnevent");
    metis_event_store_clear(&legacy);
    metis_event_store_clear(&loaded);
    metis_event_store_clear(&store);
    puts("test_event_store: OK");
    return 0;
}
