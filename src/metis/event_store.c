#include "event_store.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EVENT_MAX_COUNT 8192u
#define EVENT_MAX_STRING 65536u
#define EVENT_FILE_VERSION 2u

static const unsigned char k_magic[8] =
    {'B','N','E','V','T','1',0,0};

static char *copy_string(const char *value, int required) {
    size_t length;
    char *copy;
    if (value == NULL) return required ? NULL : calloc(1, 1);
    length = strlen(value);
    if ((required && length == 0) || length > EVENT_MAX_STRING) return NULL;
    copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, value, length + 1);
    return copy;
}

static void clear_record(metis_event_record_t *event) {
    if (event == NULL) return;
    free(event->event_id);
    free(event->episode_id);
    free(event->source_id);
    free(event->entity);
    free(event->predicate);
    free(event->value);
    free(event->valid_time);
    free(event->target_event_id);
    memset(event, 0, sizeof *event);
}

void metis_event_store_init(metis_event_store_t *store) {
    if (store != NULL) memset(store, 0, sizeof *store);
}

void metis_event_store_clear(metis_event_store_t *store) {
    if (store == NULL) return;
    for (size_t i = 0; i < store->count; ++i)
        clear_record(&store->events[i]);
    free(store->events);
    memset(store, 0, sizeof *store);
}

static int valid_operation(metis_event_operation_t operation) {
    return operation >= METIS_EVENT_ASSERT &&
           operation <= METIS_EVENT_RETRACT;
}

static int valid_memory_kind(metis_memory_kind_t kind) {
    return kind >= METIS_MEMORY_PROPERTY && kind <= METIS_MEMORY_EVENT;
}

static int valid_polarity(metis_event_polarity_t polarity) {
    return polarity >= METIS_EVENT_POSITIVE &&
           polarity <= METIS_EVENT_NEGATIVE;
}

static int valid_modality(metis_event_modality_t modality) {
    return modality >= METIS_EVENT_ACTUAL &&
           modality <= METIS_EVENT_POSSIBLE;
}

static int valid_predicate(const char *predicate) {
    size_t length;
    if (predicate == NULL) return 0;
    length = strlen(predicate);
    if (length == 0 || length > 64 ||
        predicate[0] < 'a' || predicate[0] > 'z')
        return 0;
    for (size_t i = 1; i < length; ++i) {
        unsigned char byte = (unsigned char)predicate[i];
        if (!((byte >= 'a' && byte <= 'z') ||
              (byte >= '0' && byte <= '9') || byte == '_'))
            return 0;
    }
    return 1;
}

static int span_known(size_t start, size_t end) {
    return start != METIS_EVENT_SPAN_UNKNOWN &&
           end != METIS_EVENT_SPAN_UNKNOWN;
}

static int valid_spans(const metis_event_record_t *event) {
    int subject_known = span_known(
        event->subject_start, event->subject_end);
    int value_known = span_known(event->value_start, event->value_end);
    if (!subject_known && !value_known) {
        return (event->subject_start == METIS_EVENT_SPAN_UNKNOWN ||
                event->subject_start == 0) &&
               (event->subject_end == METIS_EVENT_SPAN_UNKNOWN ||
                event->subject_end == 0) &&
               (event->value_start == METIS_EVENT_SPAN_UNKNOWN ||
                event->value_start == 0) &&
               (event->value_end == METIS_EVENT_SPAN_UNKNOWN ||
                event->value_end == 0);
    }
    return subject_known && value_known &&
           event->subject_end > event->subject_start &&
           event->value_end > event->value_start &&
           event->subject_end <= UINT32_MAX &&
           event->value_end <= UINT32_MAX;
}

static int copy_record(
    metis_event_record_t *output, const metis_event_record_t *input) {
    memset(output, 0, sizeof *output);
    output->event_id = copy_string(input->event_id, 1);
    output->episode_id = copy_string(input->episode_id, 1);
    output->source_id = copy_string(input->source_id, 1);
    output->entity = copy_string(input->entity, 1);
    output->predicate = copy_string(input->predicate, 1);
    output->value = copy_string(input->value, 1);
    output->valid_time = copy_string(input->valid_time, 0);
    output->target_event_id = copy_string(input->target_event_id, 0);
    output->operation = input->operation;
    output->memory_kind = input->memory_kind;
    output->polarity = input->polarity;
    output->modality = input->modality;
    output->active = input->active;
    output->raw_record_index = input->raw_record_index;
    output->subject_start = input->subject_start;
    output->subject_end = input->subject_end;
    output->value_start = input->value_start;
    output->value_end = input->value_end;
    if (output->event_id == NULL || output->episode_id == NULL ||
        output->source_id == NULL || output->entity == NULL ||
        output->predicate == NULL || output->value == NULL ||
        output->valid_time == NULL || output->target_event_id == NULL) {
        clear_record(output);
        return -1;
    }
    return 0;
}

const metis_event_record_t *metis_event_store_find(
    const metis_event_store_t *store, const char *event_id) {
    if (store == NULL || event_id == NULL) return NULL;
    for (size_t i = 0; i < store->count; ++i)
        if (strcmp(store->events[i].event_id, event_id) == 0)
            return &store->events[i];
    return NULL;
}

static metis_event_record_t *find_mutable(
    metis_event_store_t *store, const char *event_id) {
    return (metis_event_record_t *)metis_event_store_find(store, event_id);
}

int metis_event_store_apply(
    metis_event_store_t *store, const metis_event_record_t *event) {
    metis_event_record_t copy;
    metis_event_record_t *target = NULL;
    if (store == NULL || event == NULL ||
        !valid_operation(event->operation) ||
        !valid_memory_kind(event->memory_kind) ||
        !valid_polarity(event->polarity) ||
        !valid_modality(event->modality) ||
        !valid_predicate(event->predicate) ||
        !valid_spans(event) ||
        event->event_id == NULL || event->episode_id == NULL ||
        event->source_id == NULL || event->entity == NULL ||
        event->predicate == NULL || event->value == NULL ||
        metis_event_store_find(store, event->event_id) != NULL ||
        store->count >= EVENT_MAX_COUNT)
        return -1;
    if (event->operation != METIS_EVENT_ASSERT) {
        if (event->target_event_id == NULL ||
            event->target_event_id[0] == '\0')
            return -1;
        target = find_mutable(store, event->target_event_id);
        if (target == NULL || !target->active) return -1;
        if (strcmp(target->entity, event->entity) != 0 ||
            strcmp(target->predicate, event->predicate) != 0 ||
            target->memory_kind != event->memory_kind)
            return -1;
        if (event->operation == METIS_EVENT_SUPERSEDE &&
            event->memory_kind != METIS_MEMORY_PROPERTY)
            return -1;
    }
    if (copy_record(&copy, event) != 0) return -1;
    copy.active = event->operation != METIS_EVENT_RETRACT;
    if (store->count == store->capacity) {
        size_t capacity = store->capacity ? store->capacity * 2 : 16;
        metis_event_record_t *grown = realloc(
            store->events, capacity * sizeof *grown);
        if (grown == NULL) {
            clear_record(&copy);
            return -1;
        }
        store->events = grown;
        store->capacity = capacity;
    }
    if (target != NULL) target->active = 0;
    store->events[store->count++] = copy;
    return 0;
}

const metis_event_record_t *metis_event_store_current(
    const metis_event_store_t *store,
    const char *entity, const char *predicate) {
    if (store == NULL || entity == NULL || predicate == NULL) return NULL;
    for (size_t i = store->count; i > 0; --i) {
        const metis_event_record_t *event = &store->events[i - 1];
        if (event->active &&
            strcmp(event->entity, entity) == 0 &&
            strcmp(event->predicate, predicate) == 0)
            return event;
    }
    return NULL;
}

size_t metis_event_store_active_count(
    const metis_event_store_t *store,
    const char *entity, const char *predicate) {
    size_t count = 0;
    if (store == NULL || entity == NULL || predicate == NULL) return 0;
    for (size_t i = 0; i < store->count; ++i) {
        const metis_event_record_t *event = &store->events[i];
        if (event->active &&
            strcmp(event->entity, entity) == 0 &&
            strcmp(event->predicate, predicate) == 0)
            ++count;
    }
    return count;
}

const metis_event_record_t *metis_event_store_active_at(
    const metis_event_store_t *store,
    const char *entity, const char *predicate, size_t ordinal) {
    if (store == NULL || entity == NULL || predicate == NULL) return NULL;
    for (size_t i = store->count; i > 0; --i) {
        const metis_event_record_t *event = &store->events[i - 1];
        if (event->active &&
            strcmp(event->entity, entity) == 0 &&
            strcmp(event->predicate, predicate) == 0) {
            if (ordinal == 0) return event;
            --ordinal;
        }
    }
    return NULL;
}

static uint32_t crc32_bytes(const void *data, size_t size) {
    const unsigned char *bytes = data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^
                  (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

static int write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {value, value >> 8, value >> 16, value >> 24};
    return fwrite(bytes, 1, 4, file) == 4 ? 0 : -1;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, file) != 4) return -1;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 0;
}

static int write_string(FILE *file, const char *value) {
    uint32_t length = (uint32_t)strlen(value);
    return write_u32(file, length) ||
           write_u32(file, crc32_bytes(value, length)) ||
           fwrite(value, 1, length, file) != length ? -1 : 0;
}

static char *read_string(FILE *file) {
    uint32_t length, crc;
    char *value;
    if (read_u32(file, &length) || length > EVENT_MAX_STRING ||
        read_u32(file, &crc)) return NULL;
    value = malloc((size_t)length + 1);
    if (value == NULL || fread(value, 1, length, file) != length) {
        free(value);
        return NULL;
    }
    value[length] = '\0';
    if (crc32_bytes(value, length) != crc) {
        free(value);
        return NULL;
    }
    return value;
}

int metis_event_store_save(
    const metis_event_store_t *store, const char *path) {
    FILE *file;
    if (store == NULL || path == NULL ||
        store->count > EVENT_MAX_COUNT) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    if (fwrite(k_magic, 1, 8, file) != 8 ||
        write_u32(file, EVENT_FILE_VERSION) ||
        write_u32(file, (uint32_t)store->count)) goto fail;
    for (size_t i = 0; i < store->count; ++i) {
        const metis_event_record_t *event = &store->events[i];
        uint32_t subject_start = event->subject_start == METIS_EVENT_SPAN_UNKNOWN
            ? UINT32_MAX : (uint32_t)event->subject_start;
        uint32_t subject_end = event->subject_end == METIS_EVENT_SPAN_UNKNOWN
            ? UINT32_MAX : (uint32_t)event->subject_end;
        uint32_t value_start = event->value_start == METIS_EVENT_SPAN_UNKNOWN
            ? UINT32_MAX : (uint32_t)event->value_start;
        uint32_t value_end = event->value_end == METIS_EVENT_SPAN_UNKNOWN
            ? UINT32_MAX : (uint32_t)event->value_end;
        if (event->raw_record_index > UINT32_MAX ||
            !valid_spans(event))
            goto fail;
        if (write_u32(file, (uint32_t)event->operation) ||
            write_u32(file, (uint32_t)event->active) ||
            write_u32(file, (uint32_t)event->raw_record_index) ||
            write_u32(file, (uint32_t)event->memory_kind) ||
            write_u32(file, (uint32_t)event->polarity) ||
            write_u32(file, (uint32_t)event->modality) ||
            write_u32(file, subject_start) ||
            write_u32(file, subject_end) ||
            write_u32(file, value_start) ||
            write_u32(file, value_end) ||
            write_string(file, event->event_id) ||
            write_string(file, event->episode_id) ||
            write_string(file, event->source_id) ||
            write_string(file, event->entity) ||
            write_string(file, event->predicate) ||
            write_string(file, event->value) ||
            write_string(file, event->valid_time) ||
            write_string(file, event->target_event_id))
            goto fail;
    }
    if (fclose(file) != 0) {
        remove(path);
        return -1;
    }
    return 0;
fail:
    fclose(file);
    remove(path);
    return -1;
}

int metis_event_store_load(
    metis_event_store_t *store, const char *path) {
    metis_event_store_t loaded;
    metis_event_store_t replayed;
    FILE *file;
    unsigned char magic[8];
    uint32_t version, count;
    if (store == NULL || path == NULL) return -1;
    metis_event_store_init(&loaded);
    metis_event_store_init(&replayed);
    file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (fread(magic, 1, 8, file) != 8 ||
        memcmp(magic, k_magic, 8) != 0 ||
        read_u32(file, &version) ||
        (version != 1 && version != EVENT_FILE_VERSION) ||
        read_u32(file, &count) || count > EVENT_MAX_COUNT)
        goto fail;
    for (uint32_t i = 0; i < count; ++i) {
        metis_event_record_t event;
        uint32_t operation, active, raw_index;
        uint32_t memory_kind = METIS_MEMORY_PROPERTY;
        uint32_t polarity = METIS_EVENT_POSITIVE;
        uint32_t modality = METIS_EVENT_ACTUAL;
        uint32_t subject_start = UINT32_MAX;
        uint32_t subject_end = UINT32_MAX;
        uint32_t value_start = UINT32_MAX;
        uint32_t value_end = UINT32_MAX;
        memset(&event, 0, sizeof event);
        if (read_u32(file, &operation) ||
            read_u32(file, &active) || active > 1 ||
            read_u32(file, &raw_index) ||
            operation > METIS_EVENT_RETRACT)
            goto fail_record;
        if (version == EVENT_FILE_VERSION &&
            (read_u32(file, &memory_kind) ||
             read_u32(file, &polarity) ||
             read_u32(file, &modality) ||
             read_u32(file, &subject_start) ||
             read_u32(file, &subject_end) ||
             read_u32(file, &value_start) ||
             read_u32(file, &value_end) ||
             memory_kind > METIS_MEMORY_EVENT ||
             polarity > METIS_EVENT_NEGATIVE ||
             modality > METIS_EVENT_POSSIBLE))
            goto fail_record;
        event.operation = (metis_event_operation_t)operation;
        event.memory_kind = (metis_memory_kind_t)memory_kind;
        event.polarity = (metis_event_polarity_t)polarity;
        event.modality = (metis_event_modality_t)modality;
        event.active = (int)active;
        event.raw_record_index = raw_index;
        event.subject_start = subject_start == UINT32_MAX
            ? METIS_EVENT_SPAN_UNKNOWN : subject_start;
        event.subject_end = subject_end == UINT32_MAX
            ? METIS_EVENT_SPAN_UNKNOWN : subject_end;
        event.value_start = value_start == UINT32_MAX
            ? METIS_EVENT_SPAN_UNKNOWN : value_start;
        event.value_end = value_end == UINT32_MAX
            ? METIS_EVENT_SPAN_UNKNOWN : value_end;
        event.event_id = read_string(file);
        event.episode_id = read_string(file);
        event.source_id = read_string(file);
        event.entity = read_string(file);
        event.predicate = read_string(file);
        event.value = read_string(file);
        event.valid_time = read_string(file);
        event.target_event_id = read_string(file);
        if (event.event_id == NULL || event.episode_id == NULL ||
            event.source_id == NULL || event.entity == NULL ||
            event.predicate == NULL || event.value == NULL ||
            event.valid_time == NULL || event.target_event_id == NULL)
            goto fail_record;
        if (loaded.count == loaded.capacity) {
            size_t capacity = loaded.capacity ? loaded.capacity * 2 : 16;
            metis_event_record_t *grown = realloc(
                loaded.events, capacity * sizeof *grown);
            if (grown == NULL) goto fail_record;
            loaded.events = grown;
            loaded.capacity = capacity;
        }
        loaded.events[loaded.count++] = event;
        continue;
fail_record:
        clear_record(&event);
        goto fail;
    }
    if (fgetc(file) != EOF) goto fail;
    fclose(file);
    file = NULL;
    for (size_t i = 0; i < loaded.count; ++i) {
        if (metis_event_store_apply(
                &replayed, &loaded.events[i]) != 0)
            goto fail;
    }
    if (replayed.count != loaded.count) goto fail;
    for (size_t i = 0; i < loaded.count; ++i)
        if (replayed.events[i].active != loaded.events[i].active)
            goto fail;
    metis_event_store_clear(&loaded);
    loaded = replayed;
    metis_event_store_init(&replayed);
    metis_event_store_clear(store);
    *store = loaded;
    return 0;
fail:
    if (file != NULL) fclose(file);
    metis_event_store_clear(&replayed);
    metis_event_store_clear(&loaded);
    return -1;
}
