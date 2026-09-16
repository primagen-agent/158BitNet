#include "episodic_store.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPISODIC_MAX_RECORDS 4096u
#define EPISODIC_MAX_RECORD_BYTES 65536u

static const unsigned char k_magic[8] =
    {'B', 'N', 'E', 'P', 'I', '1', 0, 0};

static char *duplicate_text(const char *text) {
    size_t size;
    char *copy;
    if (text == NULL) return NULL;
    size = strlen(text);
    if (size == 0 || size > EPISODIC_MAX_RECORD_BYTES) return NULL;
    copy = (char *)malloc(size + 1);
    if (copy == NULL) return NULL;
    memcpy(copy, text, size + 1);
    return copy;
}

static int write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4];
    for (int i = 0; i < 4; ++i)
        bytes[i] = (unsigned char)(value >> (8 * i));
    return fwrite(bytes, 1, sizeof bytes, file) == sizeof bytes ? 0 : -1;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (fread(bytes, 1, sizeof bytes, file) != sizeof bytes) return -1;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 0;
}

static uint32_t crc32_bytes(const void *data, size_t size) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^
                  (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

static uint32_t f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static float bits_f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

void metis_episodic_init(metis_episodic_store_t *store) {
    if (store != NULL) memset(store, 0, sizeof *store);
}

void metis_episodic_clear(metis_episodic_store_t *store) {
    if (store == NULL) return;
    for (size_t i = 0; i < store->count; ++i) {
        free(store->records[i]);
        if (store->keys != NULL) free(store->keys[i]);
    }
    free(store->records);
    free(store->keys);
    free(store->priorities);
    memset(store, 0, sizeof *store);
}

void metis_episodic_free(metis_episodic_store_t *store) {
    metis_episodic_clear(store);
}

int metis_episodic_configure_keys(
    metis_episodic_store_t *store, int key_dim) {
    if (store == NULL || key_dim < 0 || key_dim > 4096) return -1;
    if (store->count > 0 && store->key_dim != 0 &&
        store->key_dim != key_dim) return -1;
    store->key_dim = key_dim;
    return 0;
}

int metis_episodic_add_with_priority(
    metis_episodic_store_t *store, const char *text, const float *key,
    float priority) {
    char *copy;
    if (store == NULL || text == NULL || text[0] == '\0') return -1;
    if (!isfinite(priority) || priority < 0.0f || priority > 1.0f)
        return -1;
    if (key != NULL && store->key_dim <= 0) return -1;
    for (size_t i = 0; i < store->count; ++i) {
        if (strcmp(store->records[i], text) == 0) {
            if (key != NULL) {
                if (store->keys[i] == NULL) {
                    store->keys[i] = (float *)malloc(
                        (size_t)store->key_dim * sizeof(float));
                    if (store->keys[i] == NULL) return -1;
                }
                memcpy(store->keys[i], key,
                       (size_t)store->key_dim * sizeof(float));
            }
            if (priority > store->priorities[i])
                store->priorities[i] = priority;
            return 0;
        }
    }
    if (store->count >= EPISODIC_MAX_RECORDS) return -1;
    if (store->count == store->capacity) {
        size_t capacity = store->capacity == 0 ? 16 : store->capacity * 2;
        char **records;
        float **keys;
        float *priorities;
        if (capacity > EPISODIC_MAX_RECORDS)
            capacity = EPISODIC_MAX_RECORDS;
        records = (char **)realloc(
            store->records, capacity * sizeof *records);
        if (records == NULL) return -1;
        store->records = records;
        keys = (float **)realloc(
            store->keys, capacity * sizeof *keys);
        if (keys == NULL) return -1;
        store->keys = keys;
        priorities = (float *)realloc(
            store->priorities, capacity * sizeof *priorities);
        if (priorities == NULL) return -1;
        store->priorities = priorities;
        for (size_t i = store->capacity; i < capacity; ++i) {
            store->keys[i] = NULL;
            store->priorities[i] = 0.0f;
        }
        store->capacity = capacity;
    }
    copy = duplicate_text(text);
    if (copy == NULL) return -1;
    store->records[store->count] = copy;
    store->keys[store->count] = NULL;
    store->priorities[store->count] = priority;
    if (key != NULL) {
        store->keys[store->count] = (float *)malloc(
            (size_t)store->key_dim * sizeof(float));
        if (store->keys[store->count] == NULL) {
            free(copy);
            store->records[store->count] = NULL;
            return -1;
        }
        memcpy(store->keys[store->count], key,
               (size_t)store->key_dim * sizeof(float));
    }
    ++store->count;
    return 0;
}
size_t metis_episodic_count(
    const metis_episodic_store_t *store) {
    return store == NULL ? 0 : store->count;
}

int metis_episodic_save(
    const metis_episodic_store_t *store, const char *path) {
    FILE *file;
    if (store == NULL || path == NULL ||
        store->count > EPISODIC_MAX_RECORDS ||
        store->key_dim < 0 || store->key_dim > 4096) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    if (fwrite(k_magic, 1, sizeof k_magic, file) != sizeof k_magic ||
        write_u32(file, 4) != 0 ||
        write_u32(file, (uint32_t)store->count) != 0 ||
        write_u32(file, (uint32_t)store->key_dim) != 0) goto fail;
    for (size_t i = 0; i < store->count; ++i) {
        size_t length = strlen(store->records[i]);
        const float *key = store->keys == NULL ? NULL : store->keys[i];
        size_t key_bytes =
            (size_t)store->key_dim * sizeof(float);
        if (length == 0 || length > EPISODIC_MAX_RECORD_BYTES ||
            write_u32(file, (uint32_t)length) != 0 ||
            write_u32(
                file, crc32_bytes(store->records[i], length)) != 0 ||
            write_u32(file, key == NULL ? 0u : 1u) != 0 ||
            (key != NULL &&
             write_u32(file, crc32_bytes(key, key_bytes)) != 0) ||
            write_u32(file, f32_bits(store->priorities[i])) != 0 ||
            fwrite(store->records[i], 1, length, file) != length ||
            (key != NULL &&
             fwrite(key, 1, key_bytes, file) != key_bytes)) goto fail;
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

int metis_episodic_load(
    metis_episodic_store_t *store, const char *path) {
    metis_episodic_store_t loaded;
    FILE *file;
    unsigned char magic[8];
    uint32_t version = 0, count = 0, key_dim = 0;
    if (store == NULL || path == NULL) return -1;
    metis_episodic_init(&loaded);
    file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (fread(magic, 1, sizeof magic, file) != sizeof magic ||
        memcmp(magic, k_magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 ||
        (version < 1 || version > 4) ||
        read_u32(file, &count) != 0 || count > EPISODIC_MAX_RECORDS ||
        (version >= 3 &&
         (read_u32(file, &key_dim) != 0 || key_dim > 4096u)))
        goto fail;
    if (version >= 3 &&
        metis_episodic_configure_keys(&loaded, (int)key_dim) != 0)
        goto fail;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t length = 0;
        uint32_t expected_crc = 0;
        uint32_t has_key = 0;
        uint32_t expected_key_crc = 0;
        uint32_t priority_bits = 0;
        float priority = 0.0f;
        char *text;
        float *key = NULL;
        size_t key_bytes = (size_t)key_dim * sizeof(float);
        if (read_u32(file, &length) != 0 || length == 0 ||
            length > EPISODIC_MAX_RECORD_BYTES) goto fail;
        if (version >= 2 && read_u32(file, &expected_crc) != 0) goto fail;
        if (version >= 3 &&
            (read_u32(file, &has_key) != 0 || has_key > 1u ||
             (has_key != 0u && key_dim == 0u) ||
             (has_key != 0u &&
              read_u32(file, &expected_key_crc) != 0)))
            goto fail;
        if (version >= 4) {
            if (read_u32(file, &priority_bits) != 0) goto fail;
            priority = bits_f32(priority_bits);
            if (!isfinite(priority) ||
                priority < 0.0f || priority > 1.0f)
                goto fail;
        }
        text = (char *)malloc((size_t)length + 1);
        if (text == NULL ||
            fread(text, 1, length, file) != length) {
            free(text);
            goto fail;
        }
        text[length] = '\0';
        if (version >= 2 &&
            crc32_bytes(text, length) != expected_crc) {
            free(text);
            goto fail;
        }
        if (has_key != 0u) {
            key = (float *)malloc(key_bytes);
            if (key == NULL ||
                fread(key, 1, key_bytes, file) != key_bytes ||
                crc32_bytes(key, key_bytes) != expected_key_crc) {
                free(key);
                free(text);
                goto fail;
            }
            for (uint32_t column = 0; column < key_dim; ++column) {
                if (!isfinite(key[column])) {
                    free(key);
                    free(text);
                    goto fail;
                }
            }
        }
        if (metis_episodic_add_with_priority(
                &loaded, text, key, priority) != 0) {
            free(key);
            free(text);
            goto fail;
        }
        free(key);
        free(text);
    }
    if (fgetc(file) != EOF) goto fail;
    fclose(file);
    metis_episodic_clear(store);
    *store = loaded;
    return 0;
fail:
    fclose(file);
    metis_episodic_clear(&loaded);
    return -1;
}
