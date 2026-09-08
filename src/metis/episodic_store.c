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

int metis_episodic_add_with_key(
    metis_episodic_store_t *store, const char *text, const float *key) {
    char *copy;
    if (store == NULL || text == NULL || text[0] == '\0') return -1;
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
            return 0;
        }
    }
    if (store->count >= EPISODIC_MAX_RECORDS) return -1;
    if (store->count == store->capacity) {
        size_t capacity = store->capacity == 0 ? 16 : store->capacity * 2;
        char **records;
        float **keys;
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
        for (size_t i = store->capacity; i < capacity; ++i)
            store->keys[i] = NULL;
        store->capacity = capacity;
    }
    copy = duplicate_text(text);
    if (copy == NULL) return -1;
    store->records[store->count] = copy;
    store->keys[store->count] = NULL;
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

int metis_episodic_add(metis_episodic_store_t *store, const char *text) {
    return metis_episodic_add_with_key(store, text, NULL);
}

size_t metis_episodic_count(const metis_episodic_store_t *store) {
    return store == NULL ? 0 : store->count;
}

static int contains_word_ci(const char *text, const char *word, size_t length) {
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor != '\0') {
        while (*cursor != '\0' && !isalnum(*cursor)) ++cursor;
        const unsigned char *start = cursor;
        while (*cursor != '\0' && isalnum(*cursor)) ++cursor;
        if ((size_t)(cursor - start) == length) {
            size_t i;
            for (i = 0; i < length; ++i)
                if (tolower(start[i]) !=
                    tolower((unsigned char)word[i])) break;
            if (i == length) return 1;
        }
    }
    return 0;
}

static int overlap_score(const char *record, const char *query) {
    const unsigned char *cursor = (const unsigned char *)query;
    int score = 0;
    while (*cursor != '\0') {
        while (*cursor != '\0' && !isalnum(*cursor)) ++cursor;
        const unsigned char *start = cursor;
        while (*cursor != '\0' && isalnum(*cursor)) ++cursor;
        size_t length = (size_t)(cursor - start);
        if (length >= 2 &&
            contains_word_ci(record, (const char *)start, length))
            ++score;
    }
    return score;
}

char *metis_episodic_build_context(
    const metis_episodic_store_t *store, const char *query, int top_k) {
    size_t *best_indices;
    int *best_scores;
    int found = 0;
    size_t bytes = 1;
    char *result;
    size_t offset = 0;
    if (store == NULL || query == NULL || top_k <= 0 ||
        store->count == 0) return NULL;
    if ((size_t)top_k > store->count) top_k = (int)store->count;
    best_indices = (size_t *)calloc((size_t)top_k, sizeof *best_indices);
    best_scores = (int *)calloc((size_t)top_k, sizeof *best_scores);
    if (best_indices == NULL || best_scores == NULL) {
        free(best_indices);
        free(best_scores);
        return NULL;
    }
    for (size_t index = 0; index < store->count; ++index) {
        int score = overlap_score(store->records[index], query);
        if (score <= 0) continue;
        int position;
        if (found < top_k) {
            position = found++;
        } else {
            position = top_k - 1;
            if (best_scores[position] > score ||
                (best_scores[position] == score &&
                 best_indices[position] > index))
                continue;
        }
        while (position > 0) {
            int previous_score = best_scores[position - 1];
            size_t previous_index = best_indices[position - 1];
            if (previous_score > score ||
                (previous_score == score && previous_index > index))
                break;
            if (position < top_k) {
                best_scores[position] = previous_score;
                best_indices[position] = previous_index;
            }
            --position;
        }
        best_scores[position] = score;
        best_indices[position] = index;
    }
    if (found == 0) {
        free(best_indices);
        free(best_scores);
        return NULL;
    }
    for (int i = 0; i < found; ++i)
        bytes += strlen(store->records[best_indices[i]]) + 32;
    result = (char *)malloc(bytes);
    if (result == NULL) {
        free(best_indices);
        free(best_scores);
        return NULL;
    }
    for (int i = 0; i < found; ++i) {
        int written = snprintf(
            result + offset, bytes - offset, "[memory %d] %s\n",
            i + 1, store->records[best_indices[i]]);
        if (written < 0 || (size_t)written >= bytes - offset) {
            free(result);
            result = NULL;
            break;
        }
        offset += (size_t)written;
    }
    free(best_indices);
    free(best_scores);
    return result;
}

typedef struct episodic_score {
    float score;
    size_t index;
} episodic_score_t;

static int compare_score_desc(const void *left, const void *right) {
    const episodic_score_t *a = (const episodic_score_t *)left;
    const episodic_score_t *b = (const episodic_score_t *)right;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->index > b->index ? -1 : (a->index < b->index ? 1 : 0);
}

static int count_words(const char *text) {
    const unsigned char *cursor = (const unsigned char *)text;
    int count = 0;
    while (*cursor != '\0') {
        while (*cursor != '\0' && !isalnum(*cursor)) ++cursor;
        if (*cursor == '\0') break;
        ++count;
        while (*cursor != '\0' && isalnum(*cursor)) ++cursor;
    }
    return count;
}

static int query_words(
    const char *query, char words[64][64], int counts[64]) {
    const unsigned char *cursor = (const unsigned char *)query;
    int count = 0;
    while (*cursor != '\0' && count < 64) {
        while (*cursor != '\0' && !isalnum(*cursor)) ++cursor;
        const unsigned char *start = cursor;
        while (*cursor != '\0' && isalnum(*cursor)) ++cursor;
        size_t length = (size_t)(cursor - start);
        if (length == 0 || length >= 64) continue;
        char word[64];
        for (size_t i = 0; i < length; ++i)
            word[i] = (char)tolower(start[i]);
        word[length] = '\0';
        int existing = -1;
        for (int i = 0; i < count; ++i)
            if (strcmp(words[i], word) == 0) {
                existing = i;
                break;
            }
        if (existing >= 0) {
            ++counts[existing];
        } else {
            memcpy(words[count], word, length + 1);
            counts[count] = 1;
            ++count;
        }
    }
    return count;
}

int metis_episodic_rank_hybrid(
    const metis_episodic_store_t *store, const char *query,
    const float *query_key, float lexical_weight,
    size_t *indices, int max_results) {
    char words[64][64];
    int word_counts[64] = {0};
    int word_count;
    int document_frequency[64] = {0};
    float average_length = 0.0f;
    episodic_score_t *lexical;
    episodic_score_t *semantic;
    episodic_score_t *combined;
    int *lexical_rank;
    int *semantic_rank;
    if (store == NULL || query == NULL || store->count == 0 ||
        indices == NULL || max_results <= 0 ||
        lexical_weight < 0.0f || lexical_weight > 1.0f)
        return -1;
    if ((size_t)max_results > store->count)
        max_results = (int)store->count;
    if (query_key == NULL) lexical_weight = 1.0f;
    word_count = query_words(query, words, word_counts);
    lexical = (episodic_score_t *)calloc(
        store->count, sizeof *lexical);
    semantic = (episodic_score_t *)calloc(
        store->count, sizeof *semantic);
    combined = (episodic_score_t *)calloc(
        store->count, sizeof *combined);
    lexical_rank = (int *)calloc(store->count, sizeof *lexical_rank);
    semantic_rank = (int *)calloc(store->count, sizeof *semantic_rank);
    if (lexical == NULL || semantic == NULL || combined == NULL ||
        lexical_rank == NULL || semantic_rank == NULL) goto fail;
    for (size_t index = 0; index < store->count; ++index) {
        average_length += (float)count_words(store->records[index]);
        for (int word = 0; word < word_count; ++word)
            document_frequency[word] += contains_word_ci(
                store->records[index], words[word],
                strlen(words[word]));
    }
    average_length /= (float)store->count;
    for (size_t index = 0; index < store->count; ++index) {
        float score = 0.0f;
        int length = count_words(store->records[index]);
        for (int word = 0; word < word_count; ++word) {
            int frequency = 0;
            const unsigned char *cursor =
                (const unsigned char *)store->records[index];
            size_t token_length = strlen(words[word]);
            while (*cursor != '\0') {
                while (*cursor != '\0' && !isalnum(*cursor)) ++cursor;
                const unsigned char *start = cursor;
                while (*cursor != '\0' && isalnum(*cursor)) ++cursor;
                if ((size_t)(cursor - start) == token_length) {
                    size_t i;
                    for (i = 0; i < token_length; ++i)
                        if (tolower(start[i]) !=
                            (unsigned char)words[word][i]) break;
                    if (i == token_length) ++frequency;
                }
            }
            if (frequency > 0) {
                float df = (float)document_frequency[word];
                float idf = logf(
                    1.0f + ((float)store->count - df + 0.5f) /
                    (df + 0.5f));
                float denominator = (float)frequency + 1.2f * (
                    0.25f + 0.75f * (float)length /
                    fmaxf(average_length, 1.0f));
                score += idf * (float)frequency * 2.2f / denominator *
                    (1.0f + 0.15f *
                     (float)(word_counts[word] > 3 ? 2 :
                             word_counts[word] - 1));
            }
        }
        lexical[index].score = score;
        lexical[index].index = index;
        semantic[index].score = -INFINITY;
        if (query_key != NULL && store->keys != NULL &&
            store->keys[index] != NULL && store->key_dim > 0) {
            float dot = 0.0f;
            for (int i = 0; i < store->key_dim; ++i)
                dot += query_key[i] * store->keys[index][i];
            semantic[index].score = dot;
        }
        semantic[index].index = index;
    }
    qsort(lexical, store->count, sizeof *lexical, compare_score_desc);
    qsort(semantic, store->count, sizeof *semantic, compare_score_desc);
    for (size_t rank = 0; rank < store->count; ++rank) {
        lexical_rank[lexical[rank].index] = (int)rank;
        semantic_rank[semantic[rank].index] = (int)rank;
    }
    for (size_t index = 0; index < store->count; ++index) {
        combined[index].index = index;
        combined[index].score =
            lexical_weight /
                (60.0f + (float)lexical_rank[index]) +
            (1.0f - lexical_weight) /
                (60.0f + (float)semantic_rank[index]);
    }
    qsort(combined, store->count, sizeof *combined, compare_score_desc);
    for (int i = 0; i < max_results; ++i)
        indices[i] = combined[i].index;
    free(lexical);
    free(semantic);
    free(combined);
    free(lexical_rank);
    free(semantic_rank);
    return max_results;
fail:
    free(lexical);
    free(semantic);
    free(combined);
    free(lexical_rank);
    free(semantic_rank);
    return -1;
}

char *metis_episodic_build_hybrid_context(
    const metis_episodic_store_t *store, const char *query,
    const float *query_key, float lexical_weight, int top_k) {
    size_t *indices;
    size_t bytes = 1, offset = 0;
    char *result;
    int found;
    if (store == NULL || query == NULL || store->count == 0 ||
        top_k <= 0) return NULL;
    if ((size_t)top_k > store->count) top_k = (int)store->count;
    indices = (size_t *)malloc((size_t)top_k * sizeof *indices);
    if (indices == NULL) return NULL;
    found = metis_episodic_rank_hybrid(
        store, query, query_key, lexical_weight, indices, top_k);
    if (found <= 0) {
        free(indices);
        return NULL;
    }
    for (int i = 0; i < found; ++i)
        bytes += strlen(store->records[indices[i]]) + 32;
    result = (char *)malloc(bytes);
    if (result == NULL) {
        free(indices);
        return NULL;
    }
    for (int i = 0; i < found; ++i) {
        int written = snprintf(
            result + offset, bytes - offset, "[memory %d] %s\n",
            i + 1, store->records[indices[i]]);
        if (written < 0 || (size_t)written >= bytes - offset) {
            free(result);
            result = NULL;
            break;
        }
        offset += (size_t)written;
    }
    free(indices);
    return result;
}

int metis_episodic_save(
    const metis_episodic_store_t *store, const char *path) {
    FILE *file;
    if (store == NULL || path == NULL ||
        store->count > EPISODIC_MAX_RECORDS) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    if (fwrite(k_magic, 1, sizeof k_magic, file) != sizeof k_magic ||
        write_u32(file, 2) != 0 ||
        write_u32(file, (uint32_t)store->count) != 0) goto fail;
    for (size_t i = 0; i < store->count; ++i) {
        size_t length = strlen(store->records[i]);
        if (length == 0 || length > EPISODIC_MAX_RECORD_BYTES ||
            write_u32(file, (uint32_t)length) != 0 ||
            write_u32(
                file, crc32_bytes(store->records[i], length)) != 0 ||
            fwrite(store->records[i], 1, length, file) != length) goto fail;
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
    uint32_t version = 0, count = 0;
    if (store == NULL || path == NULL) return -1;
    metis_episodic_init(&loaded);
    file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (fread(magic, 1, sizeof magic, file) != sizeof magic ||
        memcmp(magic, k_magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 ||
        (version != 1 && version != 2) ||
        read_u32(file, &count) != 0 || count > EPISODIC_MAX_RECORDS)
        goto fail;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t length = 0;
        uint32_t expected_crc = 0;
        char *text;
        if (read_u32(file, &length) != 0 || length == 0 ||
            length > EPISODIC_MAX_RECORD_BYTES) goto fail;
        if (version >= 2 && read_u32(file, &expected_crc) != 0) goto fail;
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
        if (metis_episodic_add(&loaded, text) != 0) {
            free(text);
            goto fail;
        }
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
