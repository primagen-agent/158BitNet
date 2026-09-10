#include "episode_pointer.h"

#include "sha256.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(FILE *file, void *output, size_t size) {
    return fread(output, 1, size, file) == size ? 0 : -1;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (read_exact(file, bytes, sizeof bytes) != 0) return -1;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 0;
}

static int read_f32(FILE *file, float *value) {
    uint32_t bits;
    if (read_u32(file, &bits) != 0) return -1;
    memcpy(value, &bits, sizeof bits);
    return 0;
}

static uint32_t crc32_bytes(const void *data, size_t size) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t index = 0; index < size; ++index) {
        crc ^= bytes[index];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^
                  (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

static void *allocate_tensor(
    FILE *file, uint32_t size, uint32_t expected, uint32_t crc) {
    void *data;
    if (size != expected) return NULL;
    data = malloc(size);
    if (data == NULL || read_exact(file, data, size) != 0 ||
        crc32_bytes(data, size) != crc) {
        free(data);
        return NULL;
    }
    return data;
}

metis_episode_pointer_t *metis_episode_pointer_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size) {
    static const unsigned char magic[8] =
        {'B', 'N', 'E', 'P', 'T', 'R', '1', 0};
    FILE *file = NULL;
    metis_episode_pointer_t *pointer = NULL;
    unsigned char actual_magic[8];
    uint8_t actual_sha256[32];
    uint32_t version, hidden, rank, max_span, tensor_count;
    uint32_t sizes[9], crcs[9];
    size_t matrix_bytes, vector_bytes;
#define FAIL(...) do { \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error != NULL && error_size > 0) error[0] = '\0';
    if (path == NULL || backbone_path == NULL || expected_hidden <= 0)
        FAIL("invalid episode pointer arguments");
    file = fopen(path, "rb");
    if (file == NULL) FAIL("cannot open episode pointer");
    if (read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &rank) != 0 ||
        read_u32(file, &max_span) != 0)
        FAIL("bad episode pointer header");
    if (hidden != (uint32_t)expected_hidden || hidden > 16384u ||
        rank < 1 || rank > 4096u || max_span < 1 || max_span > 4096u)
        FAIL("episode pointer geometry mismatch");
    pointer = (metis_episode_pointer_t *)calloc(1, sizeof *pointer);
    if (pointer == NULL) FAIL("out of memory");
    pointer->hidden_dim = (int)hidden;
    pointer->rank = (int)rank;
    pointer->max_span = (int)max_span;
    if (read_exact(
            file, pointer->backbone_sha256,
            sizeof pointer->backbone_sha256) != 0 ||
        read_u32(file, &tensor_count) != 0 || tensor_count != 9u)
        FAIL("truncated episode pointer header");
    for (uint32_t index = 0; index < tensor_count; ++index)
        if (read_u32(file, &sizes[index]) != 0 ||
            read_u32(file, &crcs[index]) != 0)
            FAIL("truncated episode pointer tensor table");
    if (bitnet_sha256_file(backbone_path, actual_sha256) != 0 ||
        memcmp(actual_sha256, pointer->backbone_sha256,
               sizeof actual_sha256) != 0)
        FAIL("episode pointer backbone SHA-256 mismatch");
    matrix_bytes = (size_t)hidden * (size_t)rank * sizeof(float);
    vector_bytes = (size_t)hidden * sizeof(float);
    pointer->query_start = (float *)allocate_tensor(
        file, sizes[0], (uint32_t)matrix_bytes, crcs[0]);
    pointer->query_end = (float *)allocate_tensor(
        file, sizes[1], (uint32_t)matrix_bytes, crcs[1]);
    pointer->source_start = (float *)allocate_tensor(
        file, sizes[2], (uint32_t)matrix_bytes, crcs[2]);
    pointer->source_end = (float *)allocate_tensor(
        file, sizes[3], (uint32_t)matrix_bytes, crcs[3]);
    pointer->local_start_weight = (float *)allocate_tensor(
        file, sizes[4], (uint32_t)vector_bytes, crcs[4]);
    if (sizes[5] != sizeof(float) ||
        read_f32(file, &pointer->local_start_bias) != 0 ||
        crc32_bytes(
            &pointer->local_start_bias, sizeof(float)) != crcs[5])
        FAIL("bad episode pointer start bias");
    pointer->local_end_weight = (float *)allocate_tensor(
        file, sizes[6], (uint32_t)vector_bytes, crcs[6]);
    if (sizes[7] != sizeof(float) ||
        read_f32(file, &pointer->local_end_bias) != 0 ||
        crc32_bytes(
            &pointer->local_end_bias, sizeof(float)) != crcs[7] ||
        sizes[8] != 2u * sizeof(float) ||
        read_exact(file, pointer->local_scale,
                   sizeof pointer->local_scale) != 0 ||
        crc32_bytes(
            pointer->local_scale,
            sizeof pointer->local_scale) != crcs[8])
        FAIL("bad episode pointer scalar tensors");
    if (pointer->query_start == NULL || pointer->query_end == NULL ||
        pointer->source_start == NULL || pointer->source_end == NULL ||
        pointer->local_start_weight == NULL ||
        pointer->local_end_weight == NULL || fgetc(file) != EOF)
        FAIL("bad episode pointer tensors");
    fclose(file);
    return pointer;
fail:
    if (file != NULL) fclose(file);
    metis_episode_pointer_free(pointer);
    return NULL;
#undef FAIL
}

void metis_episode_pointer_free(metis_episode_pointer_t *pointer) {
    if (pointer == NULL) return;
    free(pointer->query_start);
    free(pointer->query_end);
    free(pointer->source_start);
    free(pointer->source_end);
    free(pointer->local_start_weight);
    free(pointer->local_end_weight);
    free(pointer);
}

static void project_normalized(
    const float *input, const float *weight,
    int hidden, int rank, float *output) {
    float norm = 0.0f;
    for (int row = 0; row < rank; ++row) {
        float value = 0.0f;
        const float *weights = weight + (size_t)row * (size_t)hidden;
        for (int column = 0; column < hidden; ++column)
            value += weights[column] * input[column];
        output[row] = value;
        norm += value * value;
    }
    norm = sqrtf(norm);
    if (norm < 1e-12f) norm = 1e-12f;
    for (int row = 0; row < rank; ++row) output[row] /= norm;
}

static float dot(const float *left, const float *right, int count) {
    float value = 0.0f;
    for (int index = 0; index < count; ++index)
        value += left[index] * right[index];
    return value;
}

static float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

int metis_episode_pointer_select(
    const metis_episode_pointer_t *pointer,
    const float *source_hidden, size_t source_count,
    const float *query_hidden, size_t query_count,
    size_t *start, size_t *end, float *score) {
    float *query_start = NULL, *query_end = NULL;
    float *source_start = NULL, *source_end = NULL;
    float *start_scores = NULL, *end_scores = NULL;
    float best = -FLT_MAX;
    size_t best_start = 0, best_end = 0;
    const int hidden = pointer == NULL ? 0 : pointer->hidden_dim;
    const int rank = pointer == NULL ? 0 : pointer->rank;
    if (pointer == NULL || source_hidden == NULL || query_hidden == NULL ||
        source_count == 0 || query_count == 0 ||
        start == NULL || end == NULL || score == NULL)
        return -1;
    query_start = (float *)malloc(
        query_count * (size_t)rank * sizeof(float));
    query_end = (float *)malloc(
        query_count * (size_t)rank * sizeof(float));
    source_start = (float *)malloc((size_t)rank * sizeof(float));
    source_end = (float *)malloc((size_t)rank * sizeof(float));
    start_scores = (float *)malloc(source_count * sizeof(float));
    end_scores = (float *)malloc(source_count * sizeof(float));
    if (query_start == NULL || query_end == NULL ||
        source_start == NULL || source_end == NULL ||
        start_scores == NULL || end_scores == NULL)
        goto fail;
    for (size_t query = 0; query < query_count; ++query) {
        const float *row = query_hidden + query * (size_t)hidden;
        project_normalized(
            row, pointer->query_start, hidden, rank,
            query_start + query * (size_t)rank);
        project_normalized(
            row, pointer->query_end, hidden, rank,
            query_end + query * (size_t)rank);
    }
    for (size_t token = 0; token < source_count; ++token) {
        const float *row = source_hidden + token * (size_t)hidden;
        float start_match = -FLT_MAX, end_match = -FLT_MAX;
        float local_start = pointer->local_start_bias;
        float local_end = pointer->local_end_bias;
        project_normalized(
            row, pointer->source_start, hidden, rank, source_start);
        project_normalized(
            row, pointer->source_end, hidden, rank, source_end);
        for (size_t query = 0; query < query_count; ++query) {
            float candidate = dot(
                query_start + query * (size_t)rank,
                source_start, rank);
            if (candidate > start_match) start_match = candidate;
            candidate = dot(
                query_end + query * (size_t)rank,
                source_end, rank);
            if (candidate > end_match) end_match = candidate;
        }
        for (int column = 0; column < hidden; ++column) {
            local_start += pointer->local_start_weight[column] * row[column];
            local_end += pointer->local_end_weight[column] * row[column];
        }
        start_scores[token] =
            start_match * sqrtf((float)rank) +
            sigmoid(pointer->local_scale[0]) * local_start;
        end_scores[token] =
            end_match * sqrtf((float)rank) +
            sigmoid(pointer->local_scale[1]) * local_end;
    }
    for (size_t left = 0; left < source_count; ++left) {
        size_t limit = left + (size_t)pointer->max_span;
        if (limit > source_count) limit = source_count;
        for (size_t right = left; right < limit; ++right) {
            float candidate = start_scores[left] + end_scores[right];
            if (candidate > best) {
                best = candidate;
                best_start = left;
                best_end = right;
            }
        }
    }
    free(query_start);
    free(query_end);
    free(source_start);
    free(source_end);
    free(start_scores);
    free(end_scores);
    *start = best_start;
    *end = best_end;
    *score = best;
    return 1;
fail:
    free(query_start);
    free(query_end);
    free(source_start);
    free(source_end);
    free(start_scores);
    free(end_scores);
    return -1;
}
