#include "memory_pointer.h"

#include "sha256.h"

#include <float.h>
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
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^
                  (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

metis_memory_pointer_t *metis_memory_pointer_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size) {
    FILE *file = NULL;
    metis_memory_pointer_t *pointer = NULL;
    unsigned char magic[8];
    uint8_t actual_sha256[32];
    uint32_t version, hidden, max_span, tensor_count;
    uint32_t sizes[6] = {0}, crcs[6] = {0};
    size_t weight_bytes, null_weight_bytes;
#define FAIL(...) do { \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error != NULL && error_size > 0) error[0] = '\0';
    if (path == NULL || backbone_path == NULL || expected_hidden <= 0)
        FAIL("invalid pointer arguments");
    file = fopen(path, "rb");
    if (file == NULL) FAIL("cannot open pointer");
    if (read_exact(file, magic, sizeof magic) != 0 ||
        (memcmp(magic, "BNPTR1\0\0", sizeof magic) != 0 &&
         memcmp(magic, "BNPTR2\0\0", sizeof magic) != 0))
        FAIL("bad pointer magic");
    if (read_u32(file, &version) != 0 ||
        (version != 1 && version != 2) ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &max_span) != 0)
        FAIL("bad pointer header");
    pointer = (metis_memory_pointer_t *)calloc(1, sizeof *pointer);
    if (pointer == NULL) FAIL("out of memory");
    pointer->hidden_dim = (int)hidden;
    pointer->max_span = (int)max_span;
    if (hidden != (uint32_t)expected_hidden || max_span < 1 ||
        max_span > 1024)
        FAIL("pointer geometry mismatch");
    if (read_f32(file, &pointer->threshold) != 0 ||
        read_exact(file, pointer->backbone_sha256,
                   sizeof pointer->backbone_sha256) != 0 ||
        read_u32(file, &tensor_count) != 0 ||
        tensor_count != (version == 1 ? 4u : 6u))
        FAIL("truncated pointer header");
    for (uint32_t i = 0; i < tensor_count; ++i) {
        if (read_u32(file, &sizes[i]) != 0 ||
            read_u32(file, &crcs[i]) != 0)
            FAIL("truncated pointer tensor table");
    }
    weight_bytes = (size_t)hidden * sizeof(float);
    null_weight_bytes = 4u * weight_bytes;
    if (sizes[0] != weight_bytes || sizes[1] != sizeof(float) ||
        sizes[2] != weight_bytes || sizes[3] != sizeof(float) ||
        (version == 2 &&
         (sizes[4] != null_weight_bytes ||
          sizes[5] != 2u * sizeof(float))))
        FAIL("pointer tensor size mismatch");
    if (bitnet_sha256_file(backbone_path, actual_sha256) != 0 ||
        memcmp(actual_sha256, pointer->backbone_sha256,
               sizeof actual_sha256) != 0)
        FAIL("pointer backbone SHA-256 mismatch");
    pointer->start_weight = (float *)malloc(weight_bytes);
    pointer->end_weight = (float *)malloc(weight_bytes);
    if (version == 2) {
        pointer->has_null = 1;
        pointer->null_weight = (float *)malloc(null_weight_bytes);
    }
    if (pointer->start_weight == NULL || pointer->end_weight == NULL ||
        (version == 2 && pointer->null_weight == NULL))
        FAIL("out of memory");
    if (read_exact(file, pointer->start_weight, weight_bytes) != 0 ||
        read_f32(file, &pointer->start_bias) != 0 ||
        read_exact(file, pointer->end_weight, weight_bytes) != 0 ||
        read_f32(file, &pointer->end_bias) != 0)
        FAIL("truncated or trailing pointer data");
    if (version == 2 &&
        (read_exact(file, pointer->null_weight, null_weight_bytes) != 0 ||
         read_exact(file, pointer->null_bias,
                    sizeof pointer->null_bias) != 0))
        FAIL("truncated pointer null tensors");
    if (fgetc(file) != EOF) FAIL("trailing pointer data");
    if (crc32_bytes(pointer->start_weight, weight_bytes) != crcs[0] ||
        crc32_bytes(&pointer->start_bias, sizeof(float)) != crcs[1] ||
        crc32_bytes(pointer->end_weight, weight_bytes) != crcs[2] ||
        crc32_bytes(&pointer->end_bias, sizeof(float)) != crcs[3] ||
        (version == 2 &&
         (crc32_bytes(pointer->null_weight, null_weight_bytes) != crcs[4] ||
          crc32_bytes(pointer->null_bias,
                      sizeof pointer->null_bias) != crcs[5])))
        FAIL("pointer CRC mismatch");
    fclose(file);
    return pointer;
fail:
    if (file != NULL) fclose(file);
    metis_memory_pointer_free(pointer);
    return NULL;
#undef FAIL
}

void metis_memory_pointer_free(metis_memory_pointer_t *pointer) {
    if (pointer == NULL) return;
    free(pointer->start_weight);
    free(pointer->end_weight);
    free(pointer->null_weight);
    free(pointer);
}

static float score_token(
    const float *hidden, const float *weight, int hidden_dim, float bias) {
    float score = bias;
    for (int i = 0; i < hidden_dim; ++i) score += hidden[i] * weight[i];
    return score;
}

int metis_memory_pointer_select(
    const metis_memory_pointer_t *pointer, const float *hidden,
    size_t token_count, size_t *start, size_t *end, float *confidence) {
    float *start_scores = NULL, *end_scores = NULL;
    float best = -FLT_MAX, second = -FLT_MAX;
    float null_score = 0.0f;
    size_t best_start = 0, best_end = 0;
    if (pointer == NULL || hidden == NULL || token_count == 0 ||
        start == NULL || end == NULL || confidence == NULL)
        return -1;
    start_scores = (float *)malloc(token_count * sizeof(float));
    end_scores = (float *)malloc(token_count * sizeof(float));
    if (start_scores == NULL || end_scores == NULL) {
        free(start_scores);
        free(end_scores);
        return -1;
    }
    for (size_t token = 0; token < token_count; ++token) {
        const float *row =
            hidden + token * (size_t)pointer->hidden_dim;
        start_scores[token] = score_token(
            row, pointer->start_weight, pointer->hidden_dim,
            pointer->start_bias);
        end_scores[token] = score_token(
            row, pointer->end_weight, pointer->hidden_dim,
            pointer->end_bias);
    }
    if (pointer->has_null) {
        size_t pooled_dim = 2u * (size_t)pointer->hidden_dim;
        float *pooled = (float *)calloc(pooled_dim, sizeof(float));
        if (pooled == NULL) {
            free(start_scores);
            free(end_scores);
            return -1;
        }
        for (size_t token = 0; token < token_count; ++token)
            for (int column = 0; column < pointer->hidden_dim; ++column)
                pooled[column] +=
                    hidden[token * (size_t)pointer->hidden_dim +
                           (size_t)column] / (float)token_count;
        memcpy(
            pooled + pointer->hidden_dim,
            hidden + (token_count - 1u) * (size_t)pointer->hidden_dim,
            (size_t)pointer->hidden_dim * sizeof(float));
        for (int output = 0; output < 2; ++output) {
            float value = pointer->null_bias[output];
            const float *weight =
                pointer->null_weight + (size_t)output * pooled_dim;
            for (size_t column = 0; column < pooled_dim; ++column)
                value += weight[column] * pooled[column];
            null_score += value;
        }
        free(pooled);
    }
    for (size_t span_start = 0; span_start < token_count; ++span_start) {
        size_t limit = span_start + (size_t)pointer->max_span;
        if (limit > token_count) limit = token_count;
        for (size_t span_end = span_start; span_end < limit; ++span_end) {
            float score =
                start_scores[span_start] + end_scores[span_end];
            if (score > best) {
                second = best;
                best = score;
                best_start = span_start;
                best_end = span_end;
            } else if (score > second) {
                second = score;
            }
        }
    }
    free(start_scores);
    free(end_scores);
    *start = best_start;
    *end = best_end;
    *confidence = best - (pointer->has_null ? null_score : second);
    return *confidence >= pointer->threshold ? 1 : 0;
}
