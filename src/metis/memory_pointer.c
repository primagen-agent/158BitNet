#include "memory_pointer.h"

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
    uint32_t rank = 0, max_token_bytes = 0;
    uint32_t sizes[11] = {0}, crcs[11] = {0};
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
         memcmp(magic, "BNPTR2\0\0", sizeof magic) != 0 &&
         memcmp(magic, "BNPTR5\0\0", sizeof magic) != 0))
        FAIL("bad pointer magic");
    if (read_u32(file, &version) != 0 ||
        (version != 1 && version != 2 && version != 5) ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &max_span) != 0)
        FAIL("bad pointer header");
    pointer = (metis_memory_pointer_t *)calloc(1, sizeof *pointer);
    if (pointer == NULL) FAIL("out of memory");
    pointer->version = (int)version;
    pointer->hidden_dim = (int)hidden;
    pointer->max_span = (int)max_span;
    if (hidden != (uint32_t)expected_hidden || hidden > 16384u ||
        max_span < 1 || max_span > (version == 5 ? 65536u : 1024u))
        FAIL("pointer geometry mismatch");
    if (read_f32(file, &pointer->threshold) != 0)
        FAIL("truncated pointer header");
    if (version == 5 &&
        (read_u32(file, &rank) != 0 ||
         read_u32(file, &max_token_bytes) != 0 ||
         rank < 1 || rank > 4096u ||
         max_token_bytes < 1 || max_token_bytes > 1024u))
        FAIL("bad byte pointer geometry");
    pointer->byte_mode = version == 5;
    pointer->rank = (int)rank;
    pointer->max_token_bytes = (int)max_token_bytes;
    if (
        read_exact(file, pointer->backbone_sha256,
                   sizeof pointer->backbone_sha256) != 0 ||
        read_u32(file, &tensor_count) != 0 ||
        tensor_count != (
            version == 1 ? 4u : (version == 2 ? 6u : 11u)))
        FAIL("truncated pointer header");
    for (uint32_t i = 0; i < tensor_count; ++i) {
        if (read_u32(file, &sizes[i]) != 0 ||
            read_u32(file, &crcs[i]) != 0)
            FAIL("truncated pointer tensor table");
    }
    weight_bytes = (size_t)hidden * sizeof(float);
    null_weight_bytes = 4u * weight_bytes;
    if (version < 5) {
        if (sizes[0] != weight_bytes || sizes[1] != sizeof(float) ||
            sizes[2] != weight_bytes || sizes[3] != sizeof(float) ||
            (version == 2 &&
             (sizes[4] != null_weight_bytes ||
              sizes[5] != 2u * sizeof(float))))
            FAIL("pointer tensor size mismatch");
    } else {
        const size_t rank_bytes = (size_t)rank * sizeof(float);
        if (sizes[0] != (size_t)rank * weight_bytes ||
            sizes[1] != 257u * rank_bytes ||
            sizes[2] != 257u * rank_bytes ||
            sizes[3] != 257u * rank_bytes ||
            sizes[4] != (size_t)max_token_bytes * rank_bytes ||
            sizes[5] != rank_bytes || sizes[6] != sizeof(float) ||
            sizes[7] != rank_bytes || sizes[8] != sizeof(float) ||
            sizes[9] != rank_bytes || sizes[10] != sizeof(float))
            FAIL("byte pointer tensor size mismatch");
    }
    if (bitnet_sha256_file(backbone_path, actual_sha256) != 0 ||
        memcmp(actual_sha256, pointer->backbone_sha256,
               sizeof actual_sha256) != 0)
        FAIL("pointer backbone SHA-256 mismatch");
    if (version < 5) {
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
            (read_exact(
                 file, pointer->null_weight, null_weight_bytes) != 0 ||
             read_exact(file, pointer->null_bias,
                        sizeof pointer->null_bias) != 0))
            FAIL("truncated pointer null tensors");
    } else {
#define ALLOC_TENSOR(field, index) do { \
    pointer->field = (float *)malloc(sizes[index]); \
    if (pointer->field == NULL || \
        read_exact(file, pointer->field, sizes[index]) != 0) \
        FAIL("truncated byte pointer tensors"); \
} while (0)
        ALLOC_TENSOR(token_weight, 0);
        ALLOC_TENSOR(byte_weight, 1);
        ALLOC_TENSOR(previous_byte_weight, 2);
        ALLOC_TENSOR(next_byte_weight, 3);
        ALLOC_TENSOR(offset_weight, 4);
        ALLOC_TENSOR(start_weight, 5);
        if (read_f32(file, &pointer->start_bias) != 0)
            FAIL("truncated byte pointer start bias");
        ALLOC_TENSOR(end_weight, 7);
        if (read_f32(file, &pointer->end_bias) != 0)
            FAIL("truncated byte pointer end bias");
        ALLOC_TENSOR(inside_weight, 9);
        if (read_f32(file, &pointer->inside_bias) != 0)
            FAIL("truncated byte pointer inside bias");
#undef ALLOC_TENSOR
    }
    if (fgetc(file) != EOF) FAIL("trailing pointer data");
    if (version < 5) {
        if (crc32_bytes(pointer->start_weight, weight_bytes) != crcs[0] ||
            crc32_bytes(&pointer->start_bias, sizeof(float)) != crcs[1] ||
            crc32_bytes(pointer->end_weight, weight_bytes) != crcs[2] ||
            crc32_bytes(&pointer->end_bias, sizeof(float)) != crcs[3] ||
            (version == 2 &&
             (crc32_bytes(
                  pointer->null_weight, null_weight_bytes) != crcs[4] ||
              crc32_bytes(pointer->null_bias,
                          sizeof pointer->null_bias) != crcs[5])))
            FAIL("pointer CRC mismatch");
    } else {
        const void *payloads[11] = {
            pointer->token_weight, pointer->byte_weight,
            pointer->previous_byte_weight, pointer->next_byte_weight,
            pointer->offset_weight, pointer->start_weight,
            &pointer->start_bias, pointer->end_weight,
            &pointer->end_bias, pointer->inside_weight,
            &pointer->inside_bias,
        };
        for (uint32_t i = 0; i < 11; ++i)
            if (crc32_bytes(payloads[i], sizes[i]) != crcs[i])
                FAIL("byte pointer CRC mismatch");
    }
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
    free(pointer->token_weight);
    free(pointer->byte_weight);
    free(pointer->previous_byte_weight);
    free(pointer->next_byte_weight);
    free(pointer->offset_weight);
    free(pointer->inside_weight);
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
    if (pointer->byte_mode) return -1;
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

static float silu(float value) {
    return value / (1.0f + expf(-value));
}

int metis_memory_pointer_select_bytes(
    const metis_memory_pointer_t *pointer, const float *hidden,
    size_t token_count, const uint8_t *bytes,
    const size_t *byte_token_indices, const uint32_t *byte_token_offsets,
    size_t byte_count, size_t *start, size_t *end, float *confidence) {
    float *token_features = NULL;
    float *start_scores = NULL;
    float *end_scores = NULL;
    float *inside_prefix = NULL;
    float best = -FLT_MAX, second = -FLT_MAX;
    size_t best_start = 0, best_end = 0;
    if (pointer == NULL || !pointer->byte_mode || hidden == NULL ||
        token_count == 0 || bytes == NULL ||
        byte_token_indices == NULL || byte_token_offsets == NULL ||
        byte_count == 0 || start == NULL || end == NULL ||
        confidence == NULL)
        return -1;
    token_features = (float *)malloc(
        token_count * (size_t)pointer->rank * sizeof(float));
    start_scores = (float *)malloc(byte_count * sizeof(float));
    end_scores = (float *)malloc(byte_count * sizeof(float));
    inside_prefix = (float *)calloc(byte_count + 1u, sizeof(float));
    if (token_features == NULL || start_scores == NULL ||
        end_scores == NULL || inside_prefix == NULL)
        goto fail;
    for (size_t token = 0; token < token_count; ++token) {
        const float *row =
            hidden + token * (size_t)pointer->hidden_dim;
        for (int rank = 0; rank < pointer->rank; ++rank) {
            const float *weight = pointer->token_weight +
                (size_t)rank * (size_t)pointer->hidden_dim;
            float value = 0.0f;
            for (int column = 0; column < pointer->hidden_dim; ++column)
                value += row[column] * weight[column];
            token_features[
                token * (size_t)pointer->rank + (size_t)rank] = value;
        }
    }
    for (size_t byte = 0; byte < byte_count; ++byte) {
        const size_t token = byte_token_indices[byte];
        const uint32_t offset = byte_token_offsets[byte];
        const unsigned int previous = byte == 0 ? 256u : bytes[byte - 1u];
        const unsigned int next =
            byte + 1u == byte_count ? 256u : bytes[byte + 1u];
        float start_value = pointer->start_bias;
        float end_value = pointer->end_bias;
        float inside_value = pointer->inside_bias;
        if (token >= token_count ||
            offset >= (uint32_t)pointer->max_token_bytes)
            goto fail;
        for (int rank = 0; rank < pointer->rank; ++rank) {
            float feature =
                token_features[
                    token * (size_t)pointer->rank + (size_t)rank] +
                pointer->byte_weight[
                    (size_t)bytes[byte] * (size_t)pointer->rank +
                    (size_t)rank] +
                pointer->previous_byte_weight[
                    (size_t)previous * (size_t)pointer->rank +
                    (size_t)rank] +
                pointer->next_byte_weight[
                    (size_t)next * (size_t)pointer->rank +
                    (size_t)rank] +
                pointer->offset_weight[
                    (size_t)offset * (size_t)pointer->rank +
                    (size_t)rank];
            feature = silu(feature);
            start_value += feature * pointer->start_weight[rank];
            end_value += feature * pointer->end_weight[rank];
            inside_value += feature * pointer->inside_weight[rank];
        }
        start_scores[byte] = start_value;
        end_scores[byte] = end_value;
        inside_prefix[byte + 1u] = inside_prefix[byte] + inside_value;
    }
    for (size_t span_start = 0; span_start < byte_count; ++span_start) {
        size_t limit = span_start + (size_t)pointer->max_span;
        if (limit > byte_count) limit = byte_count;
        for (size_t span_end = span_start; span_end < limit; ++span_end) {
            float score =
                start_scores[span_start] + end_scores[span_end] +
                inside_prefix[span_end + 1u] - inside_prefix[span_start];
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
    free(token_features);
    free(start_scores);
    free(end_scores);
    free(inside_prefix);
    *start = best_start;
    *end = best_end;
    *confidence = best - second;
    return best >= pointer->threshold ? 1 : 0;
fail:
    free(token_features);
    free(start_scores);
    free(end_scores);
    free(inside_prefix);
    return -1;
}
