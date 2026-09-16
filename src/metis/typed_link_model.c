#include "typed_link_model.h"

#include "sha256.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TYPED_LINK_TENSOR_COUNT = 8,
    TYPED_LINK_EXISTS_FEATURES = 5
};

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

static int checked_product(
    size_t left, size_t right, size_t *output) {
    if (right != 0 && left > SIZE_MAX / right) return -1;
    *output = left * right;
    return 0;
}

static int tensor_bytes(
    size_t rows, size_t columns, uint32_t *output) {
    size_t elements;
    size_t bytes;
    if (checked_product(rows, columns, &elements) != 0 ||
        checked_product(elements, sizeof(float), &bytes) != 0 ||
        bytes > UINT32_MAX)
        return -1;
    *output = (uint32_t)bytes;
    return 0;
}

static float *read_tensor(
    FILE *file, uint32_t size, uint32_t expected, uint32_t crc) {
    float *data;
    if (size != expected || size == 0) return NULL;
    data = (float *)malloc(size);
    if (data == NULL || read_exact(file, data, size) != 0 ||
        crc32_bytes(data, size) != crc) {
        free(data);
        return NULL;
    }
    return data;
}

metis_typed_link_model_t *metis_typed_link_model_load(
    const char *path, const char *backbone_path,
    char *error, size_t error_size) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'L', 'I', 'N', 'K', '1'};
    FILE *file = NULL;
    metis_typed_link_model_t *model = NULL;
    unsigned char actual_magic[8];
    uint8_t actual_sha256[32];
    uint32_t version;
    uint32_t set_link_version;
    uint32_t rank;
    uint32_t feature_dim;
    uint32_t joint_hidden;
    uint32_t exists_features;
    uint32_t tensor_count;
    uint32_t sizes[TYPED_LINK_TENSOR_COUNT];
    uint32_t crcs[TYPED_LINK_TENSOR_COUNT];
    uint32_t expected[TYPED_LINK_TENSOR_COUNT];
#define FAIL(...) do { \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error != NULL && error_size > 0) error[0] = '\0';
    if (path == NULL || backbone_path == NULL)
        FAIL("invalid typed-link model arguments");
    file = fopen(path, "rb");
    if (file == NULL) FAIL("cannot open typed-link model");
    if (read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &set_link_version) != 0 ||
        set_link_version != 2 ||
        read_u32(file, &rank) != 0 ||
        read_u32(file, &feature_dim) != 0 ||
        read_u32(file, &joint_hidden) != 0 ||
        read_u32(file, &exists_features) != 0 ||
        read_u32(file, &tensor_count) != 0)
        FAIL("bad typed-link model header");
    if (rank < 1 || rank > 4096u ||
        feature_dim < 1 || feature_dim > 16384u ||
        joint_hidden != rank * 2u ||
        exists_features != TYPED_LINK_EXISTS_FEATURES ||
        tensor_count != TYPED_LINK_TENSOR_COUNT)
        FAIL("typed-link model geometry mismatch");
    model = (metis_typed_link_model_t *)calloc(
        1, sizeof *model);
    if (model == NULL) FAIL("out of memory");
    model->rank = (int)rank;
    model->pair_feature_dim = (int)feature_dim;
    model->joint_hidden_dim = (int)joint_hidden;
    if (read_exact(file, model->backbone_sha256,
                   sizeof model->backbone_sha256) != 0)
        FAIL("truncated typed-link model header");
    for (uint32_t index = 0; index < tensor_count; ++index)
        if (read_u32(file, &sizes[index]) != 0 ||
            read_u32(file, &crcs[index]) != 0)
            FAIL("truncated typed-link tensor table");
    if (bitnet_sha256_file(backbone_path, actual_sha256) != 0 ||
        memcmp(actual_sha256, model->backbone_sha256,
               sizeof actual_sha256) != 0)
        FAIL("typed-link backbone SHA-256 mismatch");
    if (tensor_bytes(joint_hidden, feature_dim * 2u, &expected[0]) != 0 ||
        tensor_bytes(joint_hidden, 1, &expected[1]) != 0 ||
        tensor_bytes(1, joint_hidden, &expected[2]) != 0 ||
        tensor_bytes(1, 1, &expected[3]) != 0 ||
        tensor_bytes(rank, exists_features, &expected[4]) != 0 ||
        tensor_bytes(rank, 1, &expected[5]) != 0 ||
        tensor_bytes(1, rank, &expected[6]) != 0 ||
        tensor_bytes(1, 1, &expected[7]) != 0)
        FAIL("typed-link tensor geometry overflow");
#define LOAD(index, field) do { \
    model->field = read_tensor( \
        file, sizes[index], expected[index], crcs[index]); \
    if (model->field == NULL) \
        FAIL("bad typed-link tensor %d", index); \
} while (0)
    LOAD(0, joint_hidden_weight);
    LOAD(1, joint_hidden_bias);
    LOAD(2, joint_output_weight);
#undef LOAD
#define LOAD_SCALAR(index, field) do { \
    float *scalar = read_tensor( \
        file, sizes[index], expected[index], crcs[index]); \
    if (scalar == NULL) \
        FAIL("bad typed-link tensor %d", index); \
    model->field = scalar[0]; \
    free(scalar); \
} while (0)
    LOAD_SCALAR(3, joint_output_bias);
#define LOAD(index, field) do { \
    model->field = read_tensor( \
        file, sizes[index], expected[index], crcs[index]); \
    if (model->field == NULL) \
        FAIL("bad typed-link tensor %d", index); \
} while (0)
    LOAD(4, exists_hidden_weight);
    LOAD(5, exists_hidden_bias);
    LOAD(6, exists_output_weight);
#undef LOAD
    LOAD_SCALAR(7, exists_output_bias);
#undef LOAD_SCALAR
    if (!isfinite(model->joint_output_bias) ||
        !isfinite(model->exists_output_bias))
        FAIL("non-finite typed-link scalar");
    if (fgetc(file) != EOF)
        FAIL("typed-link model has trailing data");
    fclose(file);
    return model;
fail:
    if (file != NULL) fclose(file);
    metis_typed_link_model_free(model);
    return NULL;
#undef FAIL
}

void metis_typed_link_model_free(
    metis_typed_link_model_t *model) {
    if (model == NULL) return;
    free(model->joint_hidden_weight);
    free(model->joint_hidden_bias);
    free(model->joint_output_weight);
    free(model->exists_hidden_weight);
    free(model->exists_hidden_bias);
    free(model->exists_output_weight);
    free(model);
}

static float gelu(float value) {
    return 0.5f * value *
           (1.0f + erff(value * 0.7071067811865475f));
}

static float dense_gelu_row(
    const float *weight, const float *input,
    size_t count, float bias) {
    float value = bias;
    for (size_t index = 0; index < count; ++index)
        value += weight[index] * input[index];
    return gelu(value);
}

int metis_typed_link_score_pairs(
    const metis_typed_link_model_t *model,
    const float *entity_features,
    const float *predicate_features,
    const float *entity_logits,
    const float *predicate_logits,
    size_t pair_count,
    float *joint_scores) {
    size_t feature_dim;
    size_t input_dim;
    float *hidden;
    if (model == NULL || entity_features == NULL ||
        predicate_features == NULL || entity_logits == NULL ||
        predicate_logits == NULL || pair_count == 0 ||
        joint_scores == NULL)
        return -1;
    feature_dim = (size_t)model->pair_feature_dim;
    input_dim = feature_dim * 2u;
    hidden = (float *)malloc(
        (size_t)model->joint_hidden_dim * sizeof(float));
    if (hidden == NULL) return -1;
    for (size_t pair = 0; pair < pair_count; ++pair) {
        const float *entity =
            entity_features + pair * feature_dim;
        const float *predicate =
            predicate_features + pair * feature_dim;
        for (int row = 0; row < model->joint_hidden_dim; ++row) {
            const float *weight = model->joint_hidden_weight +
                (size_t)row * input_dim;
            float value = model->joint_hidden_bias[row];
            for (size_t column = 0; column < feature_dim; ++column)
                value += weight[column] * entity[column];
            for (size_t column = 0; column < feature_dim; ++column)
                value += weight[feature_dim + column] *
                         predicate[column];
            hidden[row] = gelu(value);
        }
        float residual = model->joint_output_bias;
        for (int row = 0; row < model->joint_hidden_dim; ++row)
            residual += model->joint_output_weight[row] * hidden[row];
        joint_scores[pair] =
            fminf(entity_logits[pair], predicate_logits[pair]) +
            residual;
    }
    free(hidden);
    return 0;
}

int metis_typed_link_predecessor_exists(
    const metis_typed_link_model_t *model,
    const float *joint_scores,
    size_t pair_count,
    float *exists_score) {
    float features[TYPED_LINK_EXISTS_FEATURES];
    float *normalized = NULL;
    float *hidden = NULL;
    float top1 = -FLT_MAX;
    float top2 = -FLT_MAX;
    float mean = 0.0f;
    float variance = 0.0f;
    float stddev;
    float maximum = -FLT_MAX;
    float total = 0.0f;
    float entropy = 0.0f;
    if (model == NULL || joint_scores == NULL ||
        pair_count == 0 || exists_score == NULL)
        return -1;
    normalized = (float *)malloc(pair_count * sizeof(float));
    hidden = (float *)malloc((size_t)model->rank * sizeof(float));
    if (normalized == NULL || hidden == NULL) {
        free(normalized);
        free(hidden);
        return -1;
    }
    for (size_t index = 0; index < pair_count; ++index) {
        const float value = joint_scores[index];
        mean += value;
        if (value > top1) {
            top2 = top1;
            top1 = value;
        } else if (value > top2) {
            top2 = value;
        }
    }
    mean /= (float)pair_count;
    for (size_t index = 0; index < pair_count; ++index) {
        const float delta = joint_scores[index] - mean;
        variance += delta * delta;
    }
    stddev = sqrtf(variance / (float)pair_count);
    if (stddev < 1e-4f) stddev = 1e-4f;
    if (pair_count == 1) top2 = top1;
    for (size_t index = 0; index < pair_count; ++index) {
        normalized[index] = (joint_scores[index] - mean) / stddev;
        if (normalized[index] > maximum)
            maximum = normalized[index];
    }
    for (size_t index = 0; index < pair_count; ++index) {
        normalized[index] = expf(normalized[index] - maximum);
        total += normalized[index];
    }
    if (total < 1e-20f) total = 1e-20f;
    for (size_t index = 0; index < pair_count; ++index) {
        const float probability = normalized[index] / total;
        if (probability > 0.0f)
            entropy -= probability * logf(probability);
        normalized[index] = probability;
    }
    features[0] = (top1 - mean) / stddev;
    features[1] = (top1 - top2) / stddev;
    features[2] = 0.0f;
    for (size_t index = 0; index < pair_count; ++index)
        if (normalized[index] > features[2])
            features[2] = normalized[index];
    features[3] = pair_count > 1
        ? 1.0f - entropy / logf((float)pair_count)
        : 1.0f;
    features[4] = log1pf((float)pair_count);
    for (int row = 0; row < model->rank; ++row)
        hidden[row] = dense_gelu_row(
            model->exists_hidden_weight +
                (size_t)row * TYPED_LINK_EXISTS_FEATURES,
            features, TYPED_LINK_EXISTS_FEATURES,
            model->exists_hidden_bias[row]);
    *exists_score = model->exists_output_bias;
    for (int row = 0; row < model->rank; ++row)
        *exists_score +=
            model->exists_output_weight[row] * hidden[row];
    free(normalized);
    free(hidden);
    return isfinite(*exists_score) ? 0 : -1;
}

int metis_typed_link_select_predecessor(
    const metis_typed_link_model_t *model,
    const float *joint_scores,
    size_t pair_count,
    size_t *selected_index,
    float *exists_score) {
    size_t best = 0;
    if (selected_index == NULL ||
        metis_typed_link_predecessor_exists(
            model, joint_scores, pair_count,
            exists_score) != 0)
        return -1;
    for (size_t index = 1; index < pair_count; ++index)
        if (joint_scores[index] > joint_scores[best])
            best = index;
    *selected_index = *exists_score > 0.0f ? best : SIZE_MAX;
    return 0;
}
