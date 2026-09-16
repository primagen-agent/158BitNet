#include "typed_query_activator.h"

#include "sha256.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TYPED_QUERY_TENSOR_COUNT = 8,
    TYPED_QUERY_SET_FEATURES = 10
};

static int read_exact(FILE *file, void *output, size_t size) {
    return fread(output, 1, size, file) == size ? 0 : -1;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (read_exact(file, bytes, sizeof bytes) != 0) return -1;
    *value = (uint32_t)bytes[0] |
             ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) |
             ((uint32_t)bytes[3] << 24);
    return 0;
}

static uint32_t crc32_bytes(const void *data, size_t size) {
    const unsigned char *bytes = (const unsigned char *)data;
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

static int tensor_bytes(
    size_t rows, size_t columns, uint32_t *output) {
    if (columns != 0 && rows > SIZE_MAX / columns)
        return -1;
    size_t elements = rows * columns;
    if (elements > SIZE_MAX / sizeof(float) ||
        elements * sizeof(float) > UINT32_MAX)
        return -1;
    *output = (uint32_t)(elements * sizeof(float));
    return 0;
}

static float *read_tensor(
    FILE *file, uint32_t size,
    uint32_t expected, uint32_t crc) {
    float *data;
    if (size != expected || size == 0) return NULL;
    data = (float *)malloc(size);
    if (data == NULL ||
        read_exact(file, data, size) != 0 ||
        crc32_bytes(data, size) != crc) {
        free(data);
        return NULL;
    }
    return data;
}

metis_typed_query_activator_t *
metis_typed_query_activator_load(
    const char *path,
    const char *backbone_path,
    const char *pair_model_path,
    int expected_input_width,
    char *error,
    size_t error_size) {
    static const unsigned char magic[8] =
        {'B','N','T','Q','A','C','T','1'};
    FILE *file = NULL;
    metis_typed_query_activator_t *model = NULL;
    unsigned char actual_magic[8];
    uint8_t actual_sha256[32];
    uint32_t version, rank, input_width;
    uint32_t candidate_hidden, set_features;
    uint32_t null_hidden, tensor_count;
    uint32_t sizes[TYPED_QUERY_TENSOR_COUNT];
    uint32_t crcs[TYPED_QUERY_TENSOR_COUNT];
    uint32_t expected[TYPED_QUERY_TENSOR_COUNT];
#define FAIL(...) do { \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error != NULL && error_size > 0) error[0] = '\0';
    if (path == NULL || backbone_path == NULL ||
        pair_model_path == NULL || expected_input_width <= 0)
        FAIL("invalid typed-query model arguments");
    file = fopen(path, "rb");
    if (file == NULL) FAIL("cannot open typed-query model");
    if (read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &rank) != 0 ||
        read_u32(file, &input_width) != 0 ||
        read_u32(file, &candidate_hidden) != 0 ||
        read_u32(file, &set_features) != 0 ||
        read_u32(file, &null_hidden) != 0 ||
        read_u32(file, &tensor_count) != 0)
        FAIL("bad typed-query model header");
    if (rank < 1 || rank > 4096u ||
        input_width != (uint32_t)expected_input_width ||
        candidate_hidden != rank * 2u ||
        set_features != TYPED_QUERY_SET_FEATURES ||
        null_hidden != rank ||
        tensor_count != TYPED_QUERY_TENSOR_COUNT)
        FAIL("typed-query model geometry mismatch");
    model = (metis_typed_query_activator_t *)calloc(
        1, sizeof *model);
    if (model == NULL) FAIL("out of memory");
    model->rank = (int)rank;
    model->input_width = (int)input_width;
    model->candidate_hidden = (int)candidate_hidden;
    model->set_feature_count = (int)set_features;
    model->null_hidden = (int)null_hidden;
    if (read_exact(
            file, model->backbone_sha256,
            sizeof model->backbone_sha256) != 0 ||
        read_exact(
            file, model->pair_model_sha256,
            sizeof model->pair_model_sha256) != 0)
        FAIL("truncated typed-query model header");
    for (uint32_t index = 0;
         index < tensor_count; ++index)
        if (read_u32(file, &sizes[index]) != 0 ||
            read_u32(file, &crcs[index]) != 0)
            FAIL("truncated typed-query tensor table");
    if (bitnet_sha256_file(
            backbone_path, actual_sha256) != 0 ||
        memcmp(
            actual_sha256, model->backbone_sha256,
            sizeof actual_sha256) != 0)
        FAIL("typed-query backbone SHA-256 mismatch");
    if (bitnet_sha256_file(
            pair_model_path, actual_sha256) != 0 ||
        memcmp(
            actual_sha256, model->pair_model_sha256,
            sizeof actual_sha256) != 0)
        FAIL("typed-query pair-model SHA-256 mismatch");
    if (tensor_bytes(
            candidate_hidden, input_width,
            &expected[0]) != 0 ||
        tensor_bytes(
            candidate_hidden, 1, &expected[1]) != 0 ||
        tensor_bytes(
            1, candidate_hidden, &expected[2]) != 0 ||
        tensor_bytes(1, 1, &expected[3]) != 0 ||
        tensor_bytes(
            null_hidden, set_features,
            &expected[4]) != 0 ||
        tensor_bytes(null_hidden, 1, &expected[5]) != 0 ||
        tensor_bytes(1, null_hidden, &expected[6]) != 0 ||
        tensor_bytes(1, 1, &expected[7]) != 0)
        FAIL("typed-query tensor geometry overflow");
#define LOAD(index, field) do { \
    model->field = read_tensor( \
        file, sizes[index], expected[index], crcs[index]); \
    if (model->field == NULL) \
        FAIL("bad typed-query tensor %d", index); \
} while (0)
    LOAD(0, candidate_hidden_weight);
    LOAD(1, candidate_hidden_bias);
    LOAD(2, candidate_output_weight);
#undef LOAD
#define LOAD_SCALAR(index, field) do { \
    float *scalar = read_tensor( \
        file, sizes[index], expected[index], crcs[index]); \
    if (scalar == NULL) \
        FAIL("bad typed-query tensor %d", index); \
    model->field = scalar[0]; \
    free(scalar); \
} while (0)
    LOAD_SCALAR(3, candidate_output_bias);
#define LOAD(index, field) do { \
    model->field = read_tensor( \
        file, sizes[index], expected[index], crcs[index]); \
    if (model->field == NULL) \
        FAIL("bad typed-query tensor %d", index); \
} while (0)
    LOAD(4, null_hidden_weight);
    LOAD(5, null_hidden_bias);
    LOAD(6, null_output_weight);
#undef LOAD
    LOAD_SCALAR(7, null_output_bias);
#undef LOAD_SCALAR
    if (!isfinite(model->candidate_output_bias) ||
        !isfinite(model->null_output_bias))
        FAIL("non-finite typed-query scalar");
    if (fgetc(file) != EOF)
        FAIL("typed-query model has trailing data");
    fclose(file);
    return model;
fail:
    if (file != NULL) fclose(file);
    metis_typed_query_activator_free(model);
    return NULL;
#undef FAIL
}

void metis_typed_query_activator_free(
    metis_typed_query_activator_t *model) {
    if (model == NULL) return;
    free(model->candidate_hidden_weight);
    free(model->candidate_hidden_bias);
    free(model->candidate_output_weight);
    free(model->null_hidden_weight);
    free(model->null_hidden_bias);
    free(model->null_output_weight);
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

int metis_typed_query_select(
    const metis_typed_query_activator_t *model,
    const float *pair_features,
    const float *entity_logits,
    const float *predicate_logits,
    size_t pair_count,
    size_t *selected_index,
    float *activation_score,
    float *null_score) {
    float features[TYPED_QUERY_SET_FEATURES];
    float *scores = NULL;
    float *hidden = NULL;
    float *probabilities = NULL;
    float top1 = -FLT_MAX, top2 = -FLT_MAX;
    float mean = 0.0f, variance = 0.0f;
    float stddev, minimum = FLT_MAX;
    float maximum = -FLT_MAX, total = 0.0f;
    float entropy = 0.0f;
    float max_entity = -FLT_MAX;
    float max_predicate = -FLT_MAX;
    size_t best = 0;
    if (model == NULL || pair_features == NULL ||
        entity_logits == NULL || predicate_logits == NULL ||
        pair_count == 0 || selected_index == NULL ||
        activation_score == NULL || null_score == NULL)
        return -1;
    scores = (float *)malloc(pair_count * sizeof(float));
    hidden = (float *)malloc(
        (size_t)model->candidate_hidden * sizeof(float));
    probabilities = (float *)malloc(
        pair_count * sizeof(float));
    if (scores == NULL || hidden == NULL ||
        probabilities == NULL)
        goto fail;
    for (size_t pair = 0; pair < pair_count; ++pair) {
        const float *input = pair_features +
            pair * (size_t)model->input_width;
        for (int row = 0;
             row < model->candidate_hidden; ++row)
            hidden[row] = dense_gelu_row(
                model->candidate_hidden_weight +
                    (size_t)row *
                        (size_t)model->input_width,
                input, (size_t)model->input_width,
                model->candidate_hidden_bias[row]);
        float residual = model->candidate_output_bias;
        for (int row = 0;
             row < model->candidate_hidden; ++row)
            residual +=
                model->candidate_output_weight[row] *
                hidden[row];
        scores[pair] =
            fminf(entity_logits[pair],
                  predicate_logits[pair]) +
            residual;
        mean += scores[pair];
        if (scores[pair] < minimum)
            minimum = scores[pair];
        if (entity_logits[pair] > max_entity)
            max_entity = entity_logits[pair];
        if (predicate_logits[pair] > max_predicate)
            max_predicate = predicate_logits[pair];
        if (scores[pair] > top1) {
            top2 = top1;
            top1 = scores[pair];
            best = pair;
        } else if (scores[pair] > top2) {
            top2 = scores[pair];
        }
    }
    mean /= (float)pair_count;
    if (pair_count == 1) top2 = top1;
    for (size_t pair = 0; pair < pair_count; ++pair) {
        const float delta = scores[pair] - mean;
        variance += delta * delta;
    }
    stddev = sqrtf(variance / (float)pair_count);
    if (stddev < 1e-4f) stddev = 1e-4f;
    for (size_t pair = 0; pair < pair_count; ++pair) {
        probabilities[pair] =
            (scores[pair] - mean) / stddev;
        if (probabilities[pair] > maximum)
            maximum = probabilities[pair];
    }
    for (size_t pair = 0; pair < pair_count; ++pair) {
        probabilities[pair] =
            expf(probabilities[pair] - maximum);
        total += probabilities[pair];
    }
    if (total < 1e-20f) total = 1e-20f;
    for (size_t pair = 0; pair < pair_count; ++pair) {
        float probability = probabilities[pair] / total;
        if (probability > 0.0f)
            entropy -= probability * logf(probability);
    }
    if (pair_count > 1)
        entropy /= logf((float)pair_count);
    else
        entropy = 0.0f;
    features[0] = top1;
    features[1] = top2;
    features[2] = top1 - top2;
    features[3] = mean;
    features[4] = stddev;
    features[5] = minimum;
    features[6] = entropy;
    features[7] = log1pf((float)pair_count);
    features[8] = max_entity;
    features[9] = max_predicate;
    for (int row = 0; row < model->null_hidden; ++row)
        hidden[row] = dense_gelu_row(
            model->null_hidden_weight +
                (size_t)row *
                    TYPED_QUERY_SET_FEATURES,
            features, TYPED_QUERY_SET_FEATURES,
            model->null_hidden_bias[row]);
    *null_score = model->null_output_bias;
    for (int row = 0; row < model->null_hidden; ++row)
        *null_score +=
            model->null_output_weight[row] * hidden[row];
    *activation_score = scores[best];
    *selected_index =
        *null_score > scores[best] ? SIZE_MAX : best;
    free(scores);
    free(hidden);
    free(probabilities);
    return isfinite(*activation_score) &&
           isfinite(*null_score) ? 0 : -1;
fail:
    free(scores);
    free(hidden);
    free(probabilities);
    return -1;
}
