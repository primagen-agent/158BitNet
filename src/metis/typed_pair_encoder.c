#include "typed_pair_encoder.h"

#include "sha256.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PAIR_ENCODER_TENSOR_COUNT = 25,
    FIELD_COUNT = 2
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

metis_typed_pair_encoder_t *metis_typed_pair_encoder_load(
    const char *path, const char *backbone_path,
    int expected_hidden, char *error, size_t error_size) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'P', 'A', 'I', 'R', '1'};
    FILE *file = NULL;
    metis_typed_pair_encoder_t *model = NULL;
    unsigned char actual_magic[8];
    uint8_t actual_sha256[32];
    uint32_t version;
    uint32_t hidden;
    uint32_t rank;
    uint32_t bands;
    uint32_t layers;
    uint32_t pair_width;
    uint32_t head_width;
    uint32_t tensor_count;
    uint32_t sizes[PAIR_ENCODER_TENSOR_COUNT];
    uint32_t crcs[PAIR_ENCODER_TENSOR_COUNT];
    uint32_t expected[PAIR_ENCODER_TENSOR_COUNT];
#define FAIL(...) do { \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error != NULL && error_size > 0) error[0] = '\0';
    if (path == NULL || backbone_path == NULL || expected_hidden <= 0)
        FAIL("invalid typed-pair encoder arguments");
    file = fopen(path, "rb");
    if (file == NULL) FAIL("cannot open typed-pair encoder");
    if (read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &rank) != 0 ||
        read_u32(file, &bands) != 0 ||
        read_u32(file, &layers) != 0 ||
        read_u32(file, &pair_width) != 0 ||
        read_u32(file, &head_width) != 0 ||
        read_u32(file, &tensor_count) != 0)
        FAIL("bad typed-pair encoder header");
    if (hidden != (uint32_t)expected_hidden || hidden > 16384u ||
        rank < 1 || rank > 4096u ||
        bands < 2 || bands > 256u ||
        layers < bands || layers > 4096u ||
        pair_width != rank * 2u + 4u ||
        head_width != pair_width + 1u ||
        tensor_count != PAIR_ENCODER_TENSOR_COUNT)
        FAIL("typed-pair encoder geometry mismatch");
    model = (metis_typed_pair_encoder_t *)calloc(
        1, sizeof *model);
    if (model == NULL) FAIL("out of memory");
    model->hidden_dim = (int)hidden;
    model->rank = (int)rank;
    model->band_count = (int)bands;
    model->layer_count = (int)layers;
    model->pair_width = (int)pair_width;
    model->head_width = (int)head_width;
    model->band_start = (uint32_t *)malloc(
        (size_t)bands * sizeof(uint32_t));
    model->band_end = (uint32_t *)malloc(
        (size_t)bands * sizeof(uint32_t));
    if (model->band_start == NULL || model->band_end == NULL)
        FAIL("out of memory");
    for (uint32_t band = 0; band < bands; ++band) {
        if (read_u32(file, &model->band_start[band]) != 0 ||
            read_u32(file, &model->band_end[band]) != 0)
            FAIL("truncated typed-pair bands");
        if (model->band_start[band] > model->band_end[band] ||
            model->band_start[band] !=
                (band == 0 ? 0u :
                 model->band_end[band - 1] + 1u))
            FAIL("typed-pair bands are not contiguous");
    }
    if (model->band_end[bands - 1] + 1u != layers)
        FAIL("typed-pair bands do not cover all layers");
    if (read_exact(file, model->backbone_sha256,
                   sizeof model->backbone_sha256) != 0)
        FAIL("truncated typed-pair encoder header");
    for (uint32_t index = 0; index < tensor_count; ++index)
        if (read_u32(file, &sizes[index]) != 0 ||
            read_u32(file, &crcs[index]) != 0)
            FAIL("truncated typed-pair tensor table");
    if (bitnet_sha256_file(backbone_path, actual_sha256) != 0 ||
        memcmp(actual_sha256, model->backbone_sha256,
               sizeof actual_sha256) != 0)
        FAIL("typed-pair backbone SHA-256 mismatch");
    if (tensor_bytes(2, bands, &expected[0]) != 0 ||
        tensor_bytes(rank, hidden, &expected[1]) != 0 ||
        tensor_bytes(rank, hidden, &expected[2]) != 0 ||
        tensor_bytes(rank, hidden, &expected[3]) != 0 ||
        tensor_bytes(rank, hidden, &expected[4]) != 0 ||
        tensor_bytes(2, bands, &expected[5]) != 0 ||
        tensor_bytes(rank, hidden, &expected[6]) != 0 ||
        tensor_bytes(rank, hidden, &expected[7]) != 0 ||
        tensor_bytes(2, rank, &expected[8]) != 0)
        FAIL("typed-pair tensor geometry overflow");
    for (int field = 0; field < FIELD_COUNT; ++field) {
        const int fusion = 9 + field * 4;
        const int head = 17 + field * 4;
        if (tensor_bytes(rank, pair_width * 2u,
                         &expected[fusion]) != 0 ||
            tensor_bytes(rank, 1,
                         &expected[fusion + 1]) != 0 ||
            tensor_bytes(1, rank,
                         &expected[fusion + 2]) != 0 ||
            tensor_bytes(1, 1,
                         &expected[fusion + 3]) != 0 ||
            tensor_bytes(rank * 2u, head_width,
                         &expected[head]) != 0 ||
            tensor_bytes(rank * 2u, 1,
                         &expected[head + 1]) != 0 ||
            tensor_bytes(1, rank * 2u,
                         &expected[head + 2]) != 0 ||
            tensor_bytes(1, 1,
                         &expected[head + 3]) != 0)
            FAIL("typed-pair tensor geometry overflow");
    }
#define LOAD(index, field) do { \
    model->field = read_tensor( \
        file, sizes[index], expected[index], crcs[index]); \
    if (model->field == NULL) \
        FAIL("bad typed-pair tensor %d", index); \
} while (0)
    LOAD(0, band_logits);
    LOAD(1, projection[0]);
    LOAD(2, projection[1]);
    LOAD(3, identity_projection[0]);
    LOAD(4, identity_projection[1]);
    LOAD(5, localizer_band_logits);
    LOAD(6, localizer_projection[0]);
    LOAD(7, localizer_projection[1]);
    LOAD(8, token_keys);
    for (int field = 0; field < FIELD_COUNT; ++field) {
        const int fusion = 9 + field * 4;
        LOAD(fusion, fusion_hidden_weight[field]);
        LOAD(fusion + 1, fusion_hidden_bias[field]);
        LOAD(fusion + 2, fusion_output_weight[field]);
        {
            float *scalar = read_tensor(
                file, sizes[fusion + 3],
                expected[fusion + 3],
                crcs[fusion + 3]);
            if (scalar == NULL)
                FAIL("bad typed-pair tensor %d",
                     fusion + 3);
            model->fusion_output_bias[field] = scalar[0];
            free(scalar);
        }
    }
    for (int field = 0; field < FIELD_COUNT; ++field) {
        const int head = 17 + field * 4;
        LOAD(head, head_hidden_weight[field]);
        LOAD(head + 1, head_hidden_bias[field]);
        LOAD(head + 2, head_output_weight[field]);
        {
            float *scalar = read_tensor(
                file, sizes[head + 3],
                expected[head + 3],
                crcs[head + 3]);
            if (scalar == NULL)
                FAIL("bad typed-pair tensor %d",
                     head + 3);
            model->head_output_bias[field] = scalar[0];
            free(scalar);
        }
    }
#undef LOAD
    if (fgetc(file) != EOF)
        FAIL("typed-pair encoder has trailing data");
    fclose(file);
    return model;
fail:
    if (file != NULL) fclose(file);
    metis_typed_pair_encoder_free(model);
    return NULL;
#undef FAIL
}

void metis_typed_pair_encoder_free(
    metis_typed_pair_encoder_t *model) {
    if (model == NULL) return;
    free(model->band_start);
    free(model->band_end);
    free(model->band_logits);
    free(model->localizer_band_logits);
    free(model->token_keys);
    for (int field = 0; field < FIELD_COUNT; ++field) {
        free(model->projection[field]);
        free(model->identity_projection[field]);
        free(model->localizer_projection[field]);
        free(model->fusion_hidden_weight[field]);
        free(model->fusion_hidden_bias[field]);
        free(model->fusion_output_weight[field]);
        free(model->head_hidden_weight[field]);
        free(model->head_hidden_bias[field]);
        free(model->head_output_weight[field]);
    }
    free(model);
}

static float gelu(float value) {
    return 0.5f * value *
           (1.0f + erff(value * 0.7071067811865475f));
}

static void softmax(
    const float *input, size_t count, float *output) {
    float maximum = input[0];
    float total = 0.0f;
    for (size_t index = 1; index < count; ++index)
        if (input[index] > maximum) maximum = input[index];
    for (size_t index = 0; index < count; ++index) {
        output[index] = expf(input[index] - maximum);
        total += output[index];
    }
    if (total < 1e-20f) total = 1e-20f;
    for (size_t index = 0; index < count; ++index)
        output[index] /= total;
}

static void normalize(float *values, int count) {
    float sum = 0.0f;
    for (int index = 0; index < count; ++index)
        sum += values[index] * values[index];
    const float scale = 1.0f / sqrtf(fmaxf(sum, 1e-24f));
    for (int index = 0; index < count; ++index)
        values[index] *= scale;
}

static int project_contextual(
    const metis_typed_pair_encoder_t *model,
    int field, const float *hidden, size_t token_count,
    const float *band_logits, const float *projection,
    float *output) {
    float *band_weight = (float *)malloc(
        (size_t)model->band_count * sizeof(float));
    float *mixed = (float *)malloc(
        (size_t)model->hidden_dim * sizeof(float));
    if (band_weight == NULL || mixed == NULL) {
        free(band_weight);
        free(mixed);
        return -1;
    }
    softmax(
        band_logits + (size_t)field * (size_t)model->band_count,
        (size_t)model->band_count, band_weight);
    for (size_t token = 0; token < token_count; ++token) {
        float *row = output + token * (size_t)model->rank;
        for (int column = 0;
             column < model->hidden_dim; ++column) {
            mixed[column] = 0.0f;
            for (int band = 0;
                 band < model->band_count; ++band)
                mixed[column] += band_weight[band] *
                    hidden[
                        (token * (size_t)model->band_count +
                         (size_t)band) *
                            (size_t)model->hidden_dim +
                        (size_t)column];
        }
        for (int rank = 0; rank < model->rank; ++rank) {
            float value = 0.0f;
            const float *weight = projection +
                (size_t)rank * (size_t)model->hidden_dim;
            for (int column = 0;
                 column < model->hidden_dim; ++column)
                value += weight[column] * mixed[column];
            row[rank] = value;
        }
        normalize(row, model->rank);
    }
    free(band_weight);
    free(mixed);
    return 0;
}

static void project_identity(
    const metis_typed_pair_encoder_t *model,
    const float *identity, size_t token_count,
    const float *projection, float *output) {
    for (size_t token = 0; token < token_count; ++token) {
        const float *input = identity +
            token * (size_t)model->hidden_dim;
        float *row = output + token * (size_t)model->rank;
        for (int rank = 0; rank < model->rank; ++rank) {
            const float *weight = projection +
                (size_t)rank * (size_t)model->hidden_dim;
            float value = 0.0f;
            for (int column = 0;
                 column < model->hidden_dim; ++column)
                value += weight[column] * input[column];
            row[rank] = value;
        }
        normalize(row, model->rank);
    }
}

static int localizer_attention(
    const metis_typed_pair_encoder_t *model,
    int field, const float *hidden, size_t token_count,
    float *attention) {
    size_t token_values;
    float *tokens;
    float *scores;
    if (checked_product(
            token_count, (size_t)model->rank,
            &token_values) != 0 ||
        token_values > SIZE_MAX / sizeof(float) ||
        token_count > SIZE_MAX / sizeof(float))
        return -1;
    tokens = (float *)malloc(token_values * sizeof(float));
    scores = (float *)malloc(token_count * sizeof(float));
    if (tokens == NULL || scores == NULL) {
        free(tokens);
        free(scores);
        return -1;
    }
    if (project_contextual(
            model, field, hidden, token_count,
            model->localizer_band_logits,
            model->localizer_projection[field],
            tokens) != 0) {
        free(tokens);
        free(scores);
        return -1;
    }
    const float *key = model->token_keys +
        (size_t)field * (size_t)model->rank;
    for (size_t token = 0; token < token_count; ++token) {
        scores[token] = 0.0f;
        for (int rank = 0; rank < model->rank; ++rank)
            scores[token] +=
                tokens[token * (size_t)model->rank +
                       (size_t)rank] * key[rank];
    }
    softmax(scores, token_count, attention);
    free(tokens);
    free(scores);
    return 0;
}

static float dot(
    const float *left, const float *right, int count) {
    float value = 0.0f;
    for (int index = 0; index < count; ++index)
        value += left[index] * right[index];
    return value;
}

static int path_features(
    const metis_typed_pair_encoder_t *model,
    const float *left, size_t left_count,
    const float *right, size_t right_count,
    const float *left_attention,
    const float *right_attention,
    float *output) {
    float *left_state = NULL;
    float *right_state = NULL;
    float *left_best = NULL;
    float *right_best = NULL;
    float left_mean = 0.0f;
    float right_mean = 0.0f;
    float left_max = -FLT_MAX;
    float right_max = -FLT_MAX;
    if (left_count > SIZE_MAX / sizeof(float) ||
        right_count > SIZE_MAX / sizeof(float))
        return -1;
    left_state = (float *)calloc(
        (size_t)model->rank, sizeof(float));
    right_state = (float *)calloc(
        (size_t)model->rank, sizeof(float));
    left_best = (float *)malloc(left_count * sizeof(float));
    right_best = (float *)malloc(right_count * sizeof(float));
    if (left_state == NULL || right_state == NULL ||
        left_best == NULL || right_best == NULL)
        goto fail;
    for (int rank = 0; rank < model->rank; ++rank) {
        for (size_t token = 0; token < left_count; ++token)
            left_state[rank] += left_attention[token] *
                left[token * (size_t)model->rank + (size_t)rank];
        for (size_t token = 0; token < right_count; ++token)
            right_state[rank] += right_attention[token] *
                right[token * (size_t)model->rank + (size_t)rank];
    }
    normalize(left_state, model->rank);
    normalize(right_state, model->rank);
    for (int rank = 0; rank < model->rank; ++rank) {
        output[rank] = left_state[rank] * right_state[rank];
        output[model->rank + rank] =
            fabsf(left_state[rank] - right_state[rank]);
    }
    for (size_t token = 0; token < left_count; ++token)
        left_best[token] = -FLT_MAX;
    for (size_t token = 0; token < right_count; ++token)
        right_best[token] = -FLT_MAX;
    for (size_t left_token = 0;
         left_token < left_count; ++left_token)
        for (size_t right_token = 0;
             right_token < right_count; ++right_token) {
            const float similarity = dot(
                left + left_token * (size_t)model->rank,
                right + right_token * (size_t)model->rank,
                model->rank);
            if (similarity > left_best[left_token])
                left_best[left_token] = similarity;
            if (similarity > right_best[right_token])
                right_best[right_token] = similarity;
        }
    for (size_t token = 0; token < left_count; ++token) {
        left_mean += left_best[token];
        if (left_best[token] > left_max)
            left_max = left_best[token];
    }
    for (size_t token = 0; token < right_count; ++token) {
        right_mean += right_best[token];
        if (right_best[token] > right_max)
            right_max = right_best[token];
    }
    output[model->rank * 2] =
        left_mean / (float)left_count;
    output[model->rank * 2 + 1] =
        right_mean / (float)right_count;
    output[model->rank * 2 + 2] = left_max;
    output[model->rank * 2 + 3] = right_max;
    free(left_state);
    free(right_state);
    free(left_best);
    free(right_best);
    return 0;
fail:
    free(left_state);
    free(right_state);
    free(left_best);
    free(right_best);
    return -1;
}

static float dense_gelu(
    const float *weight, const float *input,
    int input_count, float bias) {
    float value = bias;
    for (int index = 0; index < input_count; ++index)
        value += weight[index] * input[index];
    return gelu(value);
}

static int field_score(
    const metis_typed_pair_encoder_t *model,
    int field,
    const float *left_hidden, size_t left_count,
    const float *left_identity,
    const float *right_hidden, size_t right_count,
    const float *right_identity,
    float *feature, float *logit) {
    size_t left_values;
    size_t right_values;
    float *left_context = NULL;
    float *right_context = NULL;
    float *left_token_identity = NULL;
    float *right_token_identity = NULL;
    float *left_attention = NULL;
    float *right_attention = NULL;
    float *contextual = NULL;
    float *identity = NULL;
    float *gate_hidden = NULL;
    float *head_hidden = NULL;
    float gate_input;
    if (checked_product(
            left_count, (size_t)model->rank,
            &left_values) != 0 ||
        checked_product(
            right_count, (size_t)model->rank,
            &right_values) != 0 ||
        left_values > SIZE_MAX / sizeof(float) ||
        right_values > SIZE_MAX / sizeof(float) ||
        left_count > SIZE_MAX / sizeof(float) ||
        right_count > SIZE_MAX / sizeof(float))
        return -1;
    left_context = (float *)malloc(left_values * sizeof(float));
    right_context = (float *)malloc(right_values * sizeof(float));
    left_token_identity =
        (float *)malloc(left_values * sizeof(float));
    right_token_identity =
        (float *)malloc(right_values * sizeof(float));
    left_attention = (float *)malloc(left_count * sizeof(float));
    right_attention = (float *)malloc(right_count * sizeof(float));
    contextual = (float *)malloc(
        (size_t)model->pair_width * sizeof(float));
    identity = (float *)malloc(
        (size_t)model->pair_width * sizeof(float));
    gate_hidden = (float *)malloc(
        (size_t)model->rank * sizeof(float));
    head_hidden = (float *)malloc(
        (size_t)model->rank * 2u * sizeof(float));
    if (left_context == NULL || right_context == NULL ||
        left_token_identity == NULL || right_token_identity == NULL ||
        left_attention == NULL || right_attention == NULL ||
        contextual == NULL || identity == NULL ||
        gate_hidden == NULL || head_hidden == NULL)
        goto fail;
    if (project_contextual(
            model, field, left_hidden, left_count,
            model->band_logits, model->projection[field],
            left_context) != 0 ||
        project_contextual(
            model, field, right_hidden, right_count,
            model->band_logits, model->projection[field],
            right_context) != 0)
        goto fail;
    project_identity(
        model, left_identity, left_count,
        model->identity_projection[field],
        left_token_identity);
    project_identity(
        model, right_identity, right_count,
        model->identity_projection[field],
        right_token_identity);
    if (localizer_attention(
            model, field, left_hidden,
            left_count, left_attention) != 0 ||
        localizer_attention(
            model, field, right_hidden,
            right_count, right_attention) != 0 ||
        path_features(
            model, left_context, left_count,
            right_context, right_count,
            left_attention, right_attention,
            contextual) != 0 ||
        path_features(
            model, left_token_identity, left_count,
            right_token_identity, right_count,
            left_attention, right_attention,
            identity) != 0)
        goto fail;
    for (int row = 0; row < model->rank; ++row) {
        const float *weight =
            model->fusion_hidden_weight[field] +
            (size_t)row * (size_t)model->pair_width * 2u;
        float value = model->fusion_hidden_bias[field][row];
        for (int column = 0;
             column < model->pair_width; ++column) {
            value += weight[column] * contextual[column];
            value += weight[model->pair_width + column] *
                     identity[column];
        }
        gate_hidden[row] = gelu(value);
    }
    gate_input = model->fusion_output_bias[field];
    for (int row = 0; row < model->rank; ++row)
        gate_input +=
            model->fusion_output_weight[field][row] *
            gate_hidden[row];
    const float gate = 1.0f / (1.0f + expf(-gate_input));
    for (int column = 0;
         column < model->pair_width; ++column)
        feature[column] =
            gate * contextual[column] +
            (1.0f - gate) * identity[column];
    feature[model->pair_width] = gate;
    for (int row = 0; row < model->rank * 2; ++row)
        head_hidden[row] = dense_gelu(
            model->head_hidden_weight[field] +
                (size_t)row * (size_t)model->head_width,
            feature, model->head_width,
            model->head_hidden_bias[field][row]);
    *logit = model->head_output_bias[field];
    for (int row = 0; row < model->rank * 2; ++row)
        *logit +=
            model->head_output_weight[field][row] *
            head_hidden[row];
    free(left_context);
    free(right_context);
    free(left_token_identity);
    free(right_token_identity);
    free(left_attention);
    free(right_attention);
    free(contextual);
    free(identity);
    free(gate_hidden);
    free(head_hidden);
    return isfinite(*logit) ? 0 : -1;
fail:
    free(left_context);
    free(right_context);
    free(left_token_identity);
    free(right_token_identity);
    free(left_attention);
    free(right_attention);
    free(contextual);
    free(identity);
    free(gate_hidden);
    free(head_hidden);
    return -1;
}

int metis_typed_pair_encoder_score(
    const metis_typed_pair_encoder_t *model,
    const float *left_hidden, size_t left_count,
    const float *left_identity,
    const float *right_hidden, size_t right_count,
    const float *right_identity,
    float *entity_feature,
    float *predicate_feature,
    float *entity_logit,
    float *predicate_logit) {
    if (model == NULL || left_hidden == NULL ||
        left_identity == NULL || right_hidden == NULL ||
        right_identity == NULL || left_count == 0 ||
        right_count == 0 || entity_feature == NULL ||
        predicate_feature == NULL || entity_logit == NULL ||
        predicate_logit == NULL)
        return -1;
    if (field_score(
            model, 0,
            left_hidden, left_count, left_identity,
            right_hidden, right_count, right_identity,
            entity_feature, entity_logit) != 0 ||
        field_score(
            model, 1,
            left_hidden, left_count, left_identity,
            right_hidden, right_count, right_identity,
            predicate_feature, predicate_logit) != 0)
        return -1;
    return 0;
}
