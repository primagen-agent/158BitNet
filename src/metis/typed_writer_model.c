#include "typed_writer_model.h"

#include "sha256.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TYPED_WRITER_TENSOR_COUNT = 43
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
                  (0xEDB88320u &
                   (uint32_t)-(int32_t)(crc & 1u));
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
    size_t first, size_t second, size_t third,
    uint32_t *output) {
    size_t elements, bytes;
    if (checked_product(first, second, &elements) != 0 ||
        checked_product(elements, third, &elements) != 0 ||
        checked_product(elements, sizeof(float), &bytes) != 0 ||
        bytes == 0 || bytes > UINT32_MAX)
        return -1;
    *output = (uint32_t)bytes;
    return 0;
}

static float *read_tensor(
    FILE *file, uint32_t size,
    uint32_t expected, uint32_t crc) {
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

metis_typed_writer_model_t *metis_typed_writer_model_load(
    const char *path, const char *backbone_path,
    int expected_hidden, char *error, size_t error_size) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'W', 'R', 'I', 'T', 'E'};
    FILE *file = NULL;
    metis_typed_writer_model_t *model = NULL;
    unsigned char actual_magic[8];
    uint8_t actual_sha256[32];
    uint32_t version, hidden, rank, bands, layers;
    uint32_t max_span, tensor_count;
    uint32_t sizes[TYPED_WRITER_TENSOR_COUNT];
    uint32_t crcs[TYPED_WRITER_TENSOR_COUNT];
    uint32_t expected[TYPED_WRITER_TENSOR_COUNT];
    int tensor = 0;
#define FAIL(...) do { \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error != NULL && error_size > 0) error[0] = '\0';
    if (path == NULL || backbone_path == NULL ||
        expected_hidden <= 0)
        FAIL("invalid typed-writer arguments");
    file = fopen(path, "rb");
    if (file == NULL) FAIL("cannot open typed-writer model");
    if (read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || (version != 1 && version != 2) ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &rank) != 0 ||
        read_u32(file, &bands) != 0 ||
        read_u32(file, &layers) != 0 ||
        read_u32(file, &max_span) != 0 ||
        read_u32(file, &tensor_count) != 0)
        FAIL("bad typed-writer header");
    model = (metis_typed_writer_model_t *)calloc(
        1, sizeof *model);
    if (model == NULL) FAIL("out of memory");
    if (read_f32(file, &model->predicate_anchor_weight) != 0 ||
        read_f32(file, &model->value_anchor_weight) != 0)
        FAIL("truncated typed-writer header");
    if (hidden != (uint32_t)expected_hidden ||
        hidden > 16384u || rank < 1 || rank > 4096u ||
        bands < 2 || bands > 256u ||
        layers < bands || layers > 4096u ||
        max_span < 1 || max_span > 256u ||
        tensor_count != (version == 2 ? 43u : 39u) ||
        !isfinite(model->predicate_anchor_weight) ||
        !isfinite(model->value_anchor_weight))
        FAIL("typed-writer geometry mismatch");
    model->hidden_dim = (int)hidden;
    model->rank = (int)rank;
    model->band_count = (int)bands;
    model->layer_count = (int)layers;
    model->max_span = (int)max_span;
    model->band_start = (uint32_t *)malloc(
        (size_t)bands * sizeof(uint32_t));
    model->band_end = (uint32_t *)malloc(
        (size_t)bands * sizeof(uint32_t));
    if (model->band_start == NULL || model->band_end == NULL)
        FAIL("out of memory");
    for (uint32_t band = 0; band < bands; ++band) {
        if (read_u32(file, &model->band_start[band]) != 0 ||
            read_u32(file, &model->band_end[band]) != 0)
            FAIL("truncated typed-writer bands");
        if (model->band_start[band] > model->band_end[band] ||
            model->band_start[band] !=
                (band == 0 ? 0u :
                 model->band_end[band - 1] + 1u))
            FAIL("typed-writer bands are not contiguous");
    }
    if (model->band_end[bands - 1] + 1u != layers)
        FAIL("typed-writer bands do not cover all layers");
    if (read_exact(
            file, model->backbone_sha256,
            sizeof model->backbone_sha256) != 0)
        FAIL("truncated typed-writer header");
    for (uint32_t index = 0; index < tensor_count; ++index)
        if (read_u32(file, &sizes[index]) != 0 ||
            read_u32(file, &crcs[index]) != 0)
            FAIL("truncated typed-writer tensor table");
    if (bitnet_sha256_file(backbone_path, actual_sha256) != 0 ||
        memcmp(
            actual_sha256, model->backbone_sha256,
            sizeof actual_sha256) != 0)
        FAIL("typed-writer backbone SHA-256 mismatch");

    if (tensor_bytes(4, bands, 1, &expected[tensor++]) != 0)
        FAIL("typed-writer tensor geometry overflow");
    for (int field = 0; field < 4; ++field)
        if (tensor_bytes(
                rank, hidden, 1, &expected[tensor++]) != 0)
            FAIL("typed-writer tensor geometry overflow");
    if (tensor_bytes(2, bands, 1, &expected[tensor++]) != 0)
        FAIL("typed-writer tensor geometry overflow");
    for (int field = 0; field < 2; ++field)
        if (tensor_bytes(
                rank, hidden, 1, &expected[tensor++]) != 0)
            FAIL("typed-writer tensor geometry overflow");
    if (tensor_bytes(4, rank, 1, &expected[tensor++]) != 0 ||
        tensor_bytes(4, 2, rank, &expected[tensor++]) != 0 ||
        tensor_bytes(rank, rank * 4u, 1,
                     &expected[tensor++]) != 0 ||
        tensor_bytes(rank, 1, 1, &expected[tensor++]) != 0 ||
        tensor_bytes(2, rank, 1, &expected[tensor++]) != 0 ||
        tensor_bytes(2, 1, 1, &expected[tensor++]) != 0 ||
        tensor_bytes(2, rank, 1, &expected[tensor++]) != 0)
        FAIL("typed-writer tensor geometry overflow");
    for (int field = 0; field < 4; ++field) {
        if (tensor_bytes(
                rank, rank, 3, &expected[tensor++]) != 0 ||
            tensor_bytes(rank, 1, 1,
                         &expected[tensor++]) != 0 ||
            tensor_bytes(1, rank, 1,
                         &expected[tensor++]) != 0 ||
            tensor_bytes(1, 1, 1,
                         &expected[tensor++]) != 0 ||
            tensor_bytes(max_span, 1, 1,
                         &expected[tensor++]) != 0 ||
            tensor_bytes(1, 1, 1,
                         &expected[tensor++]) != 0)
            FAIL("typed-writer tensor geometry overflow");
    }
    if (version == 2 &&
        (tensor_bytes(rank, hidden * 2u, 1, &expected[tensor++]) ||
         tensor_bytes(rank, 1, 1, &expected[tensor++]) ||
         tensor_bytes(2, rank, 1, &expected[tensor++]) ||
         tensor_bytes(2, 1, 1, &expected[tensor++])))
        FAIL("context operation geometry overflow");
    if ((uint32_t)tensor != tensor_count)
        FAIL("typed-writer internal tensor mismatch");

    tensor = 0;
#define LOAD(member) do { \
    (member) = read_tensor( \
        file, sizes[tensor], expected[tensor], crcs[tensor]); \
    if ((member) == NULL) \
        FAIL("bad typed-writer tensor %d", tensor); \
    ++tensor; \
} while (0)
    LOAD(model->band_logits);
    for (int field = 0; field < 4; ++field)
        LOAD(model->projection[field]);
    LOAD(model->localizer_band_logits);
    for (int field = 0; field < 2; ++field)
        LOAD(model->localizer_projection[field]);
    LOAD(model->token_keys);
    LOAD(model->span_boundary_keys);
    LOAD(model->operation_hidden_weight);
    LOAD(model->operation_hidden_bias);
    LOAD(model->operation_output_weight);
    LOAD(model->operation_output_bias);
    LOAD(model->adapted_anchor_keys);
    for (int field = 0; field < 4; ++field) {
        float *scalar;
        LOAD(model->field[field].context_weight);
        LOAD(model->field[field].context_bias);
        LOAD(model->field[field].output_weight);
        LOAD(scalar);
        model->field[field].output_bias = scalar[0];
        free(scalar);
        LOAD(model->field[field].length_logits);
        LOAD(scalar);
        model->field[field].residual_scale = scalar[0];
        free(scalar);
        if (!isfinite(model->field[field].output_bias) ||
            !isfinite(model->field[field].residual_scale))
            FAIL("non-finite typed-writer scalar");
    }
    if (version == 2) {
        LOAD(model->context_operation_weight);
        LOAD(model->context_operation_bias);
        LOAD(model->context_operation_output_weight);
        LOAD(model->context_operation_output_bias);
    }
#undef LOAD
    if (fgetc(file) != EOF)
        FAIL("typed-writer model has trailing data");
    fclose(file);
    return model;
fail:
#undef FAIL
    if (file != NULL) fclose(file);
    metis_typed_writer_model_free(model);
    return NULL;
}

void metis_typed_writer_model_free(
    metis_typed_writer_model_t *model) {
    if (model == NULL) return;
    free(model->band_start);
    free(model->band_end);
    free(model->band_logits);
    for (int field = 0; field < 4; ++field)
        free(model->projection[field]);
    free(model->localizer_band_logits);
    for (int field = 0; field < 2; ++field)
        free(model->localizer_projection[field]);
    free(model->token_keys);
    free(model->span_boundary_keys);
    free(model->operation_hidden_weight);
    free(model->operation_hidden_bias);
    free(model->operation_output_weight);
    free(model->operation_output_bias);
    free(model->context_operation_weight);
    free(model->context_operation_bias);
    free(model->context_operation_output_weight);
    free(model->context_operation_output_bias);
    free(model->adapted_anchor_keys);
    for (int field = 0; field < 4; ++field) {
        free(model->field[field].context_weight);
        free(model->field[field].context_bias);
        free(model->field[field].output_weight);
        free(model->field[field].length_logits);
    }
    free(model);
}

static void normalize(float *values, int count) {
    float norm = 0.0f;
    for (int index = 0; index < count; ++index)
        norm += values[index] * values[index];
    norm = sqrtf(norm);
    if (norm < 1e-12f) norm = 1e-12f;
    for (int index = 0; index < count; ++index)
        values[index] /= norm;
}

static void softmax(
    const float *input, size_t count, float *output) {
    float maximum = -FLT_MAX;
    float total = 0.0f;
    for (size_t index = 0; index < count; ++index)
        if (input[index] > maximum) maximum = input[index];
    for (size_t index = 0; index < count; ++index) {
        output[index] = expf(input[index] - maximum);
        total += output[index];
    }
    if (total < 1e-20f) total = 1e-20f;
    for (size_t index = 0; index < count; ++index)
        output[index] /= total;
}

static int project_contextual(
    const metis_typed_writer_model_t *model,
    const float *hidden, size_t token_count,
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
        band_logits, (size_t)model->band_count, band_weight);
    for (size_t token = 0; token < token_count; ++token) {
        float *row = output + token * (size_t)model->rank;
        for (int column = 0;
             column < model->hidden_dim; ++column) {
            mixed[column] = 0.0f;
            for (int band = 0; band < model->band_count; ++band)
                mixed[column] += band_weight[band] *
                    hidden[
                        (token * (size_t)model->band_count +
                         (size_t)band) *
                            (size_t)model->hidden_dim +
                        (size_t)column];
        }
        for (int row_index = 0;
             row_index < model->rank; ++row_index) {
            float value = 0.0f;
            const float *weight = projection +
                (size_t)row_index * (size_t)model->hidden_dim;
            for (int column = 0;
                 column < model->hidden_dim; ++column)
                value += weight[column] * mixed[column];
            row[row_index] = value;
        }
        normalize(row, model->rank);
    }
    free(band_weight);
    free(mixed);
    return 0;
}

static void project_identity(
    const metis_typed_writer_model_t *model,
    const float *identity, size_t token_count,
    const float *projection, float *output) {
    for (size_t token = 0; token < token_count; ++token) {
        const float *input = identity +
            token * (size_t)model->hidden_dim;
        float *row = output + token * (size_t)model->rank;
        for (int row_index = 0;
             row_index < model->rank; ++row_index) {
            const float *weight = projection +
                (size_t)row_index * (size_t)model->hidden_dim;
            float value = 0.0f;
            for (int column = 0;
                 column < model->hidden_dim; ++column)
                value += weight[column] * input[column];
            row[row_index] = value;
        }
        normalize(row, model->rank);
    }
}

static float dot(
    const float *left, const float *right, int count) {
    float value = 0.0f;
    for (int index = 0; index < count; ++index)
        value += left[index] * right[index];
    return value;
}

static float gelu(float value) {
    return 0.5f * value *
        (1.0f + erff(value * 0.7071067811865475f));
}

int metis_typed_writer_predict(
    const metis_typed_writer_model_t *model,
    const float *hidden, const float *identity,
    size_t token_count, metis_typed_writer_result_t *output) {
    size_t token_values, all_token_values;
    float *score_tokens = NULL;
    float *address_tokens = NULL;
    float *attention_scores = NULL;
    float *attention = NULL;
    float *states = NULL;
    float *operation_hidden = NULL;
    float *convolution = NULL;
    float *emissions = NULL;
    int status = -1;
    if (model == NULL || hidden == NULL || identity == NULL ||
        token_count == 0 || output == NULL)
        return -1;
    if (checked_product(
            token_count, (size_t)model->rank,
            &token_values) != 0 ||
        checked_product(
            token_values, METIS_TYPED_WRITER_FIELD_COUNT,
            &all_token_values) != 0 ||
        all_token_values > SIZE_MAX / sizeof(float))
        return -1;
    score_tokens = (float *)malloc(
        all_token_values * sizeof(float));
    address_tokens = (float *)malloc(
        token_values * sizeof(float));
    attention_scores = (float *)malloc(
        token_count * sizeof(float));
    attention = (float *)malloc(
        token_count * sizeof(float));
    states = (float *)malloc(
        METIS_TYPED_WRITER_FIELD_COUNT *
        (size_t)model->rank * sizeof(float));
    operation_hidden = (float *)malloc(
        (size_t)model->rank * sizeof(float));
    convolution = (float *)malloc(
        (size_t)model->rank * sizeof(float));
    emissions = (float *)malloc(
        token_count * sizeof(float));
    if (score_tokens == NULL || address_tokens == NULL ||
        attention_scores == NULL || attention == NULL ||
        states == NULL || operation_hidden == NULL ||
        convolution == NULL || emissions == NULL)
        goto cleanup;
    memset(output, 0, sizeof *output);
    for (int field = 0;
         field < METIS_TYPED_WRITER_FIELD_COUNT; ++field) {
        float *score = score_tokens +
            (size_t)field * token_values;
        float *state = states +
            (size_t)field * (size_t)model->rank;
        const float *original_key = model->token_keys +
            (size_t)field * (size_t)model->rank;
        const float *anchor_key = original_key;
        if (field < 2) {
            if (project_contextual(
                    model, hidden, token_count,
                    model->localizer_band_logits +
                        (size_t)field *
                        (size_t)model->band_count,
                    model->localizer_projection[field],
                    score) != 0)
                goto cleanup;
            project_identity(
                model, identity, token_count,
                model->projection[field], address_tokens);
        } else {
            if (project_contextual(
                    model, hidden, token_count,
                    model->band_logits +
                        (size_t)field *
                        (size_t)model->band_count,
                    model->projection[field], score) != 0)
                goto cleanup;
            memcpy(
                address_tokens, score,
                token_values * sizeof(float));
        }
        for (size_t token = 0; token < token_count; ++token)
            attention_scores[token] = dot(
                score + token * (size_t)model->rank,
                original_key, model->rank);
        softmax(attention_scores, token_count, attention);
        memset(
            state, 0, (size_t)model->rank * sizeof(float));
        for (size_t token = 0; token < token_count; ++token)
            for (int rank = 0; rank < model->rank; ++rank)
                state[rank] += attention[token] *
                    address_tokens[
                        token * (size_t)model->rank +
                        (size_t)rank];
        normalize(state, model->rank);
        if (field == METIS_TYPED_WRITER_PREDICATE)
            anchor_key = model->adapted_anchor_keys;
        else if (field == METIS_TYPED_WRITER_VALUE)
            anchor_key = model->adapted_anchor_keys +
                (size_t)model->rank;
        output->anchor[field] = 0;
        {
            float best = -FLT_MAX;
            for (size_t token = 0; token < token_count; ++token) {
                float value = dot(
                    score + token * (size_t)model->rank,
                    anchor_key, model->rank);
                if (value > best) {
                    best = value;
                    output->anchor[field] = token;
                }
            }
        }
    }
    for (int row = 0; row < model->rank; ++row) {
        float value = model->operation_hidden_bias[row];
        const float *weight =
            model->operation_hidden_weight +
            (size_t)row *
            (size_t)model->rank *
            METIS_TYPED_WRITER_FIELD_COUNT;
        for (int column = 0;
             column < model->rank *
                      METIS_TYPED_WRITER_FIELD_COUNT;
             ++column)
            value += weight[column] * states[column];
        operation_hidden[row] = gelu(value);
    }
    for (int operation = 0; operation < 2; ++operation) {
        output->operation_logits[operation] =
            model->operation_output_bias[operation] +
            dot(
                model->operation_output_weight +
                    (size_t)operation * (size_t)model->rank,
                operation_hidden, model->rank);
    }
    if (model->context_operation_weight != NULL) {
        int width = model->hidden_dim;
        float *features = (float *)calloc((size_t)width * 2u, sizeof(float));
        size_t first = token_count > 1 ? 1u : 0u;
        if (features == NULL) goto cleanup;
        for (size_t token = first; token < token_count; ++token)
            for (int band = 0; band < model->band_count; ++band)
                for (int column = 0; column < width; ++column) {
                    float value = hidden[(token * (size_t)model->band_count + (size_t)band) * (size_t)width + (size_t)column]
                                  / (float)model->band_count;
                    features[column] += value / (float)(token_count - first);
                    if (token == token_count - 1u) features[width + column] += value;
                }
        normalize(features, width);
        normalize(features + width, width);
        for (int row = 0; row < model->rank; ++row)
            operation_hidden[row] = gelu(model->context_operation_bias[row] + dot(
                model->context_operation_weight + (size_t)row * (size_t)width * 2u,
                features, width * 2));
        for (int operation = 0; operation < 2; ++operation)
            output->operation_logits[operation] = model->context_operation_output_bias[operation] + dot(
                model->context_operation_output_weight + (size_t)operation * (size_t)model->rank,
                operation_hidden, model->rank);
        free(features);
    }
    output->operation =
        output->operation_logits[1] >
        output->operation_logits[0] ? 1 : 0;

    for (int field = 0;
         field < METIS_TYPED_WRITER_FIELD_COUNT; ++field) {
        const metis_typed_writer_field_t *head =
            &model->field[field];
        const float *tokens = score_tokens +
            (size_t)field * token_values;
        const float *start_key = model->span_boundary_keys +
            ((size_t)field * 2u) * (size_t)model->rank;
        const float *end_key = start_key + model->rank;
        float scale = expf(head->residual_scale);
        float best_score = -FLT_MAX;
        float anchor_weight = 0.0f;
        if (scale > 20.0f) scale = 20.0f;
        if (field == METIS_TYPED_WRITER_PREDICATE)
            anchor_weight = model->predicate_anchor_weight;
        else if (field == METIS_TYPED_WRITER_VALUE)
            anchor_weight = model->value_anchor_weight;
        for (size_t token = 0; token < token_count; ++token) {
            for (int row = 0; row < model->rank; ++row) {
                float value = head->context_bias[row];
                const float *weight = head->context_weight +
                    (size_t)row *
                    (size_t)model->rank * 3u;
                for (int column = 0;
                     column < model->rank; ++column) {
                    if (token > 0)
                        value += weight[
                            (size_t)column * 3u] *
                            tokens[
                                (token - 1u) *
                                    (size_t)model->rank +
                                (size_t)column];
                    value += weight[
                        (size_t)column * 3u + 1u] *
                        tokens[
                            token * (size_t)model->rank +
                            (size_t)column];
                    if (token + 1u < token_count)
                        value += weight[
                            (size_t)column * 3u + 2u] *
                            tokens[
                                (token + 1u) *
                                    (size_t)model->rank +
                                (size_t)column];
                }
                convolution[row] = gelu(value);
            }
            emissions[token] = head->output_bias +
                dot(
                    head->output_weight,
                    convolution, model->rank);
        }
        for (size_t start = 0; start < token_count; ++start) {
            float emission_sum = 0.0f;
            float start_score = dot(
                tokens + start * (size_t)model->rank,
                start_key, model->rank);
            size_t maximum = token_count - start;
            if (maximum > (size_t)model->max_span)
                maximum = (size_t)model->max_span;
            for (size_t length = 1; length <= maximum; ++length) {
                size_t end = start + length - 1u;
                float candidate;
                emission_sum += emissions[end];
                candidate = start_score +
                    dot(
                        tokens + end * (size_t)model->rank,
                        end_key, model->rank) +
                    scale * emission_sum +
                    head->length_logits[length - 1u];
                if (anchor_weight != 0.0f &&
                    output->anchor[field] >= start &&
                    output->anchor[field] <= end)
                    candidate += anchor_weight;
                if (candidate > best_score) {
                    best_score = candidate;
                    output->span_start[field] = start;
                    output->span_end[field] = end;
                }
            }
        }
    }
    status = 0;
cleanup:
    free(score_tokens);
    free(address_tokens);
    free(attention_scores);
    free(attention);
    free(states);
    free(operation_hidden);
    free(convolution);
    free(emissions);
    return status;
}
