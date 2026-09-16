#include "bitnet.h"
#include "metis/typed_pair_encoder.h"
#include "sha256.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
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

static int read_values(
    FILE *file, size_t count, size_t size, void **output) {
    if (count == 0 || size == 0 || count > SIZE_MAX / size)
        return -1;
    *output = malloc(count * size);
    if (*output == NULL ||
        read_exact(file, *output, count * size) != 0) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

static int encode_tokens(
    bitnet_model_t *backbone,
    const int *band_start, const int *band_end,
    int bands, const int *tokens, int token_count,
    float **hidden_output, float **identity_output) {
    bitnet_context_t *context = NULL;
    const float *hidden;
    const int hidden_dim = bitnet_embedding_length(backbone);
    size_t hidden_count;
    size_t identity_count;
    float *hidden_copy = NULL;
    float *identity = NULL;
    if (hidden_output != NULL) *hidden_output = NULL;
    if (identity_output != NULL) *identity_output = NULL;
    if (backbone == NULL || band_start == NULL ||
        band_end == NULL || bands < 1 || tokens == NULL ||
        token_count < 1 || hidden_dim < 1 ||
        hidden_output == NULL || identity_output == NULL ||
        (size_t)token_count > SIZE_MAX / (size_t)bands ||
        (size_t)token_count * (size_t)bands >
            SIZE_MAX / (size_t)hidden_dim)
        return -1;
    hidden_count = (size_t)token_count *
                   (size_t)bands * (size_t)hidden_dim;
    identity_count =
        (size_t)token_count * (size_t)hidden_dim;
    context = bitnet_create_context(backbone, token_count);
    hidden_copy = (float *)malloc(
        hidden_count * sizeof(float));
    identity = (float *)malloc(
        identity_count * sizeof(float));
    if (context == NULL || hidden_copy == NULL ||
        identity == NULL ||
        bitnet_context_configure_layer_bands(
            context, band_start, band_end, bands) != 0 ||
        bitnet_eval_hidden(context, tokens, token_count) != 0)
        goto fail;
    hidden = bitnet_get_last_eval_layer_bands(context);
    if (hidden == NULL ||
        bitnet_last_eval_layer_band_token_count(context)
            != token_count ||
        bitnet_layer_band_count(context) != bands ||
        bitnet_token_embedding_lookup(
            backbone, tokens, token_count,
            identity, identity_count) != 0)
        goto fail;
    memcpy(hidden_copy, hidden,
           hidden_count * sizeof(float));
    bitnet_free_context(context);
    *hidden_output = hidden_copy;
    *identity_output = identity;
    return 0;
fail:
    bitnet_free_context(context);
    free(hidden_copy);
    free(identity);
    return -1;
}

static void error_metrics(
    const float *actual, const float *expected,
    size_t count, float *maximum, double *rmse) {
    double squared = 0.0;
    *maximum = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        const float delta = actual[index] - expected[index];
        const float absolute = fabsf(delta);
        if (absolute > *maximum) *maximum = absolute;
        squared += (double)delta * (double)delta;
    }
    *rmse = sqrt(squared / (double)count);
}

static void cosine_metrics(
    const float *actual, const float *expected,
    size_t row_count, size_t row_width,
    double *mean, float *minimum) {
    double total = 0.0;
    *minimum = FLT_MAX;
    for (size_t row = 0; row < row_count; ++row) {
        double dot = 0.0;
        double actual_norm = 0.0;
        double expected_norm = 0.0;
        const float *left = actual + row * row_width;
        const float *right = expected + row * row_width;
        for (size_t column = 0; column < row_width; ++column) {
            dot += (double)left[column] * (double)right[column];
            actual_norm += (double)left[column] * (double)left[column];
            expected_norm +=
                (double)right[column] * (double)right[column];
        }
        const float cosine = (float)(
            dot / sqrt(fmax(actual_norm * expected_norm, 1e-30)));
        total += cosine;
        if (cosine < *minimum) *minimum = cosine;
    }
    *mean = total / (double)row_count;
}

int main(int argc, char **argv) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'R', 'P', 'A', 'R', '1'};
    unsigned char actual_magic[8];
    unsigned char expected_sha[32];
    unsigned char actual_sha[32];
    uint32_t version;
    uint32_t hidden;
    uint32_t bands;
    uint32_t layers;
    uint32_t feature_dim;
    uint32_t left_count;
    uint32_t right_count;
    int *band_start = NULL;
    int *band_end = NULL;
    int *left_tokens = NULL;
    int *right_tokens = NULL;
    float *expected_left_hidden = NULL;
    float *expected_right_hidden = NULL;
    float *expected_left_identity = NULL;
    float *expected_right_identity = NULL;
    float *expected_entity = NULL;
    float *expected_predicate = NULL;
    float expected_entity_logit;
    float expected_predicate_logit;
    float *actual_left_hidden = NULL;
    float *actual_right_hidden = NULL;
    float *actual_left_identity = NULL;
    float *actual_right_identity = NULL;
    float *actual_entity = NULL;
    float *actual_predicate = NULL;
    float actual_entity_logit;
    float actual_predicate_logit;
    float hidden_max;
    float identity_max;
    float feature_max;
    float logit_max;
    float left_hidden_max;
    float right_hidden_max;
    float left_identity_max;
    float right_identity_max;
    float left_cos_min;
    float right_cos_min;
    double hidden_rmse;
    double identity_rmse;
    double feature_rmse;
    double left_hidden_rmse;
    double right_hidden_rmse;
    double left_identity_rmse;
    double right_identity_rmse;
    double left_cos_mean;
    double right_cos_mean;
    size_t left_hidden_count;
    size_t right_hidden_count;
    size_t left_identity_count;
    size_t right_identity_count;
    FILE *file = NULL;
    bitnet_model_t *backbone = NULL;
    metis_typed_pair_encoder_t *encoder = NULL;
    char error[160];
    int status = 1;

    if (argc != 4) {
        fprintf(stderr,
                "usage: %s MODEL.bntpair BACKBONE.gguf SAMPLE.bin\n",
                argv[0]);
        return 2;
    }
    file = fopen(argv[3], "rb");
    if (file == NULL ||
        read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &bands) != 0 ||
        read_u32(file, &layers) != 0 ||
        read_u32(file, &feature_dim) != 0 ||
        read_u32(file, &left_count) != 0 ||
        read_u32(file, &right_count) != 0 ||
        hidden == 0 || bands < 2 || layers < bands ||
        feature_dim == 0 || left_count == 0 ||
        right_count == 0)
        goto cleanup;
    band_start = (int *)malloc(bands * sizeof(int));
    band_end = (int *)malloc(bands * sizeof(int));
    if (band_start == NULL || band_end == NULL)
        goto cleanup;
    for (uint32_t band = 0; band < bands; ++band) {
        uint32_t start;
        uint32_t end;
        if (read_u32(file, &start) != 0 ||
            read_u32(file, &end) != 0 ||
            start > INT32_MAX || end > INT32_MAX)
            goto cleanup;
        band_start[band] = (int)start;
        band_end[band] = (int)end;
    }
    if (read_exact(file, expected_sha, sizeof expected_sha) != 0 ||
        read_values(
            file, left_count, sizeof(int),
            (void **)&left_tokens) != 0 ||
        read_values(
            file, right_count, sizeof(int),
            (void **)&right_tokens) != 0)
        goto cleanup;
    left_hidden_count =
        (size_t)left_count * bands * hidden;
    right_hidden_count =
        (size_t)right_count * bands * hidden;
    left_identity_count =
        (size_t)left_count * hidden;
    right_identity_count =
        (size_t)right_count * hidden;
    if (read_values(
            file, left_hidden_count, sizeof(float),
            (void **)&expected_left_hidden) != 0 ||
        read_values(
            file, right_hidden_count, sizeof(float),
            (void **)&expected_right_hidden) != 0 ||
        read_values(
            file, left_identity_count, sizeof(float),
            (void **)&expected_left_identity) != 0 ||
        read_values(
            file, right_identity_count, sizeof(float),
            (void **)&expected_right_identity) != 0 ||
        read_values(
            file, feature_dim, sizeof(float),
            (void **)&expected_entity) != 0 ||
        read_values(
            file, feature_dim, sizeof(float),
            (void **)&expected_predicate) != 0 ||
        read_exact(file, &expected_entity_logit,
                   sizeof expected_entity_logit) != 0 ||
        read_exact(file, &expected_predicate_logit,
                   sizeof expected_predicate_logit) != 0 ||
        fgetc(file) != EOF ||
        bitnet_sha256_file(argv[2], actual_sha) != 0 ||
        memcmp(actual_sha, expected_sha, sizeof actual_sha) != 0)
        goto cleanup;
    fclose(file);
    file = NULL;
    backbone = bitnet_load_model(argv[2]);
    if (backbone == NULL ||
        bitnet_embedding_length(backbone) != (int)hidden)
        goto cleanup;
    encoder = metis_typed_pair_encoder_load(
        argv[1], argv[2], (int)hidden,
        error, sizeof error);
    if (encoder == NULL ||
        encoder->band_count != (int)bands ||
        encoder->layer_count != (int)layers ||
        encoder->head_width != (int)feature_dim)
        goto cleanup;
    if (encode_tokens(
            backbone, band_start, band_end, (int)bands,
            left_tokens, (int)left_count,
            &actual_left_hidden,
            &actual_left_identity) != 0 ||
        encode_tokens(
            backbone, band_start, band_end, (int)bands,
            right_tokens, (int)right_count,
            &actual_right_hidden,
            &actual_right_identity) != 0)
        goto cleanup;
    actual_entity = (float *)malloc(
        feature_dim * sizeof(float));
    actual_predicate = (float *)malloc(
        feature_dim * sizeof(float));
    if (actual_entity == NULL || actual_predicate == NULL ||
        metis_typed_pair_encoder_score(
            encoder,
            actual_left_hidden, left_count,
            actual_left_identity,
            actual_right_hidden, right_count,
            actual_right_identity,
            actual_entity, actual_predicate,
            &actual_entity_logit,
            &actual_predicate_logit) != 0)
        goto cleanup;
    error_metrics(
        actual_left_hidden, expected_left_hidden,
        left_hidden_count, &left_hidden_max,
        &left_hidden_rmse);
    error_metrics(
        actual_right_hidden, expected_right_hidden,
        right_hidden_count, &right_hidden_max,
        &right_hidden_rmse);
    hidden_max = fmaxf(left_hidden_max, right_hidden_max);
    hidden_rmse = sqrt(
        (left_hidden_rmse * left_hidden_rmse *
             (double)left_hidden_count +
         right_hidden_rmse * right_hidden_rmse *
             (double)right_hidden_count) /
        (double)(left_hidden_count + right_hidden_count));
    error_metrics(
        actual_left_identity, expected_left_identity,
        left_identity_count, &left_identity_max,
        &left_identity_rmse);
    error_metrics(
        actual_right_identity, expected_right_identity,
        right_identity_count, &right_identity_max,
        &right_identity_rmse);
    identity_max =
        fmaxf(left_identity_max, right_identity_max);
    identity_rmse = sqrt(
        (left_identity_rmse * left_identity_rmse *
             (double)left_identity_count +
         right_identity_rmse * right_identity_rmse *
             (double)right_identity_count) /
        (double)(left_identity_count + right_identity_count));
    cosine_metrics(
        actual_left_hidden, expected_left_hidden,
        (size_t)left_count * bands, hidden,
        &left_cos_mean, &left_cos_min);
    cosine_metrics(
        actual_right_hidden, expected_right_hidden,
        (size_t)right_count * bands, hidden,
        &right_cos_mean, &right_cos_min);
    error_metrics(
        actual_entity, expected_entity,
        feature_dim, &feature_max, &feature_rmse);
    {
        float predicate_max;
        double predicate_rmse;
        error_metrics(
            actual_predicate, expected_predicate,
            feature_dim, &predicate_max, &predicate_rmse);
        feature_max = fmaxf(feature_max, predicate_max);
        feature_rmse = sqrt(
            (feature_rmse * feature_rmse +
             predicate_rmse * predicate_rmse) / 2.0);
    }
    logit_max = fmaxf(
        fabsf(actual_entity_logit - expected_entity_logit),
        fabsf(actual_predicate_logit -
              expected_predicate_logit));
    printf(
        "test_typed_runtime_parity: "
        "identity_max=%.9g identity_rmse=%.9g "
        "hidden_max=%.9g hidden_rmse=%.9g "
        "hidden_cos_mean=%.9g/%.9g "
        "hidden_cos_min=%.9g/%.9g "
        "feature_max=%.9g feature_rmse=%.9g "
        "logit_max=%.9g\n",
        identity_max, identity_rmse,
        hidden_max, hidden_rmse,
        left_cos_mean, right_cos_mean,
        left_cos_min, right_cos_min,
        feature_max, feature_rmse, logit_max);
    if (!isfinite(hidden_rmse) ||
        !isfinite(feature_rmse) ||
        !isfinite(logit_max) ||
        identity_max > 1e-7f)
        goto cleanup;
    status = 0;
cleanup:
    if (file != NULL) fclose(file);
    bitnet_free_model(backbone);
    metis_typed_pair_encoder_free(encoder);
    free(band_start);
    free(band_end);
    free(left_tokens);
    free(right_tokens);
    free(expected_left_hidden);
    free(expected_right_hidden);
    free(expected_left_identity);
    free(expected_right_identity);
    free(expected_entity);
    free(expected_predicate);
    free(actual_left_hidden);
    free(actual_right_hidden);
    free(actual_left_identity);
    free(actual_right_identity);
    free(actual_entity);
    free(actual_predicate);
    return status;
}
