#include "metis/typed_pair_encoder.h"

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

static int read_floats(
    FILE *file, size_t count, float **output) {
    if (count == 0 || count > SIZE_MAX / sizeof(float))
        return -1;
    *output = (float *)malloc(count * sizeof(float));
    if (*output == NULL ||
        read_exact(file, *output,
                   count * sizeof(float)) != 0) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'P', 'A', 'R', 'F', '1'};
    unsigned char actual_magic[8];
    uint32_t version;
    uint32_t hidden;
    uint32_t bands;
    uint32_t feature_dim;
    uint32_t left_count;
    uint32_t right_count;
    size_t left_hidden_count;
    size_t right_hidden_count;
    float *left_hidden = NULL;
    float *left_identity = NULL;
    float *right_hidden = NULL;
    float *right_identity = NULL;
    float *expected_entity = NULL;
    float *expected_predicate = NULL;
    float expected_entity_logit;
    float expected_predicate_logit;
    float *actual_entity = NULL;
    float *actual_predicate = NULL;
    float actual_entity_logit;
    float actual_predicate_logit;
    float max_feature_error = 0.0f;
    float logit_error;
    FILE *file = NULL;
    metis_typed_pair_encoder_t *model = NULL;
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
        read_u32(file, &feature_dim) != 0 ||
        read_u32(file, &left_count) != 0 ||
        read_u32(file, &right_count) != 0 ||
        hidden == 0 || bands < 2 || feature_dim == 0 ||
        left_count == 0 || right_count == 0)
        goto cleanup;
    left_hidden_count =
        (size_t)left_count * bands * hidden;
    right_hidden_count =
        (size_t)right_count * bands * hidden;
    if (read_floats(
            file, left_hidden_count, &left_hidden) != 0 ||
        read_floats(
            file, (size_t)left_count * hidden,
            &left_identity) != 0 ||
        read_floats(
            file, right_hidden_count, &right_hidden) != 0 ||
        read_floats(
            file, (size_t)right_count * hidden,
            &right_identity) != 0 ||
        read_floats(
            file, feature_dim, &expected_entity) != 0 ||
        read_floats(
            file, feature_dim, &expected_predicate) != 0 ||
        read_exact(file, &expected_entity_logit,
                   sizeof expected_entity_logit) != 0 ||
        read_exact(file, &expected_predicate_logit,
                   sizeof expected_predicate_logit) != 0 ||
        fgetc(file) != EOF)
        goto cleanup;
    fclose(file);
    file = NULL;
    model = metis_typed_pair_encoder_load(
        argv[1], argv[2], (int)hidden,
        error, sizeof error);
    if (model == NULL) {
        fprintf(stderr,
                "cannot load typed-pair encoder: %s\n",
                error);
        goto cleanup;
    }
    if (model->band_count != (int)bands ||
        model->head_width != (int)feature_dim) {
        fprintf(stderr,
                "typed-pair sample geometry mismatch: "
                "bands=%d/%u feature=%d/%u\n",
                model->band_count, bands,
                model->head_width, feature_dim);
        goto cleanup;
    }
    actual_entity = (float *)malloc(
        feature_dim * sizeof(float));
    actual_predicate = (float *)malloc(
        feature_dim * sizeof(float));
    if (actual_entity == NULL || actual_predicate == NULL ||
        metis_typed_pair_encoder_score(
            model,
            left_hidden, left_count, left_identity,
            right_hidden, right_count, right_identity,
            actual_entity, actual_predicate,
            &actual_entity_logit,
            &actual_predicate_logit) != 0)
    {
        fprintf(stderr,
                "typed-pair encoder scoring failed\n");
        goto cleanup;
    }
    for (uint32_t index = 0; index < feature_dim; ++index) {
        float value = fabsf(
            actual_entity[index] -
            expected_entity[index]);
        if (value > max_feature_error)
            max_feature_error = value;
        value = fabsf(
            actual_predicate[index] -
            expected_predicate[index]);
        if (value > max_feature_error)
            max_feature_error = value;
    }
    logit_error = fmaxf(
        fabsf(actual_entity_logit -
              expected_entity_logit),
        fabsf(actual_predicate_logit -
              expected_predicate_logit));
    if (max_feature_error > 2e-4f ||
        logit_error > 2e-4f) {
        fprintf(stderr,
                "typed-pair parity mismatch: "
                "feature=%.9g logit=%.9g\n",
                max_feature_error, logit_error);
        goto cleanup;
    }
    printf(
        "test_typed_pair_parity: OK "
        "feature_error=%.9g logit_error=%.9g\n",
        max_feature_error, logit_error);
    status = 0;
cleanup:
    if (file != NULL) fclose(file);
    metis_typed_pair_encoder_free(model);
    free(left_hidden);
    free(left_identity);
    free(right_hidden);
    free(right_identity);
    free(expected_entity);
    free(expected_predicate);
    free(actual_entity);
    free(actual_predicate);
    return status;
}
