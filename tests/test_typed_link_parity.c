#include "metis/typed_link_model.h"

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

static int close_enough(float left, float right) {
    const float scale = fmaxf(1.0f, fmaxf(fabsf(left), fabsf(right)));
    return fabsf(left - right) <= 5e-5f * scale;
}

int main(int argc, char **argv) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'L', 'P', 'A', 'R', '1'};
    unsigned char actual_magic[8];
    uint32_t version;
    uint32_t feature_dim;
    uint32_t pair_count;
    uint32_t expected_selected;
    float *entity = NULL;
    float *predicate = NULL;
    float *entity_logits = NULL;
    float *predicate_logits = NULL;
    float *expected_joint = NULL;
    float *actual_joint = NULL;
    float expected_exists;
    float actual_exists;
    float max_joint_error = 0.0f;
    size_t actual_selected = SIZE_MAX;
    FILE *file = NULL;
    metis_typed_link_model_t *model = NULL;
    char error[160];
    int status = 1;
    size_t feature_count;

    if (argc != 4) {
        fprintf(stderr,
                "usage: %s MODEL.bntlink BACKBONE.gguf SAMPLE.bin\n",
                argv[0]);
        return 2;
    }
    model = metis_typed_link_model_load(
        argv[1], argv[2], error, sizeof error);
    if (model == NULL) {
        fprintf(stderr, "cannot load typed-link model: %s\n", error);
        goto cleanup;
    }
    file = fopen(argv[3], "rb");
    if (file == NULL ||
        read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &feature_dim) != 0 ||
        read_u32(file, &pair_count) != 0 ||
        feature_dim != (uint32_t)model->pair_feature_dim ||
        pair_count < 2)
        goto cleanup;
    feature_count = (size_t)feature_dim * pair_count;
    entity = (float *)malloc(feature_count * sizeof(float));
    predicate = (float *)malloc(feature_count * sizeof(float));
    entity_logits = (float *)malloc(pair_count * sizeof(float));
    predicate_logits = (float *)malloc(pair_count * sizeof(float));
    expected_joint = (float *)malloc(pair_count * sizeof(float));
    actual_joint = (float *)malloc(pair_count * sizeof(float));
    if (entity == NULL || predicate == NULL ||
        entity_logits == NULL || predicate_logits == NULL ||
        expected_joint == NULL || actual_joint == NULL ||
        read_exact(file, entity,
                   feature_count * sizeof(float)) != 0 ||
        read_exact(file, predicate,
                   feature_count * sizeof(float)) != 0 ||
        read_exact(file, entity_logits,
                   pair_count * sizeof(float)) != 0 ||
        read_exact(file, predicate_logits,
                   pair_count * sizeof(float)) != 0 ||
        read_exact(file, expected_joint,
                   pair_count * sizeof(float)) != 0 ||
        read_exact(file, &expected_exists,
                   sizeof expected_exists) != 0 ||
        read_u32(file, &expected_selected) != 0 ||
        fgetc(file) != EOF)
        goto cleanup;
    if (metis_typed_link_score_pairs(
            model, entity, predicate,
            entity_logits, predicate_logits,
            pair_count, actual_joint) != 0 ||
        metis_typed_link_select_predecessor(
            model, actual_joint, pair_count,
            &actual_selected, &actual_exists) != 0)
        goto cleanup;
    for (uint32_t index = 0; index < pair_count; ++index)
        if (!close_enough(
                actual_joint[index],
                expected_joint[index])) {
            fprintf(stderr,
                    "joint parity mismatch at %u: %.9g vs %.9g\n",
                    index, actual_joint[index],
                    expected_joint[index]);
            goto cleanup;
        } else {
            const float error_value = fabsf(
                actual_joint[index] -
                expected_joint[index]);
            if (error_value > max_joint_error)
                max_joint_error = error_value;
        }
    if (!close_enough(actual_exists, expected_exists) ||
        actual_selected != (
            expected_selected == UINT32_MAX
                ? SIZE_MAX : (size_t)expected_selected)) {
        fprintf(stderr,
                "selection parity mismatch: %.9g vs %.9g, %zu vs %u\n",
                actual_exists, expected_exists,
                actual_selected, expected_selected);
        goto cleanup;
    }
    printf(
        "test_typed_link_parity: OK pairs=%u "
        "max_joint_error=%.9g exists_error=%.9g\n",
        pair_count, max_joint_error,
        fabsf(actual_exists - expected_exists));
    status = 0;
cleanup:
    if (file != NULL) fclose(file);
    metis_typed_link_model_free(model);
    free(entity);
    free(predicate);
    free(entity_logits);
    free(predicate_logits);
    free(expected_joint);
    free(actual_joint);
    return status;
}
