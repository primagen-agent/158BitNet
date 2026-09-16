#include "metis/typed_writer_model.h"

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

int main(int argc, char **argv) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'W', 'P', 'A', 'R', '1'};
    unsigned char actual_magic[8];
    uint32_t version, hidden, bands, tokens, operation;
    uint32_t expected_start[4], expected_end[4], expected_anchor[4];
    float expected_logits[2];
    float *hidden_values = NULL;
    float *identity = NULL;
    FILE *file = NULL;
    metis_typed_writer_model_t *model = NULL;
    metis_typed_writer_result_t result;
    char error[256];
    size_t hidden_count;
    int status = 1;
    if (argc != 4) {
        fprintf(
            stderr,
            "usage: %s MODEL.bntwrite BACKBONE.gguf SAMPLE.bin\n",
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
        read_u32(file, &tokens) != 0 ||
        read_u32(file, &operation) != 0 ||
        read_f32(file, &expected_logits[0]) != 0 ||
        read_f32(file, &expected_logits[1]) != 0)
        goto cleanup;
    for (int field = 0; field < 4; ++field)
        if (read_u32(file, &expected_start[field]) != 0 ||
            read_u32(file, &expected_end[field]) != 0 ||
            read_u32(file, &expected_anchor[field]) != 0)
            goto cleanup;
    if (hidden == 0 || bands == 0 || tokens == 0 ||
        (size_t)tokens > SIZE_MAX / (size_t)bands ||
        (size_t)tokens * (size_t)bands >
            SIZE_MAX / (size_t)hidden)
        goto cleanup;
    hidden_count = (size_t)tokens *
        (size_t)bands * (size_t)hidden;
    hidden_values = (float *)malloc(
        hidden_count * sizeof(float));
    identity = (float *)malloc(
        (size_t)tokens * (size_t)hidden * sizeof(float));
    if (hidden_values == NULL || identity == NULL ||
        read_exact(
            file, hidden_values,
            hidden_count * sizeof(float)) != 0 ||
        read_exact(
            file, identity,
            (size_t)tokens * (size_t)hidden *
                sizeof(float)) != 0 ||
        fgetc(file) != EOF)
        goto cleanup;
    model = metis_typed_writer_model_load(
        argv[1], argv[2], (int)hidden, error, sizeof error);
    if (model == NULL) {
        fprintf(stderr, "load failed: %s\n", error);
        goto cleanup;
    }
    if (model->band_count != (int)bands ||
        metis_typed_writer_predict(
            model, hidden_values, identity,
            tokens, &result) != 0)
        goto cleanup;
    if (result.operation != (int)operation) {
        fprintf(stderr, "operation mismatch: %d != %u\n",
                result.operation, operation);
        goto cleanup;
    }
    for (int index = 0; index < 2; ++index) {
        float difference = fabsf(
            result.operation_logits[index] -
            expected_logits[index]);
        if (difference > 2e-4f) {
            fprintf(
                stderr,
                "operation logit %d mismatch: %.8f != %.8f\n",
                index, result.operation_logits[index],
                expected_logits[index]);
            goto cleanup;
        }
    }
    for (int field = 0; field < 4; ++field) {
        if (result.span_start[field] != expected_start[field] ||
            result.span_end[field] != expected_end[field] ||
            result.anchor[field] != expected_anchor[field]) {
            fprintf(
                stderr,
                "field %d mismatch: span=%zu:%zu anchor=%zu "
                "expected=%u:%u anchor=%u\n",
                field,
                result.span_start[field],
                result.span_end[field],
                result.anchor[field],
                expected_start[field],
                expected_end[field],
                expected_anchor[field]);
            goto cleanup;
        }
    }
    printf(
        "typed writer parity passed: tokens=%u operation=%u\n",
        tokens, operation);
    status = 0;
cleanup:
    if (file != NULL) fclose(file);
    metis_typed_writer_model_free(model);
    free(hidden_values);
    free(identity);
    return status;
}
