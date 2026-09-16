#include "metis/typed_link_model.h"
#include "sha256.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    RANK = 2,
    FEATURE_DIM = 3,
    JOINT_HIDDEN = 4,
    TENSOR_COUNT = 8
};

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

static void write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4];
    for (int index = 0; index < 4; ++index)
        bytes[index] = (unsigned char)(value >> (8 * index));
    fwrite(bytes, 1, sizeof bytes, file);
}

static int write_model(
    const char *path, const uint8_t sha[32]) {
    static const float joint_hidden_weight[24] = {
        1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1,
        0, 0, 1, 0, 1, 0,
    };
    static const float joint_hidden_bias[4] =
        {0.1f, -0.2f, 0.3f, -0.4f};
    static const float joint_output_weight[4] =
        {0.5f, -0.25f, 0.1f, 0.2f};
    static const float joint_output_bias = 0.05f;
    static const float exists_hidden_weight[10] = {
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
    };
    static const float exists_hidden_bias[2] =
        {0.1f, -0.1f};
    static const float exists_output_weight[2] =
        {0.4f, -0.3f};
    static const float exists_output_bias = 0.2f;
    const void *payloads[TENSOR_COUNT] = {
        joint_hidden_weight,
        joint_hidden_bias,
        joint_output_weight,
        &joint_output_bias,
        exists_hidden_weight,
        exists_hidden_bias,
        exists_output_weight,
        &exists_output_bias,
    };
    const uint32_t sizes[TENSOR_COUNT] = {
        sizeof joint_hidden_weight,
        sizeof joint_hidden_bias,
        sizeof joint_output_weight,
        sizeof joint_output_bias,
        sizeof exists_hidden_weight,
        sizeof exists_hidden_bias,
        sizeof exists_output_weight,
        sizeof exists_output_bias,
    };
    FILE *file = fopen(path, "wb");
    if (file == NULL) return -1;
    fwrite("BNTLINK1", 1, 8, file);
    write_u32(file, 1);
    write_u32(file, 2);
    write_u32(file, RANK);
    write_u32(file, FEATURE_DIM);
    write_u32(file, JOINT_HIDDEN);
    write_u32(file, 5);
    write_u32(file, TENSOR_COUNT);
    fwrite(sha, 1, 32, file);
    for (int index = 0; index < TENSOR_COUNT; ++index) {
        write_u32(file, sizes[index]);
        write_u32(file, crc32_bytes(
            payloads[index], sizes[index]));
    }
    for (int index = 0; index < TENSOR_COUNT; ++index)
        fwrite(payloads[index], 1, sizes[index], file);
    fclose(file);
    return 0;
}

static float gelu(float value) {
    return 0.5f * value *
           (1.0f + erff(value * 0.7071067811865475f));
}

static float reference_joint(
    const float *entity, const float *predicate,
    float entity_logit, float predicate_logit) {
    const float hidden[4] = {
        gelu(entity[0] + 0.1f),
        gelu(predicate[0] - 0.2f),
        gelu(entity[1] + predicate[2] + 0.3f),
        gelu(entity[2] + predicate[1] - 0.4f),
    };
    const float residual =
        0.05f + 0.5f * hidden[0] -
        0.25f * hidden[1] + 0.1f * hidden[2] +
        0.2f * hidden[3];
    return fminf(entity_logit, predicate_logit) + residual;
}

static float reference_exists(
    const float *scores, size_t count) {
    float top1 = -FLT_MAX;
    float top2 = -FLT_MAX;
    float mean = 0.0f;
    float variance = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        mean += scores[index];
        if (scores[index] > top1) {
            top2 = top1;
            top1 = scores[index];
        } else if (scores[index] > top2) {
            top2 = scores[index];
        }
    }
    mean /= (float)count;
    for (size_t index = 0; index < count; ++index) {
        const float delta = scores[index] - mean;
        variance += delta * delta;
    }
    float stddev = sqrtf(variance / (float)count);
    if (stddev < 1e-4f) stddev = 1e-4f;
    if (count == 1) top2 = top1;
    const float feature0 = (top1 - mean) / stddev;
    const float feature1 = (top1 - top2) / stddev;
    return 0.2f +
           0.4f * gelu(feature0 + 0.1f) -
           0.3f * gelu(feature1 - 0.1f);
}

int main(void) {
    const char *backbone = "test_typed_link.backbone";
    const char *wrong_backbone =
        "test_typed_link.wrong_backbone";
    const char *path = "test_typed_link.bntlink";
    const float entity_features[6] = {
        1.0f, 0.2f, -0.4f,
        -0.3f, 0.8f, 0.5f,
    };
    const float predicate_features[6] = {
        0.5f, -0.1f, 0.7f,
        0.9f, 0.4f, -0.2f,
    };
    const float entity_logits[2] = {1.2f, 0.3f};
    const float predicate_logits[2] = {0.8f, 0.7f};
    float scores[2] = {0};
    float exists = 0;
    float expected_exists;
    size_t selected = SIZE_MAX;
    uint8_t sha[32];
    char error[160];
    FILE *file = NULL;
    metis_typed_link_model_t *model = NULL;
    int status = 1;

    file = fopen(backbone, "wb");
    if (file == NULL) goto cleanup;
    fwrite("backbone", 1, 8, file);
    fclose(file);
    file = NULL;
    if (bitnet_sha256_file(backbone, sha) != 0 ||
        write_model(path, sha) != 0)
        goto cleanup;
    model = metis_typed_link_model_load(
        path, backbone, error, sizeof error);
    if (model == NULL ||
        model->rank != RANK ||
        model->pair_feature_dim != FEATURE_DIM ||
        metis_typed_link_score_pairs(
            model, entity_features, predicate_features,
            entity_logits, predicate_logits, 2, scores) != 0)
        goto cleanup;
    for (size_t index = 0; index < 2; ++index) {
        const float expected = reference_joint(
            entity_features + index * FEATURE_DIM,
            predicate_features + index * FEATURE_DIM,
            entity_logits[index], predicate_logits[index]);
        if (fabsf(scores[index] - expected) > 1e-5f) {
            fprintf(stderr,
                    "joint mismatch at %zu: %.9g vs %.9g\n",
                    index, scores[index], expected);
            goto cleanup;
        }
    }
    expected_exists = reference_exists(scores, 2);
    if (metis_typed_link_select_predecessor(
            model, scores, 2, &selected, &exists) != 0 ||
        fabsf(exists - expected_exists) > 1e-5f ||
        selected != (
            exists > 0.0f
                ? (scores[1] > scores[0] ? 1u : 0u)
                : SIZE_MAX)) {
        fprintf(stderr,
                "exists/select mismatch: %.9g vs %.9g index=%zu\n",
                exists, expected_exists, selected);
        goto cleanup;
    }
    metis_typed_link_model_free(model);
    model = NULL;

    file = fopen(wrong_backbone, "wb");
    if (file == NULL) goto cleanup;
    fwrite("different", 1, 9, file);
    fclose(file);
    file = NULL;
    model = metis_typed_link_model_load(
        path, wrong_backbone, error, sizeof error);
    if (model != NULL ||
        strstr(error, "SHA-256 mismatch") == NULL) {
        fprintf(stderr,
                "wrong backbone was not rejected: %s\n", error);
        goto cleanup;
    }
    status = 0;
    puts("test_typed_link_model: OK");
cleanup:
    if (file != NULL) fclose(file);
    metis_typed_link_model_free(model);
    remove(path);
    remove(backbone);
    remove(wrong_backbone);
    return status;
}
