#include "memory_controller.h"

#include "sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char k_magic[8] =
    {'B', 'N', 'C', 'T', 'R', 'L', '1', 0};

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

metis_memory_controller_t *metis_memory_controller_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size) {
    FILE *file = NULL;
    metis_memory_controller_t *controller = NULL;
    unsigned char magic[8];
    uint32_t version, hidden, rank, pooling;
    uint32_t query_crc, entry_crc;
    uint8_t actual_sha256[32];
    size_t count, bytes;
#define FAIL(...) do { \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error != NULL && error_size > 0) error[0] = '\0';
    if (path == NULL || backbone_path == NULL || expected_hidden <= 0)
        FAIL("invalid controller arguments");
    file = fopen(path, "rb");
    if (file == NULL) FAIL("cannot open controller");
    if (read_exact(file, magic, sizeof magic) != 0 ||
        memcmp(magic, k_magic, sizeof magic) != 0)
        FAIL("bad controller magic");
    if (read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &rank) != 0 ||
        read_u32(file, &pooling) != 0)
        FAIL("bad controller header");
    if (hidden != (uint32_t)expected_hidden || rank < 1 || rank > 1024 ||
        (pooling != 1 && pooling != 2))
        FAIL("controller geometry mismatch");
    controller = (metis_memory_controller_t *)calloc(
        1, sizeof *controller);
    if (controller == NULL) FAIL("out of memory");
    controller->hidden_dim = (int)hidden;
    controller->rank = (int)rank;
    controller->pooling = (int)pooling;
    if (read_f32(file, &controller->temperature) != 0 ||
        !(controller->temperature > 0.0f))
        FAIL("bad controller temperature");
    if (read_exact(
            file, controller->backbone_sha256,
            sizeof controller->backbone_sha256) != 0 ||
        read_u32(file, &query_crc) != 0 ||
        read_u32(file, &entry_crc) != 0)
        FAIL("truncated controller header");
    if (bitnet_sha256_file(backbone_path, actual_sha256) != 0 ||
        memcmp(actual_sha256, controller->backbone_sha256,
               sizeof actual_sha256) != 0)
        FAIL("controller backbone SHA-256 mismatch");
    count = (size_t)hidden * (size_t)rank;
    if (count > SIZE_MAX / sizeof(float)) FAIL("controller too large");
    bytes = count * sizeof(float);
    controller->query_projection = (float *)malloc(bytes);
    controller->entry_projection = (float *)malloc(bytes);
    if (controller->query_projection == NULL ||
        controller->entry_projection == NULL)
        FAIL("out of memory");
    if (read_exact(file, controller->query_projection, bytes) != 0 ||
        read_exact(file, controller->entry_projection, bytes) != 0 ||
        fgetc(file) != EOF)
        FAIL("truncated or trailing controller data");
    if (crc32_bytes(controller->query_projection, bytes) != query_crc ||
        crc32_bytes(controller->entry_projection, bytes) != entry_crc)
        FAIL("controller CRC mismatch");
    fclose(file);
    return controller;
fail:
    if (file != NULL) fclose(file);
    metis_memory_controller_free(controller);
    return NULL;
#undef FAIL
}

void metis_memory_controller_free(
    metis_memory_controller_t *controller) {
    if (controller == NULL) return;
    free(controller->query_projection);
    free(controller->entry_projection);
    free(controller);
}

int metis_memory_controller_project(
    const metis_memory_controller_t *controller, const float *hidden,
    int is_query, float *output) {
    const float *weight;
    float norm = 0.0f;
    if (controller == NULL || hidden == NULL || output == NULL) return -1;
    weight = is_query ? controller->query_projection :
                        controller->entry_projection;
    for (int row = 0; row < controller->rank; ++row) {
        const float *weight_row =
            weight + (size_t)row * (size_t)controller->hidden_dim;
        float value = 0.0f;
        for (int column = 0; column < controller->hidden_dim; ++column)
            value += weight_row[column] * hidden[column];
        output[row] = value;
        norm += value * value;
    }
    norm = 1.0f / sqrtf(norm + 1e-12f);
    for (int row = 0; row < controller->rank; ++row)
        output[row] *= norm;
    return 0;
}

float metis_memory_controller_score(
    const metis_memory_controller_t *controller,
    const float *query_key, const float *entry_key) {
    float score = 0.0f;
    if (controller == NULL || query_key == NULL || entry_key == NULL)
        return -INFINITY;
    for (int i = 0; i < controller->rank; ++i)
        score += query_key[i] * entry_key[i];
    return score / controller->temperature;
}
