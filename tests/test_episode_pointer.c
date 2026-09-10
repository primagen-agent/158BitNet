#include "metis/episode_pointer.h"
#include "sha256.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void tensor_entry(FILE *file, const void *data, uint32_t size) {
    write_u32(file, size);
    write_u32(file, crc32_bytes(data, size));
}

int main(void) {
    const char *backbone = "test_episode_pointer.backbone";
    const char *wrong_backbone = "test_episode_pointer.wrong_backbone";
    const char *model = "test_episode_pointer.bneptr";
    const float query_start[4] = {1, 0, 0, 1};
    const float query_end[4] = {1, 0, 0, 1};
    const float source_start[4] = {1, 0, 0, 1};
    const float source_end[4] = {1, 0, 0, 1};
    const float local_weight[2] = {0, 0};
    const float zero = 0;
    const float local_scale[2] = {-20, -20};
    const void *payloads[9] = {
        query_start, query_end, source_start, source_end,
        local_weight, &zero, local_weight, &zero, local_scale,
    };
    const uint32_t sizes[9] = {
        16, 16, 16, 16, 8, 4, 8, 4, 8,
    };
    const float source_hidden[6] = {
        0, 1,
        1, 0,
        0, 1,
    };
    const float query_hidden[2] = {1, 0};
    uint8_t sha[32];
    FILE *file;
    metis_episode_pointer_t *pointer;
    size_t start = 0, end = 0;
    float score = 0;
    char error[128];

    file = fopen(backbone, "wb");
    if (file == NULL) return 1;
    fwrite("backbone", 1, 8, file);
    fclose(file);
    if (bitnet_sha256_file(backbone, sha) != 0) return 1;
    file = fopen(model, "wb");
    if (file == NULL) return 1;
    fwrite("BNEPTR1\0", 1, 8, file);
    write_u32(file, 1);
    write_u32(file, 2);
    write_u32(file, 2);
    write_u32(file, 3);
    fwrite(sha, 1, sizeof sha, file);
    write_u32(file, 9);
    for (int index = 0; index < 9; ++index)
        tensor_entry(file, payloads[index], sizes[index]);
    for (int index = 0; index < 9; ++index)
        fwrite(payloads[index], 1, sizes[index], file);
    fclose(file);

    pointer = metis_episode_pointer_load(
        model, backbone, 2, error, sizeof error);
    if (pointer == NULL ||
        metis_episode_pointer_select(
            pointer, source_hidden, 3, query_hidden, 1,
            &start, &end, &score) != 1 ||
        start != 1 || end != 1 || !isfinite(score)) {
        fprintf(stderr, "episode pointer failed: %s %zu..%zu\n",
                error, start, end);
        return 1;
    }
    metis_episode_pointer_free(pointer);
    file = fopen(wrong_backbone, "wb");
    if (file == NULL) return 1;
    fwrite("different", 1, 9, file);
    fclose(file);
    pointer = metis_episode_pointer_load(
        model, wrong_backbone, 2, error, sizeof error);
    if (pointer != NULL ||
        strstr(error, "SHA-256 mismatch") == NULL) {
        fprintf(stderr, "wrong backbone was not rejected: %s\n", error);
        metis_episode_pointer_free(pointer);
        return 1;
    }
    remove(model);
    remove(backbone);
    remove(wrong_backbone);
    puts("test_episode_pointer: OK");
    return 0;
}
