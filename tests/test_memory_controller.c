#include "metis/memory_controller.h"
#include "sha256.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t crc32_bytes(const void *data, size_t size) {
    const unsigned char *bytes = data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^
                  (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

static int write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4];
    for (int i = 0; i < 4; ++i)
        bytes[i] = (unsigned char)(value >> (8 * i));
    return fwrite(bytes, 1, 4, file) == 4 ? 0 : -1;
}

static int write_f32(FILE *file, float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return write_u32(file, bits);
}

static int write_fixture(
    const char *path, const uint8_t sha[32], int version) {
    const unsigned char magic_v1[8] = {'B','N','C','T','R','L','1',0};
    const unsigned char magic_v3[8] = {'B','N','C','T','R','L','3',0};
    const float query[8] = {1,0,0,0, 0,1,0,0};
    const float entry[8] = {0.5f,0.5f,0,0, 0,0,1,0};
    const float action[16] = {
        -1,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float action_bias[4] = {0,0,0,0};
    FILE *file = fopen(path, "wb");
    if (file == NULL) return -1;
    int failed =
        fwrite(version == 3 ? magic_v3 : magic_v1, 1, 8, file) != 8 ||
        write_u32(file, (uint32_t)version) || write_u32(file, 4) ||
        write_u32(file, 2) || write_u32(file, 2) ||
        write_f32(file, 0.07f) ||
        (version == 3 && write_f32(file, 0.85f)) ||
        fwrite(sha, 1, 32, file) != 32 ||
        write_u32(file, crc32_bytes(query, sizeof query)) ||
        write_u32(file, crc32_bytes(entry, sizeof entry)) ||
        (version == 3 &&
         (write_u32(file, crc32_bytes(action, sizeof action)) ||
          write_u32(file, crc32_bytes(
              action_bias, sizeof action_bias)))) ||
        fwrite(query, 1, sizeof query, file) != sizeof query ||
        fwrite(entry, 1, sizeof entry, file) != sizeof entry ||
        (version == 3 &&
         (fwrite(action, 1, sizeof action, file) != sizeof action ||
          fwrite(action_bias, 1, sizeof action_bias, file) !=
              sizeof action_bias));
    if (fclose(file) != 0) failed = 1;
    if (failed) remove(path);
    return failed ? -1 : 0;
}

static int verify(const char *controller_path, const char *backbone_path,
                  int hidden_dim, int expected_action) {
    char error[256];
    metis_memory_controller_t *controller =
        metis_memory_controller_load(
            controller_path, backbone_path, hidden_dim,
            error, sizeof error);
    if (controller == NULL) {
        fprintf(stderr, "load failed: %s\n", error);
        return -1;
    }
    float *hidden = calloc((size_t)hidden_dim, sizeof(float));
    float *query = calloc((size_t)controller->rank, sizeof(float));
    float *entry = calloc((size_t)controller->rank, sizeof(float));
    if (hidden == NULL || query == NULL || entry == NULL) return -1;
    for (int i = 0; i < hidden_dim; ++i) hidden[i] = (float)(i + 1);
    int failed =
        metis_memory_controller_project(controller, hidden, 1, query) ||
        metis_memory_controller_project(controller, hidden, 0, entry);
    float norm = 0.0f;
    for (int i = 0; i < controller->rank; ++i)
        norm += query[i] * query[i];
    failed |= !isfinite(norm) || fabsf(norm - 1.0f) > 1e-4f;
    failed |= !isfinite(metis_memory_controller_score(
        controller, query, entry));
    if (controller->action_projection != NULL) {
        float confidence = 0.0f;
        int action = metis_memory_controller_classify(
            controller, hidden, &confidence);
        failed |= action < 0 || action > 3;
        if (expected_action >= 0) failed |= action != expected_action;
        failed |= !(confidence > 0.0f);
        if (expected_action >= 0)
            failed |= fabsf(
                controller->address_threshold - 0.85f) > 1e-6f;
        else
            failed |= controller->address_threshold < -1.0f ||
                      controller->address_threshold > 1.0f;
    } else {
        failed |= metis_memory_controller_classify(
            controller, hidden, NULL) != -1;
    }
    free(entry);
    free(query);
    free(hidden);
    metis_memory_controller_free(controller);
    return failed ? -1 : 0;
}

static int self_test(void) {
    const char *backbone = "/tmp/bnctrl-backbone-a";
    const char *other = "/tmp/bnctrl-backbone-b";
    const char *controller_path = "/tmp/bnctrl-test";
    uint8_t sha[32];
    char error[256];
    FILE *file = fopen(backbone, "wb");
    if (file == NULL || fwrite("backbone-a", 1, 10, file) != 10) return 1;
    fclose(file);
    file = fopen(other, "wb");
    if (file == NULL || fwrite("backbone-b", 1, 10, file) != 10) return 1;
    fclose(file);
    if (bitnet_sha256_file(backbone, sha) ||
        write_fixture(controller_path, sha, 1) ||
        verify(controller_path, backbone, 4, -1))
        return 1;
    if (write_fixture(controller_path, sha, 3) ||
        verify(controller_path, backbone, 4, 3))
        return 1;

    metis_memory_controller_t *bad = metis_memory_controller_load(
        controller_path, other, 4, error, sizeof error);
    if (bad != NULL) return 1;

    file = fopen(controller_path, "r+b");
    if (file == NULL || fseek(file, -1, SEEK_END)) return 1;
    int byte = fgetc(file);
    if (byte == EOF || fseek(file, -1, SEEK_CUR)) return 1;
    fputc(byte ^ 1, file);
    fclose(file);
    bad = metis_memory_controller_load(
        controller_path, backbone, 4, error, sizeof error);
    if (bad != NULL) return 1;

    remove(controller_path);
    remove(other);
    remove(backbone);
    printf("test_memory_controller: OK\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 1) return self_test();
    if (argc == 4)
        return verify(argv[1], argv[2], atoi(argv[3]), -1) == 0 ? 0 : 1;
    fprintf(stderr,
            "usage: %s [<controller.bnctrl> <model.gguf> <hidden>]\n",
            argv[0]);
    return 2;
}
