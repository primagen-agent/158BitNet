#include "metis/memory_retriever.h"
#include "sha256.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t crc32_bytes(const void *data, size_t size) {
    const unsigned char *p = data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= p[i];
        for (int b = 0; b < 8; ++b)
            crc = (crc >> 1) ^
                  (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

static void u32(FILE *f, uint32_t v) {
    unsigned char b[4] = {v, v >> 8, v >> 16, v >> 24};
    fwrite(b, 1, 4, f);
}

static void f32(FILE *f, float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    u32(f, bits);
}

int main(void) {
    const char *backbone = "/tmp/bnret-backbone";
    const char *path = "/tmp/bnret-test";
    const unsigned char magic[8] = {'B','N','R','E','T','1',0,0};
    float identity[8] = {1,0,0,0, 0,1,0,0};
    uint8_t sha[32];
    FILE *f = fopen(backbone, "wb");
    if (!f) return 1;
    fwrite("backbone", 1, 8, f);
    fclose(f);
    if (bitnet_sha256_file(backbone, sha)) return 1;
    f = fopen(path, "wb");
    if (!f) return 1;
    fwrite(magic, 1, 8, f);
    u32(f, 1); u32(f, 4); u32(f, 2); u32(f, 2);
    f32(f, 0.5f); f32(f, 0.49f);
    fwrite(sha, 1, 32, f);
    u32(f, 4);
    for (int i = 0; i < 4; ++i) {
        u32(f, sizeof identity);
        u32(f, crc32_bytes(identity, sizeof identity));
    }
    for (int i = 0; i < 4; ++i)
        fwrite(identity, 1, sizeof identity, f);
    fclose(f);
    char error[128];
    metis_memory_retriever_t *r = metis_memory_retriever_load(
        path, backbone, 4, error, sizeof error);
    if (!r) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    float hidden[8] = {1,0,0,0, 0,1,0,0};
    float qg[2], eg[2], qt[4], et[4];
    int failed =
        metis_memory_retriever_encode(r, hidden, 2, 1, qg, qt) ||
        metis_memory_retriever_encode(r, hidden, 2, 0, eg, et);
    float score = metis_memory_retriever_score(
        r, qg, qt, 2, eg, et, 2);
    float residual = metis_memory_retriever_residual(r, score);
    failed |= !isfinite(score) || !(residual > 0.0f && residual < 0.5f);
    metis_memory_retriever_free(r);
    remove(path);
    remove(backbone);
    if (failed) return 1;
    puts("test_memory_retriever: OK");
    return 0;
}
