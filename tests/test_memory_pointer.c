#include "metis/memory_pointer.h"
#include "sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)value, (unsigned char)(value >> 8),
        (unsigned char)(value >> 16), (unsigned char)(value >> 24)};
    fwrite(bytes, 1, sizeof bytes, file);
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

int main(void) {
    const char *backbone = "test_memory_pointer.backbone";
    const char *model = "test_memory_pointer.bnptr";
    const char *model_v2 = "test_memory_pointer_v2.bnptr";
    const unsigned char magic[8] = {'B','N','P','T','R','1',0,0};
    const float start_w[2] = {1.0f, 0.0f};
    const float start_b = 0.0f;
    const float end_w[2] = {0.0f, 1.0f};
    const float end_b = 0.0f;
    const float hidden[8] = {
        0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f};
    uint8_t sha[32];
    char error[128];
    size_t start, end;
    float confidence;
    FILE *file = fopen(backbone, "wb");
    if (file == NULL) return 1;
    fwrite("backbone", 1, 8, file);
    fclose(file);
    if (bitnet_sha256_file(backbone, sha) != 0) return 1;
    file = fopen(model, "wb");
    if (file == NULL) return 1;
    fwrite(magic, 1, sizeof magic, file);
    put_u32(file, 1);
    put_u32(file, 2);
    put_u32(file, 3);
    {
        float threshold = 0.5f;
        fwrite(&threshold, sizeof threshold, 1, file);
    }
    fwrite(sha, 1, sizeof sha, file);
    put_u32(file, 4);
    put_u32(file, sizeof start_w);
    put_u32(file, crc32_bytes(start_w, sizeof start_w));
    put_u32(file, sizeof start_b);
    put_u32(file, crc32_bytes(&start_b, sizeof start_b));
    put_u32(file, sizeof end_w);
    put_u32(file, crc32_bytes(end_w, sizeof end_w));
    put_u32(file, sizeof end_b);
    put_u32(file, crc32_bytes(&end_b, sizeof end_b));
    fwrite(start_w, 1, sizeof start_w, file);
    fwrite(&start_b, 1, sizeof start_b, file);
    fwrite(end_w, 1, sizeof end_w, file);
    fwrite(&end_b, 1, sizeof end_b, file);
    fclose(file);

    metis_memory_pointer_t *pointer = metis_memory_pointer_load(
        model, backbone, 2, error, sizeof error);
    if (pointer == NULL) {
        fprintf(stderr, "load failed: %s\n", error);
        return 1;
    }
    if (metis_memory_pointer_select(
            pointer, hidden, 4, &start, &end, &confidence) != 1 ||
        start != 1 || end != 2 || confidence < 0.5f) {
        fprintf(stderr, "bad span: %zu..%zu margin=%g\n",
                start, end, confidence);
        return 1;
    }
    metis_memory_pointer_free(pointer);

    {
        const unsigned char magic_v2[8] =
            {'B','N','P','T','R','2',0,0};
        const float null_w[8] = {0};
        const float null_b[2] = {1.0f, 1.0f};
        file = fopen(model_v2, "wb");
        if (file == NULL) return 1;
        fwrite(magic_v2, 1, sizeof magic_v2, file);
        put_u32(file, 2);
        put_u32(file, 2);
        put_u32(file, 3);
        {
            float threshold = 0.5f;
            fwrite(&threshold, sizeof threshold, 1, file);
        }
        fwrite(sha, 1, sizeof sha, file);
        put_u32(file, 6);
#define PUT_DESC(value) do { \
    put_u32(file, sizeof(value)); \
    put_u32(file, crc32_bytes(&(value), sizeof(value))); \
} while (0)
        PUT_DESC(start_w);
        PUT_DESC(start_b);
        PUT_DESC(end_w);
        PUT_DESC(end_b);
        PUT_DESC(null_w);
        PUT_DESC(null_b);
#undef PUT_DESC
        fwrite(start_w, 1, sizeof start_w, file);
        fwrite(&start_b, 1, sizeof start_b, file);
        fwrite(end_w, 1, sizeof end_w, file);
        fwrite(&end_b, 1, sizeof end_b, file);
        fwrite(null_w, 1, sizeof null_w, file);
        fwrite(null_b, 1, sizeof null_b, file);
        fclose(file);
        pointer = metis_memory_pointer_load(
            model_v2, backbone, 2, error, sizeof error);
        if (pointer == NULL || !pointer->has_null ||
            metis_memory_pointer_select(
                pointer, hidden, 4, &start, &end, &confidence) != 1 ||
            start != 1 || end != 2 || confidence < 4.9f) {
            fprintf(stderr, "v2 pointer failed: %s margin=%g\n",
                    error, confidence);
            return 1;
        }
        pointer->null_bias[0] = 5.0f;
        pointer->null_bias[1] = 5.0f;
        if (metis_memory_pointer_select(
                pointer, hidden, 4, &start, &end, &confidence) != 0) {
            fprintf(stderr, "v2 null rejection failed: margin=%g\n",
                    confidence);
            return 1;
        }
        metis_memory_pointer_free(pointer);
    }
    remove(model);
    remove(model_v2);
    remove(backbone);
    puts("test_memory_pointer: OK");
    return 0;
}
