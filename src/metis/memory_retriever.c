#include "memory_retriever.h"

#include "sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char k_magic[8] =
    {'B','N','R','E','T','1',0,0};

static int read_exact(FILE *file, void *out, size_t size) {
    return fread(out, 1, size, file) == size ? 0 : -1;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char b[4];
    if (read_exact(file, b, 4) != 0) return -1;
    *value = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
             ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static int read_f32(FILE *file, float *value) {
    uint32_t bits;
    if (read_u32(file, &bits) != 0) return -1;
    memcpy(value, &bits, 4);
    return 0;
}

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

metis_memory_retriever_t *metis_memory_retriever_load(
    const char *path, const char *backbone_path, int expected_hidden,
    char *error, size_t error_size) {
    FILE *file = NULL;
    metis_memory_retriever_t *r = NULL;
    unsigned char magic[8];
    uint32_t version, hidden, rank, pooling, count;
    uint32_t crc[4], bytes32;
    uint8_t actual[32];
    size_t elements, bytes;
#define FAIL(...) do { \
    if (error && error_size) snprintf(error, error_size, __VA_ARGS__); \
    goto fail; \
} while (0)
    if (error && error_size) error[0] = '\0';
    if (!path || !backbone_path || expected_hidden <= 0)
        FAIL("invalid retriever arguments");
    file = fopen(path, "rb");
    if (!file) FAIL("cannot open retriever");
    if (read_exact(file, magic, 8) ||
        memcmp(magic, k_magic, 8) != 0 ||
        read_u32(file, &version) || version != 1 ||
        read_u32(file, &hidden) || read_u32(file, &rank) ||
        read_u32(file, &pooling) || pooling != 2 ||
        hidden != (uint32_t)expected_hidden || rank < 1 || rank > 1024)
        FAIL("bad retriever header");
    r = calloc(1, sizeof *r);
    if (!r) FAIL("out of memory");
    r->hidden_dim = (int)hidden;
    r->rank = (int)rank;
    if (read_f32(file, &r->late_weight) ||
        read_f32(file, &r->max_residual) ||
        !(r->late_weight >= 0.0f && r->late_weight <= 1.0f) ||
        !(r->max_residual >= 0.0f && r->max_residual < 0.5f) ||
        read_exact(file, r->backbone_sha256, 32) ||
        read_u32(file, &count) || count != 4)
        FAIL("bad retriever parameters");
    for (int i = 0; i < 4; ++i) {
        if (read_u32(file, &bytes32) || read_u32(file, &crc[i]))
            FAIL("truncated retriever payload table");
        elements = (size_t)hidden * (size_t)rank;
        bytes = elements * sizeof(float);
        if (bytes32 != bytes) FAIL("retriever payload size mismatch");
    }
    if (bitnet_sha256_file(backbone_path, actual) ||
        memcmp(actual, r->backbone_sha256, 32) != 0)
        FAIL("retriever backbone SHA-256 mismatch");
    elements = (size_t)hidden * (size_t)rank;
    bytes = elements * sizeof(float);
    r->query_global = malloc(bytes);
    r->entry_global = malloc(bytes);
    r->query_token = malloc(bytes);
    r->entry_token = malloc(bytes);
    if (!r->query_global || !r->entry_global ||
        !r->query_token || !r->entry_token)
        FAIL("out of memory");
    float *payloads[4] = {
        r->query_global, r->entry_global,
        r->query_token, r->entry_token};
    for (int i = 0; i < 4; ++i)
        if (read_exact(file, payloads[i], bytes) ||
            crc32_bytes(payloads[i], bytes) != crc[i])
            FAIL("retriever CRC mismatch");
    if (fgetc(file) != EOF) FAIL("trailing retriever data");
    fclose(file);
    return r;
fail:
    if (file) fclose(file);
    metis_memory_retriever_free(r);
    return NULL;
#undef FAIL
}

void metis_memory_retriever_free(metis_memory_retriever_t *r) {
    if (!r) return;
    free(r->query_global);
    free(r->entry_global);
    free(r->query_token);
    free(r->entry_token);
    free(r);
}

static void project_normalized(
    const float *weight, const float *input, int hidden, int rank,
    float *output) {
    float norm = 0.0f;
    for (int row = 0; row < rank; ++row) {
        float value = 0.0f;
        const float *w = weight + (size_t)row * (size_t)hidden;
        for (int col = 0; col < hidden; ++col)
            value += w[col] * input[col];
        output[row] = value;
        norm += value * value;
    }
    norm = 1.0f / sqrtf(norm + 1e-12f);
    for (int row = 0; row < rank; ++row) output[row] *= norm;
}

int metis_memory_retriever_encode(
    const metis_memory_retriever_t *r, const float *hidden, size_t tokens,
    int is_query, float *global_key, float *token_keys) {
    float *pooled;
    const float *global_w, *token_w;
    if (!r || !hidden || !tokens || !global_key || !token_keys) return -1;
    pooled = calloc((size_t)r->hidden_dim, sizeof(float));
    if (!pooled) return -1;
    for (size_t token = 0; token < tokens; ++token)
        for (int col = 0; col < r->hidden_dim; ++col)
            pooled[col] += hidden[token * (size_t)r->hidden_dim + col] /
                           (float)tokens;
    for (int col = 0; col < r->hidden_dim; ++col)
        pooled[col] += hidden[
            (tokens - 1) * (size_t)r->hidden_dim + col];
    global_w = is_query ? r->query_global : r->entry_global;
    token_w = is_query ? r->query_token : r->entry_token;
    project_normalized(
        global_w, pooled, r->hidden_dim, r->rank, global_key);
    for (size_t token = 0; token < tokens; ++token)
        project_normalized(
            token_w, hidden + token * (size_t)r->hidden_dim,
            r->hidden_dim, r->rank,
            token_keys + token * (size_t)r->rank);
    free(pooled);
    return 0;
}

float metis_memory_retriever_score(
    const metis_memory_retriever_t *r,
    const float *qg, const float *qt, size_t nq,
    const float *eg, const float *et, size_t ne) {
    float global = 0.0f, late = 0.0f;
    if (!r || !qg || !qt || !nq || !eg || !et || !ne) return -INFINITY;
    for (int k = 0; k < r->rank; ++k) global += qg[k] * eg[k];
    for (size_t i = 0; i < nq; ++i) {
        float best = -INFINITY;
        for (size_t j = 0; j < ne; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < r->rank; ++k)
                dot += qt[i * (size_t)r->rank + k] *
                       et[j * (size_t)r->rank + k];
            if (dot > best) best = dot;
        }
        late += best;
    }
    return global + r->late_weight * late / (float)nq;
}

float metis_memory_retriever_residual(
    const metis_memory_retriever_t *r, float score) {
    return r ? r->max_residual * tanhf(score) : 0.0f;
}
