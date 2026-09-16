#include "bitnet.h"
#include "metis/memory_controller.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}
static int f32(FILE *f, float *v) {
    uint32_t b;
    if (u32(f, &b)) return -1;
    memcpy(v, &b, 4);
    return !isfinite(*v);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s GATE.bnctrl MODEL.gguf VALID.parity\n", argv[0]); return 2; }
    int status = 1, math_errors = 0, disagreements = 0, false_writes = 0, positives = 0, accepted = 0, negatives = 0;
    FILE *file = fopen(argv[3], "rb");
    bitnet_model_t *model = NULL;
    metis_memory_controller_t *gate = NULL;
    float *hidden = NULL;
    unsigned char magic[8];
    uint32_t width, count;
    char error[256];
    if (!file || fread(magic, 1, 8, file) != 8 || memcmp(magic, "BGATEP01", 8) ||
        u32(file, &width) || u32(file, &count) || !width || width > 65536 || !count || count > 100000) goto done;
    model = bitnet_load_model(argv[2]);
    if (!model || bitnet_embedding_length(model) != (int)width) goto done;
    gate = metis_memory_controller_load(argv[1], argv[2], (int)width, error, sizeof error);
    if (!gate) { fprintf(stderr, "%s\n", error); goto done; }
    hidden = malloc(width * sizeof(float));
    if (!hidden) goto done;
    for (uint32_t row = 0; row < count; ++row) {
        uint32_t n, label, expected, token;
        int tokens[512]; float margin, actual_margin;
        if (u32(file, &n) || !n || n > 512 || u32(file, &label) || label > 1 ||
            u32(file, &expected) || expected > 1 || f32(file, &margin)) goto done;
        for (uint32_t i = 0; i < n; ++i) {
            if (u32(file, &token) || token >= (uint32_t)bitnet_vocab_size(model)) goto done;
            tokens[i] = (int)token;
        }
        for (uint32_t i = 0; i < width; ++i) if (f32(file, &hidden[i])) goto done;
        int action = metis_memory_controller_classify(gate, hidden, &actual_margin);
        math_errors += action != (int)expected || fabsf(actual_margin - margin) > 1e-3f;
        bitnet_context_t *context = bitnet_create_context(model, n < 128 ? 128 : (int)n + 1);
        if (!context) goto done;
        if (bitnet_eval_hidden(context, tokens, (int)n) != 0) { bitnet_free_context(context); goto done; }
        action = metis_memory_controller_classify(gate, gate->pooling == 2 ?
            bitnet_get_last_pooled_hidden(context) : bitnet_get_last_hidden(context), &actual_margin);
        bitnet_free_context(context);
        if (action < 0 || action > 1) goto done;
        disagreements += action != (int)expected;
        positives += label == 1; negatives += label == 0;
        accepted += label == 1 && action == 1; false_writes += label == 0 && action == 1;
        printf("{\"index\":%u,\"label\":%u,\"python_action\":%u,\"c_action\":%d,\"margin\":%.8g}\n", row, label, expected, action, actual_margin);
        fflush(stdout);
    }
    if (fgetc(file) != EOF) goto done;
    printf("{\"phase\":\"gate_c_validation\",\"total\":%u,\"math_errors\":%d,\"native_disagreements\":%d,\"write_correct\":%d,\"write_total\":%d,\"false_writes\":%d,\"negative_total\":%d}\n",
        count, math_errors, disagreements, accepted, positives, false_writes, negatives);
    status = math_errors || disagreements ? 1 : 0;
done:
    free(hidden);
    metis_memory_controller_free(gate);
    if (model) bitnet_free_model(model);
    if (file) fclose(file);
    return status;
}
