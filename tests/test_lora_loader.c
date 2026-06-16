#include "bitnet.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

#ifndef BITNET_BINARY_DIR
#define BITNET_BINARY_DIR "."
#endif

static int write_u32(FILE *f, unsigned int value) {
    return fwrite(&value, sizeof(value), 1, f) == 1 ? 0 : -1;
}

static int write_f32(FILE *f, float value) {
    return fwrite(&value, sizeof(value), 1, f) == 1 ? 0 : -1;
}

static int write_test_lora(const char *path) {
    static const char magic[8] = {'B', 'N', 'L', 'O', 'R', 'A', '1', '\0'};
    static const char persona[] = "你名字叫小丽，说话温暖、俏皮、贴心。";
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;

    if (fwrite(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        write_u32(f, 1) != 0 ||
        write_u32(f, 1) != 0 ||
        write_f32(f, 1.0f) != 0 ||
        write_u32(f, (unsigned int)strlen(persona)) != 0 ||
        fwrite(persona, 1, strlen(persona), f) != strlen(persona)) {
        fclose(f);
        return -1;
    }

    /* block 0 ffn_down: in=6144, out=2048, rank=1 */
    if (write_u32(f, 0) != 0 ||
        write_u32(f, 6) != 0 ||
        write_u32(f, 6144) != 0 ||
        write_u32(f, 2048) != 0 ||
        write_u32(f, 1) != 0 ||
        write_f32(f, 0.35f) != 0) {
        fclose(f);
        return -1;
    }

    for (int i = 0; i < 6144; ++i) {
        float v = (i < 64) ? ((i % 2) == 0 ? 0.004f : -0.003f) : 0.0f;
        if (write_f32(f, v) != 0) {
            fclose(f);
            return -1;
        }
    }
    for (int i = 0; i < 2048; ++i) {
        float v = (i % 17 == 0) ? 0.025f : ((i % 19 == 0) ? -0.018f : 0.0f);
        if (write_f32(f, v) != 0) {
            fclose(f);
            return -1;
        }
    }

    return fclose(f);
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    static const char lora_path[] = BITNET_BINARY_DIR "/test_lora_loader.bnlora";
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int tokens[64];
    float *base_logits = NULL;
    const float *logits = NULL;
    int n_tokens = 0;
    int vocab = 0;
    float max_delta = 0.0f;

    if (write_test_lora(lora_path) != 0) {
        fprintf(stderr, "FAIL write lora\n");
        return 1;
    }

    model = bitnet_load_model(model_path);
    if (model == NULL) {
        fprintf(stderr, "FAIL load model\n");
        return 2;
    }

    if (bitnet_lora_count(model) != 0) {
        fprintf(stderr, "FAIL unexpected lora count\n");
        return 3;
    }

    n_tokens = bitnet_tokenize(model, "hello world", tokens, 64);
    if (n_tokens <= 0) {
        fprintf(stderr, "FAIL tokenize\n");
        return 5;
    }

    ctx = bitnet_create_context(model, 32);
    if (ctx == NULL || bitnet_eval(ctx, tokens, n_tokens) != 0) {
        fprintf(stderr, "FAIL base eval\n");
        return 6;
    }
    vocab = 73448;
    base_logits = (float *)malloc((size_t)vocab * sizeof(*base_logits));
    if (base_logits == NULL) {
        fprintf(stderr, "FAIL alloc logits\n");
        return 7;
    }
    logits = bitnet_get_logits(ctx);
    memcpy(base_logits, logits, (size_t)vocab * sizeof(*base_logits));
    bitnet_free_context(ctx);
    ctx = NULL;

    if (bitnet_load_lora(model, lora_path, 1.0f) != 0) {
        fprintf(stderr, "FAIL load lora\n");
        return 8;
    }
    if (bitnet_lora_count(model) != 1) {
        fprintf(stderr, "FAIL lora count\n");
        return 9;
    }

    ctx = bitnet_create_context(model, 32);
    if (ctx == NULL || bitnet_eval(ctx, tokens, n_tokens) != 0) {
        fprintf(stderr, "FAIL lora eval\n");
        return 11;
    }
    logits = bitnet_get_logits(ctx);
    for (int i = 0; i < vocab; ++i) {
        float d = fabsf(logits[i] - base_logits[i]);
        if (d > max_delta) max_delta = d;
    }
    if (max_delta <= 1e-7f) {
        fprintf(stderr, "FAIL logits unchanged\n");
        return 12;
    }

    free(base_logits);
    bitnet_free_context(ctx);
    bitnet_free_model(model);
    remove(lora_path);
    return 0;
}
