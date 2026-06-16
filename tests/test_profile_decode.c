#include "bitnet.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

#define MAX_TOKENS 128
#define DECODE_COUNT 16

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char **argv) {
    static const char default_model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    static const char prompt[] = "The capital of France is";
    const char *model_path = default_model_path;
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int tokens[MAX_TOKENS];
    int n_tokens = 0;
    int next_token = 0;
    int kv_type = BITNET_KV_CACHE_F32;
    struct timespec load_start, load_end, prefill_end;
    double decode_total_sec = 0.0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "q8") == 0) {
            kv_type = BITNET_KV_CACHE_Q8;
        } else {
            model_path = argv[i];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &load_start);
    model = bitnet_load_model(model_path);
    if (model == NULL) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &load_end);

    ctx = bitnet_create_context(model, MAX_TOKENS);
    if (ctx == NULL) {
        fprintf(stderr, "context failed\n");
        bitnet_free_model(model);
        return 1;
    }
    if (kv_type != BITNET_KV_CACHE_F32 && bitnet_set_kv_cache_type(ctx, kv_type) != 0) {
        fprintf(stderr, "kv cache mode failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    n_tokens = bitnet_tokenize(model, prompt, tokens, MAX_TOKENS);
    if (n_tokens <= 0 || bitnet_eval(ctx, tokens, n_tokens) != 0) {
        fprintf(stderr, "prefill failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &prefill_end);

    for (int i = 0; i < DECODE_COUNT; ++i) {
        struct timespec t0, t1;
        next_token = bitnet_sample_greedy(ctx);
        if (next_token < 0) {
            fprintf(stderr, "sample failed\n");
            bitnet_free_context(ctx);
            bitnet_free_model(model);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (bitnet_eval(ctx, &next_token, 1) != 0) {
            fprintf(stderr, "decode failed\n");
            bitnet_free_context(ctx);
            bitnet_free_model(model);
            return 1;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        decode_total_sec += elapsed_sec(t0, t1);
    }

    printf("load_sec=%.6f\n", elapsed_sec(load_start, load_end));
    printf("kv_cache_type=%s\n", kv_type == BITNET_KV_CACHE_Q8 ? "q8" : "f32");
    printf("kv_cache_bytes=%llu\n", bitnet_kv_cache_bytes(ctx));
    printf("prefill_tokens=%d\n", n_tokens);
    printf("prefill_sec=%.6f\n", elapsed_sec(load_end, prefill_end));
    printf("decode_tokens=%d\n", DECODE_COUNT);
    printf("decode_total_sec=%.6f\n", decode_total_sec);
    printf("decode_avg_sec=%.6f\n", decode_total_sec / (double)DECODE_COUNT);
    printf("decode_tok_s=%.6f\n", (double)DECODE_COUNT / decode_total_sec);

    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
