#include "bitnet.h"

#include <stdio.h>
#include <time.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    static const char prompt[] = "The capital of France is";
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int tokens[64];
    int n_tokens = 0;
    int next_token = 0;
    struct timespec t0, t1, t2, t3;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    model = bitnet_load_model(model_path);
    if (model == NULL) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    ctx = bitnet_create_context(model, 128);
    if (ctx == NULL) {
        fprintf(stderr, "context failed\n");
        bitnet_free_model(model);
        return 1;
    }

    n_tokens = bitnet_tokenize(model, prompt, tokens, 64);
    if (n_tokens <= 0) {
        fprintf(stderr, "tokenize failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    if (bitnet_eval(ctx, tokens, n_tokens) != 0) {
        fprintf(stderr, "prefill failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);

    next_token = bitnet_sample_greedy(ctx);
    if (next_token < 0 || bitnet_eval(ctx, &next_token, 1) != 0) {
        fprintf(stderr, "decode failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t3);

    printf("load_sec=%.6f\n", elapsed_sec(t0, t1));
    printf("prefill_tokens=%d\n", n_tokens);
    printf("prefill_sec=%.6f\n", elapsed_sec(t1, t2));
    printf("prefill_tok_s=%.6f\n", (double)n_tokens / elapsed_sec(t1, t2));
    printf("decode_sec=%.6f\n", elapsed_sec(t2, t3));
    printf("decode_tok_s=%.6f\n", 1.0 / elapsed_sec(t2, t3));

    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
