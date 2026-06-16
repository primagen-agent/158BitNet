#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

#define MAX_CONTEXT_TOKENS 256
#define MAX_TURN_TOKENS 64
#define MAX_TOTAL_TOKENS 256

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static int append_turn(bitnet_model_t *model,
                       const char *text,
                       int add_bos,
                       int *turn_tokens,
                       int *turn_counts,
                       int turn_idx,
                       int *all_tokens,
                       int *total_tokens) {
    int *dst = turn_tokens + (size_t)turn_idx * MAX_TURN_TOKENS;
    int n = bitnet_tokenize_ex(model, text, dst, MAX_TURN_TOKENS, add_bos);

    if (n <= 0 || *total_tokens + n > MAX_TOTAL_TOKENS) {
        return -1;
    }

    turn_counts[turn_idx] = n;
    memcpy(all_tokens + *total_tokens, dst, (size_t)n * sizeof(*dst));
    *total_tokens += n;
    return 0;
}

static int create_context(bitnet_model_t *model, int kv_type, bitnet_context_t **out) {
    bitnet_context_t *ctx = bitnet_create_context(model, MAX_CONTEXT_TOKENS);
    if (ctx == NULL) {
        return -1;
    }
    if (kv_type != BITNET_KV_CACHE_F32 && bitnet_set_kv_cache_type(ctx, kv_type) != 0) {
        bitnet_free_context(ctx);
        return -1;
    }
    *out = ctx;
    return 0;
}

int main(int argc, char **argv) {
    static const char default_model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    static const char *turns[] = {
        "You are a concise assistant. Remember that my project is BitNet inference.",
        "User: What is the main bottleneck in decode?",
        "Assistant: Matrix-vector projections dominate short decode latency.",
        "User: Now keep the context and answer a follow-up.",
        "Assistant: Reusing the KV cache avoids recomputing the previous turns.",
        "User: Summarize the optimization strategy in one sentence."
    };
    const int n_turns = (int)(sizeof(turns) / sizeof(turns[0]));
    const char *model_path = default_model_path;
    bitnet_model_t *model = NULL;
    bitnet_context_t *recompute_ctx = NULL;
    bitnet_context_t *reuse_ctx = NULL;
    int turn_tokens[6 * MAX_TURN_TOKENS];
    int turn_counts[6];
    int all_tokens[MAX_TOTAL_TOKENS];
    int total_tokens = 0;
    int prefix_tokens = 0;
    int kv_type = BITNET_KV_CACHE_F32;
    int recompute_token = -1;
    int reuse_token = -1;
    double recompute_sec = 0.0;
    double reuse_sec = 0.0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "q8") == 0) {
            kv_type = BITNET_KV_CACHE_Q8;
        } else {
            model_path = argv[i];
        }
    }

    model = bitnet_load_model(model_path);
    if (model == NULL) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    if (create_context(model, kv_type, &recompute_ctx) != 0 ||
        create_context(model, kv_type, &reuse_ctx) != 0) {
        fprintf(stderr, "context failed\n");
        bitnet_free_context(recompute_ctx);
        bitnet_free_context(reuse_ctx);
        bitnet_free_model(model);
        return 1;
    }

    for (int i = 0; i < n_turns; ++i) {
        if (append_turn(model, turns[i], i == 0, turn_tokens, turn_counts,
                        i, all_tokens, &total_tokens) != 0) {
            fprintf(stderr, "tokenize failed\n");
            bitnet_free_context(recompute_ctx);
            bitnet_free_context(reuse_ctx);
            bitnet_free_model(model);
            return 1;
        }
    }

    for (int i = 0; i < n_turns; ++i) {
        struct timespec t0, t1;
        prefix_tokens += turn_counts[i];
        if (bitnet_reset_context(recompute_ctx) != 0) {
            fprintf(stderr, "reset failed\n");
            return 1;
        }
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (bitnet_eval(recompute_ctx, all_tokens, prefix_tokens) != 0) {
            fprintf(stderr, "recompute eval failed\n");
            return 1;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        recompute_sec += elapsed_sec(t0, t1);
    }
    recompute_token = bitnet_sample_greedy(recompute_ctx);

    prefix_tokens = 0;
    for (int i = 0; i < n_turns; ++i) {
        struct timespec t0, t1;
        const int *tokens = turn_tokens + (size_t)i * MAX_TURN_TOKENS;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (bitnet_eval(reuse_ctx, tokens, turn_counts[i]) != 0) {
            fprintf(stderr, "reuse eval failed\n");
            return 1;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        reuse_sec += elapsed_sec(t0, t1);
        prefix_tokens += turn_counts[i];
    }
    reuse_token = bitnet_sample_greedy(reuse_ctx);

    printf("kv_cache_type=%s\n", kv_type == BITNET_KV_CACHE_Q8 ? "q8" : "f32");
    printf("kv_cache_bytes=%llu\n", bitnet_kv_cache_bytes(reuse_ctx));
    printf("turns=%d\n", n_turns);
    printf("total_tokens=%d\n", total_tokens);
    printf("recompute_eval_sec=%.6f\n", recompute_sec);
    printf("reuse_eval_sec=%.6f\n", reuse_sec);
    printf("reuse_speedup=%.6f\n", recompute_sec / reuse_sec);
    printf("recompute_next_token=%d\n", recompute_token);
    printf("reuse_next_token=%d\n", reuse_token);

    if (recompute_token != reuse_token) {
        fprintf(stderr, "final token mismatch\n");
        bitnet_free_context(recompute_ctx);
        bitnet_free_context(reuse_ctx);
        bitnet_free_model(model);
        return 1;
    }

    bitnet_free_context(recompute_ctx);
    bitnet_free_context(reuse_ctx);
    bitnet_free_model(model);
    return 0;
}
