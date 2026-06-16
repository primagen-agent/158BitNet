#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GENERATE 4
#define MIN_CONTEXT_TOKENS 128
#define DEFAULT_MAX_CONTEXT_TOKENS 2048
#define DEFAULT_REPEAT_LAST_N 64
#define DEFAULT_REPEAT_PENALTY 1.1f

static int env_int_or_default(const char *name, int fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    long parsed = 0;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    parsed = strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > 32768) {
        return fallback;
    }
    return (int)parsed;
}

static float env_float_or_default(const char *name, float fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    float parsed = 0.0f;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    parsed = strtof(value, &end);
    if (end == value || parsed < 1.0f || parsed > 10.0f) {
        return fallback;
    }
    return parsed;
}

int main(int argc, char **argv) {
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int *prompt_tokens = NULL;
    int *history_tokens = NULL;
    int n_prompt_tokens = 0;
    int history_count = 0;
    int context_tokens = 0;
    int max_context_tokens = env_int_or_default("BITNET_MAX_CONTEXT", DEFAULT_MAX_CONTEXT_TOKENS);
    int repeat_last_n = env_int_or_default("BITNET_REPEAT_LAST_N", DEFAULT_REPEAT_LAST_N);
    float repeat_penalty = env_float_or_default("BITNET_REPEAT_PENALTY", DEFAULT_REPEAT_PENALTY);
    char decoded[256];
    int i = 0;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt> [num_tokens]\n", argv[0]);
        return 1;
    }

    int num_tokens = MAX_GENERATE;
    if (argc >= 4) {
        num_tokens = atoi(argv[3]);
        if (num_tokens <= 0) num_tokens = MAX_GENERATE;
    }

    model = bitnet_load_model(argv[1]);
    if (model == NULL) {
        fprintf(stderr, "Failed to load model from: %s\n", argv[1]);
        return 1;
    }

    if (max_context_tokens < MIN_CONTEXT_TOKENS) {
        max_context_tokens = MIN_CONTEXT_TOKENS;
    }

    prompt_tokens = (int *)calloc((size_t)max_context_tokens, sizeof(*prompt_tokens));
    if (prompt_tokens == NULL) {
        fprintf(stderr, "Failed to allocate prompt buffer\n");
        bitnet_free_model(model);
        return 1;
    }

    n_prompt_tokens = bitnet_tokenize(model, argv[2], prompt_tokens, max_context_tokens);
    if (n_prompt_tokens <= 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }
    if (n_prompt_tokens >= max_context_tokens - 1) {
        fprintf(stderr, "Prompt is too long for BITNET_MAX_CONTEXT=%d\n", max_context_tokens);
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }

    if (n_prompt_tokens + num_tokens + 1 > max_context_tokens) {
        int clamped = max_context_tokens - n_prompt_tokens - 1;
        if (clamped <= 0) {
            fprintf(stderr, "No room left for generation in context\n");
            free(prompt_tokens);
            bitnet_free_model(model);
            return 1;
        }
        fprintf(stderr, "Clamping generation from %d to %d tokens for context size %d\n",
                num_tokens, clamped, max_context_tokens);
        num_tokens = clamped;
    }

    context_tokens = n_prompt_tokens + num_tokens + 1;
    if (context_tokens < MIN_CONTEXT_TOKENS) {
        context_tokens = MIN_CONTEXT_TOKENS;
    }

    ctx = bitnet_create_context(model, context_tokens);
    if (ctx == NULL) {
        fprintf(stderr, "Failed to create context\n");
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }

    history_tokens = (int *)calloc((size_t)context_tokens, sizeof(*history_tokens));
    if (history_tokens == NULL) {
        fprintf(stderr, "Failed to allocate generation history\n");
        bitnet_free_context(ctx);
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }
    memcpy(history_tokens, prompt_tokens, (size_t)n_prompt_tokens * sizeof(*history_tokens));
    history_count = n_prompt_tokens;

    printf("Prompt tokens (%d):", n_prompt_tokens);
    for (i = 0; i < n_prompt_tokens; ++i) {
        printf(" %d", prompt_tokens[i]);
    }
    printf("\n");

    if (bitnet_eval(ctx, prompt_tokens, n_prompt_tokens) != 0) {
        fprintf(stderr, "Eval failed\n");
        free(history_tokens);
        free(prompt_tokens);
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    for (i = 0; i < num_tokens; ++i) {
        int recent_count = history_count < repeat_last_n ? history_count : repeat_last_n;
        const int *recent = history_tokens + history_count - recent_count;
        int next_token = bitnet_sample_greedy_repetition_penalty(ctx, recent, recent_count, repeat_penalty);
        if (next_token < 0) {
            fprintf(stderr, "Sample failed\n");
            break;
        }
        if (bitnet_token_is_eog(model, next_token)) {
            break;
        }

        if (bitnet_decode_token(model, next_token, decoded, (int)sizeof(decoded)) > 0) {
            printf("%s", decoded);
            fflush(stdout);
        }

        if (history_count < context_tokens) {
            history_tokens[history_count++] = next_token;
        }
        if (bitnet_eval(ctx, &next_token, 1) != 0) {
            fprintf(stderr, "Eval failed for token %d\n", next_token);
            break;
        }
    }

    printf("\n");

    free(history_tokens);
    free(prompt_tokens);
    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
