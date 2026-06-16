#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

#define MAX_PROMPT_TOKENS 128
#define GENERATE_COUNT 20
#define MAX_DECODE_SIZE 256

typedef struct {
    const char *prompt;
    const char *description;
} test_prompt_t;

static const test_prompt_t PROMPTS[] = {
    {"The capital of France is", "English: simple completion"},
    {"Once upon a time in a distant land, there lived a", "English: story opening"},
    {"人工智能是",                            "Chinese: AI topic"},
};

static double elapsed_sec(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static int run_benchmark(const char *model_path, const test_prompt_t *tp, int prompt_idx) {
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int prompt_tokens[MAX_PROMPT_TOKENS];
    int n_prompt_tokens = 0;
    char decoded[MAX_DECODE_SIZE];
    struct timespec t_start, t_end, t_load_end, t_prefill_end;
    double load_time = 0.0, prefill_time = 0.0;
    double gen_times[GENERATE_COUNT];
    int i = 0;
    int total_generated = 0;

    printf("\n============================================================\n");
    printf("[Test %d] %s\n", prompt_idx, tp->description);
    printf("Prompt: \"%s\"\n", tp->prompt);
    printf("------------------------------------------------------------\n");

    clock_gettime(CLOCK_MONOTONIC, &t_start);

    model = bitnet_load_model(model_path);
    if (model == NULL) {
        printf("ERROR: Failed to load model\n");
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_load_end);
    load_time = elapsed_sec(t_start, t_load_end);
    printf("Model load time: %.2f s\n", load_time);

    ctx = bitnet_create_context(model, MAX_PROMPT_TOKENS + GENERATE_COUNT);
    if (ctx == NULL) {
        printf("ERROR: Failed to create context\n");
        bitnet_free_model(model);
        return -1;
    }

    n_prompt_tokens = bitnet_tokenize(model, tp->prompt, prompt_tokens, MAX_PROMPT_TOKENS);
    if (n_prompt_tokens <= 0) {
        printf("ERROR: Failed to tokenize prompt\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return -1;
    }
    printf("Prompt tokens: %d\n", n_prompt_tokens);

    if (bitnet_eval(ctx, prompt_tokens, n_prompt_tokens) != 0) {
        printf("ERROR: Prefill eval failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_prefill_end);
    prefill_time = elapsed_sec(t_load_end, t_prefill_end);
    printf("Prefill time: %.2f s  |  Processing speed: %.2f tokens/s\n",
           prefill_time, (double)n_prompt_tokens / prefill_time);

    printf("\nGenerated output: \"");
    fflush(stdout);

    for (i = 0; i < GENERATE_COUNT; ++i) {
        int next_token;
        struct timespec gen_start, gen_end;

        clock_gettime(CLOCK_MONOTONIC, &gen_start);

        next_token = bitnet_sample_greedy(ctx);
        if (next_token < 0) {
            printf("\"\nERROR: Sample failed at token %d\n", i);
            break;
        }

        if (bitnet_decode_token(model, next_token, decoded, MAX_DECODE_SIZE) > 0) {
            printf("%s", decoded);
            fflush(stdout);
            total_generated++;
        }

        if (i < GENERATE_COUNT - 1) {
            if (bitnet_eval(ctx, &next_token, 1) != 0) {
                printf("\"\nERROR: Eval failed for token %d\n", next_token);
                break;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &gen_end);
        gen_times[i] = elapsed_sec(gen_start, gen_end);
    }

    printf("\"\n\nPer-token generation times:\n");
    for (i = 0; i < total_generated; ++i) {
        printf("  Token %d: %.2f s (%.2f tokens/s)\n", i + 1, gen_times[i], 1.0 / gen_times[i]);
    }

    if (total_generated > 0) {
        double avg_sec = 0.0;
        for (i = 0; i < total_generated; ++i) avg_sec += gen_times[i];
        avg_sec /= (double)total_generated;
        printf("  Average: %.2f s/token  |  Generation speed: %.2f tokens/s\n",
               avg_sec, 1.0 / avg_sec);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    printf("Total test time: %.2f s\n", elapsed_sec(t_start, t_end));

    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    int num_prompts = (int)(sizeof(PROMPTS) / sizeof(PROMPTS[0]));
    int failed = 0;
    int i = 0;

    printf("============================================================\n");
    printf("  BitNet 1.58B C Inference Benchmark\n");
    printf("  Model: bitcpm4-1b-tq2_0.gguf\n");
    printf("  Generate tokens per prompt: %d\n", GENERATE_COUNT);
    printf("============================================================\n");

    for (i = 0; i < num_prompts; ++i) {
        if (run_benchmark(model_path, &PROMPTS[i], i + 1) != 0) {
            failed++;
        }
    }

    printf("\n============================================================\n");
    printf("  Summary: %d/%d prompts completed successfully\n",
           num_prompts - failed, num_prompts);
    if (failed > 0) {
        printf("  WARNING: %d prompt(s) failed\n", failed);
    }
    printf("============================================================\n");

    return failed > 0 ? 1 : 0;
}
