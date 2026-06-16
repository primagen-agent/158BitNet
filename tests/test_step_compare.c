/*
 * Step-by-step token generation with top-5 logit printing.
 * Compares our C engine output with llama-cpp-python reference.
 */

#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

#define MAX_TOKENS 64
#define NUM_GENERATE 10
#define TOP_K 5
#define VOCAB_SIZE 73448

typedef struct {
    int token_id;
    float logit;
} token_logit_t;

static int cmp_logit_desc(const void *a, const void *b) {
    float la = ((const token_logit_t *)a)->logit;
    float lb = ((const token_logit_t *)b)->logit;
    if (lb > la) return 1;
    if (lb < la) return -1;
    return 0;
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int tokens[MAX_TOKENS];
    int n_tokens = 0;
    const float *logits = NULL;
    int i = 0;

    model = bitnet_load_model(model_path);
    if (model == NULL) {
        printf("ERROR: Failed to load model\n");
        return 1;
    }

    ctx = bitnet_create_context(model, MAX_TOKENS);
    if (ctx == NULL) {
        printf("ERROR: Failed to create context\n");
        bitnet_free_model(model);
        return 1;
    }

    /* Tokenize the prompt */
    const char *prompt = "The capital of France is";
    n_tokens = bitnet_tokenize(model, prompt, tokens, MAX_TOKENS);
    if (n_tokens <= 0) {
        printf("ERROR: Failed to tokenize\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    printf("Prompt tokens (%d):", n_tokens);
    for (i = 0; i < n_tokens; ++i) {
        printf(" %d", tokens[i]);
    }
    printf("\n");

    /* Print prompt token texts */
    printf("Prompt token texts:\n");
    for (i = 0; i < n_tokens; ++i) {
        char decoded[256];
        bitnet_decode_token(model, tokens[i], decoded, (int)sizeof(decoded));
        printf("  Token %d: %s\n", tokens[i], decoded);
    }

    /* Run eval on prompt tokens */
    if (bitnet_eval(ctx, tokens, n_tokens) != 0) {
        printf("ERROR: Eval failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    /* Generate tokens step by step */
    int generated_tokens[NUM_GENERATE];
    printf("\n");

    for (int step = 0; step < NUM_GENERATE; ++step) {
        logits = bitnet_get_logits(ctx);
        if (logits == NULL) {
            printf("ERROR: No logits at step %d\n", step);
            break;
        }

        /* Find top-K tokens */
        token_logit_t *all_logits = (token_logit_t *)malloc(VOCAB_SIZE * sizeof(token_logit_t));
        if (all_logits == NULL) {
            printf("ERROR: malloc failed\n");
            break;
        }
        for (int j = 0; j < VOCAB_SIZE; ++j) {
            all_logits[j].token_id = j;
            all_logits[j].logit = logits[j];
        }
        qsort(all_logits, VOCAB_SIZE, sizeof(token_logit_t), cmp_logit_desc);

        printf("=== Step %d ===\n", step);
        for (int k = 0; k < TOP_K; ++k) {
            char decoded[256];
            bitnet_decode_token(model, all_logits[k].token_id, decoded, (int)sizeof(decoded));
            printf("  Token %d: logit=%.6f text=%s\n",
                   all_logits[k].token_id, all_logits[k].logit, decoded);
        }

        int next_token = all_logits[0].token_id;
        char sel_decoded[256];
        bitnet_decode_token(model, next_token, sel_decoded, (int)sizeof(sel_decoded));
        printf("  >> Selected: Token %d = %s\n", next_token, sel_decoded);

        generated_tokens[step] = next_token;
        free(all_logits);

        /* Feed back the selected token */
        if (bitnet_eval(ctx, &next_token, 1) != 0) {
            printf("ERROR: Eval failed for token %d\n", next_token);
            break;
        }
    }

    /* Summary */
    printf("\n=== Summary ===\n");
    printf("Prompt: \"%s\"\n", prompt);
    printf("Generated tokens:");
    for (i = 0; i < NUM_GENERATE; ++i) {
        printf(" %d", generated_tokens[i]);
    }
    printf("\n");
    printf("Generated text: ");
    for (i = 0; i < NUM_GENERATE; ++i) {
        char decoded[256];
        bitnet_decode_token(model, generated_tokens[i], decoded, (int)sizeof(decoded));
        printf("%s", decoded);
    }
    printf("\n");

    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
