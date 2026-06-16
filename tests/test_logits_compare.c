#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif
#ifndef BITNET_BINARY_DIR
#define BITNET_BINARY_DIR "."
#endif

#define MAX_TOKENS 64

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int tokens[MAX_TOKENS];
    int n_tokens = 0;
    const float *logits = NULL;
    int vocab_size = 0;
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

    /* Tokenize "The" — should produce [BOS, 1507] */
    n_tokens = bitnet_tokenize(model, "The", tokens, MAX_TOKENS);
    if (n_tokens <= 0) {
        printf("ERROR: Failed to tokenize\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    printf("Tokens (%d):", n_tokens);
    for (i = 0; i < n_tokens; ++i) {
        printf(" %d", tokens[i]);
    }
    printf("\n");

    /* Run forward pass */
    if (bitnet_eval(ctx, tokens, n_tokens) != 0) {
        printf("ERROR: Eval failed\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    logits = bitnet_get_logits(ctx);
    if (logits == NULL) {
        printf("ERROR: No logits\n");
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    /* Find vocab size from model — we know it's 73448 */
    vocab_size = 73448;

    /* Print top-10 tokens */
    printf("\nTop-10 tokens by logit value:\n");
    {
        int selected[10];
        for (int rank = 0; rank < 10; ++rank) {
            int best_idx = -1;
            float best_val = -1e30f;
            for (int j = 0; j < vocab_size; ++j) {
                int skip = 0;
                for (int k = 0; k < rank; ++k) {
                    if (selected[k] == j) { skip = 1; break; }
                }
                if (skip) continue;
                if (logits[j] > best_val) {
                    best_val = logits[j];
                    best_idx = j;
                }
            }
            selected[rank] = best_idx;
            char decoded[256];
            bitnet_decode_token(model, best_idx, decoded, sizeof(decoded));
            printf("  #%d: token=%d logit=%.4f text=%s\n", rank + 1, best_idx, best_val, decoded);
        }
    }

    /* Print specific token logits */
    printf("\nSpecific token logits:\n");
    int specific_ids[] = {0, 1, 1507, 8107};
    for (i = 0; i < 4; ++i) {
        int tid = specific_ids[i];
        if (tid < vocab_size) {
            char decoded[256];
            bitnet_decode_token(model, tid, decoded, sizeof(decoded));
            printf("  Token %d: logit=%.4f text=%s\n", tid, logits[tid], decoded);
        }
    }

    /* Print logit statistics */
    float min_logit = 1e30f, max_logit = -1e30f, sum_logit = 0.0f;
    for (i = 0; i < vocab_size; ++i) {
        if (logits[i] < min_logit) min_logit = logits[i];
        if (logits[i] > max_logit) max_logit = logits[i];
        sum_logit += logits[i];
    }
    printf("\nLogit stats: min=%.4f max=%.4f mean=%.4f\n", min_logit, max_logit, sum_logit / vocab_size);

    /* Save logits to binary file for Python comparison */
    {
        FILE *fp = fopen(BITNET_BINARY_DIR "/our_logits.bin", "wb");
        if (fp != NULL) {
            fwrite(logits, sizeof(float), vocab_size, fp);
            fclose(fp);
            printf("Saved our logits to build/our_logits.bin\n");
        }
    }

    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
