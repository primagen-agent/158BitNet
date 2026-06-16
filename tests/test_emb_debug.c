#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *model_path = "models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;

    if (argc > 1) model_path = argv[1];

    model = bitnet_load_model(model_path);
    if (model == NULL) { fprintf(stderr, "Failed to load model\n"); return 1; }

    ctx = bitnet_create_context(model, 256);

    /* Eval BOS token and check logits */
    {
        int bos = 1;
        int ret = bitnet_eval(ctx, &bos, 1);
        if (ret != 0) { fprintf(stderr, "Eval failed\n"); return 1; }

        const float *logits = bitnet_get_logits(ctx);
        float max_l = -1e30f, min_l = 1e30f, sum_abs = 0.0f;
        for (int j = 0; j < 73448; ++j) {
            if (logits[j] > max_l) max_l = logits[j];
            if (logits[j] < min_l) min_l = logits[j];
            sum_abs += fabsf(logits[j]);
        }
        printf("After BOS logits: min=%.4f max=%.4f mean_abs=%.4f\n", min_l, max_l, sum_abs / 73448);
    }

    /* Eval full prompt */
    bitnet_free_context(ctx);
    ctx = bitnet_create_context(model, 256);

    {
        int tokens[16];
        int n = bitnet_tokenize(model, "The capital of France is", tokens, 16);
        printf("Tokens (%d):", n);
        for (int i = 0; i < n; ++i) {
            char dec[64];
            bitnet_decode_token(model, tokens[i], dec, sizeof(dec));
            printf(" %d[%s]", tokens[i], dec);
        }
        printf("\n");

        int ret = bitnet_eval(ctx, tokens, n);
        if (ret != 0) { fprintf(stderr, "Full eval failed\n"); return 1; }

        const float *logits = bitnet_get_logits(ctx);
        float max_l = -1e30f, min_l = 1e30f, sum_abs = 0.0f;
        for (int j = 0; j < 73448; ++j) {
            if (logits[j] > max_l) max_l = logits[j];
            if (logits[j] < min_l) min_l = logits[j];
            sum_abs += fabsf(logits[j]);
        }
        printf("After prompt logits: min=%.4f max=%.4f mean_abs=%.4f\n", min_l, max_l, sum_abs / 73448);

        /* Top-5 */
        typedef struct { int id; float val; } iv_t;
        iv_t *items = (iv_t *)calloc(73448, sizeof(iv_t));
        for (int j = 0; j < 73448; ++j) { items[j].id = j; items[j].val = logits[j]; }
        /* Simple selection sort for top 5 */
        for (int k = 0; k < 5; ++k) {
            int best = k;
            for (int j = k+1; j < 73448; ++j) {
                if (items[j].val > items[best].val) best = j;
            }
            iv_t tmp = items[k]; items[k] = items[best]; items[best] = tmp;
            char dec[64];
            bitnet_decode_token(model, items[k].id, dec, sizeof(dec));
            printf("  Top %d: token %d [%s] logit=%.4f\n", k+1, items[k].id, dec, items[k].val);
        }
        free(items);
    }

    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
