#include "bitnet.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int tokens[64];
    char buf[128];
    int n_tokens = 0;
    int next_token = 0;
    int rc = 0;
    int embedding_length = 0;
    float *raw_embeddings = NULL;
    const int band_start[4] = {0, 7, 14, 21};
    const int band_end[4] = {6, 13, 20, 27};
    const float *band_hidden = NULL;

    model = bitnet_load_model(model_path);
    if (model == NULL) { fprintf(stderr, "FAIL load\n"); return 1; }

    ctx = bitnet_create_context(model, 32);
    if (ctx == NULL) { fprintf(stderr, "FAIL ctx\n"); return 2; }

    n_tokens = bitnet_tokenize(model, "hello world", tokens, 64);
    if (n_tokens <= 0) { fprintf(stderr, "FAIL tokenize\n"); return 3; }
    embedding_length = bitnet_embedding_length(model);
    if (embedding_length <= 0 ||
        (size_t)n_tokens > SIZE_MAX / (size_t)embedding_length) {
        fprintf(stderr, "FAIL embedding geometry\n");
        return 4;
    }
    raw_embeddings = (float *)malloc(
        (size_t)n_tokens * (size_t)embedding_length *
        sizeof(float));
    if (raw_embeddings == NULL ||
        bitnet_token_embedding_lookup(
            model, tokens, n_tokens, raw_embeddings,
            (size_t)n_tokens *
                (size_t)embedding_length) != 0 ||
        !isfinite(raw_embeddings[0]) ||
        bitnet_token_embedding_lookup(
            model, tokens, n_tokens, raw_embeddings,
            (size_t)n_tokens *
                (size_t)embedding_length - 1u) == 0) {
        fprintf(stderr, "FAIL raw embedding lookup\n");
        return 5;
    }

    rc = bitnet_context_configure_layer_bands(
        ctx, band_start, band_end, 4);
    if (rc != 0) {
        fprintf(stderr, "FAIL layer bands: %d\n", rc);
        return 6;
    }
    rc = bitnet_eval(ctx, tokens, n_tokens);
    if (rc != 0) { fprintf(stderr, "FAIL eval: %d\n", rc); return 7; }
    band_hidden = bitnet_get_last_eval_layer_bands(ctx);
    if (band_hidden == NULL ||
        bitnet_last_eval_layer_band_token_count(ctx) != n_tokens ||
        bitnet_layer_band_count(ctx) != 4 ||
        !isfinite(band_hidden[0])) {
        fprintf(stderr, "FAIL layer band capture\n");
        return 8;
    }

    next_token = bitnet_sample_greedy(ctx);
    if (next_token < 0) { fprintf(stderr, "FAIL sample\n"); return 9; }

    rc = bitnet_decode_token(model, next_token, buf, sizeof(buf));
    if (rc <= 0) { fprintf(stderr, "FAIL decode\n"); return 10; }

    free(raw_embeddings);
    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
