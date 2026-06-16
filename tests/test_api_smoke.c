#include "bitnet.h"

#include <stdio.h>

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

    model = bitnet_load_model(model_path);
    if (model == NULL) { fprintf(stderr, "FAIL load\n"); return 1; }

    ctx = bitnet_create_context(model, 32);
    if (ctx == NULL) { fprintf(stderr, "FAIL ctx\n"); return 2; }

    n_tokens = bitnet_tokenize(model, "hello world", tokens, 64);
    if (n_tokens <= 0) { fprintf(stderr, "FAIL tokenize\n"); return 3; }

    rc = bitnet_eval(ctx, tokens, n_tokens);
    if (rc != 0) { fprintf(stderr, "FAIL eval: %d\n", rc); return 4; }

    next_token = bitnet_sample_greedy(ctx);
    if (next_token < 0) { fprintf(stderr, "FAIL sample\n"); return 5; }

    rc = bitnet_decode_token(model, next_token, buf, sizeof(buf));
    if (rc <= 0) { fprintf(stderr, "FAIL decode\n"); return 6; }

    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
