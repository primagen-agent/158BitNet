#include "bitnet.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

static int fail(const char *msg) {
    fprintf(stderr, "FAIL %s\n", msg);
    return 1;
}

static int logits_close(const float *a, const float *b, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(a[i] - b[i]) > 1e-4f) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    bitnet_context_t *fresh = NULL;
    int tokens[64];
    int no_bos_tokens[64];
    int n_tokens = 0;
    int n_no_bos = 0;
    float rewound_logits[16];
    float fresh_logits[16];

    model = bitnet_load_model(model_path);
    if (model == NULL) return fail("load model");

    ctx = bitnet_create_context(model, 12);
    fresh = bitnet_create_context(model, 12);
    if (ctx == NULL || fresh == NULL) return fail("create contexts");

    if (bitnet_context_pos(ctx) != 0) return fail("initial position");
    if (bitnet_context_capacity(ctx) != 12) return fail("capacity");
    if (bitnet_get_kv_cache_type(ctx) != BITNET_KV_CACHE_F32) return fail("default kv type");

    n_tokens = bitnet_tokenize(model, "hello world", tokens, 64);
    n_no_bos = bitnet_tokenize_ex(model, "hello world", no_bos_tokens, 64, 0);
    if (n_tokens <= 1 || n_no_bos <= 0) return fail("tokenize");
    if (n_no_bos != n_tokens - 1) return fail("tokenize without bos length");
    if (memcmp(tokens + 1, no_bos_tokens, (size_t)n_no_bos * sizeof(tokens[0])) != 0) {
        return fail("tokenize without bos payload");
    }

    if (bitnet_eval(ctx, tokens, n_tokens) != 0) return fail("eval prompt");
    if (bitnet_context_pos(ctx) != n_tokens) return fail("position after eval");

    if (bitnet_rewind_context(ctx, 1) != 0) return fail("rewind");
    if (bitnet_context_pos(ctx) != 1) return fail("position after rewind");
    if (bitnet_eval(ctx, tokens + 1, n_tokens - 1) != 0) return fail("eval suffix after rewind");
    memcpy(rewound_logits, bitnet_get_logits(ctx), sizeof(rewound_logits));

    if (bitnet_eval(fresh, tokens, n_tokens) != 0) return fail("fresh eval");
    memcpy(fresh_logits, bitnet_get_logits(fresh), sizeof(fresh_logits));
    if (!logits_close(rewound_logits, fresh_logits, 16)) return fail("rewind logits");

    if (bitnet_reset_context(ctx) != 0) return fail("reset");
    if (bitnet_context_pos(ctx) != 0) return fail("position after reset");
    if (bitnet_eval(ctx, tokens, 13) == 0) return fail("capacity overflow rejected");
    if (bitnet_context_pos(ctx) != 0) return fail("position unchanged after overflow");

    {
        unsigned long long f32_bytes = bitnet_kv_cache_bytes(ctx);
        if (bitnet_set_kv_cache_type(ctx, BITNET_KV_CACHE_Q8) != 0) return fail("set q8 kv");
        if (bitnet_get_kv_cache_type(ctx) != BITNET_KV_CACHE_Q8) return fail("q8 kv type");
        if (bitnet_kv_cache_bytes(ctx) >= f32_bytes) return fail("q8 kv bytes smaller");
        if (bitnet_eval(ctx, tokens, n_tokens) != 0) return fail("q8 eval");
        if (bitnet_sample_greedy(ctx) < 0) return fail("q8 sample");
    }

    bitnet_free_context(ctx);
    bitnet_free_context(fresh);
    bitnet_free_model(model);
    return 0;
}
