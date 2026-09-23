/* Diagnostic only. Fresh context and one full prefill per input; no history reuse.
 * Local binary protocol: native-endian uint32/int32/float32, little-endian hosts.
 * K/V scratch allocated internally by bitnet_eval is not a persisted memory. */
#include "bitnet.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void normalized_pair(float *out, const float *hidden, const float *lexical, uint32_t width) {
    const float *parts[2] = {hidden, lexical};
    for (int part = 0; part < 2; ++part) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < width; ++i) sum += parts[part][i] * parts[part][i];
        float norm = fmaxf(sqrtf(sum), 1e-12f);
        for (uint32_t i = 0; i < width; ++i) out[(size_t)part * width + i] = parts[part][i] / norm;
    }
}

static int write_block(FILE *f, const void *p, size_t size, size_t count) {
    return p && fwrite(p, size, count, f) == count ? 0 : -1;
}

int main(int argc, char **argv) {
    FILE *in = NULL, *out = NULL;
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    char *text = NULL;
    float *lex = NULL, *features = NULL;
    uint32_t rows, bytes, hidden, n, layers;
    int ids[512], starts[128], ends[128], rc = 1;
    char *end = NULL;
    long requested;
    uint32_t endian = 1;
    int reader_format = argc == 6 && strcmp(argv[5], "reader-v1") == 0;
    if ((argc != 5 && !reader_format) || sizeof(float) != 4 || sizeof(int) != 4 || *(unsigned char *)&endian != 1) return 2;
    requested = strtol(argv[4], &end, 10);
    if (*end || requested < 1 || requested > 128) return 2;
    layers = (uint32_t)requested;
    in = fopen(argv[2], "rb");
    /* Never overwrite a previous diagnostic artifact. */
    out = fopen(argv[3], "wbx");
    if (!in || !out || fread(&rows, 4, 1, in) != 1 || rows < 1 || rows > 1000) goto done;
    model = bitnet_load_model(argv[1]);
    if (!model) goto done;
    hidden = (uint32_t)bitnet_embedding_length(model);
    if (!hidden || write_block(out, reader_format ? "BNEN0001" : "BNFP0001", 1, 8) || write_block(out, &rows, 4, 1) ||
        write_block(out, &hidden, 4, 1) || write_block(out, &layers, 4, 1)) goto done;
    for (uint32_t i = 0; i < layers; ++i) starts[i] = ends[i] = (int)i;
    for (uint32_t row = 0; row < rows; ++row) {
        if (fread(&bytes, 4, 1, in) != 1 || !bytes || bytes > 65536) goto done;
        text = malloc((size_t)bytes + 1);
        if (!text || fread(text, 1, bytes, in) != bytes || memchr(text, 0, bytes)) goto done;
        text[bytes] = 0;
        int tokens = bitnet_tokenize_ex(model, text, ids, 512, 1);
        if (tokens < 2 || tokens >= 512) goto done;
        n = (uint32_t)tokens;
        ctx = bitnet_create_context(model, 512);
        if (!ctx || bitnet_context_pos(ctx) != 0 ||
            bitnet_context_configure_layer_bands(ctx, starts, ends, (int)layers) ||
            bitnet_eval_hidden(ctx, ids, tokens) || bitnet_last_eval_hidden_count(ctx) != tokens ||
            bitnet_last_eval_layer_band_token_count(ctx) != tokens) goto done;
        lex = malloc((size_t)n * hidden * sizeof(float));
        if (!lex || bitnet_token_embedding_lookup(model, ids, tokens, lex, (size_t)n * hidden)) goto done;
        if (write_block(out, &n, 4, 1) || write_block(out, ids, 4, n)) goto done;
        if (reader_format) {
            features = malloc((size_t)(n - 1) * hidden * 2 * sizeof(float));
            if (!features) goto done;
            const float *h = bitnet_get_last_eval_hidden(ctx);
            for (uint32_t t = 1; t < n; ++t)
                normalized_pair(features + (size_t)(t - 1) * hidden * 2, h + (size_t)t * hidden, lex + (size_t)t * hidden, hidden);
            if (write_block(out, features, 4, (size_t)(n - 1) * hidden * 2)) goto done;
            free(features); features = NULL;
        } else if (write_block(out, lex, 4, (size_t)n * hidden) ||
            write_block(out, bitnet_get_last_eval_hidden(ctx), 4, (size_t)n * hidden) ||
            write_block(out, bitnet_get_last_eval_layer_bands(ctx), 4, (size_t)n * layers * hidden)) goto done;
        free(lex); lex = NULL;
        free(text); text = NULL;
        bitnet_free_context(ctx); ctx = NULL;
    }
    if (fgetc(in) != EOF || ferror(in)) goto done;
    rc = 0;
done:
    free(features); free(lex); free(text); bitnet_free_context(ctx); bitnet_free_model(model);
    if (in) fclose(in);
    if (out && fclose(out)) rc = 1;
    if (rc) fprintf(stderr, "memory feature probe failed; partial output must not be consumed\n");
    return rc;
}
