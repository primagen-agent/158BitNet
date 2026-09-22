/* Ordinary, memory-free, fresh-context forward probe for differential tests.
 * Usage: inference_probe model.gguf input output-prefix text|tokens|tokenize
 * Writes token IDs as text and unmodified last-position F32 logits as binary. */
#include "bitnet.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int rc = 1, ids[4096], n = 0;
    char path[4096], text[65536];
    FILE *in = NULL, *out = NULL;
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    bitnet_tokenizer_t *tokenizer = NULL;
    if (argc != 5 || (strcmp(argv[4], "text") && strcmp(argv[4], "tokens") && strcmp(argv[4], "tokenize"))) return 2;
    in = fopen(argv[2], "rb");
    if (!in) goto done;
    int tokens_only = strcmp(argv[4], "tokenize") == 0;
    if (tokens_only) {
        if (bitnet_tokenizer_load(&tokenizer, argv[1])) goto done;
    } else {
        model = bitnet_load_model(argv[1]);
        if (!model) goto done;
    }
    if (!strcmp(argv[4], "tokens")) {
        while (n < 4096 && fscanf(in, "%d", ids + n) == 1) ++n;
        if (!feof(in) || n >= 4096) goto done;
    } else {
        size_t bytes = fread(text, 1, sizeof(text) - 1, in);
        if (ferror(in) || fgetc(in) != EOF || memchr(text, 0, bytes)) goto done;
        text[bytes] = 0;
        if (tokens_only) {
            int bos = bitnet_tokenizer_bos_id(tokenizer);
            if (bos < 0) goto done;
            ids[0] = bos;
            n = bitnet_tokenizer_encode(tokenizer, text, ids + 1, 4095);
            if (n < 0) goto done;
            ++n;
        } else n = bitnet_tokenize_ex(model, text, ids, 4096, 1);
    }
    fclose(in); in = NULL;
    if (n <= 0 || n >= 4096) goto done;
    int len = snprintf(path, sizeof(path), "%s.ids", argv[3]);
    if (len < 0 || len >= (int)sizeof(path) || !(out = fopen(path, "wb"))) goto done;
    for (int i = 0; i < n; ++i) if (fprintf(out, "%d\n", ids[i]) < 0) goto done;
    if (fclose(out)) { out = NULL; goto done; } out = NULL;
    if (tokens_only) { rc = 0; goto done; }
    int vocab = bitnet_vocab_size(model);
    for (int i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= vocab) goto done;
    ctx = bitnet_create_context(model, n + 16);
    if (!ctx || bitnet_context_pos(ctx) != 0 || bitnet_eval(ctx, ids, n)) goto done;
    len = snprintf(path, sizeof(path), "%s.logits", argv[3]);
    if (len < 0 || len >= (int)sizeof(path) || !(out = fopen(path, "wb"))) goto done;
    if (fwrite(bitnet_get_logits(ctx), sizeof(float), (size_t)vocab, out) != (size_t)vocab) goto done;
    if (fclose(out)) { out = NULL; goto done; } out = NULL;
    printf("tokens=%d vocab=%d argmax=%d fresh_context=1\n", n, vocab, bitnet_sample_greedy(ctx));
    rc = 0;
done:
    if (in) fclose(in);
    if (out) fclose(out);
    bitnet_free_context(ctx); bitnet_free_model(model); bitnet_tokenizer_free(tokenizer);
    if (rc) fprintf(stderr, "inference probe failed; do not consume partial output\n");
    return rc;
}
