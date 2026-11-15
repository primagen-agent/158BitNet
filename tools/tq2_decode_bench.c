/* Decode benchmark for quantized GGUF models through the bitnet runtime.
 *
 *   tq2_decode_bench -m model.gguf [-p "prompt"] [-n 32] [-c 512] [-r 1]
 *
 * Loads the model, prefills the prompt, then times `n` greedy decode steps and
 * reports prompt and decode throughput.  Greedy so every run sees the same
 * token stream; `-r` repeats the decode phase on a fresh context and prints
 * each repetition, which is how variance from thermal drift shows up. */

#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "The capital of France is";
    int n_decode = 32, ctx_size = 512, reps = 1, i, r;
    bitnet_model_t *model;
    double t0;

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(a, "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(a, "-n") == 0 && i + 1 < argc) n_decode = atoi(argv[++i]);
        else if (strcmp(a, "-c") == 0 && i + 1 < argc) ctx_size = atoi(argv[++i]);
        else if (strcmp(a, "-r") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s -m MODEL [-p PROMPT] [-n N] [-c CTX] [-r REPS]\n", argv[0]); return 2; }
    }
    if (model_path == NULL) {
        fprintf(stderr, "usage: %s -m MODEL [-p PROMPT] [-n N] [-c CTX] [-r REPS]\n", argv[0]);
        return 2;
    }
    if (ctx_size < n_decode + 64) ctx_size = n_decode + 64;

    t0 = now_sec();
    model = bitnet_load_model(model_path);
    if (model == NULL) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    fprintf(stderr, "load: %.2f s, vocab %d, embd %d\n", now_sec() - t0,
            bitnet_vocab_size(model), bitnet_embedding_length(model));

    for (r = 0; r < reps; ++r) {
        bitnet_context_t *ctx = bitnet_create_context(model, ctx_size);
        int tokens[4096];
        int n, t, n_gen = 0;
        double prefill, decode_total = 0.0;

        if (ctx == NULL) { fprintf(stderr, "ctx failed\n"); return 1; }
        n = bitnet_tokenize(model, (char *)prompt, tokens, 4096);
        if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        t0 = now_sec();
        if (bitnet_eval(ctx, tokens, n) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
        prefill = now_sec() - t0;
        fprintf(stderr, "prompt tokens: %d", n);
        for (t = 0; t < n && t < 16; ++t) fprintf(stderr, " %d", tokens[t]);
        fprintf(stderr, "\n");

        for (t = 0; t < n_decode; ++t) {
            int next = bitnet_sample_greedy(ctx);
            char piece[64];
            int len;
            double s;
            if (next < 0) break;
            if (bitnet_token_is_eog(model, next)) break;
            len = bitnet_decode_token(model, next, piece, (int)sizeof(piece));
            if (len > 0) fwrite(piece, 1, (size_t)len, stderr);
            s = now_sec();
            if (bitnet_eval(ctx, &next, 1) != 0) break;
            decode_total += now_sec() - s;
            ++n_gen;
        }
        fprintf(stderr, "\n");
        printf("rep %d: prompt %d tokens in %.3f s (%.2f tok/s), decode %d tokens in %.3f s (%.2f tok/s, %.1f ms/token)\n",
               r, n, prefill, n > 0 ? n / prefill : 0.0, n_gen, decode_total,
               decode_total > 0 ? n_gen / decode_total : 0.0,
               n_gen > 0 ? 1000.0 * decode_total / n_gen : 0.0);
        fflush(stdout);
        bitnet_free_context(ctx);
    }
    bitnet_free_model(model);
    return 0;
}
