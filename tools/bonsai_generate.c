/* Minimal CLI for the Qwen3.5-hybrid Ternary-Bonsai GGUF runtime.
 *
 *   bonsai_generate -m model.gguf -p "prompt" -n 32 [-t threads] [-c ctx]
 *   bonsai_generate -m model.gguf --tokens "prompt"     (token ids only)
 */

#include "bonsai.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Threads for the CPU half of a hybrid decode.
 *
 * A claim-a-row pool finishes a matrix only when its last thread does, so the
 * efficiency cores set the pace: on an Adreno 740 phone five threads, exactly
 * its five performance cores, beat eight by 3-4% with the GPU running and
 * matched eight without it.  The GPU also needs a core for its driver, which
 * using only the performance cores leaves free.
 *
 * Both counts come from the cpufreq topology rather than sysconf, which
 * returns -1 in this cross-built binary; out_total falls back to 8 when the
 * topology cannot be read at all. */
static int default_threads(int *out_total) {
    long freq[64], best = 0;
    int i, n = 0, total = 0;
    char path[80];

    for (i = 0; i < 64; ++i) {
        FILE *f;
        long v = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        f = fopen(path, "r");
        if (f == NULL) break;
        if (fscanf(f, "%ld", &v) != 1) v = 0;
        fclose(f);
        if (v <= 0) continue;
        freq[total++] = v;
        if (v > best) best = v;
    }
    if (total <= 0) {
        if (out_total != NULL) *out_total = 8;
        return 8;
    }
    if (out_total != NULL) *out_total = total;
    /* within 20% of the fastest core counts as a performance core */
    for (i = 0; i < total; ++i)
        if (freq[i] * 5 >= best * 4) ++n;
    return n > 0 ? n : total;
}

static void usage(void) {
    fprintf(stderr,
            "usage: bonsai_generate -m MODEL [-p PROMPT] [-n TOKENS] [-t THREADS] [-c CTX]\n"
            "       bonsai_generate -m MODEL --tokens TEXT\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "The capital of France is";
    int max_tokens = 32;
    int n_threads = 0;   /* 0 picks a default once the backend is known */
    int ctx_size = 2048;
    int tokens_only = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(a, "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(a, "-n") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(a, "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(a, "-c") == 0 && i + 1 < argc) ctx_size = atoi(argv[++i]);
        else if (strcmp(a, "--tokens") == 0 && i + 1 < argc) { tokens_only = 1; prompt = argv[++i]; }
        else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else { usage(); return 2; }
    }
    if (model_path == NULL) {
        usage();
        return 2;
    }
    if (n_threads <= 0) {
        const char *gpu = getenv("BQ_GPU");
        int gpu_on = (gpu == NULL || gpu[0] != '0');
        int total = 8;
        int per = default_threads(&total);
        n_threads = gpu_on ? per : total;
    }

    double t0 = now_sec();
    bonsai_model_t *model = bonsai_load_model(model_path, n_threads);
    if (model == NULL) return 1;
    fprintf(stderr, "loaded in %.1fs: layers=%d embd=%d vocab=%d\n",
            now_sec() - t0, bonsai_layer_count(model),
            bonsai_embedding_length(model), bonsai_vocab_size(model));

    {
        int cap = (int)strlen(prompt) * 4 + 8;
        int *ids = (int *)malloc((size_t)cap * sizeof(int));
        int n = bonsai_tokenize(model, prompt, ids, cap);
        if (n < 0) {
            fprintf(stderr, "tokenize failed\n");
            free(ids);
            bonsai_free_model(model);
            return 1;
        }
        if (tokens_only) {
            for (i = 0; i < n; ++i) printf("%d%c", ids[i], i + 1 == n ? '\n' : ' ');
            free(ids);
            bonsai_free_model(model);
            return 0;
        }

        bonsai_context_t *ctx = bonsai_create_context(model, ctx_size);
        if (ctx == NULL) {
            fprintf(stderr, "cannot create context\n");
            free(ids);
            bonsai_free_model(model);
            return 1;
        }

        double t1 = now_sec();
        if (bonsai_eval(ctx, ids, n) != 0) {
            fprintf(stderr, "eval failed\n");
            free(ids);
            bonsai_free_context(ctx);
            bonsai_free_model(model);
            return 1;
        }
        double t2 = now_sec();
        fprintf(stderr, "prompt: %d tokens in %.2fs (%.2f s/tok)\n", n, t2 - t1, (t2 - t1) / (n > 0 ? n : 1));
        printf("%s", prompt);
        fflush(stdout);

        for (i = 0; i < max_tokens; ++i) {
            int tok = bonsai_sample_greedy(ctx);
            char buf[512];
            int len;
            if (bonsai_token_is_eog(model, tok)) {
                fprintf(stderr, "\n[eog]\n");
                break;
            }
            len = bonsai_decode_token(model, tok, buf, (int)sizeof(buf));
            if (len > 0) {
                fwrite(buf, 1, (size_t)len, stdout);
                fflush(stdout);
            }
            {
                double ta = now_sec();
                if (bonsai_eval(ctx, &tok, 1) != 0) {
                    fprintf(stderr, "\neval failed\n");
                    break;
                }
                fprintf(stderr, "[tok %d: %.2fs] ", tok, now_sec() - ta);
            }
        }
        printf("\n");
        bonsai_profile_dump();
        {
            double t3 = now_sec();
            fprintf(stderr, "generate: pos=%d\n", bonsai_context_pos(ctx));
            (void)t3;
        }

        free(ids);
        bonsai_free_context(ctx);
    }

    bonsai_free_model(model);
    return 0;
}
