/* Standalone end-to-end benchmark: load, tokenize, prefill/TTFT, decode speed,
 * continuation NLL/PPL on a fixed bilingual panel, and a fixed greedy
 * arithmetic panel. Plain backbone path only; no memory model, no LoRA.
 * Build: cc -O2 tools/bench_e2e.c -I include -I src -o build/bench_e2e \
 *          build/libbitnet.a -lm -lpthread -ldl
 */
#include "bitnet.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CTX_PPL 512
#define CTX_GEN 160
#define SPEED_DECODE_TOKENS 64
#define ARITH_MAX_TOKENS 24

static const char *k_ppl_texts[] = {
    "Ternary language models use weights with three possible values. Each weight can be negative one, "
    "zero, or positive one. This reduces the memory needed to store large models and lets custom chips "
    "compute much faster. Researchers train these networks with methods that keep every weight close to "
    "one of the three allowed values while the loss still goes down.",
    "Lima is the capital of Peru. The city sits near the Pacific coast and has mild weather all year. "
    "Morgan moved there last spring and enjoys the weekend food markets. Her brother still lives in Oslo "
    "and plans to visit next summer with his daughter.",
    "三值语言模型把每个权重限制为三个可能的取值：负一、零或正一。这样的设计大幅减少了存储大模型所需的"
    "内存，也让专用芯片的计算速度快得多。研究人员使用专门的训练方法，让每个权重在损失下降的同时保持在"
    "允许的取值附近。",
    "利马是秘鲁的首都。这座城市靠近太平洋海岸，全年气候温和。摩根去年春天搬到那里，很喜欢周末的食品"
    "市场。她的哥哥仍然住在奥斯陆，计划明年夏天带女儿去看她。",
};
static const int k_ppl_texts_count = (int)(sizeof(k_ppl_texts) / sizeof(k_ppl_texts[0]));

static const char *k_speed_prompt = "The capital of France is";

typedef struct { const char *prompt; int expected; } arith_item_t;
static const arith_item_t k_arith[] = {
    {"What is 3 plus 7?", 10}, {"What is 12 plus 5?", 17}, {"What is 9 minus 4?", 5},
    {"What is 6 times 3?", 18}, {"What is 20 minus 8?", 12},
    {"3加7等于多少？", 10}, {"12加5等于多少？", 17}, {"9减4等于多少？", 5},
    {"6乘3等于多少？", 18}, {"20减8等于多少？", 12},
};
static const int k_arith_count = (int)(sizeof(k_arith) / sizeof(k_arith[0]));

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double logsumexp(const float *x, int n) {
    float m = x[0];
    double s = 0.0;
    int i;
    for (i = 1; i < n; ++i) if (x[i] > m) m = x[i];
    for (i = 0; i < n; ++i) s += exp((double)x[i] - (double)m);
    return (double)m + log(s);
}

static void json_escape(const char *s, int len) {
    int i;
    putchar('"');
    for (i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

/* Extracts the first decimal integer from text; returns found flag. */
static int first_int(const char *s, long *out) {
    const char *p = s;
    while (*p) {
        if (*p >= '0' && *p <= '9') {
            char *end = NULL;
            long v = strtol(p, &end, 10);
            if (end != p) { *out = v; return 1; }
            p++;
        } else {
            p++;
        }
    }
    return 0;
}

static int run_ppl(bitnet_model_t *model, int vocab, int text_index, double *mean_nll, int *n_cont,
                   double *prefill_sec) {
    int tokens[CTX_PPL];
    int n = bitnet_tokenize(model, (char *)k_ppl_texts[text_index], tokens, CTX_PPL);
    int k = n / 3, i;
    double t0, t1, total = 0.0;
    bitnet_context_t *ctx;
    if (n < 24 || k < 8 || n - k < 16) return -1;
    ctx = bitnet_create_context(model, CTX_PPL);
    if (ctx == NULL) return -1;
    t0 = now_sec();
    if (bitnet_eval(ctx, tokens, k) != 0) { bitnet_free_context(ctx); return -1; }
    t1 = now_sec();
    *prefill_sec = t1 - t0;
    for (i = k; i < n; ++i) {
        const float *logits = bitnet_get_logits(ctx);
        double lse;
        if (logits == NULL || tokens[i] < 0 || tokens[i] >= vocab) { bitnet_free_context(ctx); return -1; }
        lse = logsumexp(logits, vocab);
        total += lse - (double)logits[tokens[i]];
        if (i + 1 < n && bitnet_eval(ctx, &tokens[i], 1) != 0) { bitnet_free_context(ctx); return -1; }
    }
    *mean_nll = total / (double)(n - k);
    *n_cont = n - k;
    bitnet_free_context(ctx);
    return 0;
}

int main(int argc, char **argv) {
    bitnet_model_t *model = NULL;
    int vocab = 0, i, t;
    double load_t0, load_sec;
    const char *threads = getenv("BITNET_NUM_THREADS");
    char wrap[512];

    if (argc != 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 2; }
    load_t0 = now_sec();
    model = bitnet_load_model(argv[1]);
    if (model == NULL) { fprintf(stderr, "load failed\n"); return 1; }
    load_sec = now_sec() - load_t0;
    vocab = bitnet_vocab_size(model);

    printf("{\"model\":\"%s\",\"vocab\":%d,\"embedding\":%d,\"threads_env\":\"%s\",\"load_sec\":%.6f",
           argv[1], vocab, bitnet_embedding_length(model), threads ? threads : "default", load_sec);

    printf(",\"ppl\":[");
    for (i = 0; i < k_ppl_texts_count; ++i) {
        double mean_nll = 0.0, prefill = 0.0;
        int n_cont = 0;
        if (run_ppl(model, vocab, i, &mean_nll, &n_cont, &prefill) != 0) {
            printf("]"); fprintf(stderr, "\nppl %d failed\n", i); return 1;
        }
        printf("%s{\"text\":%d,\"cont_tokens\":%d,\"mean_nll\":%.6f,\"ppl\":%.6f,\"prefill_sec\":%.6f}",
               i ? "," : "", i, n_cont, mean_nll, exp(mean_nll), prefill);
    }
    printf("]");

    {
        /* Decode speed from a fixed short prompt, plain greedy, fixed positions. */
        int tokens[CTX_GEN], n, gen[SPEED_DECODE_TOKENS], n_gen = 0, distinct = 0;
        double prefill = 0.0, decode_total = 0.0, t0, t1;
        bitnet_context_t *ctx = bitnet_create_context(model, CTX_GEN);
        if (ctx == NULL) { fprintf(stderr, "\nctx failed\n"); return 1; }
        n = bitnet_tokenize(model, (char *)k_speed_prompt, tokens, CTX_GEN);
        t0 = now_sec();
        if (n <= 0 || bitnet_eval(ctx, tokens, n) != 0) { fprintf(stderr, "\nprefill failed\n"); return 1; }
        t1 = now_sec();
        prefill = t1 - t0;
        for (t = 0; t < SPEED_DECODE_TOKENS; ++t) {
            int next = bitnet_sample_greedy(ctx);
            int j, seen = 0;
            if (next < 0) { fprintf(stderr, "\nsample failed\n"); return 1; }
            if (bitnet_token_is_eog(model, next)) break;
            for (j = 0; j < n_gen; ++j) if (gen[j] == next) { seen = 1; break; }
            if (!seen) distinct++;
            gen[n_gen++] = next;
            t0 = now_sec();
            if (bitnet_eval(ctx, &next, 1) != 0) { fprintf(stderr, "\ndecode failed\n"); return 1; }
            t1 = now_sec();
            decode_total += t1 - t0;
        }
        printf(",\"speed\":{\"prompt_tokens\":%d,\"ttft_sec\":%.6f,\"decode_tokens\":%d,"
               "\"decode_total_sec\":%.6f,\"decode_tok_s\":%.6f,\"distinct_generated\":%d}",
               n, prefill, n_gen, decode_total, n_gen > 0 ? (double)n_gen / decode_total : 0.0, distinct);
        bitnet_free_context(ctx);
    }

    printf(",\"arith\":[");
    {
        int score = 0;
        for (i = 0; i < k_arith_count; ++i) {
            int tokens[CTX_GEN], n, produced = 0, matched;
            char out[2048], piece[256];
            bitnet_context_t *ctx;
            long got = 0;
            snprintf(wrap, sizeof(wrap), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                     k_arith[i].prompt);
            ctx = bitnet_create_context(model, CTX_GEN);
            if (ctx == NULL) { fprintf(stderr, "\nctx failed\n"); return 1; }
            n = bitnet_tokenize(model, wrap, tokens, CTX_GEN);
            if (n <= 0 || bitnet_eval(ctx, tokens, n) != 0) { fprintf(stderr, "\narith %d prefill failed\n", i); return 1; }
            out[0] = '\0';
            for (t = 0; t < ARITH_MAX_TOKENS; ++t) {
                int next = bitnet_sample_greedy(ctx);
                int len;
                if (next < 0 || bitnet_token_is_eog(model, next)) break;
                len = bitnet_decode_token(model, next, piece, (int)sizeof(piece));
                if (len > 0 && produced + len < (int)sizeof(out) - 1) {
                    memcpy(out + produced, piece, (size_t)len);
                    produced += len;
                    out[produced] = '\0';
                }
                if (bitnet_eval(ctx, &next, 1) != 0) break;
            }
            matched = first_int(out, &got) && got == k_arith[i].expected;
            if (matched) score++;
            printf("%s{\"item\":%d,\"expected\":%d,\"got\":%ld,\"matched\":%s,\"reply\":",
                   i ? "," : "", i, k_arith[i].expected, got, matched ? "true" : "false");
            json_escape(out, produced);
            putchar('}');
            bitnet_free_context(ctx);
        }
        printf("],\"arith_score\":%d,\"arith_total\":%d}", score, k_arith_count);
    }

    bitnet_free_model(model);
    putchar('\n');
    return 0;
}
