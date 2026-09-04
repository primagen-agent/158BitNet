/* tok_probe.c -- tokenizer + logits probe for the Phase-A Python trainer.
 *
 * NEW STANDALONE FILE (not in CMake). Compile on the server:
 *   gcc -O2 tok_probe.c -I include -I src -o build/tok_probe build/libbitnet.a -lm -lpthread -ldl
 *
 * Modes:
 *   tok_probe <model.gguf> --encode     : reads UTF-8 text on stdin, prints
 *                                         space-separated token ids (no BOS)
 *   tok_probe <model.gguf> --eos        : prints the EOS token id
 *   tok_probe <model.gguf> --serve-decode: reads token-id lines and returns
 *                                          decoded bytes as lowercase hex
 *   tok_probe <model.gguf> --logits <json-ish-ids> : evaluates the ids,
 *                                         prints top-5 ids+logits per position
 *                                         (G1 backbone parity probe)
 *   tok_probe <model.gguf> --meta       : prints eos/bos/pad + vocab + eps
 */
#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> --encode|--eos|--serve-decode|--meta|--logits ids...\n",
                argv[0]);
        return 2;
    }
    bitnet_model_t *model = bitnet_load_model(argv[1]);
    if (model == NULL) { fprintf(stderr, "load failed\n"); return 1; }

    if (strcmp(argv[2], "--eos") == 0) {
        printf("%d\n", bitnet_eos_token(model));
    } else if (strcmp(argv[2], "--serve") == 0) {
        /* persistent tokenizer: one line of text per stdin line -> one line
         * of space-separated ids (no BOS). Model loads once. Lines are
         * arbitrary text; NUL bytes are not supported (fine for JSONL
         * renders). A line consisting of only "EOS?" prints the EOS id. */
        static char buf[1 << 20];
        static int toks[1 << 15];
        while (fgets(buf, sizeof buf, stdin) != NULL) {
            size_t n = strlen(buf);
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
                buf[--n] = '\0';
            if (strcmp(buf, "EOS?") == 0) {
                printf("%d\n", bitnet_eos_token(model));
                fflush(stdout);
                continue;
            }
            if (strcmp(buf, "BOS?") == 0) {
                /* The runtime prepends the tokenizer's own BOS when
                 * add_bos=1 (bitnet_tokenize_ex). We learn its id by
                 * tokenizing the empty string WITH add_bos: the single
                 * returned token (if any) is the BOS. */
                int cnt = bitnet_tokenize_ex(model, "", toks, 8, 1);
                printf("%d\n", cnt == 1 ? toks[0] : 1);
                fflush(stdout);
                continue;
            }
            int cnt = bitnet_tokenize_ex(model, buf, toks, 32768, 0);
            if (cnt <= 0) cnt = 0;
            for (int i = 0; i < cnt; ++i)
                printf("%d%s", toks[i], i + 1 < cnt ? " " : "");
            printf("\n");
            fflush(stdout);
        }
    } else if (strcmp(argv[2], "--serve-esc") == 0) {
        /* Like --serve but input lines use '\\n' as an escaped newline
         * (rendered prompts contain real newlines). Also handles EOS?/BOS?.*/
        static char buf[1 << 20];
        static char un[1 << 20];
        static int toks[1 << 15];
        while (fgets(buf, sizeof buf, stdin) != NULL) {
            size_t n = strlen(buf);
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
                buf[--n] = '\0';
            if (strcmp(buf, "EOS?") == 0) {
                printf("%d\n", bitnet_eos_token(model));
                fflush(stdout);
                continue;
            }
            if (strcmp(buf, "BOS?") == 0) {
                int cnt = bitnet_tokenize_ex(model, "", toks, 8, 1);
                printf("%d\n", cnt == 1 ? toks[0] : 1);
                fflush(stdout);
                continue;
            }
            /* unescape \n -> newline, \\ -> backslash */
            size_t u = 0;
            for (size_t i = 0; i < n && u + 1 < sizeof un; ++i) {
                if (buf[i] == '\\' && i + 1 < n && buf[i + 1] == 'n') {
                    un[u++] = '\n'; i++;
                } else if (buf[i] == '\\' && i + 1 < n && buf[i + 1] == '\\') {
                    un[u++] = '\\'; i++;
                } else {
                    un[u++] = buf[i];
                }
            }
            un[u] = '\0';
            int cnt = bitnet_tokenize_ex(model, un, toks, 32768, 0);
            if (cnt <= 0) cnt = 0;
            for (int i = 0; i < cnt; ++i)
                printf("%d%s", toks[i], i + 1 < cnt ? " " : "");
            printf("\n");
            fflush(stdout);
        }
    } else if (strcmp(argv[2], "--serve-decode") == 0) {
        static char line[1 << 20];
        while (fgets(line, sizeof line, stdin) != NULL) {
            char *save = NULL;
            for (char *item = strtok_r(line, " ,\r\n", &save);
                 item != NULL;
                 item = strtok_r(NULL, " ,\r\n", &save)) {
                char decoded[512];
                int n = bitnet_decode_token(
                    model, atoi(item), decoded, (int)sizeof decoded);
                for (int i = 0; i < n; ++i)
                    printf("%02x", (unsigned char)decoded[i]);
            }
            printf("\n");
            fflush(stdout);
        }
    } else if (strcmp(argv[2], "--meta") == 0) {
        printf("eos %d\n", bitnet_eos_token(model));
        bitnet_context_t *c = bitnet_create_context(model, 8);
        /* bos/pad not in public API; print tokenizer-visible basics */
        printf("vocab %d\n", bitnet_vocab_size(model));
        bitnet_free_context(c);
    } else if (strcmp(argv[2], "--encode") == 0) {
        static char buf[1 << 20];
        size_t n = fread(buf, 1, sizeof buf - 1, stdin);
        buf[n] = '\0';
        /* strip trailing newline to match C trainer semantics (it tokenizes
         * rendered strings that never end in a stray newline) */
        while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
        static int toks[1 << 15];
        int cnt = bitnet_tokenize_ex(model, buf, toks, 32768, 0);
        if (cnt <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        for (int i = 0; i < cnt; ++i) printf("%d%s", toks[i], i + 1 < cnt ? " " : "\n");
    } else if (strcmp(argv[2], "--logits") == 0 && argc >= 4) {
        int toks[2048];
        int n = 0;
        char *save = NULL;
        for (char *t = strtok_r(argv[3], " ,", &save); t && n < 2048;
             t = strtok_r(NULL, " ,", &save))
            toks[n++] = atoi(t);
        if (n <= 0) { fprintf(stderr, "no tokens\n"); return 1; }
        fprintf(stderr, "n_tokens=%d\n", n);
        bitnet_context_t *ctx = bitnet_create_context(model, 4096);
        if (ctx == NULL) { fprintf(stderr, "ctx failed\n"); return 1; }
        const float *lg = NULL;
        int vocab = bitnet_vocab_size(model);
        for (int pos = 0; pos < n; ++pos) {
            /* incremental eval: context pos advances; logits after each
             * eval are the LAST position's predictions -- so evaluate one
             * token at a time instead of re-running prefixes. */
            int rc = bitnet_eval(ctx, toks + pos, 1);
            if (rc != 0) { fprintf(stderr, "eval pos %d rc %d\n", pos, rc); return 1; }
            lg = bitnet_get_logits(ctx);
            int top[5];
            for (int k = 0; k < 5; ++k) top[k] = -1;
            for (int k = 0; k < 5; ++k) {
                int best = -1;
                for (int v = 0; v < vocab; ++v) {
                    int seen = 0;
                    for (int j = 0; j < k; ++j) if (top[j] == v) seen = 1;
                    if (seen) continue;
                    if (best < 0 || lg[v] > lg[best]) best = v;
                }
                top[k] = best;
            }
            printf("pos %d:", pos);
            for (int k = 0; k < 5; ++k)
                printf(" %d:%.4f", top[k], lg[top[k]]);
            printf("\n");
        }
        bitnet_free_context(ctx);
    } else {
        fprintf(stderr, "bad mode\n");
        return 2;
    }
    bitnet_free_model(model);
    return 0;
}
