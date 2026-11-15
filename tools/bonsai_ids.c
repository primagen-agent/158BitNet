/* Decode a list of token ids and print each one's bytes in hex, so a
 * tokenizer problem can be told apart from a model problem.
 *
 *   bonsai_ids -m model.gguf ID [ID...]
 */
#include "bonsai.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *model_path = NULL;
    bonsai_model_t *m;
    int i;
    char buf[256];

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else break;
    }
    if (model_path == NULL || i >= argc) {
        fprintf(stderr, "usage: bonsai_ids -m model.gguf ID [ID...]\n");
        return 1;
    }
    m = bonsai_load_model(model_path, 1);
    if (m == NULL) { fprintf(stderr, "load failed\n"); return 1; }
    printf("eos=%d bos=%d\n", bonsai_eos_token(m), bonsai_bos_token(m));
    for (; i < argc; ++i) {
        int id = atoi(argv[i]);
        int n, k;
        memset(buf, 0, sizeof(buf));
        n = bonsai_decode_token(m, id, buf, (int)sizeof(buf));
        printf("%7d  n=%3d  ", id, n);
        for (k = 0; k < n && k < 24; ++k) printf("%02x ", (unsigned char)buf[k]);
        printf("  |");
        for (k = 0; k < n && k < 24; ++k) {
            unsigned char ch = (unsigned char)buf[k];
            putchar(ch >= 32 && ch < 127 ? ch : '.');
        }
        printf("|\n");
    }
    bonsai_free_model(m);
    return 0;
}
