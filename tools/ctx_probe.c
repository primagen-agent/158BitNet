/* What context a model actually allows, independent of any --ctx flag.
 *
 *   ctx_probe model.gguf [requested]
 *
 * bonsai_create_context() clamps to the model's declared
 * general.architecture.context_length, so the second number is the real limit
 * of the pair (file, length): ask for 1e6 and a 32k model still builds 32k.
 * With no requested length the probe only prints the model's shape.
 */
#include <stdio.h>
#include <stdlib.h>

#include "bonsai.h"

int main(int argc, char **argv) {
    bonsai_model_t *m;
    bonsai_context_t *c;
    int req = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [requested positions]\n", argv[0]);
        return 2;
    }
    if (argc >= 3) req = atoi(argv[2]);
    m = bonsai_load_model(argv[1], 1);
    if (m == NULL) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    printf("%s: arch %s, %d layers, %d embedding, vocab %d, declares %d positions\n",
           argv[1], bonsai_arch_name(bonsai_model_arch(m)), bonsai_layer_count(m),
           bonsai_embedding_length(m), bonsai_vocab_size(m), bonsai_context_limit(m));
    if (req > 0) {
        c = bonsai_create_context(m, req);
        if (c == NULL) {
            printf("requested %d: refused (not enough memory)\n", req);
            bonsai_free_model(m);
            return 1;
        }
        printf("requested %d -> capacity %d%s\n", req, bonsai_context_capacity(c),
               bonsai_context_capacity(c) < req ? "  (clamped to the model)" : "");
        bonsai_free_context(c);
    }
    bonsai_free_model(m);
    return 0;
}
