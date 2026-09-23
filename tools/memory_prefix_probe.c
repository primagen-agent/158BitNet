/* Batch extraction of frozen C prefix features. Reuse weights, never contexts. */
#include "bitnet.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    FILE *in = NULL, *out = NULL;
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    uint32_t rows, n, endian = 1, dims[3], positions[3];
    int ids[128], rc = 1;
    char magic[8];
    if (argc != 4 || sizeof(float) != 4 || sizeof(int) != 4 || *(unsigned char *)&endian != 1) return 2;
    in = fopen(argv[2], "rb"); out = fopen(argv[3], "wbx");
    if (!in || !out || fread(magic, 1, 8, in) != 8 || memcmp(magic, "BNPI0001", 8) ||
        fread(&rows, 4, 1, in) != 1 || rows < 1 || rows > 10000) goto done;
    model = bitnet_load_model(argv[1]);
    if (!model || bitnet_embedding_length(model) != 1024 || bitnet_vocab_size(model) != 73448) goto done;
    dims[0] = rows; dims[1] = 1024; dims[2] = 73448;
    if (fwrite("BNPO0001", 1, 8, out) != 8 || fwrite(dims, 4, 3, out) != 3) goto done;
    for (uint32_t row = 0; row < rows; ++row) {
        if (fread(&n, 4, 1, in) != 1 || n < 2 || n > 128 || fread(ids, 4, n, in) != n) goto done;
        for (uint32_t i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= 73448) goto done;
        ctx = bitnet_create_context(model, 512);
        if (!ctx || bitnet_context_pos(ctx) != 0 || bitnet_eval(ctx, ids, (int)n) || bitnet_context_pos(ctx) != (int)n) goto done;
        positions[0] = n; positions[1] = 0; positions[2] = (uint32_t)bitnet_context_pos(ctx);
        if (fwrite(positions, 4, 3, out) != 3 || fwrite(bitnet_get_last_hidden(ctx), 4, 1024, out) != 1024 ||
            fwrite(bitnet_get_logits(ctx), 4, 73448, out) != 73448) goto done;
        bitnet_free_context(ctx); ctx = NULL;
    }
    if (fgetc(in) != EOF || ferror(in)) goto done;
    fprintf(stderr, "BNP_TRACE rows=%u fresh_contexts=%u\n", rows, rows);
    rc = 0;
done:
    bitnet_free_context(ctx); bitnet_free_model(model);
    if (in) fclose(in);
    if (out && fclose(out)) rc = 1;
    if (rc) fprintf(stderr, "prefix extraction failed; do not consume partial output\n");
    return rc;
}
