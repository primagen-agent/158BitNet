/* Isolated one-position residual experiment. No serving API or model mutation.
 * The reference target compiles against ordinary libbitnet without injection. */
#include "bitnet.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float injection[1024], before[1024], after[1024];
static uint32_t injection_layer, injection_token, enabled;
#ifndef BITNET_GRADIENT_REFERENCE
static int captures;
static FILE *cell_trace;
static int cell_error;
static void cell_float(int stage, int layer, int token, const float *data, int size) {
    if (!cell_trace || cell_error || layer != (int)injection_layer || token != (int)injection_token || stage < 11 || stage > 16) return;
    uint32_t header[3] = {(uint32_t)stage, 1, (uint32_t)size};
    if (size <= 0 || fwrite(header, 4, 3, cell_trace) != 3 || fwrite(data, 4, (size_t)size, cell_trace) != (size_t)size) cell_error = 1;
}
static void cell_quant(int stage, int layer, int token, const int8_t *data, int size, float scale) {
    if (!cell_trace || cell_error || token != (int)injection_token) return;
    if (!((layer == (int)injection_layer && (stage == 18 || stage == 20)) ||
          (layer == (int)injection_layer + 1 && stage == 21))) return;
    uint32_t header[3] = {(uint32_t)stage, 2, (uint32_t)size};
    if (size <= 0 || fwrite(header, 4, 3, cell_trace) != 3 || fwrite(&scale, 4, 1, cell_trace) != 1 ||
        fwrite(data, 1, (size_t)size, cell_trace) != (size_t)size) cell_error = 1;
}
static void inject_residual(int layer, int token, float *hidden, int size) {
    if (layer != (int)injection_layer || token != (int)injection_token) return;
    if (size != 1024) return;
    ++captures;
    memcpy(before, hidden, sizeof before);
    if (enabled) for (int i = 0; i < 1024; ++i) hidden[i] += injection[i];
    memcpy(after, hidden, sizeof after);
}
#define BITNET_DIAGNOSTIC_INJECT inject_residual
#define BITNET_TRACE_FLOAT cell_float
#define BITNET_TRACE_Q8 cell_quant
#include "../src/bitnet.c"
#endif

int main(int argc, char **argv) {
    FILE *in = NULL, *out = NULL;
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    char magic[8]; uint32_t n, hidden = 1024, vocab = 73448, endian = 1;
    int ids[128], rc = 1;
    if ((argc != 4 && argc != 5) || sizeof(float) != 4 || sizeof(int) != 4 || *(unsigned char *)&endian != 1) return 2;
#ifdef BITNET_GRADIENT_REFERENCE
    if (argc != 4) return 2;
#else
    if (argc == 5) {
        cell_trace = fopen(argv[4], "wbx");
        if (!cell_trace) return 2;
        if (fwrite("BNGC0001", 1, 8, cell_trace) != 8) cell_error = 1;
    }
#endif
    in = fopen(argv[2], "rb"); out = fopen(argv[3], "wbx");
    if (!in || !out || fread(magic, 1, 8, in) != 8 || memcmp(magic, "BNGI0001", 8) ||
        fread(&n, 4, 1, in) != 1 || n < 2 || n > 128 || fread(ids, 4, n, in) != n ||
        fread(&injection_layer, 4, 1, in) != 1 || fread(&injection_token, 4, 1, in) != 1 ||
        fread(&enabled, 4, 1, in) != 1 || fread(injection, 4, 1024, in) != 1024 || fgetc(in) != EOF || ferror(in)) goto done;
    if (injection_layer != 23 || injection_token != n - 1 || enabled > 1) goto done;
    for (uint32_t i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= (int)vocab) goto done;
    for (int i = 0; i < 1024; ++i) if (!isfinite(injection[i])) goto done;
    model = bitnet_load_model(argv[1]);
    if (!model || bitnet_embedding_length(model) != (int)hidden || bitnet_vocab_size(model) != (int)vocab) goto done;
    ctx = bitnet_create_context(model, 512);
    if (!ctx || bitnet_context_pos(ctx) != 0 || bitnet_eval(ctx, ids, (int)n)) goto done;
#ifndef BITNET_GRADIENT_REFERENCE
    if (captures != 1 || cell_error) goto done;
#endif
    uint32_t dimensions[5] = {n, hidden, vocab, injection_layer, injection_token};
    if (fwrite("BNGO0001", 1, 8, out) != 8 || fwrite(dimensions, 4, 5, out) != 5 ||
        fwrite(before, 4, hidden, out) != hidden || fwrite(after, 4, hidden, out) != hidden ||
        fwrite(bitnet_get_last_hidden(ctx), 4, hidden, out) != hidden || fwrite(bitnet_get_logits(ctx), 4, vocab, out) != vocab) goto done;
    rc = 0;
done:
#ifndef BITNET_GRADIENT_REFERENCE
    if (cell_trace && fclose(cell_trace)) rc = 1;
#endif
    if (in) fclose(in);
    if (out && fclose(out)) rc = 1;
    bitnet_free_context(ctx); bitnet_free_model(model);
    if (rc) fprintf(stderr, "native gradient probe failed; do not consume partial output\n");
    return rc;
}
