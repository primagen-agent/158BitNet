/* Isolated DG-003 diagnostic: ordinary native base + unquantized memory delta.
 * Access to frozen GGUF output weights is confined to this research executable. */
#include "../src/bitnet.c"

int main(int argc, char **argv) {
    FILE *input = NULL, *output = NULL;
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    uint32_t n, enabled, endian = 1, dims[3];
    int ids[128], rc = 1;
    float delta[1024], block[256], *correction = NULL, *combined = NULL;
    char magic[8];
    if (argc != 4 || sizeof(float) != 4 || sizeof(int) != 4 || *(unsigned char *)&endian != 1) return 2;
    input = fopen(argv[2], "rb"); output = fopen(argv[3], "wbx");
    if (!input || !output || fread(magic, 1, 8, input) != 8 || memcmp(magic, "BNCI0001", 8) ||
        fread(&n, 4, 1, input) != 1 || n < 2 || n > 128 || fread(ids, 4, n, input) != n ||
        fread(&enabled, 4, 1, input) != 1 || enabled > 1 || fread(delta, 4, 1024, input) != 1024 || fgetc(input) != EOF || ferror(input)) goto done;
    int nonzero = 0;
    for (uint32_t i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= 73448) goto done;
    for (int i = 0; i < 1024; ++i) { if (!isfinite(delta[i])) goto done; if (delta[i] != 0) nonzero = 1; }
    model = bitnet_load_model(argv[1]);
    if (!model || model->embedding_length != 1024 || model->vocab_size != 73448 || model->block_count != 24) goto done;
    const gguf_tensor_t *projection = model->tensor_cache.output ? model->tensor_cache.output : model->tensor_cache.token_embd;
    if (!projection || (projection->type != BITNET_TARGET_TENSOR_TYPE_Q6_K && projection->type != BITNET_TARGET_TENSOR_TYPE_F16)) goto done;
    ctx = bitnet_create_context(model, 512);
    if (!ctx || bitnet_context_pos(ctx) != 0 || bitnet_eval(ctx, ids, (int)n)) goto done;
    if (bitnet_context_pos(ctx) != (int)n) goto done;
    /* One process, one new context, one complete prefill; no decode KV reuse. */
    fprintf(stderr, "BNC_TRACE {\"initial_position\":0,\"final_position\":%u,\"eval_calls\":1,\"prefix_tokens\":%u}\n", n, n);
    correction = calloc(73448, sizeof(float)); combined = malloc(73448 * sizeof(float));
    if (!correction || !combined) goto done;
    memcpy(combined, bitnet_get_logits(ctx), 73448 * sizeof(float));
    if (enabled && nonzero) {
        const uint8_t *weights = gguf_get_tensor_ptr(&model->gguf, projection);
        if (!weights) goto done;
        for (int row = 0; row < 73448; ++row) {
            float sum = 0.0f;
            if (projection->type == BITNET_TARGET_TENSOR_TYPE_F16) {
                const uint16_t *w = (const uint16_t *)weights + (size_t)row * 1024;
                for (int i = 0; i < 1024; ++i) sum += bitnet_fp16_to_fp32(w[i]) * delta[i];
            } else for (int b = 0; b < 4; ++b) {
                const bitnet_q6k_block_t *w = (const bitnet_q6k_block_t *)(weights + ((size_t)row * 4 + (size_t)b) * BITNET_Q6K_BLOCK_SIZE);
                if (bitnet_q6k_dequantize_block(w, block, 256)) goto done;
                for (int i = 0; i < 256; ++i) sum += block[i] * delta[b * 256 + i];
            }
            if (model->logit_scale != 0.0f) sum /= model->logit_scale;
            if (!isfinite(sum)) goto done;
            correction[row] = sum; combined[row] += sum;
        }
    }
    dims[0] = n; dims[1] = 1024; dims[2] = 73448;
    if (fwrite("BNCO0001", 1, 8, output) != 8 || fwrite(dims, 4, 3, output) != 3 ||
        fwrite(bitnet_get_last_hidden(ctx), 4, 1024, output) != 1024 ||
        fwrite(bitnet_get_logits(ctx), 4, 73448, output) != 73448 || fwrite(correction, 4, 73448, output) != 73448 ||
        fwrite(combined, 4, 73448, output) != 73448) goto done;
    rc = 0;
done:
    if (input) fclose(input);
    if (output && fclose(output)) rc = 1;
    free(correction); free(combined); bitnet_free_context(ctx); bitnet_free_model(model);
    if (rc) fprintf(stderr, "continuous-memory probe failed; output is incomplete\n");
    return rc;
}
