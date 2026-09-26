#ifndef BITNET_H
#define BITNET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitnet_model bitnet_model_t;
typedef struct bitnet_context bitnet_context_t;

#define BITNET_KV_CACHE_F32 0
#define BITNET_KV_CACHE_Q8 1

bitnet_model_t *bitnet_load_model(const char *path);
bitnet_context_t *bitnet_create_context(bitnet_model_t *model, int max_tokens);
int bitnet_load_lora(bitnet_model_t *model, const char *path, float scale);
int bitnet_lora_count(const bitnet_model_t *model);
int bitnet_embedding_length(const bitnet_model_t *model);
int bitnet_vocab_size(const bitnet_model_t *model);
/*
 * Read raw token-embedding rows exactly as stored in the GGUF tensor.
 * Output is [token_count][bitnet_embedding_length(model)] in float32.
 * Model-specific embedding scaling used by transformer inference is
 * intentionally not applied, matching typed-memory training inputs.
 */
int bitnet_token_embedding_lookup(
    const bitnet_model_t *model,
    const int *tokens, int token_count,
    float *output, size_t output_count);
int bitnet_chat_template_kind(const bitnet_model_t *model);
const float *bitnet_get_last_hidden(const bitnet_context_t *ctx);
/* Mean of output-normalized hidden rows from the most recent bitnet_eval,
 * plus its final row. This matches the portable memory-controller pooling
 * used during training. */
const float *bitnet_get_last_pooled_hidden(const bitnet_context_t *ctx);
/* Output-normalized hidden rows from the most recent bitnet_eval, laid out as
 * [bitnet_last_eval_hidden_count(ctx)][bitnet_embedding_length(model)]. */
const float *bitnet_get_last_eval_hidden(const bitnet_context_t *ctx);
int bitnet_last_eval_hidden_count(const bitnet_context_t *ctx);
/*
 * Capture RMS-normalized residual states after transformer layers and average
 * them into contiguous layer bands. Bands must cover every model layer once,
 * in order. The most recent eval is exposed as
 * [token][band][embedding_length]. This is an internal-state observation
 * path: it does not read, write, or reuse the KV cache.
 */
int bitnet_context_configure_layer_bands(
    bitnet_context_t *ctx, const int *band_start,
    const int *band_end, int band_count);
const float *bitnet_get_last_eval_layer_bands(
    const bitnet_context_t *ctx);
int bitnet_last_eval_layer_band_token_count(
    const bitnet_context_t *ctx);
int bitnet_layer_band_count(const bitnet_context_t *ctx);

int bitnet_tokenize(bitnet_model_t *model, const char *text, int *tokens, int max_tokens);
int bitnet_tokenize_ex(bitnet_model_t *model, const char *text, int *tokens, int max_tokens, int add_bos);
int bitnet_decode_token(bitnet_model_t *model, int token, char *out, int out_size);
int bitnet_eos_token(bitnet_model_t *model);
int bitnet_pad_token(bitnet_model_t *model);
int bitnet_token_is_eog(bitnet_model_t *model, int token);

int bitnet_eval(bitnet_context_t *ctx, const int *tokens, int n_tokens);
/* Run the same causal transformer path but stop after hidden-state capture.
 * Intended for neural side-car encoders that do not consume vocabulary
 * logits. It never reuses state from another context. */
int bitnet_eval_hidden(
    bitnet_context_t *ctx, const int *tokens, int n_tokens);
int bitnet_sample_greedy(bitnet_context_t *ctx);
int bitnet_sample_greedy_repetition_penalty(bitnet_context_t *ctx,
                                            const int *tokens,
                                            int n_tokens,
                                            float penalty);
const float *bitnet_get_logits(const bitnet_context_t *ctx);

/* Project a FINAL-NORMED hidden vector (as returned by bitnet_get_last_hidden)
 * through the model's output projection (Q6_K path) without a transformer
 * forward. Neural-memory uncertainty uses this to inject a residual into the
 * hidden state: logits = OutputProj(hidden + residual).
 * Returns the context's logits buffer (vocab-sized), or NULL when the model's
 * output path is unsupported. The buffer is overwritten by the next eval. */
const float *bitnet_project_hidden_to_logits(bitnet_context_t *ctx,
                                             const float *hidden_normed);
void bitnet_apply_repetition_penalty(float *logits,
                                     int vocab_size,
                                     const int *tokens,
                                     int n_tokens,
                                     float penalty);

int bitnet_context_pos(const bitnet_context_t *ctx);
int bitnet_context_capacity(const bitnet_context_t *ctx);
int bitnet_reset_context(bitnet_context_t *ctx);
int bitnet_rewind_context(bitnet_context_t *ctx, int n_pos);
int bitnet_set_kv_cache_type(bitnet_context_t *ctx, int cache_type);
int bitnet_get_kv_cache_type(const bitnet_context_t *ctx);
unsigned long long bitnet_kv_cache_bytes(const bitnet_context_t *ctx);

void bitnet_free_context(bitnet_context_t *ctx);
void bitnet_free_model(bitnet_model_t *model);

#ifdef __cplusplus
}
#endif

#endif
