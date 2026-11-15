#ifndef MINICPM_H
#define MINICPM_H

/* The "minicpm" family graph (MiniCPM4 / BitCPM-CANN): a dense decoder with
 * GQA attention, SwiGLU feed-forward, TQ2_0 ternary projections, a Q4_K
 * embedding table, a Q6_K head, embedding/residual/logit scales and
 * LongRoPE.  The same process serves the qwen35 family through bonsai.c;
 * this module is selected from general.architecture and never from a name
 * passed on the command line.
 *
 * The forward pass is written for the block verify: B activation rows share
 * one pass over each weight matrix.  B=1 is the ordinary decode path and it
 * uses the identical kernels, so speculative decoding cannot drift from
 * plain decoding by construction.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct minicpm_model minicpm_model_t;
typedef struct minicpm_context minicpm_context_t;

minicpm_model_t *minicpm_load_model(const char *path, int n_threads);
void minicpm_free_model(minicpm_model_t *model);

int minicpm_embedding_length(const minicpm_model_t *model);
int minicpm_vocab_size(const minicpm_model_t *model);
int minicpm_layer_count(const minicpm_model_t *model);

int minicpm_tokenize(minicpm_model_t *model, const char *text, int *tokens, int max_tokens);
int minicpm_decode_token(minicpm_model_t *model, int token, char *out, int out_size);
int minicpm_eos_token(const minicpm_model_t *model);
int minicpm_bos_token(const minicpm_model_t *model);
int minicpm_token_is_eog(minicpm_model_t *model, int token);

minicpm_context_t *minicpm_create_context(minicpm_model_t *model, int max_tokens);
void minicpm_free_context(minicpm_context_t *ctx);
int minicpm_reset_context(minicpm_context_t *ctx);
int minicpm_context_pos(const minicpm_context_t *ctx);
int minicpm_context_capacity(const minicpm_context_t *ctx);
/* Positions the file declares (general.<arch>.context_length); 0 = unknown. */
int minicpm_context_limit(const minicpm_model_t *model);

/* Append tokens; the last token's logits land in minicpm_get_logits(). */
int minicpm_eval(minicpm_context_t *ctx, const int *tokens, int n_tokens);
const float *minicpm_get_logits(const minicpm_context_t *ctx);
int minicpm_sample_greedy(const minicpm_context_t *ctx);

/* Block verify: score n tokens in one pass over the weights, writing one
 * logits row per position `stride` floats apart (stride == vocab is the
 * fast path).  Rows are chunked at 16 internally. */
int minicpm_eval_block(minicpm_context_t *ctx, const int *tokens, int n,
                       float *logits, int stride);

/* KV rows past the new end are simply overwritten by the next pass, so the
 * snapshot carries nothing but the position counter. */
size_t minicpm_state_bytes(const minicpm_context_t *ctx);
void minicpm_state_save(const minicpm_context_t *ctx, float *dst);
void minicpm_state_load(minicpm_context_t *ctx, const float *src);
void minicpm_rewind(minicpm_context_t *ctx, int n);

#ifdef __cplusplus
}
#endif

#endif
