#ifndef BONSAI_H
#define BONSAI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Inference for the Qwen3.5-hybrid "Ternary-Bonsai" GGUF family
 * (architecture "qwen35", PTQ1_0 ternary weights, Prism Hadamard folding). */

#include <stddef.h>

typedef struct bonsai_model bonsai_model_t;
typedef struct bonsai_context bonsai_context_t;

/* The model family, decided from the file's general.architecture when it is
 * opened.  One binary serves every format listed here: the loader reads the
 * family, then the runtime runs the graph and kernels that family needs. */
typedef enum {
    BONSAI_ARCH_UNKNOWN = 0,
    BONSAI_ARCH_QWEN35  = 1,   /* Ternary-Bonsai: qwen35 graph, PTQ1_0 weights */
    BONSAI_ARCH_MINICPM = 2,   /* BitCPM-CANN: minicpm graph, TQ2_0 weights */
    BONSAI_ARCH_LLAMA   = 3    /* MiniCPM4 1B/3B: the same graph without the
                                * MiniCPM embedding/residual/logit scales, which
                                * the loader defaults to 1 */
} bonsai_arch_t;

/* Name -> family.  NULL or an unknown name gives BONSAI_ARCH_UNKNOWN. */
bonsai_arch_t bonsai_arch_from_name(const char *name);
const char *bonsai_arch_name(bonsai_arch_t arch);
/* The families this build can actually run, for error messages. */
const char *bonsai_arch_supported(void);
/* The family of an opened model. */
bonsai_arch_t bonsai_model_arch(const bonsai_model_t *model);

bonsai_model_t *bonsai_load_model(const char *path, int n_threads);
void bonsai_free_model(bonsai_model_t *model);

int bonsai_embedding_length(const bonsai_model_t *model);
int bonsai_vocab_size(const bonsai_model_t *model);
int bonsai_layer_count(const bonsai_model_t *model);

int bonsai_tokenize(bonsai_model_t *model, const char *text, int *tokens, int max_tokens);
int bonsai_decode_token(bonsai_model_t *model, int token, char *out, int out_size);
int bonsai_eos_token(const bonsai_model_t *model);
int bonsai_bos_token(const bonsai_model_t *model);
int bonsai_token_is_eog(bonsai_model_t *model, int token);

bonsai_context_t *bonsai_create_context(bonsai_model_t *model, int max_tokens);
void bonsai_free_context(bonsai_context_t *ctx);
int bonsai_reset_context(bonsai_context_t *ctx);
int bonsai_context_pos(const bonsai_context_t *ctx);
/* Positions the model file declares; a context is clamped to this unless
 * BQ_CTX_EXTEND=1 asks for beyond the declared range (MiniCPM ships LongRoPE
 * factors for its long variants, so extension is possible but not declared). */
int bonsai_context_limit(const bonsai_model_t *model);
int bonsai_context_capacity(const bonsai_context_t *ctx);

int bonsai_eval(bonsai_context_t *ctx, const int *tokens, int n_tokens);
const float *bonsai_get_logits(const bonsai_context_t *ctx);
int bonsai_sample_greedy(const bonsai_context_t *ctx);


/* ---- block verification, the target side of speculative decoding ------- */

/* Score `n` tokens in one weight pass (n is split into chunks of at most 16).
 * logits, when non-NULL, receives one row per position, `stride` floats
 * apart.  hid, when non-NULL, receives the residual stream after each layer
 * listed in sel_layers as [n][n_sel*n_embd] - the view a draft model such as
 * DSpine conditions on.  layer_in, when non-NULL, records the residual stream
 * entering every layer as [n_layer][n][n_embd]; bonsai_replay_state() needs
 * it after a rejected tail. */
int bonsai_eval_block(bonsai_context_t *ctx, const int *tokens, int n,
                      float *logits, int stride,
                      const int *sel_layers, int n_sel, float *hid,
                      float *layer_in);

/* Recurrent state (gated delta net + convolution history) snapshots. */
size_t bonsai_state_bytes(const bonsai_context_t *ctx);
void bonsai_state_save(const bonsai_context_t *ctx, float *dst);
void bonsai_state_load(bonsai_context_t *ctx, const float *src);

/* Drop the last n positions from the context (attention is positional). */
void bonsai_rewind(bonsai_context_t *ctx, int n);

/* Rebuild the recurrent state for the first n positions from the residual
 * streams recorded by bonsai_eval_block().  `stride`, when given by the
 * strided form, is the block length those rows were recorded with. */
void bonsai_replay_state(bonsai_context_t *ctx, const float *layer_in, int n);
/* Rollback cache: keeps the gated delta net inputs of the last evaluated block
 * so a rewind rebuilds the state without a second weight pass.  Optional; the
 * plain replay works without it. */
int bonsai_replay_cache_init(bonsai_context_t *ctx);
int bonsai_replay_state_cached(bonsai_context_t *ctx, const float *layer_in,
                               int stride, int n);
void bonsai_replay_state_stride(bonsai_context_t *ctx, const float *layer_in,
                                int stride, int n);

/* ---- draft-model support (speculative decoding) ----------------------- */

/* Shared thread pool of the target model. */
void *bonsai_pool(const bonsai_model_t *model);
int bonsai_pool_threads(const bonsai_model_t *model);
void bonsai_pool_run(void *pool, void (*fn)(void *, int, int), void *arg);

int bonsai_block_max(void);
int bonsai_fold_block(const bonsai_model_t *model);
const float *bonsai_fold_signs(const bonsai_model_t *model, int width);
void bonsai_fold_hadamard(float *x, int n, int block);

/* Embedding and LM-head rows in the true (unfolded) basis. */
void bonsai_embed_row(const bonsai_model_t *model, int token, float *out);
void bonsai_head_row(const bonsai_model_t *model, int token, float *out);

/* Target LM head over up to 16 already-final-normed rows: logits[b][:] for
 * b < n, written `stride` floats apart. */
int bonsai_head_logits(bonsai_context_t *ctx, const float *x, int n,
                       float *logits, int stride);

/* Print per-stage timings to stderr when BQ_PROFILE=1. */
void bonsai_profile_dump(void);

/* Park the shared TQ2 row pool.  Its workers spin between dispatches on the
 * assumption that the next token follows within microseconds; that assumption
 * holds inside a generation and not at the end of one, where the spin is
 * charged to whatever runs next (a BitCPM request followed by a 27B+DSpine
 * request cost the latter 20% until it parked).  Call at the end of a
 * request, not between tokens. */
void bonsai_park_workers(void);

#ifdef __cplusplus
}
#endif

#endif
