#ifndef QWEN3_H
#define QWEN3_H

#include <stdint.h>

/* Dense Qwen3 GGUF runtime for the ternary (Q2_0 g128) Bonsai-8B packing.
 * The engine is built for speculative decoding: eval() takes a block of
 * tokens, verifies them in one weight pass, and can hand out the hidden
 * states of the layers the DSpine draft conditions on. */

typedef struct qwen3_model qwen3_model_t;
typedef struct qwen3_ctx qwen3_ctx_t;

qwen3_model_t *qwen3_load(const char *path, int n_threads);
void qwen3_free(qwen3_model_t *m);

int qwen3_n_vocab(const qwen3_model_t *m);
int qwen3_n_embd(const qwen3_model_t *m);
int qwen3_n_layer(const qwen3_model_t *m);
int qwen3_eos(const qwen3_model_t *m);

int qwen3_tokenize(qwen3_model_t *m, const char *text, int *out, int max_out);
int qwen3_detokenize(qwen3_model_t *m, int id, char *out, int out_size);

qwen3_ctx_t *qwen3_ctx_new(qwen3_model_t *m, int n_ctx);
void qwen3_ctx_free(qwen3_ctx_t *c);
void qwen3_ctx_reset(qwen3_ctx_t *c);
int qwen3_pos(const qwen3_ctx_t *c);

/* Evaluate `n` tokens starting at the context position.  When logits != NULL
 * it receives one row per position ([n][stride]); when hid != NULL it receives
 * the post-block hidden states of sel_layers ([n][n_sel*n_embd]). */
int qwen3_eval(qwen3_ctx_t *c, const int *tokens, int n,
               float *logits, int stride,
               const int *sel_layers, int n_sel, float *hid);
int qwen3_kv_commit(qwen3_ctx_t *c, int n);   /* advance past verified tokens */
void qwen3_kv_rewind(qwen3_ctx_t *c, int n);  /* drop the last n positions */

#endif
