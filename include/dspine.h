#ifndef DSPINE_H
#define DSPINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* The DSpine block-draft model (tencent Qwen3.8-27B-DSpine-b16) as the
 * speculative-decoding drafter for the Ternary-Bonsai 27B target.
 *
 * A draft round is:
 *   - append the target's residual-stream rows (the `hid` buffer from
 *     bonsai_eval_block) to the draft's context K/V;
 *   - propose() a block of `n` token ids after the anchor;
 *   - the caller verifies them with bonsai_eval_block and keeps a prefix. */

#include <stddef.h>

struct bonsai_model;
struct bonsai_context;
typedef struct bonsai_model bonsai_model_t;
typedef struct bonsai_context bonsai_context_t;

typedef struct dspine_model dspine_model_t;

/* Loads the safetensors checkpoint and binds it to the target.  The draft
 * borrows the target's embedding, LM head and thread pool. */
dspine_model_t *dspine_load(const char *path, bonsai_model_t *target,
                            bonsai_context_t *ctx);
void dspine_free(dspine_model_t *d);

/* Number of drafts one round can propose (block size - 1). */
int dspine_max_drafts(const dspine_model_t *d);
int dspine_mask_token(const dspine_model_t *d);
int dspine_context_len(const dspine_model_t *d);

/* Forget the context cache (start of a new sequence). */
void dspine_reset(dspine_model_t *d);

/* Append n positions of target residual stream.  `hid` is the buffer filled by
 * bonsai_eval_block(sel_layers=target_layer_ids), i.e. n rows of
 * n_sel*n_embd floats in layer order. */
int dspine_append_context(dspine_model_t *d, const float *hid, int n);

/* Propose `n` token ids following `anchor`; writes them to drafts[0..n-1].
 * The context must already contain every position before the anchor. */
int dspine_propose(dspine_model_t *d, int anchor, int n, int *drafts);

/* Timing counters (seconds) for the last propose() call. */
void dspine_timings(const dspine_model_t *d, double *block, double *tail);

#ifdef __cplusplus
}
#endif

#endif
