#ifndef NEURAL_MEMORY_H
#define NEURAL_MEMORY_H

#include <stddef.h>

/* Pure C neural memory inference: FineSpanReader + QueryOnlyUncertainty.
 * All weights loaded from binary files exported from PyTorch training. */

typedef struct neural_memory_ctx nm_ctx_t;

typedef struct {
    int route;      /* 0=normal, 1=supported, 2=insufficient */
    int mode;       /* 0=generate, 1=start value */
    float logit_normal, logit_supported, logit_insufficient;
} nm_route_t;

/* Initialize: load neural weights from .bnmodel file.
 * No output_head needed — the C backbone's quantized output projection
 * handles the final logits computation from the modified hidden state. */
nm_ctx_t *nm_init(const char *model_path);
void nm_free(nm_ctx_t *ctx);

/* Uncertainty branch: compute residual and add to hidden state.
 * The C backbone's quantized output projection then produces final logits
 * from the modified hidden — no separate output head file needed.
 * hidden: [1024] C backbone hidden state
 * modified_hidden: [1024] hidden + residual / logit_scale */
/* Wide 2048-dim span selection: score candidate spans using full backbone features.
 * Returns index of best value span, or -1. Writes span token range to best_span_idx. */
int nm_wide_select_value(nm_ctx_t *ctx,
                          const float *query_features,  /* [1×2048] */
                          const float *source_features, /* [n_source×2048] */
                          int n_source,
                          int *span_starts, int *span_ends,
                          int n_spans,
                          int *best_span_idx);

void nm_apply_residual(nm_ctx_t *ctx,
                        const float *hidden,
                        float *modified_hidden);

/* Per-token relevance score using the trained wide fact head.
 * wide_in: [4096] (query[2048] + token[2048])
 * h1: [256] intermediate buffer
 * score: [1] output */
void nm_wide_score(nm_ctx_t *ctx, const float *wide_in, float *h1, float *score);

/* Episode relevance, joint-pair form (EC-005, .bnmodel v3): the only
 * architecture that learned subject identity (EC-002 lineage). Score =
 * sigmoid over per-pair joint features [q;e;q*e;|q-e|] -> tanh -> per-dim
 * max -> mix with means. q_rows/ep_rows: [n x 2048] feature rows (the
 * episode side comes from write-time row caching); means are [2048].
 * Returns a probability in [0,1), or -1 when absent. */
int nm_has_episode_joint(const nm_ctx_t *ctx);
float nm_episode_joint_relevance(nm_ctx_t *ctx,
                                 const float *q_rows, int n_query,
                                 const float *q_mean,
                                 const float *ep_rows, int n_ep_rows,
                                 const float *ep_mean);

/* --- Training-exact reader route path (route parity, LC-002) --- */

/* Frame an episode message exactly like training (json.dumps sort_keys,
 * compact separators, escaped text). */
void nm_frame_text(const char *text, char *out, int out_size);

/* Frame the QUERY exactly like training: encoder_texts renders the context
 * list, so a query is a one-element ARRAY of the message object. */
void nm_frame_query(const char *text, char *out, int out_size);

/* Dynamic variants for arbitrary-length text (unbounded memory content).
 * buf/cap grow via realloc and are reused across calls; free once. */
void nm_frame_text_dyn(const char *text, char **buf, size_t *cap);
void nm_frame_query_dyn(const char *text, char **buf, size_t *cap);

/* Rebuild the training span layout from the framed source text and its
 * decoded token pieces (mirrors payload_mask + source_alignment +
 * ByteLayout.endpoints + candidate_spans, max_span=16).
 * pieces must reconstruct the framed text (optionally with a leading space
 * from the BOS decode artifact). Writes allowed[n_pieces] and the candidate
 * span token ranges. Returns n_spans, or -1 on framing mismatch. */
int nm_build_spans(const char *framed, const char *episode_text,
                   char **pieces, int n_pieces,
                   int *allowed_out, int *span_start_out, int *span_end_out,
                   int max_spans);

/* Route decision with the trained reader's span-based evidence
 * (fact-probability-weighted span representations). allowed/len come from
 * nm_build_spans; span arrays hold n_spans token ranges. */
nm_route_t nm_route_decision(nm_ctx_t *ctx,
                             const float *query_features, int n_query,
                             const float *source_features, int n_source,
                             const int *allowed, int n_allowed,
                             const int *span_starts, const int *span_ends,
                             int n_spans);

#endif /* NEURAL_MEMORY_H */
