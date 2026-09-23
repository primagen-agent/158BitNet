#ifndef NEURAL_MEMORY_H
#define NEURAL_MEMORY_H

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

/* Route prediction: given query/source C backbone features and prefix hidden.
 * query_features: [n_query × 2048] (from memory_feature_probe)
 * source_features: [n_source × 2048] or NULL if no stored facts
 * prefix_hidden: [1024] (from memory_gradient_reference)
 * Returns route decision (normal/supported/insufficient + mode). */
nm_route_t nm_predict_route(nm_ctx_t *ctx,
                             const float *query_features, int n_query,
                             const float *source_features, int n_source,
                             const float *prefix_hidden,
                             int n_allowed,
                             float *route_logits_out);

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

#endif /* NEURAL_MEMORY_H */
