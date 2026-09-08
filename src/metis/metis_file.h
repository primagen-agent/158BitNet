/* metis.h -- Metis memory model: 32 per-layer memories with
 * GQA reads through the FROZEN backbone q_proj/o_proj.
 *
 * Architecture (aligned with the reference implementation's 95%-accuracy
 * recipe; delta-rule micro-math stays our locked C-runtime semantics):
 *   - one memory (M[kv_dim x kv_dim], S[kv_dim]) per backbone layer
 *   - read: q = frozen backbone q_proj @ h_normed (q_dim, PRE-RoPE);
 *     per-head L2 normalize (head_dim), groups of q_dim/kv_dim heads read the
 *     SAME per-layer M with read = (q~ M)/(q~ S), NO +1, empty-state guard;
 *     concat groups -> [q_dim]; mem_norm (RMSNorm over q_dim, trainable);
 *     fused = frozen backbone o_proj @ mem_normed -> [d_model];
 *     attn' = gamma*attn + (1-gamma)*fused.
 *   - write: per-layer commit of that layer's captured raw residual rows:
 *     one attn_norm, w_agg scores, AlphaTopP(rho)
 *     hard select, K = L2normalize(W_k h)/sqrt(kv_dim), V = W_v h,
 *     alpha = sum w sigma(gdu_aw.h+gdu_ab)/sum w,
 *     beta_i = beta_scale * sigma(gdu_bw.h+gdu_bb),
 *     GDU with pre-decay km/ks and S never pre-scaled by alpha.
 *
 * BNMEM2 stores only trainable parameters plus the SHA-256 of the exact
 * frozen backbone GGUF. Runtime loading rejects unbound files and digest
 * mismatches. Layers count <= METIS_MAX_LAYERS(32).
 */
#ifndef METIS_V6_H
#define METIS_V6_H
#include <stdint.h>
#include <stddef.h>

/* capture window per layer, in rows. One server exchange (prefill +
 * decoded answer) is far below this; evals that would overflow are rejected
 * with -3 by the same rule as the v2 path. Matches the v2 commit slice cap
 * so a full window always commits in a single metis_commit_states pass. */
#define METIS_CAP_ROWS 1024

typedef struct metis_params {
    int has_backbone_sha256;
    uint8_t backbone_sha256[32];
    int n_layers;
    int layer_ids[32];      /* backbone block index per slot, strictly incr */
    int d_model;            /* backbone hidden (2560 for the 3B) */
    int kv_dim;             /* memory dim = n_kv_heads*head_dim (256) */
    int q_dim;              /* backbone q_proj out dim (4096) */
    int head_dim;           /* per-head L2 normalize width (128) */
    float gamma, tau, rho;
    int k_min;
    int alpha_max_tokens;       /* 0 disables the fixed AlphaTopP cap */
    float alpha_max_fraction;   /* 0 disables the relative AlphaTopP cap */
    float gdu_ab, gdu_bb;   /* fixed scalar gate biases (reference init) */
    float beta_scale;       /* reference update_ratio */
    /* v4 (reference-trained direct-use models, version 4): per-layer trained
     * query path + per-layer gate biases + +1 read denominator. When
     * version == 4, query_proj/query_norm are non-NULL and the runtime uses
     * metis_commit_states/metis_read. */
    int denom_plus_one;     /* 1: qS+1; 2: |qS|+1 */
    float *gdu_ab_v, *gdu_bb_v;   /* [n_layers] per-layer gate biases (v4) */
    float *query_proj;      /* [n_layers][q_dim x d_model] row-major (v4;
                             * NULL in v5 low-rank files) */
    float *query_norm;      /* [n_layers][head_dim] (v4) */
    /* v5 (magic BNMEM5, version 5): low-rank query factors. When query_rank
     * > 0 the read computes q = (h @ B^T) @ A^T per layer from these
     * factors instead of the folded full-rank matrix (~110MB vs 1.34GB
     * for the 3B/32-layer model at rank 128). */
    int query_rank;         /* 0 = full-rank (v4 semantics); >0 = v5 */
    int query_add_backbone; /* low-rank query is a delta on pre-RoPE Q */
    float *query_a;         /* [n_layers][q_dim x query_rank] row-major */
    float *query_b;         /* [n_layers][query_rank x d_model] row-major */
    /* Optional SVD-factorized write projections.  kv_rank == 0 keeps the
     * legacy full wk/wv tensors; kv_rank > 0 uses A(Bh) directly.  Dynamic
     * memory state M/S is deliberately unchanged and remains full-rank. */
    int kv_rank;
    float *wk_a;            /* [n_layers][kv_dim x kv_rank] */
    float *wk_b;            /* [n_layers][kv_rank x d_model] */
    float *wv_a;            /* [n_layers][kv_dim x kv_rank] */
    float *wv_b;            /* [n_layers][kv_rank x d_model] */
    /* per-layer trainable tensors, layer-major concatenation:
     * wk/wv: [n_layers][kv_dim x d_model] row-major
     * w_agg/gdu_aw/gdu_bw: [n_layers][d_model]
     * mem_norm: [n_layers][q_dim] */
    float *wk, *wv, *w_agg, *gdu_aw, *gdu_bw, *mem_norm;
} metis_params_t;

typedef struct metis_file_model {
    int version;            /* always 3 */
    metis_params_t params;
} metis_file_model_t;

/* Allocate with exact tensor sizes (all zero except mem_norm = 1.0);
 * low-rank query factors per layer (query_rank). */
int  metis_model_alloc(metis_file_model_t *m, int n_layers,
                             const int *layer_ids, int d_model, int kv_dim,
                             int q_dim, int head_dim, float gamma, float tau,
                             float rho, int k_min, float beta_scale,
                             int denom_plus_one, int query_rank);

/* Free the heap arrays inside m (pointers NULLed, struct zeroed). */
void metis_model_free_arrays(metis_file_model_t *m);

/* Free arrays AND the container (load returns a heap container). */
void metis_model_free_loaded(metis_file_model_t *m);

/* Serialize (same manifest scheme as v2: name/rows/cols/offset/crc32 per
 * tensor, trailing manifest, no trailing garbage). 0 ok, -1 I/O failure. */
int  metis_model_save(const metis_file_model_t *m, const char *path);

/* Load + validate against the backbone geometry. d/q/kv/head dims must match
 * the loaded model exactly; n_layers <= block_count; layer ids < block_count.
 * Returns a heap container or NULL + message in err. */
metis_file_model_t *metis_model_load(const char *path, int d_model_expected,
                                           int q_dim_expected, int kv_dim_expected,
                                           int head_dim_expected,
                                           int block_count_expected,
                                           char *err, size_t err_size);

/* Persist only the mutable per-context memory state. The snapshot is bound
 * to the memory model geometry and layer ids; load validates everything and
 * does not modify M/S unless the complete file and both CRCs are valid. */
int metis_state_save(const metis_params_t *p, const float *M, const float *S,
                     int active, const char *path);
int metis_state_load(const metis_params_t *p, float *M, float *S,
                     int *active, const char *path,
                     char *err, size_t err_size);

/* Quick magic probe: returns 1 when the file starts with the v3 magic. */
int  metis_file_probe(const char *path);

/* ---- runtime math (shared by the C runtime and mirrors for parity) ---- */

/* One commit pass over L normed hidden rows for ONE layer slot. h_all rows
 * are PRE-norm hiddens of this layer; the norm is recomputed here (locked
 * single norm). norm_w is this layer's attn_norm weight [d_model]. */
int metis_commit_states(const metis_params_t *p, int slot,
                           float *M, float *S, const float *h_all, int L,
                           const float *norm_w, float eps);

/* Memory read for one token at one layer: raw residual [d_model] on the
 * trained query path (metis_read).  out receives [q_dim]; the caller
 * applies mem_norm then o_proj.
 *
 * Verified line-by-line against the reference Metis stack:
 *   - WRITE: h_all rows are the layer's RAW RESIDUAL inputs, normed ONCE
 *     here with the layer's attn_norm (input_layernorm(raw_info)); gdu
 *     biases are the per-layer TRAINED scalars; S IS pre-scaled by alpha
 *     (reference: S_new = alpha*(S - K^T(beta*(KS))) + K^T beta).
 *   - READ: q = query_proj(h_raw) [q_dim]; reshape [n_heads, head_dim];
 *     query_norm RMSNorm(head_dim, eps 1e-6); per-head L2; groups of
 *     q_dim/kv_dim consecutive heads read the SAME M with
 *     out = (q~ M)/(q~ S + 1.0) (denominator +1; no empty-state guard). */

int metis_commit_states(const metis_params_t *p, int slot,
                              float *M, float *S, const float *h_all, int L,
                              const float *norm_w, float eps);

void metis_read(const metis_params_t *p, int slot,
                      const float *h_raw /*[d_model]*/,
                      const float *q_backbone /*[q_dim], nullable*/,
                      const float *M,
                      const float *S, float *out /*[q_dim]*/);

#endif
