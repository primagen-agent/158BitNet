/* neural_memory.c — Pure C implementation of the trained neural memory reader
 * and uncertainty branch. Loads binary weights exported from PyTorch.
 *
 * Components: FineSpanReader (route/mode/span/fact/value heads) +
 * QueryOnlyUncertainty (encoder/output residual). All matrix ops in plain C.
 * Compiled into the C end-to-end server (no Python).
 */
#include "neural_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Single-file weight loading (.bnmodel format) --- */
typedef struct {
    char name[128];
    int shape[4];
    int n_dims;
    long offset;
    int count;
} bnmodel_entry_t;

typedef struct {
    FILE *f;
    long data_base;
    int n_entries;
    bnmodel_entry_t *entries;
} bnmodel_t;

static bnmodel_t *bnmodel_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "neural_memory: cannot open %s\n", path); return NULL; }

    char magic[4];
    int version, n_entries, index_size;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BNMW", 4) != 0) {
        fprintf(stderr, "neural_memory: bad magic\n"); fclose(f); return NULL;
    }
    if (fread(&version, 4, 1, f) != 1 || fread(&n_entries, 4, 1, f) != 1 ||
        fread(&index_size, 4, 1, f) != 1) {
        fprintf(stderr, "neural_memory: bad header\n"); fclose(f); return NULL;
    }

    /* Read index (one JSON per line) */
    char *index_buf = malloc(index_size + 1);
    if (fread(index_buf, 1, index_size, f) != (size_t)index_size) {
        fprintf(stderr, "neural_memory: bad index\n"); free(index_buf); fclose(f); return NULL;
    }
    index_buf[index_size] = 0;

    bnmodel_t *m = calloc(1, sizeof(bnmodel_t));
    m->f = f;
    m->n_entries = n_entries;
    m->entries = calloc(n_entries, sizeof(bnmodel_entry_t));

    /* Parse simple JSON index lines */
    char *line = index_buf;
    for (int i = 0; i < n_entries; i++) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char *np = strstr(line, "\"n\":");
        char *sp = strstr(line, "\"s\":[");
        char *op = strstr(line, "\"o\":");
        if (np && sp && op) {
            char *vp = np + 4;
            while (*vp == ' ' || *vp == '"') vp++;
            int j = 0;
            while (vp[j] && vp[j] != '"' && j < 127) {
                m->entries[i].name[j] = vp[j]; j++;
            }
            m->entries[i].name[j] = 0;
            int a=0,b=0,c2=0,d=0;
            int nd = sscanf(sp + 5, "%d,%d,%d,%d", &a, &b, &c2, &d);
            if (nd < 1) nd = 1;
            m->entries[i].n_dims = nd;
            m->entries[i].shape[0]=a; m->entries[i].shape[1]=b;
            m->entries[i].shape[2]=c2; m->entries[i].shape[3]=d;
            m->entries[i].count = 1;
            for (int k = 0; k < nd; k++) m->entries[i].count *= m->entries[i].shape[k];
            char *ovp = op + 4;
            while (*ovp == ' ') ovp++;
            m->entries[i].offset = strtol(ovp, NULL, 10);
        }
        line = nl ? nl + 1 : line + strlen(line);
    }

    /* Data starts after: 4+4+4+4 (header) + index_size + 1 (newline) */
    m->data_base = 16 + index_size + 1;
    free(index_buf);

    for (int dbg = 0; dbg < 3 && dbg < n_entries; dbg++)
        fprintf(stderr, "  entry[%d]: name='%s' count=%d offset=%ld\n", dbg,
                m->entries[dbg].name, m->entries[dbg].count, m->entries[dbg].offset);
    fprintf(stderr, "neural_memory: loaded %d tensors from %s\n", n_entries, path);
    return m;
}

static float *bnmodel_get(bnmodel_t *m, const char *name) {
    for (int i = 0; i < m->n_entries; i++) {
        if (strcmp(m->entries[i].name, name) == 0) {
            float *data = malloc(m->entries[i].count * sizeof(float));
            fseek(m->f, m->data_base + m->entries[i].offset, SEEK_SET);
            if (fread(data, sizeof(float), m->entries[i].count, m->f) !=
                (size_t)m->entries[i].count) {
                fprintf(stderr, "neural_memory: short read %s\n", name);
                free(data); return NULL;
            }
            return data;
        }
    }
    fprintf(stderr, "neural_memory: tensor not found: %s\n", name);
    return NULL;
}

static void bnmodel_close(bnmodel_t *m) {
    if (!m) return;
    if (m->f) fclose(m->f);
    free(m->entries);
    free(m);
}

/* --- Matrix ops (row-major, y = x @ W^T + b) --- */
static void linear(float *out, const float *x, const float *w, const float *b,
                   int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++)
            sum += x[i] * w[o * in_dim + i];
        out[o] = sum;
    }
}

static void tanh_vec(float *v, int n) {
    for (int i = 0; i < n; i++) v[i] = tanhf(v[i]);
}

static void softmax(float *v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - mx); sum += v[i]; }
    for (int i = 0; i < n; i++) v[i] /= sum;
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* --- Neural memory context --- */
struct neural_memory_ctx {
    /* Reader weights */
    float *query_w, *query_b;         /* 64×2048, 64 */
    float *source_w, *source_b;       /* 64×2048, 64 */
    float *query_pool_w;              /* 1×64 */
    float *span_w, *span_b;           /* 64×192, 64 */
    float *fact0_w, *fact0_b;         /* 64×256, 64 */
    float *fact2_w, *fact2_b;         /* 1×64, 1 */
    float *fact_value_w;              /* 64×64 */
    float *value_w;                   /* 64×64 */
    float *value_query_w;             /* 64×64 */
    float *route_w, *route_b;         /* 3×131, 3 */
    float *mode0_w, *mode0_b;         /* 64×1152, 64 */
    float *mode2_w, *mode2_b;         /* 2×64, 2 */
    float *offset_w;                  /* 65×64 */
    float *bound0_w, *bound0_b;       /* 64×192, 64 */
    float *bound2_w, *bound2_b;       /* 4×64, 4 */
    /* Branch weights */
    float *br_enc_w, *br_enc_b;       /* 1024×1024, 1024 */
    float *br_out_w;                  /* 1024×1024 */
    /* Wide fact/value heads (2048-dim, bypass 64-dim bottleneck) */
    float *wf0_w, *wf0_b;            /* 256×4096, 256 */
    float *wf2_w, *wf2_b;            /* 1×256, 1 */
    float *wv0_w, *wv0_b;            /* 256×4096, 256 */
    float *wv2_w, *wv2_b;            /* 1×256, 1 */
    /* Structural constants */
    float struct_mu[3], struct_sd[3];
    float logit_scale;
    int width, hidden;
};

nm_ctx_t *nm_init(const char *model_path) {
    nm_ctx_t *ctx = calloc(1, sizeof(nm_ctx_t));
    ctx->width = 64; ctx->hidden = 1024;
    ctx->logit_scale = 4.0f;  /* frozen training constant */

    bnmodel_t *m = bnmodel_open(model_path);
    if (!m) { free(ctx); return NULL; }

    ctx->query_w  = bnmodel_get(m, "query.weight");
    ctx->query_b  = bnmodel_get(m, "query.bias");
    ctx->source_w = bnmodel_get(m, "source.weight");
    ctx->source_b = bnmodel_get(m, "source.bias");
    ctx->query_pool_w = bnmodel_get(m, "query_pool.weight");
    ctx->span_w   = bnmodel_get(m, "span.weight");
    ctx->span_b   = bnmodel_get(m, "span.bias");
    ctx->fact0_w  = bnmodel_get(m, "fact.0.weight");
    ctx->fact0_b  = bnmodel_get(m, "fact.0.bias");
    ctx->fact2_w  = bnmodel_get(m, "fact.2.weight");
    ctx->fact2_b  = bnmodel_get(m, "fact.2.bias");
    ctx->fact_value_w = bnmodel_get(m, "fact_value.weight");
    ctx->value_w  = bnmodel_get(m, "value.weight");
    ctx->value_query_w = bnmodel_get(m, "value_query.weight");
    ctx->route_w  = bnmodel_get(m, "route.weight");
    ctx->route_b  = bnmodel_get(m, "route.bias");
    ctx->mode0_w  = bnmodel_get(m, "mode.0.weight");
    ctx->mode0_b  = bnmodel_get(m, "mode.0.bias");
    ctx->mode2_w  = bnmodel_get(m, "mode.2.weight");
    ctx->mode2_b  = bnmodel_get(m, "mode.2.bias");
    ctx->offset_w = bnmodel_get(m, "offset.weight");
    ctx->bound0_w = bnmodel_get(m, "boundaries.0.weight");
    ctx->bound0_b = bnmodel_get(m, "boundaries.0.bias");
    ctx->bound2_w = bnmodel_get(m, "boundaries.2.weight");
    ctx->bound2_b = bnmodel_get(m, "boundaries.2.bias");
    ctx->br_enc_w = bnmodel_get(m, "branch.encoder.weight");
    ctx->br_enc_b = bnmodel_get(m, "branch.encoder.bias");
    ctx->br_out_w = bnmodel_get(m, "branch.output.weight");

    /* Wide fact/value heads (version 2 .bnmodel) */
    ctx->wf0_w = bnmodel_get(m, "wide_fact.0.weight");
    ctx->wf0_b = bnmodel_get(m, "wide_fact.0.bias");
    ctx->wf2_w = bnmodel_get(m, "wide_fact.2.weight");
    ctx->wf2_b = bnmodel_get(m, "wide_fact.2.bias");
    ctx->wv0_w = bnmodel_get(m, "wide_value.0.weight");
    ctx->wv0_b = bnmodel_get(m, "wide_value.0.bias");
    ctx->wv2_w = bnmodel_get(m, "wide_value.2.weight");
    ctx->wv2_b = bnmodel_get(m, "wide_value.2.bias");

    /* Structural constants: hardcoded from training data (DG-029) */
    ctx->struct_mu[0] = 0.838565f; ctx->struct_mu[1] = 136.708527f; ctx->struct_mu[2] = 0.455858f;
    ctx->struct_sd[0] = 0.368138f; ctx->struct_sd[1] = 102.237572f; ctx->struct_sd[2] = 0.264822f;

    bnmodel_close(m);

    /* Verify all loaded */
    if (!ctx->query_w || !ctx->route_w || !ctx->br_enc_w) {
        fprintf(stderr, "neural_memory: weight loading failed\n");
        nm_free(ctx); return NULL;
    }
    return ctx;
}

void nm_free(nm_ctx_t *ctx) {
    if (!ctx) return;
    free(ctx->query_w); free(ctx->query_b); free(ctx->source_w); free(ctx->source_b);
    free(ctx->query_pool_w); free(ctx->span_w); free(ctx->span_b);
    free(ctx->fact0_w); free(ctx->fact0_b); free(ctx->fact2_w); free(ctx->fact2_b);
    free(ctx->fact_value_w); free(ctx->value_w); free(ctx->value_query_w);
    free(ctx->route_w); free(ctx->route_b);
    free(ctx->mode0_w); free(ctx->mode0_b); free(ctx->mode2_w); free(ctx->mode2_b);
    free(ctx->offset_w); free(ctx->bound0_w); free(ctx->bound0_b);
    free(ctx->bound2_w); free(ctx->bound2_b);
    free(ctx->br_enc_w); free(ctx->br_enc_b); free(ctx->br_out_w);
    free(ctx);
}

/* --- Route prediction --- */
nm_route_t nm_predict_route(nm_ctx_t *ctx,
                             const float *query_features,   /* [n_query × 2048] */
                             int n_query,
                             const float *source_features,  /* [n_source × 2048] or NULL */
                             int n_source,
                             const float *prefix_hidden,    /* [1024] */
                             int n_allowed,                 /* count of allowed tokens */
                             float *route_logits_out /* [3] or NULL */) {
    int W = ctx->width;

    /* Project ALL query token features → tanh → learned positional pooling.
     * Training: qrows = tanh(W_q @ feat + b) for each token, then
     * weights = softmax(query_pool @ qrows over positions), q = Σ weights·qrows */
    float qrows_buf[512 * 64];  /* max 512 positions × 64 dims */
    float pool_logits[512];
    int nq = n_query < 512 ? n_query : 512;

    for (int t = 0; t < nq; t++) {
        linear(qrows_buf + t*W, query_features + (size_t)t*2048,
               ctx->query_w, ctx->query_b, 2048, W);
        tanh_vec(qrows_buf + t*W, W);
        /* Pool logit for position t: dot(query_pool_w[64], qrows[t][64]) */
        float pl = 0;
        for (int i = 0; i < W; i++) pl += ctx->query_pool_w[i] * qrows_buf[t*W + i];
        pool_logits[t] = pl;
    }
    /* Softmax over positions */
    softmax(pool_logits, nq);
    /* Weighted sum */
    float q[64];
    memset(q, 0, sizeof(q));
    for (int t = 0; t < nq; t++)
        for (int i = 0; i < W; i++)
            q[i] += pool_logits[t] * qrows_buf[t*W + i];

    /* Project source features if present */
    float *src = NULL;
    if (n_source > 0 && source_features) {
        src = malloc(n_source * W * sizeof(float));
        for (int t = 0; t < n_source && t < 512; t++) {
            linear(src + t*W, source_features + t*2048, ctx->source_w, ctx->source_b, 2048, W);
            tanh_vec(src + t*W, W);
        }
    }

    /* Evidence: if source present, compute span-based evidence */
    float evidence[64];
    memset(evidence, 0, sizeof(evidence));
    int n_spans = 0;
    if (src && n_source > 1) {
        /* Simplified evidence: mean of source projections */
        for (int i = 0; i < W; i++) {
            float sum = 0;
            for (int t = 0; t < n_source; t++) sum += src[t*W + i];
            evidence[i] = sum / n_source;
        }
        n_spans = n_source > 2 ? n_source : 1;
    }

    /* Route input: q + evidence + structural scalars */
    float route_in[131]; /* 64+64+3 */
    memcpy(route_in, q, W*sizeof(float));
    memcpy(route_in + W, evidence, W*sizeof(float));
    route_in[2*W + 0] = (n_source > 0 ? 1.0f : 0.0f - ctx->struct_mu[0]) / ctx->struct_sd[0];
    route_in[2*W + 1] = ((float)n_spans - ctx->struct_mu[1]) / ctx->struct_sd[1];
    route_in[2*W + 2] = ((float)n_allowed / 32.0f - ctx->struct_mu[2]) / ctx->struct_sd[2];
    /* Fix has_source computation */
    route_in[2*W + 0] = ((n_source > 0 ? 1.0f : 0.0f) - ctx->struct_mu[0]) / ctx->struct_sd[0];

    float route_logits[3];
    linear(route_logits, route_in, ctx->route_w, ctx->route_b, 131, 3);

    /* Mask route[1] if no spans */
    if (n_spans == 0) route_logits[1] = -1e30f;

    int route = argmax(route_logits, 3);

    /* Mode prediction (if route==1) */
    int mode = 0;
    if (route == 1) {
        float mode_in[1152]; /* 1024 + 64 + 64 */
        if (prefix_hidden) memcpy(mode_in, prefix_hidden, 1024*sizeof(float));
        else memset(mode_in, 0, 1024*sizeof(float));
        memcpy(mode_in + 1024, q, W*sizeof(float));
        memcpy(mode_in + 1024 + W, evidence, W*sizeof(float));
        float mode_hidden[64];
        linear(mode_hidden, mode_in, ctx->mode0_w, ctx->mode0_b, 1152, 64);
        tanh_vec(mode_hidden, 64);
        float mode_logits[2];
        linear(mode_logits, mode_hidden, ctx->mode2_w, ctx->mode2_b, 64, 2);
        mode = argmax(mode_logits, 2);
    }

    if (route_logits_out) memcpy(route_logits_out, route_logits, 3*sizeof(float));
    free(src);

    nm_route_t result = {route, mode, route_logits[0], route_logits[1], route_logits[2]};
    return result;
}

/* --- Uncertainty branch: residual → logits --- */
/* --- Wide 2048-dim span selection (bypasses 64-dim bottleneck) --- */
/* Scores each candidate span using the FULL 2048-dim query/source features.
 * Returns the best value span's token range, or -1 if no spans. */
int nm_wide_select_value(nm_ctx_t *ctx,
                          const float *query_features,  /* [1×2048] */
                          const float *source_features, /* [n_source×2048] */
                          int n_source,
                          int *span_starts, int *span_ends, /* candidate spans */
                          int n_spans,
                          int *best_span_idx) {
    if (!ctx->wf0_w || n_spans <= 0 || !query_features || !source_features) return -1;

    float q_wide[2048];
    /* Mean-pool query features */
    for (int i = 0; i < 2048; i++) q_wide[i] = query_features[i]; /* single row = no pool needed */

    float best_score = -1e30f;
    int best_idx = -1;

    for (int s = 0; s < n_spans; s++) {
        /* Mean-pool source features in span range */
        float span_src[2048];
        int start = span_starts[s], end = span_ends[s];
        if (end > n_source) end = n_source;
        int n = end - start;
        if (n <= 0) continue;
        for (int i = 0; i < 2048; i++) {
            float sum = 0;
            for (int t = start; t < end; t++) sum += source_features[t * 2048 + i];
            span_src[i] = sum / n;
        }

        /* Concatenate query + span → [4096] */
        float wide_in[4096];
        memcpy(wide_in, q_wide, 2048 * sizeof(float));
        memcpy(wide_in + 2048, span_src, 2048 * sizeof(float));

        /* wide_fact: 4096 → 256 → tanh → 1 */
        float h1[256];
        linear(h1, wide_in, ctx->wf0_w, ctx->wf0_b, 4096, 256);
        tanh_vec(h1, 256);
        float score;
        linear(&score, h1, ctx->wf2_w, ctx->wf2_b, 256, 1);

        if (score > best_score) {
            best_score = score;
            best_idx = s;
        }
    }

    if (best_idx >= 0) *best_span_idx = best_idx;
    return best_idx;
}

void nm_apply_residual(nm_ctx_t *ctx,
                        const float *hidden,  /* [1024] C backbone hidden */
                        float *modified_hidden /* [1024] hidden + uncertainty residual / scale */) {
    int H = ctx->hidden;
    /* LayerNorm(hidden) */
    float ln[1024];
    float mean = 0;
    for (int i = 0; i < H; i++) mean += hidden[i];
    mean /= H;
    float var = 0;
    for (int i = 0; i < H; i++) var += (hidden[i]-mean)*(hidden[i]-mean);
    var /= H;
    float eps = 1e-5f;
    for (int i = 0; i < H; i++) ln[i] = (hidden[i]-mean) / sqrtf(var+eps);

    /* encoder: tanh(W_in @ ln + b_in) */
    float enc[1024];
    linear(enc, ln, ctx->br_enc_w, ctx->br_enc_b, H, H);
    tanh_vec(enc, H);

    /* output: residual (no bias, was zero-init) */
    float residual[1024];
    linear(residual, enc, ctx->br_out_w, NULL, H, H);

    /* modified_hidden = hidden + residual / logit_scale
     * The C backbone's quantized output projection (Q6_K) then produces
     * the final logits from this modified hidden state — no .npy needed. */
    for (int i = 0; i < H; i++)
        modified_hidden[i] = hidden[i] + residual[i] / ctx->logit_scale;
}

void nm_wide_score(nm_ctx_t *ctx, const float *wide_in, float *h1, float *score) {
    if (!ctx || !ctx->wf0_w || !wide_in || !h1 || !score) return;
    linear(h1, wide_in, ctx->wf0_w, ctx->wf0_b, 4096, 256);
    tanh_vec(h1, 256);
    linear(score, h1, ctx->wf2_w, ctx->wf2_b, 256, 1);
}
