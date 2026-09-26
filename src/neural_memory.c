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

static bnmodel_t *bnmodel_open(const char *path, int *version_out) {
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
    if (version < 2 || version > 3) {
        fprintf(stderr, "neural_memory: unsupported version %d\n", version);
        fclose(f); return NULL;
    }
    if (version_out) *version_out = version;

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

/* Exact-shape requirement for version-gated tensors (corruption guard). */
static int bnmodel_shape_is(bnmodel_t *m, const char *name, int dims, int r, int c) {
    for (int i = 0; i < m->n_entries; i++) {
        if (strcmp(m->entries[i].name, name) == 0) {
            if (m->entries[i].n_dims != dims) return 0;
            if (dims >= 1 && m->entries[i].shape[0] != r) return 0;
            if (dims >= 2 && m->entries[i].shape[1] != c) return 0;
            return 1;
        }
    }
    return 0;
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
    /* Wide fact/value heads (version 2 .bnmodel) */
    float *wf0_w, *wf0_b;            /* 256×4096, 256 */
    float *wf2_w, *wf2_b;            /* 1×256, 1 */
    float *wv0_w, *wv0_b;            /* 256×4096, 256 */
    float *wv2_w, *wv2_b;            /* 1×256, 1 */
    /* Episode joint-relevance head (EC-005, version 3) */
    float *jp_pair_w, *jp_pair_b;    /* 64×8192, 64 */
    float *jp_mix_w, *jp_mix_b;      /* 64×4160, 64 */
    float *jp_out_w, *jp_out_b;      /* 1×64, 1 */
    int has_ep_joint;
    /* Structural constants */
    float struct_mu[3], struct_sd[3];
    float logit_scale;
    int width, hidden;
};

nm_ctx_t *nm_init(const char *model_path) {
    nm_ctx_t *ctx = calloc(1, sizeof(nm_ctx_t));
    ctx->width = 64; ctx->hidden = 1024;
    ctx->logit_scale = 4.0f;  /* frozen training constant */

    int file_version = 0;
    bnmodel_t *m = bnmodel_open(model_path, &file_version);
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

    /* Episode joint-relevance head: optional at v2, required at v3 */
    ctx->jp_pair_w = bnmodel_get(m, "ep_joint.pair.weight");
    ctx->jp_pair_b = bnmodel_get(m, "ep_joint.pair.bias");
    ctx->jp_mix_w = bnmodel_get(m, "ep_joint.mix.weight");
    ctx->jp_mix_b = bnmodel_get(m, "ep_joint.mix.bias");
    ctx->jp_out_w = bnmodel_get(m, "ep_joint.out.weight");
    ctx->jp_out_b = bnmodel_get(m, "ep_joint.out.bias");
    ctx->has_ep_joint = (ctx->jp_pair_w && ctx->jp_pair_b && ctx->jp_mix_w &&
                         ctx->jp_mix_b && ctx->jp_out_w && ctx->jp_out_b) ? 1 : 0;
    if (file_version >= 3) {
        if (!ctx->has_ep_joint ||
            !bnmodel_shape_is(m, "ep_joint.pair.weight", 2, 64, 8192) ||
            !bnmodel_shape_is(m, "ep_joint.pair.bias", 1, 64, 0) ||
            !bnmodel_shape_is(m, "ep_joint.mix.weight", 2, 64, 4160) ||
            !bnmodel_shape_is(m, "ep_joint.mix.bias", 1, 64, 0) ||
            !bnmodel_shape_is(m, "ep_joint.out.weight", 2, 1, 64) ||
            !bnmodel_shape_is(m, "ep_joint.out.bias", 1, 1, 0)) {
            fprintf(stderr, "neural_memory: corrupt ep_joint tensors at version %d\n",
                    file_version);
            bnmodel_close(m);
            nm_free(ctx);
            return NULL;
        }
    }

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
    free(ctx->wf0_w); free(ctx->wf0_b); free(ctx->wf2_w); free(ctx->wf2_b);
    free(ctx->wv0_w); free(ctx->wv0_b); free(ctx->wv2_w); free(ctx->wv2_b);
    free(ctx->jp_pair_w); free(ctx->jp_pair_b); free(ctx->jp_mix_w);
    free(ctx->jp_mix_b); free(ctx->jp_out_w); free(ctx->jp_out_b);
    free(ctx);
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

/* --- Episode joint relevance (EC-005): per-pair joint features, max agg ---
 * f_ij = tanh(W_pair [q_i; e_j; q_i*e_j; |q_i - e_j|] + b)   (64)
 * agg  = per-dim max over pairs; mix with means; sigmoid. */
int nm_has_episode_joint(const nm_ctx_t *ctx) {
    return (ctx && ctx->has_ep_joint) ? 1 : 0;
}

float nm_episode_joint_relevance(nm_ctx_t *ctx,
                                 const float *q_rows, int n_query,
                                 const float *q_mean,
                                 const float *ep_rows, int n_ep_rows,
                                 const float *ep_mean) {
    if (!ctx || !ctx->has_ep_joint || !q_rows || n_query <= 0 ||
        !ep_rows || n_ep_rows <= 0 || !q_mean || !ep_mean)
        return -1.0f;

    float agg[64];
    for (int d = 0; d < 64; d++) agg[d] = -1e30f;

    float pair_in[8192];
    float f[64];
    for (int i = 0; i < n_query && i < 512; i++) {
        const float *q = q_rows + (size_t)i * 2048;
        memcpy(pair_in, q, 2048 * sizeof(float));
        for (int j = 0; j < n_ep_rows && j < 512; j++) {
            const float *e = ep_rows + (size_t)j * 2048;
            memcpy(pair_in + 2048, e, 2048 * sizeof(float));
            for (int d = 0; d < 2048; d++) {
                float a = q[d], b = e[d];
                pair_in[4096 + d] = a * b;
                pair_in[6144 + d] = fabsf(a - b);
            }
            linear(f, pair_in, ctx->jp_pair_w, ctx->jp_pair_b, 8192, 64);
            tanh_vec(f, 64);
            for (int d = 0; d < 64; d++)
                if (f[d] > agg[d]) agg[d] = f[d];
        }
    }

    float mix_in[4160];
    memcpy(mix_in, agg, 64 * sizeof(float));
    memcpy(mix_in + 64, q_mean, 2048 * sizeof(float));
    memcpy(mix_in + 64 + 2048, ep_mean, 2048 * sizeof(float));
    float h[64], logit;
    linear(h, mix_in, ctx->jp_mix_w, ctx->jp_mix_b, 4160, 64);
    tanh_vec(h, 64);
    linear(&logit, h, ctx->jp_out_w, ctx->jp_out_b, 64, 1);
    return 1.0f / (1.0f + expf(-logit));
}

/* --- Training-exact route path (route parity, LC-002) --- */

/* Escape a string for a JSON string literal like Python's json.dumps with
 * ensure_ascii=False: quote, backslash, and C0 controls; non-ASCII raw. */
static void nm_json_escape(const char *in, char *out, int out_size) {
    int o = 0;
    for (const unsigned char *p = (const unsigned char *)in;
         *p && o + 8 < out_size; p++) {
        switch (*p) {
        case '"':  out[o++] = '\\'; out[o++] = '"';  break;
        case '\\': out[o++] = '\\'; out[o++] = '\\'; break;
        case '\n': out[o++] = '\\'; out[o++] = 'n';  break;
        case '\t': out[o++] = '\\'; out[o++] = 't';  break;
        case '\r': out[o++] = '\\'; out[o++] = 'r';  break;
        case '\b': out[o++] = '\\'; out[o++] = 'b';  break;
        case '\f': out[o++] = '\\'; out[o++] = 'f';  break;
        default:
            if (*p < 0x20) o += snprintf(out + o, 7, "\\u%04x", *p);
            else out[o++] = (char)*p;
        }
    }
    out[o] = 0;
}

void nm_frame_text(const char *text, char *out, int out_size) {
    char esc[4096];
    nm_json_escape(text, esc, sizeof esc);
    snprintf(out, (size_t)out_size,
             "{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"%s\"}", esc);
}

void nm_frame_query(const char *text, char *out, int out_size) {
    char esc[4096];
    nm_json_escape(text, esc, sizeof esc);
    snprintf(out, (size_t)out_size,
             "[{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"%s\"}]", esc);
}

static void frame_dyn(const char *text, char **buf, size_t *cap,
                      const char *prefix, const char *suffix) {
    /* worst case: 6x escaped text + prefix + suffix + NUL */
    size_t worst = 6 * strlen(text) + strlen(prefix) + strlen(suffix) + 64;
    if (*cap < worst) {
        size_t new_cap = *cap ? *cap : 256;
        while (new_cap < worst) new_cap *= 2;
        char *grown = realloc(*buf, new_cap);
        if (!grown) { *buf = NULL; return; }
        *buf = grown;
        *cap = new_cap;
    }
    size_t prefix_len = strlen(prefix);
    memcpy(*buf, prefix, prefix_len);
    size_t o = prefix_len;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        switch (*p) {
        case '"':  (*buf)[o++] = '\\'; (*buf)[o++] = '"';  break;
        case '\\': (*buf)[o++] = '\\'; (*buf)[o++] = '\\'; break;
        case '\n': (*buf)[o++] = '\\'; (*buf)[o++] = 'n';  break;
        case '\t': (*buf)[o++] = '\\'; (*buf)[o++] = 't';  break;
        case '\r': (*buf)[o++] = '\\'; (*buf)[o++] = 'r';  break;
        case '\b': (*buf)[o++] = '\\'; (*buf)[o++] = 'b';  break;
        case '\f': (*buf)[o++] = '\\'; (*buf)[o++] = 'f';  break;
        default:
            if (*p < 0x20) o += (size_t)snprintf(*buf + o, 7, "\\u%04x", *p);
            else (*buf)[o++] = (char)*p;
        }
    }
    size_t suffix_len = strlen(suffix);
    memcpy(*buf + o, suffix, suffix_len);
    (*buf)[o + suffix_len] = 0;
}

void nm_frame_text_dyn(const char *text, char **buf, size_t *cap) {
    if (!buf || !cap) return;
    frame_dyn(text, buf, cap,
              "{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"", "\"}");
}

void nm_frame_query_dyn(const char *text, char **buf, size_t *cap) {
    if (!buf || !cap) return;
    frame_dyn(text, buf, cap,
              "[{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"", "\"}]");
}

int nm_build_spans(const char *framed, const char *episode_text,
                   char **pieces, int n_pieces,
                   int *allowed_out, int *span_start_out, int *span_end_out,
                   int max_spans) {
    if (!framed || !episode_text || !pieces || n_pieces <= 0) return -1;
    size_t flen = strlen(framed);

    char esc[4096];
    nm_json_escape(episode_text, esc, sizeof esc);
    int escaped = strcmp(esc, episode_text) != 0;

    /* pieces must reconstruct the frame (optionally with the BOS space) */
    size_t total = 1;
    for (int i = 0; i < n_pieces; i++) total += strlen(pieces[i]);
    char *decoded = malloc(total + 1);
    if (!decoded) return -1;
    size_t o = 0;
    for (int i = 0; i < n_pieces; i++) {
        size_t pl = strlen(pieces[i]);
        memcpy(decoded + o, pieces[i], pl);
        o += pl;
    }
    decoded[o] = 0;
    /* pieces reconstruct the frame (optionally with the BOS space), or a
     * PREFIX of it when the feature window (512 tokens) truncated a long
     * episode — the layout then covers only the window. */
    const char *dec_body = (decoded[0] == ' ') ? decoded + 1 : decoded;
    size_t body_len = strlen(dec_body);
    int pad = (dec_body != decoded) ? 1 : 0;
    int truncated;
    if (strcmp(dec_body, framed) == 0) {
        truncated = 0;
    } else if (strlen(framed) > body_len &&
               strncmp(framed, dec_body, body_len) == 0) {
        truncated = 1;
    } else {
        free(decoded);
        return -2;  /* decode mismatch */
    }

    char literal[4200];
    snprintf(literal, sizeof literal, "\"%s\"", esc);
    size_t lit_len = strlen(literal);
    static const char FRAME_PREFIX[] =
        "{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"";
    size_t start, end;
    if (!truncated) {
        if (flen < lit_len + 1 ||
            strncmp(framed + flen - lit_len - 1, literal, lit_len) != 0 ||
            framed[flen - 1] != '}') {
            free(decoded);
            return -3;  /* literal bound mismatch */
        }
        /* text bytes span [start, end) in decoded coordinates */
        start = flen - lit_len + pad;
        end = flen - 2 + pad;
    } else {
        if (strncmp(framed, FRAME_PREFIX, sizeof FRAME_PREFIX - 1) != 0) {
            free(decoded);
            return -3;
        }
        /* window covers the frame from its start: text begins after the
         * frame prefix and runs to the end of the window */
        start = sizeof FRAME_PREFIX - 1 + pad;
        end = strlen(decoded);
    }

    /* escaped payloads have no addressable tokens (future byte channel) */
    if (escaped) {
        for (int i = 0; i < n_pieces; i++) allowed_out[i] = 0;
        free(decoded);
        return 0;
    }

    /* allowed mask + per-token byte ranges (relative to the opening quote,
     * exactly like source_alignment) */
    long *range_s = malloc((size_t)n_pieces * sizeof(long));
    long *range_e = malloc((size_t)n_pieces * sizeof(long));
    if (!range_s || !range_e) { free(decoded); free(range_s); free(range_e); return -1; }
    size_t offset = 0;
    for (int i = 0; i < n_pieces; i++) {
        size_t pl = strlen(pieces[i]);
        size_t stop = offset + pl;
        const char *pc = pieces[i];
        int control = pl >= 4 && pc[0] == '<' && pc[1] == '|' &&
                      pc[pl - 1] == '>' && pc[pl - 2] == '|';
        int ok = pl > 0 && offset >= start && stop <= end && !control;
        allowed_out[i] = ok;
        range_s[i] = ok ? (long)(offset - start) : -1;
        range_e[i] = ok ? (long)(stop - start) : -1;
        offset = stop;
    }
    free(decoded);

    /* UTF-8 character boundaries of the episode text (ByteLayout.cuts) */
    int cuts[1100];
    int n_cuts = 1;
    cuts[0] = 0;
    size_t tpos = 0;
    while (episode_text[tpos] && n_cuts < 1100) {
        unsigned char c = (unsigned char)episode_text[tpos];
        int clen = 1;
        if ((c & 0xE0) == 0xC0) clen = 2;
        else if ((c & 0xF0) == 0xE0) clen = 3;
        else if ((c & 0xF8) == 0xF0) clen = 4;
        tpos += (size_t)clen;
        cuts[n_cuts++] = (int)tpos;
    }

    /* first/last cut inside each allowed token's byte range */
    int *first_cut = malloc((size_t)n_pieces * sizeof(int));
    int *last_cut = malloc((size_t)n_pieces * sizeof(int));
    if (!first_cut || !last_cut) {
        free(range_s); free(range_e); free(first_cut); free(last_cut);
        return -1;
    }
    for (int i = 0; i < n_pieces; i++) {
        first_cut[i] = -1;
        last_cut[i] = -1;
        if (!allowed_out[i]) continue;
        for (int c = 0; c < n_cuts; c++) {
            int p = cuts[c];
            if (p >= range_s[i] && p < range_e[i] && first_cut[i] < 0)
                first_cut[i] = p;
            if (p > range_s[i] && p <= range_e[i])
                last_cut[i] = p;
        }
    }
    free(range_s); free(range_e);

    /* candidate_spans(allowed, max_span=16) + endpoint filter */
    int n_spans = 0;
    for (int s = 0; s < n_pieces; s++) {
        for (int e = s + 1; e <= n_pieces && e <= s + 16; e++) {
            if (!allowed_out[e - 1]) break;
            if (first_cut[s] < 0 || last_cut[e - 1] < 0) continue;
            if (first_cut[s] >= last_cut[e - 1]) continue;
            if (n_spans < max_spans) {
                span_start_out[n_spans] = s;
                span_end_out[n_spans] = e;
                n_spans++;
            }
        }
    }
    free(first_cut); free(last_cut);
    return n_spans;
}

nm_route_t nm_route_decision(nm_ctx_t *ctx,
                             const float *query_features, int n_query,
                             const float *source_features, int n_source,
                             const int *allowed, int n_allowed,
                             const int *span_starts, const int *span_ends,
                             int n_spans) {
    nm_route_t result = {0, 0, 0.0f, 0.0f, 0.0f};
    if (!ctx || !query_features || n_query <= 0) return result;
    int W = ctx->width;

    /* q = softmax(query_pool(qrows)) · qrows */
    float qrows_buf[512 * 64];
    float pool_logits[512];
    int nq = n_query < 512 ? n_query : 512;
    for (int t = 0; t < nq; t++) {
        linear(qrows_buf + t * W, query_features + (size_t)t * 2048,
               ctx->query_w, ctx->query_b, 2048, W);
        tanh_vec(qrows_buf + t * W, W);
        float pl = 0;
        for (int i = 0; i < W; i++) pl += ctx->query_pool_w[i] * qrows_buf[t * W + i];
        pool_logits[t] = pl;
    }
    softmax(pool_logits, nq);
    float q[64];
    memset(q, 0, sizeof q);
    for (int t = 0; t < nq; t++)
        for (int i = 0; i < W; i++)
            q[i] += pool_logits[t] * qrows_buf[t * W + i];

    /* source rows + prefix sums */
    int ns = n_source > 0 ? (n_source < 512 ? n_source : 512) : 0;
    float *src = NULL;
    double *sums = NULL;
    if (ns > 0 && source_features) {
        src = malloc((size_t)ns * W * sizeof(float));
        sums = malloc(((size_t)ns + 1) * W * sizeof(double));
        if (!src || !sums) { free(src); free(sums); return result; }
        for (int t = 0; t < ns; t++) {
            linear(src + t * W, source_features + (size_t)t * 2048,
                   ctx->source_w, ctx->source_b, 2048, W);
            tanh_vec(src + t * W, W);
        }
        for (int i = 0; i < W; i++) sums[i] = 0.0;
        for (int t = 0; t < ns; t++)
            for (int i = 0; i < W; i++)
                sums[(size_t)(t + 1) * W + i] = sums[(size_t)t * W + i] + src[t * W + i];
    }

    /* spans → span representations → fact scores → evidence */
    float evidence[64];
    memset(evidence, 0, sizeof evidence);
    float *h = NULL, *fs = NULL;
    if (n_spans > 0 && src) {
        h = malloc((size_t)n_spans * W * sizeof(float));
        fs = malloc((size_t)n_spans * sizeof(float));
        if (!h || !fs) { free(h); free(fs); free(src); free(sums); return result; }
        for (int i = 0; i < n_spans; i++) {
            int s = span_starts[i], e = span_ends[i];
            if (e > ns) e = ns;
            int len = e - s;
            if (len <= 0 || s >= ns) { len = 1; }
            float span_in[192];
            memcpy(span_in, src + (size_t)s * W, W * sizeof(float));
            memcpy(span_in + W, src + (size_t)(e - 1) * W, W * sizeof(float));
            for (int d = 0; d < W; d++)
                span_in[2 * W + d] =
                    (float)((sums[(size_t)e * W + d] - sums[(size_t)s * W + d]) / len);
            float hv[64];
            linear(hv, span_in, ctx->span_w, ctx->span_b, 3 * W, W);
            tanh_vec(hv, W);
            memcpy(h + (size_t)i * W, hv, W * sizeof(float));
        }
        for (int i = 0; i < n_spans; i++) {
            float fact_in[256];
            for (int d = 0; d < W; d++) {
                float qv = q[d], hv = h[(size_t)i * W + d];
                fact_in[d] = qv;
                fact_in[W + d] = hv;
                fact_in[2 * W + d] = qv * hv;
                fact_in[3 * W + d] = fabsf(qv - hv);
            }
            float fh[64];
            linear(fh, fact_in, ctx->fact0_w, ctx->fact0_b, 4 * W, W);
            tanh_vec(fh, W);
            linear(&fs[i], fh, ctx->fact2_w, ctx->fact2_b, W, 1);
        }
        /* evidence = softmax(fact scores) · h */
        float *pw = malloc((size_t)n_spans * sizeof(float));
        if (pw) {
            memcpy(pw, fs, (size_t)n_spans * sizeof(float));
            softmax(pw, n_spans);
            for (int i = 0; i < n_spans; i++)
                for (int d = 0; d < W; d++)
                    evidence[d] += pw[i] * h[(size_t)i * W + d];
            free(pw);
        }
    }

    /* structural scalars (mu/sd hardcoded from training, DG-029) */
    int allowed_sum = 0;
    for (int i = 0; i < n_allowed && allowed; i++) allowed_sum += allowed[i] ? 1 : 0;
    float route_in[131];
    memcpy(route_in, q, W * sizeof(float));
    memcpy(route_in + W, evidence, W * sizeof(float));
    route_in[2 * W + 0] = ((ns > 0 ? 1.0f : 0.0f) - ctx->struct_mu[0]) / ctx->struct_sd[0];
    route_in[2 * W + 1] = ((float)n_spans - ctx->struct_mu[1]) / ctx->struct_sd[1];
    route_in[2 * W + 2] = ((float)allowed_sum / 32.0f - ctx->struct_mu[2]) / ctx->struct_sd[2];

    float route_logits[3];
    linear(route_logits, route_in, ctx->route_w, ctx->route_b, 2 * W + 3, 3);
    if (n_spans == 0 || !src) route_logits[1] = -1e30f;

    free(src); free(sums); free(h); free(fs);

    result.route = argmax(route_logits, 3);
    result.mode = 0;
    result.logit_normal = route_logits[0];
    result.logit_supported = route_logits[1];
    result.logit_insufficient = route_logits[2];
    return result;
}
