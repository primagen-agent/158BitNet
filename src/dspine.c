/* DSpine block-draft model (tencent/Qwen3.8-27B-DSpine-b16) as the drafter
 * for speculative decoding of the Ternary-Bonsai 27B target.
 *
 * One round drafts a block of `block_size` tokens in a single pass of a
 * 5-layer Qwen3 backbone that attends to K/V projected from the target's own
 * residual stream.  A per-layer code bridge (a 1024-wide "code" written back
 * into the residual stream) carries state across the block, and a refinement
 * tail resolves the last write against the target's LM head over a 16-token
 * candidate set.
 *
 * Reference: dspine_mlx/{model,bridge,propose}.py (DSpine-main).  The code
 * below follows it op for op; float32 replaces the reference's BF16 rounding
 * points, which only removes noise. */

#include "dspine.h"
#include "bonsai.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define DS_NEON 1
#endif

#include "quant_ptq1_0.h"

#define DS_MAX_L   8
#define DS_MAX_BLK 16

/* One matrix: either bf16 (w) or the PTQ1_0 form of it (q), which is what
 * the chessboard kernel wants: 28 bytes per 128 weights against 256. */
typedef struct {
    const uint16_t *w;
    int n_out, n_in;
    const uint8_t *q;   /* PTQ1_0 rows, NULL for the bf16 path */
    uint8_t *p2;        /* the same rows 2-bit repacked for i8mm, or NULL */
    float *p2d;         /* their scales, 2*nb floats per row pair */
} ds_w;

typedef struct {
    ds_w q, k, v, o, gate, up, down;
    float *in_norm, *post_norm, *q_norm, *k_norm;
} ds_layer;

struct dspine_model {
    void *map;
    size_t map_len;
    char *hdr_buf;
    const char *hdr;
    size_t hdr_len;
    size_t data_off;

    bonsai_model_t *tgt;
    bonsai_context_t *ctx;
    void *pool;
    int nthreads;

    ds_w fc;
    float *hidden_norm;
    float *norm;
    ds_layer L[DS_MAX_L];
    int nl;

    const uint16_t *codebook;    /* [V][C] */
    const uint16_t *early_read;  /* [nl-1][C][H] */
    const uint16_t *read;        /* [C][H] */
    const uint16_t *feat_write;  /* [nl][H][R] */
    const uint16_t *gate_w;      /* [R][C] */
    const uint16_t *gate_b;      /* [R]   (bf16) */
    const uint16_t *pred_gate;   /* [R][C] */
    const uint16_t *prev_w;      /* [R][C] */

    int H, nff, nh, nkv, hd, C, R, V, concat, ntap;
    int block_size, mask_id, topk, window;
    float eps, rope_theta;
    float *rope_freq;
    float *gate_bf;              /* gate.bias as float32 [R] */
    float *fw_f;                 /* feature_write of the last layer, float32 [H][R] */
    float *pg_f, *pv_f;          /* predecessor gate / prev, float32 [R][C] */

    int cap, ctx_len;
    float *ck[DS_MAX_L], *cv[DS_MAX_L];

    /* scratch (DS_MAX_BLK rows) */
    float *x, *xn, *ab, *raw, *dbuf;
    float *qb, *kb, *vb, *ob;
    float *gb, *ub;
    float *codes, *projected, *qpack, *spack, *vpack, *msg;
    float *rms;
    float *libuf;
    float *hrow, *uhat, *tg, *mm, *tv;
    float *logits;
    float *ascore;
    int anchor_cur;
    int drafts_cur[DS_MAX_BLK];
    int cand[DS_MAX_BLK][16];
    float base_[DS_MAX_BLK][16];
    float score[16][16];
    int idx[16][16];

    double t_block, t_tail;
    double t_mv, t_attn;         /* draft block: matmul and attention shares */

    /* PTQ1_0 form of the bound matrices, plus the int8 form of one activation
     * block.  The draft reads 4.05 GB a round as bf16 and 0.5 GB this way. */
    void **qmem;
    int nqmem, cqmem;
    int8_t *qact;
    float *qasc;
    long qact_cap;
    void *p2prep;          /* activation tiles + block sums for the i8mm matmat */
    size_t p2prep_size;
    int nqmat;             /* how many weight matrices went through PTQ1_0 */
    long p2_bytes;
    long q_bytes;
};

/* ------------------------------------------------------------------ */
/* float helpers                                                      */
/* ------------------------------------------------------------------ */

static inline float ds_bf16(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static float ds_dot_bf16(const uint16_t *w, const float *x, int n) {
#if defined(DS_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    float s;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + i), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + i + 4), 16));
        float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + i + 8), 16));
        float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + i + 12), 16));
        a0 = vfmaq_f32(a0, w0, vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, w1, vld1q_f32(x + i + 4));
        a2 = vfmaq_f32(a2, w2, vld1q_f32(x + i + 8));
        a3 = vfmaq_f32(a3, w3, vld1q_f32(x + i + 12));
    }
    s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; ++i) s += ds_bf16(w[i]) * x[i];
    return s;
#else
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        acc0 += ds_bf16(w[i + 0]) * x[i + 0];
        acc1 += ds_bf16(w[i + 1]) * x[i + 1];
        acc2 += ds_bf16(w[i + 2]) * x[i + 2];
        acc3 += ds_bf16(w[i + 3]) * x[i + 3];
    }
    for (; i < n; ++i) acc0 += ds_bf16(w[i]) * x[i];
    return (acc0 + acc1) + (acc2 + acc3);
#endif
}

static void ds_rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double s = 0.0;
    float sc;
    int i;
    for (i = 0; i < n; ++i) s += (double)x[i] * x[i];
    sc = 1.0f / sqrtf((float)(s / n) + eps);
    for (i = 0; i < n; ++i) out[i] = x[i] * sc * w[i];
}

static void ds_rope(float *v, int n_heads, int head_dim, int pos, const float *freq) {
    const int half = head_dim / 2;
    int h, i;
    for (h = 0; h < n_heads; ++h) {
        float *p = v + (size_t)h * head_dim;
        for (i = 0; i < half; ++i) {
            float th = (float)pos * freq[i];
            float cs = cosf(th), sn = sinf(th);
            float x0 = p[i], x1 = p[i + half];
            p[i] = x0 * cs - x1 * sn;
            p[i + half] = x0 * sn + x1 * cs;
        }
    }
}

static float ds_silu(float x) { return x / (1.0f + expf(-x)); }
static float ds_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ------------------------------------------------------------------ */
/* batched bf16 matvec: y[b][j] = W[j] . x[b]                         */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint16_t *w;
    const uint8_t *q;      /* PTQ1_0 rows, NULL for the bf16 path */
    size_t qb;             /* bytes per PTQ1_0 row */
    const int8_t *xq;      /* int8 activations, one row per position */
    const float *xs;       /* one scale per position */
    const uint8_t *p2;     /* repacked row pairs, when the i8mm path is usable */
    const float *p2d;
    const void *pprep;     /* prepared activation tiles for those pairs */
    int n_out, n_in;
    const float *x;
    float *y;
    int B, next;
} ds_mm_t;

/* One weight row against up to four activation rows.  The row is decoded once
 * per block of four positions instead of once per position: at B=8 the row
 * used to be read and expanded eight times, which turned a pass that should
 * be bound by the 3.3 GB of draft weights into a load-bound one (375 ms a
 * round against the 200 ms the stream alone costs). */
static void ds_mm_row4(const uint16_t *r, const float *x, int B, int n,
                       float *y, int n_out, int row) {
#if defined(DS_NEON)
    int b0, i;
    for (b0 = 0; b0 < B; b0 += 4) {
        if (b0 + 3 < B) {
            const float *x0 = x + (size_t)(b0 + 0) * n;
            const float *x1 = x + (size_t)(b0 + 1) * n;
            const float *x2 = x + (size_t)(b0 + 2) * n;
            const float *x3 = x + (size_t)(b0 + 3) * n;
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (i = 0; i + 4 <= n; i += 4) {
                const float32x4_t w =
                    vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(r + i), 16));
                a0 = vfmaq_f32(a0, w, vld1q_f32(x0 + i));
                a1 = vfmaq_f32(a1, w, vld1q_f32(x1 + i));
                a2 = vfmaq_f32(a2, w, vld1q_f32(x2 + i));
                a3 = vfmaq_f32(a3, w, vld1q_f32(x3 + i));
            }
            s0 = vaddvq_f32(a0); s1 = vaddvq_f32(a1);
            s2 = vaddvq_f32(a2); s3 = vaddvq_f32(a3);
            for (; i < n; ++i) {
                const float w = ds_bf16(r[i]);
                s0 += w * x0[i]; s1 += w * x1[i];
                s2 += w * x2[i]; s3 += w * x3[i];
            }
            y[(size_t)(b0 + 0) * n_out + row] = s0;
            y[(size_t)(b0 + 1) * n_out + row] = s1;
            y[(size_t)(b0 + 2) * n_out + row] = s2;
            y[(size_t)(b0 + 3) * n_out + row] = s3;
        } else {
            int b;
            for (b = b0; b < B; ++b)
                y[(size_t)b * n_out + row] = ds_dot_bf16(r, x + (size_t)b * n, n);
        }
    }
#else
    int b;
    for (b = 0; b < B; ++b)
        y[(size_t)b * n_out + row] = ds_dot_bf16(r, x + (size_t)b * n, n);
#endif
}

static void ds_mm_job(void *p, int tid, int nthreads) {
    ds_mm_t *a = (ds_mm_t *)p;
    (void)tid;
    (void)nthreads;
    for (;;) {
        int st = __atomic_fetch_add(&a->next, 4, __ATOMIC_RELAXED);
        int en = st + 4, j;
        if (st >= a->n_out) break;
        if (en > a->n_out) en = a->n_out;
        j = st;
        if (a->p2 != NULL) {
            const size_t pb = bitnet_ptq1_0_p2_pair_bytes(a->n_in);
            const size_t nb = (size_t)(a->n_in / 128);
            if (a->B <= 2)
                for (; j + 3 < en; j += 4)
                    bitnet_ptq1_0_matmat_p2q(a->p2 + (size_t)(j / 2) * pb,
                                             a->p2d + (size_t)(j / 2) * nb * 2u,
                                             a->pprep, a->xs, a->B, a->n_out, j, a->y);
            for (; j + 1 < en; j += 2)
                bitnet_ptq1_0_matmat_p2(a->p2 + (size_t)(j / 2) * pb,
                                        a->p2d + (size_t)(j / 2) * nb * 2u,
                                        a->pprep, a->xs, a->B, a->n_out, j, a->y);
        }
        for (; j < en; ++j) {
            if (a->q != NULL) {
                float tmp[DS_MAX_BLK];
                int b2;
                bitnet_ptq1_0_matmat_row(a->q + (size_t)j * a->qb, a->xq, a->xs,
                                         a->n_in, a->B, tmp);
                for (b2 = 0; b2 < a->B; ++b2)
                    a->y[(size_t)b2 * a->n_out + j] = tmp[b2];
            } else {
                ds_mm_row4(a->w + (size_t)j * a->n_in, a->x, a->B, a->n_in,
                           a->y, a->n_out, j);
            }
        }
    }
}

/* Quantize all B activation rows once; every PTQ1_0 matrix of a layer reads
 * the same block. */
static const int8_t *ds_qact(dspine_model_t *d, int B, int n, const float *x) {
    long need = (long)B * (long)n;
    int b;
    if (need > d->qact_cap) {
        int8_t *p = (int8_t *)realloc(d->qact, (size_t)need);
        if (p == NULL) return NULL;
        d->qact = p;
        d->qact_cap = need;
    }
    if (d->qasc == NULL && (d->qasc = (float *)malloc(DS_MAX_BLK * sizeof(float))) == NULL)
        return NULL;
    for (b = 0; b < B; ++b)
        bitnet_quantize_row_i8(x + (size_t)b * n, d->qact + (size_t)b * n,
                               &d->qasc[b], n);
    return d->qact;
}

static double ds_now(void);

static void ds_mv(dspine_model_t *d, const ds_w *w, const float *x, int B, float *y) {
    ds_mm_t a;
    const double tm0 = ds_now();
    const int8_t *xq = NULL;
    if (w->q != NULL && B >= 1 && B <= DS_MAX_BLK) xq = ds_qact(d, B, w->n_in, x);
    a.w = w->w;
    a.q = xq != NULL ? w->q : NULL;
    a.p2 = NULL;
    a.p2d = NULL;
    a.pprep = NULL;
    if (a.q != NULL && w->p2 != NULL) {
        size_t need = bitnet_ptq1_0_p2_prep_bytes(w->n_in, B);
        if (d->p2prep == NULL || d->p2prep_size < need) {
            void *np = realloc(d->p2prep, need);
            if (np != NULL) { d->p2prep = np; d->p2prep_size = need; }
        }
        if (d->p2prep != NULL) {
            bitnet_ptq1_0_p2_prep(d->p2prep, xq, w->n_in, B);
            a.p2 = w->p2;
            a.p2d = w->p2d;
            a.pprep = d->p2prep;
        }
    }
    a.qb = bitnet_ptq1_0_row_size(w->n_in);
    a.xq = xq;
    a.xs = d->qasc;
    a.n_out = w->n_out;
    a.n_in = w->n_in;
    a.x = x;
    a.y = y;
    a.B = B;
    a.next = 0;
    if (w->n_out >= 256) bonsai_pool_run(d->pool, ds_mm_job, &a);
    else ds_mm_job(&a, 0, 1);
    d->t_mv += ds_now() - tm0;
}

/* ------------------------------------------------------------------ */
/* safetensors header scan                                            */
/* ------------------------------------------------------------------ */

static const char *ds_object(dspine_model_t *d, const char *name) {
    char pat[160];
    const char *p, *q, *end;
    snprintf(pat, sizeof(pat), "\"%s\"", name);
    p = strstr(d->hdr, pat);
    if (p == NULL) return NULL;
    /* the value must be an object, and the object ends at the matching '}' */
    q = strchr(p, '{');
    if (q == NULL) return NULL;
    end = strchr(q, '}');
    if (end == NULL) return NULL;
    if (end < q) return NULL;
    return q;
}

static const char *ds_field(const char *obj, const char *key) {
    char pat[64];
    const char *q;
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    q = strstr(obj, pat);
    if (q == NULL) return NULL;
    q += strlen(pat);
    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') ++q;
    if (*q != ':') return NULL;
    ++q;
    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') ++q;
    return q;
}

static int ds_int_list(const char *p, long long *out, int max, int *n) {
    int k = 0;
    if (p == NULL || *p != '[') return -1;
    ++p;
    for (;;) {
        while (*p == ' ') ++p;
        if (*p == ']') break;
        {
            char *endp = NULL;
            long long v = strtoll(p, &endp, 10);
            if (endp == p || k >= max) return -1;
            out[k++] = v;
            p = endp;
        }
        while (*p == ' ') ++p;
        if (*p == ',') { ++p; continue; }
        if (*p == ']') break;
        return -1;
    }
    *n = k;
    return 0;
}

/* Fetches a tensor and checks its shape; returns a pointer to its bf16 data. */
static const uint16_t *ds_tensor(dspine_model_t *d, const char *name,
                                 const int *shape, int nd) {
    const char *obj = ds_object(d, name);
    const char *dt;
    long long got[8], off[8];
    int ng = 0, no = 0, i;
    size_t want = 2;
    if (obj == NULL) {
        fprintf(stderr, "dspine: missing tensor %s\n", name);
        return NULL;
    }
    dt = ds_field(obj, "dtype");
    if (dt == NULL || strncmp(dt, "\"BF16\"", 6) != 0) {
        fprintf(stderr, "dspine: %s is not BF16\n", name);
        return NULL;
    }
    if (ds_int_list(ds_field(obj, "shape"), got, 8, &ng) != 0 || ng != nd) {
        fprintf(stderr, "dspine: %s has an unexpected rank\n", name);
        return NULL;
    }
    for (i = 0; i < nd; ++i) {
        if (got[i] != (long long)shape[i]) {
            fprintf(stderr, "dspine: %s shape mismatch (dim %d: %lld vs %d)\n",
                    name, i, got[i], shape[i]);
            return NULL;
        }
        want *= (size_t)shape[i];
    }
    if (ds_int_list(ds_field(obj, "data_offsets"), off, 8, &no) != 0 || no != 2) {
        fprintf(stderr, "dspine: %s has no data_offsets\n", name);
        return NULL;
    }
    if ((size_t)(off[1] - off[0]) != want) {
        fprintf(stderr, "dspine: %s byte span %lld != %zu\n", name, off[1] - off[0], want);
        return NULL;
    }
    return (const uint16_t *)((const char *)d->map + d->data_off + (size_t)off[0]);
}

/* BF16 vector from the file, widened once (norm weights, gate bias). */
static float *ds_fvec(dspine_model_t *d, const char *name, int n) {
    int shape[2];
    const uint16_t *p;
    float *out;
    int i;
    if (n == 1) {
        shape[0] = 1;
        p = ds_tensor(d, name, shape, 1);
    } else {
        shape[0] = n;
        p = ds_tensor(d, name, shape, 1);
    }
    if (p == NULL) return NULL;
    out = (float *)malloc((size_t)n * sizeof(float));
    if (out == NULL) return NULL;
    for (i = 0; i < n; ++i) out[i] = ds_bf16(p[i]);
    return out;
}

/* ------------------------------------------------------------------ */
/* load                                                               */
/* ------------------------------------------------------------------ */

/* bf16 rows -> PTQ1_0, if the shape allows and DS_PTQ is not 0. */
/* Owned allocation, freed by dspine_free(). */
static int ds_qmem_add(dspine_model_t *d, void *mem) {
    if (d->nqmem == d->cqmem) {
        int c = d->cqmem ? d->cqmem * 2 : 64;
        void **np = (void **)realloc(d->qmem, (size_t)c * sizeof(void *));
        if (np == NULL) return 0;
        d->qmem = np;
        d->cqmem = c;
    }
    d->qmem[d->nqmem++] = mem;
    return 1;
}

static int ds_p2_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("DS_P2");
        cached = (e == NULL || atoi(e) != 0) && bitnet_ptq1_0_p2_available();
    }
    return cached;
}

static int ds_to_ptq(dspine_model_t *d, ds_w *w) {
    size_t rb;
    uint8_t *q;
    int r;
    const char *env = getenv("DS_PTQ");
    if (env != NULL && atoi(env) == 0) return 0;
    if (w->n_in % BITNET_PTQ1_0_QK != 0) return 0;
    if ((size_t)w->n_out * (size_t)w->n_in < (size_t)1 << 20) return 0;
    rb = bitnet_ptq1_0_row_size(w->n_in);
    q = (uint8_t *)malloc(rb * (size_t)w->n_out);
    if (q == NULL) return 0;
    for (r = 0; r < w->n_out; ++r)
        if (bitnet_ptq1_0_quantize_row(w->w + (size_t)r * w->n_in,
                                       q + (size_t)r * rb, w->n_in) != 0) {
            free(q);
            return 0;
        }
    if (!ds_qmem_add(d, q)) { free(q); return 0; }
    w->q = q;
    w->w = NULL;
    ++d->nqmat;
    d->q_bytes += (long)(rb * (size_t)w->n_out);

    /* The draft used to run only the 28-byte SDOT path.  Its matrices are
     * smaller than the target's but the shapes are identical, so the same
     * 2-bit repack applies; at B=2 that is worth ~2.4x on this kernel. */
    if (ds_p2_enabled()) {
        const size_t pb = bitnet_ptq1_0_p2_pair_bytes(w->n_in);
        const size_t nb = (size_t)(w->n_in / 128);
        const int npair = w->n_out / 2;
        uint8_t *p2 = (uint8_t *)malloc(pb * (size_t)npair);
        float *p2d = (float *)malloc(sizeof(float) * (size_t)npair * nb * 2u);
        if (p2 != NULL && p2d != NULL) {
            for (r = 0; r + 1 < w->n_out; r += 2)
                bitnet_ptq1_0_p2_pack(q + (size_t)r * rb, q + (size_t)(r + 1) * rb,
                                      w->n_in, p2 + (size_t)(r / 2) * pb,
                                      p2d + (size_t)(r / 2) * nb * 2u);
            if (ds_qmem_add(d, p2) && ds_qmem_add(d, p2d)) {
                w->p2 = p2;
                w->p2d = p2d;
                d->p2_bytes += (long)(pb * (size_t)npair +
                                      sizeof(float) * (size_t)npair * nb * 2u);
            } else {
                free(p2); free(p2d);
            }
        } else {
            free(p2); free(p2d);
        }
    }
    return 1;
}

static int ds_bind2(dspine_model_t *d, const char *name, int n_out, int n_in, ds_w *w) {
    int sh[2];
    sh[0] = n_out;
    sh[1] = n_in;
    w->q = NULL;
    w->p2 = NULL;
    w->p2d = NULL;
    w->w = ds_tensor(d, name, sh, 2);
    if (w->w == NULL) return -1;
    w->n_out = n_out;
    w->n_in = n_in;
    ds_to_ptq(d, w);
    return 0;
}

static long long ds_scalar(dspine_model_t *d, const char *name, int want) {
    const char *obj = ds_object(d, name);
    const char *dt, *of;
    long long off[8];
    int no = 0;
    if (obj == NULL) {
        fprintf(stderr, "dspine: missing %s\n", name);
        return -1;
    }
    dt = ds_field(obj, "dtype");
    if (dt == NULL || strncmp(dt, "\"I64\"", 5) != 0) return -1;
    of = ds_field(obj, "data_offsets");
    if (ds_int_list(of, off, 8, &no) != 0 || no != 2) return -1;
    {
        int64_t v;
        memcpy(&v, (const char *)d->map + d->data_off + (size_t)off[0], sizeof(v));
        if ((int)v != want) {
            fprintf(stderr, "dspine: %s = %d, expected %d\n", name, (int)v, want);
            return -1;
        }
        return v;
    }
}

static int ds_bind_layer(dspine_model_t *d, int l) {
    char nm[128];
    ds_layer *L = &d->L[l];
    const int H = d->H, hd = d->hd, nff = d->nff;
    const int qd = d->nh * hd, kd = d->nkv * hd;

#define DSP(fmt, ...) (snprintf(nm, sizeof(nm), fmt, __VA_ARGS__), nm)
    L->in_norm = ds_fvec(d, DSP("layers.%d.input_layernorm.weight", l), H);
    L->post_norm = ds_fvec(d, DSP("layers.%d.post_attention_layernorm.weight", l), H);
    L->q_norm = ds_fvec(d, DSP("layers.%d.self_attn.q_norm.weight", l), hd);
    L->k_norm = ds_fvec(d, DSP("layers.%d.self_attn.k_norm.weight", l), hd);
    if (!L->in_norm || !L->post_norm || !L->q_norm || !L->k_norm) return -1;
    if (ds_bind2(d, DSP("layers.%d.self_attn.q_proj.weight", l), qd, H, &L->q)) return -1;
    if (ds_bind2(d, DSP("layers.%d.self_attn.k_proj.weight", l), kd, H, &L->k)) return -1;
    if (ds_bind2(d, DSP("layers.%d.self_attn.v_proj.weight", l), kd, H, &L->v)) return -1;
    if (ds_bind2(d, DSP("layers.%d.self_attn.o_proj.weight", l), H, qd, &L->o)) return -1;
    if (ds_bind2(d, DSP("layers.%d.mlp.gate_proj.weight", l), nff, H, &L->gate)) return -1;
    if (ds_bind2(d, DSP("layers.%d.mlp.up_proj.weight", l), nff, H, &L->up)) return -1;
    if (ds_bind2(d, DSP("layers.%d.mlp.down_proj.weight", l), H, nff, &L->down)) return -1;
#undef DSP
    return 0;
}

static int ds_reserve(dspine_model_t *d, int need) {
    int l;
    if (need <= d->cap) return 0;
    if (need < 256) need = 256;
    for (l = 0; l < d->nl; ++l) {
        float *nk = (float *)realloc(d->ck[l], (size_t)need * d->nkv * d->hd * sizeof(float));
        float *nv;
        if (nk == NULL) return -1;
        d->ck[l] = nk;
        nv = (float *)realloc(d->cv[l], (size_t)need * d->nkv * d->hd * sizeof(float));
        if (nv == NULL) return -1;
        d->cv[l] = nv;
    }
    {
        float *na = (float *)realloc(d->ascore,
                                     (size_t)d->nthreads * (size_t)(need + DS_MAX_BLK) * sizeof(float));
        if (na == NULL) return -1;
        d->ascore = na;
    }
    d->cap = need;
    return 0;
}

void dspine_free(dspine_model_t *d) {
    int l;
    if (d == NULL) return;
    for (l = 0; l < DS_MAX_L; ++l) {
        free(d->ck[l]);
        free(d->cv[l]);
        if (l < d->nl) {
            free(d->L[l].in_norm);
            free(d->L[l].post_norm);
            free(d->L[l].q_norm);
            free(d->L[l].k_norm);
        }
    }
    free(d->hidden_norm);
    free(d->norm);
    free(d->rope_freq);
    free(d->gate_bf);
    free(d->fw_f);
    free(d->pg_f);
    free(d->pv_f);
    free(d->x); free(d->xn); free(d->ab); free(d->raw); free(d->dbuf);
    free(d->qb); free(d->kb); free(d->vb); free(d->ob);
    free(d->gb); free(d->ub);
    free(d->codes); free(d->projected); free(d->msg);
    free(d->qpack); free(d->spack); free(d->vpack);
    free(d->rms); free(d->libuf);
    free(d->hrow); free(d->uhat); free(d->tg); free(d->mm); free(d->tv);
    free(d->logits); free(d->ascore);
    free(d->hdr_buf);
    if (d->qmem != NULL) {
        int i;
        for (i = 0; i < d->nqmem; ++i) free(d->qmem[i]);
        free(d->qmem);
    }
    free(d->qact);
    free(d->qasc);
    free(d->p2prep);
    if (d->map != NULL) munmap(d->map, d->map_len);
    free(d);
}

dspine_model_t *dspine_load(const char *path, bonsai_model_t *target,
                            bonsai_context_t *ctx) {
    dspine_model_t *d;
    struct stat st;
    int fd;
    uint64_t hlen;
    int sh[3];
    int l;
    const int T = DS_MAX_BLK;

    if (path == NULL || target == NULL || ctx == NULL) return NULL;
    d = (dspine_model_t *)calloc(1, sizeof(*d));
    if (d == NULL) return NULL;
    d->tgt = target;
    d->ctx = ctx;
    d->pool = bonsai_pool(target);
    d->nthreads = bonsai_pool_threads(target);
    if (d->nthreads < 1) d->nthreads = 1;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "dspine: cannot open %s\n", path);
        dspine_free(d);
        return NULL;
    }
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < 16) {
        close(fd);
        dspine_free(d);
        return NULL;
    }
    d->map_len = (size_t)st.st_size;
    d->map = mmap(NULL, d->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (d->map == MAP_FAILED) {
        d->map = NULL;
        dspine_free(d);
        return NULL;
    }
    memcpy(&hlen, d->map, sizeof(hlen));
    if (hlen > d->map_len - 8) {
        fprintf(stderr, "dspine: bad safetensors header length\n");
        dspine_free(d);
        return NULL;
    }
    d->hdr_len = (size_t)hlen;
    d->hdr_buf = (char *)malloc(d->hdr_len + 1);
    if (d->hdr_buf == NULL) {
        dspine_free(d);
        return NULL;
    }
    memcpy(d->hdr_buf, (const char *)d->map + 8, d->hdr_len);
    d->hdr_buf[d->hdr_len] = '\0';
    d->hdr = d->hdr_buf;
    d->data_off = 8 + (size_t)hlen;

    if (ds_scalar(d, "draft_block_causal_version", 1) < 0 ||
        ds_scalar(d, "last_write_refine_version", 1) < 0 ||
        ds_scalar(d, "selected_prefix_bridge.independent_write_version", 1) < 0 ||
        ds_scalar(d, "selected_prefix_bridge.terminal_disabled_version", 1) < 0) {
        fprintf(stderr, "dspine: unsupported checkpoint version\n");
        dspine_free(d);
        return NULL;
    }

    d->H = bonsai_embedding_length(target);
    d->V = bonsai_vocab_size(target);
    d->ntap = 5;
    d->concat = d->ntap * d->H;
    d->nl = 5;
    d->nff = 17408;
    d->nh = 32;
    d->nkv = 8;
    d->hd = 128;
    d->C = 1024;
    d->R = 512;
    d->block_size = 16;
    d->mask_id = 248070;
    d->topk = 16;
    d->window = 8192;
    d->eps = 1e-6f;
    d->rope_theta = 1e7f;

    if (d->H != 5120 || d->V != 248320) {
        fprintf(stderr, "dspine: target mismatch (hidden %d, vocab %d)\n", d->H, d->V);
        dspine_free(d);
        return NULL;
    }

    if (ds_bind2(d, "fc.weight", d->H, d->concat, &d->fc)) { dspine_free(d); return NULL; }
    d->hidden_norm = ds_fvec(d, "hidden_norm.weight", d->H);
    d->norm = ds_fvec(d, "norm.weight", d->H);
    if (d->hidden_norm == NULL || d->norm == NULL) { dspine_free(d); return NULL; }
    for (l = 0; l < d->nl; ++l)
        if (ds_bind_layer(d, l) != 0) { dspine_free(d); return NULL; }

    sh[0] = d->V; sh[1] = d->C;
    d->codebook = ds_tensor(d, "selected_prefix_bridge.codebook", sh, 2);
    sh[0] = d->nl - 1; sh[1] = d->C; sh[2] = d->H;
    d->early_read = ds_tensor(d, "selected_prefix_bridge.early_read_weights", sh, 3);
    sh[0] = d->C; sh[1] = d->H;
    d->read = ds_tensor(d, "selected_prefix_bridge.read_weight", sh, 2);
    sh[0] = d->nl; sh[1] = d->H; sh[2] = d->R;
    d->feat_write = ds_tensor(d, "selected_prefix_bridge.feature_write_weights", sh, 3);
    sh[0] = d->R; sh[1] = d->C;
    d->gate_w = ds_tensor(d, "selected_prefix_bridge.gate.weight", sh, 2);
    d->pred_gate = ds_tensor(d, "selected_prefix_bridge.predecessor_gate.weight", sh, 2);
    d->prev_w = ds_tensor(d, "selected_prefix_bridge.prev.weight", sh, 2);
    sh[0] = d->R;
    d->gate_b = ds_tensor(d, "selected_prefix_bridge.gate.bias", sh, 1);
    if (!d->codebook || !d->early_read || !d->read || !d->feat_write ||
        !d->gate_w || !d->pred_gate || !d->prev_w || !d->gate_b) {
        dspine_free(d);
        return NULL;
    }
    d->gate_bf = (float *)malloc((size_t)d->R * sizeof(float));
    if (d->gate_bf == NULL) { dspine_free(d); return NULL; }
    for (l = 0; l < d->R; ++l) d->gate_bf[l] = ds_bf16(d->gate_b[l]);

    /* ds_tail reads these as bf16 in scalar loops: 42 M MACs for the feature
     * write alone, which made the draft's post-block tail 17 ms - as much as
     * the transformer pass it follows.  Widen once here so the loops can use
     * NEON fma. */
    {
        const size_t nfw = (size_t)d->H * (size_t)d->R;
        const size_t nrc = (size_t)d->R * (size_t)d->C;
        const uint16_t *fw = d->feat_write + (size_t)(d->nl - 1) * nfw;
        size_t i2;
        d->fw_f = (float *)malloc(nfw * sizeof(float));
        d->pg_f = (float *)malloc(nrc * sizeof(float));
        d->pv_f = (float *)malloc(nrc * sizeof(float));
        if (d->fw_f == NULL || d->pg_f == NULL || d->pv_f == NULL) {
            dspine_free(d);
            return NULL;
        }
        for (i2 = 0; i2 < nfw; ++i2) d->fw_f[i2] = ds_bf16(fw[i2]);
        for (i2 = 0; i2 < nrc; ++i2) d->pg_f[i2] = ds_bf16(d->pred_gate[i2]);
        for (i2 = 0; i2 < nrc; ++i2) d->pv_f[i2] = ds_bf16(d->prev_w[i2]);
    }

    d->rope_freq = (float *)malloc((size_t)(d->hd / 2) * sizeof(float));
    if (d->rope_freq == NULL) { dspine_free(d); return NULL; }
    for (l = 0; l < d->hd / 2; ++l)
        d->rope_freq[l] = powf(d->rope_theta, -2.0f * (float)l / (float)d->hd);

    /* scratch */
    d->x = (float *)calloc((size_t)T * d->H, sizeof(float));
    d->xn = (float *)calloc((size_t)T * d->H, sizeof(float));
    d->ab = (float *)calloc((size_t)T * d->H, sizeof(float));
    d->raw = (float *)calloc((size_t)T * d->H, sizeof(float));
    d->dbuf = (float *)calloc((size_t)T * d->H, sizeof(float));
    d->qb = (float *)calloc((size_t)T * d->nh * d->hd, sizeof(float));
    d->kb = (float *)calloc((size_t)T * d->nkv * d->hd, sizeof(float));
    d->vb = (float *)calloc((size_t)T * d->nkv * d->hd, sizeof(float));
    d->ob = (float *)calloc((size_t)T * d->nh * d->hd, sizeof(float));
    d->gb = (float *)calloc((size_t)T * d->nff, sizeof(float));
    d->ub = (float *)calloc((size_t)T * d->nff, sizeof(float));
    d->codes = (float *)calloc((size_t)T * d->C, sizeof(float));
    d->projected = (float *)calloc((size_t)T * d->C, sizeof(float));
    d->qpack = (float *)calloc((size_t)T * d->R, sizeof(float));
    d->spack = (float *)calloc((size_t)T * d->R, sizeof(float));
    d->vpack = (float *)calloc((size_t)T * d->R, sizeof(float));
    d->msg = (float *)calloc((size_t)T * d->R, sizeof(float));
    d->rms = (float *)calloc((size_t)T, sizeof(float));
    d->libuf = (float *)calloc((size_t)T * d->H, sizeof(float));
    d->hrow = (float *)calloc((size_t)DS_MAX_BLK * d->H, sizeof(float));
    d->uhat = (float *)calloc((size_t)DS_MAX_BLK * d->H, sizeof(float));
    d->tg = (float *)calloc((size_t)DS_MAX_BLK * d->R, sizeof(float));
    d->mm = (float *)calloc((size_t)DS_MAX_BLK * d->R, sizeof(float));
    d->tv = (float *)calloc((size_t)DS_MAX_BLK * 2 * d->R, sizeof(float));
    d->logits = (float *)calloc((size_t)T * d->V, sizeof(float));
    if (!d->x || !d->xn || !d->ab || !d->raw || !d->dbuf || !d->qb || !d->kb ||
        !d->vb || !d->ob || !d->gb || !d->ub || !d->codes || !d->projected ||
        !d->qpack || !d->spack || !d->vpack || !d->msg || !d->rms || !d->libuf || !d->hrow || !d->uhat ||
        !d->tg || !d->mm || !d->tv || !d->logits || ds_reserve(d, 256) != 0) {
        dspine_free(d);
        return NULL;
    }
    if (d->q_bytes > 0) {
        double mb = (double)d->q_bytes / 1048576.0;
        fprintf(stderr, "dspine: PTQ1_0 weights %.0f MB in %d matrices "
                        "(bf16 would be %.0f MB), i8mm repack %.0f MB%s\n",
                mb, d->nqmat, mb * 256.0 / 28.0,
                (double)d->p2_bytes / 1048576.0,
                d->p2_bytes > 0 ? "" : " (off)");
    }
    return d;
}

int dspine_max_drafts(const dspine_model_t *d) { return d->block_size - 1; }
int dspine_mask_token(const dspine_model_t *d) { return d->mask_id; }
int dspine_context_len(const dspine_model_t *d) { return d->ctx_len; }
void dspine_reset(dspine_model_t *d) { if (d != NULL) d->ctx_len = 0; }
void dspine_timings(const dspine_model_t *d, double *block, double *tail) {
    if (block != NULL) *block = d->t_block;
    if (tail != NULL) *tail = d->t_tail;
}

/* ------------------------------------------------------------------ */
/* target-context K/V                                                */
/* ------------------------------------------------------------------ */

int dspine_append_context(dspine_model_t *d, const float *hid, int n) {
    const int H = d->H, kd = d->nkv * d->hd;
    int off;
    if (d == NULL || hid == NULL || n < 0) return -1;
    if (n == 0) return 0;
    if (ds_reserve(d, d->ctx_len + n) != 0) return -1;
    for (off = 0; off < n; off += DS_MAX_BLK) {
        const int B = (n - off) < DS_MAX_BLK ? (n - off) : DS_MAX_BLK;
        int l, b, h;
        ds_mv(d, &d->fc, hid + (size_t)off * d->concat, B, d->libuf);
        for (b = 0; b < B; ++b)
            ds_rmsnorm(d->libuf + (size_t)b * H, d->hidden_norm, H, d->eps,
                       d->libuf + (size_t)b * H);
        for (l = 0; l < d->nl; ++l) {
            float *KK = d->ck[l] + (size_t)(d->ctx_len + off) * kd;
            float *VV = d->cv[l] + (size_t)(d->ctx_len + off) * kd;
            ds_mv(d, &d->L[l].k, d->libuf, B, d->kb);
            ds_mv(d, &d->L[l].v, d->libuf, B, d->vb);
            for (b = 0; b < B; ++b) {
                float *kb = d->kb + (size_t)b * kd;
                float *vb = d->vb + (size_t)b * kd;
                for (h = 0; h < d->nkv; ++h)
                    ds_rmsnorm(kb + (size_t)h * d->hd, d->L[l].k_norm, d->hd, d->eps,
                               kb + (size_t)h * d->hd);
                ds_rope(kb, d->nkv, d->hd, d->ctx_len + off + b, d->rope_freq);
                memcpy(KK + (size_t)b * kd, kb, (size_t)kd * sizeof(float));
                memcpy(VV + (size_t)b * kd, vb, (size_t)kd * sizeof(float));
            }
        }
    }
    d->ctx_len += n;
    return 0;
}

/* ------------------------------------------------------------------ */
/* block attention over [context | block], causal, windowed           */
/* ------------------------------------------------------------------ */

static float ds_dotf(const float *a, const float *b, int n) {
#if defined(DS_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float s;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    int i;
    for (i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
#endif
}

typedef struct {
    dspine_model_t *d;
    const float *q, *k, *v, *K, *V;
    int p0, ctx, T, next;
} ds_attn_t;

static void ds_attn_job(void *p, int tid, int nthreads) {
    ds_attn_t *a = (ds_attn_t *)p;
    dspine_model_t *d = a->d;
    const int nh = d->nh, nkv = d->nkv, hd = d->hd;
    const int rep = nh / nkv;
    const float scale = 1.0f / sqrtf((float)hd);
    float *sc = d->ascore + (size_t)tid * (size_t)(d->cap + DS_MAX_BLK);
    (void)nthreads;
    for (;;) {
        int id = __atomic_fetch_add(&a->next, 1, __ATOMIC_RELAXED);
        int t, h, kh, i, nk, start, dd;
        const float *qh;
        float maxv, sum;
        float *o;
        if (id >= a->T * nh) break;
        t = id / nh;
        h = id % nh;
        kh = h / rep;
        qh = a->q + (size_t)t * nh * hd + (size_t)h * hd;
        start = a->p0 + t - d->window + 1;
        if (start < 0) start = 0;
        nk = a->p0 + t - start + 1;
        maxv = -INFINITY;
        for (i = 0; i < nk; ++i) {
            const int kpos = start + i;
            const float *kt = (kpos < a->ctx)
                ? (a->K + (size_t)kpos * nkv * hd + (size_t)kh * hd)
                : (a->k + (size_t)(kpos - a->ctx) * nkv * hd + (size_t)kh * hd);
            float s = ds_dotf(qh, kt, hd) * scale;
            sc[i] = s;
            if (s > maxv) maxv = s;
        }
        sum = 0.0f;
        for (i = 0; i < nk; ++i) {
            float e = expf(sc[i] - maxv);
            sc[i] = e;
            sum += e;
        }
        o = d->ob + (size_t)t * nh * hd + (size_t)h * hd;
        for (dd = 0; dd < hd; ++dd) o[dd] = 0.0f;
        for (i = 0; i < nk; ++i) {
            const int kpos = start + i;
            const float pr = sc[i] / sum;
            const float *vt = (kpos < a->ctx)
                ? (a->V + (size_t)kpos * nkv * hd + (size_t)kh * hd)
                : (a->v + (size_t)(kpos - a->ctx) * nkv * hd + (size_t)kh * hd);
            for (dd = 0; dd < hd; ++dd) o[dd] += pr * vt[dd];
        }
    }
}

/* ------------------------------------------------------------------ */
/* proposal                                                           */
/* ------------------------------------------------------------------ */

static double ds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* The k largest logits, ties broken by the lower id, then sorted by id. */
static int ds_topk(const float *lg, int V, int K, int *ids) {
    float vals[32];
    int n = 0, i, j;
    for (i = 0; i < V; ++i) {
        const float v = lg[i];
        if (n == K && !(v > vals[K - 1])) continue;
        j = (n < K) ? n : K - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            --j;
        }
        vals[j] = v;
        ids[j] = i;
        if (n < K) ++n;
    }
    /* ascending ids, as the reference's topk_ids returns */
    for (i = 1; i < n; ++i) {
        int v = ids[i], j2 = i - 1;
        while (j2 >= 0 && ids[j2] > v) { ids[j2 + 1] = ids[j2]; --j2; }
        ids[j2 + 1] = v;
    }
    return n;
}

/* The code bridge: one write into the residual stream after every layer. */
static void ds_bridge_write(dspine_model_t *d, int l, int T, int anchor) {
    const int H = d->H, C = d->C, R = d->R;
    const float sqrt_c = sqrtf((float)C);
    const uint16_t *rd = (l == d->nl - 1) ? d->read : (d->early_read + (size_t)l * C * H);
    ds_w rw;
    int t, i;
    float *normed = d->libuf;

    rw.w = rd; rw.n_out = C; rw.n_in = H; rw.q = NULL;
    for (t = 0; t < T; ++t) {
        float *f = d->x + (size_t)t * H;
        double s = 0.0;
        float rms;
        for (i = 0; i < H; ++i) s += (double)f[i] * f[i];
        rms = sqrtf((float)(s / H));
        if (rms < 1.17549435e-38f) rms = 1.17549435e-38f;
        d->rms[t] = rms;
        if (t == 0) continue;                     /* the anchor row keeps the codebook code */
        for (i = 0; i < H; ++i) normed[i] = f[i] / (rms + d->eps);
        ds_mv(d, &rw, normed, 1, d->projected + (size_t)t * C);
        {
            float *pr = d->projected + (size_t)t * C;
            double pn = 0.0;
            for (i = 0; i < C; ++i) pn += (double)pr[i] * pr[i];
            pn = sqrt(pn);
            if (pn > 1e-20) {
                for (i = 0; i < C; ++i)
                    d->codes[(size_t)t * C + i] = sqrt_c * (float)(pr[i] / (pn + d->eps));
            } else {
                memset(d->codes + (size_t)t * C, 0, (size_t)C * sizeof(float));
            }
        }
    }
    {
        float *c0 = d->codes;
        const uint16_t *cb = d->codebook + (size_t)anchor * C;
        for (i = 0; i < C; ++i) c0[i] = sqrt_c * ds_bf16(cb[i]);
    }
    {
        ds_w ag, ap, av;
        ag.w = d->gate_w; ag.n_out = R; ag.n_in = C; ag.q = NULL;
        ap.w = d->pred_gate; ap.n_out = R; ap.n_in = C; ap.q = NULL;
        av.w = d->prev_w; av.n_out = R; av.n_in = C; av.q = NULL;
        ds_mv(d, &ag, d->codes, T, d->qpack);
        ds_mv(d, &ap, d->codes, T, d->spack);
        ds_mv(d, &av, d->codes, T, d->vpack);
    }
    memset(d->msg, 0, (size_t)R * sizeof(float));
    for (t = 1; t < T; ++t) {
        const float *q1 = d->qpack + (size_t)t * R;
        const float *s0 = d->spack + (size_t)(t - 1) * R;
        const float *v0 = d->vpack + (size_t)(t - 1) * R;
        float *m = d->msg + (size_t)t * R;
        for (i = 0; i < R; ++i)
            m[i] = ds_sigmoid(q1[i] + s0[i] + d->gate_bf[i]) * v0[i];
    }
    {
        ds_w fw;
        fw.w = d->feat_write + (size_t)l * H * R;
        fw.q = NULL;
        fw.n_out = H;
        fw.n_in = R;
        ds_mv(d, &fw, d->msg, T, d->dbuf);
    }
    for (t = 1; t < T; ++t) {
        float *f = d->x + (size_t)t * H;
        const float *dl = d->dbuf + (size_t)t * H;
        const float r = d->rms[t];
        for (i = 0; i < H; ++i) f[i] += r * dl[i];
    }
}

/* Last-write refinement: resolve the block against the target's LM head. */
/* DS_TIMING=1 splits the draft's tail into the target LM head and the
 * candidate ranking, which used to be one number. */
static int ds_timing_on(void) {
    static int c = -1;
    if (c < 0) { const char *e = getenv("DS_TIMING"); c = e != NULL; }
    return c;
}

static double ds_t_tk, ds_t_hr, ds_t_tg, ds_t_tv, ds_t_sc;

/* tg[k] = uhat[k] . feature_write, the rank-R readout of the write.  This is
 * 16 independent products against the same 10.5 MB table, so the candidates
 * split across the pool and each thread streams the table once; serially it
 * was 7.3 ms of the draft's 8 ms tail. */
static void ds_tg_rows(dspine_model_t *d, int k0, int k1) {
    const int H = d->H, R = d->R;
    const float *fw = d->fw_f;
    int k, c, hh, rr;
#if defined(DS_NEON)
    for (k = k0; k < k1; k += 4) {
        for (rr = 0; rr < R; rr += 8) {
            float32x4_t a0[4], a1[4];
            for (c = 0; c < 4; ++c) a0[c] = a1[c] = vdupq_n_f32(0.0f);
            for (hh = 0; hh < H; ++hh) {
                const float *fr = fw + (size_t)hh * R + rr;
                const float32x4_t f0 = vld1q_f32(fr);
                const float32x4_t f1 = vld1q_f32(fr + 4);
                for (c = 0; c < 4; ++c) {
                    const int kk = (k + c < k1) ? k + c : k1 - 1;
                    const float u = d->uhat[(size_t)kk * H + hh];
                    a0[c] = vmlaq_n_f32(a0[c], f0, u);
                    a1[c] = vmlaq_n_f32(a1[c], f1, u);
                }
            }
            for (c = 0; c < 4 && k + c < k1; ++c) {
                vst1q_f32(d->tg + (size_t)(k + c) * R + rr, a0[c]);
                vst1q_f32(d->tg + (size_t)(k + c) * R + rr + 4, a1[c]);
            }
        }
    }
#else
    for (k = k0; k < k1; ++k) {
        const float *uh = d->uhat + (size_t)k * H;
        float *tg = d->tg + (size_t)k * R;
        memset(tg, 0, (size_t)R * sizeof(float));
        for (hh = 0; hh < H; ++hh) {
            const float u = uh[hh];
            const float *fr = fw + (size_t)hh * R;
            for (rr = 0; rr < R; ++rr) tg[rr] += u * fr[rr];
        }
    }
#endif
}

typedef struct { dspine_model_t *d; int next, K; } ds_tg_t;

static void ds_tg_job(void *pv, int tid, int nthreads) {
    ds_tg_t *a = (ds_tg_t *)pv;
    (void)tid; (void)nthreads;
    for (;;) {
        int k = __atomic_fetch_add(&a->next, 4, __ATOMIC_RELAXED);
        int en = k + 4;
        if (k >= a->K) break;
        if (en > a->K) en = a->K;
        ds_tg_rows(a->d, k, en);
    }
}

static void ds_tail(dspine_model_t *d, int nr) {
    const double tall0 = ds_now();
    double tq = tall0;
    ds_t_tk = ds_t_hr = ds_t_tg = ds_t_tv = ds_t_sc = 0.0;
    const int H = d->H, C = d->C, R = d->R, V = d->V, K = d->topk;
    const float sqrt_c = sqrtf((float)C);
    int i, k;

    for (i = 0; i < nr; ++i) {
        const float *lg = d->logits + (size_t)i * V;
        const float *raw = d->raw + (size_t)(i + 1) * H;
        const float rms = d->rms[i + 1];
        const float *qq = d->qpack + (size_t)(i + 1) * R;
        int *cand = d->cand[i];

        ds_topk(lg, V, K, cand);
        { const double q1 = ds_now(); ds_t_tk += q1 - tq; tq = q1; }
        /* dense head rows and their dot with the pre-write state */
        for (k = 0; k < K; ++k) {
            float *hr = d->hrow + (size_t)k * H;
            float *uh = d->uhat + (size_t)k * H;
            int dd;
            bonsai_head_row(d->tgt, cand[k], hr);
            for (dd = 0; dd < H; ++dd) uh[dd] = hr[dd] * d->norm[dd];
            d->base_[i][k] = ds_dotf(uh, raw, H);
        }
        { const double q1 = ds_now(); ds_t_hr += q1 - tq; tq = q1; }
        /* tg[k] = uhat[k] . feature_write (rank-R readout of the write). */
        {
            ds_tg_t ta;
            ta.d = d;
            ta.K = K;
            ta.next = 0;
            bonsai_pool_run(d->pool, ds_tg_job, &ta);
        }
        { const double q1 = ds_now(); ds_t_tg += q1 - tq; tq = q1; }
        /* predecessor state: the anchor for row 0, else the previous candidate */
        {
            int prev_sid = -1;
            for (k = 0; k < K; ++k) {
                const int sid = (i == 0) ? d->anchor_cur : d->cand[i - 1][k];
                float *tvg = d->tv + (size_t)k * 2 * R;
                float *tvv = tvg + R;
                int cc, rr;
                if (sid == prev_sid) {
                    /* row 0 shares one anchor across every candidate */
                    memcpy(tvg, d->tv, (size_t)2 * R * sizeof(float));
                    continue;
                }
                prev_sid = sid;
                {
                    const uint16_t *cb = d->codebook + (size_t)sid * C;
                    for (cc = 0; cc < C; ++cc) d->libuf[cc] = sqrt_c * ds_bf16(cb[cc]);
                }
                for (rr = 0; rr < R; ++rr) {
                    const float *pg = d->pg_f + (size_t)rr * C;
                    const float *pv = d->pv_f + (size_t)rr * C;
#if defined(DS_NEON)
                    float32x4_t s1 = vdupq_n_f32(0.0f), s2 = s1;
                    for (cc = 0; cc + 4 <= C; cc += 4) {
                        const float32x4_t e = vld1q_f32(d->libuf + cc);
                        s1 = vmlaq_f32(s1, e, vld1q_f32(pg + cc));
                        s2 = vmlaq_f32(s2, e, vld1q_f32(pv + cc));
                    }
                    tvg[rr] = vaddvq_f32(s1);
                    tvv[rr] = vaddvq_f32(s2);
#else
                    float a1 = 0.0f, a2 = 0.0f;
                    for (cc = 0; cc < C; ++cc) {
                        const float e = d->libuf[cc];
                        a1 += e * pg[cc];
                        a2 += e * pv[cc];
                    }
                    tvg[rr] = a1;
                    tvv[rr] = a2;
#endif
                }
            }
        }
        { const double q1 = ds_now(); ds_t_tv += q1 - tq; tq = q1; }
        for (k = 0; k < K; ++k) {
            const float *tvg = d->tv + (size_t)k * 2 * R;
            const float *tvv = tvg + R;
            float *m = d->mm + (size_t)k * R;
            int rr;
            for (rr = 0; rr < R; ++rr)
                m[rr] = ds_sigmoid(qq[rr] + tvg[rr] + d->gate_bf[rr]) * tvv[rr];
        }
        for (k = 0; k < K; ++k) {
            const float *m = d->mm + (size_t)k * R;
            int k2, rr, best = 0;
            float bv = -INFINITY;
            for (k2 = 0; k2 < K; ++k2) {
                const float *tg = d->tg + (size_t)k2 * R;
                float s = 0.0f;
                for (rr = 0; rr < R; ++rr) s += m[rr] * tg[rr];
                d->score[k][k2] = d->base_[i][k2] + rms * s;
                if (d->score[k][k2] > bv) { bv = d->score[k][k2]; best = k2; }
            }
            d->idx[i][k] = best;
        }
    }
    ds_t_sc += ds_now() - tq;
    if (ds_timing_on())
        fprintf(stderr, "#tk %.1f hr %.1f tg %.1f tv %.1f sc %.1f tot %.1f\n",
                ds_t_tk * 1e3, ds_t_hr * 1e3, ds_t_tg * 1e3, ds_t_tv * 1e3,
                ds_t_sc * 1e3, (ds_now() - tall0) * 1e3);
    {
        int j = d->idx[0][0];
        for (i = 0; i < nr; ++i) {
            d->drafts_cur[i] = d->cand[i][j];
            if (i + 1 < nr) j = d->idx[i + 1][j];
        }
    }
}

int dspine_propose(dspine_model_t *d, int anchor, int n, int *drafts) {
    const int H = d->H, qd = d->nh * d->hd, kd = d->nkv * d->hd, nff = d->nff;
    const int T = n + 1, p0 = d->ctx_len;
    const float eps = d->eps;
    int l, t, h, i;
    double t0;

    if (d == NULL || drafts == NULL || n < 1) return -1;
    if (T > DS_MAX_BLK || T > d->block_size) return -1;

    for (t = 0; t < T; ++t)
        bonsai_embed_row(d->tgt, (t == 0) ? anchor : d->mask_id, d->x + (size_t)t * H);
    d->anchor_cur = anchor;

    d->t_mv = 0.0;
    d->t_attn = 0.0;
    t0 = ds_now();
    for (t = 0; t < T; ++t)
        ds_rmsnorm(d->x + (size_t)t * H, d->L[0].in_norm, H, eps, d->xn + (size_t)t * H);
    for (l = 0; l < d->nl; ++l) {
        ds_layer *L = &d->L[l];
        ds_attn_t aa;
        const float *nw;

        ds_mv(d, &L->q, d->xn, T, d->qb);
        ds_mv(d, &L->k, d->xn, T, d->kb);
        ds_mv(d, &L->v, d->xn, T, d->vb);
        for (t = 0; t < T; ++t) {
            float *q = d->qb + (size_t)t * qd;
            float *k2 = d->kb + (size_t)t * kd;
            for (h = 0; h < d->nh; ++h)
                ds_rmsnorm(q + (size_t)h * d->hd, L->q_norm, d->hd, eps,
                           q + (size_t)h * d->hd);
            for (h = 0; h < d->nkv; ++h)
                ds_rmsnorm(k2 + (size_t)h * d->hd, L->k_norm, d->hd, eps,
                           k2 + (size_t)h * d->hd);
            ds_rope(q, d->nh, d->hd, p0 + t, d->rope_freq);
            ds_rope(k2, d->nkv, d->hd, p0 + t, d->rope_freq);
        }
        aa.d = d;
        aa.q = d->qb;
        aa.k = d->kb;
        aa.v = d->vb;
        aa.K = d->ck[l];
        aa.V = d->cv[l];
        aa.p0 = p0;
        aa.ctx = d->ctx_len;
        aa.T = T;
        aa.next = 0;
        {
            const double ta0 = ds_now();
            if (T * d->nh < 8) ds_attn_job(&aa, 0, 1);
            else bonsai_pool_run(d->pool, ds_attn_job, &aa);
            d->t_attn += ds_now() - ta0;
        }

        ds_mv(d, &L->o, d->ob, T, d->ab);
        for (t = 0; t < T; ++t) {
            float *x = d->x + (size_t)t * H;
            const float *a2 = d->ab + (size_t)t * H;
            for (i = 0; i < H; ++i) x[i] += a2[i];
        }
        for (t = 0; t < T; ++t)
            ds_rmsnorm(d->x + (size_t)t * H, L->post_norm, H, eps, d->xn + (size_t)t * H);
        ds_mv(d, &L->gate, d->xn, T, d->gb);
        ds_mv(d, &L->up, d->xn, T, d->ub);
        for (t = 0; t < T; ++t) {
            float *g2 = d->gb + (size_t)t * nff;
            const float *u2 = d->ub + (size_t)t * nff;
            for (i = 0; i < nff; ++i) g2[i] = ds_silu(g2[i]) * u2[i];
        }
        ds_mv(d, &L->down, d->gb, T, d->dbuf);
        for (t = 0; t < T; ++t) {
            float *x = d->x + (size_t)t * H;
            const float *dd = d->dbuf + (size_t)t * H;
            for (i = 0; i < H; ++i) x[i] += dd[i];
        }
        if (l == d->nl - 1) memcpy(d->raw, d->x, (size_t)T * H * sizeof(float));
        ds_bridge_write(d, l, T, anchor);
        nw = (l + 1 < d->nl) ? d->L[l + 1].in_norm : d->norm;
        for (t = 0; t < T; ++t)
            ds_rmsnorm(d->x + (size_t)t * H, nw, H, eps, d->xn + (size_t)t * H);
    }
    d->t_block = ds_now() - t0;

    t0 = ds_now();
    if (bonsai_head_logits(d->ctx, d->xn + (size_t)H, n, d->logits, d->V) != 0)
        return -1;
    if (ds_timing_on())
        fprintf(stderr, "MV %.1f ATTN %.1f REST %.1f |", d->t_mv * 1e3, d->t_attn * 1e3,
                (d->t_block - d->t_mv - d->t_attn) * 1e3);
    if (ds_timing_on()) fprintf(stderr, " HEAD %.1f", (ds_now() - t0) * 1e3);
    t0 = ds_now();
    ds_tail(d, n);
    if (ds_timing_on()) fprintf(stderr, " TAIL %.1f\n", (ds_now() - t0) * 1e3);
    for (i = 0; i < n; ++i) drafts[i] = d->drafts_cur[i];
    d->t_tail = ds_now() - t0;
    return n;
}
