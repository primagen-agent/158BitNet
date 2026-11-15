/* Dense Qwen3 GGUF runtime (ternary Q2_0 g128 packing) with block verify.
 *
 * Built for speculative decoding with the DSpine draft: eval() runs a block
 * of tokens through one weight pass, records the hidden states of the target
 * layers the draft conditions on, and leaves the KV cache trimmable so only
 * the accepted prefix survives. */
#include "qwen3.h"
#include "gguf.h"
#include "unicode_data.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define Q3_NEON 1
#include <arm_neon.h>
#endif

#define Q3_QK      128
#define Q3_BLK     34
#define Q3_TYPE_Q2 42
#define Q3_TYPE_F32 0
#define Q3_TYPE_BF16 30
#define Q3_MAX_B   16

/* ------------------------------------------------------------------ */
/* thread pool                                                        */
/* ------------------------------------------------------------------ */

typedef void (*q3_job_fn)(void *arg, int tid, int nthreads);

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv_start;
    pthread_cond_t cv_done;
    pthread_t th[16];
    int nthreads;
    volatile int started[16];
    int last[16];
    unsigned gen;
    int active;
    int quit;
    q3_job_fn fn;
    void *arg;
} q3_pool;

static void *q3_worker(void *p) {
    q3_pool *pool = (q3_pool *)p;
    int tid = -1, i;
    pthread_mutex_lock(&pool->mu);
    for (i = 1; i < pool->nthreads; ++i) {
        if (!pool->started[i]) { pool->started[i] = 1; tid = i; break; }
    }
    if (tid < 0) { pthread_mutex_unlock(&pool->mu); return NULL; }
    for (;;) {
        while (pool->gen == (unsigned)pool->last[tid] && !pool->quit)
            pthread_cond_wait(&pool->cv_start, &pool->mu);
        if (pool->quit) break;
        pool->last[tid] = (int)pool->gen;
        {
            q3_job_fn fn = pool->fn;
            void *arg = pool->arg;
            pthread_mutex_unlock(&pool->mu);
            fn(arg, tid, pool->nthreads);
            pthread_mutex_lock(&pool->mu);
        }
        if (--pool->active == 0) pthread_cond_signal(&pool->cv_done);
    }
    pthread_mutex_unlock(&pool->mu);
    return NULL;
}

static void q3_pool_init(q3_pool *pool, int nthreads) {
    int i;
    memset(pool, 0, sizeof(*pool));
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    pool->nthreads = nthreads;
    pthread_mutex_init(&pool->mu, NULL);
    pthread_cond_init(&pool->cv_start, NULL);
    pthread_cond_init(&pool->cv_done, NULL);
    pool->started[0] = 1;
    for (i = 1; i < nthreads; ++i) {
        if (pthread_create(&pool->th[i], NULL, q3_worker, pool) != 0) {
            pool->nthreads = i;
            break;
        }
    }
    if (pool->nthreads > 1) {
        for (;;) {
            int ready = 1;
            for (i = 1; i < pool->nthreads; ++i) if (!pool->started[i]) ready = 0;
            if (ready) break;
        }
    }
}

static void q3_pool_run(q3_pool *pool, q3_job_fn fn, void *arg) {
    if (pool->nthreads <= 1) { fn(arg, 0, 1); return; }
    pthread_mutex_lock(&pool->mu);
    pool->fn = fn;
    pool->arg = arg;
    pool->active = pool->nthreads - 1;
    pool->gen++;
    pthread_cond_broadcast(&pool->cv_start);
    pthread_mutex_unlock(&pool->mu);
    fn(arg, 0, pool->nthreads);
    pthread_mutex_lock(&pool->mu);
    while (pool->active > 0) pthread_cond_wait(&pool->cv_done, &pool->mu);
    pthread_mutex_unlock(&pool->mu);
}

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static float q3_f16(uint16_t v) {
    uint32_t sign = (uint32_t)(v >> 15) << 31;
    uint32_t exp = (v >> 10) & 0x1f;
    uint32_t man = v & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((man & 0x400) == 0) { man <<= 1; --e; }
            man &= 0x3ff;
            bits = sign | (uint32_t)(e + 127) << 23 | man << 13;
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | man << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | man << 13;
    }
    {
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
}

static float q3_bf16(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void q3_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float s = 0.0f;
    int i;
    for (i = 0; i < n; ++i) s += x[i] * x[i];
    s = 1.0f / sqrtf(s / (float)n + eps);
    for (i = 0; i < n; ++i) out[i] = x[i] * s * w[i];
}

static float q3_silu(float x) { return x / (1.0f + expf(-x)); }

static int64_t q3_md_int(const gguf_file_t *g, const char *key, int64_t def) {
    const gguf_metadata_t *e = gguf_find_metadata(g, key);
    if (e == NULL) return def;
    if (e->type == GGUF_TYPE_UINT32) return (int64_t)e->value.u64;
    if (e->type == GGUF_TYPE_INT32) return (int64_t)e->value.i64;
    return (int64_t)e->value.u64;
}

static float q3_md_float(const gguf_file_t *g, const char *key, float def) {
    const gguf_metadata_t *e = gguf_find_metadata(g, key);
    if (e == NULL) return def;
    if (e->type == GGUF_TYPE_FLOAT32) return e->array_floats ? e->array_floats[0] : (float)e->value.f64;
    return (float)e->value.f64;
}

static const char *q3_md_str(const gguf_file_t *g, const char *key) {
    const gguf_metadata_t *e = gguf_find_metadata(g, key);
    return e == NULL ? NULL : e->string_value;
}

/* ------------------------------------------------------------------ */
/* weights                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *data;
    int type;
    int n_in;
    int n_out;
    size_t stride;
} q3_w;

typedef struct {
    q3_w attn_norm, attn_q_norm, attn_k_norm;
    q3_w attn_q, attn_k, attn_v, attn_o;
    q3_w ffn_norm, ffn_gate, ffn_up, ffn_down;
} q3_layer;

struct q3_tok;
typedef struct q3_tok q3_tok;

struct qwen3_model {
    gguf_file_t g;
    char path[512];
    int n_vocab, n_embd, n_layer, n_head, n_head_kv, head_dim, n_ff;
    int kv_dim;
    int rope_n_dims;
    float eps, rope_base, rope_scale, rope_orig;
    int has_yarn;
    float *rope_cos;
    float *rope_sin;
    int n_rope_pos;
    q3_w tok_embd, out_norm, out;
    q3_layer *layers;
    int eos;
    q3_pool pool;
    q3_tok *tok;
};

struct qwen3_ctx {
    qwen3_model_t *m;
    int n_ctx;
    float *kcache;
    float *vcache;
    int pos;
    float *x, *xn, *q, *k, *v, *att, *proj, *gate, *up;
    int8_t *xp;
    float *xscale;
    int32_t *bsums;
    float *scores;
};

static int q3_find(const gguf_file_t *g, const char *name, const gguf_tensor_t **out) {
    uint64_t i;
    for (i = 0; i < g->tensor_count; ++i) {
        if (strcmp(g->tensors[i].name, name) == 0) { *out = &g->tensors[i]; return 0; }
    }
    fprintf(stderr, "qwen3: missing tensor %s\n", name);
    return -1;
}

static int q3_bind(const gguf_file_t *g, const char *name, int n_in, int n_out, q3_w *w) {
    const gguf_tensor_t *t;
    if (q3_find(g, name, &t) != 0) return -1;
    w->data = (uint8_t *)gguf_get_tensor_ptr(g, t);
    if (w->data == NULL) { fprintf(stderr, "qwen3: no data for %s\n", name); return -1; }
    w->type = (int)t->type;
    w->n_in = n_in;
    w->n_out = n_out;
    if (t->type == Q3_TYPE_Q2) {
        if ((n_in % Q3_QK) != 0) { fprintf(stderr, "qwen3: %s n_in=%d\n", name, n_in); return -1; }
        w->stride = (size_t)(n_in / Q3_QK) * Q3_BLK;
    } else if (t->type == Q3_TYPE_F32) {
        w->stride = (size_t)n_in * 4u;
    } else if (t->type == Q3_TYPE_BF16) {
        w->stride = (size_t)n_in * 2u;
    } else {
        fprintf(stderr, "qwen3: %s unsupported type %u\n", name, t->type);
        return -1;
    }
    if (t->dims[0] != (uint64_t)n_in || (n_out > 1 && t->dims[1] != (uint64_t)n_out)) {
        fprintf(stderr, "qwen3: %s shape [%llu,%llu] wanted [%d,%d]\n", name,
                (unsigned long long)t->dims[0],
                t->dims[1] ? (unsigned long long)t->dims[1] : 0ull, n_in, n_out);
        return -1;
    }
    return 0;
}

static float q3_w_get(const q3_w *w, int row, int col) {
    if (w->type == Q3_TYPE_F32) return ((const float *)w->data)[(size_t)row * w->n_in + col];
    if (w->type == Q3_TYPE_BF16) return q3_bf16(((const uint16_t *)w->data)[(size_t)row * w->n_in + col]);
    {
        const uint8_t *blk = w->data + (size_t)row * w->stride + (size_t)(col / Q3_QK) * Q3_BLK;
        float d = q3_f16((uint16_t)(blk[32] | (blk[33] << 8)));
        int c = col % Q3_QK;
        int code = (blk[c / 4] >> (2 * (c % 4))) & 3;
        return (float)(code - 1) * d;
    }
}

/* ------------------------------------------------------------------ */
/* Q2_0 g128 matmul, batched over activations                         */
/* ------------------------------------------------------------------ */

static uint8_t q3_lut[16];

static void q3_lut_init(void) {
    int i;
    for (i = 0; i < 16; ++i) {
        signed char v[4];
        int j, n = 0;
        for (j = 0; j < 4; ++j) {
            int code = (i >> (2 * j)) & 3;
            v[j] = (signed char)(code - 1);
        }
        /* plane s collects code s of each byte */
        for (j = 3; j >= 0; --j) q3_lut[i * 4 + j] = (uint8_t)v[0];
        (void)n;
    }
}

static void q3_quant_row(int8_t *q, float *scale, int32_t *bsums, const float *x, int n) {
    float amax = 0.0f;
    int i;
    for (i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    {
        float s = amax / 127.0f;
        float inv;
        if (s < 1e-8f) s = 1e-8f;
        *scale = s;
        inv = 1.0f / s;
        for (i = 0; i < n; ++i) {
            int val = (int)lrintf(x[i] * inv);
            if (val > 127) val = 127;
            if (val < -127) val = -127;
            q[i] = (int8_t)val;
        }
    }
    for (i = 0; i < n / Q3_QK; ++i) {
        int32_t s = 0;
        int j;
        for (j = 0; j < Q3_QK; ++j) s += q[i * Q3_QK + j];
        bsums[i] = s;
    }
}

/* Reorder a quantized row into the plane/lane layout the VTBL unpack expects. */
static void q3_scatter(const int8_t *q, int8_t *xp, int n) {
    int k, c, s, j;
    for (k = 0; k < n / Q3_QK; ++k) {
        for (c = 0; c < 2; ++c) {
            for (s = 0; s < 4; ++s) {
                for (j = 0; j < 16; ++j) {
                    xp[(k * 2 + c) * 64 + s * 16 + j] = q[k * Q3_QK + c * 64 + 4 * j + s];
                }
            }
        }
    }
}

typedef struct {
    const q3_w *w;
    const int8_t *xp;
    const float *xscale;
    const int32_t *bsums;
    float *y;
    int B;
    int row_begin, row_end;
} q3_mm_job;

static void q3_mm_rows(const q3_mm_job *j) {
    const q3_w *w = j->w;
    int nblk = w->n_in / Q3_QK;
    int row, b, k, c;
    (void)c;
    for (row = j->row_begin; row < j->row_end; ++row) {
        const uint8_t *r = w->data + (size_t)row * w->stride;
        float accf[Q3_MAX_B];
        for (b = 0; b < j->B; ++b) accf[b] = 0.0f;
        if (w->type != Q3_TYPE_Q2) {
            for (b = 0; b < j->B; ++b) {
                float s = 0.0f;
                for (k = 0; k < w->n_in; ++k) {
                    s += q3_w_get(w, row, k) * ((const float *)j->xscale)[b * w->n_in + k];
                }
                accf[b] = s;
            }
        } else {
            for (k = 0; k < nblk; ++k) {
                const uint8_t *blk = r + (size_t)k * Q3_BLK;
                float d = q3_f16((uint16_t)(blk[32] | (blk[33] << 8)));
                for (c = 0; c < 2; ++c) {
                    int32_t acci[Q3_MAX_B];
                    for (b = 0; b < j->B; ++b) acci[b] = 0;
#if defined(Q3_NEON)
                    {
                        uint8x16_t lut = vld1q_u8(q3_lut);
                        uint8x16_t codes = vld1q_u8(blk + c * 16);
                        int8x16_t c0 = vreinterpretq_s8_u8(vqtbl1q_u8(lut, vandq_u8(codes, vdupq_n_u8(0x0f))));
                        int8x16_t c1 = vreinterpretq_s8_u8(vqtbl1q_u8(lut, vandq_u8(vshrq_n_u8(codes, 2), vdupq_n_u8(0x0f))));
                        int8x16_t c2 = vreinterpretq_s8_u8(vqtbl1q_u8(lut, vshrq_n_u8(codes, 4)));
                        int8x16_t c3 = vreinterpretq_s8_u8(vqtbl1q_u8(lut, vshrq_n_u8(codes, 6)));
                        for (b = 0; b < j->B; ++b) {
                            const int8_t *xb = j->xp + (size_t)b * w->n_in + (size_t)(k * 2 + c) * 64;
                            int32x4_t t = vdupq_n_s32(0);
                            t = vdotq_s32(t, c0, vld1q_s8(xb + 0));
                            t = vdotq_s32(t, c1, vld1q_s8(xb + 16));
                            t = vdotq_s32(t, c2, vld1q_s8(xb + 32));
                            t = vdotq_s32(t, c3, vld1q_s8(xb + 48));
                            acci[b] = vaddvq_s32(t);
                        }
                    }
#else
                    for (b = 0; b < j->B; ++b) {
                        const int8_t *xb = j->xp + (size_t)b * w->n_in + (size_t)(k * 2 + c) * 64;
                        int32_t s = 0;
                        int t;
                        for (t = 0; t < 64; ++t) {
                            int col = c * 64 + t;
                            int code = (blk[col / 4] >> (2 * (col % 4))) & 3;
                            s += (code - 1) * (int)xb[t];
                        }
                        acci[b] = s;
                    }
#endif
                    for (b = 0; b < j->B; ++b) {
                        int64_t sv = (int64_t)acci[b] - (int64_t)j->bsums[(size_t)b * nblk + k];
                        accf[b] += (float)sv * d * j->xscale[b];
                    }
                }
            }
        }
        for (b = 0; b < j->B; ++b) j->y[(size_t)b * w->n_out + row] = accf[b];
    }
}

static void q3_mm_worker(void *arg, int tid, int nthreads) {
    q3_mm_job *j = (q3_mm_job *)arg;
    int total = j->row_end - j->row_begin;
    int per = (total + nthreads - 1) / nthreads;
    q3_mm_job sub = *j;
    int start = j->row_begin + tid * per;
    int end = start + per;
    if (start >= j->row_end) return;
    if (end > j->row_end) end = j->row_end;
    sub.row_begin = start;
    sub.row_end = end;
    q3_mm_rows(&sub);
}

/* y[b][row] = sum_k W[row][k] * xn[b][k]; xp/xscale/bsums must be prepared. */
static void q3_mm(qwen3_model_t *m, const q3_w *w,
                  const int8_t *xp, const float *xscale, const int32_t *bsums,
                  float *y, int B) {
    q3_mm_job j;
    j.w = w;
    j.xp = xp;
    j.xscale = xscale;
    j.bsums = bsums;
    j.y = y;
    j.B = B;
    j.row_begin = 0;
    j.row_end = w->n_out;
    q3_pool_run(&m->pool, q3_mm_worker, &j);
}

/* Prepare the quantized/permuted activation block for a batch of rows. */
static void q3_prep(qwen3_ctx_t *c, const float *xn, int B, int n) {
    int b;
    for (b = 0; b < B; ++b) {
        int8_t *q = c->xq + (size_t)b * n;
        q3_quant_row(q, &c->xscale[b], c->bsums + (size_t)b * (n / Q3_QK), xn + (size_t)b * n, n);
        q3_scatter(q, c->xp + (size_t)b * n, n);
    }
}
