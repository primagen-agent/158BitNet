/* Pure-C inference for the Qwen3.5-hybrid "Ternary-Bonsai" GGUF family:
 * architecture "qwen35", PTQ1_0 ternary weights and Prism Hadamard folding.
 *
 * Layout of the model (Ternary-Bonsai-2-27B-PTQ1_0):
 *   64 blocks, hidden 5120, FFN 17408, 24 q heads / 4 kv heads of 256,
 *   every 4th block is full attention, the rest are gated delta net (GDN).
 *   Weights are folded (W_f = W*D*H), so every matmul input is first
 *   transformed as x' = H(D*x); the embedding lookup is the inverse
 *   (x = D*H*z), and the GDN output projection also permutes its heads. */

#include "bonsai.h"
#include "gguf.h"
#include "minicpm.h"
#include "quant_ptq1_0.h"
#include "quant_tq2_0.h"
#include "unicode_data.h"
#include "gpu_ops.h"
#if defined(BITNET_ENABLE_VULKAN)
#include "vk_backend.h"
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define BQ_F32  0u
#define BQ_BF16 30u
#define BQ_PTQ  143u

/* ---- stage profiling (BQ_PROFILE=1) -------------------------------- */
static double bq_prof_clock(void);
static int bq_prof_enabled(void);
/* non-matvec stage split, printed by bonsai_profile_dump */
static double g_bq_t_prep, g_bq_t_gdn, g_bq_t_attn, g_bq_t_norm, g_bq_t_ew, g_bq_t_conv;


/* ------------------------------------------------------------------ */
/* basic helpers                                                      */
/* ------------------------------------------------------------------ */

static float bq_bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float bq_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float bq_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float bq_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static void bq_rms_norm(const float *x, const float *w, int n, float eps, float *out) {
    double t_n = bq_prof_enabled() ? bq_prof_clock() : 0.0;
    double sum = 0.0;
    float scale;
    int i;
    for (i = 0; i < n; ++i) sum += (double)x[i] * x[i];
    scale = 1.0f / sqrtf((float)(sum / n) + eps);
    for (i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
    if (t_n != 0.0) g_bq_t_norm += bq_prof_clock() - t_n;
}

static void bq_l2_norm(float *x, int n, float eps) {
    double t_n = bq_prof_enabled() ? bq_prof_clock() : 0.0;
    double sum = 0.0;
    float scale;
    int i;
    for (i = 0; i < n; ++i) sum += (double)x[i] * x[i];
    scale = 1.0f / fmaxf(sqrtf((float)sum), eps);
    for (i = 0; i < n; ++i) x[i] *= scale;
    if (t_n != 0.0) g_bq_t_norm += bq_prof_clock() - t_n;
}

/* normalized Sylvester-Walsh Hadamard transform over 1024-wide blocks */
static void bq_fwht_block(float *x, int n) {
    int len, i, j;
    for (len = 1; len < n; len <<= 1) {
        for (i = 0; i < n; i += 2 * len) {
            float *a = x + i;
            float *b = x + i + len;
            for (j = 0; j < len; ++j) {
                float u = a[j];
                float v = b[j];
                a[j] = u + v;
                b[j] = u - v;
            }
        }
    }
}

static void bq_hadamard(float *x, int n, int block) {
    const float scale = 1.0f / sqrtf((float)block);
    int i, k;
    for (i = 0; i < n; i += block) {
        bq_fwht_block(x + i, block);
        for (k = 0; k < block; ++k) x[i + k] *= scale;
    }
}

/* x' = H(D*x), the transform every folded weight expects */
static void bq_prep_input(float *dst, const float *src, int n, const float *signs, int block) {
    int i;
    if (signs != NULL) {
        for (i = 0; i < n; ++i) dst[i] = src[i] * signs[i];
    } else if (dst != src) {
        memcpy(dst, src, (size_t)n * sizeof(float));
    }
    bq_hadamard(dst, n, block);
}

/* ------------------------------------------------------------------ */
/* thread pool                                                        */
/* ------------------------------------------------------------------ */

typedef void (*bq_job_fn)(void *arg, int tid, int nthreads);

typedef struct {
    int nthreads;
    pthread_t th[16];
    pthread_mutex_t mu;
    pthread_cond_t cv_start;
    pthread_cond_t cv_done;
    int gen;
    int quit;
    int active;
    int started[16];
    int last[16];
    bq_job_fn fn;
    void *arg;
} bq_pool_t;

static void *bq_worker(void *p) {
    bq_pool_t *pool = (bq_pool_t *)p;
    int tid = -1;
    int i;
    pthread_mutex_lock(&pool->mu);
    for (i = 1; i < pool->nthreads; ++i) {
        if (!pool->started[i]) {
            pool->started[i] = 1;
            tid = i;
            break;
        }
    }
    if (tid < 0) {
        pthread_mutex_unlock(&pool->mu);
        return NULL;
    }
    for (;;) {
        while (pool->gen == pool->last[tid] && !pool->quit) {
            pthread_cond_wait(&pool->cv_start, &pool->mu);
        }
        if (pool->quit) break;
        pool->last[tid] = pool->gen;
        {
            bq_job_fn fn = pool->fn;
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

static int bq_pool_init(bq_pool_t *pool, int nthreads) {
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
        if (pthread_create(&pool->th[i], NULL, bq_worker, pool) != 0) {
            /* keep the workers we already have */
            pool->nthreads = i;
            break;
        }
    }
    if (pool->nthreads > 1) {
        /* wait until every worker has claimed its id */
        for (;;) {
            int ready = 1;
            pthread_mutex_lock(&pool->mu);
            for (i = 1; i < pool->nthreads; ++i) if (!pool->started[i]) ready = 0;
            pthread_mutex_unlock(&pool->mu);
            if (ready) break;
        }
    }
    return 0;
}

static void bq_pool_run(bq_pool_t *pool, bq_job_fn fn, void *arg) {
    if (pool->nthreads <= 1) {
        fn(arg, 0, 1);
        return;
    }
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

/* Block-parallel prep transform.  The Walsh-Hadamard blocks are independent,
 * and this runs four times per layer over 5-27 blocks, which makes it the
 * largest non-matvec cost of a forward pass.  Each block applies its own signs
 * before transforming, so splitting by block does not change any arithmetic. */
typedef struct {
    float *dst;
    const float *signs;
    int block;
    int nblocks;
    int next;
} bq_prep_arg_t;

static void bq_prep_job(void *p, int tid, int nthreads) {
    bq_prep_arg_t *a = (bq_prep_arg_t *)p;
    const float scale = 1.0f / sqrtf((float)a->block);
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int b = __atomic_fetch_add(&a->next, 1, __ATOMIC_RELAXED);
        float *blk;
        int k;
        if (b >= a->nblocks) break;
        blk = a->dst + (size_t)b * a->block;
        if (a->signs != NULL) {
            const float *sg = a->signs + (size_t)b * a->block;
            for (k = 0; k < a->block; ++k) blk[k] *= sg[k];
        }
        bq_fwht_block(blk, a->block);
        for (k = 0; k < a->block; ++k) blk[k] *= scale;
    }
}

static void bq_prep_input_p(bq_pool_t *pool, float *dst, const float *src, int n,
                            const float *signs, int block) {
    double t_pi = bq_prof_enabled() ? bq_prof_clock() : 0.0;
    bq_prep_arg_t a;
    if (block <= 0 || n % block != 0) {
        bq_prep_input(dst, src, n, signs, block);
        return;
    }
    if (dst != src) memcpy(dst, src, (size_t)n * sizeof(float));
    a.dst = dst;
    a.signs = signs;
    a.block = block;
    a.nblocks = n / block;
    a.next = 0;
    /* The transform is a few microseconds of work per block, and waking the
     * pool costs an order of magnitude more than that for the sizes a decode
     * step uses, so stay on one thread until the job is big enough to pay for
     * the handoff. */
    if (n < 32768) bq_prep_job(&a, 0, 1);
    else bq_pool_run(pool, bq_prep_job, &a);
    if (t_pi != 0.0) g_bq_t_prep += bq_prof_clock() - t_pi;
}

static void bq_pool_free(bq_pool_t *pool) {
    int i;
    if (pool->nthreads > 1) {
        pthread_mutex_lock(&pool->mu);
        pool->quit = 1;
        pthread_cond_broadcast(&pool->cv_start);
        pthread_mutex_unlock(&pool->mu);
        for (i = 1; i < pool->nthreads; ++i) pthread_join(pool->th[i], NULL);
    }
    pthread_cond_destroy(&pool->cv_done);
    pthread_cond_destroy(&pool->cv_start);
    pthread_mutex_destroy(&pool->mu);
    pool->nthreads = 0;
}

/* ------------------------------------------------------------------ */
/* weights and matvec                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *data;
    uint32_t type;
    int n_in;
    int n_out;
    size_t row_stride;
    void *vk_buf;   /* GPU-visible copy of the rows, NULL when CPU-only */
    void *p2;       /* 2-bit repacked row pairs for the i8mm kernel, or NULL */
    float *p2d;     /* their per-row-pair scales, 2*nb floats per pair */
} bq_w_t;

static size_t bq_row_stride(uint32_t type, int n_in) {
    switch (type) {
        case BQ_PTQ: return bitnet_ptq1_0_row_size(n_in);
        case BQ_BF16: return (size_t)n_in * 2;
        default: return (size_t)n_in * 4;
    }
}

static float bq_dot_row(const bq_w_t *w, int row, const float *x) {
    const uint8_t *r = w->data + (size_t)row * w->row_stride;
    int i;
    float sum;
    switch (w->type) {
        case BQ_PTQ:
            return bitnet_ptq1_0_dot_row(r, x, w->n_in);
        case BQ_BF16: {
            const uint16_t *p = (const uint16_t *)r;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (i = 0; i + 3 < w->n_in; i += 4) {
                acc0 += bq_bf16_to_f32(p[i + 0]) * x[i + 0];
                acc1 += bq_bf16_to_f32(p[i + 1]) * x[i + 1];
                acc2 += bq_bf16_to_f32(p[i + 2]) * x[i + 2];
                acc3 += bq_bf16_to_f32(p[i + 3]) * x[i + 3];
            }
            for (; i < w->n_in; ++i) acc0 += bq_bf16_to_f32(p[i]) * x[i];
            sum = (acc0 + acc1) + (acc2 + acc3);
            return sum;
        }
        default: {
            const float *p = (const float *)r;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (i = 0; i + 3 < w->n_in; i += 4) {
                acc0 += p[i + 0] * x[i + 0];
                acc1 += p[i + 1] * x[i + 1];
                acc2 += p[i + 2] * x[i + 2];
                acc3 += p[i + 3] * x[i + 3];
            }
            for (; i < w->n_in; ++i) acc0 += p[i] * x[i];
            sum = (acc0 + acc1) + (acc2 + acc3);
            return sum;
        }
    }
}

typedef struct {
    const bq_w_t *w;
    const float *x;
    const void *xq;  /* q8_0 form of x, used for PTQ rows (NULL otherwise) */
    float *y;
    int row0;   /* first row this call owns; the GPU took [0,row0) */
    int row1;
    int next;   /* next row to claim, shared by all workers */
    int chunk;  /* rows claimed per atomic */
} bq_mv_arg_t;

/* Chunked work stealing.  Static round-robin splits rows evenly per thread,
 * which leaves the fast cores idle while the little cores finish their share;
 * this phone is 4xA55 / 3xA77 / 1 prime.  Claiming runs of rows keeps every
 * core busy until the work is gone, and each run is contiguous in memory. */
static void bq_mv_job(void *p, int tid, int nthreads) {
    bq_mv_arg_t *a = (bq_mv_arg_t *)p;
    const int last = a->row1;
    const int step = a->chunk;
    (void)tid;
    (void)nthreads;
    for (;;) {
        int start = __atomic_fetch_add(&a->next, step, __ATOMIC_RELAXED);
        int end = start + step;
        if (start >= last) break;
        if (end > last) end = last;
        if (a->xq != NULL) {
            const int n_in = a->w->n_in;
            const size_t stride = a->w->row_stride;
            for (int j = start; j < end; ++j) {
                a->y[j] = bitnet_ptq1_0_dot_row_q8(a->w->data + (size_t)j * stride, a->xq, n_in);
            }
        } else {
            for (int j = start; j < end; ++j) a->y[j] = bq_dot_row(a->w, j, a->x);
        }
    }
}


/* ------------------------------------------------------------------ */
/* tokenizer: GPT-2 byte-level BPE with the Qwen3.5 pre-tokenizer      */
/* ------------------------------------------------------------------ */

typedef struct {
    char **tokens;      /* owned by the gguf file */
    int32_t *types;
    int n_vocab;

    int32_t *tok_head;
    int32_t *tok_next;
    uint32_t tok_mask;

    char **merges;      /* owned by the gguf file: "left right" */
    int32_t *merge_head;
    int32_t *merge_next;
    int32_t *merge_lpos;  /* offset of the separating space */
    uint32_t *merge_hash;
    uint32_t merge_mask;
    int n_merges;

    int32_t byte_to_cpt[256];
    int32_t cpt_to_byte[0x200];

    int eos;
    int bos;
    int pad;
    int add_bos;
} bq_tok_t;

static uint32_t bq_hash_bytes(const char *s, int n) {
    uint32_t h = 2166136261u;
    int i;
    for (i = 0; i < n; ++i) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t bq_hash_str(const char *s) {
    return bq_hash_bytes(s, (int)strlen(s));
}

static uint32_t bq_hash_pair(const char *a, int an, const char *b, int bn) {
    uint32_t h = bq_hash_bytes(a, an);
    int i;
    for (i = 0; i < bn; ++i) {
        h ^= (uint8_t)b[i];
        h *= 16777619u;
    }
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

static uint32_t bq_next_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static void bq_byte_tables(bq_tok_t *t) {
    int i, n = 0;
    int seen[256];
    memset(seen, 0, sizeof(seen));
    for (i = 0; i < 0x200; ++i) t->cpt_to_byte[i] = -1;

    for (i = 0x21; i <= 0x7E; ++i) seen[i] = 1;
    for (i = 0xA1; i <= 0xAC; ++i) seen[i] = 1;
    for (i = 0xAE; i <= 0xFF; ++i) seen[i] = 1;
    for (i = 0; i < 256; ++i) {
        int cpt;
        if (seen[i]) {
            cpt = i;
        } else {
            cpt = 256 + n;
            ++n;
        }
        t->byte_to_cpt[i] = cpt;
        t->cpt_to_byte[cpt] = i;
    }
}

static int bq_tok_lookup(const bq_tok_t *t, const char *s, int n) {
    uint32_t h = bq_hash_bytes(s, n);
    int idx = t->tok_head[h & t->tok_mask];
    while (idx >= 0) {
        const char *tok = t->tokens[idx];
        if ((int)strlen(tok) == n && memcmp(tok, s, (size_t)n) == 0) return idx;
        idx = t->tok_next[idx];
    }
    return -1;
}

static int bq_merge_rank(const bq_tok_t *t, const char *a, int an, const char *b, int bn) {
    uint32_t h = bq_hash_pair(a, an, b, bn);
    int idx = t->merge_head[h & t->merge_mask];
    while (idx >= 0) {
        if (t->merge_hash[idx] == h) {
            const char *m = t->merges[idx];
            int lp = t->merge_lpos[idx];
            int rl = (int)strlen(m) - lp - 1;
            if (lp == an && rl == bn && memcmp(m, a, (size_t)an) == 0 && memcmp(m + lp + 1, b, (size_t)bn) == 0) {
                return idx;
            }
        }
        idx = t->merge_next[idx];
    }
    return -1;
}

static int bq_tok_load(bq_tok_t *t, gguf_file_t *g) {
    const gguf_metadata_t *e;
    int i;

    memset(t, 0, sizeof(*t));
    bq_byte_tables(t);

    e = gguf_find_metadata(g, "tokenizer.ggml.tokens");
    if (e == NULL || e->array_strings == NULL) return -1;
    t->tokens = e->array_strings;
    t->n_vocab = (int)e->array_count;

    e = gguf_find_metadata(g, "tokenizer.ggml.token_type");
    if (e == NULL || e->array_ints == NULL) return -1;
    t->types = e->array_ints;

    e = gguf_find_metadata(g, "tokenizer.ggml.merges");
    if (e == NULL || e->array_strings == NULL) return -1;
    t->merges = e->array_strings;
    t->n_merges = (int)e->array_count;

    e = gguf_find_metadata(g, "tokenizer.ggml.eos_token_id");
    t->eos = e ? (int)e->value.u64 : -1;
    e = gguf_find_metadata(g, "tokenizer.ggml.bos_token_id");
    t->bos = e ? (int)e->value.u64 : -1;
    e = gguf_find_metadata(g, "tokenizer.ggml.padding_token_id");
    t->pad = e ? (int)e->value.u64 : t->bos;
    e = gguf_find_metadata(g, "tokenizer.ggml.add_bos_token");
    t->add_bos = e ? (int)e->value.u64 : 0;

    t->tok_mask = bq_next_pow2((uint32_t)t->n_vocab * 2) - 1;
    t->tok_head = (int32_t *)malloc(((size_t)t->tok_mask + 1) * sizeof(int32_t));
    t->tok_next = (int32_t *)malloc((size_t)t->n_vocab * sizeof(int32_t));
    if (t->tok_head == NULL || t->tok_next == NULL) return -1;
    for (i = 0; i <= (int)t->tok_mask; ++i) t->tok_head[i] = -1;
    for (i = 0; i < t->n_vocab; ++i) {
        uint32_t h = bq_hash_str(t->tokens[i]);
        t->tok_next[i] = t->tok_head[h & t->tok_mask];
        t->tok_head[h & t->tok_mask] = i;
    }

    t->merge_mask = bq_next_pow2((uint32_t)t->n_merges * 2) - 1;
    t->merge_head = (int32_t *)malloc(((size_t)t->merge_mask + 1) * sizeof(int32_t));
    t->merge_next = (int32_t *)malloc((size_t)t->n_merges * sizeof(int32_t));
    t->merge_lpos = (int32_t *)malloc((size_t)t->n_merges * sizeof(int32_t));
    t->merge_hash = (uint32_t *)malloc((size_t)t->n_merges * sizeof(uint32_t));
    if (!t->merge_head || !t->merge_next || !t->merge_lpos || !t->merge_hash) return -1;
    for (i = 0; i <= (int)t->merge_mask; ++i) t->merge_head[i] = -1;
    for (i = 0; i < t->n_merges; ++i) {
        const char *m = t->merges[i];
        const char *sp = strchr(m, ' ');
        int lp = sp ? (int)(sp - m) : 0;
        uint32_t h;
        t->merge_lpos[i] = lp;
        h = bq_hash_pair(m, lp, m + lp + 1, (int)strlen(m) - lp - 1);
        t->merge_hash[i] = h;
        t->merge_next[i] = t->merge_head[h & t->merge_mask];
        t->merge_head[h & t->merge_mask] = i;
    }
    return 0;
}

static void bq_tok_free(bq_tok_t *t) {
    free(t->tok_head);
    free(t->tok_next);
    free(t->merge_head);
    free(t->merge_next);
    free(t->merge_lpos);
    free(t->merge_hash);
    memset(t, 0, sizeof(*t));
}

/* UTF-8 helpers */
static int bq_utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

static int bq_utf8_decode(const char *s, int len, uint32_t *out, int max) {
    int i = 0, n = 0;
    while (i < len && n < max) {
        unsigned char c = (unsigned char)s[i];
        int k = bq_utf8_len(c);
        uint32_t cp;
        int j;
        if (i + k > len) k = 1;
        if (k == 1) cp = c;
        else if (k == 2) cp = ((uint32_t)(c & 0x1F) << 6) | (uint32_t)(s[i + 1] & 0x3F);
        else if (k == 3) cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) | (uint32_t)(s[i + 2] & 0x3F);
        else cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12) |
                  ((uint32_t)(s[i + 2] & 0x3F) << 6) | (uint32_t)(s[i + 3] & 0x3F);
        for (j = 1; j < k; ++j) {
            if (((unsigned char)s[i + j] & 0xC0) != 0x80) { k = 1; cp = c; break; }
        }
        out[n++] = cp;
        i += k;
    }
    return n;
}

static int bq_utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int bq_is_letter(uint32_t f) { return (f & BQ_U_LETTER) != 0; }
static int bq_is_mark(uint32_t f) { return (f & BQ_U_ACCENT_MARK) != 0; }
static int bq_is_number(uint32_t f) { return (f & BQ_U_NUMBER) != 0; }
static int bq_is_space(uint32_t f) { return (f & BQ_U_WHITESPACE) != 0; }

/* Qwen3.5 pre-tokenizer, a port of llama.cpp's
 * unicode_regex_split_custom_qwen35():
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
 *   |\s*[\r\n]+|\s+(?!\S)|\s+
 * Writes the codepoint length of each word into lens[] and returns the count. */
static int bq_pretok_qwen35(const uint32_t *cpts, int n, int *lens, int max_words) {
    int nw = 0;
    int pos = 0;

    while (pos < n) {
        uint32_t cpt = cpts[pos];
        uint32_t f = bq_unicode_flags(cpt);
        int start = pos;

        /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t c1 = cpts[pos + 1];
            if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
                goto emit;
            }
            if (pos + 2 < n) {
                uint32_t c2 = cpts[pos + 2];
                if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                    goto emit;
                }
            }
        }

        /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
        if (!(cpt == '\r' || cpt == '\n' || bq_is_number(f))) {
            uint32_t fn = bq_unicode_flags(cpts[pos + 1 < n ? pos + 1 : pos]);
            if (bq_is_letter(f) || bq_is_mark(f) ||
                (pos + 1 < n && (bq_is_letter(fn) || bq_is_mark(fn)))) {
                ++pos;
                while (pos < n) {
                    uint32_t g = bq_unicode_flags(cpts[pos]);
                    if (!bq_is_letter(g) && !bq_is_mark(g)) break;
                    ++pos;
                }
                goto emit;
            }
        }

        /* \p{N} */
        if (bq_is_number(f)) {
            ++pos;
            goto emit;
        }

        /*  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* */
        {
            uint32_t f2 = (cpt == ' ' && pos + 1 < n) ? bq_unicode_flags(cpts[pos + 1]) : f;
            if (!(bq_is_space(f2) || bq_is_letter(f2) || bq_is_mark(f2) || bq_is_number(f2)) && f != 0) {
                if (cpt == ' ') ++pos;
                for (;;) {
                    uint32_t fcur;
                    if (pos >= n) break;
                    fcur = bq_unicode_flags(cpts[pos]);
                    if (fcur == 0 || bq_is_space(fcur) || bq_is_letter(fcur) ||
                        bq_is_mark(fcur) || bq_is_number(fcur)) break;
                    ++pos;
                }
                while (pos < n && (cpts[pos] == '\r' || cpts[pos] == '\n')) ++pos;
                goto emit;
            }
        }

        /* whitespace runs */
        {
            int nws = 0;
            int last_rn = 0;
            while (pos + nws < n && bq_is_space(bq_unicode_flags(cpts[pos + nws]))) {
                if (cpts[pos + nws] == '\r' || cpts[pos + nws] == '\n') last_rn = pos + nws + 1;
                ++nws;
            }
            if (last_rn > 0) {
                pos = last_rn;
                goto emit;
            }
            if (nws > 1 && pos + nws < n) {
                pos += nws - 1;
                goto emit;
            }
            if (nws > 0) {
                pos += nws;
                goto emit;
            }
        }

        /* no match: consume one codepoint */
        ++pos;

    emit:
        if (nw < max_words) lens[nw++] = pos - start;
    }
    return nw;
}

/* byte-level encode a UTF-8 word: every byte becomes a GPT-2 unicode codepoint */
static int bq_byte_encode_word(const bq_tok_t *t, const uint32_t *cpts, int n, char *out, int out_size) {
    int i, o = 0;
    char tmp[4];
    for (i = 0; i < n; ++i) {
        int len = bq_utf8_encode(cpts[i], tmp);
        int j;
        for (j = 0; j < len; ++j) {
            unsigned char b = (unsigned char)tmp[j];
            int k = bq_utf8_encode((uint32_t)t->byte_to_cpt[b], out + o);
            o += k;
            if (o >= out_size) return -1;
        }
    }
    return o;
}

static int bq_bpe_word(bq_tok_t *t, const char *enc, int len, int *out, int max_out) {
    typedef struct { int off; int len; } sym_t;
    sym_t sym[256];
    int nsym = 0;
    int i, o = 0;

    for (i = 0; i < len;) {
        int k = bq_utf8_len((unsigned char)enc[i]);
        if (i + k > len) k = len - i;
        if (nsym >= 256) k = len - i;
        sym[nsym].off = i;
        sym[nsym].len = k;
        ++nsym;
        i += k;
        if (nsym >= 256) {
            /* absurdly long word: emit the rest as one symbol */
            if (i < len) { sym[255].len = len - sym[255].off; }
            break;
        }
    }

    for (;;) {
        int best = -1;
        int best_rank = 0x7fffffff;
        for (i = 0; i + 1 < nsym; ++i) {
            int r = bq_merge_rank(t, enc + sym[i].off, sym[i].len,
                                  enc + sym[i + 1].off, sym[i + 1].len);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best = i;
            }
        }
        if (best < 0) break;
        sym[best].len += sym[best + 1].len;
        memmove(&sym[best + 1], &sym[best + 2], (size_t)(nsym - best - 2) * sizeof(sym_t));
        --nsym;
    }

    for (i = 0; i < nsym; ++i) {
        int id = bq_tok_lookup(t, enc + sym[i].off, sym[i].len);
        if (id >= 0) {
            if (o >= max_out) return -1;
            out[o++] = id;
        } else {
            /* fall back to single byte tokens */
            int p;
            for (p = 0; p < sym[i].len; ++p) {
                int k = bq_utf8_len((unsigned char)enc[sym[i].off + p]);
                int one = bq_tok_lookup(t, enc + sym[i].off + p, k);
                if (one >= 0) {
                    if (o >= max_out) return -1;
                    out[o++] = one;
                }
                p += k - 1;
            }
        }
    }
    return o;
}

static int bq_tokenize_plain(bq_tok_t *t, const char *text, int len, int *out, int max_out) {
    uint32_t *cpts;
    int *lens;
    int n, nw, i, start, o = 0;
    char *enc;

    if (len <= 0) return 0;
    cpts = (uint32_t *)malloc((size_t)len * sizeof(uint32_t));
    lens = (int *)malloc(((size_t)len + 1) * sizeof(int));
    enc = (char *)malloc((size_t)len * 4 + 8);
    if (!cpts || !lens || !enc) {
        free(cpts); free(lens); free(enc);
        return -1;
    }

    n = bq_utf8_decode(text, len, cpts, len);
    nw = bq_pretok_qwen35(cpts, n, lens, n + 1);

    start = 0;
    for (i = 0; i < nw; ++i) {
        int wlen = bq_byte_encode_word(t, cpts + start, lens[i], enc, len * 4);
        int rc;
        start += lens[i];
        if (wlen < 0) { free(cpts); free(lens); free(enc); return -1; }
        rc = bq_bpe_word(t, enc, wlen, out + o, max_out - o);
        if (rc < 0) { free(cpts); free(lens); free(enc); return -1; }
        o += rc;
    }

    free(cpts);
    free(lens);
    free(enc);
    return o;
}

static int bq_tokenize(bq_tok_t *t, const char *text, int *out, int max_out) {
    int len = (int)strlen(text);
    int o = 0, i = 0, seg = 0;

    while (i + 1 < len) {
        if (text[i] == '<' && text[i + 1] == '|') {
            const char *endp = strstr(text + i, "|>");
            if (endp != NULL) {
                int n = (int)(endp - (text + i)) + 2;
                if (n <= 512) {
                    int id = bq_tok_lookup(t, text + i, n);
                    if (id >= 0 && (t->types[id] == 3 || t->types[id] == 4)) {
                        int rc = bq_tokenize_plain(t, text + seg, i - seg, out + o, max_out - o);
                        if (rc < 0) return -1;
                        o += rc;
                        if (o >= max_out) return -1;
                        out[o++] = id;
                        i += n;
                        seg = i;
                        continue;
                    }
                }
            }
        }
        ++i;
    }

    if (seg < len) {
        int rc = bq_tokenize_plain(t, text + seg, len - seg, out + o, max_out - o);
        if (rc < 0) return -1;
        o += rc;
    }
    return o;
}

static int bq_detokenize(const bq_tok_t *t, int id, char *out, int out_size) {
    const char *s;
    uint32_t cpts[256];
    int n, i, o = 0;

    if (id < 0 || id >= t->n_vocab) return 0;
    if (t->types[id] == 3) return 0; /* control tokens carry no text */
    s = t->tokens[id];
    n = bq_utf8_decode(s, (int)strlen(s), cpts, 256);
    for (i = 0; i < n; ++i) {
        int32_t b = cpts[i] < 0x200 ? t->cpt_to_byte[cpts[i]] : -1;
        if (b >= 0) {
            if (o < out_size) out[o++] = (char)b;
        } else {
            char tmp[4];
            int k = bq_utf8_encode(cpts[i], tmp);
            int j;
            for (j = 0; j < k && o < out_size; ++j) out[o++] = tmp[j];
        }
    }
    return o;
}

/* ------------------------------------------------------------------ */
/* model                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int is_full;
    float *attn_norm;
    float *post_attn_norm;

    /* gated delta net */
    bq_w_t wqkv;
    bq_w_t wqkv_gate;
    bq_w_t wssm_out;
    bq_w_t walpha;
    bq_w_t wbeta;
    float *ssm_a;
    float *ssm_dt;
    float *ssm_norm;
    float *ssm_conv;

    /* full attention */
    bq_w_t wq;
    bq_w_t wk;
    bq_w_t wv;
    bq_w_t wo;
    float *q_norm;
    float *k_norm;

    /* ffn */
    bq_w_t ffn_up;
    bq_w_t ffn_gate;
    bq_w_t ffn_down;
} bq_layer_t;

struct bonsai_model {
    gguf_file_t gguf;
    bq_tok_t tok;
    bq_pool_t *pool;

    /* Which family this GGUF is, decided from general.architecture at load
     * time, so one binary serves every format it knows.  arch_prefix is the
     * same name again, used to read "<arch>.<key>" metadata without a literal
     * per family. */
    bonsai_arch_t arch;
    char arch_prefix[16];

    /* Set when the file's family is minicpm: its graph, weights and kernels
     * live in minicpm.c and every public entry point below dispatches to
     * them.  NULL for qwen35, the family this file implements itself. */
    void *mc;

    int n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, n_rot;
    int n_ctx_train, full_interval, block_size, gdn_v_grouped;
    int ssm_conv_k, ssm_state, ssm_groups, ssm_dt_rank, ssm_inner;
    int n_vocab, n_gdn, n_full;

    float eps, rope_base;
    int rope_sections[4];

    int n_sign;
    int sign_width[8];
    float *sign_data[8];

    float *rope_freq;

    bq_w_t tok_embd;
    float *out_norm;
    bq_w_t output;

    bq_layer_t *layers;

    void **owned;   /* tensor copies that outlive the file mapping */
    int n_owned;
    int mmap_live;  /* some weight still points into the file mapping */

#if defined(BITNET_ENABLE_VULKAN)
    bq_vk_t *vk;
    bq_vk_buf_t **vk_bufs;   /* GPU-side weight allocations, freed with the model */
    int n_vk_bufs;
    int gpu_pct;             /* share of each matvec's rows given to the GPU */
    int mm_gpu_pct;          /* the same for a block matmul (block verify) */
    int gpu_narrow;          /* the same for a narrow batch, 0 = the wide share */
    int gpu_min_rows;        /* below this the launch cost outweighs the win */
#endif
};

/* ---- stage profiling (BQ_PROFILE=1) -------------------------------- */
static double bq_prof_clock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double g_bq_t_mv, g_bq_t_fwd;
static double g_bq_t_mm;   /* time inside bq_matmat (block verify) */
static double g_bq_w_mv, g_bq_w_gpu;
static long long g_bq_n_mv, g_bq_n_fwd;
static int g_bq_prof = -1;

static int bq_prof_enabled(void) {
    if (g_bq_prof < 0) {
        const char *e = getenv("BQ_PROFILE");
        g_bq_prof = (e != NULL && e[0] == '1') ? 1 : 0;
    }
    return g_bq_prof;
}

void bonsai_profile_dump(void) {
    if (!bq_prof_enabled() || g_bq_n_fwd == 0) return;
    fprintf(stderr, "[prof] forward %.3f s over %lld evals; matvec %.3f s (%.1f%%), %.2f G weight/token on average\n",
            g_bq_t_fwd, g_bq_n_fwd, g_bq_t_mv, 100.0 * g_bq_t_mv / g_bq_t_fwd, g_bq_w_mv / (double)g_bq_n_fwd / 1e9);
    if (g_bq_n_mv > 0) {
        fprintf(stderr, "[prof] matvec: %lld calls, %.2f G weights/s, %.3f ms/call, %.1f%% of weights on the GPU\n",
                g_bq_n_mv, g_bq_w_mv / g_bq_t_mv / 1e9, 1e3 * g_bq_t_mv / (double)g_bq_n_mv,
                100.0 * g_bq_w_gpu / g_bq_w_mv);
    }
    fprintf(stderr, "[prof] other: prep %.3f s, gdn %.3f s, attn %.3f s, norm %.3f s, "
                    "elt %.3f s, conv %.3f s (over %lld evals)\n",
            g_bq_t_prep, g_bq_t_gdn, g_bq_t_attn, g_bq_t_norm, g_bq_t_ew, g_bq_t_conv, g_bq_n_fwd);
    fprintf(stderr, "[prof] block matmat %.3f s of forward %.3f s (%.1f%%), rest %.3f s\n",
            g_bq_t_mm, g_bq_t_fwd,
            g_bq_t_fwd > 0.0 ? 100.0 * g_bq_t_mm / g_bq_t_fwd : 0.0,
            g_bq_t_fwd - g_bq_t_mm);
}

/* ---- GPU/CPU agreement check (BQ_VERIFY=1) ------------------------- */
#if defined(BITNET_ENABLE_VULKAN)
static int g_bq_verify = -1;

static int bq_verify_enabled(void) {
    if (g_bq_verify < 0) {
        const char *e = getenv("BQ_VERIFY");
        g_bq_verify = (e != NULL && e[0] == '1') ? 1 : 0;
    }
    return g_bq_verify;
}

/* Recompute the rows the GPU produced with the plain float kernel and report
 * the worst relative difference.  Both engines then evaluate the same
 * expression over the same data, so only summation order differs and the
 * error should sit at rounding level; anything larger means the shader's
 * decode or activation layout is wrong. */
static void bq_verify_rows(const bq_w_t *w, const float *x, const float *y, int n) {
    static long long n_calls, n_bad;
    static double worst;
    int i;
    long long bad_here = 0;
    double worst_here = 0.0;
    for (i = 0; i < n; ++i) {
        const float ref = bq_dot_row(w, i, x);
        const double d = fabs((double)y[i] - (double)ref);
        const double denom = fabs((double)ref) > 1e-6 ? fabs((double)ref) : 1e-6;
        const double rel = d / denom;
        if (rel > worst_here) worst_here = rel;
        if (fabs((double)y[i] - (double)ref) > 1e-4 * (fabs((double)ref) > 1.0 ? fabs((double)ref) : 1.0))
            ++bad_here;
    }
    if (worst_here > worst) worst = worst_here;
    n_bad += bad_here;
    ++n_calls;
    /* A forward pass calls this several hundred times, so report the first
     * few, anything that mismatches, and then a heartbeat. */
    if (n_calls <= 20 || bad_here > 0 || n_calls % 100 == 0)
        fprintf(stderr, "[verify] %dx%d gpu_rows=%d worst_rel=%.3e worst_rel_all=%.3e mismatched=%lld/%lld\n",
                w->n_out, w->n_in, n, worst_here, worst, n_bad, n_calls * (long long)n);
}
#endif

/* One matvec's share of a batched group. */
typedef struct {
    const bq_w_t *w;
    const float *x;
    float *y;
    int grow;      /* GPU row count; the CPU takes the rest */
} bq_mv_seg_t;

/* Rows of `w` the GPU should take, or 0 to keep the whole matrix on the CPU. */
static int bq_mv_gpu_rows(const bonsai_model_t *m, const bq_w_t *w) {
#if defined(BITNET_ENABLE_VULKAN)
    int grow;
    if (m->vk == NULL || w->vk_buf == NULL || m->gpu_pct <= 0) return 0;
    if (w->n_out < m->gpu_min_rows || w->n_in % 128 != 0) return 0;
    if ((int64_t)w->row_stride * w->n_out > (int64_t)((size_t)128 << 20) * 64) return 0;
    grow = (int)((int64_t)w->n_out * m->gpu_pct / 100);
    if (grow < 1 || grow > w->n_out) return 0;
    return grow;
#else
    (void)m; (void)w;
    return 0;
#endif
}

/* The CPU rows of one matvec, from arg.row0 up. */
static void bq_mv_cpu(bonsai_model_t *m, const bq_w_t *w, const float *x, float *y, int row0) {
    bq_pool_t *pool = m->pool;
    bq_mv_arg_t arg;
    int chunk = w->n_out / (pool->nthreads * 64);
    void *xq = NULL;
    if (chunk < 1) chunk = 1;
    if (chunk > 64) chunk = 64;
    /* The activations are shared by every row, so quantize them once and run
     * the integer dot kernel over the whole matrix. */
    if (w->type == BQ_PTQ && w->n_in % BITNET_Q8_0_QK == 0) {
        xq = malloc(bitnet_q8_0_row_size(w->n_in));
        if (xq != NULL) bitnet_quantize_row_q8_0(x, xq, w->n_in);
    }
    arg.w = w;
    arg.x = x;
    arg.xq = xq;
    arg.y = y;
    arg.row0 = row0;
    arg.row1 = w->n_out;
    arg.next = row0;
    arg.chunk = chunk;
    if (row0 < w->n_out) bq_pool_run(pool, bq_mv_job, &arg);
    free(xq);
}

/* Runs a group of matvecs that share one activation and are independent of
 * each other.  Their GPU slices go into a single submission -- one round trip
 * costs about 0.6 ms on this driver whatever it carries -- and the CPU pool
 * works its rows of every matrix while that whole batch is in flight. */
/* BQ_VK_BATCH=0 restores one submission per matvec; it is the A/B reference
 * and the escape hatch if a driver dislikes a command buffer with several
 * dispatches. */
static int bq_batch_off(void) {
    static int off = -1;
    if (off < 0) {
        const char *e = getenv("BQ_VK_BATCH");
        off = (e != NULL && e[0] == '0');
    }
    return off;
}

/* Rows the GPU took in the last group.  The split depends on the batch width
 * and on the configured share, so a test that wants to check the device's rows
 * has to ask what the split actually did rather than assume it. */
int bq_last_gpu_rows;

static void bq_matvec_group(bonsai_model_t *m, bq_mv_seg_t *s, int nseg) {
    double t0 = bq_prof_enabled() ? bq_prof_clock() : 0.0;
    int i, total = 0, batched = 0;
#if !defined(BITNET_ENABLE_VULKAN)
    (void)total;
    (void)batched;
#endif
    if (nseg > 1 && bq_batch_off()) {
        for (i = 0; i < nseg; ++i) bq_matvec_group(m, &s[i], 1);
        return;
    }
    for (i = 0; i < nseg; ++i) { s[i].grow = bq_mv_gpu_rows(m, s[i].w); total += s[i].grow; }
    /* A narrow batch is where the GPU stops paying its way: one 3072-row
     * dispatch runs at about 17 G weights/s against 45 for a wide one, close
     * to what the CPU manages on the same rows, so the same share makes the
     * GPU the straggler and the caller waits on it (240-430 us per group).
     * Measured on an Adreno 740, 60% of the rows is the best split there and
     * it is its own number, not the wide share minus a margin: giving narrow
     * batches 62% costs 5% of the eval, and so does 54%. */
#if defined(BITNET_ENABLE_VULKAN)
    if (total > 0 && total < 8192) {
        int pct = m->gpu_narrow > 0 ? m->gpu_narrow : m->gpu_pct;
        /* Two narrow shapes want different splits: a 5120-wide output
         * projection leaves the GPU ~25 G weights/s against the CPU's 23, so
         * 52% balances where 60% leaves the GPU the straggler (180 us a
         * layer); the 17408-wide down projection leaves the GPU at 43 against
         * 28 and is balanced at 60%.  Narrow the share only for the short
         * rows.  Worth 2.5% of the eval over the single number. */
        if (s[0].w->n_in <= 8192 && pct > 8) pct -= 8;
        total = 0;
        for (i = 0; i < nseg; ++i) {
            if (s[i].grow <= 0) continue;
            s[i].grow = (int)((int64_t)s[i].w->n_out * pct / 100);
            total += s[i].grow;
        }
    }
    if (total > 0 && bq_vk_ptq_batch_begin(m->vk, s[0].x, s[0].w->n_in, total) == 0) {
        int ok = 1;
        for (i = 0; i < nseg && ok; ++i) {
            if (s[i].grow <= 0) continue;
            /* only segments over the packed activation can join */
            if (s[i].x != s[0].x || s[i].w->n_in != s[0].w->n_in) { s[i].grow = 0; continue; }
            if (bq_vk_ptq_batch_add(m->vk, (const bq_vk_buf_t *)s[i].w->vk_buf, s[i].w->row_stride,
                                    0, s[i].grow, s[i].y) != 0) ok = 0;
        }
        if (ok && bq_vk_ptq_batch_end(m->vk) == 0) batched = 1;
    }
    if (!batched) for (i = 0; i < nseg; ++i) s[i].grow = 0;
#endif
    {
        int g = 0;
        for (i = 0; i < nseg; ++i) g += s[i].grow;
        bq_last_gpu_rows = g;
    }
    {
        const int diag = getenv("BQ_VK_GROUPS") != NULL;
        double t_cpu = diag ? bq_prof_clock() : 0.0, t_wait = 0.0;
        for (i = 0; i < nseg; ++i) bq_mv_cpu(m, s[i].w, s[i].x, s[i].y, s[i].grow);
        if (diag) t_cpu = bq_prof_clock() - t_cpu;
#if defined(BITNET_ENABLE_VULKAN)
        if (batched) {
            if (diag) t_wait = bq_prof_clock();
            bq_vk_wait(m->vk);
            if (diag) t_wait = bq_prof_clock() - t_wait;
            if (bq_verify_enabled())
                for (i = 0; i < nseg; ++i)
                    if (s[i].grow > 0) bq_verify_rows(s[i].w, s[i].x, s[i].y, s[i].grow);
        }
#endif
        if (diag) {
            int tiny = 0, widest = 0;
            for (i = 0; i < nseg; ++i) {
                if (s[i].w->n_out < tiny || tiny == 0) tiny = s[i].w->n_out;
                if (s[i].w->n_out > widest) widest = s[i].w->n_out;
            }
            fprintf(stderr, "[grp] n=%d n_out=%d..%d rows=%d cpu=%.0fus wait=%.0fus\n",
                    nseg, tiny, widest, total, t_cpu * 1e6, t_wait * 1e6);
        }
    }
    if (t0 != 0.0) {
        g_bq_t_mv += bq_prof_clock() - t0;
        for (i = 0; i < nseg; ++i) {
            g_bq_w_mv += (double)s[i].w->n_out * (double)s[i].w->n_in;
            g_bq_w_gpu += (double)s[i].grow * (double)s[i].w->n_in;
            ++g_bq_n_mv;
        }
    }
}

/* Several independent matvecs over one activation go out as one submission;
 * a round trip costs far more than the dispatch it carries. */
#define BQ_MV(w_, x_, y_) { (w_), (x_), (y_), 0 }
#define BQ_MV_GROUP(m_, ...) do { \
    bq_mv_seg_t bq_segs_[] = { __VA_ARGS__ }; \
    bq_matvec_group((m_), bq_segs_, (int)(sizeof(bq_segs_) / sizeof(bq_segs_[0]))); \
} while (0)

static void bq_matvec(bonsai_model_t *m, const bq_w_t *w, const float *x, float *y) {
    bq_mv_seg_t s;
    s.w = w;
    s.x = x;
    s.y = y;
    s.grow = 0;
    bq_matvec_group(m, &s, 1);
}

static const gguf_tensor_t *bq_find_tensor(const gguf_file_t *g, const char *name) {
    uint64_t i;
    for (i = 0; i < g->tensor_count; ++i) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

static int bq_md_int(const gguf_file_t *g, const char *key, int def) {
    const gguf_metadata_t *e = gguf_find_metadata(g, key);
    if (e == NULL) return def;
    return (int)e->value.u64;
}

/* "<arch>.<suffix>": the same helper for a model whose family is decided at
 * runtime, so the loader has no per-family literals. */
static const gguf_metadata_t *bq_md_a(const bonsai_model_t *m, const char *suffix) {
    char key[96];
    snprintf(key, sizeof(key), "%s.%s", m->arch_prefix, suffix);
    return gguf_find_metadata(&m->gguf, key);
}
static int bq_md_int_a(const bonsai_model_t *m, const char *suffix, int def) {
    const gguf_metadata_t *e = bq_md_a(m, suffix);
    if (e == NULL) return def;
    return (int)e->value.u64;
}
static float bq_md_float_a(const bonsai_model_t *m, const char *suffix, float def) {
    const gguf_metadata_t *e = bq_md_a(m, suffix);
    if (e == NULL) return def;
    return (float)e->value.f64;
}


static const float *bq_signs(const bonsai_model_t *m, int width) {
    int i;
    for (i = 0; i < m->n_sign; ++i) {
        if (m->sign_width[i] == width) return m->sign_data[i];
    }
    return NULL;
}

/* bind a weight tensor; the input transform is implicit: every PTQ weight is
 * folded, the rest are plain floats read in the primal basis */
#if defined(BITNET_ENABLE_VULKAN)
static double g_bq_t_alloc, g_bq_t_read;
static size_t g_bq_b_alloc;
#endif

static int bq_p2_enabled(void);
static void *bq_own(bonsai_model_t *m, void *ptr);

/* Repacking is a load-time cost paid once per tensor, and it is pure decode
 * work: one thread per row pair stripe. */
#define BQ_PACK_THREADS 8
typedef struct {
    const bq_w_t *w;
    uint8_t *p2;
    float *p2d;
    size_t pb;
    int nb, tid;
} bq_pack_arg_t;

static void *bq_pack_worker(void *vp) {
    const bq_pack_arg_t *a = (const bq_pack_arg_t *)vp;
    int r;
    for (r = 2 * a->tid; r + 1 < a->w->n_out; r += 2 * BQ_PACK_THREADS)
        bitnet_ptq1_0_p2_pack(a->w->data + (size_t)r * a->w->row_stride,
                              a->w->data + (size_t)(r + 1) * a->w->row_stride,
                              a->w->n_in, a->p2 + (size_t)(r / 2) * a->pb,
                              a->p2d + (size_t)(r / 2) * (size_t)a->nb * 2u);
    return NULL;
}

static int bq_bind(bonsai_model_t *m, bq_w_t *w, const char *name) {
    const gguf_tensor_t *t = bq_find_tensor(&m->gguf, name);
    if (t == NULL) {
        fprintf(stderr, "bonsai: missing tensor %s\n", name);
        return -1;
    }
    w->data = (const uint8_t *)gguf_get_tensor_ptr(&m->gguf, t);
    w->type = t->type;
    w->n_in = (int)t->dims[0];
    w->n_out = t->n_dims > 1 ? (int)t->dims[1] : 1;
    w->row_stride = bq_row_stride(t->type, w->n_in);
    w->vk_buf = NULL;
    w->p2 = NULL;
    w->p2d = NULL;
    if (w->data == NULL) return -1;
    if (w->type != BQ_PTQ || w->n_in % 128 != 0) m->mmap_live = 1;
#if defined(BITNET_ENABLE_VULKAN)
    /* Copy the packed matrix into a buffer the GPU can bind.  The allocation
     * is host-visible, so the CPU keeps reading it directly: both engines work
     * out of one copy instead of two. */
    if (m->vk != NULL && w->type == BQ_PTQ && w->n_in % 128 == 0 && w->n_out > 0) {
        size_t bytes = (size_t)w->n_out * w->row_stride;
        bq_vk_buf_t *vb = NULL;
        double t0 = bq_prof_clock();
        void *dst = bq_vk_buf_alloc_cached(m->vk, bytes, &vb);
        g_bq_t_alloc += bq_prof_clock() - t0;
        if (dst != NULL) {
            bq_vk_buf_t **grown = (bq_vk_buf_t **)realloc(
                m->vk_bufs, (size_t)(m->n_vk_bufs + 1) * sizeof(*grown));
            {
                double t1 = bq_prof_clock();
                if (gguf_read_tensor(&m->gguf, t, dst, bytes) != 0) {
                    memcpy(dst, w->data, bytes);
                }
                bq_vk_buf_flush(m->vk, vb);
                g_bq_t_read += bq_prof_clock() - t1;
                g_bq_b_alloc += bytes;
            }
            if (grown != NULL) {
                m->vk_bufs = grown;
                m->vk_bufs[m->n_vk_bufs++] = vb;
            }
            w->data = (const uint8_t *)dst;
            w->vk_buf = vb;
        } else {
            m->mmap_live = 1;
        }
    } else {
        m->mmap_live = 1;
    }
#else
    (void)m;
#endif
    if (w->type == BQ_PTQ && w->n_in % 128 == 0 && w->n_out > 0 && (w->n_out & 1) == 0 &&
        bq_p2_enabled()) {
        const int nb = w->n_in / 128;
        const size_t pb = bitnet_ptq1_0_p2_pair_bytes(w->n_in);
        const int npair = w->n_out / 2;
        uint8_t *p2 = (uint8_t *)bq_own(m, malloc(pb * (size_t)npair));
        float *p2d = (float *)bq_own(m, malloc(sizeof(float) * (size_t)npair * (size_t)nb * 2u));
        if (p2 != NULL && p2d != NULL) {
#if defined(BITNET_ENABLE_VULKAN)
            double t0 = bq_prof_clock();
#endif
            bq_pack_arg_t args[BQ_PACK_THREADS];
            pthread_t th[BQ_PACK_THREADS];
            int made[BQ_PACK_THREADS];
            int k;
            for (k = 0; k < BQ_PACK_THREADS; ++k) {
                args[k].w = w;
                args[k].p2 = p2;
                args[k].p2d = p2d;
                args[k].pb = pb;
                args[k].nb = nb;
                args[k].tid = k;
                made[k] = pthread_create(&th[k], NULL, bq_pack_worker, &args[k]) == 0;
            }
            for (k = 0; k < BQ_PACK_THREADS; ++k)
                if (made[k]) pthread_join(th[k], NULL);
#if defined(BITNET_ENABLE_VULKAN)
            g_bq_t_alloc += bq_prof_clock() - t0;
#endif
            w->p2 = p2;
            w->p2d = p2d;
        }
    }
    return 0;
}

/* Takes ownership of `ptr`, freeing it with the model. */
static void *bq_own(bonsai_model_t *m, void *ptr) {
    void **grown = (void **)realloc(m->owned, (size_t)(m->n_owned + 1) * sizeof(*grown));
    if (grown == NULL) return ptr;
    m->owned = grown;
    m->owned[m->n_owned++] = ptr;
    return ptr;
}

/* Small float tensors (norms, gates, conv taps) are copied out of the mapping
 * so the model file can be unmapped once loading finishes. */
static float *bq_bind_f32(bonsai_model_t *m, const char *name, int n_expected) {
    const gguf_tensor_t *t = bq_find_tensor(&m->gguf, name);
    float *src, *dst;
    if (t == NULL || t->type != BQ_F32) {
        fprintf(stderr, "bonsai: missing f32 tensor %s\n", name);
        return NULL;
    }
    if (n_expected > 0 && (int)t->dims[0] != n_expected) {
        fprintf(stderr, "bonsai: tensor %s has width %d, expected %d\n", name, (int)t->dims[0], n_expected);
        return NULL;
    }
    src = (float *)gguf_get_tensor_ptr(&m->gguf, t);
    if (src == NULL) return NULL;
    dst = (float *)malloc((size_t)t->dims[0] * sizeof(float));
    if (dst == NULL) return NULL;
    memcpy(dst, src, (size_t)t->dims[0] * sizeof(float));
    return (float *)bq_own(m, dst);
}

static int bq_load_layer(bonsai_model_t *m, int il) {
    bq_layer_t *L = &m->layers[il];
    char name[128];

    L->is_full = (il % m->full_interval) == (m->full_interval - 1);

    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);
    L->attn_norm = bq_bind_f32(m, name, m->n_embd);
    snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", il);
    L->post_attn_norm = bq_bind_f32(m, name, m->n_embd);
    if (!L->attn_norm || !L->post_attn_norm) return -1;

    snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);
    if (bq_bind(m, &L->ffn_up, name) != 0) return -1;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);
    if (bq_bind(m, &L->ffn_gate, name) != 0) return -1;
    snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);
    if (bq_bind(m, &L->ffn_down, name) != 0) return -1;

    if (L->is_full) {
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);
        if (bq_bind(m, &L->wq, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);
        if (bq_bind(m, &L->wk, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);
        if (bq_bind(m, &L->wv, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il);
        if (bq_bind(m, &L->wo, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", il);
        L->q_norm = bq_bind_f32(m, name, m->head_dim);
        snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", il);
        L->k_norm = bq_bind_f32(m, name, m->head_dim);
        if (!L->q_norm || !L->k_norm) return -1;
    } else {
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", il);
        if (bq_bind(m, &L->wqkv, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", il);
        if (bq_bind(m, &L->wqkv_gate, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", il);
        if (bq_bind(m, &L->wssm_out, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", il);
        if (bq_bind(m, &L->walpha, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", il);
        if (bq_bind(m, &L->wbeta, name) != 0) return -1;
        snprintf(name, sizeof(name), "blk.%d.ssm_a", il);
        L->ssm_a = bq_bind_f32(m, name, m->ssm_dt_rank);
        snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", il);
        L->ssm_dt = bq_bind_f32(m, name, m->ssm_dt_rank);
        snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", il);
        L->ssm_norm = bq_bind_f32(m, name, m->ssm_state);
        snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", il);
        {
            const gguf_tensor_t *t = bq_find_tensor(&m->gguf, name);
            if (t == NULL || t->type != BQ_F32 || t->n_dims != 2 ||
                (int)t->dims[0] != m->ssm_conv_k) {
                fprintf(stderr, "bonsai: bad tensor %s\n", name);
                return -1;
            }
            {
                const float *src = (const float *)gguf_get_tensor_ptr(&m->gguf, t);
                size_t n = (size_t)t->dims[0] * (size_t)t->dims[1];
                float *dst = (float *)malloc(n * sizeof(float));
                if (dst == NULL) return -1;
                memcpy(dst, src, n * sizeof(float));
                L->ssm_conv = (float *)bq_own(m, dst);
            }
        }
        if (!L->ssm_a || !L->ssm_dt || !L->ssm_norm || !L->ssm_conv) return -1;
    }
    return 0;
}

void bonsai_free_model(bonsai_model_t *m) {
    int i;
    if (m == NULL) return;
    if (m->mc != NULL) {
        minicpm_free_model((minicpm_model_t *)m->mc);
        free(m);
        return;
    }
    if (m->pool != NULL) {
        bq_pool_free(m->pool);
        free(m->pool);
    }
    bq_tok_free(&m->tok);
    for (i = 0; i < m->n_sign; ++i) free(m->sign_data[i]);
    free(m->rope_freq);
    free(m->layers);
    for (i = 0; i < m->n_owned; ++i) free(m->owned[i]);
    free(m->owned);
#if defined(BITNET_ENABLE_VULKAN)
    if (m->vk != NULL) {
        for (i = 0; i < m->n_vk_bufs; ++i) bq_vk_buf_free(m->vk, m->vk_bufs[i]);
        free(m->vk_bufs);
        bq_vk_destroy(m->vk);
    }
#endif
    gguf_close(&m->gguf);
    free(m);
}

static int bq_parse_signs(bonsai_model_t *m) {
    const gguf_metadata_t *w = gguf_find_metadata(&m->gguf, "prism.hadamard.sign_widths");
    const gguf_metadata_t *v = gguf_find_metadata(&m->gguf, "prism.hadamard.sign_values");
    size_t off = 0;
    int i, j;

    if (w == NULL || v == NULL) return -1;
    if (w->array_count > 8) return -1;
    m->n_sign = (int)w->array_count;

    for (i = 0; i < m->n_sign; ++i) {
        int width = w->array_ints ? w->array_ints[i] : 0;
        float *dst;
        if (width <= 0 || width > (int)v->array_count || off + (size_t)width > v->array_count) return -1;
        dst = (float *)malloc((size_t)width * sizeof(float));
        if (dst == NULL) return -1;
        for (j = 0; j < width; ++j) dst[j] = (float)v->array_ints[off + (size_t)j];
        m->sign_width[i] = width;
        m->sign_data[i] = dst;
        off += (size_t)width;
    }
    return off == (size_t)v->array_count ? 0 : -1;
}

bonsai_arch_t bonsai_arch_from_name(const char *name) {
    if (name == NULL) return BONSAI_ARCH_UNKNOWN;
    if (strcmp(name, "qwen35") == 0) return BONSAI_ARCH_QWEN35;
    if (strcmp(name, "minicpm") == 0) return BONSAI_ARCH_MINICPM;
    if (strcmp(name, "llama") == 0) return BONSAI_ARCH_LLAMA;
    return BONSAI_ARCH_UNKNOWN;
}

const char *bonsai_arch_name(bonsai_arch_t arch) {
    switch (arch) {
    case BONSAI_ARCH_QWEN35:  return "qwen35";
    case BONSAI_ARCH_MINICPM: return "minicpm";
    case BONSAI_ARCH_LLAMA:   return "llama";
    default:                  return "unknown";
    }
}

/* Families with a graph in this build. */
const char *bonsai_arch_supported(void) { return "qwen35, minicpm, llama"; }

bonsai_arch_t bonsai_model_arch(const bonsai_model_t *model) {
    return model != NULL ? model->arch : BONSAI_ARCH_UNKNOWN;
}

bonsai_model_t *bonsai_load_model(const char *path, int n_threads) {
    bonsai_model_t *m;
    const gguf_metadata_t *e;
    int i;

    m = (bonsai_model_t *)calloc(1, sizeof(*m));
    if (m == NULL) return NULL;

    if (gguf_open(path, &m->gguf) != 0) {
        fprintf(stderr, "bonsai: cannot open %s\n", path);
        free(m);
        return NULL;
    }

    /* Read the family from the file.  This is the single place the runtime
     * decides what it is looking at: everything below reads "<arch>.<key>"
     * metadata and, once the graph exists for that family, runs it. */
    e = gguf_find_metadata(&m->gguf, "general.architecture");
    m->arch = bonsai_arch_from_name(e != NULL ? e->string_value : NULL);
    if (m->arch == BONSAI_ARCH_UNKNOWN) {
        fprintf(stderr, "bonsai: %s: unsupported architecture '%s' (this build knows %s)\n",
                path, e != NULL && e->string_value != NULL ? e->string_value : "?",
                bonsai_arch_supported());
        bonsai_free_model(m);
        return NULL;
    }
    snprintf(m->arch_prefix, sizeof(m->arch_prefix), "%s", e->string_value);
    if (m->arch == BONSAI_ARCH_MINICPM || m->arch == BONSAI_ARCH_LLAMA) {
        /* Same binary, other family.  The minicpm loader opens the file for
         * itself, so this parse is dropped rather than held twice. */
        gguf_close(&m->gguf);
        memset(&m->gguf, 0, sizeof(m->gguf));
        m->gguf.fd = -1;
        m->mc = minicpm_load_model(path, n_threads);
        if (m->mc == NULL) {
            free(m);
            return NULL;
        }
        /* Context clamping is generic, so give the outer struct the length the
         * minicpm file declares; without it a llama/minicpm model would accept
         * any max_tokens and allocate KV for positions it was never trained
         * for. */
        m->n_ctx_train = minicpm_context_limit((const minicpm_model_t *)m->mc);
        return m;
    }

#if defined(BITNET_ENABLE_VULKAN)
    {
        /* Bring the GPU up before the weights are bound, so every ternary
         * matrix is staged into memory both engines can reach. */
        const char *off = getenv("BQ_GPU");
        if (off == NULL || off[0] != '0') {
            const char *pct = getenv("BQ_GPU_PCT");
            const char *minr = getenv("BQ_GPU_MIN_ROWS");
            /* How much of each matvec's rows the GPU takes.  69% measured
             * best on an Adreno 740 phone after the CPU side was vectorised
             * (0.364 s/eval at 69 over three reps against 0.370 at 68, 0.374
             * at 66, 0.382 at 70 and 0.408 at 60); the curve falls away on
             * both sides.  The optimum follows the two engines' throughputs,
             * so it moves whenever either kernel changes. */
            m->gpu_pct = pct != NULL ? atoi(pct) : 69;
            {
                const char *nar = getenv("BQ_GPU_NARROW");
                /* 60% measured best for a batch under 8192 total rows, where
                 * the dispatch is too narrow to fill the GPU. */
                m->gpu_narrow = nar != NULL ? atoi(nar) : 60;
                if (m->gpu_narrow < 0) m->gpu_narrow = 0;
                if (m->gpu_narrow > 100) m->gpu_narrow = 100;
            }
            m->gpu_min_rows = minr != NULL ? atoi(minr) : 512;
            if (m->gpu_pct < 0) m->gpu_pct = 0;
            {
                /* Rows of each block matmul the GPU takes.  The two engines
                 * meet where the CPU's share (n_threads cores) and the GPU's
                 * share cost the same wall time; with the rollback cache in
                 * place the CPU has a little less work, which moves the
                 * meeting point from 40% to 45% (35% and 48% are both worse). */
                const char *mp = getenv("BQ_MM_GPU");
                m->mm_gpu_pct = mp != NULL ? atoi(mp) : 45;
                if (m->mm_gpu_pct < 0) m->mm_gpu_pct = 0;
                if (m->mm_gpu_pct > 100) m->mm_gpu_pct = 100;
            }
            if (m->gpu_pct > 100) m->gpu_pct = 100;
            m->vk = bq_vk_create();
            if (m->vk != NULL && !bq_vk_ready(m->vk)) {
                bq_vk_destroy(m->vk);
                m->vk = NULL;
            }
            if (getenv("BQ_VK_DEBUG") != NULL) {
                fprintf(stderr, "bonsai: vulkan %s, gpu_pct=%d, min_rows=%d\n",
                        m->vk != NULL ? "ready" : "unavailable", m->gpu_pct, m->gpu_min_rows);
            }
        }
    }
#endif

    m->n_layer = bq_md_int_a(m, "block_count", 0);
    m->n_embd = bq_md_int_a(m, "embedding_length", 0);
    m->n_ff = bq_md_int_a(m, "feed_forward_length", 0);
    m->n_head = bq_md_int_a(m, "attention.head_count", 0);
    m->n_head_kv = bq_md_int_a(m, "attention.head_count_kv", 0);
    m->head_dim = bq_md_int_a(m, "attention.key_length", 0);
    m->n_rot = bq_md_int_a(m, "rope.dimension_count", 0);
    m->n_ctx_train = bq_md_int_a(m, "context_length", 0);
    m->full_interval = bq_md_int_a(m, "full_attention_interval", 4);
    m->ssm_conv_k = bq_md_int_a(m, "ssm.conv_kernel", 4);
    m->ssm_state = bq_md_int_a(m, "ssm.state_size", 128);
    m->ssm_groups = bq_md_int_a(m, "ssm.group_count", 16);
    m->ssm_dt_rank = bq_md_int_a(m, "ssm.time_step_rank", 48);
    m->ssm_inner = bq_md_int_a(m, "ssm.inner_size", 6144);
    m->eps = bq_md_float_a(m, "attention.layer_norm_rms_epsilon", 1e-6f);
    m->rope_base = bq_md_float_a(m, "rope.freq_base", 10000.0f);
    m->block_size = bq_md_int(&m->gguf, "prism.hadamard.block_size", 0);
    m->gdn_v_grouped = bq_md_int(&m->gguf, "prism.hadamard.gdn_v_grouped", 0);

    e = bq_md_a(m, "rope.dimension_sections");
    if (e != NULL && e->array_ints != NULL && e->array_count >= 4) {
        for (i = 0; i < 4; ++i) m->rope_sections[i] = e->array_ints[i];
    } else {
        m->rope_sections[0] = m->n_rot / 2;
    }

    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_head <= 0 || m->head_dim <= 0 ||
        m->block_size <= 0 || m->n_rot <= 0 || m->n_rot > m->head_dim) {
        fprintf(stderr, "bonsai: bad model hyperparameters\n");
        bonsai_free_model(m);
        return NULL;
    }

    if (bq_parse_signs(m) != 0) {
        fprintf(stderr, "bonsai: bad prism.hadamard metadata\n");
        bonsai_free_model(m);
        return NULL;
    }

    m->rope_freq = (float *)malloc((size_t)(m->n_rot / 2) * sizeof(float));
    if (m->rope_freq == NULL) {
        bonsai_free_model(m);
        return NULL;
    }
    for (i = 0; i < m->n_rot / 2; ++i) {
        m->rope_freq[i] = powf(m->rope_base, -2.0f * (float)i / (float)m->n_rot);
    }

    if (bq_bind(m, &m->tok_embd, "token_embd.weight") != 0) {
        bonsai_free_model(m);
        return NULL;
    }
    m->out_norm = bq_bind_f32(m, "output_norm.weight", m->n_embd);
    if (m->out_norm == NULL || bq_bind(m, &m->output, "output.weight") != 0) {
        bonsai_free_model(m);
        return NULL;
    }
    m->n_vocab = m->tok_embd.n_out;

    m->layers = (bq_layer_t *)calloc((size_t)m->n_layer, sizeof(bq_layer_t));
    if (m->layers == NULL) {
        bonsai_free_model(m);
        return NULL;
    }
    m->n_gdn = 0;
    m->n_full = 0;
    for (i = 0; i < m->n_layer; ++i) {
        if (bq_load_layer(m, i) != 0) {
            bonsai_free_model(m);
            return NULL;
        }
        if (m->layers[i].is_full) ++m->n_full;
        else ++m->n_gdn;
    }

    if (bq_tok_load(&m->tok, &m->gguf) != 0) {
        fprintf(stderr, "bonsai: bad tokenizer data\n");
        bonsai_free_model(m);
        return NULL;
    }

    m->pool = (bq_pool_t *)malloc(sizeof(bq_pool_t));
    if (m->pool == NULL || bq_pool_init(m->pool, n_threads) != 0) {
        bonsai_free_model(m);
        return NULL;
    }

    /* Every tensor the runtime keeps now lives in its own allocation, so the
     * file mapping serves no purpose and its page cache would otherwise fight
     * the GPU's copy for memory. */
    if (!m->mmap_live) gguf_release_data(&m->gguf);
#if defined(BITNET_ENABLE_VULKAN)
    if (getenv("BQ_VK_DEBUG") != NULL) {
        fprintf(stderr, "bonsai: staged %.2f GiB for the GPU: alloc %.1f s, read %.1f s (%.0f MB/s)\n",
                (double)g_bq_b_alloc / 1073741824.0, g_bq_t_alloc, g_bq_t_read,
                g_bq_t_read > 0 ? (double)g_bq_b_alloc / g_bq_t_read / 1e6 : 0.0);
    }
#endif
    return m;
}

/* ------------------------------------------------------------------ */
/* context                                                            */
/* ------------------------------------------------------------------ */

typedef struct bq_blk bq_blk_t;
static void bq_blk_free(bq_blk_t *b);
static bq_blk_t *bq_blk_new(bonsai_model_t *m, int B, int capacity);

struct bonsai_context {
    bonsai_model_t *model;
    void *alt;      /* minicpm_context_t when the model's family is minicpm */
    bq_blk_t *blk;
    int capacity;
    int pos;

    float *x;
    float *xn;
    float *resid;
    float *prep;
    float *prep2;
    float *qg;
    float *qkv;
    float *conv_out;
    float *gdn_out;
    float *attn_out;
    float *z;
    float *ffn_gate;
    float *ffn_up;
    float *alpha;
    float *beta;
    float *logits;
    float *scores;

    float *ssm;
    float *convh;
    float *kvc_k;
    float *kvc_v;
    /* Rollback scratch: the gated delta net's own inputs (qkv, alpha, beta)
     * for every position of the last evaluated block.  Rebuilding the state
     * after a rewind then costs only the elementwise recurrence, not another
     * pass over those weights.  rp_npos is 0 when the cache does not cover
     * the whole of the last block. */
    float *rp;
    int rp_npos;
    int rp_row;
    int rp_layers;

#if defined(BITNET_ENABLE_VULKAN)
    /* Set when attention runs on the GPU: the cache then lives in memory the
     * shaders can read, and the CPU keeps writing it through the same map. */
    int gpu_stages;
    float *gpu_q;
    bq_vk_buf_t *kvc_k_buf;
    bq_vk_buf_t *kvc_v_buf;
    bq_vk_buf_t *gpu_q_buf;
#endif
};

void bonsai_free_context(bonsai_context_t *c) {
    if (c == NULL) return;
    if (c->alt != NULL) {
        minicpm_free_context((minicpm_context_t *)c->alt);
        free(c);
        return;
    }
    bq_blk_free(c->blk);
    free(c->x);
    free(c->xn);
    free(c->resid);
    free(c->prep);
    free(c->prep2);
    free(c->qg);
    free(c->qkv);
    free(c->conv_out);
    free(c->gdn_out);
    free(c->attn_out);
    free(c->z);
    free(c->ffn_gate);
    free(c->ffn_up);
    free(c->alpha);
    free(c->beta);
    free(c->logits);
    free(c->scores);
    free(c->ssm);
    free(c->convh);
    free(c->rp);
#if defined(BITNET_ENABLE_VULKAN)
    if (c->gpu_stages && c->model != NULL && c->model->vk != NULL) {
        bq_vk_buf_free(c->model->vk, c->kvc_k_buf);
        bq_vk_buf_free(c->model->vk, c->kvc_v_buf);
        bq_vk_buf_free(c->model->vk, c->gpu_q_buf);
    } else
#endif
    {
        free(c->kvc_k);
        free(c->kvc_v);
    }
    free(c);
}

/* The declared context_length is the range the model was trained for; going
 * past it is only advisory because the Minicpm long variants ship LongRoPE
 * factors for it, so the caller can opt in (BQ_CTX_EXTEND=1) instead of the
 * limit being silently ignored. */
static int bq_ctx_extend(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("BQ_CTX_EXTEND");
        on = (e != NULL && e[0] == '1') ? 1 : 0;
    }
    return on;
}

bonsai_context_t *bonsai_create_context(bonsai_model_t *model, int max_tokens) {
    bonsai_context_t *c;
    int i;
    size_t n_embd, n_ff, n_qkv, n_inner;

    if (model == NULL) return NULL;
    if (max_tokens <= 0) max_tokens = 2048;
    if (model->n_ctx_train > 0 && max_tokens > model->n_ctx_train && !bq_ctx_extend())
        max_tokens = model->n_ctx_train;

    if (model->mc != NULL) {
        minicpm_context_t *mc = minicpm_create_context((minicpm_model_t *)model->mc, max_tokens);
        if (mc == NULL) return NULL;
        c = (bonsai_context_t *)calloc(1, sizeof(*c));
        if (c == NULL) {
            minicpm_free_context(mc);
            return NULL;
        }
        c->model = model;
        c->alt = mc;
        c->capacity = max_tokens;
        return c;
    }

    c = (bonsai_context_t *)calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    c->model = model;
    c->capacity = max_tokens;

    n_embd = (size_t)model->n_embd;
    n_ff = (size_t)model->n_ff;
    n_qkv = (size_t)(2 * model->ssm_groups * model->ssm_state + model->ssm_inner);
    n_inner = (size_t)model->ssm_inner;

    c->x = (float *)calloc(n_embd, sizeof(float));
    c->xn = (float *)calloc(n_embd, sizeof(float));
    c->resid = (float *)calloc(n_embd, sizeof(float));
    c->prep = (float *)calloc(n_ff, sizeof(float));
    c->prep2 = (float *)calloc(n_inner > n_embd ? n_inner : n_embd, sizeof(float));
    c->qg = (float *)calloc((size_t)(2 * model->head_dim * model->n_head), sizeof(float));
    c->qkv = (float *)calloc(n_qkv, sizeof(float));
    c->conv_out = (float *)calloc(n_qkv, sizeof(float));
    c->gdn_out = (float *)calloc(n_inner, sizeof(float));
    c->attn_out = (float *)calloc(n_inner, sizeof(float));
    c->z = (float *)calloc(n_inner, sizeof(float));
    c->ffn_gate = (float *)calloc(n_ff, sizeof(float));
    c->ffn_up = (float *)calloc(n_ff, sizeof(float));
    c->alpha = (float *)calloc((size_t)model->ssm_dt_rank, sizeof(float));
    c->beta = (float *)calloc((size_t)model->ssm_dt_rank, sizeof(float));
    c->logits = (float *)calloc((size_t)model->n_vocab, sizeof(float));
    c->scores = (float *)calloc((size_t)max_tokens, sizeof(float));

    c->ssm = (float *)calloc((size_t)model->n_gdn * model->ssm_dt_rank *
                             model->ssm_state * model->ssm_state, sizeof(float));
    c->convh = (float *)calloc((size_t)model->n_gdn * model->ssm_conv_k * n_qkv, sizeof(float));
#if defined(BITNET_ENABLE_VULKAN)
    {
        const char *st = getenv("BQ_GPU_STAGES");
        c->gpu_stages = (model->vk != NULL && st != NULL && atoi(st) != 0);
    }
    if (c->gpu_stages) {
        const size_t kv_bytes = (size_t)model->n_full * (size_t)max_tokens *
                                model->n_head_kv * model->head_dim * sizeof(float);
        c->kvc_k = (float *)bq_vk_buf_alloc(model->vk, kv_bytes, &c->kvc_k_buf);
        c->kvc_v = (float *)bq_vk_buf_alloc(model->vk, kv_bytes, &c->kvc_v_buf);
        c->gpu_q = (float *)bq_vk_buf_alloc(model->vk,
                                            (size_t)model->n_head * model->head_dim * sizeof(float),
                                            &c->gpu_q_buf);
        if (c->kvc_k == NULL || c->kvc_v == NULL || c->gpu_q == NULL) {
            /* not enough GPU-visible memory: stay on the CPU path */
            if (c->kvc_k != NULL) { bq_vk_buf_free(model->vk, c->kvc_k_buf); c->kvc_k = NULL; }
            if (c->kvc_v != NULL) { bq_vk_buf_free(model->vk, c->kvc_v_buf); c->kvc_v = NULL; }
            if (c->gpu_q != NULL) { bq_vk_buf_free(model->vk, c->gpu_q_buf); c->gpu_q = NULL; }
            c->gpu_stages = 0;
        }
    }
#endif
    if (c->kvc_k == NULL) {
        c->kvc_k = (float *)calloc((size_t)model->n_full * (size_t)max_tokens *
                                   model->n_head_kv * model->head_dim, sizeof(float));
    }
    if (c->kvc_v == NULL) {
        c->kvc_v = (float *)calloc((size_t)model->n_full * (size_t)max_tokens *
                                   model->n_head_kv * model->head_dim, sizeof(float));
    }

    if (!c->x || !c->xn || !c->resid || !c->prep || !c->prep2 || !c->qg || !c->qkv ||
        !c->conv_out || !c->gdn_out || !c->attn_out || !c->z || !c->ffn_gate || !c->ffn_up ||
        !c->alpha || !c->beta || !c->logits || !c->scores || !c->ssm || !c->convh || !c->kvc_k || !c->kvc_v) {
        bonsai_free_context(c);
        return NULL;
    }
    for (i = 0; i < model->n_vocab; ++i) c->logits[i] = -INFINITY;
    return c;
}

static void bq_reset_context(bonsai_context_t *c) {
    bonsai_model_t *m = c->model;
    size_t n_qkv = (size_t)(2 * m->ssm_groups * m->ssm_state + m->ssm_inner);
    c->pos = 0;
    memset(c->ssm, 0, (size_t)m->n_gdn * m->ssm_dt_rank * m->ssm_state * m->ssm_state * sizeof(float));
    memset(c->convh, 0, (size_t)m->n_gdn * m->ssm_conv_k * n_qkv * sizeof(float));
    memset(c->kvc_k, 0, (size_t)m->n_full * (size_t)c->capacity * m->n_head_kv * m->head_dim * sizeof(float));
    memset(c->kvc_v, 0, (size_t)m->n_full * (size_t)c->capacity * m->n_head_kv * m->head_dim * sizeof(float));
}

/* ------------------------------------------------------------------ */
/* forward pass                                                       */
/* ------------------------------------------------------------------ */

static void bq_rope(float *v, int n_heads, int head_dim, int n_rot, int pos, const float *freq) {
    int h, i;
    int half = n_rot / 2;
    for (h = 0; h < n_heads; ++h) {
        float *p = v + (size_t)h * head_dim;
        for (i = 0; i < half; ++i) {
            float th = (float)pos * freq[i];
            float cs = cosf(th);
            float sn = sinf(th);
            float x0 = p[i];
            float x1 = p[i + half];
            p[i] = x0 * cs - x1 * sn;
            p[i + half] = x0 * sn + x1 * cs;
        }
    }
}

/* one gated delta net head: M[j][i] holds the transposed state S[i][j] */
#if defined(__aarch64__) && defined(__ARM_NEON)
/* The three inner loops of the recurrence are plain dependent float sums, which
 * a compiler will not vectorise without reassociation (and would then be too
 * inaccurate to keep).  Written out, they are worth about a quarter of the
 * stage's time: the state is S*S floats per head, so this loop walks tens of
 * megabytes per token and is the largest piece of serial CPU work left in a
 * hybrid decode. */
static inline float bq_dot_f32(const float *a, const float *b, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    int i = 0;
    float s;
    for (; i + 8 <= n; i += 8) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    if (i + 4 <= n) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
        i += 4;
    }
    s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

static void bq_gdn_head(float *M, const float *q, const float *k, const float *v,
                        float decay, float beta, float *out, int S) {
    const float scale = 1.0f / sqrtf((float)S);
    int i, j;
#if defined(__aarch64__) && defined(__ARM_NEON)
    {
        const float32x4_t dv = vdupq_n_f32(decay);
        const int n4 = (S * S) & ~3;
        for (i = 0; i < n4; i += 4)
            vst1q_f32(M + i, vmulq_f32(vld1q_f32(M + i), dv));
        for (; i < S * S; ++i) M[i] *= decay;
    }
    for (j = 0; j < S; ++j) {
        float *row = M + (size_t)j * S;
        const float d = (v[j] - bq_dot_f32(row, k, S)) * beta;
        const float32x4_t dv = vdupq_n_f32(d);
        const int s4 = S & ~3;
        for (i = 0; i < s4; i += 4)
            vst1q_f32(row + i, vfmaq_f32(vld1q_f32(row + i), dv, vld1q_f32(k + i)));
        for (; i < S; ++i) row[i] += d * k[i];
        out[j] = bq_dot_f32(row, q, S) * scale;
    }
#else
    for (i = 0; i < S * S; ++i) M[i] *= decay;
    for (j = 0; j < S; ++j) {
        float *row = M + (size_t)j * S;
        float s = 0.0f;
        float d;
        for (i = 0; i < S; ++i) s += row[i] * k[i];
        d = (v[j] - s) * beta;
        for (i = 0; i < S; ++i) row[i] += d * k[i];
        s = 0.0f;
        for (i = 0; i < S; ++i) s += row[i] * q[i];
        out[j] = s * scale;
    }
#endif
}

static void bq_attention(const bonsai_context_t *c, int full_idx, const float *q,
                         const float *K, const float *V, int pos, float *out) {
    const bonsai_model_t *m = c->model;
    const int hd = m->head_dim;
    const int nkv = m->n_head_kv;
    const int rep = m->n_head / nkv;
    const int nt = pos + 1;
    const float scale = 1.0f / sqrtf((float)hd);
    int h, t, d;
    float *scores = c->scores;
    (void)full_idx;

    for (h = 0; h < m->n_head; ++h) {
        const float *qh = q + (size_t)h * hd;
        const int kh = h / rep;
        float maxv = -INFINITY;
        float sum = 0.0f;
        float *o = out + (size_t)h * hd;

        for (t = 0; t < nt; ++t) {
            const float *kt = K + ((size_t)t * nkv + kh) * hd;
            float s = 0.0f;
            for (d = 0; d < hd; ++d) s += qh[d] * kt[d];
            s *= scale;
            scores[t] = s;
            if (s > maxv) maxv = s;
        }
        for (t = 0; t < nt; ++t) {
            float e = expf(scores[t] - maxv);
            scores[t] = e;
            sum += e;
        }
        for (d = 0; d < hd; ++d) o[d] = 0.0f;
        for (t = 0; t < nt; ++t) {
            const float p = scores[t] / sum;
            const float *vt = V + ((size_t)t * nkv + kh) * hd;
            for (d = 0; d < hd; ++d) o[d] += p * vt[d];
        }
    }
}

/* The GDN heads of one layer are independent, so the recurrence is spread
 * over the pool; it used to run on one core and was the largest non-matvec
 * cost in a decode step. */
typedef struct {
    float *ssm;          /* this layer's H_v states, S*S each */
    const float *q, *k, *v;
    const float *decay, *beta;      /* precomputed when H_v <= 128 */
    const float *a_log, *dt, *alpha, *beta_raw;
    int inline_dbe;                 /* H_v > 128: no room for a table */
    float *out;
    int n_heads, groups, S, next;
} bq_gdn_arg_t;

static void bq_gdn_job(void *p, int tid, int nthreads) {
    bq_gdn_arg_t *a = (bq_gdn_arg_t *)p;
    (void)tid;
    (void)nthreads;
    for (;;) {
        int h = __atomic_fetch_add(&a->next, 1, __ATOMIC_RELAXED);
        int kh;
        float dv, bv;
        if (h >= a->n_heads) break;
        kh = h % a->groups;
        if (a->inline_dbe) {
            dv = expf(a->a_log[h] * bq_softplus(a->alpha[h] + a->dt[h]));
            bv = bq_sigmoid(a->beta_raw[h]);
        } else {
            dv = a->decay[h];
            bv = a->beta[h];
        }
        bq_gdn_head(a->ssm + (size_t)h * a->S * a->S,
                    a->q + (size_t)kh * a->S, a->k + (size_t)kh * a->S,
                    a->v + (size_t)h * a->S, dv, bv,
                    a->out + (size_t)h * a->S, a->S);
    }
}

/* Device-side stage helpers.  Each returns 1 when the stage ran on the GPU
 * and 0 when the caller must use its CPU path, so a stage that is disabled or
 * unavailable never changes what the code does.  BQ_GPU_STAGES is a bitmask
 * (1 gated delta net, 2 convolution, 4 elementwise and norm) because a stage
 * that wins on its own loses once the dispatches contend with the matvec. */
#define BQ_STAGE_GDN  1
#define BQ_STAGE_CONV 2
#define BQ_STAGE_ELT  4
#define BQ_STAGE_SAMPLE 8
#if defined(BITNET_ENABLE_VULKAN)
static int bq_gpu_elt(bonsai_context_t *c, int mode, const float *a, const float *b,
                      const float *w, int n, int nrow, float *out) {
    if ((c->gpu_stages & BQ_STAGE_ELT) == 0) return 0;
    return bq_vk_elt(c->model->vk, mode, a, b, w, n, nrow, c->model->eps, out) == 0;
}

static int bq_gpu_gdn(bonsai_context_t *c, int slot, const float *q, const float *k,
                      const float *v, const float *decay, const float *beta,
                      int nhead, int nkv, int S, float *out) {
    if ((c->gpu_stages & BQ_STAGE_GDN) == 0) return 0;
    return bq_vk_gdn(c->model->vk, slot, q, k, v, decay, beta, nhead, nkv, S, out) == 0;
}

static int bq_gpu_conv(bonsai_context_t *c, const float *w, const float *hist,
                       int K, int nch, float *out) {
    if ((c->gpu_stages & BQ_STAGE_CONV) == 0) return 0;
    return bq_vk_conv(c->model->vk, w, hist, K, nch, out) == 0;
}
#else
static int bq_gpu_elt(bonsai_context_t *c, int mode, const float *a, const float *b,
                      const float *w, int n, int nrow, float *out) {
    (void)c; (void)mode; (void)a; (void)b; (void)w; (void)n; (void)nrow; (void)out;
    return 0;
}
static int bq_gpu_gdn(bonsai_context_t *c, int slot, const float *q, const float *k,
                      const float *v, const float *decay, const float *beta,
                      int nhead, int nkv, int S, float *out) {
    (void)c; (void)slot; (void)q; (void)k; (void)v; (void)decay; (void)beta;
    (void)nhead; (void)nkv; (void)S; (void)out;
    return 0;
}
static int bq_gpu_conv(bonsai_context_t *c, const float *w, const float *hist,
                       int K, int nch, float *out) {
    (void)c; (void)w; (void)hist; (void)K; (void)nch; (void)out;
    return 0;
}
#endif

static int bq_forward(bonsai_context_t *c, int token, int want_logits) {
    double bq_t_start = bq_prof_enabled() ? bq_prof_clock() : 0.0;
    bonsai_model_t *m = c->model;
    const int n_embd = m->n_embd;
    const int n_ff = m->n_ff;
    const int n_qkv = 2 * m->ssm_groups * m->ssm_state + m->ssm_inner;
    const int hd = m->head_dim;
    const int S = m->ssm_state;
    const int H_v = m->ssm_dt_rank;
    const float *sign_embd = bq_signs(m, n_embd);
    const float *sign_inner = bq_signs(m, m->ssm_inner);
    const float *sign_ff = bq_signs(m, n_ff);
    int i, l, gidx = 0, fidx = 0;

    if (token < 0 || token >= m->n_vocab) return -1;
    if (c->pos >= c->capacity) return -1;

    /* token embedding: the table rows are stored in the folded latent basis */
    {
        const uint8_t *row = m->tok_embd.data + (size_t)token * m->tok_embd.row_stride;
        bitnet_ptq1_0_dequantize_row(row, c->x, n_embd);
        bq_hadamard(c->x, n_embd, m->block_size);
        if (sign_embd != NULL) {
            for (i = 0; i < n_embd; ++i) c->x[i] *= sign_embd[i];
        }
    }

    for (l = 0; l < m->n_layer; ++l) {
        bq_layer_t *L = &m->layers[l];

        if (!bq_gpu_elt(c, BQ_VK_ELT_RMS, c->x, NULL, L->attn_norm, n_embd, 1, c->xn))
            bq_rms_norm(c->x, L->attn_norm, n_embd, m->eps, c->xn);
        bq_prep_input_p(m->pool, c->prep, c->xn, n_embd, sign_embd, m->block_size);

        if (L->is_full) {
            float *k = c->qkv;
            float *v = c->qkv + (size_t)m->n_head_kv * hd;
            float *q = c->z;
            float *Kc = c->kvc_k + (size_t)fidx * c->capacity * m->n_head_kv * hd;
            float *Vc = c->kvc_v + (size_t)fidx * c->capacity * m->n_head_kv * hd;
            int h, d;

            BQ_MV_GROUP(m, BQ_MV(&L->wq, c->prep, c->qg), BQ_MV(&L->wk, c->prep, k),
                        BQ_MV(&L->wv, c->prep, v));

            for (h = 0; h < m->n_head; ++h)
                memcpy(q + (size_t)h * hd, c->qg + (size_t)h * 2 * hd, (size_t)hd * sizeof(float));
            if (!(bq_gpu_elt(c, BQ_VK_ELT_RMS, q, NULL, L->q_norm, hd, m->n_head, q) &&
                  bq_gpu_elt(c, BQ_VK_ELT_RMS, k, NULL, L->k_norm, hd, m->n_head_kv, k))) {
                /* q may already carry the device result, so rebuild it first */
                for (h = 0; h < m->n_head; ++h)
                    memcpy(q + (size_t)h * hd, c->qg + (size_t)h * 2 * hd, (size_t)hd * sizeof(float));
                for (h = 0; h < m->n_head; ++h)
                    bq_rms_norm(q + (size_t)h * hd, L->q_norm, hd, m->eps, q + (size_t)h * hd);
                for (h = 0; h < m->n_head_kv; ++h)
                    bq_rms_norm(k + (size_t)h * hd, L->k_norm, hd, m->eps, k + (size_t)h * hd);
            }

            bq_rope(q, m->n_head, hd, m->n_rot, c->pos, m->rope_freq);
            bq_rope(k, m->n_head_kv, hd, m->n_rot, c->pos, m->rope_freq);

            memcpy(Kc + (size_t)c->pos * m->n_head_kv * hd, k,
                   (size_t)m->n_head_kv * hd * sizeof(float));
            memcpy(Vc + (size_t)c->pos * m->n_head_kv * hd, v,
                   (size_t)m->n_head_kv * hd * sizeof(float));

#if defined(BITNET_ENABLE_VULKAN)
            {
                int on_gpu = 0;
                if (c->gpu_stages) {
                    memcpy(c->gpu_q, q, (size_t)m->n_head * hd * sizeof(float));
                    on_gpu = bq_vk_attn(m->vk, c->gpu_q, Kc, Vc, m->n_head, m->n_head_kv, hd,
                                        c->pos + 1, m->n_head / m->n_head_kv,
                                        1.0f / sqrtf((float)hd), c->attn_out) == 0;
                }
                {
                    double t_a = bq_prof_enabled() ? bq_prof_clock() : 0.0;
                    if (!on_gpu) bq_attention(c, fidx, q, Kc, Vc, c->pos, c->attn_out);
                    if (t_a != 0.0) g_bq_t_attn += bq_prof_clock() - t_a;
                }
            }
#else
            bq_attention(c, fidx, q, Kc, Vc, c->pos, c->attn_out);
#endif

            for (h = 0; h < m->n_head; ++h) {
                for (d = 0; d < hd; ++d) {
                    float g = c->qg[(size_t)h * 2 * hd + hd + d];
                    c->attn_out[(size_t)h * hd + d] *= bq_sigmoid(g);
                }
            }

            bq_prep_input_p(m->pool, c->prep2, c->attn_out, m->ssm_inner, sign_inner, m->block_size);
            bq_matvec(m, &L->wo, c->prep2, c->xn);
            ++fidx;
        } else {
            float *hist = c->convh + (size_t)gidx * m->ssm_conv_k * n_qkv;
            float *Mall = c->ssm + (size_t)gidx * H_v * S * S;
            float *q, *k, *v;
            int h, d;

            /* alpha and beta are per-head, so their matrices are 48 rows and
             * never go to the GPU; as their own group they cost a pool wake
             * each and nothing overlaps them.  They read a different
             * activation from qkv but depend on nothing in the group, so they
             * ride along as CPU-only segments and are computed under the
             * GPU's slack. */
            BQ_MV_GROUP(m, BQ_MV(&L->wqkv, c->prep, c->qkv),
                        BQ_MV(&L->wqkv_gate, c->prep, c->z),
                        BQ_MV(&L->walpha, c->xn, c->alpha),
                        BQ_MV(&L->wbeta, c->xn, c->beta));

            /* causal depthwise conv over the raw mixed qkv */
            memmove(hist, hist + n_qkv, (size_t)(m->ssm_conv_k - 1) * n_qkv * sizeof(float));
            memcpy(hist + (size_t)(m->ssm_conv_k - 1) * n_qkv, c->qkv,
                   (size_t)n_qkv * sizeof(float));
            if (!bq_gpu_conv(c, L->ssm_conv, hist, m->ssm_conv_k, n_qkv, c->conv_out)) {
                double t_c = bq_prof_enabled() ? bq_prof_clock() : 0.0;
                for (i = 0; i < n_qkv; ++i) {
                    const float *w = L->ssm_conv + (size_t)i * m->ssm_conv_k;
                    float s = 0.0f;
                    int t;
                    for (t = 0; t < m->ssm_conv_k; ++t) {
                        s += w[t] * hist[(size_t)t * n_qkv + i];
                    }
                    c->conv_out[i] = bq_silu(s);
                }
                if (t_c != 0.0) g_bq_t_conv += bq_prof_clock() - t_c;
            }

            q = c->conv_out;
            k = c->conv_out + (size_t)m->ssm_groups * S;
            v = c->conv_out + (size_t)2 * m->ssm_groups * S;

            if (!(bq_gpu_elt(c, BQ_VK_ELT_L2, q, NULL, NULL, S, m->ssm_groups, q) &&
                  bq_gpu_elt(c, BQ_VK_ELT_L2, k, NULL, NULL, S, m->ssm_groups, k))) {
                for (h = 0; h < m->ssm_groups; ++h) {
                    bq_l2_norm(q + (size_t)h * S, S, m->eps);
                    bq_l2_norm(k + (size_t)h * S, S, m->eps);
                }
            }

            {
                float dbe[256];
                int on_gpu = 0;
                if (H_v <= 128) {
                    for (h = 0; h < H_v; ++h) {
                        float g0 = L->ssm_a[h] * bq_softplus(c->alpha[h] + L->ssm_dt[h]);
                        dbe[h] = expf(g0);
                        dbe[128 + h] = bq_sigmoid(c->beta[h]);
                    }
                    on_gpu = bq_gpu_gdn(c, gidx, q, k, v, dbe, dbe + 128, H_v, m->ssm_groups,
                                        S, c->gdn_out);
                }
                if (!on_gpu) {
                    double t_g = bq_prof_enabled() ? bq_prof_clock() : 0.0;
                    bq_gdn_arg_t ga;
                    ga.ssm = Mall;
                    ga.q = q;
                    ga.k = k;
                    ga.v = v;
                    ga.decay = dbe;
                    ga.beta = dbe + 128;
                    ga.a_log = L->ssm_a;
                    ga.dt = L->ssm_dt;
                    ga.alpha = c->alpha;
                    ga.beta_raw = c->beta;
                    ga.inline_dbe = H_v > 128;
                    ga.out = c->gdn_out;
                    ga.n_heads = H_v;
                    ga.groups = m->ssm_groups;
                    ga.S = S;
                    ga.next = 0;
                    bq_pool_run(m->pool, bq_gdn_job, &ga);
                    if (t_g != 0.0) g_bq_t_gdn += bq_prof_clock() - t_g;
                }
            }

            /* gated RMS norm, then the head permutation back to grouped order */
            if (!bq_gpu_elt(c, BQ_VK_ELT_RMS_GATE, c->gdn_out, c->z, L->ssm_norm, S,
                            H_v, c->attn_out)) {
                for (h = 0; h < H_v; ++h) {
                    bq_rms_norm(c->gdn_out + (size_t)h * S, L->ssm_norm, S, m->eps,
                                c->attn_out + (size_t)h * S);
                    for (d = 0; d < S; ++d) {
                        c->attn_out[(size_t)h * S + d] *= bq_silu(c->z[(size_t)h * S + d]);
                    }
                }
            }

            if (m->gdn_v_grouped) {
                const int rep = H_v / m->ssm_groups;
                int kk, r;
                for (kk = 0; kk < m->ssm_groups; ++kk) {
                    for (r = 0; r < rep; ++r) {
                        for (d = 0; d < S; ++d) {
                            c->prep2[(size_t)d + S * (r + rep * kk)] =
                                c->attn_out[(size_t)d + S * (kk + m->ssm_groups * r)];
                        }
                    }
                }
                bq_prep_input_p(m->pool, c->prep2, c->prep2, m->ssm_inner, sign_inner, m->block_size);
            } else {
                bq_prep_input_p(m->pool, c->prep2, c->attn_out, m->ssm_inner, sign_inner, m->block_size);
            }

            bq_matvec(m, &L->wssm_out, c->prep2, c->xn);
            ++gidx;
        }

        if (!bq_gpu_elt(c, BQ_VK_ELT_ADD, c->x, c->xn, NULL, n_embd, 1, c->x))
            for (i = 0; i < n_embd; ++i) c->x[i] += c->xn[i];

        memcpy(c->resid, c->x, (size_t)n_embd * sizeof(float));
        if (!bq_gpu_elt(c, BQ_VK_ELT_RMS, c->x, NULL, L->post_attn_norm, n_embd, 1, c->xn))
            bq_rms_norm(c->x, L->post_attn_norm, n_embd, m->eps, c->xn);
        bq_prep_input_p(m->pool, c->prep, c->xn, n_embd, sign_embd, m->block_size);

        BQ_MV_GROUP(m, BQ_MV(&L->ffn_gate, c->prep, c->ffn_gate),
                    BQ_MV(&L->ffn_up, c->prep, c->ffn_up));
        if (!bq_gpu_elt(c, BQ_VK_ELT_SWIGLU, c->ffn_gate, c->ffn_up, NULL, n_ff, 1, c->ffn_gate))
            for (i = 0; i < n_ff; ++i) c->ffn_gate[i] = bq_silu(c->ffn_gate[i]) * c->ffn_up[i];

        bq_prep_input_p(m->pool, c->prep, c->ffn_gate, n_ff, sign_ff, m->block_size);
        bq_matvec(m, &L->ffn_down, c->prep, c->xn);

        if (!bq_gpu_elt(c, BQ_VK_ELT_ADD, c->resid, c->xn, NULL, n_embd, 1, c->x))
            for (i = 0; i < n_embd; ++i) c->x[i] = c->resid[i] + c->xn[i];
    }

    c->pos++;

    if (want_logits) {
        if (!bq_gpu_elt(c, BQ_VK_ELT_RMS, c->x, NULL, m->out_norm, n_embd, 1, c->xn))
            bq_rms_norm(c->x, m->out_norm, n_embd, m->eps, c->xn);
        bq_prep_input_p(m->pool, c->prep, c->xn, n_embd, sign_embd, m->block_size);
        bq_matvec(m, &m->output, c->prep, c->logits);
    }

    if (bq_t_start != 0.0) {
        g_bq_t_fwd += bq_prof_clock() - bq_t_start;
        ++g_bq_n_fwd;
    }
    return 0;
}


/* ------------------------------------------------------------------ */
/* block verify: one weight pass over several positions               */
/* ------------------------------------------------------------------ */

/* Speculative decoding needs the target to score a whole block of draft
 * tokens in one pass.  The weights are read once per layer either way, so
 * batching the activations is pure arithmetic gain; the pieces that are
 * sequential in the model - the gated delta net recurrence, the causal
 * convolution history and attention itself - still run a position at a time,
 * in order.  Hidden states of chosen layers can be handed back for a draft
 * model that conditions on them. */

#define BQ_BLK_MAX 16

struct bq_blk {
    int n_embd, n_ff, n_qkv, n_inner, hd, n_head, n_head_kv;
    float *x, *xn, *res, *prep, *fg, *fu, *att, *gdn, *z, *conv;
    float *qg, *qkv, *kbuf, *vbuf, *alpha, *beta, *qcur, *lg;
    float *ascore;   /* [nthreads][capacity] attention score scratch */
    int cap;
    int8_t *q8;
    int8_t *qgrp;    /* the same activation rows in the Vulkan group layout */
    float *qs, *qs2;
    void *p2prep;    /* activation tiles + block sums for the i8mm matmat */
    size_t p2prep_size;
};

static void bq_blk_free(bq_blk_t *b) {
    if (b == NULL) return;
    free(b->x); free(b->xn); free(b->res); free(b->prep); free(b->fg); free(b->fu);
    free(b->att); free(b->gdn); free(b->z); free(b->conv); free(b->qg); free(b->qkv);
    free(b->kbuf); free(b->vbuf); free(b->alpha); free(b->beta); free(b->qcur); free(b->lg);
    free(b->ascore); free(b->q8); free(b->qgrp); free(b->qs); free(b->qs2);
    free(b->p2prep); free(b);
}

static bq_blk_t *bq_blk_new(bonsai_model_t *m, int B, int capacity) {
    bq_blk_t *b = (bq_blk_t *)calloc(1, sizeof(*b));
    const int n_embd = m->n_embd, n_ff = m->n_ff, n_inner = m->ssm_inner;
    const int n_qkv = 2 * m->ssm_groups * m->ssm_state + m->ssm_inner;
    const int q_dim = m->n_head * m->head_dim;
    const int kv_dim = m->n_head_kv * m->head_dim;
    const size_t nw = (size_t)B;
    size_t nmax = (size_t)n_ff;
    if ((size_t)n_inner > nmax) nmax = (size_t)n_inner;
    if ((size_t)n_qkv > nmax) nmax = (size_t)n_qkv;
    if ((size_t)(2 * q_dim) > nmax) nmax = (size_t)(2 * q_dim);
    if (b == NULL) return NULL;
    b->n_embd = n_embd; b->n_ff = n_ff; b->n_qkv = n_qkv; b->n_inner = n_inner;
    b->hd = m->head_dim; b->n_head = m->n_head; b->n_head_kv = m->n_head_kv;
    b->x = (float *)calloc(nw * n_embd, sizeof(float));
    b->xn = (float *)calloc(nw * n_embd, sizeof(float));
    b->res = (float *)calloc(nw * n_embd, sizeof(float));
    b->prep = (float *)calloc(nw * nmax, sizeof(float));
    b->fg = (float *)calloc(nw * n_ff, sizeof(float));
    b->fu = (float *)calloc(nw * n_ff, sizeof(float));
    b->att = (float *)calloc(nw * n_inner, sizeof(float));
    b->gdn = (float *)calloc(nw * n_inner, sizeof(float));
    b->z = (float *)calloc(nw * n_inner, sizeof(float));
    b->conv = (float *)calloc(nw * n_qkv, sizeof(float));
    b->qg = (float *)calloc(nw * 2 * q_dim, sizeof(float));
    b->qkv = (float *)calloc(nw * n_qkv, sizeof(float));
    b->kbuf = (float *)calloc(nw * kv_dim, sizeof(float));
    b->vbuf = (float *)calloc(nw * kv_dim, sizeof(float));
    b->alpha = (float *)calloc(nw * m->ssm_dt_rank, sizeof(float));
    b->beta = (float *)calloc(nw * m->ssm_dt_rank, sizeof(float));
    b->qcur = (float *)calloc(nw * q_dim, sizeof(float));
    b->q8 = (int8_t *)calloc(nw * nmax, sizeof(int8_t));
    b->qgrp = (int8_t *)calloc(nw * nmax, sizeof(int8_t));
    b->qs = (float *)calloc(nw, sizeof(float));
    b->qs2 = (float *)calloc(nw, sizeof(float));
    b->lg = (float *)calloc(nw * (size_t)m->n_vocab, sizeof(float));
    b->cap = capacity;
    b->ascore = (float *)calloc((size_t)m->pool->nthreads * (size_t)capacity, sizeof(float));
    if (!b->x || !b->xn || !b->res || !b->prep || !b->fg || !b->fu || !b->att || !b->gdn ||
        !b->z || !b->conv || !b->qg || !b->qgrp || !b->qkv || !b->kbuf || !b->vbuf || !b->alpha ||
        !b->beta || !b->qcur || !b->ascore || !b->q8 || !b->qs || !b->lg) {
        bq_blk_free(b);
        return NULL;
    }
    return b;
}

typedef struct {
    bq_blk_t *blk;
    const bq_w_t *w;
    const float *xb;
    float *yb;
    int B;
    int next;
    const void *p2;   /* prepared activation tiles when the i8mm path is used */
} bq_mm_arg_t;

static void bq_mm_job(void *p, int tid, int nthreads) {
    bq_mm_arg_t *a = (bq_mm_arg_t *)p;
    const int n_out = a->w->n_out;
    const int n_in = a->w->n_in;
    (void)tid;
    (void)nthreads;
    for (;;) {
        int start = __atomic_fetch_add(&a->next, 8, __ATOMIC_RELAXED);
        int end = start + 8;
        int j, b2;
        if (start >= n_out) break;
        if (end > n_out) end = n_out;
        if (a->p2 != NULL) {
            const size_t pb = bitnet_ptq1_0_p2_pair_bytes(n_in);
            const size_t db = (size_t)(n_in / 128) * 2u;
            j = start;
            if (a->B <= 2)      /* two row pairs share the tile loads */
                for (; j + 3 < end; j += 4)
                    bitnet_ptq1_0_matmat_p2q((const uint8_t *)a->w->p2 + (size_t)(j / 2) * pb,
                                             a->w->p2d + (size_t)(j / 2) * db,
                                             a->p2, a->blk->qs, a->B, n_out, j, a->yb);
            for (; j < end; j += 2)
                bitnet_ptq1_0_matmat_p2((const uint8_t *)a->w->p2 + (size_t)(j / 2) * pb,
                                        a->w->p2d + (size_t)(j / 2) * db,
                                        a->p2, a->blk->qs, a->B, n_out, j, a->yb);
        } else if (a->w->type == BQ_PTQ) {
            for (j = start; j < end; ++j) {
                float tmp[BQ_BLK_MAX];
                bitnet_ptq1_0_matmat_row(a->w->data + (size_t)j * a->w->row_stride,
                                         a->blk->q8, a->blk->qs, n_in, a->B, tmp);
                for (b2 = 0; b2 < a->B; ++b2) a->yb[(size_t)b2 * n_out + j] = tmp[b2];
            }
        } else {
            for (j = start; j < end; ++j) {
                for (b2 = 0; b2 < a->B; ++b2) {
                    a->yb[(size_t)b2 * n_out + j] = bq_dot_row(a->w, j, a->xb + (size_t)b2 * n_in);
                }
            }
        }
    }
}

/* Rows of a block matmul the GPU should take, or 0 to keep it all on the CPU.
 * Unlike a matvec the two engines run the same integer kernel at about the
 * same rate here (158 against 141 G MAC/s measured on an Adreno 740 and an
 * X3/A715 mix), but a block verify is a chain of 62 layers of small matmuls,
 * so each dispatch also pays a launch and a fence: 40% to the GPU leaves the
 * CPU enough rows to fill that gap, and 30% or 50% both cost 10% more
 * (measured at B=3: 615 ms/round against 714 and 685).  A block is only worth
 * a dispatch from two positions up, where the CPU is compute bound rather
 * than waiting on the weights it streams. */
static int bq_mm_gpu_rows(bonsai_model_t *m, const bq_w_t *w, int B) {
#if defined(BITNET_ENABLE_VULKAN)
    int grow;
    if (m->vk == NULL || w->vk_buf == NULL || m->mm_gpu_pct <= 0) return 0;
    if (B < 2 || B > 16) return 0;
    if (!bq_vk_ptq_matmat_available(m->vk)) return 0;
    if (w->n_out < m->gpu_min_rows || w->n_in % 128 != 0) return 0;
    grow = (int)((int64_t)w->n_out * m->mm_gpu_pct / 100);
    if (grow < 1 || grow > w->n_out) return 0;
    return grow;
#else
    (void)m; (void)w; (void)B;
    return 0;
#endif
}

/* Quantize one activation block.  The CPU kernel reads the digit-major
 * packing, the GPU shader the group-major one, so the shader's copy is only
 * made when the GPU is actually given rows. */
static void bq_mm_quant(bq_blk_t *blk, const bq_w_t *w, const float *xb, int B, int groups) {
    int b;
    for (b = 0; b < B; ++b) {
        bitnet_quantize_row_i8(xb + (size_t)b * w->n_in, blk->q8 + (size_t)b * w->n_in,
                               &blk->qs[b], w->n_in);
#if defined(BITNET_ENABLE_VULKAN)
        if (groups)
            bitnet_quantize_row_i8_groups(xb + (size_t)b * w->n_in,
                                          blk->qgrp + (size_t)b * w->n_in,
                                          &blk->qs2[b], w->n_in);
#else
        (void)groups;
#endif
    }
}

/* Rows are eaten two at a time by the i8mm kernel, which packs an output row
 * pair into one stream: the GPU share has to hand back an even boundary. */
static int n_out_left(const bq_w_t *w, int from) { return w->n_out - from >= 2; }

/* 2-bit repacked weights for the i8mm matmat: 64 bytes per 256 weights
 * against 56 for the 28-byte layout, so it is not a memory saving, and at B=1
 * half its lanes idle.  Opt-in for that reason — the repack is a load-time
 * pass plus 0.28 bytes/weight of RAM and only the block verify (B >= 2) earns
 * it back, so the speculative path turns it on for itself. */
static int bq_p2_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("BQ_P2");
        cached = (e != NULL && atoi(e) != 0) ? bitnet_ptq1_0_p2_available() : 0;
    }
    return cached;
}

/* CPU part of a block matmul: rows [from, n_out). */
static void bq_mm_cpu(bonsai_model_t *m, bq_blk_t *blk, const bq_w_t *w,
                      const float *xb, int B, int from, float *yb) {
    bq_mm_arg_t a;
    a.blk = blk;
    a.w = w;
    a.xb = xb;
    a.yb = yb;
    a.B = B;
    a.next = from;
    a.p2 = NULL;
    if (w->p2 != NULL && B > 0 && B <= BQ_BLK_MAX && (from & 1) == 0 && n_out_left(w, from)) {
        const size_t need = bitnet_ptq1_0_p2_prep_bytes(w->n_in, B);
        if (blk->p2prep == NULL || blk->p2prep_size < need) {
            void *np = realloc(blk->p2prep, need);
            if (np != NULL) { blk->p2prep = np; blk->p2prep_size = need; }
        }
        if (blk->p2prep != NULL) {
            bitnet_ptq1_0_p2_prep(blk->p2prep, blk->q8, w->n_in, B);
            a.p2 = blk->p2prep;
        }
    }
    if ((size_t)(w->n_out - from) < 256) bq_mm_job(&a, 0, 1);
    else bq_pool_run(m->pool, bq_mm_job, &a);
}

/* yb[b][row] = W[row] . xb[b] for b < B */
static void bq_matmat(bonsai_model_t *m, bq_blk_t *blk, const bq_w_t *w,
                      const float *xb, int B, float *yb) {
    int grow = 0, submitted = 0;
    double t_mm = bq_prof_enabled() ? bq_prof_clock() : 0.0;
    if (w->type == BQ_PTQ) {
        grow = bq_mm_gpu_rows(m, w, B);
        if (w->p2 != NULL && (grow & 1)) --grow;
        bq_mm_quant(blk, w, xb, B, grow > 0);
#if defined(BITNET_ENABLE_VULKAN)
        if (grow > 0) {
            if (bq_vk_ptq_matmat_begin(m->vk, blk->qgrp, blk->qs, w->n_in, B, grow) == 0 &&
                bq_vk_ptq_matmat_add(m->vk, (const bq_vk_buf_t *)w->vk_buf, w->row_stride,
                                     0, grow, w->n_out, yb) == 0 &&
                bq_vk_ptq_matmat_end(m->vk) == 0) {
                submitted = 1;
            } else {
                grow = 0;
            }
        }
        (void)submitted;
#endif
    }
    bq_mm_cpu(m, blk, w, xb, B, grow, yb);
#if defined(BITNET_ENABLE_VULKAN)
    if (submitted) bq_vk_wait(m->vk);
#endif
    if (t_mm != 0.0) g_bq_t_mm += bq_prof_clock() - t_mm;
}

/* Several matrices fed by one activation, in a single dispatch round trip.
 * Every output here is on the critical path, so the extra dispatches ride in
 * the same command buffer while the CPU works through the earlier matrices'
 * rows: one fence wait for the whole group instead of one per matrix.  On a
 * driver where a round trip costs far more than the dispatch it carries (the
 * Mali driver here answers a fence in about 0.6 ms) that is the difference
 * between waiting on the GPU and waiting on the clock.  Splitting the group
 * back into separate bq_matmat() calls is exact, only slower. */
#define BQ_MULTI_MAX 8
typedef struct { const bq_w_t *w; float *y; } bq_mmi_t;

static void bq_matmat_multi(bonsai_model_t *m, bq_blk_t *blk, const bq_mmi_t *it, int n,
                            const float *xb, int B) {
    int g[BQ_MULTI_MAX], submitted = 0, all_ptq = 1, sum = 0, i;
    double t_mm = bq_prof_enabled() ? bq_prof_clock() : 0.0;
    if (n > BQ_MULTI_MAX) n = BQ_MULTI_MAX;
    for (i = 0; i < n; ++i)
        if (it[i].w->type != BQ_PTQ || it[i].w->n_in != it[0].w->n_in) all_ptq = 0;
    if (!all_ptq) {
        for (i = 0; i < n; ++i) bq_matmat(m, blk, it[i].w, xb, B, it[i].y);
        if (t_mm != 0.0) g_bq_t_mm += bq_prof_clock() - t_mm;
        return;
    }
    for (i = 0; i < n; ++i) {
        g[i] = bq_mm_gpu_rows(m, it[i].w, B);
        if (it[i].w->p2 != NULL && (g[i] & 1)) --g[i];
        sum += g[i];
    }
    bq_mm_quant(blk, it[0].w, xb, B, sum > 0);
#if defined(BITNET_ENABLE_VULKAN)
    if (sum > 0) {
        int ok;
        ok = bq_vk_ptq_matmat_begin(m->vk, blk->qgrp, blk->qs, it[0].w->n_in, B, sum) == 0;
        for (i = 0; ok && i < n; ++i) {
            if (g[i] > 0 &&
                bq_vk_ptq_matmat_add(m->vk, (const bq_vk_buf_t *)it[i].w->vk_buf,
                                     it[i].w->row_stride, 0, g[i], it[i].w->n_out,
                                     it[i].y) != 0)
                ok = 0;
        }
        if (ok && bq_vk_ptq_matmat_end(m->vk) == 0) submitted = 1;
        else for (i = 0; i < n; ++i) g[i] = 0;
    }
    (void)submitted;
#endif
    for (i = 0; i < n; ++i) bq_mm_cpu(m, blk, it[i].w, xb, B, g[i], it[i].y);
#if defined(BITNET_ENABLE_VULKAN)
    if (submitted) bq_vk_wait(m->vk);
#endif
    if (t_mm != 0.0) g_bq_t_mm += bq_prof_clock() - t_mm;
}

static void bq_matmat_pair(bonsai_model_t *m, bq_blk_t *blk,
                           const bq_w_t *wa, float *ya, const bq_w_t *wb, float *yb,
                           const float *xb, int B) {
    bq_mmi_t it[2];
    it[0].w = wa; it[0].y = ya;
    it[1].w = wb; it[1].y = yb;
    bq_matmat_multi(m, blk, it, 2, xb, B);
}

/* attention for a whole block: one job per (position, head) */
typedef struct {
    const bonsai_model_t *m;
    bq_blk_t *blk;
    const float *q;      /* [B][n_head*hd] */
    const float *K, *V;  /* [capacity][n_kv*hd], this layer */
    int pos0;            /* context position of block token 0 */
    int B;
    int next;
} bq_attn_arg_t;

static void bq_attn_job(void *p, int tid, int nthreads) {
    bq_attn_arg_t *a = (bq_attn_arg_t *)p;
    const int n_head = a->m->n_head;
    const int hd = a->m->head_dim;
    const int nkv = a->m->n_head_kv;
    const int rep = n_head / nkv;
    const float scale = 1.0f / sqrtf((float)hd);
    float *sc = a->blk->ascore + (size_t)tid * (size_t)a->blk->cap;
    (void)nthreads;
    for (;;) {
        int idx = __atomic_fetch_add(&a->next, 1, __ATOMIC_RELAXED);
        int b, h, t, d, nt;
        const float *qh;
        int kh;
        float maxv, sum;
        float *o, *s;
        if (idx >= a->B * n_head) break;
        b = idx / n_head;
        h = idx % n_head;
        nt = a->pos0 + b + 1;
        s = sc;
        qh = a->q + (size_t)b * n_head * hd + (size_t)h * hd;
        kh = h / rep;
        maxv = -INFINITY;
        for (t = 0; t < nt; ++t) {
            const float *kt = a->K + ((size_t)t * nkv + kh) * hd;
            float v = 0.0f;
#if defined(__aarch64__) && defined(__ARM_NEON)
            v = bq_dot_f32(qh, kt, hd);
#else
            for (d = 0; d < hd; ++d) v += qh[d] * kt[d];
#endif
            v *= scale;
            s[t] = v;
            if (v > maxv) maxv = v;
        }
        sum = 0.0f;
        for (t = 0; t < nt; ++t) {
            float e = expf(s[t] - maxv);
            s[t] = e;
            sum += e;
        }
        o = a->blk->att + (size_t)b * a->m->ssm_inner + (size_t)h * hd;
        for (d = 0; d < hd; ++d) o[d] = 0.0f;
        for (t = 0; t < nt; ++t) {
            const float p = s[t] / sum;
            const float *vt = a->V + ((size_t)t * nkv + kh) * hd;
            for (d = 0; d < hd; ++d) o[d] += p * vt[d];
        }
    }
}


/* one position of a gated delta net layer, state carried in the context */
static void bq_gdn_step(bonsai_context_t *c, bq_blk_t *blk, const bq_layer_t *L, int gidx,
                        const float *qkv, const float *z, const float *alpha,
                        const float *beta, float *out, int b) {
    bonsai_model_t *m = c->model;
    const int n_qkv = blk->n_qkv;
    const int S = m->ssm_state;
    const int H_v = m->ssm_dt_rank;
    float *hist = c->convh + (size_t)gidx * m->ssm_conv_k * n_qkv;
    float *cv = blk->conv + (size_t)b * n_qkv;
    float *gd = blk->gdn + (size_t)b * blk->n_inner;
    float *q, *k, *v;
    int i, h, d;

    memmove(hist, hist + n_qkv, (size_t)(m->ssm_conv_k - 1) * n_qkv * sizeof(float));
    memcpy(hist + (size_t)(m->ssm_conv_k - 1) * n_qkv, qkv, (size_t)n_qkv * sizeof(float));
    for (i = 0; i < n_qkv; ++i) {
        const float *w = L->ssm_conv + (size_t)i * m->ssm_conv_k;
        float sum = 0.0f;
        int t;
        for (t = 0; t < m->ssm_conv_k; ++t) sum += w[t] * hist[(size_t)t * n_qkv + i];
        cv[i] = bq_silu(sum);
    }
    q = cv;
    k = cv + (size_t)m->ssm_groups * S;
    v = cv + (size_t)2 * m->ssm_groups * S;
    for (h = 0; h < m->ssm_groups; ++h) {
        bq_l2_norm(q + (size_t)h * S, S, m->eps);
        bq_l2_norm(k + (size_t)h * S, S, m->eps);
    }
    {
        float dbe[256];
        bq_gdn_arg_t ga;
        for (h = 0; h < H_v && h < 128; ++h) {
            dbe[h] = expf(L->ssm_a[h] * bq_softplus(alpha[h] + L->ssm_dt[h]));
            dbe[128 + h] = bq_sigmoid(beta[h]);
        }
        ga.ssm = c->ssm + (size_t)gidx * H_v * S * S;
        ga.q = q;
        ga.k = k;
        ga.v = v;
        ga.decay = dbe;
        ga.beta = dbe + 128;
        ga.a_log = L->ssm_a;
        ga.dt = L->ssm_dt;
        ga.alpha = alpha;
        ga.beta_raw = beta;
        ga.inline_dbe = 0;
        ga.out = gd;
        ga.n_heads = H_v;
        ga.groups = m->ssm_groups;
        ga.S = S;
        ga.next = 0;
        bq_pool_run(m->pool, bq_gdn_job, &ga);
    }
    for (h = 0; h < H_v; ++h) {
        bq_rms_norm(gd + (size_t)h * S, L->ssm_norm, S, m->eps, out + (size_t)h * S);
        for (d = 0; d < S; ++d) out[(size_t)h * S + d] *= bq_silu(z[(size_t)h * S + d]);
    }
}

/* Regroup the value heads the way wssm_out expects (prism.gdn_v_grouped). */
static void bq_gdn_permute(const bonsai_model_t *m, const float *src, float *dst) {
    const int S = m->ssm_state;
    const int H_v = m->ssm_dt_rank;
    const int rep = H_v / m->ssm_groups;
    int kk, r, d;
    for (kk = 0; kk < m->ssm_groups; ++kk) {
        for (r = 0; r < rep; ++r) {
            for (d = 0; d < S; ++d) {
                dst[(size_t)d + S * (r + rep * kk)] =
                    src[(size_t)d + S * (kk + m->ssm_groups * r)];
            }
        }
    }
}

static bq_blk_t *bq_blk_ensure(bonsai_context_t *c) {
    if (c->blk == NULL) c->blk = bq_blk_new(c->model, BQ_BLK_MAX, c->capacity);
    return c->blk;
}

/* Evaluate `n` tokens in one pass.  When logits != NULL it receives one row
 * per position (stride floats apart); when hid != NULL the block hidden
 * states after layers sel_layers[] are written as [n][n_sel*n_embd]; when
 * layer_in != NULL the residual stream entering every layer is recorded as
 * [n_layer][n][n_embd] for bonsai_replay_state(). */
int bonsai_eval_block(bonsai_context_t *c, const int *tokens, int n,
                      float *logits, int stride,
                      const int *sel_layers, int n_sel, float *hid,
                      float *layer_in) {
    bonsai_model_t *m;
    bq_blk_t *blk;
    if (c != NULL && c->alt != NULL)
        return minicpm_eval_block((minicpm_context_t *)c->alt, tokens, n, logits, stride);
    const int n_embd = c->model->n_embd;
    const int n_ff = c->model->n_ff;
    const int n_qkv = 2 * c->model->ssm_groups * c->model->ssm_state + c->model->ssm_inner;
    const int hd = c->model->head_dim;
    const int n_head = c->model->n_head;
    const int kv_dim = c->model->n_head_kv * hd;
    const float *sign_embd, *sign_inner, *sign_ff;
    int off, l, b, i;

    double t_blk = bq_prof_enabled() ? bq_prof_clock() : 0.0;

    if (c == NULL || tokens == NULL || n <= 0) return -1;
    m = c->model;
    if (c->pos + n > c->capacity) return -1;
    blk = bq_blk_ensure(c);
    if (blk == NULL) return -1;
    sign_embd = bq_signs(m, n_embd);
    sign_inner = bq_signs(m, m->ssm_inner);
    sign_ff = bq_signs(m, n_ff);

    for (off = 0; off < n; off += BQ_BLK_MAX) {
        const int B = (n - off) < BQ_BLK_MAX ? (n - off) : BQ_BLK_MAX;
        int gidx = 0, fidx = 0;

        for (b = 0; b < B; ++b) {
            int tok = tokens[off + b];
            float *x = blk->x + (size_t)b * n_embd;
            if (tok < 0 || tok >= m->n_vocab) return -1;
            bitnet_ptq1_0_dequantize_row(m->tok_embd.data + (size_t)tok * m->tok_embd.row_stride,
                                         x, n_embd);
            bq_hadamard(x, n_embd, m->block_size);
            if (sign_embd != NULL) for (i = 0; i < n_embd; ++i) x[i] *= sign_embd[i];
        }

        for (l = 0; l < m->n_layer; ++l) {
            const bq_layer_t *L = &m->layers[l];
            float *x = blk->x, *xn = blk->xn, *prep = blk->prep;

            if (layer_in != NULL) {
                memcpy(layer_in + ((size_t)l * n + off) * n_embd,
                       blk->x, (size_t)B * n_embd * sizeof(float));
            }
            for (b = 0; b < B; ++b) {
                bq_rms_norm(x + (size_t)b * n_embd, L->attn_norm, n_embd, m->eps,
                            xn + (size_t)b * n_embd);
                bq_prep_input(prep + (size_t)b * n_embd, xn + (size_t)b * n_embd, n_embd,
                              sign_embd, m->block_size);
            }

            if (L->is_full) {
                float *Kc = c->kvc_k + (size_t)fidx * c->capacity * kv_dim;
                float *Vc = c->kvc_v + (size_t)fidx * c->capacity * kv_dim;
                int h, d;

                {
                    bq_mmi_t qkv[3];
                    qkv[0].w = &L->wq; qkv[0].y = blk->qg;
                    qkv[1].w = &L->wk; qkv[1].y = blk->kbuf;
                    qkv[2].w = &L->wv; qkv[2].y = blk->vbuf;
                    bq_matmat_multi(m, blk, qkv, 3, prep, B);
                }

                for (b = 0; b < B; ++b) {
                    float *q = blk->qcur + (size_t)b * n_head * hd;
                    float *kb = blk->kbuf + (size_t)b * kv_dim;
                    for (h = 0; h < n_head; ++h)
                        memcpy(q + (size_t)h * hd, blk->qg + ((size_t)b * 2 * n_head + 2 * h) * hd,
                               (size_t)hd * sizeof(float));
                    for (h = 0; h < n_head; ++h)
                        bq_rms_norm(q + (size_t)h * hd, L->q_norm, hd, m->eps, q + (size_t)h * hd);
                    for (h = 0; h < m->n_head_kv; ++h)
                        bq_rms_norm(kb + (size_t)h * hd, L->k_norm, hd, m->eps, kb + (size_t)h * hd);
                    bq_rope(q, n_head, hd, m->n_rot, c->pos + b, m->rope_freq);
                    bq_rope(kb, m->n_head_kv, hd, m->n_rot, c->pos + b, m->rope_freq);
                    memcpy(Kc + (size_t)(c->pos + b) * kv_dim, kb, (size_t)kv_dim * sizeof(float));
                    memcpy(Vc + (size_t)(c->pos + b) * kv_dim,
                           blk->vbuf + (size_t)b * kv_dim, (size_t)kv_dim * sizeof(float));
                }

                {
                    bq_attn_arg_t aa;
                    aa.m = m;
                    aa.blk = blk;
                    aa.q = blk->qcur;
                    aa.K = Kc;
                    aa.V = Vc;
                    aa.pos0 = c->pos;
                    aa.B = B;
                    aa.next = 0;
                    if (B * n_head < 8) bq_attn_job(&aa, 0, 1);
                    else bq_pool_run(m->pool, bq_attn_job, &aa);
                }

                for (b = 0; b < B; ++b) {
                    float *at = blk->att + (size_t)b * blk->n_inner;
                    for (h = 0; h < n_head; ++h) {
                        for (d = 0; d < hd; ++d) {
                            float g = blk->qg[((size_t)b * 2 * n_head + 2 * h) * hd + hd + d];
                            at[(size_t)h * hd + d] *= bq_sigmoid(g);
                        }
                    }
                    bq_prep_input(prep + (size_t)b * blk->n_inner, at, m->ssm_inner,
                                  sign_inner, m->block_size);
                }
                bq_matmat(m, blk, &L->wo, prep, B, xn);
                ++fidx;
            } else {
                bq_matmat_pair(m, blk, &L->wqkv, blk->qkv, &L->wqkv_gate, blk->z,
                               prep, B);
                bq_matmat_pair(m, blk, &L->walpha, blk->alpha, &L->wbeta, blk->beta,
                               xn, B);

                if (c->rp != NULL) {
                    const size_t qrow = (size_t)c->rp_row;
                    for (b = 0; b < B; ++b) {
                        float *dst;
                        if (off + b >= BQ_BLK_MAX) break;
                        dst = c->rp + ((size_t)gidx * (size_t)BQ_BLK_MAX +
                                       (size_t)(off + b)) * qrow;
                        memcpy(dst, blk->qkv + (size_t)b * n_qkv,
                               (size_t)n_qkv * sizeof(float));
                        memcpy(dst + n_qkv, blk->alpha + (size_t)b * m->ssm_dt_rank,
                               (size_t)m->ssm_dt_rank * sizeof(float));
                        memcpy(dst + n_qkv + m->ssm_dt_rank,
                               blk->beta + (size_t)b * m->ssm_dt_rank,
                               (size_t)m->ssm_dt_rank * sizeof(float));
                    }
                }

                for (b = 0; b < B; ++b) {
                    float *at = blk->att + (size_t)b * blk->n_inner;
                    bq_gdn_step(c, blk, L, gidx,
                                blk->qkv + (size_t)b * n_qkv,
                                blk->z + (size_t)b * blk->n_inner,
                                blk->alpha + (size_t)b * m->ssm_dt_rank,
                                blk->beta + (size_t)b * m->ssm_dt_rank,
                                at, b);
                    if (m->gdn_v_grouped) {
                        float *pp = prep + (size_t)b * blk->n_inner;
                        bq_gdn_permute(m, at, pp);
                        bq_prep_input(pp, pp, m->ssm_inner, sign_inner, m->block_size);
                    } else {
                        bq_prep_input(prep + (size_t)b * blk->n_inner, at, m->ssm_inner,
                                      sign_inner, m->block_size);
                    }
                }
                bq_matmat(m, blk, &L->wssm_out, prep, B, xn);
                ++gidx;
            }

            for (b = 0; b < B; ++b) {
                float *xb = x + (size_t)b * n_embd;
                float *rr = blk->res + (size_t)b * n_embd;
                for (i = 0; i < n_embd; ++i) xb[i] += xn[(size_t)b * n_embd + i];
                memcpy(rr, xb, (size_t)n_embd * sizeof(float));
                bq_rms_norm(xb, L->post_attn_norm, n_embd, m->eps,
                            xn + (size_t)b * n_embd);
                bq_prep_input(prep + (size_t)b * n_embd, xn + (size_t)b * n_embd, n_embd,
                              sign_embd, m->block_size);
            }

            bq_matmat_pair(m, blk, &L->ffn_gate, blk->fg, &L->ffn_up, blk->fu,
                           prep, B);
            for (b = 0; b < B; ++b) {
                float *g2 = blk->fg + (size_t)b * n_ff;
                float *u2 = blk->fu + (size_t)b * n_ff;
                float *pp = prep + (size_t)b * n_ff;
                for (i = 0; i < n_ff; ++i) g2[i] = bq_silu(g2[i]) * u2[i];
                bq_prep_input(pp, g2, n_ff, sign_ff, m->block_size);
            }
            bq_matmat(m, blk, &L->ffn_down, prep, B, xn);

            for (b = 0; b < B; ++b) {
                float *xb = x + (size_t)b * n_embd;
                const float *rr = blk->res + (size_t)b * n_embd;
                for (i = 0; i < n_embd; ++i) xb[i] = rr[i] + xn[(size_t)b * n_embd + i];
            }

            if (hid != NULL && n_sel > 0) {
                for (i = 0; i < n_sel; ++i) {
                    if (sel_layers[i] == l) {
                        int bb;
                        for (bb = 0; bb < B; ++bb) {
                            memcpy(hid + ((size_t)(off + bb) * n_sel + i) * n_embd,
                                   x + (size_t)bb * n_embd, (size_t)n_embd * sizeof(float));
                        }
                        break;
                    }
                }
            }
            (void)n_qkv;
        }

        if (logits != NULL) {
            for (b = 0; b < B; ++b) {
                bq_rms_norm(blk->x + (size_t)b * n_embd, m->out_norm, n_embd, m->eps,
                            blk->xn + (size_t)b * n_embd);
                bq_prep_input(blk->prep + (size_t)b * n_embd, blk->xn + (size_t)b * n_embd,
                              n_embd, sign_embd, m->block_size);
            }
            bq_matmat(m, blk, &m->output, blk->prep, B, blk->lg);
            for (b = 0; b < B; ++b) {
                memcpy(logits + (size_t)(off + b) * stride, blk->lg + (size_t)b * m->n_vocab,
                       (size_t)m->n_vocab * sizeof(float));
            }
        }
        c->pos += B;
    }
    /* The cache holds one block's worth of positions, so it is only usable
     * when the call was a single chunk. */
    if (c->rp != NULL) c->rp_npos = (n <= BQ_BLK_MAX) ? n : 0;
    if (t_blk != 0.0) {
        g_bq_t_fwd += bq_prof_clock() - t_blk;
        g_bq_n_fwd += n;
    }
    return 0;
}

/* ---- recurrent state: snapshot, restore, replay a prefix ---------------- */

static size_t bq_ssm_floats(const bonsai_context_t *c) {
    const bonsai_model_t *m = c->model;
    const int n_qkv = 2 * m->ssm_groups * m->ssm_state + m->ssm_inner;
    return (size_t)m->n_gdn * m->ssm_dt_rank * m->ssm_state * m->ssm_state +
           (size_t)m->n_gdn * m->ssm_conv_k * n_qkv;
}

size_t bonsai_state_bytes(const bonsai_context_t *c) {
    if (c == NULL) return 0;
    if (c->alt != NULL) return minicpm_state_bytes((const minicpm_context_t *)c->alt);
    return bq_ssm_floats(c) * sizeof(float);
}

void bonsai_state_save(const bonsai_context_t *c, float *dst) {
    bonsai_model_t *m = c->model;
    if (c->alt != NULL) {
        minicpm_state_save((const minicpm_context_t *)c->alt, dst);
        return;
    }
    const int n_qkv = 2 * m->ssm_groups * m->ssm_state + m->ssm_inner;
    const size_t n_ssm = (size_t)m->n_gdn * m->ssm_dt_rank * m->ssm_state * m->ssm_state;
    const size_t n_conv = (size_t)m->n_gdn * m->ssm_conv_k * n_qkv;
    memcpy(dst, c->ssm, n_ssm * sizeof(float));
    memcpy(dst + n_ssm, c->convh, n_conv * sizeof(float));
}

void bonsai_state_load(bonsai_context_t *c, const float *src) {
    bonsai_model_t *m = c->model;
    if (c->alt != NULL) {
        minicpm_state_load((minicpm_context_t *)c->alt, src);
        return;
    }
    const int n_qkv = 2 * m->ssm_groups * m->ssm_state + m->ssm_inner;
    const size_t n_ssm = (size_t)m->n_gdn * m->ssm_dt_rank * m->ssm_state * m->ssm_state;
    const size_t n_conv = (size_t)m->n_gdn * m->ssm_conv_k * n_qkv;
    memcpy(c->ssm, src, n_ssm * sizeof(float));
    memcpy(c->convh, src + n_ssm, n_conv * sizeof(float));
}

/* Rewind the position counter to where the accepted prefix ends.  Attention
 * is addressed by position, so stale entries past the new end are simply
 * overwritten by the next pass. */
void bonsai_rewind(bonsai_context_t *c, int n) {
    if (c == NULL || n <= 0) return;
    if (c->alt != NULL) {
        minicpm_rewind((minicpm_context_t *)c->alt, n);
        return;
    }
    if (n > c->pos) n = c->pos;
    c->pos -= n;
}

/* Recompute the recurrent state for the first n positions after a rewind.
 * The residual stream each layer saw is in layer_in ([n_layer][n][n_embd]),
 * which is valid for any accepted prefix because the model is causal.  Only
 * the gated delta net's own inputs matter for its state, so the value
 * projection, the output projection and the feed-forward network are skipped. */
static double g_rp_mm, g_rp_step, g_rp_mm_seen, g_rp_step_seen;
static int bq_prof_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("BQ_PROFILE"); on = (e != NULL && e[0] == '1'); }
    return on;
}
static double bq_prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void bonsai_replay_state_stride(bonsai_context_t *c, const float *layer_in,
                               int stride, int n) {
    bonsai_model_t *m = c->model;
    if (c->alt != NULL) return;   /* KV cache is positional: rewind is enough */
    bq_blk_t *blk = bq_blk_ensure(c);
    const int n_embd = m->n_embd;
    const float *sign_embd = bq_signs(m, n_embd);
    int l, p, gidx = 0;

    if (blk == NULL || layer_in == NULL || n <= 0) return;
    if (stride < n) stride = n;
    if (n > BQ_BLK_MAX) n = BQ_BLK_MAX;
    for (l = 0; l < m->n_layer; ++l) {
        const bq_layer_t *L = &m->layers[l];
        const int n_qkv_l = 2 * m->ssm_groups * m->ssm_state + m->ssm_inner;
        if (L->is_full) continue;
        /* All accepted positions of a layer in one weight pass.  Run one
         * position at a time this re-streams the layer's qkv/alpha/beta
         * weights per position, which is 94 ms of DRAM traffic each (it was
         * 284 ms of a round at B=8); the rows are independent of each other
         * until the state update, which stays sequential below. */
        for (p = 0; p < n; ++p) {
            const float *x = layer_in + ((size_t)l * stride + p) * n_embd;
            float *xn = blk->xn + (size_t)p * n_embd;
            bq_rms_norm(x, L->attn_norm, n_embd, m->eps, xn);
            bq_prep_input(blk->prep + (size_t)p * n_embd, xn, n_embd, sign_embd,
                          m->block_size);
        }
        {
            double _t0 = bq_prof_on() ? bq_prof_now() : 0.0;
            bq_matmat(m, blk, &L->wqkv, blk->prep, n, blk->qkv);
            bq_matmat_pair(m, blk, &L->walpha, blk->alpha, &L->wbeta, blk->beta,
                           blk->xn, n);
            if (_t0 != 0.0) g_rp_mm += bq_prof_now() - _t0;
        }
        {
            double _t0 = bq_prof_on() ? bq_prof_now() : 0.0;
            for (p = 0; p < n; ++p) {
                bq_gdn_step(c, blk, L, gidx,
                            blk->qkv + (size_t)p * n_qkv_l,
                            blk->z,
                            blk->alpha + (size_t)p * m->ssm_dt_rank,
                            blk->beta + (size_t)p * m->ssm_dt_rank,
                            blk->att + (size_t)p * blk->n_inner, p);
            }
            if (_t0 != 0.0) g_rp_step += bq_prof_now() - _t0;
        }
        ++gidx;
    }
    if (bq_prof_on()) {
        fprintf(stderr, "[replay] n=%d mm %.1f ms step %.1f ms\n", n,
                1e3 * (g_rp_mm - g_rp_mm_seen), 1e3 * (g_rp_step - g_rp_step_seen));
        g_rp_mm_seen = g_rp_mm;
        g_rp_step_seen = g_rp_step;
    }
}

/* Allocate the rollback cache.  Without it the rollback re-runs the qkv and
 * alpha/beta matmuls to rebuild the recurrent state; with it those outputs,
 * which the verify already produced, are reused and only the recurrence runs.
 * Costs n_gdn * 16 * (n_qkv + 2*dt_rank) floats (a few MB). */
int bonsai_replay_cache_init(bonsai_context_t *c) {
    const bonsai_model_t *m;
    size_t n;

    if (c == NULL || c->model == NULL) return -1;
    if (c->alt != NULL) return -1;   /* nothing to cache for the minicpm graph */
    m = c->model;
    if (c->rp != NULL) return 0;
    c->rp_row = 2 * m->ssm_groups * m->ssm_state + m->ssm_inner + 2 * m->ssm_dt_rank;
    c->rp_layers = m->n_gdn;
    n = (size_t)m->n_gdn * (size_t)BQ_BLK_MAX * (size_t)c->rp_row;
    c->rp = (float *)malloc(n * sizeof(float));
    c->rp_npos = 0;
    if (c->rp == NULL) { c->rp_row = 0; c->rp_layers = 0; return -1; }
    return 0;
}

/* Rebuild the recurrent state for the first n positions of the block that was
 * just evaluated, reusing the gated delta net inputs recorded by that pass.
 * Falls back to the full replay when the cache is absent or stale. */
int bonsai_replay_state_cached(bonsai_context_t *c, const float *layer_in,
                               int stride, int n) {
    bonsai_model_t *m;
    bq_blk_t *blk;
    int l, p, gidx = 0;

    if (c == NULL) return -1;
    if (c->alt != NULL) return 0;
    m = c->model;
    if (c->rp == NULL || n <= 0 || n > c->rp_npos) {
        bonsai_replay_state_stride(c, layer_in, stride, n);
        return 0;
    }
    blk = bq_blk_ensure(c);
    if (blk == NULL) return -1;
    for (l = 0; l < m->n_layer; ++l) {
        const bq_layer_t *L = &m->layers[l];
        const int n_qkv = blk->n_qkv;
        if (L->is_full) continue;
        for (p = 0; p < n; ++p) {
            const float *src = c->rp +
                ((size_t)gidx * (size_t)BQ_BLK_MAX + (size_t)p) * (size_t)c->rp_row;
            bq_gdn_step(c, blk, L, gidx, src, blk->z,
                        src + n_qkv, src + n_qkv + m->ssm_dt_rank,
                        blk->att + (size_t)p * blk->n_inner, p);
        }
        ++gidx;
    }
    return 0;
}

void bonsai_replay_state(bonsai_context_t *c, const float *layer_in, int n) {
    bonsai_replay_state_stride(c, layer_in, n, n);
}

/* ------------------------------------------------------------------ */
/* public API                                                         */
/* ------------------------------------------------------------------ */

int bonsai_embedding_length(const bonsai_model_t *m) {
    return m->mc != NULL ? minicpm_embedding_length((const minicpm_model_t *)m->mc) : m->n_embd;
}
int bonsai_vocab_size(const bonsai_model_t *m) {
    return m->mc != NULL ? minicpm_vocab_size((const minicpm_model_t *)m->mc) : m->n_vocab;
}
int bonsai_layer_count(const bonsai_model_t *m) {
    return m->mc != NULL ? minicpm_layer_count((const minicpm_model_t *)m->mc) : m->n_layer;
}

int bonsai_tokenize(bonsai_model_t *m, const char *text, int *tokens, int max_tokens) {
    if (m->mc != NULL) return minicpm_tokenize((minicpm_model_t *)m->mc, text, tokens, max_tokens);
    return bq_tokenize(&m->tok, text, tokens, max_tokens);
}

int bonsai_decode_token(bonsai_model_t *m, int token, char *out, int out_size) {
    if (m->mc != NULL)
        return minicpm_decode_token((minicpm_model_t *)m->mc, token, out, out_size);
    return bq_detokenize(&m->tok, token, out, out_size);
}

int bonsai_eos_token(const bonsai_model_t *m) {
    return m->mc != NULL ? minicpm_eos_token((const minicpm_model_t *)m->mc) : m->tok.eos;
}
int bonsai_bos_token(const bonsai_model_t *m) {
    return m->mc != NULL ? minicpm_bos_token((const minicpm_model_t *)m->mc) : m->tok.bos;
}

int bonsai_token_is_eog(bonsai_model_t *m, int token) {
    if (m->mc != NULL) return minicpm_token_is_eog((minicpm_model_t *)m->mc, token);
    return token == m->tok.eos;
}

int bonsai_reset_context(bonsai_context_t *c) {
    if (c == NULL) return -1;
    if (c->alt != NULL) return minicpm_reset_context((minicpm_context_t *)c->alt);
    bq_reset_context(c);
    return 0;
}

int bonsai_context_pos(const bonsai_context_t *c) {
    return c->alt != NULL ? minicpm_context_pos((const minicpm_context_t *)c->alt) : c->pos;
}
int bonsai_context_limit(const bonsai_model_t *m) {
    if (m == NULL) return 0;
    if (m->mc != NULL) return minicpm_context_limit((const minicpm_model_t *)m->mc);
    return m->n_ctx_train;
}

void bonsai_park_workers(void) {
    bitnet_tq2_0_park_workers();
}

int bonsai_context_capacity(const bonsai_context_t *c) {
    return c->alt != NULL ? minicpm_context_capacity((const minicpm_context_t *)c->alt)
                          : c->capacity;
}

int bonsai_eval(bonsai_context_t *c, const int *tokens, int n_tokens) {
    int i;
    if (c == NULL || n_tokens < 0) return -1;
    if (c->alt != NULL) return minicpm_eval((minicpm_context_t *)c->alt, tokens, n_tokens);
    if (c->pos + n_tokens > c->capacity) return -1;
    for (i = 0; i < n_tokens; ++i) {
        if (bq_forward(c, tokens[i], i == n_tokens - 1) != 0) return -1;
    }
    return 0;
}

const float *bonsai_get_logits(const bonsai_context_t *c) {
    if (c->alt != NULL) return minicpm_get_logits((const minicpm_context_t *)c->alt);
    return c->logits;
}

int bonsai_sample_greedy(const bonsai_context_t *c) {
    const bonsai_model_t *m = c->model;
    int best = 0, i;
    float bv = -INFINITY;
    if (c->alt != NULL) return minicpm_sample_greedy((const minicpm_context_t *)c->alt);
#if defined(BITNET_ENABLE_VULKAN)
    {
        uint32_t tok;
        if ((c->gpu_stages & BQ_STAGE_SAMPLE) &&
            bq_vk_sample(m->vk, c->logits, m->n_vocab, 0, 0u, 1, 0.0f, 1.0f, &tok) == 0)
            return (int)tok;
    }
#endif
    for (i = 0; i < m->n_vocab; ++i) {
        if (c->logits[i] > bv) {
            bv = c->logits[i];
            best = i;
        }
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* draft-model support                                                */
/* ------------------------------------------------------------------ */
/* A DSpine-style draft conditions on the target's residual stream, reuses
 * the target's token embedding, and scores its proposals with the target's
 * LM head.  These entry points expose exactly those three things in the
 * true (unfolded) basis, plus the shared thread pool, so the draft can live
 * in its own translation unit without reaching into this one's statics. */

void *bonsai_pool(const bonsai_model_t *m) {
    return (m != NULL) ? (void *)m->pool : NULL;
}

int bonsai_pool_threads(const bonsai_model_t *m) {
    return (m != NULL && m->pool != NULL) ? m->pool->nthreads : 1;
}

void bonsai_pool_run(void *pool, void (*fn)(void *, int, int), void *arg) {
    if (pool == NULL || fn == NULL) {
        if (fn != NULL) fn(arg, 0, 1);
        return;
    }
    bq_pool_run((bq_pool_t *)pool, fn, arg);
}

int bonsai_block_max(void) { return BQ_BLK_MAX; }
int bonsai_fold_block(const bonsai_model_t *m) {
    return m->mc != NULL ? 0 : m->block_size;
}
const float *bonsai_fold_signs(const bonsai_model_t *m, int width) {
    return m->mc != NULL ? NULL : bq_signs(m, width);
}
void bonsai_fold_hadamard(float *x, int n, int block) { bq_hadamard(x, n, block); }

/* Dense (float32) row of a weight tensor, unpacked. */
static void bq_w_row_dense(const bq_w_t *w, int row, float *out) {
    const uint8_t *r = w->data + (size_t)row * w->row_stride;
    int i;
    switch (w->type) {
        case BQ_PTQ:
            bitnet_ptq1_0_dequantize_row(r, out, w->n_in);
            break;
        case BQ_BF16: {
            const uint16_t *p = (const uint16_t *)r;
            for (i = 0; i < w->n_in; ++i) out[i] = bq_bf16_to_f32(p[i]);
            break;
        }
        default:
            memcpy(out, r, (size_t)w->n_in * sizeof(float));
            break;
    }
}

/* Token embedding row in the true basis: E = D*H*z, the layout the residual
 * stream uses (see the file comment). */
void bonsai_embed_row(const bonsai_model_t *m, int token, float *out) {
    const float *signs;
    int i;
    if (m == NULL || out == NULL || m->mc != NULL) return;
    if (token < 0 || token >= m->n_vocab) return;
    bq_w_row_dense(&m->tok_embd, token, out);
    bq_hadamard(out, m->n_embd, m->block_size);
    signs = bq_signs(m, m->n_embd);
    if (signs != NULL) for (i = 0; i < m->n_embd; ++i) out[i] *= signs[i];
}

/* LM head row in the true basis: the same D*H as the embedding, because the
 * fold acts on the matmul input (W . (H D x)) and (H D)^T = D H. */
void bonsai_head_row(const bonsai_model_t *m, int token, float *out) {
    const float *signs;
    int i;
    if (m == NULL || out == NULL || m->mc != NULL) return;
    if (token < 0 || token >= m->n_vocab) return;
    bq_w_row_dense(&m->output, token, out);
    bq_hadamard(out, m->n_embd, m->block_size);
    signs = bq_signs(m, m->n_embd);
    if (signs != NULL) for (i = 0; i < m->n_embd; ++i) out[i] *= signs[i];
}

/* logits[b][:] = head . x[b] for b < n (n <= 16), x already final-normed. */
int bonsai_head_logits(bonsai_context_t *c, const float *x, int n,
                       float *logits, int stride) {
    bonsai_model_t *m;
    bq_blk_t *blk;
    const int n_embd = (c != NULL) ? c->model->n_embd : 0;
    const float *signs;
    int b;
    if (c == NULL || x == NULL || logits == NULL || n <= 0 || n > BQ_BLK_MAX) return -1;
    if (c->alt != NULL) return -1;   /* DSpine drafts pair with the qwen35 target */
    m = c->model;
    blk = bq_blk_ensure(c);
    if (blk == NULL) return -1;
    signs = bq_signs(m, n_embd);
    for (b = 0; b < n; ++b)
        bq_prep_input_p(m->pool, blk->prep + (size_t)b * n_embd,
                        x + (size_t)b * n_embd, n_embd, signs, m->block_size);
    bq_matmat(m, blk, &m->output, blk->prep, n, blk->lg);
    for (b = 0; b < n; ++b)
        memcpy(logits + (size_t)b * stride, blk->lg + (size_t)b * m->n_vocab,
               (size_t)m->n_vocab * sizeof(float));
    return 0;
}
