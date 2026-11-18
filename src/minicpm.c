/* MiniCPM4 / BitCPM-CANN runtime -- the "minicpm" family.
 *
 * The framework decides the family from general.architecture when the file
 * is opened; nothing here is chosen by a build flag or a model name.  The
 * graph is the dense decoder src/bitnet.c already implements for this
 * family: RMSNorm, GQA attention with LongRoPE, SwiGLU feed-forward, and the
 * three scale knobs the MiniCPM4 config turns on (embedding_scale on the
 * token embedding, residual_scale on every sublayer output, logit_scale on
 * the final logits).
 *
 * Every projection goes through bitnet_tq2_0_matmul_batched_i8(), the kernel
 * that dots one 256-weight TQ2_0 block against B activation rows.  A block
 * verify (B = 1..16) therefore reads each matrix once, while B = 1 is the
 * ordinary decode path over the very same kernel -- a token the verify
 * accepts is built by the arithmetic a plain decode would use.
 *
 * The head is a Q6_K matrix (or the tied f16 embedding on the 0.5B).  Either
 * way it is prepared once at load into the int8 layout the engine's output
 * pool dots (src/output_head.h) and read through that pool, whose workers
 * split the vocabulary across the same thread count as the projections.  The
 * per-token dequantize path it replaced was single-threaded and cost 15 ms of
 * the 20 ms a 0.5B token took.  A TQ2_0 head keeps the batched projection.
 */

#include "minicpm.h"

#include "gguf.h"
#include "ops.h"
#include "quant_q4k.h"
#include "quant_q6k.h"
#include "quant_q8k.h"
#include "quant_tq2_0.h"
#include "output_head.h"

#include "tensor.h"
#include "tokenizer.h"

#if defined(BITNET_ENABLE_VULKAN)
#include "vk_backend.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define MC_TYPE_F32   0u
#define MC_TYPE_F16   1u
#define MC_TYPE_Q4_K 12u
#define MC_TYPE_Q6_K 14u
#define MC_TYPE_TQ2_0 35u

#define MC_MAX_B 16

/* Rows scored per prefill chunk.  Every buffer here is sized for MC_MAX_B, so
 * this is a scheduling choice and not a capacity one.  The batch kernel's cost
 * per row is not flat in B: at B=16 the sixteen decoded code vectors and their
 * accumulator pairs no longer fit in the A77's vector registers, so the
 * default branch reloads them from the stack, while B=4 rides the quad form at
 * the chip's one vdot per cycle.  The 8B measured 14.9 tok/s of prefill at 4
 * rows against 11.0 at 16 on three threads (13.2 at 8); the 0.5B and 1B are
 * within 7% either way.  Four rows read the matrix four times as often as
 * sixteen, but at 8B that is 8 GB/s against a 23.5 GB/s read ceiling.  A call that scores
 * at most MC_MAX_B rows is a speculative verify and is never split -- it gets
 * the matrix in one pass.  BQ_MC_CHUNK overrides the prefill chunk. */
static int mc_chunk_b(int n) {
    static int chunk = -1;
    if (chunk < 0) {
        const char *e = getenv("BQ_MC_CHUNK");
        chunk = 4;
        if (e != NULL && e[0] != '\0') {
            int v = atoi(e);
            if (v >= 1 && v <= MC_MAX_B) chunk = v;
        }
    }
    return n <= MC_MAX_B ? n : chunk;
}

/* A verify dispatch costs about 0.6 ms of round trip whatever it carries, so
 * a matrix smaller than this is not worth splitting: the CPU finishes it
 * before the GPU could report back. */
#define MC_GPU_MIN_ROWS 512

/* ------------------------------------------------------------------ */
/* weights                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *data;
    float *scales;   /* [n_out][n_in/256] */
    int n_in;
    int n_out;
    int bpr;
    /* Under a block verify the GPU takes the last vk_rows rows of the matrix
     * while the CPU pool takes the head.  Only that tail is staged: the head
     * is read from the mapping, so the staged copy never doubles the model. */
    void *vk_buf;    /* bq_vk_buf_t *, NULL when unaccelerated */
    int vk_rows;
} mc_w_t;

typedef struct {
    const float *attn_norm;
    const float *ffn_norm;
    mc_w_t q, k, v, o, gate, up, down;
} mc_layer_t;

struct minicpm_model {
    gguf_file_t gguf;
    char family[24];   /* "minicpm" or "llama": metadata lives under this prefix */
    bitnet_tokenizer_t *tok;

#if defined(BITNET_ENABLE_VULKAN)
    bq_vk_t *vk;     /* Vulkan compute, shared by every projection */
    int gpu_pct;     /* rows of each verify matmul given to the GPU */
#endif

    int n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, n_rot;
    int q_dim, kv_dim, n_vocab, n_ctx_train;
    float eps, rope_base, emb_scale, res_scale, logit_scale;

    const float *rope_factors_short;
    const float *rope_factors_long;
    int rope_factor_count;

    const float *out_norm;
    const uint8_t *emb_data;
    const uint8_t *head_data;
    uint32_t emb_type, head_type;
    float *head_scales;   /* only when the head is TQ2_0 */

    /* Prepared output projection (src/output_head.h).  Built once at load from
     * whatever the head tensor is, so a decoded token dots int8 rows through
     * the engine's output pool instead of dequantizing the head on the
     * calling thread: the per-token dequantize path costs 15 ms of the 20 ms
     * a 0.5B token took.  NULL when the head keeps its native path. */
    int8_t *head_q8;        /* [n_vocab][n_embd] int8 rows (f16 head only) */
    float *head_q8_d;       /* [n_vocab][bpr] block scales */
    uint8_t *head_tq2;      /* [n_vocab][bpr] TQ2_0 blocks (BQ_MC_HEAD=2) */
    float *head_tq2_scales; /* [n_vocab][bpr] */

    mc_layer_t *layer;
};

/* ------------------------------------------------------------------ */
/* metadata                                                           */
/* ------------------------------------------------------------------ */

/* Everything is read under the family prefix the file itself carries, so a
 * renamed or re-quantised checkpoint with the same architecture loads. */
static const gguf_metadata_t *mc_md(const minicpm_model_t *m, const char *suffix) {
    char key[96];
    snprintf(key, sizeof(key), "%s.%s", m->family[0] ? m->family : "minicpm", suffix);
    return gguf_find_metadata(&m->gguf, key);
}

static int mc_md_int(const minicpm_model_t *m, const char *suffix, int def) {
    const gguf_metadata_t *e = mc_md(m, suffix);
    if (e == NULL) return def;
    switch (e->type) {
        case GGUF_TYPE_UINT32: return (int)e->value.u64;
        case GGUF_TYPE_INT32:  return (int)e->value.i64;
        case GGUF_TYPE_UINT64: return (int)e->value.u64;
        case GGUF_TYPE_INT64:  return (int)e->value.i64;
        default: return def;
    }
}

static float mc_md_float(const minicpm_model_t *m, const char *suffix, float def) {
    const gguf_metadata_t *e = mc_md(m, suffix);
    if (e == NULL) return def;
    switch (e->type) {
        case GGUF_TYPE_FLOAT32: return (float)e->value.f64;
        case GGUF_TYPE_FLOAT64: return (float)e->value.f64;
        default: return def;
    }
}

/* ------------------------------------------------------------------ */
/* loading                                                            */
/* ------------------------------------------------------------------ */

static int mc_bind_w(minicpm_model_t *m, const char *name, int n_in, int n_out, mc_w_t *w) {
    const gguf_tensor_t *t = gguf_find_tensor(&m->gguf, name);
    size_t count;

    memset(w, 0, sizeof(*w));
    if (t == NULL) {
        fprintf(stderr, "minicpm: missing tensor %s\n", name);
        return -1;
    }
    if (t->type != MC_TYPE_TQ2_0) {
        fprintf(stderr, "minicpm: tensor %s is %s, this graph needs TQ2_0\n",
                name, gguf_tensor_type_name(t->type));
        return -1;
    }
    w->data = (const uint8_t *)gguf_get_tensor_ptr(&m->gguf, t);
    w->n_in = n_in;
    w->n_out = n_out;
    w->bpr = n_in / 256;
    if (w->data == NULL) return -1;
    count = bitnet_tq2_0_scale_float_count(n_out, n_in);
    w->scales = (float *)malloc(count * sizeof(float));
    if (w->scales == NULL) return -1;
    if (bitnet_tq2_0_build_scales(w->data, n_out, n_in, w->scales, count) != 0) {
        free(w->scales);
        w->scales = NULL;
        return -1;
    }
#if defined(BITNET_ENABLE_VULKAN)
    /* Stage the tail the GPU will own.  The read goes straight from the file,
     * so those pages never enter the mapping and there is still only one
     * resident copy of the matrix. */
    if (m->vk != NULL && m->gpu_pct > 0) {
        const size_t row_bytes = (size_t)w->bpr * BITNET_TQ2_0_BLOCK_SIZE;
        int rows = (int)((int64_t)n_out * m->gpu_pct / 100);
        if (rows > n_out - 1) rows = n_out - 1;
        if (rows >= MC_GPU_MIN_ROWS) {
            bq_vk_buf_t *vb = NULL;
            void *dst = bq_vk_buf_alloc_cached(m->vk, (size_t)rows * row_bytes, &vb);
            if (dst != NULL &&
                gguf_read_tensor_at(&m->gguf, t, (size_t)(n_out - rows) * row_bytes,
                                    dst, (size_t)rows * row_bytes) == 0) {
                bq_vk_buf_flush(m->vk, vb);
                w->vk_buf = vb;
                w->vk_rows = rows;
            } else if (vb != NULL) {
                bq_vk_buf_free(m->vk, vb);
            }
        }
    }
#endif
    return 0;
}

static void mc_unbind_w(minicpm_model_t *m, mc_w_t *w) {
    free(w->scales);
    w->scales = NULL;
#if defined(BITNET_ENABLE_VULKAN)
    if (w->vk_buf != NULL && m->vk != NULL) bq_vk_buf_free(m->vk, (bq_vk_buf_t *)w->vk_buf);
#else
    (void)m;
#endif
    w->vk_buf = NULL;
    w->vk_rows = 0;
}

/* How to hold the head between the checkpoint and the per-token dot:
 *   2 (default)  TQ2_0 rows, 2.125 bits per weight, plus an exact rescore of
 *                the best rows from the checkpoint form (f16 or Q6_K, see
 *                mc_head_row).  A decode reads 4x less of the head.  On the
 *                865 board that is 0.5B 112 -> 155-162 tok/s, and 1B
 *                45.6 -> 51.8, 3B 20.8 -> 21.8, 8B 9.1 -> 9.4 as well,
 *                because their Q6_K output.weight alone is 124-249 MB per
 *                token;
 *   1            an int8 copy, one byte per weight (an f16 head only; a Q6_K
 *                head is read in place by the shared Q8_K matmul);
 *   0            no copy at all: the native per-token dequantize path. */
static int mc_head_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("BQ_MC_HEAD");
        mode = 2;
        if (e != NULL) {
            if (e[0] == '0') mode = 0;
            else if (e[0] == '1') mode = 1;
            else if (e[0] == '2') mode = 2;
        }
    }
    return mode;
}

/* One head row as f32, whichever form the checkpoint stores it in. */
static int mc_head_row_f32(const minicpm_model_t *m, int row, float *dst) {
    const int E = m->n_embd;
    const int bpr = E / 256;
    int i, b;

    if (m->head_type == MC_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)m->head_data + (size_t)row * (size_t)E;
        for (i = 0; i < E; ++i) dst[i] = bitnet_fp16_to_fp32(src[i]);
        return 0;
    }
    if (m->head_type == MC_TYPE_Q6_K) {
        const uint8_t *src = (const uint8_t *)m->head_data +
                             (size_t)row * (size_t)bpr * BITNET_Q6K_BLOCK_SIZE;
        for (b = 0; b < bpr; ++b)
            if (bitnet_q6k_dequantize_block(
                    (const bitnet_q6k_block_t *)(src + (size_t)b * BITNET_Q6K_BLOCK_SIZE),
                    dst + (size_t)b * 256, 256) != 0) return -1;
        return 0;
    }
    return -1;
}

/* The ternary copy of a head, in the same format the projections ship in:
 * one TQ2_0 block (64 packed trits + an f16 step) per 256 weights.  A decode
 * reads 19 MB instead of the 75 MB int8 copy (0.5B) or 124-300 MB of Q6_K
 * (1B/3B/8B).  Weights go through one f32 row at a time, so the head's
 * original form is only ever read once, here. */
static int mc_build_head_tq2(minicpm_model_t *m, int bpr) {
    const int E = m->n_embd;
    const size_t row_bytes = (size_t)bpr * BITNET_TQ2_0_BLOCK_SIZE;
    size_t count = bitnet_tq2_0_scale_float_count(m->n_vocab, E);
    float *fv;
    int row, b, i;

    if (count == 0) return -1;
    fv = (float *)malloc((size_t)E * sizeof(float));
    if (fv == NULL) return -1;
    m->head_tq2 = (uint8_t *)malloc((size_t)m->n_vocab * row_bytes);
    m->head_tq2_scales = (float *)malloc(count * sizeof(float));
    if (m->head_tq2 == NULL || m->head_tq2_scales == NULL) { free(fv); return -1; }

    for (row = 0; row < m->n_vocab; ++row) {
        uint8_t *dst_row = m->head_tq2 + (size_t)row * row_bytes;
        if (mc_head_row_f32(m, row, fv) != 0) { free(fv); return -1; }
        for (b = 0; b < bpr; ++b) {
            const float *src = fv + (size_t)b * 256;
            bitnet_tq2_0_block_t *blk = (bitnet_tq2_0_block_t *)(dst_row + (size_t)b * BITNET_TQ2_0_BLOCK_SIZE);
            float sum = 0.0f, d;
            for (i = 0; i < 256; ++i) sum += fabsf(src[i]);
            memset(blk->qs, 0, sizeof blk->qs);
            d = sum / 256.0f;
            blk->d = bitnet_fp32_to_fp16(d);
            if (d <= 0.0f) continue;
            for (i = 0; i < 256; ++i) {
                int q = (int)(src[i] / d + (src[i] >= 0.0f ? 0.5f : -0.5f));
                int pos, l, mm;
                if (q > 1) q = 1;
                if (q < -1) q = -1;
                /* Interleaved layout: byte g*32+m carries position
                 * g*128 + l*32 + m in bits 2l..2l+1. */
                pos = i;
                l = (pos % 128) / 32;
                mm = pos % 32;
                blk->qs[(pos / 128) * 32 + mm] |= (uint8_t)((q + 1) << (2 * l));
            }
        }
    }
    free(fv);
    if (bitnet_tq2_0_build_scales(m->head_tq2, m->n_vocab, E,
                                  m->head_tq2_scales, count) != 0) return -1;
    return 0;
}

/* Prepare the head, in the form mc_head_mode() picks.
 *
 * The int8 copy only serves the tied-embedding case: the 0.5B has no
 * output.weight at all (its head is the f16 token_embd) and dequantizing
 * that per token on the calling thread cost 15 ms of a 20 ms token.  A Q6_K
 * head needs no int8 copy -- it is already read in place by the shared Q8_K
 * matmul, which measured no slower than dotting an expanded copy (1B 53.5 vs
 * 52.0, 3B 25.4 vs 23.5, 8B 13.0 vs 12.0 tok/s) -- but it does get the
 * ternary copy in mode 2. */
static int mc_prepare_head(minicpm_model_t *m) {
    const int bpr = m->n_embd / 256;
    const int mode = mc_head_mode();
    /* BQ_MC_HEAD=0 keeps the head in its native form: same logits, 3-4x
     * slower decode, one matrix less per model (matters on a board that has
     * to hold a large target next to it). */
    if (mode == 0) return 0;
    if (m->n_embd <= 0 || m->n_embd % 256 != 0 || m->n_vocab <= 0) return 0;

    /* The Q6_K rescore quantizes the activation into a fixed Q8_K block
     * array, so a wider model than that array holds would overrun it. */
    if (mode == 2 && (m->head_type == MC_TYPE_F16 ||
                      (m->head_type == MC_TYPE_Q6_K &&
                       m->n_embd <= BITNET_Q8K_MAX_BLOCKS * 256))) {
        if (mc_build_head_tq2(m, bpr) == 0) return 0;
        /* Out of memory for the small copy: fall back to the int8 one. */
        free(m->head_tq2);
        free(m->head_tq2_scales);
        m->head_tq2 = NULL;
        m->head_tq2_scales = NULL;
    }

    if (m->head_type == MC_TYPE_F16) {
        const uint16_t *w = (const uint16_t *)m->head_data;
        int row;
        m->head_q8 = (int8_t *)malloc((size_t)m->n_vocab * (size_t)m->n_embd);
        m->head_q8_d = (float *)malloc((size_t)m->n_vocab * (size_t)bpr * sizeof(float));
        if (m->head_q8 == NULL || m->head_q8_d == NULL) goto drop;
        for (row = 0; row < m->n_vocab; ++row) {
            const uint16_t *src_row = w + (size_t)row * (size_t)m->n_embd;
            int8_t *dst_row = m->head_q8 + (size_t)row * (size_t)m->n_embd;
            float *d_row = m->head_q8_d + (size_t)row * (size_t)bpr;
            int b, i;
            for (b = 0; b < bpr; ++b) {
                const uint16_t *src_block = src_row + (size_t)b * 256;
                int8_t *dst_block = dst_row + (size_t)b * 256;
                float max_abs = 0.0f, inv;
                for (i = 0; i < 256; ++i) {
                    float a = fabsf(bitnet_fp16_to_fp32(src_block[i]));
                    if (a > max_abs) max_abs = a;
                }
                if (max_abs == 0.0f) {
                    memset(dst_block, 0, 256 * sizeof(*dst_block));
                    d_row[b] = 0.0f;
                    continue;
                }
                d_row[b] = max_abs / 127.0f;
                inv = 127.0f / max_abs;
                for (i = 0; i < 256; ++i) {
                    float scaled = bitnet_fp16_to_fp32(src_block[i]) * inv;
                    int q = (int)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    dst_block[i] = (int8_t)q;
                }
            }
        }
        return 0;
    }
    return 0;

drop:
    /* Not fatal: mc_head_row falls back to dequantizing the head per token. */
    fprintf(stderr, "minicpm: head is too large to prepare, using the per-token path\n");
    free(m->head_q8);
    free(m->head_q8_d);
    m->head_q8 = NULL;
    m->head_q8_d = NULL;
    return 0;
}

minicpm_model_t *minicpm_load_model(const char *path, int n_threads) {
    minicpm_model_t *m;
    const gguf_tensor_t *t;
    int i;
    (void)n_threads;   /* the shared row pool reads BITNET_NUM_THREADS */

    if (path == NULL) return NULL;
    m = (minicpm_model_t *)calloc(1, sizeof(*m));
    if (m == NULL) return NULL;
    if (gguf_open(path, &m->gguf) != 0) {
        fprintf(stderr, "minicpm: cannot open %s\n", path);
        free(m);
        return NULL;
    }

    /* MiniCPM4 1B/3B ship as general.architecture "llama": the same tensors
     * and the same graph, minus the MiniCPM4 scales, whose defaults are 1. */
    {
        const gguf_metadata_t *arch = gguf_find_metadata(&m->gguf, "general.architecture");
        const char *name = arch != NULL && arch->string_value != NULL ? arch->string_value
                                                                     : "minicpm";
        snprintf(m->family, sizeof m->family, "%s", name);
    }

    m->n_layer  = mc_md_int(m, "block_count", 0);
    m->n_embd   = mc_md_int(m, "embedding_length", 0);
    m->n_ff     = mc_md_int(m, "feed_forward_length", 0);
    m->n_head   = mc_md_int(m, "attention.head_count", 0);
    m->n_head_kv = mc_md_int(m, "attention.head_count_kv", 0);
    m->n_ctx_train = mc_md_int(m, "context_length", 0);
    m->eps      = mc_md_float(m, "attention.layer_norm_rms_epsilon", 1e-6f);
    m->rope_base = mc_md_float(m, "rope.freq_base", 10000.0f);
    m->emb_scale = mc_md_float(m, "embedding_scale", 1.0f);
    m->res_scale = mc_md_float(m, "residual_scale", 1.0f);
    m->logit_scale = mc_md_float(m, "logit_scale", 1.0f);

    t = gguf_find_tensor(&m->gguf, "token_embd.weight");
    if (t == NULL || t->n_dims != 2) {
        fprintf(stderr, "minicpm: missing token_embd.weight\n");
        goto fail;
    }
    if (m->n_embd == 0) m->n_embd = (int)t->dims[0];
    if (m->n_vocab == 0) m->n_vocab = (int)t->dims[1];
    m->emb_type = t->type;
    m->emb_data = (const uint8_t *)gguf_get_tensor_ptr(&m->gguf, t);

    t = gguf_find_tensor(&m->gguf, "output.weight");
    if (t == NULL || t->n_dims != 2) {
        /* Tied embeddings: the BitCPM 0.5B checkpoint has no output.weight and
         * reuses token_embd, which is legal GGUF (and what a small draft
         * model usually does). */
        t = gguf_find_tensor(&m->gguf, "token_embd.weight");
        if (t == NULL || t->n_dims != 2) {
            fprintf(stderr, "minicpm: no output.weight and no token_embd.weight\n");
            goto fail;
        }
        fprintf(stderr, "minicpm: no output.weight, head tied to token_embd (%s)\n",
                gguf_tensor_type_name(t->type));
    }
    if (m->n_vocab == 0) m->n_vocab = (int)t->dims[1];
    m->head_type = t->type;
    m->head_data = (const uint8_t *)gguf_get_tensor_ptr(&m->gguf, t);
    if (m->head_type != MC_TYPE_Q6_K && m->head_type != MC_TYPE_TQ2_0 &&
        m->head_type != MC_TYPE_F16) {
        fprintf(stderr, "minicpm: output weight is %s, expected Q6_K, TQ2_0 or F16\n",
                gguf_tensor_type_name(m->head_type));
        goto fail;
    }
    if (m->head_type == MC_TYPE_TQ2_0) {
        size_t count = bitnet_tq2_0_scale_float_count(m->n_vocab, m->n_embd);
        m->head_scales = (float *)malloc(count * sizeof(float));
        if (m->head_scales == NULL) goto fail;
        if (bitnet_tq2_0_build_scales(m->head_data, m->n_vocab, m->n_embd,
                                      m->head_scales, count) != 0) goto fail;
    }
    if (mc_prepare_head(m) != 0) goto fail;

    t = gguf_find_tensor(&m->gguf, "output_norm.weight");
    if (t == NULL) {
        fprintf(stderr, "minicpm: missing output_norm.weight\n");
        goto fail;
    }
    m->out_norm = (const float *)gguf_get_tensor_ptr(&m->gguf, t);

    {
        const gguf_tensor_t *q0 = gguf_find_tensor(&m->gguf, "blk.0.attn_q.weight");
        const gguf_tensor_t *k0 = gguf_find_tensor(&m->gguf, "blk.0.attn_k.weight");
        const gguf_tensor_t *v0 = gguf_find_tensor(&m->gguf, "blk.0.attn_v.weight");
        if (q0 == NULL || k0 == NULL || v0 == NULL) {
            fprintf(stderr, "minicpm: missing blk.0 attention tensors\n");
            goto fail;
        }
        m->q_dim = (int)q0->dims[1];
        m->kv_dim = (int)k0->dims[1];
        if (m->n_head == 0 || m->q_dim % m->n_head) {
            fprintf(stderr, "minicpm: bad head_count\n");
            goto fail;
        }
        m->head_dim = m->q_dim / m->n_head;
        if (m->n_head_kv == 0) m->n_head_kv = m->n_head;
        if (m->kv_dim != m->n_head_kv * m->head_dim) {
            fprintf(stderr, "minicpm: kv head layout %d != %d*%d\n",
                    m->kv_dim, m->n_head_kv, m->head_dim);
            goto fail;
        }
        /* The TQ2_0 kernels need the *input* widths on a 256 block boundary
         * (embedding, q_dim, ff); kv_dim is only an output width, so a 2x64
         * kv head (the 0.5B) is fine. */
        if (m->n_embd % 256 || m->n_ff % 256 || m->q_dim % 256 ||
            m->n_embd != (int)v0->dims[0]) {
            fprintf(stderr, "minicpm: unsupported dimensions\n");
            goto fail;
        }
    }
    m->n_rot = mc_md_int(m, "rope.dimension_count", m->head_dim);
    if (m->n_rot % 2 || m->n_rot > m->head_dim) {
        fprintf(stderr, "minicpm: bad rope.dimension_count %d\n", m->n_rot);
        goto fail;
    }
    {
        const gguf_metadata_t *s = mc_md(m, "rope.scaling.type");
        const gguf_tensor_t *ts = gguf_find_tensor(&m->gguf, "rope_factors_short.weight");
        const gguf_tensor_t *tl = gguf_find_tensor(&m->gguf, "rope_factors_long.weight");
        size_t needed = (size_t)m->n_rot / 2u;
        if (s != NULL && s->type == GGUF_TYPE_STRING && s->string_value != NULL &&
            strcmp(s->string_value, "longrope") == 0 && ts != NULL && tl != NULL &&
            ts->type == MC_TYPE_F32 && tl->type == MC_TYPE_F32 && ts->n_dims == 1 &&
            tl->n_dims == 1 && ts->dims[0] >= needed && tl->dims[0] >= needed) {
            m->rope_factors_short = (const float *)gguf_get_tensor_ptr(&m->gguf, ts);
            m->rope_factors_long = (const float *)gguf_get_tensor_ptr(&m->gguf, tl);
            m->rope_factor_count = (int)needed;
        } else if (s != NULL && s->type == GGUF_TYPE_STRING && s->string_value != NULL &&
                   strcmp(s->string_value, "longrope") == 0) {
            /* The published BitCPM-CANN / MiniCPM4.1 1B and 3B GGUFs declare
             * longrope but ship without the factor tensors, so decoding would
             * silently use plain RoPE and lose accuracy.  Say so once. */
            fprintf(stderr, "minicpm: %s: longrope factors missing from the file; "
                            "plain RoPE in use (repair with tools/gguf_rope_fix.c)\n",
                    path);
        }
    }

    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_vocab <= 0 || m->n_ff <= 0) {
        fprintf(stderr, "minicpm: incomplete model metadata\n");
        goto fail;
    }
    if (m->emb_type != MC_TYPE_Q4_K && m->emb_type != MC_TYPE_F32 &&
        m->emb_type != MC_TYPE_Q6_K && m->emb_type != MC_TYPE_F16) {
        fprintf(stderr, "minicpm: token_embd is %s, unsupported\n",
                gguf_tensor_type_name(m->emb_type));
        goto fail;
    }

    /* The verify's CPU and GPU shares are tuned against each other, so the
     * device is set up here, before any matrix is staged.  MiniCPM's batched
     * TQ2_0 kernel runs at roughly the rate of the Adreno DP4A matmat, and
     * the two engines split the rows rather than one replacing the other. */
#if defined(BITNET_ENABLE_VULKAN)
    {
        /* Rows of each verify matmul the GPU owns.  The two engines meet
         * where their shares cost the same wall time; the CPU pool's share
         * loses a little to the memory traffic the GPU adds, so the meeting
         * point sits above the 47% the raw rates would suggest (measured on
         * the Adreno 740: 25/40/50/65/75% all cost more than 55%). */
        const char *gp = getenv("BQ_MM_TQ2_GPU");
        m->gpu_pct = gp != NULL ? atoi(gp) : 55;
        if (m->gpu_pct < 0) m->gpu_pct = 0;
        if (m->gpu_pct > 95) m->gpu_pct = 95;
        if (m->gpu_pct > 0) {
            m->vk = bq_vk_create();
            if (m->vk != NULL && !bq_vk_ready(m->vk)) {
                bq_vk_destroy(m->vk);
                m->vk = NULL;
            }
            if (m->vk != NULL && !bq_vk_tq2_matmat_available(m->vk)) {
                bq_vk_destroy(m->vk);
                m->vk = NULL;
            }
        }
        if (getenv("BQ_VK_DEBUG") != NULL)
            fprintf(stderr, "minicpm: vulkan %s, verify_gpu=%d%%\n",
                    m->vk != NULL ? "ready" : "unavailable", m->gpu_pct);
    }
#endif

    m->layer = (mc_layer_t *)calloc((size_t)m->n_layer, sizeof(*m->layer));
    if (m->layer == NULL) goto fail;
    for (i = 0; i < m->n_layer; ++i) {
        char name[64];
        mc_layer_t *L = &m->layer[i];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        t = gguf_find_tensor(&m->gguf, name);
        if (t == NULL) goto fail_layer;
        L->attn_norm = (const float *)gguf_get_tensor_ptr(&m->gguf, t);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        t = gguf_find_tensor(&m->gguf, name);
        if (t == NULL) goto fail_layer;
        L->ffn_norm = (const float *)gguf_get_tensor_ptr(&m->gguf, t);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        if (mc_bind_w(m, name, m->n_embd, m->q_dim, &L->q) != 0) goto fail_layer;
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
        if (mc_bind_w(m, name, m->n_embd, m->kv_dim, &L->k) != 0) goto fail_layer;
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
        if (mc_bind_w(m, name, m->n_embd, m->kv_dim, &L->v) != 0) goto fail_layer;
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
        if (mc_bind_w(m, name, m->q_dim, m->n_embd, &L->o) != 0) goto fail_layer;
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        if (mc_bind_w(m, name, m->n_embd, m->n_ff, &L->gate) != 0) goto fail_layer;
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        if (mc_bind_w(m, name, m->n_embd, m->n_ff, &L->up) != 0) goto fail_layer;
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        if (mc_bind_w(m, name, m->n_ff, m->n_embd, &L->down) != 0) goto fail_layer;
        continue;
fail_layer:
        fprintf(stderr, "minicpm: layer %d is incomplete\n", i);
        goto fail;
    }

    if (bitnet_tokenizer_load(&m->tok, path) != 0 || m->tok == NULL) {
        fprintf(stderr, "minicpm: tokenizer load failed\n");
        goto fail;
    }
    return m;

fail:
    minicpm_free_model(m);
    return NULL;
}

void minicpm_free_model(minicpm_model_t *m) {
    int i;
    if (m == NULL) return;
    if (m->layer != NULL) {
        for (i = 0; i < m->n_layer; ++i) {
            mc_unbind_w(m, &m->layer[i].q);
            mc_unbind_w(m, &m->layer[i].k);
            mc_unbind_w(m, &m->layer[i].v);
            mc_unbind_w(m, &m->layer[i].o);
            mc_unbind_w(m, &m->layer[i].gate);
            mc_unbind_w(m, &m->layer[i].up);
            mc_unbind_w(m, &m->layer[i].down);
        }
        free(m->layer);
    }
    free(m->head_scales);
    free(m->head_q8);
    free(m->head_q8_d);
    free(m->head_tq2);
    free(m->head_tq2_scales);
#if defined(BITNET_ENABLE_VULKAN)
    if (m->vk != NULL) bq_vk_destroy(m->vk);
#endif
    if (m->tok != NULL) bitnet_tokenizer_free(m->tok);
    gguf_close(&m->gguf);
    free(m);
}

int minicpm_embedding_length(const minicpm_model_t *m) { return m != NULL ? m->n_embd : 0; }
int minicpm_vocab_size(const minicpm_model_t *m) { return m != NULL ? m->n_vocab : 0; }
int minicpm_layer_count(const minicpm_model_t *m) { return m != NULL ? m->n_layer : 0; }

int minicpm_tokenize(minicpm_model_t *m, const char *text, int *tokens, int max_tokens) {
    int n = 0;
    if (m == NULL || m->tok == NULL || tokens == NULL || max_tokens <= 0) return -1;
    if (bitnet_tokenizer_add_bos(m->tok)) {
        int bos = bitnet_tokenizer_bos_id(m->tok);
        if (bos >= 0) {
            tokens[0] = bos;
            n = 1;
        }
    }
    {
        int enc = bitnet_tokenizer_encode(m->tok, text, tokens + n, max_tokens - n);
        if (enc < 0) return enc;
        n += enc;
    }
    return n;
}

int minicpm_decode_token(minicpm_model_t *m, int token, char *out, int out_size) {
    if (m == NULL || m->tok == NULL) return -1;
    return bitnet_tokenizer_decode(m->tok, token, out, out_size);
}

int minicpm_eos_token(const minicpm_model_t *m) {
    return m != NULL && m->tok != NULL ? bitnet_tokenizer_eos_id(m->tok) : -1;
}

int minicpm_bos_token(const minicpm_model_t *m) {
    return m != NULL && m->tok != NULL ? bitnet_tokenizer_bos_id(m->tok) : -1;
}

int minicpm_token_is_eog(minicpm_model_t *m, int token) {
    int eos = minicpm_eos_token(m);
    int pad = m != NULL && m->tok != NULL ? bitnet_tokenizer_pad_id(m->tok) : -1;
    return (eos >= 0 && token == eos) || (pad >= 0 && token == pad);
}

/* ------------------------------------------------------------------ */
/* context                                                            */
/* ------------------------------------------------------------------ */

/* One quantized activation set: B rows of `dim` int8 codes, each with a
 * per-row scale and its per-256 block sums, the layout the batched kernel
 * expects. */
typedef struct {
    int8_t *q;
    float *s;
    int32_t *bs;
    int dim;
    int bpr;
} mc_act_t;

struct minicpm_context {
    minicpm_model_t *m;
    int n_ctx;
    int n_pos;

    float *kcache;   /* [layer][kv_head][pos][head_dim] */
    float *vcache;

    float *hidden;   /* [MC_MAX_B][n_embd] */
    float *xn;       /* [MC_MAX_B][n_embd] */
    float *q;        /* [MC_MAX_B][q_dim]  */
    float *att;      /* [MC_MAX_B][q_dim]  */
    float *kk;       /* [MC_MAX_B][kv_dim] */
    float *vv;       /* [MC_MAX_B][kv_dim] */
    float *gate;     /* [MC_MAX_B][n_ff]   */
    float *up;       /* [MC_MAX_B][n_ff]   */
    float *down;     /* [MC_MAX_B][n_embd] */

    mc_act_t act_e;  /* n_embd wide */
    mc_act_t act_f;  /* n_ff wide */
    mc_act_t act_a;  /* q_dim wide (attention output) */

    float *scores;   /* [n_head][n_ctx] */
    float *rope_cos; /* [n_ctx][n_rot/2] */
    float *rope_sin;
    float *logits;   /* [n_vocab] */
    float *rowlogits;
};

static size_t mc_kv_off(const minicpm_context_t *c, int l, int kvh, int pos) {
    const minicpm_model_t *m = c->m;
    return (((size_t)l * (size_t)m->n_head_kv + (size_t)kvh) * (size_t)c->n_ctx +
            (size_t)pos) * (size_t)m->head_dim;
}

static int mc_act_init(mc_act_t *a, int dim) {
    a->dim = dim;
    a->bpr = dim / 256;
    a->q = (int8_t *)malloc((size_t)MC_MAX_B * (size_t)dim);
    a->s = (float *)malloc((size_t)MC_MAX_B * sizeof(float));
    a->bs = (int32_t *)malloc((size_t)MC_MAX_B * (size_t)a->bpr * sizeof(int32_t));
    return (a->q != NULL && a->s != NULL && a->bs != NULL) ? 0 : -1;
}

static void mc_act_free(mc_act_t *a) {
    free(a->q);
    free(a->s);
    free(a->bs);
    a->q = NULL;
    a->s = NULL;
    a->bs = NULL;
}

static int mc_quant(mc_act_t *a, const float *x, int b) {
    return bitnet_tq2_0_quantize_vec_i8_impl(x, a->dim, a->q + (size_t)b * (size_t)a->dim,
                                             &a->s[b], a->bs + (size_t)b * (size_t)a->bpr);
}

/* Rows of this matrix the GPU should take, or 0 to keep it all on the CPU. */
static int mc_gpu_rows(const minicpm_model_t *m, const mc_w_t *w, int B) {
#if defined(BITNET_ENABLE_VULKAN)
    if (m->vk == NULL || w->vk_buf == NULL) return 0;
    /* Only a block verify has B rows to amortise a decode over; a plain
     * decode is a matvec, and B > 8 is a prefill chunk the shader's row
     * window does not cover. */
    if (B < 2 || B > 8) return 0;
    if (w->vk_rows < MC_GPU_MIN_ROWS || w->vk_rows <= 0) return 0;
    return w->vk_rows;
#else
    (void)m; (void)w; (void)B;
    return 0;
#endif
}

/* One projection of a group: which matrix, where its [B][n_out] result goes,
 * and how many rows the GPU took. */
typedef struct {
    const mc_w_t *w;
    float *dst;
    int grow;
} mc_pj_t;

/* Y[b][row] = W[row] . x_b for every projection in `pj`, which all read the
 * same activation rows.  The GPU's share of all of them goes into one command
 * buffer and one submission, so a layer pays four round trips (q|k|v, o,
 * gate|up, down) instead of seven; the CPU pool then runs the head rows of
 * each matrix while the GPU works on the tails, and the results are joined by
 * a single wait before the caller reads them. */
static int mc_proj_multi(minicpm_context_t *c, mc_pj_t *pj, int n,
                         const mc_act_t *a, int B) {
    const minicpm_model_t *m = c->m;
    int i, sum = 0, submitted = 0;

    if (pj == NULL || n <= 0 || n > 8 || B < 1 || B > MC_MAX_B) return -1;
    for (i = 0; i < n; ++i) {
        if (pj[i].w == NULL || pj[i].dst == NULL || a->dim != pj[i].w->n_in) return -1;
        pj[i].grow = mc_gpu_rows(m, pj[i].w, B);
        sum += pj[i].grow;
    }
#if defined(BITNET_ENABLE_VULKAN)
    if (sum > 0) {
        int begun = bq_vk_tq2_matmat_begin(c->m->vk, a->q, a->s, a->dim, B, sum) == 0;
        int ok = begun;
        for (i = 0; ok && i < n; ++i) {
            const mc_w_t *w = pj[i].w;
            if (pj[i].grow <= 0) continue;
            /* The staged buffer starts at row n_out - grow, so it is windowed
             * from row 0 while the results belong at the matrix's real rows. */
            if (bq_vk_tq2_matmat_add(c->m->vk, (const bq_vk_buf_t *)w->vk_buf, 0,
                                     w->n_out - pj[i].grow, w->n_out,
                                     w->n_out, pj[i].dst) != 0)
                ok = 0;
        }
        if (ok && bq_vk_tq2_matmat_end(c->m->vk) == 0) {
            submitted = 1;
        } else {
            /* Anything already recorded is still submitted; wait it out before
             * the CPU rewrites those rows, and let the CPU take everything. */
            if (begun) bq_vk_tq2_matmat_end(c->m->vk);
            if (begun) bq_vk_wait(c->m->vk);
            for (i = 0; i < n; ++i) pj[i].grow = 0;
        }
    }
#endif
    for (i = 0; i < n; ++i) {
        const mc_w_t *w = pj[i].w;
        const int cpu_rows = w->n_out - pj[i].grow;
        if (cpu_rows <= 0) continue;
        if (bitnet_tq2_0_matmul_batched_i8_rows(w->data, w->scales, w->n_out,
                                                0, cpu_rows, w->n_in,
                                                a->q, a->s, a->bs, B,
                                                pj[i].dst) != 0) {
#if defined(BITNET_ENABLE_VULKAN)
            if (submitted) bq_vk_wait(c->m->vk);
#endif
            return -1;
        }
    }
#if defined(BITNET_ENABLE_VULKAN)
    if (submitted && bq_vk_wait(c->m->vk) != 0) return -1;
#else
    (void)submitted;
    (void)sum;
#endif
    return 0;
}

/* Y[b][row] = W[row] . x_b, one pass over W for all B rows. */
static int mc_proj(minicpm_context_t *c, const mc_w_t *w, const mc_act_t *a, int B, float *dst) {
    mc_pj_t pj;
    pj.w = w;
    pj.dst = dst;
    pj.grow = 0;
    return mc_proj_multi(c, &pj, 1, a, B);
}

static int mc_build_rope(minicpm_context_t *c) {
    const minicpm_model_t *m = c->m;
    const int half = m->n_rot / 2;
    const float *factors = c->n_ctx > m->n_ctx_train ? m->rope_factors_long
                                                     : m->rope_factors_short;
    const float theta_scale = powf(m->rope_base, -2.0f / (float)m->n_rot);
    int pos;

    c->rope_cos = (float *)calloc((size_t)c->n_ctx * (size_t)half, sizeof(float));
    c->rope_sin = (float *)calloc((size_t)c->n_ctx * (size_t)half, sizeof(float));
    if (c->rope_cos == NULL || c->rope_sin == NULL) return -1;
    for (pos = 0; pos < c->n_ctx; ++pos) {
        float angle = (float)pos;
        int j;
        for (j = 0; j < half; ++j) {
            float theta = angle;
            if (factors != NULL && j < m->rope_factor_count && factors[j] > 0.0f)
                theta /= factors[j];
            c->rope_cos[(size_t)pos * (size_t)half + (size_t)j] = cosf(theta);
            c->rope_sin[(size_t)pos * (size_t)half + (size_t)j] = sinf(theta);
            angle *= theta_scale;
        }
    }
    return 0;
}

minicpm_context_t *minicpm_create_context(minicpm_model_t *m, int max_tokens) {
    minicpm_context_t *c;
    const int B = MC_MAX_B;
    size_t kv_floats;

    if (m == NULL || max_tokens <= 0) return NULL;
    c = (minicpm_context_t *)calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    c->m = m;
    c->n_ctx = max_tokens;

    kv_floats = (size_t)m->n_layer * (size_t)m->n_head_kv * (size_t)max_tokens *
                (size_t)m->head_dim;
    c->kcache = (float *)calloc(kv_floats, sizeof(float));
    c->vcache = (float *)calloc(kv_floats, sizeof(float));
    c->hidden = (float *)calloc((size_t)B * (size_t)m->n_embd, sizeof(float));
    c->xn     = (float *)calloc((size_t)B * (size_t)m->n_embd, sizeof(float));
    c->q      = (float *)calloc((size_t)B * (size_t)m->q_dim, sizeof(float));
    c->att    = (float *)calloc((size_t)B * (size_t)m->q_dim, sizeof(float));
    c->kk     = (float *)calloc((size_t)B * (size_t)m->kv_dim, sizeof(float));
    c->vv     = (float *)calloc((size_t)B * (size_t)m->kv_dim, sizeof(float));
    c->gate   = (float *)calloc((size_t)B * (size_t)m->n_ff, sizeof(float));
    c->up     = (float *)calloc((size_t)B * (size_t)m->n_ff, sizeof(float));
    c->down   = (float *)calloc((size_t)B * (size_t)m->n_embd, sizeof(float));
    c->scores = (float *)calloc((size_t)m->n_head * (size_t)max_tokens, sizeof(float));
    c->logits = (float *)calloc((size_t)m->n_vocab, sizeof(float));
    c->rowlogits = (float *)calloc((size_t)m->n_vocab, sizeof(float));

    if (c->kcache == NULL || c->vcache == NULL || c->hidden == NULL || c->xn == NULL ||
        c->q == NULL || c->att == NULL || c->kk == NULL || c->vv == NULL ||
        c->gate == NULL || c->up == NULL || c->down == NULL || c->scores == NULL ||
        c->logits == NULL || c->rowlogits == NULL ||
        mc_act_init(&c->act_e, m->n_embd) != 0 ||
        mc_act_init(&c->act_f, m->n_ff) != 0 ||
        mc_act_init(&c->act_a, m->q_dim) != 0 ||
        mc_build_rope(c) != 0) {
        fprintf(stderr, "minicpm: context allocation failed\n");
        minicpm_free_context(c);
        return NULL;
    }
    return c;
}

void minicpm_free_context(minicpm_context_t *c) {
    if (c == NULL) return;
    free(c->kcache);
    free(c->vcache);
    free(c->hidden);
    free(c->xn);
    free(c->q);
    free(c->att);
    free(c->kk);
    free(c->vv);
    free(c->gate);
    free(c->up);
    free(c->down);
    free(c->scores);
    free(c->rope_cos);
    free(c->rope_sin);
    free(c->logits);
    free(c->rowlogits);
    mc_act_free(&c->act_e);
    mc_act_free(&c->act_f);
    mc_act_free(&c->act_a);
    free(c);
}

int minicpm_reset_context(minicpm_context_t *c) {
    if (c == NULL) return -1;
    c->n_pos = 0;
    return 0;
}

int minicpm_context_pos(const minicpm_context_t *c) { return c != NULL ? c->n_pos : -1; }
int minicpm_context_capacity(const minicpm_context_t *c) { return c != NULL ? c->n_ctx : 0; }

int minicpm_context_limit(const minicpm_model_t *m) {
    return m != NULL ? m->n_ctx_train : 0;
}

/* ------------------------------------------------------------------ */
/* graph                                                              */
/* ------------------------------------------------------------------ */

/* fp16 -> fp32 for one embedding row.  The tied 0.5B head reads the same
 * matrix, so this stays a plain convert-and-store rather than a dequantized
 * copy: 150 MB of fp16 streams better than 300 MB of fp32. */
static void mc_f16_row(const uint8_t *base, int row, int n, float *out) {
    const uint16_t *src = (const uint16_t *)base + (size_t)row * (size_t)n;
    int i = 0;
#if defined(__ARM_NEON)
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const float16_t *)(src + i));
        vst1q_f32(out + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(out + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
#endif
    for (; i < n; ++i) out[i] = bitnet_fp16_to_fp32(src[i]);
}

/* Dot of a float activation against an fp16 weight row, without materialising
 * the row. */
static float mc_dot_f16(const float *a, const uint16_t *b, int n) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    float sum;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float16x8_t h0 = vld1q_f16((const float16_t *)(b + i));
        float16x8_t h1 = vld1q_f16((const float16_t *)(b + i + 8));
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vcvt_f32_f16(vget_low_f16(h0)));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vcvt_f32_f16(vget_high_f16(h0)));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vcvt_f32_f16(vget_low_f16(h1)));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vcvt_f32_f16(vget_high_f16(h1)));
    }
    sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s2), vaddq_f32(s1, s3)));
    for (; i < n; ++i) sum += a[i] * bitnet_fp16_to_fp32(b[i]);
    return sum;
#else
    float sum = 0.0f;
    int i;
    for (i = 0; i < n; ++i) sum += a[i] * bitnet_fp16_to_fp32(b[i]);
    return sum;
#endif
}

static int mc_embed_row(const minicpm_model_t *m, int token, float *out) {
    if (token < 0 || token >= m->n_vocab) return -1;
    switch (m->emb_type) {
        case MC_TYPE_Q4_K:
            return bitnet_q4k_embedding_lookup(m->emb_data, token, m->n_embd,
                                               m->n_vocab, out, (size_t)m->n_embd);
        case MC_TYPE_F32:
            memcpy(out, (const float *)m->emb_data + (size_t)token * (size_t)m->n_embd,
                   (size_t)m->n_embd * sizeof(float));
            return 0;
        case MC_TYPE_F16:
            mc_f16_row(m->emb_data, token, m->n_embd, out);
            return 0;
        case MC_TYPE_Q6_K: {
            const int bpr = m->n_embd / BITNET_Q6K_QK;
            const uint8_t *row = m->emb_data + (size_t)token * (size_t)bpr *
                                 BITNET_Q6K_BLOCK_SIZE;
            int k;
            for (k = 0; k < bpr; ++k) {
                const bitnet_q6k_block_t *blk =
                    (const bitnet_q6k_block_t *)(row + (size_t)k * BITNET_Q6K_BLOCK_SIZE);
                if (bitnet_q6k_dequantize_block(blk, out + (size_t)k * BITNET_Q6K_QK,
                                                BITNET_Q6K_QK) != 0) return -1;
            }
            return 0;
        }
        default:
            return -1;
    }
}

static float mc_dot(const float *a, const float *b, int n) {
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    int i = 0;
    float sum;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i + 0), vld1q_f32(b + i + 0));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float s = 0.0f;
    int i;
    for (i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
#endif
}

/* One query row against the cached keys/values of its kv head.  The heads of
 * a group share the kv rows, so the loop runs per kv head and keeps them
 * resident for every query head in the group. */
static void mc_attention(const minicpm_context_t *c, int l, int npos,
                         const float *qrow, float *out) {
    const minicpm_model_t *m = c->m;
    const int hd = m->head_dim;
    const int gqa = m->n_head / m->n_head_kv;
    const float inv = 1.0f / sqrtf((float)hd);
    int kvh;

    for (kvh = 0; kvh < m->n_head_kv; ++kvh) {
        const float *kb = c->kcache + mc_kv_off(c, l, kvh, 0);
        const float *vb = c->vcache + mc_kv_off(c, l, kvh, 0);
        int g;
        for (g = 0; g < gqa; ++g) {
            const int h = kvh * gqa + g;
            const float *qh = qrow + (size_t)h * (size_t)hd;
            float *hs = c->scores + (size_t)h * (size_t)c->n_ctx;
            float *o = out + (size_t)h * (size_t)hd;
            int p, d;
            for (p = 0; p < npos; ++p)
                hs[p] = mc_dot(qh, kb + (size_t)p * (size_t)hd, hd) * inv;
            bitnet_softmax_impl(hs, npos);
            memset(o, 0, (size_t)hd * sizeof(float));
            for (p = 0; p < npos; ++p) {
                const float *vp = vb + (size_t)p * (size_t)hd;
                const float e = hs[p];
                d = 0;
#if defined(__ARM_NEON)
                {
                    const float32x4_t ev = vdupq_n_f32(e);
                    for (; d + 3 < hd; d += 4)
                        vst1q_f32(o + d, vfmaq_f32(vld1q_f32(o + d), vld1q_f32(vp + d), ev));
                }
#endif
                for (; d < hd; ++d) o[d] += e * vp[d];
            }
        }
    }
}

/* Rewrite the K best rows of a coarse (ternary) logits row with the exact
 * head value.  A ternary head costs 4x less traffic than the int8 copy, and
 * the only logits a sampler can act on are the ones near the top, so paying
 * K exact dots keeps greedy picks identical to the uncompressed head while
 * the full vocabulary is scanned in 19 MB (0.5B) or 4x less of 124-249 MB
 * (1B/3B/8B). */
#define MC_HEAD_RESCORE 128
#define MC_HEAD_RESCORE_MAX 1024

/* How many coarse rows to rescore.  Each costs ~2 KB of the f16 mapping;
 * 128 measured the same decode rate and the same text as 64, and 256 cost
 * 3%, so this is the margin without the traffic.  A Q6_K row is 70 B per
 * 256 weights, 7x cheaper per candidate, so the ceiling is set higher for
 * the experiment of widening it. */
static int mc_head_rescore_k(void) {
    static int k = -1;
    if (k < 0) {
        const char *e = getenv("BQ_MC_RESCORE");
        k = MC_HEAD_RESCORE;
        if (e != NULL && e[0] != '\0') {
            int v = atoi(e);
            if (v > 0) k = v;
        }
        if (k > MC_HEAD_RESCORE_MAX) k = MC_HEAD_RESCORE_MAX;
    }
    return k;
}

/* The K highest coarse logits, ascending.  Both heads below pick the same
 * candidates, so the sampler always sees the best exact value of each. */
static int mc_head_topk(const float *logits, int n_vocab, int K,
                        int *idx, float *val) {
    int n = 0, j, i;
    for (j = 0; j < n_vocab; ++j) {
        const float v = logits[j];
        if (n < K) {
            i = n++;
            while (i > 0 && val[i - 1] > v) { val[i] = val[i - 1]; idx[i] = idx[i - 1]; --i; }
            val[i] = v;
            idx[i] = j;
        } else if (v > val[0]) {
            i = 0;
            while (i + 1 < K && val[i + 1] < v) { val[i] = val[i + 1]; idx[i] = idx[i + 1]; ++i; }
            val[i] = v;
            idx[i] = j;
        }
    }
    return n;
}

static void mc_head_rescore_f16(const minicpm_model_t *m, const float *x, float *logits) {
    const uint16_t *w = (const uint16_t *)m->head_data;
    int idx[MC_HEAD_RESCORE_MAX];
    float val[MC_HEAD_RESCORE_MAX];
    int n, i;

    n = mc_head_topk(logits, m->n_vocab, mc_head_rescore_k(), idx, val);
    for (i = 0; i < n; ++i) {
        float s = mc_dot_f16(x, w + (size_t)idx[i] * (size_t)m->n_embd, m->n_embd);
        if (m->logit_scale != 0.0f) s /= m->logit_scale;
        logits[idx[i]] = s;
    }
}

/* Same rescore against the Q6_K head the 1B/3B/8B checkpoints carry.  A
 * candidate costs 210 B per 256 weights instead of 512 B for an f16
 * equivalent, so 128 candidates are ~0.4-1.4 MB against the 124-249 MB the
 * coarse pass saves. */
static void mc_head_rescore_q6k(const minicpm_model_t *m,
                                const bitnet_q8k_block_t *q8k, float *logits) {
    const int bpr = m->n_embd / 256;
    const size_t row_bytes = (size_t)bpr * BITNET_Q6K_BLOCK_SIZE;
    const uint8_t *w = (const uint8_t *)m->head_data;
    int idx[MC_HEAD_RESCORE_MAX];
    float val[MC_HEAD_RESCORE_MAX];
    int n, i;

    n = mc_head_topk(logits, m->n_vocab, mc_head_rescore_k(), idx, val);
    for (i = 0; i < n; ++i) {
        const uint8_t *row = w + (size_t)idx[i] * row_bytes;
        float sum = 0.0f;
        int b, ok = 1;
        /* The dot covers one 256-weight block, so a row is bpr calls.  Each
         * block carries its own activation scale, exactly as the native
         * matmul does, so a rescored logit matches the head's own value. */
        for (b = 0; b < bpr; ++b) {
            float s = 0.0f;
            if (bitnet_q6k_dot_product_i8_neon(
                    (const bitnet_q6k_block_t *)(row + (size_t)b * BITNET_Q6K_BLOCK_SIZE),
                    q8k[b].qs, q8k[b].d, 256, &s) != 0) { ok = 0; break; }
            sum += s;
        }
        /* The coarse pass folded logit_scale into the activation scale;
         * the exact value has to be in the same units. */
        if (ok && m->logit_scale != 0.0f) sum /= m->logit_scale;
        if (ok) logits[idx[i]] = sum;
    }
}

/* One logits row.  Q6_K goes through the shared Q8_K matmul (the same one
 * src/bitnet.c uses for this family); a TQ2_0 head reuses the batched
 * projection kernel with a single activation row. */
static int mc_head_row(minicpm_context_t *c, const float *x, float *out, int slot) {
    const minicpm_model_t *m = c->m;
    int i;
    if (m->head_tq2 != NULL) {
        int8_t *qrow = c->act_e.q + (size_t)slot * (size_t)m->n_embd;
        int32_t *bsum = c->act_e.bs + (size_t)slot * (size_t)c->act_e.bpr;
        float scale;
        if (bitnet_tq2_0_quantize_vec_i8_impl(x, m->n_embd, qrow, &scale, bsum) != 0) return -1;
        if (m->logit_scale != 0.0f) scale /= m->logit_scale;
        if (bitnet_tq2_0_matmul_batched_i8(m->head_tq2, m->head_tq2_scales, m->n_vocab,
                                           m->n_embd, qrow, &scale, bsum, 1, out) != 0)
            return -1;
        if (m->head_type == MC_TYPE_Q6_K) {
            bitnet_q8k_block_t q8k[BITNET_Q8K_MAX_BLOCKS];
            if (bitnet_quantize_q8k_impl(x, m->n_embd, q8k) != 0) return -1;
            mc_head_rescore_q6k(m, q8k, out);
        } else {
            mc_head_rescore_f16(m, x, out);
        }
        return 0;
    }
    if (m->head_q8 != NULL) {
        int8_t *qrow = c->act_e.q + (size_t)slot * (size_t)m->n_embd;
        int32_t *bsum = c->act_e.bs + (size_t)slot * (size_t)c->act_e.bpr;
        float scale;
        if (bitnet_tq2_0_quantize_vec_i8_impl(x, m->n_embd, qrow, &scale, bsum) != 0) return -1;
        /* logit_scale folds into the activation scale: one multiply here
         * instead of n_vocab divides after the projection. */
        if (m->logit_scale != 0.0f) scale /= m->logit_scale;
        return bitnet_output_logits_blockscale(m->head_q8, m->head_q8_d,
                                               m->n_embd, m->n_vocab, qrow, scale, out);
    }
    if (m->head_type == MC_TYPE_F16) {
        const uint16_t *w = (const uint16_t *)m->head_data;
        const int stride = m->n_embd;
        for (i = 0; i < m->n_vocab; ++i)
            out[i] = mc_dot_f16(x, w + (size_t)i * (size_t)stride, stride);
    } else if (m->head_type == MC_TYPE_Q6_K) {
        bitnet_q8k_block_t q8k[BITNET_Q8K_MAX_BLOCKS];
        if (m->n_embd > BITNET_Q8K_MAX_BLOCKS * 256) return -1;
        if (bitnet_quantize_q8k_impl(x, m->n_embd, q8k) != 0) return -1;
        if (bitnet_matmul_q8k_prepared(m->head_data, (int)MC_TYPE_Q6_K, m->n_vocab,
                                       m->n_embd, q8k, out) != 0) return -1;
    } else {
        float scale;
        int32_t bsum[BITNET_Q8K_MAX_BLOCKS];
        int8_t *qrow = c->act_e.q + (size_t)slot * (size_t)m->n_embd;
        if (bitnet_tq2_0_quantize_vec_i8_impl(x, m->n_embd, qrow, &scale, bsum) != 0)
            return -1;
        if (bitnet_tq2_0_matmul_batched_i8(m->head_data, m->head_scales, m->n_vocab,
                                           m->n_embd, qrow, &scale, bsum, 1, out) != 0)
            return -1;
    }
    if (m->logit_scale != 0.0f)
        for (i = 0; i < m->n_vocab; ++i) out[i] /= m->logit_scale;
    return 0;
}

/* Score `n` tokens in chunks of at most 16.  Rows at or past `emit_from` get
 * a logits row, written `stride` floats apart starting at index 0. */
static int mc_forward(minicpm_context_t *c, const int *tokens, int n,
                      float *logits, int stride, int emit_from) {
    const minicpm_model_t *m = c->m;
    const int E = m->n_embd, F = m->n_ff, Q = m->q_dim, KV = m->kv_dim;
    const int hd = m->head_dim;
    int base;

    if (c == NULL || tokens == NULL || n <= 0 || m == NULL) return -1;
    if (c->n_pos + n > c->n_ctx) return -1;
    if (logits != NULL && stride < m->n_vocab) return -1;

    {
    const int chunk = mc_chunk_b(n);
    for (base = 0; base < n; base += chunk) {
        const int B = (n - base < chunk) ? n - base : chunk;
        const int pos0 = c->n_pos;
        int b, l, i;

        for (b = 0; b < B; ++b) {
            float *h = c->hidden + (size_t)b * (size_t)E;
            if (mc_embed_row(m, tokens[base + b], h) != 0) return -1;
            if (m->emb_scale != 1.0f)
                for (i = 0; i < E; ++i) h[i] *= m->emb_scale;
        }

        for (l = 0; l < m->n_layer; ++l) {
            const mc_layer_t *L = &m->layer[l];

            /* attention norm -> q/k/v */
            for (b = 0; b < B; ++b) {
                float *xn = c->xn + (size_t)b * (size_t)E;
                bitnet_rms_norm_inplace_eps_impl(xn, c->hidden + (size_t)b * (size_t)E,
                                                 L->attn_norm, E, m->eps);
                if (mc_quant(&c->act_e, xn, b) != 0) return -1;
            }
            {
                mc_pj_t qkv[3];
                qkv[0] = (mc_pj_t){ &L->q, c->q, 0 };
                qkv[1] = (mc_pj_t){ &L->k, c->kk, 0 };
                qkv[2] = (mc_pj_t){ &L->v, c->vv, 0 };
                if (mc_proj_multi(c, qkv, 3, &c->act_e, B) != 0) return -1;
            }

            for (b = 0; b < B; ++b) {
                const size_t off = (size_t)(pos0 + b) * (size_t)(m->n_rot / 2);
                int kvh;
                bitnet_rope_apply_impl(c->q + (size_t)b * (size_t)Q, m->n_head, hd,
                                       m->n_rot, c->rope_cos + off, c->rope_sin + off);
                bitnet_rope_apply_impl(c->kk + (size_t)b * (size_t)KV, m->n_head_kv, hd,
                                       m->n_rot, c->rope_cos + off, c->rope_sin + off);
                for (kvh = 0; kvh < m->n_head_kv; ++kvh) {
                    const size_t ko = mc_kv_off(c, l, kvh, pos0 + b);
                    memcpy(c->kcache + ko,
                           c->kk + (size_t)b * (size_t)KV + (size_t)kvh * (size_t)hd,
                           (size_t)hd * sizeof(float));
                    memcpy(c->vcache + ko,
                           c->vv + (size_t)b * (size_t)KV + (size_t)kvh * (size_t)hd,
                           (size_t)hd * sizeof(float));
                }
            }

            for (b = 0; b < B; ++b) {
                float *arow = c->att + (size_t)b * (size_t)Q;
                mc_attention(c, l, pos0 + b + 1, c->q + (size_t)b * (size_t)Q, arow);
                if (mc_quant(&c->act_a, arow, b) != 0) return -1;
            }
            if (mc_proj(c, &L->o, &c->act_a, B, c->down) != 0) return -1;
            for (b = 0; b < B; ++b) {
                float *h = c->hidden + (size_t)b * (size_t)E;
                const float *d = c->down + (size_t)b * (size_t)E;
                if (m->res_scale == 1.0f)
                    for (i = 0; i < E; ++i) h[i] += d[i];
                else
                    for (i = 0; i < E; ++i) h[i] += m->res_scale * d[i];
            }

            /* feed-forward norm -> gate/up -> silu*up -> down */
            for (b = 0; b < B; ++b) {
                float *xn = c->xn + (size_t)b * (size_t)E;
                bitnet_rms_norm_inplace_eps_impl(xn, c->hidden + (size_t)b * (size_t)E,
                                                 L->ffn_norm, E, m->eps);
                if (mc_quant(&c->act_e, xn, b) != 0) return -1;
            }
            {
                mc_pj_t gu[2];
                gu[0] = (mc_pj_t){ &L->gate, c->gate, 0 };
                gu[1] = (mc_pj_t){ &L->up, c->up, 0 };
                if (mc_proj_multi(c, gu, 2, &c->act_e, B) != 0) return -1;
            }
            for (b = 0; b < B; ++b) {
                float *g = c->gate + (size_t)b * (size_t)F;
                bitnet_silu_mul_impl(g, c->up + (size_t)b * (size_t)F, F);
                if (mc_quant(&c->act_f, g, b) != 0) return -1;
            }
            if (mc_proj(c, &L->down, &c->act_f, B, c->down) != 0) return -1;
            for (b = 0; b < B; ++b) {
                float *h = c->hidden + (size_t)b * (size_t)E;
                const float *d = c->down + (size_t)b * (size_t)E;
                if (m->res_scale == 1.0f)
                    for (i = 0; i < E; ++i) h[i] += d[i];
                else
                    for (i = 0; i < E; ++i) h[i] += m->res_scale * d[i];
            }
        }

        c->n_pos = pos0 + B;

        if (logits != NULL) {
            for (b = 0; b < B; ++b) {
                const int row = base + b;
                float *dst;
                float *tmp;
                if (row < emit_from) continue;
                bitnet_rms_norm_inplace_eps_impl(c->xn + (size_t)b * (size_t)E,
                                                 c->hidden + (size_t)b * (size_t)E,
                                                 m->out_norm, E, m->eps);
                dst = logits + (size_t)(row - emit_from) * (size_t)stride;
                tmp = (stride == m->n_vocab) ? dst : c->rowlogits;
                if (mc_head_row(c, c->xn + (size_t)b * (size_t)E, tmp, b) != 0) return -1;
                if (tmp != dst)
                    memcpy(dst, tmp, (size_t)m->n_vocab * sizeof(float));
            }
        }
    }
    }
    return 0;
}

/* The row pool is deliberately left hot here: a decode is a token-by-token
 * stream of evals and the workers must not pay a park/unpark every 7 ms.  A
 * caller that finishes a request parks them once (bonsai_park_workers), which
 * is where the idle spin would otherwise tax the next model. */
int minicpm_eval(minicpm_context_t *c, const int *tokens, int n_tokens) {
    if (c == NULL) return -1;
    return mc_forward(c, tokens, n_tokens, c->logits, c->m->n_vocab, n_tokens - 1);
}

int minicpm_eval_block(minicpm_context_t *c, const int *tokens, int n,
                       float *logits, int stride) {
    if (c == NULL) return -1;
    if (logits == NULL) return mc_forward(c, tokens, n, NULL, 0, 0);
    return mc_forward(c, tokens, n, logits, stride, 0);
}

const float *minicpm_get_logits(const minicpm_context_t *c) {
    return c != NULL ? c->logits : NULL;
}

int minicpm_sample_greedy(const minicpm_context_t *c) {
    int best = 0, i;
    if (c == NULL || c->logits == NULL || c->m->n_vocab <= 0) return -1;
    for (i = 1; i < c->m->n_vocab; ++i)
        if (c->logits[i] > c->logits[best]) best = i;
    return best;
}

/* ---- rollback ------------------------------------------------------- */

/* Attention only reads positions below the counter, so a snapshot is the
 * counter itself: the rejected tail's KV rows are overwritten by the next
 * pass.  The driver still asks for a buffer, hence the fixed 16 bytes. */
/* Everything the minicpm graph carries across a rejected verify is the KV
 * cache, and the KV cache is addressed by position: a row the verify wrote at
 * p >= pos is never read again once the position counter is wound back.  So
 * the snapshot is empty, exactly like the qwen35 one (which also leaves `pos`
 * to minicpm_rewind / bonsai_rewind).  Putting `n_pos` in here as well would
 * make the driver's load-then-rewind restore the pre-verify position and then
 * subtract again, which lands the next round one token early per rejection. */
size_t minicpm_state_bytes(const minicpm_context_t *c) {
    (void)c;
    return 0;
}

void minicpm_state_save(const minicpm_context_t *c, float *dst) {
    (void)c;
    (void)dst;
}

void minicpm_state_load(minicpm_context_t *c, const float *src) {
    (void)c;
    (void)src;
}

void minicpm_rewind(minicpm_context_t *c, int n) {
    if (c == NULL || n <= 0) return;
    if (n > c->n_pos) n = c->n_pos;
    c->n_pos -= n;
}
