/* metis_file.c -- Metis memory model: file format + runtime math.
 *
 * File layout (little-endian), magic "BNMEM1\0\0" or "BNMEM2\0\0":
 *   u32 version (=1 legacy unbound, =2 backbone-bound)
 *   u32 n_layers
 *   u32 layer_ids[32]        (unused slots = 0xFFFFFFFF)
 *   u32 d_model, kv_dim, q_dim, head_dim
 *   f32 gamma, tau, rho
 *   u32 k_min
 *   f32 gdu_ab, gdu_bb, beta_scale
 *   v2 only: u8 backbone_sha256[32]
 *   tensor payloads in fixed order (all f32, row-major, layer-major):
 *     wk       [n_layers x kv_dim*d_model]
 *     wv       [n_layers x kv_dim*d_model]
 *     w_agg    [n_layers x d_model]
 *     gdu_aw   [n_layers x d_model]
 *     gdu_bw   [n_layers x d_model]
 *     mem_norm [n_layers x q_dim]
 *   manifest: u32 tensor_count (6), then per tensor
 *     { u32 name_len; name; u32 rows; u32 cols; u64 offset; u32 crc32 }
 *
 * v4 (magic "BNMEM4\0\0", version 4) -- reference-trained direct-use models:
 *   same header as v3, plus after beta_scale:
 *   u32 denom_plus_one       (read = (q~M)/(q~S + 1) when 1)
 *   u32 gdu_ab_v[n_layers]   bit-cast f32 per-layer TRAINED alpha bias
 *   u32 gdu_bb_v[n_layers]   bit-cast f32 per-layer TRAINED beta bias
 *   (then n_layers > 0 requires) tensor payloads add:
 *     query_proj [n_layers x q_dim*d_model]  (trained low/full-rank query)
 *     query_norm [n_layers x head_dim]       (per-head RMSNorm weight)
 *   manifest tensor_count 8 (v3's six + query_proj + query_norm).
 *   Semantics switch: commit = single-norm on RAW residual rows with
 *   alpha-scaled S; read = trained query path with the +1 flag; fusion is
 *   ALWAYS on (reference has no bypass: read=0/(0+1)=0 pre-commit).
 *
 * Fail-closed like the v2 loader: header validation, recomputed offsets,
 * dims and crc diffed against the on-disk manifest, EOF immediately after.
 */
#include "metis_file.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define METIS_FILE_VERSION_3 3u
#define METIS_FILE_VERSION 1u
#define METIS_BOUND_FILE_VERSION 2u
#define METIS_MAX_LAYERS 32
#define METIS_HDR_MAGIC_LEN 8u
#define METIS_UNUSED_LAYER 0xFFFFFFFFu
#define METIS_QUERY_ADD_BACKBONE 0x80000000u
#define METIS_QUERY_RANK_MASK 0x0000FFFFu
#define METIS_KV_RANK_MASK 0x7FFF0000u
#define METIS_KV_RANK_SHIFT 16u

static const uint8_t METIS_MAGIC[METIS_HDR_MAGIC_LEN] =
    { 'B', 'N', 'M', 'E', 'M', '1', 0, 0 };
static const uint8_t METIS_BOUND_MAGIC[METIS_HDR_MAGIC_LEN] =
    { 'B', 'N', 'M', 'E', 'M', '2', 0, 0 };
/* Accept BNMEM5 as well (same v1 format, written by the converter before
 * the rename from v5 to v1; the writer now emits BNMEM1). */
static const uint8_t METIS_MAGIC_5[METIS_HDR_MAGIC_LEN] =
    { 'B', 'N', 'M', 'E', 'M', '5', 0, 0 };

/* ------------------------------------------------------------------ */
/* little-endian primitives (shared shape with metis_model.c)          */
/* ------------------------------------------------------------------ */

static int metis_w_bytes(FILE *f, const void *b, size_t n) {
    return (n == 0 || fwrite(b, 1, n, f) == n) ? 0 : -1;
}

static int metis_r_bytes(FILE *f, void *b, size_t n) {
    return (n == 0 || fread(b, 1, n, f) == n) ? 0 : -1;
}

static int metis_w_u32(FILE *f, uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; ++i) b[i] = (uint8_t)(v >> (8 * i));
    return metis_w_bytes(f, b, 4);
}

static int metis_r_u32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (metis_r_bytes(f, b, 4) != 0) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static int metis_w_u64(FILE *f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t)(v >> (8 * i));
    return metis_w_bytes(f, b, 8);
}

static int metis_r_u64(FILE *f, uint64_t *v) {
    uint8_t b[8];
    if (metis_r_bytes(f, b, 8) != 0) return -1;
    uint64_t acc = 0;
    for (int i = 7; i >= 0; --i) acc = (acc << 8) | b[i];
    *v = acc;
    return 0;
}

static int metis_w_f32(FILE *f, float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return metis_w_u32(f, u);
}

static int metis_r_f32(FILE *f, float *v) {
    uint32_t u;
    if (metis_r_u32(f, &u) != 0) return -1;
    memcpy(v, &u, 4);
    return 0;
}

static uint32_t metis_crc32_buf(const void *buf, size_t len) {
    static uint32_t table[256];
    static int have_table = 0;
    if (!have_table) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        have_table = 1;
    }
    const uint8_t *p = (const uint8_t *)buf;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* ------------------------------------------------------------------ */
/* tensor table                                                        */
/* ------------------------------------------------------------------ */

enum {
    T_WK = 0,   /* f32[n_layers x (kv_dim*d_model)] */
    T_WV,       /* f32[n_layers x (kv_dim*d_model)] */
    T_W_AGG,    /* f32[n_layers x d_model] */
    T_GDU_AW,   /* f32[n_layers x d_model] */
    T_GDU_BW,   /* f32[n_layers x d_model] */
    T_MEM_NORM, /* f32[n_layers x q_dim] */
    T_QUERY_PROJ, /* v4 only: f32[n_layers x (q_dim*d_model)] */
    T_QUERY_NORM, /* v4/v5: f32[n_layers x head_dim] */
    T_QUERY_A,    /* v5 only: f32[n_layers x (q_dim*query_rank)] */
    T_QUERY_B,    /* v5 only: f32[n_layers x (query_rank*d_model)] */
    T_WK_A,       /* low-rank K: f32[n_layers x (kv_dim*kv_rank)] */
    T_WK_B,       /* low-rank K: f32[n_layers x (kv_rank*d_model)] */
    T_WV_A,       /* low-rank V: f32[n_layers x (kv_dim*kv_rank)] */
    T_WV_B,       /* low-rank V: f32[n_layers x (kv_rank*d_model)] */
    T_TENSOR_COUNT,
};

typedef struct {
    const char *name;
    uint32_t rows, cols;
    uint64_t nbytes;
} metis_tinfo_t;

static void metis_tinfo_fill(metis_tinfo_t *t, const char *name, uint64_t rows,
                          uint64_t cols) {
    t->name = name;
    t->rows = (uint32_t)rows;
    t->cols = (uint32_t)cols;
    t->nbytes = rows * cols * 4u;
}

static void metis_build_tinfo(const metis_params_t *p,
                           metis_tinfo_t ti[T_TENSOR_COUNT]) {
    const uint64_t NL = (uint64_t)p->n_layers;
    metis_tinfo_fill(&ti[T_WK], "wk", NL, (uint64_t)p->kv_dim * p->d_model);
    metis_tinfo_fill(&ti[T_WV], "wv", NL, (uint64_t)p->kv_dim * p->d_model);
    metis_tinfo_fill(&ti[T_W_AGG], "w_agg", NL, p->d_model);
    metis_tinfo_fill(&ti[T_GDU_AW], "gdu_aw", NL, p->d_model);
    metis_tinfo_fill(&ti[T_GDU_BW], "gdu_bw", NL, p->d_model);
    metis_tinfo_fill(&ti[T_MEM_NORM], "mem_norm", NL, p->q_dim);
    metis_tinfo_fill(&ti[T_QUERY_PROJ], "query_proj", NL,
                  (uint64_t)p->q_dim * p->d_model);
    metis_tinfo_fill(&ti[T_QUERY_NORM], "query_norm", NL, p->head_dim);
    metis_tinfo_fill(&ti[T_QUERY_A], "query_a", NL,
                  (uint64_t)p->q_dim * (uint64_t)(p->query_rank > 0 ?
                                                  p->query_rank : 1));
    metis_tinfo_fill(&ti[T_QUERY_B], "query_b", NL,
                  (uint64_t)(p->query_rank > 0 ? p->query_rank : 1) *
                  p->d_model);
    metis_tinfo_fill(&ti[T_WK_A], "wk_a", NL,
                  (uint64_t)p->kv_dim *
                  (uint64_t)(p->kv_rank > 0 ? p->kv_rank : 1));
    metis_tinfo_fill(&ti[T_WK_B], "wk_b", NL,
                  (uint64_t)(p->kv_rank > 0 ? p->kv_rank : 1) *
                  p->d_model);
    metis_tinfo_fill(&ti[T_WV_A], "wv_a", NL,
                  (uint64_t)p->kv_dim *
                  (uint64_t)(p->kv_rank > 0 ? p->kv_rank : 1));
    metis_tinfo_fill(&ti[T_WV_B], "wv_b", NL,
                  (uint64_t)(p->kv_rank > 0 ? p->kv_rank : 1) *
                  p->d_model);
}

static const void *metis_tensor_ptr(const metis_params_t *p, int idx) {
    switch (idx) {
    case T_WK:       return p->wk;
    case T_WV:       return p->wv;
    case T_W_AGG:    return p->w_agg;
    case T_GDU_AW:   return p->gdu_aw;
    case T_GDU_BW:   return p->gdu_bw;
    case T_MEM_NORM: return p->mem_norm;
    case T_QUERY_PROJ: return p->query_proj;
    case T_QUERY_A:    return p->query_a;
    case T_QUERY_B:    return p->query_b;
    case T_QUERY_NORM: return p->query_norm;
    case T_WK_A:       return p->wk_a;
    case T_WK_B:       return p->wk_b;
    case T_WV_A:       return p->wv_a;
    case T_WV_B:       return p->wv_b;
    default:          return NULL;
    }
}

static int metis_build_tmap(const metis_params_t *p,
                            int tmap[T_TENSOR_COUNT]) {
    int n = 0;
    if (p->kv_rank > 0) {
        tmap[n++] = T_WK_A;
        tmap[n++] = T_WK_B;
        tmap[n++] = T_WV_A;
        tmap[n++] = T_WV_B;
    } else {
        tmap[n++] = T_WK;
        tmap[n++] = T_WV;
    }
    tmap[n++] = T_W_AGG;
    tmap[n++] = T_GDU_AW;
    tmap[n++] = T_GDU_BW;
    tmap[n++] = T_MEM_NORM;
    tmap[n++] = T_QUERY_NORM;
    if (p->query_rank > 0) {
        tmap[n++] = T_QUERY_A;
        tmap[n++] = T_QUERY_B;
    } else {
        tmap[n++] = T_QUERY_PROJ;
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* validation                                                          */
/* ------------------------------------------------------------------ */

static int metis_validate(int n_layers, const int *layer_ids, int d_model,
                       int kv_dim, int q_dim, int head_dim, float gamma,
                       float tau, float rho) {
    if (n_layers < 1 || n_layers > METIS_MAX_LAYERS) return -1;
    if (layer_ids == NULL) return -1;
    for (int i = 0; i < n_layers; ++i) {
        if (layer_ids[i] < 0) return -1;
        if (i > 0 && layer_ids[i] <= layer_ids[i - 1]) return -1;
    }
    if (d_model <= 0 || d_model > 8192) return -1;
    if (kv_dim <= 0 || kv_dim > 512) return -1;     /* metis_read stack buf */
    if (q_dim <= 0 || q_dim % head_dim != 0) return -1;
    if (head_dim <= 0 || head_dim > 512) return -1;
    if (q_dim % kv_dim != 0) return -1;             /* GQA groups must tile */
    if (!(rho > 0.0f && rho <= 1.0f)) return -1;
    if (!(gamma > 0.0f && gamma < 1.0f)) return -1;
    if (!(tau > 0.0f)) return -1;
    return 0;
}

static void *metis_xcalloc(uint64_t n) {
    if (n == 0 || n > (uint64_t)(SIZE_MAX / sizeof(float))) return NULL;
    return calloc((size_t)n, sizeof(float));
}

/* ------------------------------------------------------------------ */
/* alloc / free                                                        */
/* ------------------------------------------------------------------ */

int metis_model_alloc(metis_file_model_t *m, int n_layers,
                            const int *layer_ids, int d_model, int kv_dim,
                            int q_dim, int head_dim, float gamma, float tau,
                            float rho, int k_min, float beta_scale,
                            int denom_plus_one, int query_rank) {
    metis_params_t *p = &m->params;
    if (m == NULL || layer_ids == NULL) return -1;
    if (query_rank < 0 || query_rank > q_dim || query_rank > d_model)
        return -1;
    memset(m, 0, sizeof *m);
    p->n_layers = n_layers;
    for (int i = 0; i < n_layers; ++i) p->layer_ids[i] = layer_ids[i];
    p->d_model = d_model; p->kv_dim = kv_dim; p->q_dim = q_dim;
    p->head_dim = head_dim;
    p->gamma = gamma; p->tau = tau; p->rho = rho; p->k_min = k_min;
    p->beta_scale = beta_scale;
    if (metis_validate(n_layers, layer_ids, d_model, kv_dim, q_dim, head_dim,
                       gamma, tau, rho) != 0)
        return -1;
    const uint64_t NL = (uint64_t)n_layers;
    p->wk = metis_xcalloc(NL * (uint64_t)kv_dim * d_model);
    p->wv = metis_xcalloc(NL * (uint64_t)kv_dim * d_model);
    p->w_agg = metis_xcalloc(NL * d_model);
    p->gdu_aw = metis_xcalloc(NL * d_model);
    p->gdu_bw = metis_xcalloc(NL * d_model);
    p->mem_norm = metis_xcalloc(NL * q_dim);
    if (!p->wk || !p->wv || !p->w_agg || !p->gdu_aw || !p->gdu_bw ||
        !p->mem_norm) {
        metis_model_free_arrays(m);
        return -1;
    }
    p->denom_plus_one = denom_plus_one;
    p->gdu_ab_v = metis_xcalloc(NL);
    p->gdu_bb_v = metis_xcalloc(NL);
    p->query_norm = metis_xcalloc(NL * head_dim);
    p->query_rank = query_rank;
    p->kv_rank = 0;
    if (query_rank > 0) {
        p->query_a = metis_xcalloc(
            NL * (uint64_t)q_dim * (uint64_t)query_rank);
        p->query_b = metis_xcalloc(
            NL * (uint64_t)query_rank * d_model);
    } else {
        p->query_proj = metis_xcalloc(
            NL * (uint64_t)q_dim * (uint64_t)d_model);
    }
    if (!p->gdu_ab_v || !p->gdu_bb_v || !p->query_norm ||
        (query_rank > 0 && (!p->query_a || !p->query_b)) ||
        (query_rank == 0 && !p->query_proj)) {
        metis_model_free_arrays(m);
        return -1;
    }
    for (uint64_t i = 0; i < NL * (uint64_t)head_dim; ++i)
        p->query_norm[i] = 1.0f;
    m->version = (int)METIS_FILE_VERSION;
    return 0;
}

void metis_model_free_arrays(metis_file_model_t *m) {
    if (m == NULL) return;
    metis_params_t *p = &m->params;
    free(p->wk); p->wk = NULL;
    free(p->wv); p->wv = NULL;
    free(p->w_agg); p->w_agg = NULL;
    free(p->gdu_aw); p->gdu_aw = NULL;
    free(p->gdu_bw); p->gdu_bw = NULL;
    free(p->mem_norm); p->mem_norm = NULL;
    free(p->gdu_ab_v); p->gdu_ab_v = NULL;
    free(p->gdu_bb_v); p->gdu_bb_v = NULL;
    free(p->query_proj); p->query_proj = NULL;
    free(p->query_norm); p->query_norm = NULL;
    free(p->query_a); p->query_a = NULL;
    free(p->query_b); p->query_b = NULL;
    free(p->wk_a); p->wk_a = NULL;
    free(p->wk_b); p->wk_b = NULL;
    free(p->wv_a); p->wv_a = NULL;
    free(p->wv_b); p->wv_b = NULL;
    memset(m, 0, sizeof *m);
}

void metis_model_free_loaded(metis_file_model_t *m) {
    if (m == NULL) return;
    metis_model_free_arrays(m);
    free(m);
}

/* ------------------------------------------------------------------ */
/* save                                                                */
/* ------------------------------------------------------------------ */

int metis_model_save(const metis_file_model_t *m, const char *path) {
    if (m == NULL || path == NULL) return -1;
    if (m->version != (int)METIS_FILE_VERSION) return -1;
    const metis_params_t *p = &m->params;
    if (metis_validate(p->n_layers, p->layer_ids, p->d_model, p->kv_dim,
                    p->q_dim, p->head_dim, p->gamma, p->tau, p->rho) != 0)
        return -1;
    if (p->gdu_ab_v == NULL || p->gdu_bb_v == NULL ||
        p->query_norm == NULL ||
        (p->kv_rank > 0 &&
         (p->wk_a == NULL || p->wk_b == NULL ||
          p->wv_a == NULL || p->wv_b == NULL)) ||
        (p->kv_rank == 0 && (p->wk == NULL || p->wv == NULL)) ||
        (p->query_rank > 0 &&
         (p->query_a == NULL || p->query_b == NULL)) ||
        (p->query_rank == 0 && p->query_proj == NULL))
        return -1;

    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;

    int tmap[T_TENSOR_COUNT];
    int n_tensors = metis_build_tmap(p, tmap);
    metis_tinfo_t ti[T_TENSOR_COUNT];
    metis_build_tinfo((metis_params_t *)p, ti);

    uint64_t off = 0;
    const uint8_t *magic =
        p->has_backbone_sha256 ? METIS_BOUND_MAGIC : METIS_MAGIC;
    uint32_t version =
        p->has_backbone_sha256 ? METIS_BOUND_FILE_VERSION :
                                 METIS_FILE_VERSION;
    if (metis_w_bytes(f, magic, METIS_HDR_MAGIC_LEN) != 0) goto io_fail;
    off += METIS_HDR_MAGIC_LEN;
    if (metis_w_u32(f, version) != 0) goto io_fail;   off += 4;
    if (metis_w_u32(f, (uint32_t)p->n_layers) != 0) goto io_fail; off += 4;
    for (int i = 0; i < METIS_MAX_LAYERS; ++i) {
        uint32_t id = (i < p->n_layers) ? (uint32_t)p->layer_ids[i]
                                        : METIS_UNUSED_LAYER;
        if (metis_w_u32(f, id) != 0) goto io_fail;
        off += 4;
    }
    if (metis_w_u32(f, (uint32_t)p->d_model) != 0) goto io_fail;    off += 4;
    if (metis_w_u32(f, (uint32_t)p->kv_dim) != 0) goto io_fail;     off += 4;
    if (metis_w_u32(f, (uint32_t)p->q_dim) != 0) goto io_fail;      off += 4;
    if (metis_w_u32(f, (uint32_t)p->head_dim) != 0) goto io_fail;   off += 4;
    if (metis_w_f32(f, p->gamma) != 0) goto io_fail;                off += 4;
    if (metis_w_f32(f, p->tau) != 0) goto io_fail;                  off += 4;
    if (metis_w_f32(f, p->rho) != 0) goto io_fail;                  off += 4;
    if (metis_w_u32(f, (uint32_t)p->k_min) != 0) goto io_fail;      off += 4;
    if (metis_w_f32(f, p->gdu_ab) != 0) goto io_fail;               off += 4;
    if (metis_w_f32(f, p->gdu_bb) != 0) goto io_fail;               off += 4;
    if (metis_w_f32(f, p->beta_scale) != 0) goto io_fail;           off += 4;
    if (metis_w_u32(f, (uint32_t)p->denom_plus_one) != 0) goto io_fail;
    off += 4;
    {
        uint32_t encoded_rank =
            ((uint32_t)p->query_rank & METIS_QUERY_RANK_MASK) |
            (((uint32_t)p->kv_rank << METIS_KV_RANK_SHIFT) &
             METIS_KV_RANK_MASK);
        if (p->query_add_backbone)
            encoded_rank |= METIS_QUERY_ADD_BACKBONE;
        if (metis_w_u32(f, encoded_rank) != 0) goto io_fail;
    }
    off += 4;
    if (p->has_backbone_sha256) {
        if (metis_w_bytes(
                f, p->backbone_sha256, sizeof p->backbone_sha256) != 0)
            goto io_fail;
        off += sizeof p->backbone_sha256;
    }
    for (int i = 0; i < p->n_layers; ++i) {
        if (metis_w_f32(f, p->gdu_ab_v[i]) != 0) goto io_fail;  off += 4;
    }
    for (int i = 0; i < p->n_layers; ++i) {
        if (metis_w_f32(f, p->gdu_bb_v[i]) != 0) goto io_fail;  off += 4;
    }

    uint64_t offs[T_TENSOR_COUNT];
    uint32_t crcs[T_TENSOR_COUNT];
    for (int i = 0; i < n_tensors; ++i) {
        const void *src = metis_tensor_ptr(p, tmap[i]);
        if (src == NULL) goto io_fail;
        offs[i] = off;
        if (metis_w_bytes(f, src, (size_t)ti[tmap[i]].nbytes) != 0) goto io_fail;
        crcs[i] = metis_crc32_buf(src, (size_t)ti[tmap[i]].nbytes);
        off += ti[tmap[i]].nbytes;
    }

    if (metis_w_u32(f, (uint32_t)n_tensors) != 0) goto io_fail;
    for (int i = 0; i < n_tensors; ++i) {
        uint32_t name_len = (uint32_t)strlen(ti[tmap[i]].name);
        if (metis_w_u32(f, name_len) != 0) goto io_fail;
        if (metis_w_bytes(f, ti[tmap[i]].name, name_len) != 0) goto io_fail;
        if (metis_w_u32(f, ti[tmap[i]].rows) != 0) goto io_fail;
        if (metis_w_u32(f, ti[tmap[i]].cols) != 0) goto io_fail;
        if (metis_w_u64(f, offs[i]) != 0) goto io_fail;
        if (metis_w_u32(f, crcs[i]) != 0) goto io_fail;
    }

    if (fclose(f) != 0) {
        remove(path);
        return -1;
    }
    return 0;

io_fail:
    fclose(f);
    remove(path);
    return -1;
}

int metis_file_probe(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return 0;
    uint8_t magic[METIS_HDR_MAGIC_LEN];
    int ok = metis_r_bytes(f, magic, sizeof magic) == 0 &&
             (memcmp(magic, METIS_MAGIC, sizeof magic) == 0 ||
              memcmp(magic, METIS_BOUND_MAGIC, sizeof magic) == 0 ||
              memcmp(magic, METIS_MAGIC_5, sizeof magic) == 0);
    fclose(f);
    return ok ? 1 : 0;
}

#define METIS_FAIL(...) do {                                                \
        if (err != NULL && err_size > 0)                                     \
            snprintf(err, err_size, __VA_ARGS__);                            \
        goto fail;                                                           \
    } while (0)

/* ------------------------------------------------------------------ */
/* load                                                                */
/* ------------------------------------------------------------------ */

metis_file_model_t *metis_model_load(const char *path,
                                     int d_model_expected,
                                     int q_dim_expected,
                                     int kv_dim_expected,
                                     int head_dim_expected,
                                     int block_count_expected,
                                     char *err, size_t err_size) {
    if (err != NULL && err_size > 0) err[0] = '\0';
    FILE *f = NULL;
    metis_file_model_t *m = NULL;

    if (path == NULL) METIS_FAIL("null path");
    if (d_model_expected <= 0 || q_dim_expected <= 0 || kv_dim_expected <= 0 ||
        head_dim_expected <= 0 || block_count_expected <= 0)
        METIS_FAIL("bad expected dims");
    f = fopen(path, "rb");
    if (f == NULL) METIS_FAIL("cannot open '%s'", path);

    uint64_t off = 0;
    uint8_t magic[METIS_HDR_MAGIC_LEN];
    if (metis_r_bytes(f, magic, sizeof magic) != 0) METIS_FAIL("truncated header");
    off += sizeof magic;
    if (memcmp(magic, METIS_MAGIC, sizeof magic) != 0 &&
    memcmp(magic, METIS_BOUND_MAGIC, sizeof magic) != 0 &&
    memcmp(magic, METIS_MAGIC_5, sizeof magic) != 0)
        METIS_FAIL("bad magic (not a metis bnmem file)");

    uint32_t version = 0, n_layers = 0;
    if (metis_r_u32(f, &version) != 0) METIS_FAIL("truncated header");
    off += 4;
    /* BNMEM5 was emitted before the format was renumbered to v1. Its payload
     * layout is identical; accept only the matching legacy magic/version
     * pair so unrelated version numbers still fail closed. */
    if (version != METIS_FILE_VERSION &&
        !(version == METIS_BOUND_FILE_VERSION &&
          memcmp(magic, METIS_BOUND_MAGIC, sizeof magic) == 0) &&
        !(version == 5u && memcmp(magic, METIS_MAGIC_5, sizeof magic) == 0))
        METIS_FAIL("unsupported version %u", version);
    if (metis_r_u32(f, &n_layers) != 0) METIS_FAIL("truncated header");
    off += 4;
    if (n_layers < 1 || n_layers > METIS_MAX_LAYERS)
        METIS_FAIL("n_layers %u out of range", n_layers);

    uint32_t lids[METIS_MAX_LAYERS];
    for (int i = 0; i < METIS_MAX_LAYERS; ++i) {
        if (metis_r_u32(f, &lids[i]) != 0) METIS_FAIL("truncated header");
        off += 4;
    }
    for (int i = 0; i < METIS_MAX_LAYERS; ++i) {
        if (i < (int)n_layers) {
            if (lids[i] == METIS_UNUSED_LAYER ||
                lids[i] >= (uint32_t)block_count_expected)
                METIS_FAIL("layer id %u >= block count %d", lids[i],
                           block_count_expected);
            if (i > 0 && lids[i] <= lids[i - 1])
                METIS_FAIL("layer ids not strictly increasing");
        } else if (lids[i] != METIS_UNUSED_LAYER) {
            METIS_FAIL("unused layer slot %d is not the sentinel", i);
        }
    }

    uint32_t d_model = 0, kv_dim = 0, q_dim = 0, head_dim = 0;
    if (metis_r_u32(f, &d_model) != 0) METIS_FAIL("truncated header");
    if (metis_r_u32(f, &kv_dim) != 0) METIS_FAIL("truncated header");
    if (metis_r_u32(f, &q_dim) != 0) METIS_FAIL("truncated header");
    if (metis_r_u32(f, &head_dim) != 0) METIS_FAIL("truncated header");
    off += 4 * 4;
    float gamma = 0, tau = 0, rho = 0;
    if (metis_r_f32(f, &gamma) != 0) METIS_FAIL("truncated header");
    if (metis_r_f32(f, &tau) != 0) METIS_FAIL("truncated header");
    if (metis_r_f32(f, &rho) != 0) METIS_FAIL("truncated header");
    off += 3 * 4;
    uint32_t k_min = 0;
    if (metis_r_u32(f, &k_min) != 0) METIS_FAIL("truncated header");
    off += 4;
    float gdu_ab = 0, gdu_bb = 0, beta_scale = 0;
    if (metis_r_f32(f, &gdu_ab) != 0) METIS_FAIL("truncated header");
    if (metis_r_f32(f, &gdu_bb) != 0) METIS_FAIL("truncated header");
    if (metis_r_f32(f, &beta_scale) != 0) METIS_FAIL("truncated header");
    off += 3 * 4;

    uint32_t denom_plus_one = 0;
    uint32_t encoded_query_rank = 0;
    if (metis_r_u32(f, &denom_plus_one) != 0) METIS_FAIL("truncated header");
    if (denom_plus_one > 2u) METIS_FAIL("denominator mode %u", denom_plus_one);
    off += 4;
    if (metis_r_u32(f, &encoded_query_rank) != 0)
        METIS_FAIL("truncated header");
    int query_add_backbone =
        (encoded_query_rank & METIS_QUERY_ADD_BACKBONE) != 0;
    uint32_t query_rank =
        encoded_query_rank & METIS_QUERY_RANK_MASK;
    uint32_t kv_rank =
        (encoded_query_rank & METIS_KV_RANK_MASK) >>
        METIS_KV_RANK_SHIFT;
    if (query_add_backbone && query_rank == 0)
        METIS_FAIL("full query cannot use backbone-delta mode");
    if (query_rank > q_dim || query_rank > d_model)
        METIS_FAIL("query_rank %u out of range", query_rank);
    if (kv_rank > kv_dim || kv_rank > d_model)
        METIS_FAIL("kv_rank %u out of range", kv_rank);
    off += 4;
    uint8_t backbone_sha256[32];
    int has_backbone_sha256 =
        version == METIS_BOUND_FILE_VERSION;
    memset(backbone_sha256, 0, sizeof backbone_sha256);
    if (has_backbone_sha256) {
        if (metis_r_bytes(f, backbone_sha256, sizeof backbone_sha256) != 0)
            METIS_FAIL("truncated backbone SHA-256");
        off += sizeof backbone_sha256;
    }

    if (d_model != (uint32_t)d_model_expected)
        METIS_FAIL("d_model mismatch: file %u, expected %d", d_model,
                   d_model_expected);
    if (q_dim != (uint32_t)q_dim_expected)
        METIS_FAIL("q_dim mismatch: file %u, expected %d", q_dim,
                   q_dim_expected);
    if (kv_dim != (uint32_t)kv_dim_expected)
        METIS_FAIL("kv_dim mismatch: file %u, expected %d", kv_dim,
                   kv_dim_expected);
    if (head_dim != (uint32_t)head_dim_expected)
        METIS_FAIL("head_dim mismatch: file %u, expected %d", head_dim,
                   head_dim_expected);
    if (!(beta_scale > 0.0f && beta_scale <= 1.0f))
        METIS_FAIL("beta_scale out of range");
    if (k_min > 1000000000u) METIS_FAIL("k_min out of range");

    int ids[METIS_MAX_LAYERS];
    for (int i = 0; i < METIS_MAX_LAYERS; ++i)
        ids[i] = (i < (int)n_layers) ? (int)lids[i] : -1;
    m = (metis_file_model_t *)calloc(1, sizeof *m);
    if (m == NULL) METIS_FAIL("out of memory");
    if (metis_model_alloc(m, (int)n_layers, ids, (int)d_model,
                          (int)kv_dim, (int)q_dim, (int)head_dim,
                          gamma, tau, rho, (int)k_min, beta_scale,
                          (int)denom_plus_one, (int)query_rank) != 0)
        METIS_FAIL("model allocation failed");
    m->params.query_add_backbone = query_add_backbone;
    m->params.has_backbone_sha256 = has_backbone_sha256;
    if (has_backbone_sha256)
        memcpy(m->params.backbone_sha256, backbone_sha256,
               sizeof backbone_sha256);
    if (kv_rank > 0) {
        metis_params_t *p = &m->params;
        free(p->wk); p->wk = NULL;
        free(p->wv); p->wv = NULL;
        p->kv_rank = (int)kv_rank;
        p->wk_a = metis_xcalloc(
            (uint64_t)n_layers * kv_dim * kv_rank);
        p->wk_b = metis_xcalloc(
            (uint64_t)n_layers * kv_rank * d_model);
        p->wv_a = metis_xcalloc(
            (uint64_t)n_layers * kv_dim * kv_rank);
        p->wv_b = metis_xcalloc(
            (uint64_t)n_layers * kv_rank * d_model);
        if (p->wk_a == NULL || p->wk_b == NULL ||
            p->wv_a == NULL || p->wv_b == NULL)
            METIS_FAIL("low-rank K/V allocation failed");
    }
    for (int i = 0; i < (int)n_layers; ++i)
        if (metis_r_f32(f, &m->params.gdu_ab_v[i]) != 0)
            METIS_FAIL("truncated alpha biases");
    for (int i = 0; i < (int)n_layers; ++i)
        if (metis_r_f32(f, &m->params.gdu_bb_v[i]) != 0)
            METIS_FAIL("truncated beta biases");
    off += 2ull * n_layers * 4u;
    m->params.gdu_ab = m->params.gdu_ab_v[0];
    m->params.gdu_bb = m->params.gdu_bb_v[0];

    int tmap[T_TENSOR_COUNT];
    int n_tensors = metis_build_tmap(&m->params, tmap);
    metis_tinfo_t ti[T_TENSOR_COUNT];
    metis_build_tinfo(&m->params, ti);
    uint64_t exp_offs[T_TENSOR_COUNT];
    uint32_t crcs[T_TENSOR_COUNT];
    for (int i = 0; i < n_tensors; ++i) {
        void *dst = (void *)(uintptr_t)metis_tensor_ptr(&m->params, tmap[i]);
        if (dst == NULL) METIS_FAIL("internal: no tensor %d", tmap[i]);
        exp_offs[i] = off;
        if (metis_r_bytes(f, dst, (size_t)ti[tmap[i]].nbytes) != 0)
            METIS_FAIL("truncated in tensor '%s'", ti[tmap[i]].name);
        crcs[i] = metis_crc32_buf(dst, (size_t)ti[tmap[i]].nbytes);
        off += ti[tmap[i]].nbytes;
    }

    uint32_t count = 0;
    if (metis_r_u32(f, &count) != 0) METIS_FAIL("truncated manifest");
    if (count != (uint32_t)n_tensors)
        METIS_FAIL("manifest count %u != %d", count, n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        uint32_t name_len = 0, rows = 0, cols = 0, crc = 0;
        uint64_t moff = 0;
        char name[64];
        if (metis_r_u32(f, &name_len) != 0) METIS_FAIL("truncated manifest");
        if (name_len == 0 || name_len >= sizeof name)
            METIS_FAIL("manifest name length %u out of range", name_len);
        if (metis_r_bytes(f, name, name_len) != 0) METIS_FAIL("truncated manifest");
        name[name_len] = '\0';
        if (metis_r_u32(f, &rows) != 0) METIS_FAIL("truncated manifest");
        if (metis_r_u32(f, &cols) != 0) METIS_FAIL("truncated manifest");
        if (metis_r_u64(f, &moff) != 0) METIS_FAIL("truncated manifest");
        if (metis_r_u32(f, &crc) != 0) METIS_FAIL("truncated manifest");
        if (strcmp(name, ti[tmap[i]].name) != 0)
            METIS_FAIL("manifest tensor %d: name '%s' != '%s'", i, name,
                    ti[tmap[i]].name);
        if (rows != ti[tmap[i]].rows || cols != ti[tmap[i]].cols)
            METIS_FAIL("manifest tensor '%s': dims %ux%u != %ux%u", name, rows,
                    cols, ti[tmap[i]].rows, ti[tmap[i]].cols);
        if (moff != exp_offs[i])
            METIS_FAIL("manifest tensor '%s': offset %llu != %llu", name,
                    (unsigned long long)moff, (unsigned long long)exp_offs[i]);
        if (crc != crcs[i])
            METIS_FAIL("manifest tensor '%s': crc mismatch", name);
    }
    if (fgetc(f) != EOF) METIS_FAIL("trailing data after manifest");

    fclose(f);
    return m;

fail:
    if (f != NULL) fclose(f);
    metis_model_free_loaded(m);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* mutable memory-state snapshot                                       */
/* ------------------------------------------------------------------ */

#define METIS_STATE_VERSION 1u
static const uint8_t METIS_STATE_MAGIC[METIS_HDR_MAGIC_LEN] =
    { 'B', 'N', 'M', 'S', 'T', '1', 0, 0 };

int metis_state_save(const metis_params_t *p, const float *M, const float *S,
                     int active, const char *path) {
    FILE *f = NULL;
    uint64_t m_count, s_count;
    uint32_t m_crc, s_crc;
    if (p == NULL || M == NULL || S == NULL || path == NULL ||
        p->n_layers < 1 || p->n_layers > METIS_MAX_LAYERS ||
        p->kv_dim < 1 || active < 0 || active > 1) {
        return -1;
    }
    m_count = (uint64_t)p->n_layers * (uint64_t)p->kv_dim *
              (uint64_t)p->kv_dim;
    s_count = (uint64_t)p->n_layers * (uint64_t)p->kv_dim;
    if (m_count > SIZE_MAX / sizeof(float) ||
        s_count > SIZE_MAX / sizeof(float)) {
        return -1;
    }
    m_crc = metis_crc32_buf(M, (size_t)m_count * sizeof(float));
    s_crc = metis_crc32_buf(S, (size_t)s_count * sizeof(float));
    f = fopen(path, "wb");
    if (f == NULL) return -1;
    if (metis_w_bytes(f, METIS_STATE_MAGIC, sizeof METIS_STATE_MAGIC) != 0 ||
        metis_w_u32(f, METIS_STATE_VERSION) != 0 ||
        metis_w_u32(f, (uint32_t)p->n_layers) != 0 ||
        metis_w_u32(f, (uint32_t)p->kv_dim) != 0 ||
        metis_w_u32(f, (uint32_t)active) != 0) goto io_fail;
    for (int i = 0; i < METIS_MAX_LAYERS; ++i) {
        uint32_t id = i < p->n_layers ? (uint32_t)p->layer_ids[i] :
                                       METIS_UNUSED_LAYER;
        if (metis_w_u32(f, id) != 0) goto io_fail;
    }
    if (metis_w_u64(f, m_count) != 0 || metis_w_u64(f, s_count) != 0 ||
        metis_w_u32(f, m_crc) != 0 || metis_w_u32(f, s_crc) != 0 ||
        metis_w_bytes(f, M, (size_t)m_count * sizeof(float)) != 0 ||
        metis_w_bytes(f, S, (size_t)s_count * sizeof(float)) != 0 ||
        fflush(f) != 0) {
        fclose(f);
        return -1;
    }
    return fclose(f) == 0 ? 0 : -1;
io_fail:
    fclose(f);
    return -1;
}

int metis_state_load(const metis_params_t *p, float *M, float *S,
                     int *active, const char *path,
                     char *err, size_t err_size) {
    FILE *f = NULL;
    float *new_M = NULL, *new_S = NULL;
    uint8_t magic[METIS_HDR_MAGIC_LEN];
    uint32_t version, n_layers, kv_dim, state_active, m_crc, s_crc;
    uint64_t m_count, s_count, expected_m, expected_s;
    if (err != NULL && err_size > 0) err[0] = '\0';
    if (p == NULL || M == NULL || S == NULL || active == NULL || path == NULL)
        METIS_FAIL("invalid state import arguments");
    f = fopen(path, "rb");
    if (f == NULL) METIS_FAIL("cannot open '%s'", path);
    if (metis_r_bytes(f, magic, sizeof magic) != 0 ||
        memcmp(magic, METIS_STATE_MAGIC, sizeof magic) != 0)
        METIS_FAIL("bad memory state magic");
    if (metis_r_u32(f, &version) != 0 || version != METIS_STATE_VERSION)
        METIS_FAIL("unsupported memory state version");
    if (metis_r_u32(f, &n_layers) != 0 ||
        metis_r_u32(f, &kv_dim) != 0 ||
        metis_r_u32(f, &state_active) != 0)
        METIS_FAIL("truncated memory state header");
    if (n_layers != (uint32_t)p->n_layers || kv_dim != (uint32_t)p->kv_dim)
        METIS_FAIL("memory state geometry mismatch");
    if (state_active > 1u) METIS_FAIL("invalid memory state active flag");
    for (int i = 0; i < METIS_MAX_LAYERS; ++i) {
        uint32_t id;
        uint32_t expected = i < p->n_layers ? (uint32_t)p->layer_ids[i] :
                                             METIS_UNUSED_LAYER;
        if (metis_r_u32(f, &id) != 0) METIS_FAIL("truncated layer ids");
        if (id != expected) METIS_FAIL("memory state layer ids mismatch");
    }
    if (metis_r_u64(f, &m_count) != 0 || metis_r_u64(f, &s_count) != 0 ||
        metis_r_u32(f, &m_crc) != 0 || metis_r_u32(f, &s_crc) != 0)
        METIS_FAIL("truncated memory state metadata");
    expected_m = (uint64_t)p->n_layers * (uint64_t)p->kv_dim *
                 (uint64_t)p->kv_dim;
    expected_s = (uint64_t)p->n_layers * (uint64_t)p->kv_dim;
    if (m_count != expected_m || s_count != expected_s)
        METIS_FAIL("memory state payload size mismatch");
    new_M = (float *)metis_xcalloc(m_count);
    new_S = (float *)metis_xcalloc(s_count);
    if (new_M == NULL || new_S == NULL) METIS_FAIL("out of memory");
    if (metis_r_bytes(f, new_M, (size_t)m_count * sizeof(float)) != 0 ||
        metis_r_bytes(f, new_S, (size_t)s_count * sizeof(float)) != 0)
        METIS_FAIL("truncated memory state payload");
    if (fgetc(f) != EOF) METIS_FAIL("trailing data after memory state");
    if (metis_crc32_buf(new_M, (size_t)m_count * sizeof(float)) != m_crc ||
        metis_crc32_buf(new_S, (size_t)s_count * sizeof(float)) != s_crc)
        METIS_FAIL("memory state CRC mismatch");
    memcpy(M, new_M, (size_t)m_count * sizeof(float));
    memcpy(S, new_S, (size_t)s_count * sizeof(float));
    *active = (int)state_active;
    free(new_M);
    free(new_S);
    fclose(f);
    return 0;
fail:
    free(new_M);
    free(new_S);
    if (f != NULL) fclose(f);
    return -1;
}

/* ---- v4 reference-semantics math ---- */

static float metis_sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

int metis_commit_states(const metis_params_t *p, int slot,
                              float *M, float *S, const float *h_all, int L,
                              const float *norm_w, float eps) {
    /* Reference StraightThroughAlphaTopPGatedDeltaRule commit:
     *   h_normed = attn_norm(raw residual)  -- SINGLE norm (no double-norm)
     *   scores = pool_score(h_normed); p = softmax(scores/tau)
     *   AlphaTopP(rho, k_min) hard select; w = p*hard/mass
     *   K = L2normalize(W_k h)/sqrt(kv_dim); V = W_v h
     *   alpha = sum w*sigmoid(gdu_aw.h + gdu_ab_v) / sum w
     *   beta_i = beta_scale * sigmoid(gdu_bw.h + gdu_bb_v)   (b_eff = w*beta
     *                                                          folded below)
     *   M <- alpha*(M - K^T(b_eff*(K M))) + K^T(b_eff*V)
     *   S <- alpha*(S - K^T(b_eff*(K S))) + K^T b_eff
     * S IS pre-scaled by alpha (reference law; differs from our v3 S). */
    if (p == NULL || M == NULL || S == NULL || h_all == NULL ||
        norm_w == NULL || slot < 0 || slot >= p->n_layers ||
        L <= 0 || L > METIS_CAP_ROWS) {
        return -1;
    }
    const int d = p->d_model;
    const int dk = p->kv_dim, dv = p->kv_dim;
    const size_t layer_off = (size_t)slot;

    float *h_normed = (float *)malloc((size_t)L * (size_t)d * sizeof(float));
    float *K = (float *)malloc((size_t)L * (size_t)dk * sizeof(float));
    float *V = (float *)malloc((size_t)L * (size_t)dv * sizeof(float));
    float *scores = (float *)malloc((size_t)L * sizeof(float));
    float *probs = (float *)malloc((size_t)L * sizeof(float));
    float *w_sel = (float *)malloc((size_t)L * sizeof(float));
    float *beta_i = (float *)malloc((size_t)L * sizeof(float));
    if (!h_normed || !K || !V || !scores || !probs || !w_sel || !beta_i) {
        free(h_normed); free(K); free(V);
        free(scores); free(probs); free(w_sel); free(beta_i);
        return -1;
    }

    /* SINGLE standard RMSNorm: normalize the RAW residual first, then apply
     * the learned backbone norm weight. */
    for (int l = 0; l < L; ++l) {
        const float *h = h_all + (size_t)l * (size_t)d;
        float *hn = h_normed + (size_t)l * (size_t)d;
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += h[i] * h[i];
        }
        float inv = 1.0f / sqrtf(sum / (float)d + eps);
        for (int i = 0; i < d; ++i) hn[i] = h[i] * inv * norm_w[i];
    }

    /* scores + softmax(scores/tau) */
    const float *w_agg = p->w_agg + layer_off * (size_t)d;
    for (int l = 0; l < L; ++l) {
        const float *hn = h_normed + (size_t)l * (size_t)d;
        float acc = 0.0f;
        for (int i = 0; i < d; ++i) acc += w_agg[i] * hn[i];
        scores[l] = acc;
    }
    {
        float mx = scores[0];
        for (int l = 1; l < L; ++l) if (scores[l] > mx) mx = scores[l];
        float zsum = 0.0f;
        for (int l = 0; l < L; ++l) {
            probs[l] = expf((scores[l] - mx) / p->tau);
            zsum += probs[l];
        }
        float inv = 1.0f / zsum;
        for (int l = 0; l < L; ++l) probs[l] *= inv;
    }
    /* AlphaTopP hard select: descending cumulative mass > rho, k_min floor */
    {
        int idx[METIS_CAP_ROWS];
        for (int l = 0; l < L; ++l) idx[l] = l;
        for (int a = 1; a < L; ++a) {
            int ii = idx[a]; int b = a - 1;
            while (b >= 0 && probs[idx[b]] < probs[ii]) { idx[b+1] = idx[b]; --b; }
            idx[b+1] = ii;
        }
        float cum = 0.0f; int k = 1;
        for (int l = 0; l < L; ++l) {
            cum += probs[idx[l]];
            if (cum > p->rho) { k = l + 1; break; }
            k = l + 1;
        }
        if (k < p->k_min) k = p->k_min;
        if (k > L) k = L;
        float mass = 0.0f;
        for (int l = 0; l < k; ++l) mass += probs[idx[l]];
        if (mass < 1e-6f) mass = 1e-6f;
        for (int l = 0; l < L; ++l) w_sel[l] = 0.0f;
        for (int l = 0; l < k; ++l) w_sel[idx[l]] = probs[idx[l]] / mass;
    }

    /* K/V model projections.  Low-rank files execute A(Bh) directly;
     * the dynamic M/S state below remains a full [kv,kv]/[kv] pair. */
    if (p->kv_rank > 0) {
        const int r = p->kv_rank;
        const float *wk_a = p->wk_a +
            layer_off * (size_t)dk * (size_t)r;
        const float *wk_b = p->wk_b +
            layer_off * (size_t)r * (size_t)d;
        const float *wv_a = p->wv_a +
            layer_off * (size_t)dv * (size_t)r;
        const float *wv_b = p->wv_b +
            layer_off * (size_t)r * (size_t)d;
        float *kh = (float *)malloc((size_t)r * sizeof(float));
        float *vh = (float *)malloc((size_t)r * sizeof(float));
        if (kh == NULL || vh == NULL) {
            free(kh); free(vh);
            free(h_normed); free(K); free(V);
            free(scores); free(probs); free(w_sel); free(beta_i);
            return -1;
        }
        for (int l = 0; l < L; ++l) {
            const float *hn = h_normed + (size_t)l * (size_t)d;
            float *k = K + (size_t)l * (size_t)dk;
            float *v = V + (size_t)l * (size_t)dv;
            for (int j = 0; j < r; ++j) {
                float ka = 0.0f, va = 0.0f;
                const float *kb = wk_b + (size_t)j * (size_t)d;
                const float *vb = wv_b + (size_t)j * (size_t)d;
                for (int i = 0; i < d; ++i) {
                    ka += kb[i] * hn[i];
                    va += vb[i] * hn[i];
                }
                kh[j] = ka;
                vh[j] = va;
            }
            for (int o = 0; o < dk; ++o) {
                float ka = 0.0f, va = 0.0f;
                const float *karow = wk_a + (size_t)o * (size_t)r;
                const float *varow = wv_a + (size_t)o * (size_t)r;
                for (int j = 0; j < r; ++j) {
                    ka += karow[j] * kh[j];
                    va += varow[j] * vh[j];
                }
                k[o] = ka;
                v[o] = va;
            }
        }
        free(kh);
        free(vh);
    } else {
        const float *wk = p->wk + layer_off * (size_t)dk * (size_t)d;
        const float *wv = p->wv + layer_off * (size_t)dv * (size_t)d;
        for (int l = 0; l < L; ++l) {
            const float *hn = h_normed + (size_t)l * (size_t)d;
            float *k = K + (size_t)l * (size_t)dk;
            float *v = V + (size_t)l * (size_t)dv;
            for (int o = 0; o < dk; ++o) {
                const float *row = wk + (size_t)o * (size_t)d;
                float acc = 0.0f;
                for (int i = 0; i < d; ++i) acc += row[i] * hn[i];
                k[o] = acc;
            }
            for (int o = 0; o < dv; ++o) {
                const float *row = wv + (size_t)o * (size_t)d;
                float acc = 0.0f;
                for (int i = 0; i < d; ++i) acc += row[i] * hn[i];
                v[o] = acc;
            }
        }
    }
    {
        const float inv_sqrt_dk = 1.0f / sqrtf((float)dk);
        for (int l = 0; l < L; ++l) {
            float *k = K + (size_t)l * (size_t)dk;
            float nrm = 0.0f;
            for (int i = 0; i < dk; ++i) nrm += k[i] * k[i];
            nrm = 1.0f / sqrtf(nrm + 1e-12f);
            for (int i = 0; i < dk; ++i) k[i] = (k[i] * nrm) * inv_sqrt_dk;
        }
    }

    /* per-layer TRAINED gate biases */
    const float *gdu_aw = p->gdu_aw + layer_off * (size_t)d;
    const float *gdu_bw = p->gdu_bw + layer_off * (size_t)d;
    const float gdu_ab = p->gdu_ab_v[layer_off];
    const float gdu_bb = p->gdu_bb_v[layer_off];
    float alpha_scalar = 0.0f;
    {
        float mass = 0.0f;
        for (int l = 0; l < L; ++l) mass += w_sel[l];
        if (mass < 1e-6f) mass = 1e-6f;
        for (int l = 0; l < L; ++l) {
            const float *hn = h_normed + (size_t)l * (size_t)d;
            float acc_a = gdu_ab, acc_b = gdu_bb;
            for (int i = 0; i < d; ++i) {
                acc_a += gdu_aw[i] * hn[i];
                acc_b += gdu_bw[i] * hn[i];
            }
            beta_i[l] = p->beta_scale * metis_sigmoid(acc_b);
            alpha_scalar += (metis_sigmoid(acc_a) * w_sel[l]) / mass;
        }
    }

    /* GDU with reference S law: S pre-scaled by alpha.
     * M_new = alpha*(M - E) + A, E = sum outer(k_i, b_i*(K_i M)),
     * A = sum outer(k_i, b_i*V_i), b_i = beta_i*w_i (metis_gdu_commit).
     * S_new = alpha*S + sum_i k_i*b_i*(1 - alpha*(K_i S_old))
     *       = alpha*(S - sum_i k_i*b_i*(K_i S_old)) + sum_i k_i*b_i
     * metis_gdu_commit computes M correctly and gives
     * S += k_i b_i (1 - alpha*ks); we then need S_new = alpha*S_old +
     * that same increment -- equivalently S_new = alpha*(S_old + inc_unscaled)
     * where inc_unscaled = sum k_i b_i (1 - alpha*ks). Since gdu_commit
     * computes S_new_v3 = S_old + inc, we post-multiply the OLD rows'
     * contribution: S_new = S_new_v3 - (1-alpha)*S_old. */
    /* GDU (reference law): M <- alpha*(M - E) + A; S <- alpha*(S - E_s) + A_s
     * E = sum_i outer(k_i, b_i*(k_i . M_old)); A = sum_i outer(k_i, b_i*V_i)
     * E_s = sum_i k_i*b_i*(k_i . S_old); A_s = sum_i k_i*b_i
     * b_i = beta_i[i] * w_sel[i] (already weighted). */
    {
        float *km = (float *)malloc((size_t)L * (size_t)dv * sizeof(float));
        float *ks = (float *)malloc((size_t)L * sizeof(float));
        if (km == NULL || ks == NULL) {
            free(km); free(ks);
            free(h_normed); free(K); free(V);
            free(scores); free(probs); free(w_sel); free(beta_i);
            return -1;
        }
        /* km[i] = K_i . M_old ; ks[i] = K_i . S_old (pre-decay) */
        for (int i = 0; i < L; ++i) {
            const float *ki = K + (size_t)i * (size_t)dk;
            float *kmi = km + (size_t)i * (size_t)dv;
            float ksi = 0.0f;
            for (int r = 0; r < dk; ++r) {
                float kr = ki[r];
                const float *mrow = M + (size_t)r * (size_t)dv;
                for (int c = 0; c < dv; ++c) kmi[c] += kr * mrow[c];
                ksi += kr * S[r];
            }
            ks[i] = ksi;
        }
        /* M <- alpha*M ; S <- alpha*S (pre-decay scale) */
        for (size_t e = 0; e < (size_t)dk * (size_t)dv; ++e) M[e] *= alpha_scalar;
        for (int r = 0; r < dk; ++r) S[r] *= alpha_scalar;
        /* M += outer(k_i, b_i*(V_i - alpha*km_i)) ; S += k_i*b_i*(1 - alpha*ks_i) */
        for (int i = 0; i < L; ++i) {
            const float *ki = K + (size_t)i * (size_t)dk;
            const float *vi = V + (size_t)i * (size_t)dv;
            const float *kmi = km + (size_t)i * (size_t)dv;
            float b = beta_i[i] * w_sel[i];
            for (int r = 0; r < dk; ++r) {
                float kr = ki[r] * b;
                if (kr == 0.0f) continue;
                float *mrow = M + (size_t)r * (size_t)dv;
                for (int c = 0; c < dv; ++c)
                    mrow[c] += kr * (vi[c] - alpha_scalar * kmi[c]);
                S[r] += kr * (1.0f - alpha_scalar * ks[i]);
            }
        }
        free(km); free(ks);
    }

    free(h_normed); free(K); free(V);
    free(scores); free(probs); free(w_sel); free(beta_i);
    return 0;
}

void metis_read(const metis_params_t *p, int slot,
                      const float *h_raw, const float *q_backbone,
                      const float *M, const float *S, float *out) {
    /* Reference NormedReweightLearnedQuery read:
     * q = query_proj(h_raw)  [q_dim]  (h_raw = layer INPUT residual)
     * per head h: q_h <- RMSNorm(q_h * query_norm, head_dim, eps 1e-6)
     * per head: L2 normalize (F.normalize eps 1e-12)
     * group g (q_dim/kv_dim groups, consecutive heads): out_g = (q~ M)/(q~ S + 1) */
    const int dk = p->kv_dim, dv = p->kv_dim;
    const int d = p->d_model;
    const int n_heads = p->q_dim / p->head_dim;
    const int groups = p->q_dim / p->kv_dim;
    const int heads_per_group = n_heads / groups;
    const float *qproj = p->query_proj +
                         (size_t)slot * (size_t)p->q_dim * (size_t)d;
    const float *qn_w = p->query_norm + (size_t)slot * (size_t)p->head_dim;

    /* project: q[o] = qproj_row_o . h_raw  (v5: via rank-r factors
     * q = (h @ B^T) @ A^T — hidden[r] = B[r,:] . h first, then
     * q[o] = A[o,:] . hidden) */
    float *q = (float *)malloc((size_t)p->q_dim * sizeof(float));
    if (q == NULL) {
        for (int i = 0; i < p->q_dim; ++i) out[i] = 0.0f;
        return;
    }
    if (p->query_rank > 0 && p->query_a != NULL && p->query_b != NULL) {
        const int r = p->query_rank;
        const float *qa = p->query_a +
                          (size_t)slot * (size_t)p->q_dim * (size_t)r;
        const float *qb = p->query_b +
                          (size_t)slot * (size_t)r * (size_t)d;
        float *hid = (float *)malloc((size_t)r * sizeof(float));
        if (hid == NULL) {
            free(q);
            for (int i = 0; i < p->q_dim; ++i) out[i] = 0.0f;
            return;
        }
        for (int k = 0; k < r; ++k) {
            const float *brow = qb + (size_t)k * (size_t)d;
            float acc = 0.0f;
            for (int i = 0; i < d; ++i) acc += brow[i] * h_raw[i];
            hid[k] = acc;
        }
        for (int o = 0; o < p->q_dim; ++o) {
            const float *arow = qa + (size_t)o * (size_t)r;
            float acc = p->query_add_backbone && q_backbone != NULL ?
                        q_backbone[o] : 0.0f;
            for (int k = 0; k < r; ++k) acc += arow[k] * hid[k];
            q[o] = acc;
        }
        free(hid);
    } else {
        for (int o = 0; o < p->q_dim; ++o) {
            const float *row = qproj + (size_t)o * (size_t)d;
            float acc = 0.0f;
            for (int i = 0; i < d; ++i) acc += row[i] * h_raw[i];
            q[o] = acc;
        }
    }
    /* per-head RMSNorm(head_dim, query_norm, eps 1e-6) then L2 */
    for (int h = 0; h < n_heads; ++h) {
        float *qh = q + (size_t)h * (size_t)p->head_dim;
        float sum = 0.0f;
        for (int i = 0; i < p->head_dim; ++i) {
            qh[i] *= qn_w[i];
            sum += qh[i] * qh[i];
        }
        float inv = 1.0f / sqrtf(sum / (float)p->head_dim + 1e-6f);
        for (int i = 0; i < p->head_dim; ++i) qh[i] *= inv;
    }
    /* group reads with +1 denominator (denom_plus_one flag governs) */
    for (int g = 0; g < groups; ++g) {
        float qg[512];
        for (int h = 0; h < heads_per_group; ++h) {
            float *qh = q + (size_t)(g * heads_per_group + h) * p->head_dim;
            float nrm = 0.0f;
            for (int i = 0; i < p->head_dim; ++i) nrm += qh[i] * qh[i];
            nrm = 1.0f / sqrtf(nrm + 1e-12f);
            for (int i = 0; i < p->head_dim; ++i)
                qg[h * p->head_dim + i] = qh[i] * nrm;
        }
        float *dst = out + (size_t)g * dv;
        if (p->denom_plus_one) {
            float denom = 0.0f;
            for (int r = 0; r < dk; ++r) denom += qg[r] * S[r];
            if (p->denom_plus_one == 2) denom = fabsf(denom);
            float inv = 1.0f / (denom + 1.0f);
            for (int c = 0; c < dv; ++c) {
                float acc = 0.0f;
                for (int r = 0; r < dk; ++r)
                    acc += qg[r] * M[(size_t)r * dv + c];
                dst[c] = acc * inv;
            }
        }
    }
    free(q);
}
