/* ggwshim.c -- standalone GGUF TQ2_0/Q6_K weight streaming shim for the
 * Phase-A Python/CUDA Metis trainer (158BitNet pivot brief).
 *
 * NEW FILE -- deliberately NOT wired into the repo's CMake build. Compile
 * standalone on the server:
 *   gcc -O2 -fPIC -shared src/ggwshim.c -I src -o build/libggwshim.so
 * then load via python/ggw.py (ctypes).
 *
 * Reuses the repo's own dequant semantics by reimplementation 1:1 with
 * src/quant_tq2_0.c bitnet_tq2_0_dequantize_block and
 * src/quant_q6k.c bitnet_q6k_dequantize_block (same loops, same fp16->fp32
 * conversion, same GGUF header walk as src/gguf.c). Output is fp32 (the
 * Python side casts to bf16/fp32 as needed) instead of bf16 so the Python
 * layer controls precision.
 *
 * GGUF note: tensor dims are [in, out] but storage is row-major with
 * dims[0] contiguous, so the flat dequantized buffer reshapes directly
 * to nn.Linear.weight layout [out, in] -- no transpose.
 */
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- ggml tensor type ids (match src/gguf.c gguf_tensor_type_name) ---- */
#define SHIM_T_F32 0u
#define SHIM_T_F16 1u
#define SHIM_T_Q4_K 12u
#define SHIM_T_Q6_K 14u
#define SHIM_T_TQ2_0 35u

#define SHIM_TQ2_QK 256
#define SHIM_TQ2_QS_SIZE 64
#define SHIM_TQ2_BLOCK_BYTES 66
#define SHIM_Q6K_QK 256
#define SHIM_Q6K_BLOCK_BYTES 210
#define SHIM_Q4K_QK 256
#define SHIM_Q4K_BLOCK_BYTES 144

#define SHIM_MAX_DIMS 4
#define SHIM_MAX_TENSORS 512
#define SHIM_MAX_META 256
#define SHIM_NAME_MAX 96

typedef struct shim_tensor {
    char name[SHIM_NAME_MAX];
    uint32_t n_dims;
    uint64_t dims[SHIM_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
} shim_tensor_t;

typedef struct shim_meta {
    char key[SHIM_NAME_MAX];
    uint32_t type;
    uint64_t u64;
    double f64;
} shim_meta_t;

typedef struct shim_model {
    FILE *fp;
    uint8_t *data;         /* whole-file buffer (fread, not mmap: portable) */
    size_t file_size;
    uint64_t data_offset;  /* section start within file */
    uint64_t alignment;
    shim_tensor_t tensors[SHIM_MAX_TENSORS];
    uint64_t n_tensors;
    shim_meta_t meta[SHIM_MAX_META];
    uint64_t n_meta;
} shim_model_t;

/* fp32 -> unused here; Q4_K scale unpack (llama.cpp k4_scale semantics) */
static void q4_k_scale(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* fp16 -> fp32, 1:1 with bitnet_fp16_to_fp32 (src/quant_tq2_0.c) */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t raw = sign << 31;
            float f;
            memcpy(&f, &raw, sizeof(f));
            return f;
        }
        int e = -14;
        while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
        mant &= 0x3FFu;
        uint32_t raw = (sign << 31) | ((uint32_t)(e + 127) << 23) | (mant << 13);
        float f;
        memcpy(&f, &raw, sizeof(f));
        return f;
    }
    if (exp == 31) {
        uint32_t raw = (sign << 31) | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &raw, sizeof(f));
        return f;
    }
    exp = exp - 15u + 127u;
    mant <<= 13;
    uint32_t raw = (sign << 31) | (exp << 23) | mant;
    float f;
    memcpy(&f, &raw, sizeof(f));
    return f;
}

/* ---- minimal GGUF reader (walks exactly like src/gguf.c gguf_open) ---- */

static int sh_read(FILE *fp, void *dst, size_t n) {
    return n == 0 || fread(dst, 1, n, fp) == n ? 0 : -1;
}
static int sh_u32(FILE *fp, uint32_t *v) {
    uint8_t b[4];
    if (sh_read(fp, b, 4) != 0) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
    return 0;
}
static int sh_u64(FILE *fp, uint64_t *v) {
    uint8_t b[8];
    if (sh_read(fp, b, 8) != 0) return -1;
    uint64_t acc = 0;
    for (int i = 7; i >= 0; --i) acc = (acc << 8) | b[i];
    *v = acc;
    return 0;
}
static int sh_string(FILE *fp, char *dst, size_t cap) {
    uint64_t len = 0;
    if (sh_u64(fp, &len) != 0) return -1;
    if (len + 1 > cap) return -1;
    if (sh_read(fp, dst, (size_t)len) != 0) return -1;
    dst[len] = '\0';
    return 0;
}
static int sh_skip_string(FILE *fp) {
    uint64_t len = 0;
    if (sh_u64(fp, &len) != 0) return -1;
    return fseek(fp, (long)len, SEEK_CUR) == 0 ? 0 : -1;
}
static size_t sh_type_size(uint32_t t) {
    switch (t) { /* gguf scalar sizes */
    case 0: case 1: case 7: return 1;   /* u8 i8 bool */
    case 2: case 3:         return 2;   /* u16 i16 */
    case 4: case 5: case 6: return 4;   /* u32 i32 f32 */
    case 10: case 11: case 12: return 8; /* u64 i64 f64 */
    default: return 0;
    }
}
static int sh_skip_value(FILE *fp, uint32_t type);
static int sh_skip_array(FILE *fp) {
    uint32_t et = 0;
    uint64_t cnt = 0;
    if (sh_u32(fp, &et) != 0 || sh_u64(fp, &cnt) != 0) return -1;
    if (et == 8) { /* string array */
        for (uint64_t i = 0; i < cnt; ++i)
            if (sh_skip_string(fp) != 0) return -1;
        return 0;
    }
    if (et == 9) return -1;
    size_t es = sh_type_size(et);
    if (es == 0) return -1;
    return fseek(fp, (long)(es * cnt), SEEK_CUR) == 0 ? 0 : -1;
}
static int sh_skip_value(FILE *fp, uint32_t type) {
    size_t es = sh_type_size(type);
    if (es != 0) return fseek(fp, (long)es, SEEK_CUR) == 0 ? 0 : -1;
    if (type == 8) return sh_skip_string(fp);
    if (type == 9) return sh_skip_array(fp);
    return -1;
}

void *shim_open(const char *gguf_path) {
    if (gguf_path == NULL) return NULL;
    shim_model_t *m = (shim_model_t *)calloc(1, sizeof(*m));
    if (m == NULL) return NULL;

    FILE *fp = fopen(gguf_path, "rb");
    if (fp == NULL) { free(m); return NULL; }

    uint32_t magic = 0, version = 0;
    uint64_t tcount = 0, mcount = 0;
    if (sh_u32(fp, &magic) != 0 || magic != 0x46554747u ||
        sh_u32(fp, &version) != 0 ||
        sh_u64(fp, &tcount) != 0 || sh_u64(fp, &mcount) != 0) {
        fclose(fp); free(m); return NULL;
    }
    if (tcount > SHIM_MAX_TENSORS || mcount > SHIM_MAX_META) {
        fclose(fp); free(m); return NULL;
    }

    m->alignment = 32;
    for (uint64_t i = 0; i < mcount; ++i) {
        shim_meta_t *e = &m->meta[m->n_meta];
        if (sh_string(fp, e->key, sizeof e->key) != 0 ||
            sh_u32(fp, &e->type) != 0) {
            fclose(fp); free(m); return NULL;
        }
        if (e->type == 9) { /* array: read etype+count, skip payload */
            if (sh_skip_array(fp) != 0) { fclose(fp); free(m); return NULL; }
            continue;
        }
        size_t es = sh_type_size(e->type);
        if (es == 0 || e->type == 8) { /* skip strings */
            if (sh_skip_value(fp, e->type) != 0) { fclose(fp); free(m); return NULL; }
            continue;
        }
        if (sh_read(fp, &e->u64, es) != 0) { fclose(fp); free(m); return NULL; }
        if (e->type == 6) { float f; memcpy(&f, &e->u64, 4); e->f64 = (double)f; }
        else if (e->type == 12) { memcpy(&e->f64, &e->u64, 8); }
        else if (e->type == 4 || e->type == 10) { /* keep raw u64 */ }
        else if (e->type == 5 || e->type == 11) { e->u64 = (uint64_t)(int64_t)e->u64; }
        if (strcmp(e->key, "general.alignment") == 0) m->alignment = e->u64;
        m->n_meta++;
    }

    for (uint64_t i = 0; i < tcount; ++i) {
        shim_tensor_t *t = &m->tensors[m->n_tensors];
        if (sh_string(fp, t->name, sizeof t->name) != 0 ||
            sh_u32(fp, &t->n_dims) != 0 || t->n_dims > SHIM_MAX_DIMS) {
            fclose(fp); free(m); return NULL;
        }
        for (uint32_t d = 0; d < t->n_dims; ++d)
            if (sh_u64(fp, &t->dims[d]) != 0) { fclose(fp); free(m); return NULL; }
        if (sh_u32(fp, &t->type) != 0 || sh_u64(fp, &t->offset) != 0) {
            fclose(fp); free(m); return NULL;
        }
        m->n_tensors++;
    }

    long end_pos = ftell(fp);
    if (end_pos < 0) { fclose(fp); free(m); return NULL; }
    if (m->alignment == 0) m->alignment = 32;
    uint64_t off = (uint64_t)end_pos;
    m->data_offset = (off + m->alignment - 1) & ~(m->alignment - 1);

    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); free(m); return NULL; }
    long fsz = ftell(fp);
    if (fsz < 0 || (uint64_t)fsz < m->data_offset) { fclose(fp); free(m); return NULL; }
    m->file_size = (size_t)fsz;
    m->data = (uint8_t *)malloc(m->file_size);
    if (m->data == NULL) { fclose(fp); free(m); return NULL; }
    if (fseek(fp, 0, SEEK_SET) != 0 || sh_read(fp, m->data, m->file_size) != 0) {
        fclose(fp); free(m->data); free(m); return NULL;
    }
    fclose(fp);
    m->fp = NULL;
    return m;
}

void shim_close(shim_model_t *m) {
    if (m == NULL) return;
    free(m->data);
    free(m);
}

const shim_tensor_t *shim_find(shim_model_t *m, const char *name) {
    if (m == NULL || name == NULL) return NULL;
    for (uint64_t i = 0; i < m->n_tensors; ++i)
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    return NULL;
}

/* metadata lookup: returns 0 and fills out_u64 or out_f64 on success */
int shim_meta(shim_model_t *m, const char *key, uint32_t *type_out,
              uint64_t *u64_out, double *f64_out) {
    if (m == NULL || key == NULL) return -1;
    for (uint64_t i = 0; i < m->n_meta; ++i) {
        if (strcmp(m->meta[i].key, key) == 0) {
            if (type_out) *type_out = m->meta[i].type;
            if (u64_out) *u64_out = m->meta[i].u64;
            if (f64_out) *f64_out = m->meta[i].f64;
            return 0;
        }
    }
    return -1;
}

static uint64_t nelem(const shim_tensor_t *t) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) n *= t->dims[i];
    return n;
}

static const uint8_t *tensor_data(const shim_model_t *m, const shim_tensor_t *t) {
    return m->data + m->data_offset + t->offset;
}

/* TQ2_0 block dequant, 1:1 with bitnet_tq2_0_dequantize_block:
 * for j in {0,32}: for l 0..3: for m 0..31:
 *   q = (qs[j+m] >> (l*2)) & 3; out[(j/32)*128 + l*32 + m] = (q-1)*d */
static void dequant_tq2_block(const uint8_t *blk, float *out256) {
    uint16_t dh;
    memcpy(&dh, blk + SHIM_TQ2_QS_SIZE, 2);
    float d = fp16_to_f32(dh);
    int oi = 0;
    for (int j = 0; j < SHIM_TQ2_QS_SIZE; j += 32)
        for (int l = 0; l < 4; ++l)
            for (int mm = 0; mm < 32; ++mm) {
                int q = (blk[j + mm] >> (l * 2)) & 3;
                out256[oi++] = (float)(q - 1) * d;
            }
}

/* Q6_K block dequant, 1:1 with bitnet_q6k_dequantize_block */
static void dequant_q6k_block(const uint8_t *blk, float *out256) {
    const float d = fp16_to_f32(*(const uint16_t *)(blk + 208));
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 128 + 64);
    float *out = out256;
    for (int n = 0; n < SHIM_Q6K_QK; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l + 0] = d * (float)sc[is + 0] * (float)q1;
            out[l + 32] = d * (float)sc[is + 2] * (float)q2;
            out[l + 64] = d * (float)sc[is + 4] * (float)q3;
            out[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        out += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

/* Dequantize one whole tensor into caller fp32 buffer (row-major, flat).
 * Returns 0 ok, -1 bad args, -2 missing, -3 unsupported type. */
int shim_dequant_tensor(shim_model_t *m, const char *name, float *out,
                        size_t out_len) {
    if (m == NULL || name == NULL || out == NULL) return -1;
    const shim_tensor_t *t = shim_find(m, name);
    if (t == NULL) return -2;
    uint64_t n = nelem(t);
    if ((size_t)n > out_len) return -1;
    const uint8_t *data = tensor_data(m, t);

    if (t->type == SHIM_T_F32) {
        memcpy(out, data, (size_t)n * 4);
        return 0;
    }
    if (t->type == SHIM_T_F16) {
        const uint16_t *h = (const uint16_t *)data;
        for (uint64_t i = 0; i < n; ++i) out[i] = fp16_to_f32(h[i]);
        return 0;
    }
    if (t->type == SHIM_T_TQ2_0) {
        if (n % SHIM_TQ2_QK != 0) return -3;
        size_t nblocks = (size_t)(n / SHIM_TQ2_QK);
        float *w = out;
        for (size_t b = 0; b < nblocks; ++b) {
            dequant_tq2_block(data + b * SHIM_TQ2_BLOCK_BYTES, w);
            w += SHIM_TQ2_QK;
        }
        return 0;
    }
    if (t->type == SHIM_T_Q4_K) {
        /* Q4_K: 144B blocks of 256 elems (d, dmin fp16; 12B k4 scales;
         * 128B packed nibbles). Layout per llama.cpp / 158BitNet. */
        if (n % SHIM_Q4K_QK != 0) return -3;
        size_t nblocks = (size_t)(n / SHIM_Q4K_QK);
        const uint8_t *blk = data;
        for (size_t b = 0; b < nblocks; ++b) {
            float d = fp16_to_f32(*(const uint16_t *)(blk + 0));
            float mn = fp16_to_f32(*(const uint16_t *)(blk + 2));
            const uint8_t *scales = blk + 4;
            const uint8_t *qs = blk + 16;
            float *base = out + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, mm;
                q4_k_scale(is + 0, scales, &sc, &mm);
                float d1 = d * (float)sc, m1 = mn * (float)mm;
                q4_k_scale(is + 1, scales, &sc, &mm);
                float d2 = d * (float)sc, m2 = mn * (float)mm;
                for (int l = 0; l < 32; ++l)
                    base[j + l] = d1 * (float)(qs[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l)
                    base[j + 32 + l] = d2 * (float)(qs[l] >> 4) - m2;
                qs += 32;
                is += 2;
            }
            blk += SHIM_Q4K_BLOCK_BYTES;
        }
        return 0;
    }
    if (t->type == SHIM_T_Q6_K) {
        if (n % SHIM_Q6K_QK != 0) return -3;
        size_t nblocks = (size_t)(n / SHIM_Q6K_QK);
        float *w = out;
        for (size_t b = 0; b < nblocks; ++b) {
            dequant_q6k_block(data + b * SHIM_Q6K_BLOCK_BYTES, w);
            w += SHIM_Q6K_QK;
        }
        return 0;
    }
    return -3;
}

/* geometry helper: out = {n_layers(block_count), hidden, kv_dim, q_dim,
 * ffn, vocab, rope_dim}. Derived from architecture metadata with tensor-dims
 * fallback, matching bitnet.c's config reading. */
int shim_geometry(shim_model_t *m, int out[7]) {
    if (m == NULL || out == NULL) return -1;
    uint64_t bc = 0, hidden = 0, kv = 0, qd = 0, ffn = 0, vocab = 0, rope = 0;
    shim_meta(m, "llama.block_count", NULL, &bc, NULL);
    if (bc == 0)
        shim_meta(m, "minicpm.block_count", NULL, &bc, NULL);
    shim_meta(m, "general.block_count", NULL, &bc, NULL);
    const shim_tensor_t *q = shim_find(m, "blk.0.attn_q.weight");
    const shim_tensor_t *k = shim_find(m, "blk.0.attn_k.weight");
    const shim_tensor_t *gate = shim_find(m, "blk.0.ffn_gate.weight");
    const shim_tensor_t *embd = shim_find(m, "token_embd.weight");
    if (q == NULL || k == NULL || gate == NULL || embd == NULL) return -1;
    hidden = q->dims[0];
    qd = q->dims[1];
    kv = k->dims[1];
    ffn = gate->dims[1];
    vocab = embd->dims[1];
    rope = qd / (qd / hidden > 0 ? (qd / hidden) : 1); /* placeholder; refined below */
    /* head_dim = q_dim / n_heads; rope_dim falls back to head_dim. */
    uint64_t n_heads = 0;
    shim_meta(m, "llama.attention.head_count", NULL, &n_heads, NULL);
    if (n_heads == 0)
        shim_meta(m, "minicpm.attention.head_count", NULL, &n_heads, NULL);
    uint64_t rope_dim = 0;
    shim_meta(m, "llama.rope.dimension_count", NULL, &rope_dim, NULL);
    if (rope_dim == 0)
        shim_meta(m, "minicpm.rope.dimension_count", NULL, &rope_dim, NULL);
    if (n_heads > 0) {
        uint64_t head_dim = qd / n_heads;
        rope = rope_dim > 0 ? rope_dim : head_dim;
    } else {
        rope = rope_dim;
    }
    if (bc == 0) {
        int max_layer = -1;
        for (uint64_t i = 0; i < m->n_tensors; ++i) {
            const char *nm = m->tensors[i].name;
            if (strncmp(nm, "blk.", 4) == 0 && nm[4] >= '0' && nm[4] <= '9') {
                int layer = atoi(nm + 4);
                if (layer > max_layer) max_layer = layer;
            }
        }
        bc = (uint64_t)(max_layer + 1);
    }
    if (bc == 0 || hidden == 0 || ffn == 0 || vocab == 0) return -1;
    out[0] = (int)bc; out[1] = (int)hidden; out[2] = (int)kv; out[3] = (int)qd;
    out[4] = (int)ffn; out[5] = (int)vocab; out[6] = (int)rope;
    return 0;
}
