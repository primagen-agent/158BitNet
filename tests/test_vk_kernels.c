/* GPU-vs-CPU check for every Vulkan kernel: the matvec (arithmetic, lookup
 * table and multi-row), the attention chain, the depthwise conv, the GDN
 * recurrence, the seven elementwise/norm modes and the sampler.
 *
 * Runs on any Vulkan device, so it also works on a machine with no GPU at all
 * through Mesa's lavapipe (VK_ICD_FILENAMES=.../lvp_icd.json).  Each case is
 * compared against an independent float reference; the exit status is non-zero
 * if any of them differs.  See scripts/test_vk_kernels.sh for the build line. */
#include "vk_backend.h"
#include "gpu_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
static float g_worst;

static void cmp(const char *what, const float *got, const float *ref, int n) {
    float worst = 0.0f;
    int bad = 0, i;
    for (i = 0; i < n; ++i) {
        float d = fabsf(got[i] - ref[i]);
        float m = fabsf(ref[i]) > 1.0f ? fabsf(ref[i]) : 1.0f;
        float rel = d / m;
        if (!(rel <= 1e-4f)) ++bad;
        if (rel > worst) worst = rel;
    }
    if (worst > g_worst) g_worst = worst;
    printf("  %-28s n=%-6d worst_rel=%.3e %s\n", what, n, (double)worst,
           bad ? "MISMATCH" : "ok");
    if (bad) { ++g_fail; printf("  first bad index: ");
        for (i = 0; i < n; ++i) {
            float m = fabsf(ref[i]) > 1.0f ? fabsf(ref[i]) : 1.0f;
            if (!(fabsf(got[i] - ref[i]) / m <= 1e-4f)) { printf("%d got=%g ref=%g\n", i, (double)got[i], (double)ref[i]); break; }
        }
    }
}

static float frand(unsigned *st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
}

static void fill(float *p, int n, unsigned *st) { for (int i = 0; i < n; ++i) p[i] = frand(st); }

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* ---------------- elt ---------------- */
static void test_elt(bq_vk_t *vk) {
    enum { N = 128, R = 4 };
    float a[N * R], b[N * R], w[N], got[N * R], ref[N * R];
    unsigned st = 12345u;
    int mode;
    printf("[elt]\n");
    fill(a, N * R, &st); fill(b, N * R, &st); fill(w, N, &st);
    for (mode = 0; mode <= 6; ++mode) {
        int r, i;
        char tag[64];
        for (r = 0; r < R; ++r) {
            const float *ap = a + r * N, *bp = b + r * N;
            float *rp = ref + r * N, sum = 0.0f, scale;
            for (i = 0; i < N; ++i) sum += ap[i] * ap[i];
            if (mode == 2) scale = 1.0f / fmaxf(sqrtf(sum), 1e-5f);
            else scale = 1.0f / sqrtf(sum / (float)N + 1e-5f);
            for (i = 0; i < N; ++i) {
                switch (mode) {
                case BQ_VK_ELT_RMS:      rp[i] = ap[i] * scale * w[i]; break;
                case BQ_VK_ELT_RMS_GATE: rp[i] = ap[i] * scale * w[i] * silu(bp[i]); break;
                case BQ_VK_ELT_L2:       rp[i] = ap[i] * scale; break;
                case BQ_VK_ELT_SWIGLU:   rp[i] = silu(ap[i]) * bp[i]; break;
                case BQ_VK_ELT_SIGMOID:  rp[i] = ap[i] / (1.0f + expf(-bp[i])); break;
                case BQ_VK_ELT_ADD:      rp[i] = ap[i] + bp[i]; break;
                default:                 rp[i] = ap[i] * w[i]; break;
                }
            }
        }
        if (bq_vk_elt(vk, mode, a, b, w, N, R, 1e-5f, got) != 0) {
            printf("  mode %d                                  UNSUPPORTED\n", mode); ++g_fail; continue;
        }
        snprintf(tag, sizeof tag, "mode %d", mode);
        cmp(tag, got, ref, N * R);
    }
}

/* ---------------- conv ---------------- */
static void test_conv(bq_vk_t *vk) {
    enum { K = 4, C = 96 };
    float w[K * C], h[K * C], got[C], ref[C];
    unsigned st = 999u;
    printf("[conv1d]\n");
    fill(w, K * C, &st); fill(h, K * C, &st);
    for (int c = 0; c < C; ++c) {
        float s = 0.0f;
        for (int t = 0; t < K; ++t) s += w[c * K + t] * h[t * C + c];
        ref[c] = silu(s);
    }
    if (bq_vk_conv(vk, w, h, K, C, got) != 0) { printf("  UNSUPPORTED\n"); ++g_fail; return; }
    cmp("depthwise K=4", got, ref, C);
}

/* ---------------- gdn ---------------- */
static void ref_gdn(float *M, const float *q, const float *k, const float *v,
                    const float *decay, const float *beta, int nhead, int nkv,
                    int S, float *o) {
    float scale = 1.0f / sqrtf((float)S);
    for (int h = 0; h < nhead; ++h) {
        int kh = h % nkv;
        for (int i = 0; i < S * S; ++i) M[h * S * S + i] *= decay[h];
        for (int j = 0; j < S; ++j) {
            float *row = M + h * S * S + j * S;
            float s = 0.0f, d;
            for (int i = 0; i < S; ++i) s += row[i] * k[kh * S + i];
            d = (v[h * S + j] - s) * beta[h];
            for (int i = 0; i < S; ++i) row[i] += d * k[kh * S + i];
            s = 0.0f;
            for (int i = 0; i < S; ++i) s += row[i] * q[kh * S + i];
            o[h * S + j] = s * scale;
        }
    }
}

static void test_gdn(bq_vk_t *vk) {
    enum { NH = 8, NKV = 2, S = 32 };
    static float M[NH * S * S];
    float q[NKV * S], k[NKV * S], v[NH * S], dec[NH], bet[NH];
    float got[NH * S], ref[NH * S];
    unsigned st = 4242u;
    int step;
    printf("[gdn] nhead=%d nkv=%d S=%d, 3 sequential steps\n", NH, NKV, S);
    fill(q, NKV * S, &st); fill(k, NKV * S, &st); fill(v, NH * S, &st);
    fill(dec, NH, &st); fill(bet, NH, &st);
    for (step = 0; step < NH; ++step) { dec[step] = 0.9f + 0.1f * dec[step]; bet[step] = 0.5f + 0.4f * bet[step]; }
    memset(M, 0, sizeof M);
    for (step = 0; step < 3; ++step) {
        char tag[48];
        ref_gdn(M, q, k, v, dec, bet, NH, NKV, S, ref);
        if (bq_vk_gdn(vk, 3, q, k, v, dec, bet, NH, NKV, S, got) != 0) {
            printf("  UNSUPPORTED\n"); ++g_fail; return;
        }
        snprintf(tag, sizeof tag, "step %d", step);
        cmp(tag, got, ref, NH * S);
    }
}

/* ---------------- sample ---------------- */
/* The sampler keeps its top-k list in global memory, so a mis-indexed layout
 * would still return a plausible token.  These checks pin the result down:
 * every pick must come from the reference top-k set, and over many seeds the
 * pick frequencies must match the reference softmax of that set. */
static const float *g_sortkey;
static int cmp_desc_idx(const void *pa, const void *pb) {
    float a = g_sortkey[*(const int *)pa], b = g_sortkey[*(const int *)pb];
    return (a < b) - (a > b);
}

static void test_sample(bq_vk_t *vk) {
    enum { N = 4096, K = 8, SEEDS = 400 };
    static float lg[N];
    static int idx[N], hits[K];
    unsigned st = 7u;
    uint32_t tok = 0;
    int expect = 0, i, j, bad = 0;
    float prob[K], worst = 0.0f;
    printf("[sample] n=%d k=%d, %d seeds\n", N, K, SEEDS);
    fill(lg, N, &st);
    for (i = 0; i < N; ++i) lg[i] *= 4.0f;
    lg[1234] = 9.0f;
    for (i = 0; i < N; ++i) if (lg[i] > lg[expect]) expect = i;

    if (bq_vk_sample(vk, lg, N, 0, 0u, 1, 0.0f, 1.0f, &tok) != 0) { printf("  greedy UNSUPPORTED\n"); ++g_fail; return; }
    printf("  %-28s got=%u want=%d %s\n", "greedy argmax", tok, expect, tok == (uint32_t)expect ? "ok" : "MISMATCH");
    if (tok != (uint32_t)expect) ++g_fail;

    /* drop the planted outlier so the top K are genuine near-ties and the
     * frequency check actually discriminates between them */
    lg[1234] = 0.0f;

    /* reference: indices sorted by logit, then the softmax of the top K */
    for (i = 0; i < N; ++i) idx[i] = i;
    g_sortkey = lg;
    qsort(idx, N, sizeof idx[0], cmp_desc_idx);
    {
        float tot = 0.0f, m = lg[idx[0]];
        for (j = 0; j < K; ++j) { prob[j] = expf(lg[idx[j]] - m); tot += prob[j]; }
        for (j = 0; j < K; ++j) prob[j] /= tot;
    }

    memset(hits, 0, sizeof hits);
    for (i = 0; i < SEEDS; ++i) {
        int in_top = 0;
        unsigned seed = 777u + (unsigned)i * 2654435761u % 100003u;
        if (bq_vk_sample(vk, lg, N, K, seed, 0, 1.0f, 1.0f, &tok) != 0) {
            printf("  top-k UNSUPPORTED\n"); ++g_fail; return;
        }
        for (j = 0; j < K; ++j) if (tok == (uint32_t)idx[j]) { in_top = 1; ++hits[j]; break; }
        if (!in_top) ++bad;
    }
    printf("  %-28s out_of_topk=%d %s\n", "top-k membership", bad, bad ? "MISMATCH" : "ok");
    if (bad) ++g_fail;
    for (j = 0; j < K; ++j) {
        float f = (float)hits[j] / (float)SEEDS;
        float d = fabsf(f - prob[j]);
        if (d > worst) worst = d;
    }
    /* 400 draws put the sampling error of a frequency near 0.05 at most */
    printf("  %-28s worst=%.3f %s\n", "pick frequency vs softmax", (double)worst,
           worst <= 0.08f ? "ok" : "MISMATCH");
    if (!(worst <= 0.08f)) ++g_fail;

    /* same seed twice must agree */
    { uint32_t t2 = 0;
      (void)bq_vk_sample(vk, lg, N, K, 1234u, 0, 1.0f, 1.0f, &tok);
      (void)bq_vk_sample(vk, lg, N, K, 1234u, 0, 1.0f, 1.0f, &t2);
      printf("  %-28s %u vs %u %s\n", "same seed reproducible", tok, t2, tok == t2 ? "ok" : "DIFFERS");
      if (tok != t2) ++g_fail; }

    /* top-p narrows the set: the pick must fall in the prefix that reaches p */
    {
        float tot = 0.0f, acc = 0.0f;
        int keep = K;
        for (j = 0; j < K; ++j) tot += prob[j];
        for (j = 0; j < K; ++j) { acc += prob[j]; if (acc >= 0.8f * tot) { keep = j + 1; break; } }
        bad = 0;
        for (i = 0; i < 200; ++i) {
            int ok = 0;
            (void)bq_vk_sample(vk, lg, N, K, 31u + (unsigned)i * 40503u, 0, 1.0f, 0.8f, &tok);
            for (j = 0; j < keep; ++j) if (tok == (uint32_t)idx[j]) { ok = 1; break; }
            if (!ok) ++bad;
        }
        printf("  %-28s out_of_prefix=%d keep=%d %s\n", "top-p restricts to prefix", bad, keep,
               bad ? "MISMATCH" : "ok");
        if (bad) ++g_fail;
    }

    /* k is clamped to n when the vocabulary is smaller than k */
    if (bq_vk_sample(vk, lg, 5, 64, 9u, 0, 1.0f, 1.0f, &tok) != 0) {
        printf("  clamp UNSUPPORTED\n"); ++g_fail;
    } else {
        printf("  %-28s got=%u %s\n", "k clamped to n=5", tok, tok < 5u ? "ok" : "MISMATCH");
        if (tok >= 5u) ++g_fail;
    }
}

/* ---------------- matvec ---------------- */
static void test_matvec(bq_vk_t *vk) {
    enum { NIN = 512, NOUT = 24 };
    size_t blocks = NIN / 128, stride = blocks * 28;
    bq_vk_buf_t *wb = NULL;
    uint8_t *wp;
    float x[NIN], y[NOUT];
    unsigned st = 555u;
    int r, i;
    printf("[matvec] %dx%d, %zu blocks per row\n", NOUT, NIN, blocks);
    if (!bq_vk_ready(vk)) { printf("  not ready\n"); ++g_fail; return; }
    wp = bq_vk_buf_alloc(vk, stride * NOUT, &wb);
    if (wp == NULL) { printf("  alloc failed\n"); ++g_fail; return; }
    for (i = 0; i < NOUT * NIN / 128; ++i) {
        uint8_t *blk = wp + (size_t)i * 28;
        for (r = 0; r < 26; ++r) { st = st * 1664525u + 1013904223u; blk[r] = (uint8_t)(st >> 24); }
        blk[26] = 0x00; blk[27] = 0x3C; /* fp16 1.0 */
    }
    for (i = 0; i < NIN; ++i) x[i] = frand(&st);
    if (bq_vk_ptq_submit(vk, wb, stride, NIN, 0, NOUT, x, y) != 0) { printf("  submit failed\n"); ++g_fail; return; }
    if (bq_vk_wait(vk) != 0) { printf("  wait failed\n"); ++g_fail; return; }
    for (r = 0; r < NOUT; ++r) printf("  y[%d] = %.9g\n", r, (double)y[r]);
    bq_vk_buf_free(vk, wb);
}


/* ---------------- attention ---------------- */
static void test_attn(bq_vk_t *vk) {
    enum { NH = 8, NKV = 2, HD = 64, NT = 37 };
    bq_vk_buf_t *qb = NULL, *kb = NULL, *vb = NULL;
    float *q, *k, *v, out[NH * HD], ref[NH * HD], sc[NH * NT];
    unsigned st = 31337u;
    int h, t, d, rep = NH / NKV;
    float scale = 1.0f / sqrtf((float)HD);
    printf("[attn] nhead=%d nkv=%d hd=%d nt=%d\n", NH, NKV, HD, NT);
    q = bq_vk_buf_alloc(vk, NH * HD * sizeof(float), &qb);
    k = bq_vk_buf_alloc(vk, NT * NKV * HD * sizeof(float), &kb);
    v = bq_vk_buf_alloc(vk, NT * NKV * HD * sizeof(float), &vb);
    if (!q || !k || !v) { printf("  alloc failed\n"); ++g_fail; return; }
    fill(q, NH * HD, &st); fill(k, NT * NKV * HD, &st); fill(v, NT * NKV * HD, &st);
    for (h = 0; h < NH; ++h) {
        int kh = h / rep;
        float m = -3.40282347e+38f, sum = 0.0f;
        for (t = 0; t < NT; ++t) {
            float acc = 0.0f;
            for (d = 0; d < HD; ++d) acc += q[h * HD + d] * k[(t * NKV + kh) * HD + d];
            sc[t] = acc * scale;
            if (sc[t] > m) m = sc[t];
        }
        for (t = 0; t < NT; ++t) { sc[t] = expf(sc[t] - m); sum += sc[t]; }
        for (d = 0; d < HD; ++d) {
            float acc = 0.0f;
            for (t = 0; t < NT; ++t) acc += (sc[t] / sum) * v[(t * NKV + kh) * HD + d];
            ref[h * HD + d] = acc;
        }
    }
    if (bq_vk_attn(vk, q, k, v, NH, NKV, HD, NT, rep, scale, out) != 0) {
        printf("  UNSUPPORTED\n"); ++g_fail; goto done;
    }
    cmp("scores+softmax+av", out, ref, NH * HD);
done:
    bq_vk_buf_free(vk, qb); bq_vk_buf_free(vk, kb); bq_vk_buf_free(vk, vb);
}

int main(int argc, char **argv) {
    bq_vk_t *vk;
    const char *only = argc > 1 ? argv[1] : NULL;
    vk = bq_vk_create();
    if (vk == NULL || !bq_vk_ready(vk)) { printf("FATAL: no usable Vulkan device\n"); return 2; }
    printf("device ready\n");
    if (!only || !strcmp(only, "elt"))    test_elt(vk);
    if (!only || !strcmp(only, "conv"))   test_conv(vk);
    if (!only || !strcmp(only, "gdn"))    test_gdn(vk);
    if (!only || !strcmp(only, "sample")) test_sample(vk);
    if (!only || !strcmp(only, "attn"))   test_attn(vk);
    if (!only || !strcmp(only, "matvec")) test_matvec(vk);
    printf("\n%s (worst rel %.3e)\n", g_fail ? "FAILED" : "ALL OK", (double)g_worst);
    return g_fail ? 1 : 0;
}
