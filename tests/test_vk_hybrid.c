/* CPU-side integration check: the hybrid GPU/CPU row split of a PTQ matvec
 * against the float reference (what BQ_VERIFY does during a real run, without
 * needing the model), and the parallel Prism input transform against the
 * serial one.  Also covers the multi-row kernel's tail rows.
 *
 * Includes src/bonsai.c so the statics under test are reachable. */
#include "../src/bonsai.c"

static int g_fail;
static unsigned g_seed;

static float rnd(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((float)((g_seed >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
}

int main(void) {
    enum { N_IN = 1024, N_OUT = 512, BLOCK = 128, NT = 8 };
    const size_t stride = (size_t)(N_IN / 128) * 28;
    const size_t wbytes = stride * N_OUT;
    bonsai_model_t m;
    bq_w_t w;
    bq_vk_buf_t *wbuf = NULL;
    uint8_t *blob;
    float *x = malloc((size_t)N_IN * sizeof(float));
    float *y = malloc((size_t)N_OUT * sizeof(float));
    float *signs = malloc((size_t)N_IN * sizeof(float));
    float *p1 = malloc((size_t)N_IN * sizeof(float));
    float *p2 = malloc((size_t)N_IN * sizeof(float));
    int i, r, pct;
    float worst = 0.0f;
    int bad = 0;

    printf("[prep] parallel vs serial transform, n=%d block=%d\n", N_IN, BLOCK);
    for (i = 0; i < N_IN; ++i) { x[i] = rnd(); signs[i] = rnd() > 0 ? 1.0f : -1.0f; }
    bq_prep_input(p1, x, N_IN, signs, BLOCK);
    memset(&m, 0, sizeof m);
    m.pool = calloc(1, sizeof(bq_pool_t));
    bq_pool_init(m.pool, NT);
    bq_prep_input_p(m.pool, p2, x, N_IN, signs, BLOCK);
    for (i = 0; i < N_IN; ++i) if (p1[i] != p2[i]) ++bad;
    printf("  bitwise differences: %d %s\n", bad, bad ? "MISMATCH" : "ok");
    if (bad) ++g_fail;
    /* the in-place form must give the same answer */
    memcpy(p2, x, (size_t)N_IN * sizeof(float));
    bq_prep_input_p(m.pool, p2, p2, N_IN, signs, BLOCK);
    for (bad = 0, i = 0; i < N_IN; ++i) if (p1[i] != p2[i]) ++bad;
    printf("  in-place differences: %d %s\n", bad, bad ? "MISMATCH" : "ok");
    if (bad) ++g_fail;

    m.eps = 1e-5f;
    m.vk = bq_vk_create();
    if (m.vk == NULL || !bq_vk_ready(m.vk)) { printf("FATAL: no Vulkan device\n"); return 2; }
    m.gpu_min_rows = 1;

    blob = bq_vk_buf_alloc(m.vk, wbytes, &wbuf);
    if (blob == NULL) { printf("FATAL: weight alloc\n"); return 2; }
    for (i = 0; i < N_OUT * (N_IN / 128); ++i) {
        uint8_t *blk = blob + (size_t)i * 28;
        for (r = 0; r < 26; ++r) blk[r] = (uint8_t)((g_seed = g_seed * 1664525u + 1013904223u) >> 24);
        blk[26] = 0x00; blk[27] = 0x3C;
    }
    for (i = 0; i < N_IN; ++i) x[i] = rnd();

    w.data = blob; w.type = BQ_PTQ; w.n_in = N_IN; w.n_out = N_OUT;
    w.row_stride = stride; w.vk_buf = wbuf;

    {
    const int pcts[] = { 0, 25, 26, 50, 77, 100 };
    int pi;
    for (pi = 0; pi < 6; ++pi) {
        int grow;
        pct = pcts[pi];
        grow = N_OUT * pct / 100;
        m.gpu_pct = pct;
        memset(y, 0, (size_t)N_OUT * sizeof(float));
        bq_matvec(&m, &w, x, y);
        /* the rows the device took, which the share alone does not determine:
         * a narrow batch is handed a smaller share */
        grow = bq_last_gpu_rows;
#ifdef BQ_WAIT_AFTER
        bq_vk_wait(m.vk);
#endif
        if (pct == 0) {
            printf("[matvec] pct=%3d  cpu reference row0=%.6f\n", pct, (double)y[0]);
            continue;
        }
        /* the rows the GPU produced must match the float reference */
        worst = 0.0f; bad = 0;
        for (r = 0; r < grow; ++r) {
            float ref = bq_dot_row(&w, r, x);
            float rel = fabsf(y[r] - ref) / fmaxf(fabsf(ref), 1.0f);
            if (!(rel <= 1e-4f)) ++bad;
            if (rel > worst) worst = rel;
        }
        printf("[matvec] pct=%3d  gpu_rows=%-4d worst_rel=%.3e %s\n",
               pct, grow, (double)worst, bad ? "MISMATCH" : "ok");
        if (bad) ++g_fail;
    }
    }

    bq_vk_buf_free(m.vk, wbuf);
    bq_vk_destroy(m.vk);
    bq_pool_free(m.pool);
    free(m.pool); free(x); free(y); free(signs); free(p1); free(p2);
    printf("\n%s\n", g_fail ? "FAILED" : "ALL OK");
    return g_fail ? 1 : 0;
}
