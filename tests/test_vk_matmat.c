/* Correctness and speed of the PTQ1_0 matmat kernel -- the block verify's
 * weight pass, where one decoded row of trits is dotted against B activation
 * rows.  Uses synthetic ternary matrices (the bytes are what the kernel
 * decodes, so random ones exercise the same work as real weights) and
 * compares against the CPU batched kernel.
 *
 *   test_vk_matmat [-r reps]
 */
#include "vk_backend.h"
#include "quant_ptq1_0.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float frand(unsigned *st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* fp16 of a small positive scale, round to nearest */
static uint16_t f2h(float f) {
    uint32_t x;
    uint16_t s = 0;
    int e;
    uint32_t m;
    memcpy(&x, &f, sizeof(x));
    s = (uint16_t)((x >> 16) & 0x8000u);
    e = (int)((x >> 23) & 0xFFu) - 127 + 15;
    m = (x >> 13) & 0x3FFu;
    if (e <= 0) return s;
    if (e >= 31) return (uint16_t)(s | 0x7C00u);
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

#define FAIL(...) do { printf("FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(int argc, char **argv) {
    static const struct { int n_in, n_out; } shapes[] = {
        { 5120, 5120 }, { 5120, 17408 }, { 17408, 5120 }
    };
    static const int Bs[] = { 1, 3, 4, 8, 16 };
    int reps = 20, si, bi, i, rc = 0, pct = 100;
    bq_vk_t *vk;
    unsigned st = 7u;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        /* -p N dispatches only the first N% of the rows, which is the shape
         * the hybrid verify actually gives the GPU (BQ_MM_GPU). */
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) pct = atoi(argv[++i]);
    }
    vk = bq_vk_create();
    if (vk == NULL || !bq_vk_ready(vk)) FAIL("no Vulkan device");
    if (!bq_vk_ptq_matmat_available(vk)) FAIL("matmat pipeline unavailable");
    printf("matmat available, reps=%d, rows %.0f%%\n", reps, (double)pct);

    for (si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); ++si) {
        const int n_in = shapes[si].n_in;
        const int n_full = shapes[si].n_out;
        const int n_out = pct >= 100 ? n_full : n_full * pct / 100;
        const size_t row = bitnet_ptq1_0_row_size(n_in);
        const int nb = n_in / 128;
        uint8_t *W = (uint8_t *)malloc(row * (size_t)n_full);
        int8_t *xq = (int8_t *)malloc((size_t)n_in * 16);
        int8_t *xg = (int8_t *)malloc((size_t)n_in * 16);
        float *x = (float *)malloc((size_t)n_in * 16 * sizeof(float));
        float *ys = (float *)malloc((size_t)n_out * 16 * sizeof(float));
        float *yr = (float *)malloc((size_t)n_out * 16 * sizeof(float));
        float *xs = (float *)malloc(16 * sizeof(float));
        float *xsg = (float *)malloc(16 * sizeof(float));
        int8_t *chk = (int8_t *)malloc((size_t)n_in);
        bq_vk_buf_t *wb = NULL;
        unsigned wst = 99u;
        int r;

        if (!W || !xq || !xg || !x || !ys || !yr || !xs || !xsg || !chk) FAIL("alloc");
        for (r = 0; r < n_full; ++r) {
            uint8_t *p = W + (size_t)r * row;
            int b;
            for (b = 0; b < nb; ++b) {
                uint8_t *blk = p + (size_t)b * 28;
                int k;
                /* 24 low bytes hold five trits each (value < 243), the two
                 * high bytes four each (value < 81) */
                for (k = 0; k < 24; ++k) blk[k] = (uint8_t)(wst = wst * 1103515245u + 12345u, (wst >> 16) % 243u);
                for (k = 0; k < 2; ++k) blk[24 + k] = (uint8_t)((wst >> 8) % 81u);
                {
                    uint16_t d = f2h(0.01f + 0.02f * (float)((wst >> 12) & 0xFFu) / 255.0f);
                    memcpy(blk + 26, &d, 2);
                }
            }
        }
        {
            /* every buffer here is host visible, so loading the weights is a
             * plain memcpy into the returned pointer */
            void *wptr = bq_vk_buf_alloc(vk, row * (size_t)n_full, &wb);
            if (wptr == NULL) FAIL("gpu weight alloc");
            memcpy(wptr, W, row * (size_t)n_full);
        }

        printf("\n[%dx%d rows=%d]\n", n_out, n_in, n_out);
        for (bi = 0; bi < (int)(sizeof(Bs) / sizeof(Bs[0])); ++bi) {
            const int B = Bs[bi];
            double t_cpu, t_gpu, tg_avg_last = 0.0;
            double wg = (double)n_out * (double)n_in;
            int b;
            for (i = 0; i < n_in * B; ++i) x[i] = frand(&st);
            for (b = 0; b < B; ++b) {
                bitnet_quantize_row_i8(x + (size_t)b * n_in, xq + (size_t)b * n_in,
                                       &xs[b], n_in);
                bitnet_quantize_row_i8_groups(x + (size_t)b * n_in, xg + (size_t)b * n_in,
                                              &xsg[b], n_in);
                /* both packings must hold the same values */
                bitnet_unpack_row_i8_groups(xg + (size_t)b * n_in, ys, xsg[b], n_in);
                for (i = 0; i < n_in; ++i) {
                    float want = (float)xq[(size_t)b * n_in + i] * xs[b];
                    if (fabsf(ys[i] - want) > 1e-6f * (1.0f + fabsf(want))) {
                        printf("FAIL: packing mismatch b=%d i=%d got %g want %g\n",
                               b, i, (double)ys[i], (double)want);
                        return 1;
                    }
                }
            }

            /* reference: CPU batched kernel over the whole matrix, scattered
             * into [b][row] as the kernel writes it */
            t_cpu = now_sec();
            for (r = 0; r < n_out; ++r) {
                float tmp[16];
                bitnet_ptq1_0_matmat_row(W + (size_t)r * row, xq, xs, n_in, B, tmp);
                for (b = 0; b < B; ++b) yr[(size_t)b * n_out + r] = tmp[b];
            }
            t_cpu = now_sec() - t_cpu;

            t_gpu = now_sec();
            if (bq_vk_ptq_matmat_begin(vk, xg, xsg, n_in, B, n_out) != 0) FAIL("begin");
            if (bq_vk_ptq_matmat_add(vk, wb, row, 0, n_out, n_out, ys) != 0) FAIL("add");
            if (bq_vk_ptq_matmat_end(vk) != 0) FAIL("end");
            if (bq_vk_wait(vk) != 0) FAIL("wait");
            t_gpu = now_sec() - t_gpu;

            {
                float worst = 0.0f;
                int bad = 0;
                for (i = 0; i < n_out * B; ++i) {
                    float m = fabsf(yr[i]) > 1.0f ? fabsf(yr[i]) : 1.0f;
                    float rel = fabsf(ys[i] - yr[i]) / m;
                    if (!(rel <= 2e-4f)) ++bad;
                    if (rel > worst) worst = rel;
                }
                printf("  B=%-2d gpu worst_rel=%.2e %-9s ", B, (double)worst,
                       bad ? "MISMATCH" : "ok");
                if (bad) { ++rc; }
            }
            /* timing: only meaningful when the result matched */
            {
                double tg = 0.0, tc = 0.0;
                int rep;
                for (rep = 0; rep < reps; ++rep) {
                    double t0 = now_sec();
                    if (bq_vk_ptq_matmat_begin(vk, xg, xsg, n_in, B, n_out) != 0) FAIL("begin2");
                    if (bq_vk_ptq_matmat_add(vk, wb, row, 0, n_out, n_out, ys) != 0) FAIL("add2");
                    if (bq_vk_ptq_matmat_end(vk) != 0) FAIL("end2");
                    if (bq_vk_wait(vk) != 0) FAIL("wait2");
                    tg += now_sec() - t0;
                }
                (void)tc;
                tg /= (double)reps;
                tg_avg_last = tg;
                printf("gpu %7.1f ms %6.1f G w/s (first %.1f ms)  ",
                       1e3 * tg, wg / tg / 1e9, 1e3 * t_gpu);
            }
            {
                double tg_avg = tg_avg_last;
                printf("cpu %7.1f ms %6.1f G w/s  speedup %.2fx\n", 1e3 * t_cpu,
                       wg / t_cpu / 1e9, tg_avg > 0.0 ? t_cpu / tg_avg : 0.0);
            }
        }
        bq_vk_buf_free(vk, wb);
        free(W); free(xq); free(xg); free(x); free(ys); free(yr); free(xs); free(xsg);
        free(chk);
    }
    bq_vk_destroy(vk);
    return rc ? 1 : 0;
}
