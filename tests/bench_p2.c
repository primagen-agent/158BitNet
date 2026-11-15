/* The i8mm pairs kernel on its own: full matrix through
 * bitnet_ptq1_0_matmat_p2, single-threaded or over `nt` threads the same way
 * the verify splits output rows.
 *
 *   bench_p2 [n_in] [n_out] [B] [nt] [reps] [two-pair]
 *
 * Reports G weight/s, which is the number the verify's roofline is written in.
 */
#include "quant_ptq1_0.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint8_t *W2;
static float *D, *xs, *Y;
static void *prep;
static int g_n_in, g_n_out, g_B, g_npair, g_q;
static size_t g_pb, g_nb;
static volatile int g_next;

static void *worker(void *p) {
    (void)p;
    for (;;) {
        int j = __atomic_fetch_add(&g_next, g_q ? 4 : 2, __ATOMIC_RELAXED);
        if (j >= g_n_out) break;
        if (g_q && j + 3 < g_n_out) {
            bitnet_ptq1_0_matmat_p2q(W2 + (size_t)(j / 2) * g_pb,
                                     D + (size_t)(j / 2) * g_nb * 2u,
                                     prep, xs, g_B, g_n_out, j, Y);
            continue;
        }
        bitnet_ptq1_0_matmat_p2(W2 + (size_t)(j / 2) * g_pb,
                                D + (size_t)(j / 2) * g_nb * 2u,
                                prep, xs, g_B, g_n_out, j, Y);
    }
    return NULL;
}

int main(int argc, char **argv) {
    const int n_in = argc > 1 ? atoi(argv[1]) : 5120;
    const int n_out = argc > 2 ? atoi(argv[2]) : 5120;
    const int B = argc > 3 ? atoi(argv[3]) : 2;
    const int nt = argc > 4 ? atoi(argv[4]) : 1;
    const int reps = argc > 5 ? atoi(argv[5]) : 9;
    const size_t row = bitnet_ptq1_0_row_size(n_in);
    const int nb = n_in / 128;
    const int npair = n_out / 2;
    uint8_t *W;
    int8_t *xq;
    float *x, tmp[16], best = 1e9f, med;
    double *ts;
    int r, b, i, rep;
    double wg = (double)n_out * (double)n_in;
    pthread_t th[32];

    if (!bitnet_ptq1_0_p2_available()) { printf("no i8mm\n"); return 1; }
    if (nt > 32 || n_out % 2 != 0) { printf("bad args\n"); return 1; }
    W = (uint8_t *)malloc(row * (size_t)n_out);
    W2 = (uint8_t *)malloc(bitnet_ptq1_0_p2_pair_bytes(n_in) * (size_t)npair);
    D = (float *)malloc(sizeof(float) * (size_t)npair * (size_t)nb * 2u);
    xq = (int8_t *)malloc((size_t)B * n_in);
    x = (float *)malloc((size_t)B * n_in * sizeof(float));
    xs = (float *)malloc((size_t)sizeof(float) * (size_t)B);
    Y = (float *)malloc(sizeof(float) * (size_t)B * (size_t)n_out);
    prep = malloc(bitnet_ptq1_0_p2_prep_bytes(n_in, B));
    ts = (double *)malloc((size_t)reps * sizeof(double));
    if (!W || !W2 || !D || !xq || !x || !xs || !Y || !prep || !ts) { printf("alloc failed\n"); return 1; }
    for (i = 0; i < B * n_in; ++i) x[i] = ((float)((i * 2654435761u) >> 8) / 65536.0f) - 1.0f;
    for (b = 0; b < B; ++b)
        bitnet_quantize_row_i8(x + (size_t)b * n_in, xq + (size_t)b * n_in, &xs[b], n_in);
    for (r = 0; r < n_out; ++r)
        if (bitnet_ptq1_0_quantize_row(x, W + (size_t)r * row, n_in) != 0) { printf("quantise failed\n"); return 1; }
    for (r = 0; r + 1 < n_out; r += 2)
        bitnet_ptq1_0_p2_pack(W + (size_t)r * row, W + (size_t)(r + 1) * row, n_in,
                              W2 + (size_t)(r / 2) * bitnet_ptq1_0_p2_pair_bytes(n_in),
                              D + (size_t)(r / 2) * (size_t)nb * 2u);
    bitnet_ptq1_0_p2_prep(prep, xq, n_in, B);

    g_q = argc > 6 ? atoi(argv[6]) : 0;
    g_n_in = n_in; g_n_out = n_out; g_B = B; g_npair = npair;
    g_pb = bitnet_ptq1_0_p2_pair_bytes(n_in);
    g_nb = (size_t)nb;

    for (rep = 0; rep < reps; ++rep) {
        double t0 = now_sec();
        if (nt == 1) {
            for (r = 0; r + 1 < n_out; r += 2) {
                if (g_q && r + 3 < n_out) {
                    bitnet_ptq1_0_matmat_p2q(W2 + (size_t)(r / 2) * g_pb,
                                             D + (size_t)(r / 2) * g_nb * 2u,
                                             prep, xs, B, n_out, r, Y);
                    ++r;
                    continue;
                }
                bitnet_ptq1_0_matmat_p2(W2 + (size_t)(r / 2) * g_pb,
                                        D + (size_t)(r / 2) * g_nb * 2u,
                                        prep, xs, B, n_out, r, Y);
            }
        } else {
            int k;
            g_next = 0;
            for (k = 0; k < nt; ++k) pthread_create(&th[k], NULL, worker, NULL);
            for (k = 0; k < nt; ++k) pthread_join(th[k], NULL);
        }
        ts[rep] = now_sec() - t0;
        if (ts[rep] < best) best = (float)ts[rep];
    }
    med = (float)ts[0];
    {
        int j, k;
        for (j = 1; j < reps; ++j) {
            double v = ts[j];
            k = j - 1;
            while (k >= 0 && ts[k] > v) { ts[k + 1] = ts[k]; --k; }
            ts[k + 1] = v;
        }
        med = (float)ts[reps / 2];
    }
    tmp[0] = Y[0];
    printf("%dx%d B=%d nt=%d  min %.3f ms %5.1f G w/s   median %.3f ms %5.1f G w/s (sink %.3f)\n",
           n_in, n_out, B, nt, 1e3 * best, wg / best / 1e9, 1e3 * med, wg / med / 1e9, tmp[0]);
    free(W); free(xq); free(x); free(ts);
    return 0;
}
