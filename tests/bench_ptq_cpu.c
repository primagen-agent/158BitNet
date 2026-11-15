/* CPU PTQ1_0 matmat kernel microbenchmark.
 *
 *   bench_ptq_cpu [n_in] [n_out] [B] [reps]
 *
 * Times bitnet_ptq1_0_matmat_row over a synthetic matrix the same way the
 * block verify calls it (one packed row against B int8 activation rows) and
 * reports the best and median of `reps` passes, since a single pass on a phone
 * is dominated by thermal noise.
 */
#include "quant_ptq1_0.h"

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

static unsigned wst = 99u;

int main(int argc, char **argv) {
    const int n_in = argc > 1 ? atoi(argv[1]) : 5120;
    const int n_out = argc > 2 ? atoi(argv[2]) : 5120;
    const int B = argc > 3 ? atoi(argv[3]) : 3;
    const int reps = argc > 4 ? atoi(argv[4]) : 9;
    const size_t row = bitnet_ptq1_0_row_size(n_in);
    const int nb = n_in / 128;
    uint8_t *W;
    int8_t *xq;
    float *x, xs[16], tmp[16];
    double best = 1e9, *ts;
    int r, b, i, rep;
    double wg = (double)n_out * (double)n_in;

    W = (uint8_t *)malloc(row * (size_t)n_out);
    xq = (int8_t *)malloc((size_t)B * n_in);
    x = (float *)malloc((size_t)B * n_in * sizeof(float));
    ts = (double *)malloc((size_t)reps * sizeof(double));
    if (!W || !xq || !x || !ts) { printf("alloc failed\n"); return 1; }

    for (r = 0; r < n_out; ++r) {
        uint8_t *p = W + (size_t)r * row;
        for (b = 0; b < nb; ++b) {
            uint8_t *blk = p + (size_t)b * BITNET_PTQ1_0_BLOCK_SIZE;
            int k;
            for (k = 0; k < 24; ++k) {
                wst = wst * 1103515245u + 12345u;
                blk[k] = (uint8_t)((wst >> 16) % 243u);
            }
            for (k = 0; k < 2; ++k) {
                wst = wst * 1103515245u + 12345u;
                blk[24 + k] = (uint8_t)((wst >> 8) % 81u);
            }
            {
                /* fp16 0.01..0.03, the scale range real blocks carry */
                uint32_t hv = 0x2400u + ((wst >> 12) & 0x1FFu);
                memcpy(blk + 26, &hv, 2);
            }
        }
    }
    for (i = 0; i < B * n_in; ++i) x[i] = ((float)((i * 2654435761u) >> 8) / 65536.0f) - 1.0f;
    for (b = 0; b < B; ++b)
        bitnet_quantize_row_i8(x + (size_t)b * n_in, xq + (size_t)b * n_in, &xs[b], n_in);

    /* warm the caches and the decode tables */
    for (r = 0; r < n_out; ++r) bitnet_ptq1_0_matmat_row(W + (size_t)r * row, xq, xs, n_in, B, tmp);

    for (rep = 0; rep < reps; ++rep) {
        double t0 = now_sec();
        for (r = 0; r < n_out; ++r)
            bitnet_ptq1_0_matmat_row(W + (size_t)r * row, xq, xs, n_in, B, tmp);
        ts[rep] = now_sec() - t0;
        if (ts[rep] < best) best = ts[rep];
    }
    for (rep = 1; rep < reps; ++rep) {
        double v = ts[rep];
        int j = rep - 1;
        while (j >= 0 && ts[j] > v) { ts[j + 1] = ts[j]; --j; }
        ts[j + 1] = v;
    }
    printf("%dx%d B=%d  min %.3f ms %5.1f G w/s   median %.3f ms %5.1f G w/s\n",
           n_in, n_out, B, 1e3 * best, wg / best / 1e9,
           1e3 * ts[reps / 2], wg / ts[reps / 2] / 1e9);
    free(W); free(xq); free(x); free(ts);
    return 0;
}
