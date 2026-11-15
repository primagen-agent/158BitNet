/* The 2-bit i8mm repack has to agree with the 28-byte path exactly: both
 * accumulate each 128-trit block in int32 and only differ in the instruction
 * that forms the products.  Skipped on machines without i8mm. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "quant_ptq1_0.h"

#define N 512
#define N_OUT 6

static float frand(unsigned *s) {
    *s = *s * 1664525u + 1013904223u;
    return ((float)(int)(*s >> 8) - 8388608.0f) / 8388608.0f;
}

static uint16_t to_bf16(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return (uint16_t)(b >> 16);
}

int main(void) {
    static uint16_t bf[N];
    static float xf[16][N], xs[16];
    static int8_t xq[16 * N];   /* one row per token, n int8 values each */
    unsigned char rows[N_OUT * 128 * BITNET_PTQ1_0_BLOCK_SIZE];
    const size_t rs = bitnet_ptq1_0_row_size(N);
    unsigned seed = 12345u;
    int i, r, b, failed = 0;
    static const int bs[] = {1, 2, 3, 4, 5, 6, 8};

    if (!bitnet_ptq1_0_p2_available()) {
        printf("skipped: no i8mm on this CPU\n");
        return 0;
    }
    if (rs * N_OUT > sizeof(rows)) { printf("row buffer too small\n"); return 1; }

    for (r = 0; r < N_OUT; ++r) {
        for (i = 0; i < N; ++i) bf[i] = to_bf16(frand(&seed));
        if (bitnet_ptq1_0_quantize_row(bf, rows + (size_t)r * rs, N) != 0) {
            printf("encode failed\n");
            return 1;
        }
    }
    for (b = 0; b < 16; ++b) {
        for (i = 0; i < N; ++i) xf[b][i] = 2.0f * frand(&seed);
        bitnet_quantize_row_i8(xf[b], xq + (size_t)b * N, &xs[b], N);
    }

    for (i = 0; i < (int)(sizeof(bs) / sizeof(bs[0])); ++i) {
        const int B = bs[i];
        const int nb = N / 128, pairs = (B + 1) / 2;
        float got[16 * N_OUT];
        double worst = 0.0;
        void *prep;
        unsigned char *p2w;
        float *p2d;

        p2w = (unsigned char *)malloc(bitnet_ptq1_0_p2_pair_bytes(N) * (size_t)(N_OUT / 2));
        p2d = (float *)malloc((size_t)(N_OUT / 2) * (size_t)nb * 2u * sizeof(float));
        prep = malloc(bitnet_ptq1_0_p2_prep_bytes(N, B));
        if (p2w == NULL || p2d == NULL || prep == NULL) { printf("oom\n"); return 1; }

        for (r = 0; r < N_OUT; r += 2)
            bitnet_ptq1_0_p2_pack(rows + (size_t)r * rs, rows + (size_t)(r + 1) * rs, N,
                                  p2w + (size_t)(r / 2) * bitnet_ptq1_0_p2_pair_bytes(N),
                                  p2d + (size_t)(r / 2) * (size_t)nb * 2u);
        /* every token sees its own block sums, so prep once per B */
        bitnet_ptq1_0_p2_prep(prep, xq, N, B);
        memset(got, 0, sizeof(got));
        for (r = 0; r < N_OUT; r += 2)
            bitnet_ptq1_0_matmat_p2(p2w + (size_t)(r / 2) * bitnet_ptq1_0_p2_pair_bytes(N),
                                    p2d + (size_t)(r / 2) * (size_t)nb * 2u, prep, xs, B,
                                    N_OUT, r, got);

        for (b = 0; b < B; ++b)
            for (r = 0; r < N_OUT; ++r) {
                float ref[16];
                double d;
                bitnet_ptq1_0_matmat_row(rows + (size_t)r * rs, xq + (size_t)b * N,
                                         xs + b, N, 1, ref);
                d = fabs((double)ref[0] - got[(size_t)b * N_OUT + r]);
                d /= fabs((double)ref[0]) > 1e-3 ? fabs((double)ref[0]) : 1e-3;
                if (d > worst) worst = d;
            }
        /* pairs=%d keeps the boundary cases (odd B, B>2) in play */
        printf("B=%d pairs=%d  worst relative difference %.3e  %s\n",
               B, pairs, worst, worst < 1e-5 ? "ok" : "MISMATCH");
        if (!(worst < 1e-5)) failed = 1;

        /* the two-pair kernel must agree with the one-pair one row for row */
        if (pairs == 1) {
            float got2[16 * N_OUT];
            double w2 = 0.0;
            memset(got2, 0, sizeof(got2));
            for (r = 0; r + 3 < N_OUT; r += 4)
                bitnet_ptq1_0_matmat_p2q(p2w + (size_t)(r / 2) * bitnet_ptq1_0_p2_pair_bytes(N),
                                         p2d + (size_t)(r / 2) * (size_t)nb * 2u, prep, xs, B,
                                         N_OUT, r, got2);
            for (b = 0; b < B; ++b)
                for (r = 0; r + 3 < N_OUT; ++r) {
                    double d = fabs((double)got[(size_t)b * N_OUT + r] -
                                    (double)got2[(size_t)b * N_OUT + r]);
                    if (d > w2) w2 = d;
                }
            printf("B=%d two-pair  worst absolute difference %.3e  %s\n",
                   B, w2, w2 < 1e-4 ? "ok" : "MISMATCH");
            if (!(w2 < 1e-4)) failed = 1;
        }
        free(prep);
        free(p2d);
        free(p2w);
    }
    return failed;
}
