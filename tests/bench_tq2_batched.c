/* TQ2_0 block verify: one weight pass against B activation rows.
 *
 *   bench_tq2_batched [-r reps] [-t threads]
 *
 * Compares the batched kernel (bitnet_tq2_0_matmul_batched_i8) against B
 * passes of the single-row kernel on the BitCPM-8B shapes.  A speculative
 * verify needs the first number: with it, a round costs one read of the
 * matrix plus B dot products; without it, B full reads.
 */
#include "quant_tq2_0.h"
#include "thread_config.h"

#include <math.h>
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

static uint16_t f2h(float f) {
    uint32_t x, m;
    int e;
    memcpy(&x, &f, sizeof(x));
    uint16_t s = (uint16_t)((x >> 16) & 0x8000u);
    e = (int)((x >> 23) & 0xFFu) - 127 + 15;
    m = (x >> 13) & 0x3FFu;
    if (e <= 0) return s;
    if (e >= 31) return (uint16_t)(s | 0x7C00u);
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

static float frand(unsigned *st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
}

int main(int argc, char **argv) {
    static const struct { int n_in, n_out; const char *name; } shapes[] = {
        { 4096, 16384, "ffn_up" },
        { 16384, 4096, "ffn_down" },
        { 4096, 4096, "attn_q" },
    };
    static const int Bs[] = { 1, 3, 4, 6, 8 };
    int reps = 10, i, si, bi;
    unsigned st = 11u;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%d", atoi(argv[++i]));
            setenv("BITNET_NUM_THREADS", buf, 1);   /* read once, before any pool */
        }
    }
    printf("threads=%d reps=%d\n", bitnet_thread_count(0), reps);

    for (si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); ++si) {
        const int n_in = shapes[si].n_in, n_out = shapes[si].n_out;
        const int nb = n_in / BITNET_TQ2_0_QK;
        const size_t row = (size_t)nb * BITNET_TQ2_0_BLOCK_SIZE;
        uint8_t *W = (uint8_t *)malloc(row * (size_t)n_out);
        float *scales = (float *)malloc((size_t)n_out * (size_t)nb * sizeof(float));
        int8_t *xq = (int8_t *)malloc((size_t)n_in * 16);
        float *xs = (float *)malloc(16 * sizeof(float));
        int32_t *bs = (int32_t *)malloc((size_t)nb * 16 * sizeof(int32_t));
        float *x = (float *)malloc((size_t)n_in * 16 * sizeof(float));
        float *yb = (float *)malloc((size_t)n_out * 16 * sizeof(float));
        float *yr = (float *)malloc((size_t)n_out * 16 * sizeof(float));
        int r, b;

        if (!W || !scales || !xq || !xs || !bs || !x || !yb || !yr) { printf("alloc\n"); return 1; }
        for (r = 0; r < n_out; ++r) {
            uint8_t *p = W + (size_t)r * row;
            for (b = 0; b < nb; ++b) {
                uint8_t *blk = p + (size_t)b * BITNET_TQ2_0_BLOCK_SIZE;
                uint16_t d = f2h(0.01f + 0.02f * (float)((st = st * 1103515245u + 12345u, (st >> 16) & 0xFFu)) / 255.0f);
                for (i = 0; i < BITNET_TQ2_0_QS_SIZE; ++i) blk[i] = (uint8_t)(st = st * 1103515245u + 12345u, st >> 17);
                memcpy(blk + BITNET_TQ2_0_QS_SIZE, &d, 2);
            }
        }
        if (bitnet_tq2_0_build_scales(W, n_out, n_in, scales,
                                      (size_t)n_out * (size_t)nb) != 0) { printf("scales\n"); return 1; }

        printf("\n[%s %dx%d]\n", shapes[si].name, n_out, n_in);
        for (b = 0; b < 16; ++b) {
            for (i = 0; i < n_in; ++i) x[(size_t)b * n_in + i] = frand(&st);
            if (bitnet_tq2_0_quantize_vec_i8(x + (size_t)b * n_in, n_in,
                                             xq + (size_t)b * n_in, &xs[b],
                                             bs + (size_t)b * nb) != 0) { printf("quant\n"); return 1; }
        }

        for (bi = 0; bi < (int)(sizeof(Bs) / sizeof(Bs[0])); ++bi) {
            const int B = Bs[bi];
            const double wg = (double)n_out * (double)n_in;
            double t_b = 0.0, t_s = 0.0;
            float worst = 0.0f;
            int bad = 0;

            memset(yb, 0, (size_t)n_out * B * sizeof(float));
            memset(yr, 0, (size_t)n_out * B * sizeof(float));
            if (bitnet_tq2_0_matmul_batched_i8(W, scales, n_out, n_in, xq, xs, bs, B, yb) != 0) {
                printf("batched failed\n"); return 1;
            }
            for (b = 0; b < B; ++b) {
                if (bitnet_tq2_0_matmul_vector_i8_neon_scales(
                        W, scales, n_out, n_in, xq + (size_t)b * n_in, xs[b],
                        bs + (size_t)b * nb, yr + (size_t)b * n_out) != 0) {
                    printf("single failed\n"); return 1;
                }
            }
            for (i = 0; i < n_out * B; ++i) {
                float m = fabsf(yr[i]) > 1.0f ? fabsf(yr[i]) : 1.0f;
                float rel = fabsf(yb[i] - yr[i]) / m;
                if (!(rel <= 2e-5f)) ++bad;
                if (rel > worst) worst = rel;
            }

            {
                /* Alternate the two forms inside the loop and keep the best
                 * time of each: the big cores throttle over a sweep and a
                 * mean would charge whichever form ran later. */
                double tb = 1e9, ts = 1e9;
                int rep;
                for (rep = 0; rep < reps; ++rep) {
                    double t0 = now_sec();
                    bitnet_tq2_0_matmul_batched_i8(W, scales, n_out, n_in, xq, xs, bs, B, yb);
                    t0 = now_sec() - t0;
                    if (t0 < tb) tb = t0;
                    t0 = now_sec();
                    for (b = 0; b < B; ++b)
                        bitnet_tq2_0_matmul_vector_i8_neon_scales(
                            W, scales, n_out, n_in, xq + (size_t)b * n_in, xs[b],
                            bs + (size_t)b * nb, yr + (size_t)b * n_out);
                    t0 = now_sec() - t0;
                    if (t0 < ts) ts = t0;
                }
                t_b = tb;
                t_s = ts;
            }
            printf("  B=%-2d %-9s worst=%.1e  batched %7.1f ms %6.1f G w/s (%.1f ms/row)  "
                   "B passes %7.1f ms %6.1f G w/s  speedup %.2fx\n",
                   B, bad ? "MISMATCH" : "ok", (double)worst, 1e3 * t_b, wg / t_b / 1e9,
                   1e3 * t_b / (double)B, 1e3 * t_s, wg / t_s / 1e9, t_s / t_b);
        }
        free(W); free(scales); free(xq); free(xs); free(bs); free(x); free(yb); free(yr);
    }
    return 0;
}
