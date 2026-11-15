/* Throughput bench and correctness check for the TQ2_0 GPU matmat.
 *
 *   test_vk_tq2_matmat_bench [-i N_IN] [-n MAX_ROWS] [-B MAXB] [-r REPEATS]
 *
 * Builds one synthetic TQ2_0 matrix of MAX_ROWS x N_IN in the stored 66-byte
 * block layout, checks the GPU's B-row windows against a CPU decode of the
 * same bytes, then times B = 1..MAXB and a row sweep through
 * bq_vk_tq2_matmat_{begin,add,end} + bq_vk_wait.  The rate that matters for a
 * speculative verify is the MAC rate (rows x B x n_in against seconds); the
 * one-row weight rate is printed beside it so the decode-sharing gain over
 * the matvec is visible directly. */

#include "vk/vk_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float half_to_float(unsigned h) {
    unsigned s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, f = h & 0x3FFu;
    unsigned bits;
    if (e == 0) bits = (s << 31) | (f ? ((127u - 15u + 1u) << 23) | (f << 13) : 0u);
    else bits = (s << 31) | ((e - 15u + 127u) << 23) | (f << 13);
    {
        float out;
        memcpy(&out, &bits, sizeof out);
        return out;
    }
}

static float tq2_weight(const unsigned char *blk, int p) {
    unsigned byte = blk[(p >> 7) * 32 + (p & 31)];
    return (float)((byte >> (((p >> 5) & 3) * 2)) & 3u) - 1.0f;
}

/* The activation quantization the runtime applies before handing the GPU its
 * rows (bitnet_quantize_row_i8): one symmetric int8 scale per row. */
static void quant_row(const float *x, int8_t *dst, float *scale, int n) {
    float amax = 0.0f, s;
    int i;
    for (i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    s = amax / 127.0f;
    if (!(s > 1e-8f)) s = 1e-8f;
    *scale = s;
    for (i = 0; i < n; ++i) {
        int v = (int)lrintf(x[i] / s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        dst[i] = (int8_t)v;
    }
}

/* CPU reference: y[b] = scale_b * sum over blocks of d * sum_p w(p) * q[p],
 * i.e. the same int8 integers the DP4A kernel multiplies. */
static double ref_row(const unsigned char *wp, int nb, const int8_t *xq, int n_in) {
    int b, p;
    double sum = 0.0;
    for (b = 0; b < nb; ++b) {
        const unsigned char *blk = wp + (size_t)b * 66u;
        float d = half_to_float((unsigned)blk[64] | ((unsigned)blk[65] << 8));
        double s = 0.0;
        const int8_t *x = xq + (size_t)b * 256u;
        for (p = 0; p < 256; ++p) s += (double)tq2_weight(blk, p) * (double)x[p];
        sum += s * (double)d;
    }
    (void)n_in;
    return sum;
}

int main(int argc, char **argv) {
    int n_in = 4096, max_rows = 16384, maxB = 8, reps = 3, i, r, b, k;
    size_t wbytes, rb;
    int nb = n_in / 256;
    bq_vk_t *vk;
    bq_vk_buf_t *wb = NULL;
    unsigned char *wp;
    float *x, *y;
    int8_t *xq;
    float xscale[8];
    unsigned st = 12345u;
    int rsweep[] = { 1024, 4096, 8192, 16384, 32768, 65536 };
    int nrsweep = (int)(sizeof rsweep / sizeof rsweep[0]);

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "-i") && i + 1 < argc) n_in = atoi(argv[++i]);
        else if (!strcmp(a, "-n") && i + 1 < argc) max_rows = atoi(argv[++i]);
        else if (!strcmp(a, "-B") && i + 1 < argc) maxB = atoi(argv[++i]);
        else if (!strcmp(a, "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-i N_IN] [-n MAX_ROWS] [-B MAXB] [-r REPS]\n", argv[0]);
            return 2;
        }
    }
    if (n_in % 256) { fprintf(stderr, "n_in must be a multiple of 256\n"); return 2; }
    if (maxB < 1 || maxB > 8) maxB = 8;
    nb = n_in / 256;

    vk = bq_vk_create();
    if (vk == NULL || !bq_vk_ready(vk)) { printf("FATAL: no usable Vulkan device\n"); return 2; }
    if (!bq_vk_tq2_matmat_available(vk)) { printf("FATAL: tq2 matmat unavailable\n"); return 2; }

    rb = (size_t)nb * 66u;
    wbytes = rb * (size_t)max_rows;
    wp = bq_vk_buf_alloc(vk, wbytes, &wb);
    if (wp == NULL) { printf("FATAL: weight alloc of %zu bytes failed\n", wbytes); return 2; }
    x = malloc((size_t)n_in * (size_t)maxB * sizeof(float));
    y = malloc((size_t)max_rows * (size_t)maxB * sizeof(float));
    xq = malloc((size_t)n_in * (size_t)maxB);
    if (!x || !y || !xq) { printf("FATAL: host alloc failed\n"); return 2; }

    for (i = 0; i < n_in * maxB; ++i) {
        st = st * 1664525u + 1013904223u;
        x[i] = (float)((int)(st >> 24) - 128) / 128.0f;
    }
    for (b = 0; b < maxB; ++b)
        quant_row(x + (size_t)b * (size_t)n_in, xq + (size_t)b * (size_t)n_in,
                  &xscale[b], n_in);
    for (r = 0; r < max_rows; ++r) {
        for (b = 0; b < nb; ++b) {
            unsigned char *blk = wp + (size_t)r * rb + (size_t)b * 66u;
            unsigned h;
            for (i = 0; i < 64; ++i) {
                st = st * 1664525u + 1013904223u;
                blk[i] = (unsigned char)(st >> 24);
            }
            h = 0x3800u + (unsigned)((r * 7 + b * 3) % 0x400u);   /* 0.5 .. 1.0 */
            blk[64] = (unsigned char)(h & 0xFFu);
            blk[65] = (unsigned char)(h >> 8);
        }
    }

    /* correctness: a few row windows at each B, decoded on the CPU */
    {
        int probe[] = { 0, 1, 2, 17, max_rows / 2, max_rows - 3 };
        int np = (int)(sizeof probe / sizeof probe[0]), bad = 0, bb;
        for (bb = 1; bb <= maxB; ++bb) {
            if (bq_vk_tq2_matmat_begin(vk, xq, xscale, n_in, bb, max_rows) != 0) {
                printf("FATAL: begin (B=%d)\n", bb); return 2;
            }
            for (i = 0; i < np; ++i) {
                int row = probe[i];
                if (bq_vk_tq2_matmat_add(vk, wb, row, row, row + 1, max_rows, y) != 0) {
                    printf("FATAL: add (B=%d)\n", bb); return 2;
                }
            }
            if (bq_vk_tq2_matmat_end(vk) != 0 || bq_vk_wait(vk) != 0) {
                printf("FATAL: submit (B=%d)\n", bb); return 2;
            }
            for (i = 0; i < np; ++i) {
                int row = probe[i];
                for (b = 0; b < bb; ++b) {
                    double want = ref_row(wp + (size_t)row * rb, nb,
                                          xq + (size_t)b * (size_t)n_in, n_in)
                                  * (double)xscale[b];
                    double got = y[(size_t)b * (size_t)max_rows + row];
                    double rel = fabs(want) > 1e-9 ? fabs(got - want) / fabs(want)
                                                   : fabs(got - want);
                    if (rel > 1e-3) {
                        printf("B=%d row %6d b %d gpu %.9g cpu %.9g rel %.2e  MISMATCH\n",
                               bb, row, b, got, want, rel);
                        ++bad;
                    }
                }
            }
        }
        if (bad) { printf("FATAL: %d results mismatch, bench data unusable\n", bad); return 1; }
        printf("correctness: %d row windows x B=1..%d agree with the CPU decode\n", np, maxB);
    }

    /* A tail slice, the shape the model runtime stages: the buffer holds only
     * rows [n0, max_rows) yet the results must land at their real row index.
     * The CPU path then fills [0, n0) in the same output array. */
    {
        int n0 = max_rows - max_rows / 3;
        size_t slice_bytes = (size_t)(max_rows - n0) * rb;
        bq_vk_buf_t *swb = NULL;
        unsigned char *sp = bq_vk_buf_alloc(vk, slice_bytes, &swb);
        int bad = 0, bb;
        if (sp == NULL) { printf("FATAL: slice alloc\n"); return 2; }
        memcpy(sp, wp + (size_t)n0 * rb, slice_bytes);
        for (bb = 2; bb <= maxB; bb += 3) {
            memset(y, 0, (size_t)max_rows * (size_t)maxB * sizeof(float));
            if (bq_vk_tq2_matmat_begin(vk, xq, xscale, n_in, bb, max_rows) != 0 ||
                bq_vk_tq2_matmat_add(vk, swb, 0, n0, max_rows, max_rows, y) != 0 ||
                bq_vk_tq2_matmat_end(vk) != 0 || bq_vk_wait(vk) != 0) {
                printf("FATAL: slice submit (B=%d)\n", bb); return 2;
            }
            for (r = n0; r < max_rows; ++r)
                for (b = 0; b < bb; ++b) {
                    double want = ref_row(wp + (size_t)r * rb, nb,
                                          xq + (size_t)b * (size_t)n_in, n_in)
                                  * (double)xscale[b];
                    double got = y[(size_t)b * (size_t)max_rows + r];
                    double rel = fabs(want) > 1e-9 ? fabs(got - want) / fabs(want)
                                                   : fabs(got - want);
                    if (rel > 1e-3) ++bad;
                }
        }
        bq_vk_buf_free(vk, swb);
        if (bad) { printf("FATAL: %d tail-slice results mismatch\n", bad); return 1; }
        printf("tail slice: rows [%d,%d) land at their own index for B>=2\n", n0, max_rows);
    }

    /* warm pass, so the sweep does not start at a low GPU clock */
    if (bq_vk_tq2_matmat_begin(vk, xq, xscale, n_in, maxB, max_rows) == 0 &&
        bq_vk_tq2_matmat_add(vk, wb, 0, 0, max_rows, max_rows, y) == 0 &&
        bq_vk_tq2_matmat_end(vk) == 0) (void)bq_vk_wait(vk);

    printf("\nB sweep at %d rows (%d x %d):\n", max_rows, max_rows, n_in);
    printf("%3s %10s %11s %11s %11s %11s\n",
           "B", "ms", "G w/s", "G MAC/s", "us/row", "GB/s");
    for (b = 1; b <= maxB; ++b) {
        double best = 1e9;
        for (k = 0; k < reps; ++k) {
            double t0, dt;
            t0 = now_sec();
            if (bq_vk_tq2_matmat_begin(vk, xq, xscale, n_in, b, max_rows) != 0 ||
                bq_vk_tq2_matmat_add(vk, wb, 0, 0, max_rows, max_rows, y) != 0 ||
                bq_vk_tq2_matmat_end(vk) != 0 || bq_vk_wait(vk) != 0) {
                printf("FATAL: run failed at B=%d\n", b); return 2;
            }
            dt = now_sec() - t0;
            if (dt < best) best = dt;
        }
        {
            double w = (double)max_rows * (double)n_in;
            double bytes = (double)max_rows * (double)rb;
            printf("%3d %10.3f %11.1f %11.1f %11.3f %11.2f\n", b, best * 1e3,
                   w / best / 1e9, w * (double)b / best / 1e9,
                   best * 1e6 / (double)max_rows, bytes / best / 1e9);
        }
    }

    {
        int b6 = maxB < 6 ? maxB : 6;
        printf("\nrow sweep at B=%d:\n", b6);
        printf("%6s %10s %11s %11s %11s %11s\n",
               "rows", "ms", "G w/s", "G MAC/s", "us/row", "GB/s");
        for (i = 0; i < nrsweep; ++i) {
            int rows = rsweep[i];
            double best = 1e9;
            if (rows > max_rows) continue;
            for (k = 0; k < reps; ++k) {
                double t0, dt;
                t0 = now_sec();
                if (bq_vk_tq2_matmat_begin(vk, xq, xscale, n_in, b6, rows) != 0 ||
                    bq_vk_tq2_matmat_add(vk, wb, 0, 0, rows, rows, y) != 0 ||
                    bq_vk_tq2_matmat_end(vk) != 0 || bq_vk_wait(vk) != 0) {
                    printf("FATAL: run failed at %d rows\n", rows); return 2;
                }
                dt = now_sec() - t0;
                if (dt < best) best = dt;
            }
            {
                double w = (double)rows * (double)n_in;
                double bytes = (double)rows * (double)rb;
                printf("%6d %10.3f %11.1f %11.1f %11.3f %11.2f\n", rows, best * 1e3,
                       w / best / 1e9, w * (double)b6 / best / 1e9,
                       best * 1e6 / (double)rows, bytes / best / 1e9);
            }
        }
    }

    bq_vk_buf_free(vk, wb);
    bq_vk_destroy(vk);
    free(x); free(y);
    return 0;
}
