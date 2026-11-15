/* Throughput bench and correctness check for the TQ2_0 GPU matvec.
 *
 *   test_vk_tq2_bench [-i N_IN] [-n MAX_ROWS] [-r REPEATS]
 *
 * Builds one synthetic TQ2_0 matrix of MAX_ROWS x N_IN (the shape of a real
 * tensor: 66-byte blocks), checks the GPU rows against a CPU decode of the
 * same bytes, then times a sweep of row counts through
 * bq_vk_tq2_batch_{begin,add,end} + bq_vk_wait, i.e. one round trip per data
 * point, exactly as the model calls it.  Reports weights/s and the bytes/s
 * that implies, so the kernel rate can be read off against the device's
 * streaming ceiling and against the PTQ1_0 kernel on the same machine. */

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

/* code of weight position p inside a block, and its fp16 scale */
static float tq2_weight(const unsigned char *blk, int p) {
    unsigned byte = blk[(p >> 7) * 32 + (p & 31)];
    return (float)((byte >> (((p >> 5) & 3) * 2)) & 3u) - 1.0f;
}

int main(int argc, char **argv) {
    int n_in = 4096, max_rows = 16384, reps = 3, i, r, b, p;
    size_t block = 66, wbytes, rb;
    int nb = n_in / 256;
    bq_vk_t *vk;
    bq_vk_buf_t *wb = NULL;
    unsigned char *wp;
    float *x, *y;
    unsigned st = 12345u;
    int sweep[] = { 1024, 4096, 8192, 16384, 32768, 65536 };
    int nsweep = (int)(sizeof sweep / sizeof sweep[0]);

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "-i") && i + 1 < argc) n_in = atoi(argv[++i]);
        else if (!strcmp(a, "-n") && i + 1 < argc) max_rows = atoi(argv[++i]);
        else if (!strcmp(a, "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [-i N_IN] [-n MAX_ROWS] [-r REPS]\n", argv[0]); return 2; }
    }
    if (n_in % 256) { fprintf(stderr, "n_in must be a multiple of 256\n"); return 2; }
    nb = n_in / 256;

    vk = bq_vk_create();
    if (vk == NULL || !bq_vk_ready(vk)) { printf("FATAL: no usable Vulkan device\n"); return 2; }
    if (!bq_vk_tq2_available(vk)) { printf("FATAL: tq2 kernel unavailable\n"); return 2; }

    rb = (size_t)nb * block;
    wbytes = rb * (size_t)max_rows;
    wp = bq_vk_buf_alloc(vk, wbytes, &wb);
    if (wp == NULL) { printf("FATAL: weight alloc of %zu bytes failed\n", wbytes); return 2; }
    x = malloc((size_t)n_in * sizeof(float));
    y = malloc((size_t)max_rows * sizeof(float));
    if (!x || !y) { printf("FATAL: host alloc failed\n"); return 2; }

    for (i = 0; i < n_in; ++i) {
        st = st * 1664525u + 1013904223u;
        x[i] = (float)((int)(st >> 24) - 128) / 128.0f;
    }
    for (r = 0; r < max_rows; ++r) {
        for (b = 0; b < nb; ++b) {
            unsigned char *blk = wp + (size_t)r * rb + (size_t)b * block;
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

    /* correctness: four rows spread over the matrix, decoded on the CPU */
    {
        int probe[] = { 0, 1, 2, max_rows / 2, max_rows - 1 };
        int np = (int)(sizeof probe / sizeof probe[0]), bad = 0;
        if (bq_vk_tq2_batch_begin(vk, x, n_in, max_rows) != 0) { printf("FATAL: begin\n"); return 2; }
        for (i = 0; i < np; ++i) {
            int row = probe[i];
            if (bq_vk_tq2_batch_add(vk, wb, n_in, row, row + 1, y) != 0) { printf("FATAL: add\n"); return 2; }
        }
        if (bq_vk_tq2_batch_end(vk) != 0 || bq_vk_wait(vk) != 0) { printf("FATAL: submit\n"); return 2; }
        for (i = 0; i < np; ++i) {
            int row = probe[i];
            double sum = 0.0, gpu = y[row];
            /* the GPU applies the block scale inside the block, so fold it the
             * same way here rather than once at the end */
            for (b = 0; b < nb; ++b) {
                const unsigned char *blk = wp + (size_t)row * rb + (size_t)b * block;
                float d = half_to_float((unsigned)blk[64] | ((unsigned)blk[65] << 8));
                double s = 0.0;
                for (p = 0; p < 256; ++p) s += (double)tq2_weight(blk, p) * x[b * 256 + p];
                sum += s * (double)d;
            }
            {
                double rel = fabs(sum) > 1e-9 ? fabs((double)gpu - sum) / fabs(sum) : fabs((double)gpu - sum);
                printf("row %6d gpu %.9g cpu %.9g rel %.2e%s\n", row, (double)gpu, sum, rel,
                       rel > 1e-3 ? "  MISMATCH" : "");
                if (rel > 1e-3) ++bad;
            }
        }
        if (bad) { printf("FATAL: %d rows mismatch, bench data unusable\n", bad); return 1; }
    }

    /* one warm pass so the sweep does not start at a low GPU clock */
    if (bq_vk_tq2_batch_begin(vk, x, n_in, max_rows) == 0 &&
        bq_vk_tq2_batch_add(vk, wb, n_in, 0, max_rows, y) == 0 &&
        bq_vk_tq2_batch_end(vk) == 0) (void)bq_vk_wait(vk);

    printf("\n%6s %10s %10s %10s %10s\n", "rows", "ms", "G w/s", "GB/s", "us/row");
    for (i = 0; i < nsweep; ++i) {
        int rows = sweep[i];
        double best = 1e9;
        int k;
        if (rows > max_rows) continue;
        for (k = 0; k < reps; ++k) {
            double t0 = now_sec(), dt;
            if (bq_vk_tq2_batch_begin(vk, x, n_in, rows) != 0 ||
                bq_vk_tq2_batch_add(vk, wb, n_in, 0, rows, y) != 0 ||
                bq_vk_tq2_batch_end(vk) != 0 || bq_vk_wait(vk) != 0) {
                printf("FATAL: run failed at %d rows\n", rows);
                return 2;
            }
            dt = now_sec() - t0;
            if (dt < best) best = dt;
        }
        {
            double w = (double)rows * (double)n_in;
            double bytes = (double)rows * (double)rb;
            printf("%6d %10.3f %10.1f %10.2f %10.3f\n", rows, best * 1e3,
                   w / best / 1e9, bytes / best / 1e9, best * 1e6 / rows);
        }
    }
    bq_vk_buf_free(vk, wb);
    bq_vk_destroy(vk);
    free(x); free(y);
    return 0;
}
