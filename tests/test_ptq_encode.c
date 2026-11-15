/* Round-trip check for the PTQ1_0 encoder used by the DSpine drafter.
 *
 *   ./test_ptq_encode [n]
 *
 * Checks two things for a row of n bf16 weights:
 *   1. the packed trits decode to exactly d * clip(round(w / d)), where d is
 *      the fp16 block scale that is stored in the block;
 *   2. the batched matmat reproduces a reference built from the dequantized
 *      weights and activations (this isolates the kernel from quantization
 *      loss).  The loss against the original bf16 weights is reported
 *      separately - for random gaussian weights a ternary encode is coarse,
 *      so that figure is a property of the format, not of the code.
 */

#include "quant_ptq1_0.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long g_st = 12345u;
static float frand(void) {
    g_st = g_st * 1103515245u + 12345u;
    return (float)((g_st >> 8) & 0xFFFF) / 65535.0f;
}
/* roughly the distribution of a trained weight matrix: gaussian, std 0.02 */
static float grand(void) {
    float u1 = frand(), u2 = frand();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2) * 0.02f;
}
static float bf16_round(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    b = (b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&f, &b, 4);
    return f;
}
static float bf16_val(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 5120;
    const int B = 4, nrow = 16;
    int i, b, r, rc = 0;

    const size_t qbytes = bitnet_ptq1_0_row_size(n);
    uint16_t *wbf = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    uint8_t *q = (uint8_t *)malloc(qbytes);
    float *dec = (float *)malloc((size_t)n * sizeof(float));
    float *x = (float *)malloc((size_t)B * n * sizeof(float));
    int8_t *xq = (int8_t *)malloc((size_t)B * n);
    float xs[16];

    setvbuf(stdout, NULL, _IONBF, 0);
    if (!wbf || !q || !dec || !x || !xq) {
        printf("alloc failed\n");
        return 1;
    }

    for (i = 0; i < n; ++i) {
        float f = bf16_round(grand());
        uint32_t bits;
        memcpy(&bits, &f, 4);
        wbf[i] = (uint16_t)(bits >> 16);
    }
    if (bitnet_ptq1_0_quantize_row(wbf, q, n) != 0) {
        printf("n=%d: quantize failed\n", n);
        return 1;
    }
    bitnet_ptq1_0_dequantize_row(q, dec, n);

    {
        int bad = 0, nz = 0, first = -1;
        double worst = 0.0;
        for (i = 0; i < n; ++i) {
            const bitnet_ptq1_0_block_t *bp = (const bitnet_ptq1_0_block_t *)
                (q + (size_t)(i / 128) * BITNET_PTQ1_0_BLOCK_SIZE);
            const float d = bitnet_ptq1_0_fp16_to_fp32(bp->d);
            const float v = bf16_val(wbf[i]);
            int t = (int)lrintf(v / d);
            float want;
            if (t < -1) t = -1;
            if (t > 1) t = 1;
            want = (float)t * d;
            if (dec[i] != want) {
                const double rel = fabs((double)dec[i] - want) / d;
                if (first < 0) first = i;
                if (rel > worst) worst = rel;
                ++bad;
            }
            if (t != 0) ++nz;
        }
        printf("n=%-6d packed %zu B  roundtrip exact=%s bad=%d first=%d worst_rel=%.2e  "
               "nonzero_trits=%d/%d\n", n, qbytes, bad ? "NO" : "yes", bad, first, worst, nz, n);
        if (bad) ++rc;
    }

    for (b = 0; b < B; ++b) {
        for (i = 0; i < n; ++i) x[(size_t)b * n + i] = grand() * 20.0f;
        bitnet_quantize_row_i8(x + (size_t)b * n, xq + (size_t)b * n, &xs[b], n);
    }
    {
        double dq_acc = 0.0, dq_ref = 0.0, tr_acc = 0.0, tr_ref = 0.0;
        for (r = 0; r < nrow; ++r) {
            float yp[16];
            bitnet_ptq1_0_matmat_row(q, xq, xs, n, B, yp);
            for (b = 0; b < B; ++b) {
                double ref_dq = 0.0, ref_true = 0.0;
                for (i = 0; i < n; ++i) {
                    const double xv = (double)xq[(size_t)b * n + i] * (double)xs[b];
                    ref_dq += (double)dec[i] * xv;
                    ref_true += (double)bf16_val(wbf[i]) * (double)x[(size_t)b * n + i];
                }
                dq_acc += fabs((double)yp[b] - ref_dq);
                dq_ref += fabs(ref_dq);
                tr_acc += fabs((double)yp[b] - ref_true);
                tr_ref += fabs(ref_true);
            }
        }
        printf("n=%-6d matmat vs dequantized ref %.4f%%  vs bf16 ref %.2f%%\n", n,
               100.0 * dq_acc / (dq_ref > 0 ? dq_ref : 1.0),
               100.0 * tr_acc / (tr_ref > 0 ? tr_ref : 1.0));
        if (100.0 * dq_acc / (dq_ref > 0 ? dq_ref : 1.0) > 0.5) ++rc;
    }

    free(x);
    free(xq);
    free(dec);
    free(q);
    free(wbf);
    printf(rc ? "FAIL\n" : "ALL OK\n");
    return rc;
}
