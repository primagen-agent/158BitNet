#include "quant_q8k.h"
#include "quant_tq2_0.h"
#include "quant_q6k.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    float input[512] = {0}, output[7];
    bitnet_q8k_block_t q[2];
    input[0] = 127.0f; input[1] = -127.0f; input[2] = 0.5f; input[3] = 1.5f;
    input[256] = -254.0f; input[257] = 127.0f;
    if (bitnet_quantize_q8k(input, 512, q) || q[0].d != -1.0f || q[1].d != 2.0f ||
        q[0].qs[0] != -127 || q[0].qs[1] != 127 || q[0].qs[2] != 0 ||
        q[0].qs[3] != -2 || q[1].qs[257 - 256] != 64) return 1;
    memset(input, 0, sizeof input);
    if (bitnet_quantize_q8k(input, 512, q) || q[0].d || q[1].d) return 2;
    if (!bitnet_quantize_q8k(input, 511, q)) return 3;
    input[3] = NAN;
    if (!bitnet_quantize_q8k(input, 512, q)) return 4;
    for (int i = 0; i < 512; ++i) input[i] = sinf((float)i * 0.73f) * (i < 256 ? 0.25f : 9.0f);
    if (bitnet_quantize_q8k(input, 512, q)) return 5;
    bitnet_tq2_0_block_t tq[14];
    bitnet_q6k_block_t q6[14];
    for (int b = 0; b < 14; ++b) {
        tq[b].d = q6[b].d = 0x3800; /* exactly representable 0.5 */
        for (int i = 0; i < 64; ++i) tq[b].qs[i] = (uint8_t)(i * 31 + b * 13);
        for (int i = 0; i < 128; ++i) q6[b].ql[i] = (uint8_t)(i * 7 + b);
        for (int i = 0; i < 64; ++i) q6[b].qh[i] = (uint8_t)(i * 11 + b);
        for (int i = 0; i < 16; ++i) q6[b].scales[i] = (int8_t)(i - 8);
    }
    for (int kind = 0; kind < 2; ++kind) {
        if (bitnet_matmul_q8k(kind ? (void *)q6 : (void *)tq, kind ? 14 : 35, 7, 512, input, output)) return 6;
        for (int row = 0; row < 7; ++row) {
            double expected = 0;
            for (int b = 0; b < 2; ++b) {
                float weights[256];
                if (kind) bitnet_q6k_dequantize_block(q6 + row * 2 + b, weights, 256);
                else bitnet_tq2_0_dequantize_block(tq + row * 2 + b, weights, 256);
                for (int i = 0; i < 256; ++i) expected += (double)weights[i] * q[b].d * q[b].qs[i];
            }
            if (fabs(output[row] - expected) > 1e-4 + 1e-5 * fabs(expected)) {
                fprintf(stderr, "q8k kind=%d row=%d got=%g expected=%g\n", kind, row, output[row], expected); return 7;
            }
        }
    }
    return 0;
}
