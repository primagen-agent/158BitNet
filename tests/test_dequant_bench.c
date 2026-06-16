#include "quant_tq2_0.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(void) {
    bitnet_tq2_0_block_t block;
    float out[BITNET_TQ2_0_QK];
    const int iterations = 5000000;

    /* Fill block with random 2-bit codes */
    for (int i = 0; i < BITNET_TQ2_0_QS_SIZE; i++) {
        block.qs[i] = (uint8_t)(rand() & 0xFF);
    }
    /* Use fp16 value 1.0: 0x3C00 */
    block.d = 0x3C00;

    /* Verify correctness first */
    memset(out, 0xCC, sizeof(out));
    if (bitnet_tq2_0_dequantize_block(&block, out, BITNET_TQ2_0_QK) != 0) {
        printf("FAIL: dequantize returned error\n");
        return 1;
    }

    /* Verify by comparing with manual computation */
    for (int i = 0; i < BITNET_TQ2_0_QS_SIZE; i++) {
        uint8_t byte = block.qs[i];
        static const float lut[4] = { -1.0f, 0.0f, 1.0f, 0.0f };
        for (int p = 0; p < 4; p++) {
            float expected = lut[(byte >> (6 - 2*p)) & 3u];
            if (out[i*4 + p] != expected) {
                printf("FAIL: mismatch at [%d]: got %f, expected %f\n",
                       i*4 + p, (double)out[i*4 + p], (double)expected);
                return 1;
            }
        }
    }
    printf("Correctness: PASS\n");

    /* Benchmark */
    double start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        bitnet_tq2_0_dequantize_block(&block, out, BITNET_TQ2_0_QK);
    }
    double elapsed = get_time_ms() - start;
    double ns_per_call = elapsed / iterations * 1e6;

    printf("Iterations: %d\n", iterations);
    printf("Total time: %.2f ms\n", elapsed);
    printf("Per call:   %.2f ns\n", ns_per_call);
    printf("Throughput: %.2f M blocks/s\n", 1000.0 / elapsed * iterations / 1e6);

    return 0;
}
