#ifndef BITNET_QUANT_Q8K_H
#define BITNET_QUANT_Q8K_H
#include <stdint.h>

/* Activation quantization shared by GGUF TQ2_0 and Q6_K dot products. */
typedef struct bitnet_q8k_block {
    float d;
    int8_t qs[256];
} bitnet_q8k_block_t;

int bitnet_quantize_q8k(const float *input, int count, bitnet_q8k_block_t *blocks);
int bitnet_matmul_q8k(const void *weights, int gguf_type, int rows, int columns,
                     const float *input, float *output);
/* Largest supported activation is 16384 floats (8b FFN) = 64 Q8_K blocks. */
#define BITNET_Q8K_MAX_BLOCKS 64
int bitnet_quantize_q8k_impl(const float *input, int count, bitnet_q8k_block_t *blocks);
int bitnet_matmul_q8k_impl(const void *weights, int gguf_type, int rows, int columns,
                          const float *input, float *output);
/* Same dot with a caller-quantized input; lets one activation vector serve
 * several projections without re-quantizing per call. */
int bitnet_matmul_q8k_prepared(const void *weights, int gguf_type, int rows, int columns,
                               const bitnet_q8k_block_t *input, float *output);

/* Internal synchronous executor. Like the other matmul entry points, runtime
 * callers hold g_eval_mutex; callbacks may not dispatch nested work. */
int bitnet_quant_run_rows(void (*compute)(void *, int, int), void *opaque, int rows);
#endif
