#ifndef QUANT_Q4K_H
#define QUANT_Q4K_H

#include <stddef.h>
#include <stdint.h>

#define BITNET_Q4K_QK 256
#define BITNET_Q4K_BLOCK_SIZE 144
#define BITNET_Q4K_QS_SIZE (BITNET_Q4K_QK / 2)
#define BITNET_Q4K_SCALES_SIZE (3 * BITNET_Q4K_QK / 64)

typedef struct bitnet_q4k_block {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[BITNET_Q4K_SCALES_SIZE];
    uint8_t qs[BITNET_Q4K_QS_SIZE];
} bitnet_q4k_block_t;

int bitnet_q4k_dequantize_block(const bitnet_q4k_block_t *block, float *out, size_t out_len);

int bitnet_q4k_embedding_lookup(const void *data, int token_id, int embedding_length,
                                int vocab_size, float *out, size_t out_len);

#endif
