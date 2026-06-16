#include "quant_q4k.h"

#include <math.h>
#include <string.h>

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t raw = sign << 31;
            float f;
            memcpy(&f, &raw, sizeof(f));
            return f;
        }
        /* Denormalized: value = (-1)^sign * 2^(-14) * (mant/1024) */
        float value = ldexpf((float)mant / 1024.0f, -14);
        return sign ? -value : value;
    }

    if (exp == 31) {
        uint32_t raw = (sign << 31) | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &raw, sizeof(f));
        return f;
    }

    exp = exp - 15u + 127u;
    mant <<= 13;

    uint32_t raw = (sign << 31) | (exp << 23) | mant;
    float f;
    memcpy(&f, &raw, sizeof(f));
    return f;
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4);
    }
}

int bitnet_q4k_dequantize_block(const bitnet_q4k_block_t *block, float *out, size_t out_len) {
    int j = 0;
    int l = 0;
    int is = 0;
    const uint8_t *qs = NULL;

    if (block == NULL || out == NULL || out_len < BITNET_Q4K_QK) {
        return -1;
    }

    float d = fp16_to_fp32(block->d);
    float min = fp16_to_fp32(block->dmin);

    qs = block->qs;
    is = 0;

    for (j = 0; j < BITNET_Q4K_QK; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = min * (float)m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        float d2 = d * (float)sc;
        float m2 = min * (float)m;

        for (l = 0; l < 32; ++l) {
            *out++ = d1 * (float)(qs[l] & 0xF) - m1;
        }
        for (l = 0; l < 32; ++l) {
            *out++ = d2 * (float)(qs[l] >> 4) - m2;
        }

        qs += 32;
        is += 2;
    }

    return 0;
}

int bitnet_q4k_embedding_lookup(const void *data, int token_id, int embedding_length,
                                int vocab_size, float *out, size_t out_len) {
    int blocks_per_row = 0;
    size_t block_offset = 0;
    size_t block_size = BITNET_Q4K_BLOCK_SIZE;
    int n_blocks = 0;
    size_t total_out_len = 0;
    int b = 0;

    if (data == NULL || out == NULL || out_len < (size_t)embedding_length) {
        return -1;
    }
    if (token_id < 0 || token_id >= vocab_size) {
        return -1;
    }
    if (embedding_length <= 0 || vocab_size <= 0) {
        return -1;
    }

    blocks_per_row = embedding_length / BITNET_Q4K_QK;
    if (embedding_length % BITNET_Q4K_QK != 0) {
        return -1;
    }

    n_blocks = blocks_per_row;
    block_offset = (size_t)token_id * (size_t)n_blocks;
    total_out_len = (size_t)n_blocks * BITNET_Q4K_QK;
    if (total_out_len < (size_t)embedding_length) {
        return -1;
    }

    for (b = 0; b < n_blocks; ++b) {
        const uint8_t *block_bytes = (const uint8_t *)data;
        const bitnet_q4k_block_t *block = (const bitnet_q4k_block_t *)(block_bytes + (block_offset + (size_t)b) * block_size);
        float buf[BITNET_Q4K_QK];
        if (bitnet_q4k_dequantize_block(block, buf, BITNET_Q4K_QK) != 0) {
            return -1;
        }
        memcpy(out + (size_t)b * BITNET_Q4K_QK, buf, BITNET_Q4K_QK * sizeof(float));
    }

    return 0;
}
