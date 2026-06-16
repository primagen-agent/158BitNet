#include "quant_q6k.h"
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

static int almost_equal(float a, float b) {
    float diff = fabsf(a - b);
    return diff < 0.001f;
}

int main(void) {
    bitnet_q6k_block_t block;
    float deq[BITNET_Q6K_QK];
    float vec[BITNET_Q6K_QK];
    int8_t qvec[BITNET_Q6K_QK];
    float expected = 0.0f;
    float expected_i8 = 0.0f;
    float got = 0.0f;
    float got_i8 = 0.0f;
    float got_expanded = 0.0f;
    float got_compact = 0.0f;
    float got_compact4[4];
    float got_compact8[8];
    float got_expanded4[4];
    float got_expanded8[8];
    float vec_scale = 1.0f / 7.0f;
    int8_t expanded_q8[BITNET_Q6K_QK];
    float expanded_scales[BITNET_Q6K_QK / 16];
    int8_t compact_scales[BITNET_Q6K_QK / 16];
    float compact_d[1];
    bitnet_q6k_block_t blocks4[4];
    int8_t expanded_q8_4[4 * BITNET_Q6K_QK];
    float expanded_scales4[4 * (BITNET_Q6K_QK / 16)];
    int8_t compact_scales4[4 * (BITNET_Q6K_QK / 16)];
    float compact_d4[4];
    bitnet_q6k_block_t blocks8[8];
    int8_t expanded_q8_8[8 * BITNET_Q6K_QK];
    float expanded_scales8[8 * (BITNET_Q6K_QK / 16)];
    int8_t compact_scales8[8 * (BITNET_Q6K_QK / 16)];
    float compact_d8[8];

    printf("sizeof(bitnet_q6k_block_t) = %zu (expected 210)\n", sizeof(bitnet_q6k_block_t));
    printf("offset of d = %zu\n", offsetof(bitnet_q6k_block_t, d));
    printf("offset of ql = %zu\n", offsetof(bitnet_q6k_block_t, ql));
    printf("offset of qh = %zu\n", offsetof(bitnet_q6k_block_t, qh));
    printf("offset of scales = %zu\n", offsetof(bitnet_q6k_block_t, scales));

    memset(&block, 0, sizeof(block));
    block.d = 0x3c00;
    for (int i = 0; i < BITNET_Q6K_QL_SIZE; ++i) {
        block.ql[i] = (uint8_t)(i * 37u + 11u);
    }
    for (int i = 0; i < BITNET_Q6K_QH_SIZE; ++i) {
        block.qh[i] = (uint8_t)(i * 13u + 7u);
    }
    for (int i = 0; i < BITNET_Q6K_SCALES_SIZE; ++i) {
        block.scales[i] = (int8_t)(i - 8);
    }
    for (int i = 0; i < BITNET_Q6K_QK; ++i) {
        vec[i] = (float)((i % 17) - 8) / 7.0f;
        qvec[i] = (int8_t)((i % 17) - 8);
    }

    if (bitnet_q6k_dequantize_block(&block, deq, BITNET_Q6K_QK) != 0) return 1;
    for (int i = 0; i < BITNET_Q6K_QK; ++i) {
        expected += deq[i] * vec[i];
    }
    if (bitnet_q6k_dot_product(&block, vec, BITNET_Q6K_QK, &got) != 0) return 2;
    if (!almost_equal(expected, got)) {
        printf("dot mismatch: expected %.6f got %.6f\n", (double)expected, (double)got);
        return 3;
    }
    for (int i = 0; i < BITNET_Q6K_QK; ++i) {
        expected_i8 += deq[i] * ((float)qvec[i] * vec_scale);
    }
    if (bitnet_q6k_dot_product_i8_neon(&block, qvec, vec_scale, BITNET_Q6K_QK, &got_i8) != 0) return 4;
    if (!almost_equal(expected_i8, got_i8)) {
        printf("i8 dot mismatch: expected %.6f got %.6f\n", (double)expected_i8, (double)got_i8);
        return 5;
    }
    if (bitnet_q6k_expand_to_q8(&block, 1, BITNET_Q6K_QK,
                                expanded_q8, expanded_scales) != 0) return 6;
    if (bitnet_q6k_dot_product_q8_neon(expanded_q8, expanded_scales,
                                       1, qvec, vec_scale, &got_expanded) != 0) return 7;
    if (!almost_equal(expected_i8, got_expanded)) {
        printf("expanded i8 dot mismatch: expected %.6f got %.6f\n",
               (double)expected_i8, (double)got_expanded);
        return 8;
    }
    if (bitnet_q6k_expand_to_q8_compact(&block, 1, BITNET_Q6K_QK,
                                        expanded_q8, compact_scales, compact_d) != 0) return 9;
    if (bitnet_q6k_dot_product_q8_compact_neon(expanded_q8, compact_scales, compact_d,
                                               1, qvec, vec_scale, &got_compact) != 0) return 10;
    if (!almost_equal(got_expanded, got_compact)) {
        printf("compact i8 dot mismatch: expected %.6f got %.6f\n",
               (double)got_expanded, (double)got_compact);
        return 11;
    }
    for (int r = 0; r < 4; ++r) {
        blocks4[r] = block;
        blocks4[r].ql[0] = (uint8_t)(blocks4[r].ql[0] + r);
        blocks4[r].qh[0] = (uint8_t)(blocks4[r].qh[0] + r * 3);
        blocks4[r].scales[0] = (int8_t)(blocks4[r].scales[0] + r);
    }
    if (bitnet_q6k_expand_to_q8(blocks4, 4, BITNET_Q6K_QK,
                                expanded_q8_4, expanded_scales4) != 0) return 12;
    if (bitnet_q6k_expand_to_q8_compact(blocks4, 4, BITNET_Q6K_QK,
                                        expanded_q8_4, compact_scales4, compact_d4) != 0) return 13;
    if (bitnet_q6k_dot_product_q8_4_neon(expanded_q8_4, expanded_scales4,
                                         BITNET_Q6K_QK, BITNET_Q6K_QK / 16,
                                         1, qvec, vec_scale, got_expanded4) != 0) return 14;
    if (bitnet_q6k_dot_product_q8_compact_4_neon(expanded_q8_4, compact_scales4, compact_d4,
                                                 BITNET_Q6K_QK, BITNET_Q6K_QK / 16, 1,
                                                 1, qvec, vec_scale, got_compact4) != 0) return 15;
    for (int r = 0; r < 4; ++r) {
        float got_single = 0.0f;
        if (bitnet_q6k_dot_product_q8_neon(expanded_q8_4 + r * BITNET_Q6K_QK,
                                           expanded_scales4 + r * (BITNET_Q6K_QK / 16),
                                           1, qvec, vec_scale, &got_single) != 0) return 16;
        if (!almost_equal(got_single, got_expanded4[r])) {
            printf("expanded 4-row mismatch row %d: expected %.6f got %.6f\n",
                   r, (double)got_single, (double)got_expanded4[r]);
            return 17;
        }
        if (!almost_equal(got_expanded4[r], got_compact4[r])) {
            printf("compact 4-row mismatch row %d: expected %.6f got %.6f\n",
                   r, (double)got_expanded4[r], (double)got_compact4[r]);
            return 18;
        }
    }

    for (int r = 0; r < 8; ++r) {
        blocks8[r] = block;
        blocks8[r].ql[0] = (uint8_t)(blocks8[r].ql[0] + r);
        blocks8[r].qh[0] = (uint8_t)(blocks8[r].qh[0] + r * 3);
        blocks8[r].scales[0] = (int8_t)(blocks8[r].scales[0] + r);
    }
    if (bitnet_q6k_expand_to_q8(blocks8, 8, BITNET_Q6K_QK,
                                expanded_q8_8, expanded_scales8) != 0) return 19;
    if (bitnet_q6k_expand_to_q8_compact(blocks8, 8, BITNET_Q6K_QK,
                                        expanded_q8_8, compact_scales8, compact_d8) != 0) return 20;
    if (bitnet_q6k_dot_product_q8_8_neon(expanded_q8_8, expanded_scales8,
                                         BITNET_Q6K_QK, BITNET_Q6K_QK / 16,
                                         1, qvec, vec_scale, got_expanded8) != 0) return 21;
    if (bitnet_q6k_dot_product_q8_compact_8_neon(expanded_q8_8, compact_scales8, compact_d8,
                                                 BITNET_Q6K_QK, BITNET_Q6K_QK / 16, 1,
                                                 1, qvec, vec_scale, got_compact8) != 0) return 22;
    for (int r = 0; r < 8; ++r) {
        float got_single = 0.0f;
        if (bitnet_q6k_dot_product_q8_neon(expanded_q8_8 + r * BITNET_Q6K_QK,
                                           expanded_scales8 + r * (BITNET_Q6K_QK / 16),
                                           1, qvec, vec_scale, &got_single) != 0) return 23;
        if (!almost_equal(got_single, got_expanded8[r])) {
            printf("expanded 8-row mismatch row %d: expected %.6f got %.6f\n",
                   r, (double)got_single, (double)got_expanded8[r]);
            return 24;
        }
        if (!almost_equal(got_expanded8[r], got_compact8[r])) {
            printf("compact 8-row mismatch row %d: expected %.6f got %.6f\n",
                   r, (double)got_expanded8[r], (double)got_compact8[r]);
            return 25;
        }
    }

    return 0;
}
