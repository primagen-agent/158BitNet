#include "quant_tq2_0.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

static int almost_equal(float a, float b) {
    float diff = a - b;
    if (diff < 0.0f) diff = -diff;
    return diff < 0.0005f;
}

int main(void) {
    bitnet_tq2_0_block_t block;
    bitnet_tq2_0_block_t weight[8];
    float values[BITNET_TQ2_0_QK];
    float dot = 0.0f;
    float vec[BITNET_TQ2_0_QK * 4];
    float matmul_ref[2];
    float matmul_lut[2];
    float matmul_i8[2];
    float matmul_neon[2];
    float matmul_neon_pair[2];
    float matmul_i16[2];
    float lut[8 * 2 * 32 * 256];
    int16_t i16_lut[8 * 2 * 32 * 256];
    float scales[8];
    int8_t qvec[BITNET_TQ2_0_QK * 4];
    int8_t qvec_known_max[BITNET_TQ2_0_QK * 4];
    float qscale = 0.0f;
    float qscale_known_max = 0.0f;
    int32_t qblock_bsums_known_max[4];
    int8_t scale_qvec[BITNET_TQ2_0_QK];
    float scale_vec[BITNET_TQ2_0_QK];
    float scale_check = 0.0f;

    memset(&block, 0, sizeof(block));

    block.qs[0] = 0;
    block.qs[1] = 1;
    block.qs[2] = 2;
    block.qs[3] = 1;
    block.d = 0x3c00;

    if (bitnet_tq2_0_dequantize_block(&block, values, BITNET_TQ2_0_QK) != 0) return 1;
    if (!almost_equal(values[0], -1.0f)) return 2;
    if (!almost_equal(values[1], 0.0f)) return 3;
    if (!almost_equal(values[2], 1.0f)) return 4;
    if (!almost_equal(values[3], 0.0f)) return 5;

    memset(scale_vec, 0, sizeof(scale_vec));
    scale_vec[0] = -1.0f;
    scale_vec[1] = 2.0f;
    scale_vec[2] = -3.0f;
    scale_vec[3] = 4.0f;
    if (bitnet_tq2_0_quantize_vec_i8(scale_vec, BITNET_TQ2_0_QK,
                                     scale_qvec, &scale_check, NULL) != 0) return 39;
    if (fabsf(scale_check - (4.0f / 127.0f)) > 0.000001f) {
        fprintf(stderr, "qscale=%.8f expected=%.8f\n", scale_check, 4.0f / 127.0f);
        return 40;
    }

    memset(vec, 0, sizeof(vec));
    vec[0] = 1.0f;
    vec[1] = 2.0f;
    vec[2] = 3.0f;
    vec[3] = 4.0f;

    if (bitnet_tq2_0_dot_product(&block, vec, BITNET_TQ2_0_QK, &dot) != 0) return 6;
    if (!almost_equal(dot, 2.0f)) return 7;

    memset(weight, 0, sizeof(weight));
    for (int b = 0; b < 8; ++b) {
        weight[b] = block;
    }
    if (bitnet_tq2_0_matmul_vector(weight, 2, BITNET_TQ2_0_QK * 4, vec, matmul_ref) != 0) return 8;
    if (bitnet_tq2_0_build_vec_lut(vec, BITNET_TQ2_0_QK * 4, lut,
                                   sizeof(lut) / sizeof(lut[0])) != 0) return 9;
    if (bitnet_tq2_0_matmul_vector_lut(weight, 2, BITNET_TQ2_0_QK * 4, lut, matmul_lut) != 0) return 10;
    if (!almost_equal(matmul_ref[0], matmul_lut[0])) return 11;
    if (!almost_equal(matmul_ref[1], matmul_lut[1])) return 12;
    if (bitnet_tq2_0_build_scales(weight, 2, BITNET_TQ2_0_QK * 4,
                                  scales, sizeof(scales) / sizeof(scales[0])) != 0) return 16;
    memset(matmul_lut, 0, sizeof(matmul_lut));
    if (bitnet_tq2_0_matmul_vector_lut_scales(weight, scales, 2, BITNET_TQ2_0_QK * 4,
                                              lut, matmul_lut) != 0) return 17;
    if (!almost_equal(matmul_ref[0], matmul_lut[0])) return 18;
    if (!almost_equal(matmul_ref[1], matmul_lut[1])) return 19;
    int32_t qblock_bsums[4];
    if (bitnet_tq2_0_quantize_vec_i8(vec, BITNET_TQ2_0_QK * 4, qvec, &qscale, qblock_bsums) != 0) return 23;
    if (bitnet_tq2_0_quantize_vec_i8_known_max(vec, BITNET_TQ2_0_QK * 4,
                                               qvec_known_max, &qscale_known_max,
                                               qblock_bsums_known_max, 4.0f) != 0) return 41;
    if (!almost_equal(qscale, qscale_known_max)) return 42;
    for (int i = 0; i < BITNET_TQ2_0_QK * 4; ++i) {
        if (qvec[i] != qvec_known_max[i]) return 43;
    }
    for (int i = 0; i < 4; ++i) {
        if (qblock_bsums[i] != qblock_bsums_known_max[i]) return 44;
    }
    if (bitnet_tq2_0_matmul_vector_i8_scales(weight, scales, 2, BITNET_TQ2_0_QK * 4,
                                             qvec, qscale, matmul_i8) != 0) return 24;
    if (fabsf(matmul_ref[0] - matmul_i8[0]) > 0.05f) return 25;
    if (fabsf(matmul_ref[1] - matmul_i8[1]) > 0.05f) return 26;
    if (bitnet_tq2_0_matmul_vector_i8_neon_scales(weight, scales, 2, BITNET_TQ2_0_QK * 4,
                                                  qvec, qscale, qblock_bsums, matmul_neon) != 0) return 31;
    if (!almost_equal(matmul_i8[0], matmul_neon[0])) {
        fprintf(stderr, "i8[0]=%.8f neon[0]=%.8f diff=%.8f\n", matmul_i8[0], matmul_neon[0], matmul_i8[0]-matmul_neon[0]);
        return 32;
    }
    if (!almost_equal(matmul_i8[1], matmul_neon[1])) {
        fprintf(stderr, "i8[1]=%.8f neon[1]=%.8f diff=%.8f\n", matmul_i8[1], matmul_neon[1], matmul_i8[1]-matmul_neon[1]);
        return 33;
    }
    if (bitnet_tq2_0_matmul_vector_i8_neon_pair_scales(weight, scales, weight, scales,
                                                       2, BITNET_TQ2_0_QK * 4,
                                                       qvec, qscale, qblock_bsums,
                                                       matmul_neon_pair, matmul_ref) != 0) return 34;
    if (!almost_equal(matmul_neon[0], matmul_neon_pair[0])) return 35;
    if (!almost_equal(matmul_neon[1], matmul_neon_pair[1])) return 36;
    if (!almost_equal(matmul_neon[0], matmul_ref[0])) return 37;
    if (!almost_equal(matmul_neon[1], matmul_ref[1])) return 38;
    if (bitnet_tq2_0_build_i16_lut(qvec, BITNET_TQ2_0_QK * 4,
                                   i16_lut, sizeof(i16_lut) / sizeof(i16_lut[0])) != 0) return 27;
    if (bitnet_tq2_0_matmul_vector_i16_lut_scales(weight, scales, 2, BITNET_TQ2_0_QK * 4,
                                                  i16_lut, qscale, matmul_i16) != 0) return 28;
    if (!almost_equal(matmul_i8[0], matmul_i16[0])) return 29;
    if (!almost_equal(matmul_i8[1], matmul_i16[1])) return 30;
    memset(matmul_lut, 0, sizeof(matmul_lut));
    if (bitnet_tq2_0_matmul_vector_lut_pair(weight, weight, 2, BITNET_TQ2_0_QK * 4,
                                            lut, matmul_lut, matmul_ref) != 0) return 13;
    if (!almost_equal(matmul_ref[0], matmul_lut[0])) return 14;
    if (!almost_equal(matmul_ref[1], matmul_lut[1])) return 15;
    memset(matmul_lut, 0, sizeof(matmul_lut));
    memset(matmul_ref, 0, sizeof(matmul_ref));
    if (bitnet_tq2_0_matmul_vector_lut_pair_scales(weight, scales, weight, scales,
                                                   2, BITNET_TQ2_0_QK * 4,
                                                   lut, matmul_lut, matmul_ref) != 0) return 20;
    if (!almost_equal(matmul_ref[0], matmul_lut[0])) return 21;
    if (!almost_equal(matmul_ref[1], matmul_lut[1])) return 22;

    return 0;
}
