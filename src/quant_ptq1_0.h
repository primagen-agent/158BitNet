#ifndef QUANT_PTQ1_0_H
#define QUANT_PTQ1_0_H

#include <stddef.h>
#include <stdint.h>

/* Prism PTQ1_0: ternary weights at group size 128.
 *
 * Block layout (28 bytes), identical to ggml's block_ptq1_0:
 *   uint8_t  qs[24] base-3 packed trits, 5 per byte (120 values)
 *   uint8_t  qh[2]  base-3 packed trits, 4 per byte (8 values)
 *   uint16_t d      fp16 scale for the group of 128 trits
 *
 * The scale sits at the END of the block.  Reference:
 * dequantize_row_ptq1_0() in the PrismML llama.cpp fork. */

#define BITNET_PTQ1_0_QK 128
#define BITNET_PTQ1_0_BLOCK_SIZE 28
#define BITNET_PTQ1_0_QS_SIZE 24
#define BITNET_PTQ1_0_QH_SIZE 2

typedef struct bitnet_ptq1_0_block {
    uint8_t qs[BITNET_PTQ1_0_QS_SIZE];
    uint8_t qh[BITNET_PTQ1_0_QH_SIZE];
    uint16_t d;
} bitnet_ptq1_0_block_t;

float bitnet_ptq1_0_fp16_to_fp32(uint16_t h);

/* Dequantize n values (n must be a multiple of 128) from packed rows. */
void bitnet_ptq1_0_dequantize_row(const void *src, float *out, int n);

/* Dot product of one packed row with a float vector. */
float bitnet_ptq1_0_dot_row(const void *src, const float *x, int n);

/* Explicit kernels, exposed so a test can compare them. */
float bitnet_ptq1_0_dot_row_scalar(const void *src, const float *x, int n);
#if defined(__aarch64__)
float bitnet_ptq1_0_dot_row_neon(const void *src, const float *x, int n);

#endif

/* Bytes per row for n columns. */
size_t bitnet_ptq1_0_row_size(int n);

/* Encode one row of bf16 weights (n a multiple of 128) into PTQ1_0 blocks,
 * block absmean scale, nearest trit.  Returns 0 or -1. */
int bitnet_ptq1_0_quantize_row(const void *src, void *dst, int n);

/* q8_0 activations: 32 values per block, fp16 scale - the same layout ggml
 * uses, so the integer dot below mirrors the reference CPU kernel. */
#define BITNET_Q8_0_QK 32
typedef struct bitnet_q8_0_block {
    uint16_t d;
    int8_t qs[BITNET_Q8_0_QK];
} bitnet_q8_0_block_t;

/* Bytes needed to hold the q8_0 form of n values. */
size_t bitnet_q8_0_row_size(int n);

/* Quantize n floats (multiple of 32) into q8_0 blocks. */
void bitnet_quantize_row_q8_0(const float *x, void *dst, int n);

/* Dot product of a packed PTQ1_0 row with q8_0 activations.  Integer inner
 * loop; uses the ARMv8.2 dot-product extension when the CPU has it. */
float bitnet_ptq1_0_dot_row_q8(const void *src, const void *xq, int n);

/* Batched form used by block verification: one packed row against `B` int8
 * activation rows (each n values, contiguous, one scale per row). */
void bitnet_quantize_row_i8(const float *x, int8_t *dst, float *scale, int n);

/* The same int8 row in the group layout the Vulkan matmat kernel reads: each
 * 128-value block becomes 32 uint32 groups of four trit lanes, in the weight
 * decoder's order.  Depends only on the block decoder, so a row packed once
 * serves every matrix in the layer that shares it. */
void bitnet_quantize_row_i8_groups(const float *x, int8_t *dst, float *scale, int n);

/* Inverse of the above, for validating the GPU layout against element order. */
void bitnet_unpack_row_i8_groups(const int8_t *src, float *out, float scale, int n);
void bitnet_ptq1_0_matmat_row(const void *src, const int8_t *xq, const float *xscale,
                              int n, int B, float *out);
/* 2-bit repacked rows for the ARMv8.6 i8mm kernels.
 *
 * Two output rows are stored interleaved so that a 16-byte SMMLA operand
 * carries 8 weights of each: four SDOT of a 128-trit block become one SMMLA
 * pair, and the decode is a shift plus an and per 16 weights instead of the
 * base-3 mul/hadd chains.  Values are the same trits, so results match the
 * 28-byte path exactly up to float summation order.  Packing is per output
 * row pair, which is why the caller has to hand rows over two at a time. */
int bitnet_ptq1_0_p2_available(void);
size_t bitnet_ptq1_0_p2_pair_bytes(int n);
size_t bitnet_ptq1_0_p2_prep_bytes(int n, int B);
void bitnet_ptq1_0_p2_pack(const void *row_a, const void *row_b, int n,
                           uint8_t *dst, float *scale);
void bitnet_ptq1_0_p2_prep(void *prep, const int8_t *xq, int n, int B);
void bitnet_ptq1_0_matmat_p2(const void *w2, const float *d, const void *prep,
                             const float *xscale, int B, int n_out, int row0, float *yb);
/* Two row pairs in one pass, sharing the sixteen activation-tile loads.  Same
 * contract as above with four output rows: vectors for pairs row0/2 and
 * row0/2+1 must be adjacent, which is how the packer stores them. */
void bitnet_ptq1_0_matmat_p2q(const void *w2, const float *d, const void *prep,
                              const float *xscale, int B, int n_out, int row0, float *yb);

#endif
