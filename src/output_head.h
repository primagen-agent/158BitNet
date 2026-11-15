#ifndef OUTPUT_HEAD_H
#define OUTPUT_HEAD_H

/* The vocabulary projection, prepared once at load and dotted every token.
 *
 * The output pool lives in src/bitnet.c: a persistent set of workers that
 * split the vocabulary into row chunks and run a dotprod kernel over each.
 * Both engines use it.  src/bitnet.c builds the caches while it loads its own
 * GGUF; src/minicpm.c builds the same layouts from its head tensor and calls
 * the entry points below, so a family that is not qwen35 still reaches the
 * vocabulary through the parallel kernel instead of dequantizing the head
 * one row at a time on the calling thread.
 *
 * The layout is "blockscale": int8 rows with one float scale per 256-weight
 * block, which is what a tied f16 embedding becomes when it is quantized on
 * load.  Handing the pool a compact Q6_K copy is possible
 * (bitnet_output_projection_q8_compact_neon_parallel) but measurably slower
 * than reading the Q6_K weights in place, so nothing here uses it.
 *
 * `emb_dim` is the activation width and must be a positive multiple of 256;
 * `qhidden` is the activation already quantized by
 * bitnet_tq2_0_quantize_vec_i8[_impl], `hidden_scale` its per-tensor scale.
 * `w8` holds `vocab` rows of `emb_dim` int8 codes. */
int bitnet_output_logits_blockscale(const int8_t *w8, const float *w8_d,
                                    int emb_dim, int vocab,
                                    const int8_t *qhidden, float hidden_scale, float *logits);

#endif
