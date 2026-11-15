#ifndef BITNET_VK_BACKEND_H
#define BITNET_VK_BACKEND_H

/* Vulkan compute backend for the Ternary-Bonsai runtime.
 *
 * The device this targets has no dedicated VRAM: every Vulkan allocation is
 * system memory that both engines can reach, so weights are allocated as
 * host-visible buffers and the CPU reads the very same pages the shaders do.
 * The GPU takes a fraction of the rows of each matvec while the CPU thread
 * pool works on the rest, and the two run concurrently.
 *
 * Everything here degrades to "unavailable" when no Vulkan device or shader
 * can be set up; callers must then keep using the CPU path. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bq_vk bq_vk_t;

/* Opaque GPU buffer whose contents the CPU can also address. */
typedef struct bq_vk_buf bq_vk_buf_t;

/* Creates the backend; returns NULL when Vulkan is unusable here. */
bq_vk_t *bq_vk_create(void);
void bq_vk_destroy(bq_vk_t *vk);
int bq_vk_ready(const bq_vk_t *vk);

/* Allocates a buffer of `bytes` that is both GPU-visible and CPU-addressable.
 * On success returns the CPU pointer and, when `out` is non-NULL, the handle
 * needed to bind it for the GPU. */
void *bq_vk_buf_alloc(bq_vk_t *vk, size_t bytes, bq_vk_buf_t **out);
void bq_vk_buf_free(bq_vk_t *vk, bq_vk_buf_t *buf);

/* The same for a buffer the CPU writes once (weights) and then streams with
 * plain loads: it uses the cached host-visible heap when the device has one,
 * which on a Mali driver is where reading from the CPU is actually fast.  Call
 * bq_vk_buf_flush() after filling it, before the first dispatch reads it. */
void *bq_vk_buf_alloc_cached(bq_vk_t *vk, size_t bytes, bq_vk_buf_t **out);
void bq_vk_buf_flush(bq_vk_t *vk, bq_vk_buf_t *buf);

/* Queues rows [row0,row1) of a PTQ1_0 matrix for the GPU and returns at once.
 * `w` must be a buffer from bq_vk_buf_alloc holding the whole matrix from row
 * 0; `stride` is the packed row size and must be n_in/128*28.  The result
 * lands in y[row0..row1) once bq_vk_wait() returns. */
int bq_vk_ptq_submit(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                     int n_in, int row0, int row1, const float *x, float *y);

/* Batched form: several independent matvecs over one shared activation go
 * into a single command buffer and a single queue submission, because one
 * round trip costs about 0.6 ms on the drivers this targets whatever its
 * payload.  Begin packs `x` and reserves `max_rows` of output space; add
 * appends one matrix's rows [row0,row1), written to y[row0..row1) on wait;
 * end submits.  All segments of a batch must share x and n_in. */
int bq_vk_ptq_batch_begin(bq_vk_t *vk, const float *x, int n_in, int max_rows);
int bq_vk_ptq_batch_add(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                        int row0, int row1, float *y);
int bq_vk_ptq_batch_end(bq_vk_t *vk);

/* ---- batched PTQ1_0 matmat, the block verify of speculative decoding ----
 * One weight pass scores up to 16 positions, so each 28-byte block of trits
 * is decoded once and used B times.  `xq` is B rows already packed by
 * bitnet_quantize_row_i8_groups() into one [B][n_in] int8 block, with one
 * scale per row; every segment of the batch shares it, exactly like the
 * matvec batch.  add() writes y[b*ystride + row] for b < B. */
int bq_vk_ptq_matmat_available(const bq_vk_t *vk);
int bq_vk_ptq_matmat_begin(bq_vk_t *vk, const int8_t *xq, const float *xscale,
                           int n_in, int B, int max_rows);
int bq_vk_ptq_matmat_add(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                         int row0, int row1, int ystride, float *y);
int bq_vk_ptq_matmat_end(bq_vk_t *vk);

/* TQ2_0 (2-bit ternary) matvec over the tensor's own bytes, so the CPU can
 * stream the very same copy: `packed` is a bq_vk_buf_alloc buffer holding the
 * whole matrix from row 0 (the tensor as stored, 66-byte blocks) and `n_in`
 * its reduction length, a multiple of 256.  `x` is a plain float activation
 * of n_in values that every segment of the batch shares, exactly like the PTQ
 * batch, so a decode step's independent matrices still cost one round trip. */
int bq_vk_tq2_available(const bq_vk_t *vk);
int bq_vk_tq2_batch_begin(bq_vk_t *vk, const float *x, int n_in, int max_rows);
int bq_vk_tq2_batch_add(bq_vk_t *vk, const bq_vk_buf_t *packed,
                        int n_in, int row0, int row1, float *y);
int bq_vk_tq2_batch_end(bq_vk_t *vk);

/* Batched TQ2_0 matmat, the block verify of speculative decoding for the
 * MiniCPM family: B int8 activation rows against one weight row per
 * invocation, so each 66-byte block of 2-bit codes is decoded once and
 * DP4A'd B times.  `xq` is B rows already packed by bitnet_quantize_row_i8()
 * into one [B][n_in] int8 block with one scale per row, staged once for the
 * whole batch; add() writes y[b*ystride + row] for b < B, row in
 * [row0,row1), once wait returns.  `wrow0` is where those rows start inside
 * `packed`, so a caller may stage only the tail of a matrix and still have
 * the results land at their true row indices (pass wrow0 == row0 when
 * `packed` holds the whole matrix).  B is 1..8; matrices of a layer that
 * share an activation go into one submission. */
int bq_vk_tq2_matmat_available(const bq_vk_t *vk);
int bq_vk_tq2_matmat_begin(bq_vk_t *vk, const int8_t *xq, const float *xscale,
                           int n_in, int B, int max_rows);
int bq_vk_tq2_matmat_add(bq_vk_t *vk, const bq_vk_buf_t *packed, int wrow0,
                         int row0, int row1, int ystride, float *y);
int bq_vk_tq2_matmat_end(bq_vk_t *vk);

/* Waits for the last submit and copies the GPU rows into the callers' y. */
int bq_vk_wait(bq_vk_t *vk);

/* Resolves a CPU pointer that came out of bq_vk_buf_alloc to its GPU buffer,
 * so callers can keep working with plain pointers. */
bq_vk_buf_t *bq_vk_buf_lookup(const bq_vk_t *vk, const void *ptr);

/* Decode-step attention over a KV cache held in GPU-visible memory.
 * q is n_head*hd floats, k and v are nt slots of nkv*hd each laid out
 * [slot][kv head][d], out receives n_head*hd.  `scale` is 1/sqrt(hd) and `rep`
 * is n_head/nkv.  All of q/k/v must point into buffers from bq_vk_buf_alloc;
 * out is ordinary memory.  Runs scores, softmax and the weighted sum as three
 * dispatches in one submission. */
int bq_vk_attn(bq_vk_t *vk, const float *q, const float *k, const float *v,
               int n_head, int nkv, int hd, int nt, int rep, float scale, float *out);

#ifdef __cplusplus
}
#endif


/* ---- staged ops ---------------------------------------------------- *
 * These run the small per-layer stages on the device.  They are optional: a
 * return of -1 means the stage is unavailable and the caller keeps its CPU
 * path.  Each is a single dispatch that stages its inputs, runs, and copies
 * the result back, so no buffer has to stay resident between calls. */
#include "gpu_ops.h"

/* nrow rows of n values each; w is one row applied to every row. */
int bq_vk_elt(bq_vk_t *vk, int mode, const float *a, const float *b, const float *w,
              int n, int nrow, float eps, float *out);

/* Gated delta-net recurrence.  The state is kept on the device per `slot`
 * (one per GDN layer) and persists across tokens. */
int bq_vk_gdn(bq_vk_t *vk, int slot, const float *q, const float *k, const float *v,
              const float *decay, const float *beta, int nhead, int nkv, int S, float *out);

/* Causal depthwise conv over a K x n_ch history, with SiLU. */
int bq_vk_conv(bq_vk_t *vk, const float *w, const float *hist, int K, int n_ch, float *out);

/* Temperature / top-k / top-p sampling over n logits. */
int bq_vk_sample(bq_vk_t *vk, const float *logits, int n, int k, unsigned seed,
                 int greedy, float temperature, float top_p, uint32_t *token);

#endif
