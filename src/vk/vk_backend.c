#include "vk_backend.h"

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#if !defined(BITNET_ENABLE_VULKAN)
/* Backend compiled out: every entry point reports "unavailable". */
bq_vk_t *bq_vk_create(void) { return NULL; }
void bq_vk_destroy(bq_vk_t *vk) { (void)vk; }
int bq_vk_ready(const bq_vk_t *vk) { (void)vk; return 0; }
void *bq_vk_buf_alloc(bq_vk_t *vk, size_t bytes, bq_vk_buf_t **out) {
    (void)vk; (void)bytes; (void)out; return NULL;
}
void bq_vk_buf_free(bq_vk_t *vk, bq_vk_buf_t *buf) { (void)vk; (void)buf; }
int bq_vk_ptq_submit(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                     int n_in, int row0, int row1, const float *x, float *y) {
    (void)vk; (void)w; (void)stride; (void)n_in; (void)row0; (void)row1;
    (void)x; (void)y; return -1;
}
int bq_vk_ptq_batch_begin(bq_vk_t *vk, const float *x, int n_in, int max_rows) {
    (void)vk; (void)x; (void)n_in; (void)max_rows; return -1;
}
int bq_vk_ptq_batch_add(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                        int row0, int row1, float *y) {
    (void)vk; (void)w; (void)stride; (void)row0; (void)row1; (void)y; return -1;
}
int bq_vk_ptq_batch_end(bq_vk_t *vk) { (void)vk; return -1; }
int bq_vk_ptq_matmat_available(const bq_vk_t *vk) { (void)vk; return 0; }
int bq_vk_ptq_matmat_begin(bq_vk_t *vk, const int8_t *xq, const float *xscale,
                           int n_in, int B, int max_rows) {
    (void)vk; (void)xq; (void)xscale; (void)n_in; (void)B; (void)max_rows; return -1;
}
int bq_vk_ptq_matmat_add(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                         int row0, int row1, int ystride, float *y) {
    (void)vk; (void)w; (void)stride; (void)row0; (void)row1; (void)ystride; (void)y;
    return -1;
}
int bq_vk_ptq_matmat_end(bq_vk_t *vk) { (void)vk; return -1; }
int bq_vk_tq2_available(const bq_vk_t *vk) { (void)vk; return 0; }
int bq_vk_tq2_batch_begin(bq_vk_t *vk, const float *x, int n_in, int max_rows) {
    (void)vk; (void)x; (void)n_in; (void)max_rows; return -1;
}
int bq_vk_tq2_batch_add(bq_vk_t *vk, const bq_vk_buf_t *packed,
                        int n_in, int row0, int row1, float *y) {
    (void)vk; (void)packed; (void)n_in; (void)row0; (void)row1; (void)y; return -1;
}
int bq_vk_tq2_batch_end(bq_vk_t *vk) { (void)vk; return -1; }
int bq_vk_tq2_matmat_available(const bq_vk_t *vk) { (void)vk; return 0; }
int bq_vk_tq2_matmat_begin(bq_vk_t *vk, const int8_t *xq, const float *xscale,
                           int n_in, int B, int max_rows) {
    (void)vk; (void)xq; (void)xscale; (void)n_in; (void)B; (void)max_rows; return -1;
}
int bq_vk_tq2_matmat_add(bq_vk_t *vk, const bq_vk_buf_t *packed, int wrow0,
                         int row0, int row1, int ystride, float *y) {
    (void)vk; (void)packed; (void)wrow0; (void)row0; (void)row1; (void)ystride;
    (void)y; return -1;
}
int bq_vk_tq2_matmat_end(bq_vk_t *vk) { (void)vk; return -1; }
int bq_vk_wait(bq_vk_t *vk) { (void)vk; return -1; }
bq_vk_buf_t *bq_vk_buf_lookup(const bq_vk_t *vk, const void *ptr) {
    (void)vk; (void)ptr; return NULL;
}
int bq_vk_attn(bq_vk_t *vk, const float *q, const float *k, const float *v,
               int n_head, int nkv, int hd, int nt, int rep, float scale, float *out) {
    (void)vk; (void)q; (void)k; (void)v; (void)n_head; (void)nkv; (void)hd;
    (void)nt; (void)rep; (void)scale; (void)out; return -1;
}
int bq_vk_elt(bq_vk_t *vk, int mode, const float *a, const float *b, const float *w,
              int n, int nrow, float eps, float *out) {
    (void)vk; (void)mode; (void)a; (void)b; (void)w; (void)n; (void)nrow;
    (void)eps; (void)out; return -1;
}
int bq_vk_gdn(bq_vk_t *vk, int slot, const float *q, const float *k, const float *v,
              const float *decay, const float *beta, int nhead, int nkv, int S, float *out) {
    (void)vk; (void)slot; (void)q; (void)k; (void)v; (void)decay; (void)beta;
    (void)nhead; (void)nkv; (void)S; (void)out; return -1;
}
int bq_vk_conv(bq_vk_t *vk, const float *w, const float *hist, int K, int n_ch, float *out) {
    (void)vk; (void)w; (void)hist; (void)K; (void)n_ch; (void)out; return -1;
}
int bq_vk_sample(bq_vk_t *vk, const float *logits, int n, int k, unsigned seed,
                 int greedy, float temperature, float top_p, uint32_t *token) {
    (void)vk; (void)logits; (void)n; (void)k; (void)seed; (void)greedy;
    (void)temperature; (void)top_p; (void)token; return -1;
}
#else

#include <vulkan/vulkan.h>

#include "ptq_matvec_spv.h"
/* Optional alternative matvec kernel that decodes trits from a lookup table
 * instead of arithmetic.  Selected at runtime with BQ_VK_LUT=1 so the two can
 * be A/B'd without a rebuild; absent when glslangValidator was unavailable. */
#if defined(__has_include)
#  if __has_include("elt_spv.h") && __has_include("gdn_spv.h") && \
      __has_include("conv1d_spv.h") && __has_include("sample_spv.h")
#    include "elt_spv.h"
#    include "gdn_spv.h"
#    include "conv1d_spv.h"
#    include "sample_spv.h"
#    define BQ_VK_HAVE_STAGE_SPV 1
#  endif
#endif
/* Shared-memory staged activation variant of the one-row kernel.  It wins on
 * wide windows and loses on narrow ones, and the barrier is only safe on the
 * vendored Qualcomm driver, so it is created alongside the plain kernel and
 * selected per dispatch. */
#if defined(__has_include)
#  if __has_include("ptq_matvec_smem_spv.h")
#    include "ptq_matvec_smem_spv.h"
#    define BQ_VK_HAVE_SMEM_SPV 1
#  endif
#endif
#if defined(__has_include)
#  if __has_include("ptq_matvec_lut_spv.h")
#    include "ptq_matvec_lut_spv.h"
#    define BQ_VK_HAVE_LUT_SPV 1
#  endif
#endif
/* Batched matmat for the block verify: one weight pass, B activations. */
#if defined(__has_include)
#  if __has_include("ptq_matmat_spv.h")
#    include "ptq_matmat_spv.h"
#    define BQ_VK_HAVE_MATM_SPV 1
#  endif
#endif
/* The same matmat with no shared-memory staging: the activation words come
 * straight from the global buffer, which trades one L1 broadcast load for the
 * two workgroup barriers every 128-weight block costs in the staged variant. */
#if defined(__has_include)
#  if __has_include("ptq_matmat_g_spv.h")
#    include "ptq_matmat_g_spv.h"
#    define BQ_VK_HAVE_MMG_SPV 1
#  endif
#endif
/* TQ2_0 (ternary, 2 bits/weight) matvec over the I2_S packed layout.  This is
 * the quantisation the MiniCPM/BitCPM checkpoints ship, and its 4-rows-per-
 * byte packing means the GPU can stream the very same bytes the CPU kernel
 * reads, so no second copy of the weights is needed. */
#if defined(__has_include)
#  if __has_include("tq2_matvec_spv.h")
#    include "tq2_matvec_spv.h"
#    define BQ_VK_HAVE_TQ2_SPV 1
#  endif
#endif
/* The TQ2_0 block matmat, B activation rows against one weight row per
 * invocation: the verify of speculative decoding for the MiniCPM family. */
#if defined(__has_include)
#  if __has_include("tq2_matmat_spv.h")
#    include "tq2_matmat_spv.h"
#    define BQ_VK_HAVE_TQ2M_SPV 1
#  endif
#endif
/* A multi-row matvec (several output rows per invocation, sharing one set of
 * activation loads) was built and measured here: 18.7 G weights/s against 50.8
 * for the one-row kernel on an Adreno 740 at 17408 rows, and slower still on
 * the Adreno 650.  The extra live rows cost more in registers than the shared
 * loads save, so it is gone; the one-row kernel with the shared-memory staging
 * variant is what the runtime selects between. */
/* Attention shaders are optional: when they are missing the matvec backend
 * still comes up and attention falls back to the CPU path. */
#if defined(__has_include)
#  if __has_include("attn_scores_spv.h") && __has_include("attn_softmax_spv.h") && \
      __has_include("attn_av_spv.h")
#    include "attn_scores_spv.h"
#    include "attn_softmax_spv.h"
#    include "attn_av_spv.h"
#    define BQ_VK_HAVE_ATTN_SPV 1
#  endif
#endif

#if defined(BQ_VK_HAVE_STAGE_SPV)
/* Defined further down, next to the pipelines it creates: the create path
 * needs to call it as soon as the first pipeline is up. */
static int vk_setup_stages(bq_vk_t *vk);
#endif

/* The driver advertises maxStorageBufferRange = 128 MiB; a descriptor range
 * above that is invalid, so wide matrices are dispatched in row windows. */
#define BQ_VK_MAX_BIND ((VkDeviceSize)128u << 20)
/* Descriptor sets are written before each submit, because updating a set that
 * a recorded-but-unsubmitted command buffer already refers to takes effect at
 * submit time, not at record time. */
#define BQ_VK_MAX_SETS 32
/* Matvecs per batched submission; a decode step has at most four independent
 * ones per layer. */
#define BQ_VK_MAX_SEG 8
#define BQ_VK_LOCAL_SIZE 128
/* Window width from which the shared-memory staged kernel beats the plain one;
 * below it the barrier costs more than the activation reloads it saves. */
#define BQ_VK_SMEM_MIN_ROWS 6144

struct bq_vk_buf {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    void *ptr;
    int noncoherent;          /* mapped memory the GPU only sees after a flush */
};

struct bq_vk {
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    VkCommandPool cpool;
    VkCommandBuffer cb;
    VkFence fence;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool dpool;
    VkPipelineLayout pl;
    VkPipeline pipe;
    VkPipeline pipe_smem;    /* one-row matvec, activation staged in shared memory */
    VkPipeline pipe_tq2;     /* TQ2_0 / I2_S one-row matvec */
    VkPipeline pipe_mm;      /* PTQ1_0 matmat, B activation rows per pass */
    VkPipeline pipe_mmg;     /* the same, activation read from global memory */
    VkPipelineLayout mpl;    /* its layout: 16 bytes of dims + 16 scales */
    VkDescriptorSetLayout tdsl; /* TQ2_0 matmat, three bindings */
    VkPipelineLayout tmpl;   /* its layout: 20 bytes of dims */
    VkPipeline pipe_tq2m;    /* TQ2_0 matmat, B activation rows per pass */
    uint32_t mem_type;
    uint32_t mem_type_w;      /* cached host-visible type, for weights */
    int mem_type_w_coherent;
    uint32_t driver_id;      /* VkDriverId; 0 when the driver would not say */
    int ready;

    bq_vk_buf_t *xa;
    bq_vk_buf_t *xb;
    bq_vk_buf_t *yb;
    bq_vk_buf_t *lut;      /* 256 x vec4 trit table, digits 0..3 */
    bq_vk_buf_t *xt;       /* TQ2_0 batch activation, float, unpacked */

    int busy;
    double t_sub_end;
    double t_sub_start;
    /* segments recorded into the command buffer being built; one per matvec
     * of a batched group, copied back to their own y on wait */
    struct bq_vk_seg { float *y; int row0; int rows; size_t yoff; } seg[BQ_VK_MAX_SEG];
    int nseg;
    VkDeviceSize seg_rows;
    int batch_nb;

    /* matmat batch (used instead of the matvec batch, never alongside it) */
    bq_vk_buf_t *xm;        /* packed int8 activations, [B][n_in] */
    bq_vk_buf_t *ymm;       /* its results, [B][rows] per segment */
    bq_vk_buf_t *luti;      /* int trit tables for the DP4A decoder */
    int mm_B;
    int mm_nin;
    float mm_scale[16];
    struct bq_vk_mseg { float *y; int row0; int rows; size_t yoff; int ystride; } mseg[BQ_VK_MAX_SEG];
    int mnseg;
    size_t mseg_floats;
    size_t mm_cap_floats;   /* begin() sized the result buffer to this */

    /* TQ2_0 matmat batch (the minicpm verify), same shape as the PTQ one */
    bq_vk_buf_t *xtm;       /* int8 activations, [8][n_in] */
    bq_vk_buf_t *ytm;       /* its results, [B][rows] per segment */
    bq_vk_buf_t *lutt;      /* the 2-bit code table for its decoder */
    int tq2m_B;
    int tq2m_nin;
    float tq2m_scale[8];
    struct bq_vk_tseg { float *y; int row0; int rows; size_t yoff; int ystride; } tseg[BQ_VK_MAX_SEG];
    int tnseg;
    size_t tseg_floats;
    size_t tm_cap_floats;

    /* attention over the KV cache */
    VkDescriptorSetLayout adsl;
    VkPipelineLayout apl;
    VkPipeline ascores, asoft, aav;
    bq_vk_buf_t *as;        /* scores, then probabilities in place, n_head*nt */
    bq_vk_buf_t *aout;      /* n_head*hd */

    /* registry so a bare CPU pointer can be resolved to its GPU buffer */
    struct bq_vk_reg { const void *ptr; size_t size; bq_vk_buf_t *buf; } reg[4096];
    int n_reg;
    /* staged ops: one descriptor layout and pipeline layout shared by every
     * stage, since they all take flat buffers plus push constants */
    VkDescriptorSetLayout sdsl;
    VkPipelineLayout spl;
    VkPipeline pelt, pgdn, pconv, psamp;
    bq_vk_buf_t *st[8];              /* staging buffers, reused by every call */
    bq_vk_buf_t *gdn_state[64];      /* per-GDN-layer recurrent state */
    size_t gdn_bytes[64];
};

#define BQ_VK_MAX(a, b) ((a) > (b) ? (a) : (b))

/* Submit/wait accounting, printed at teardown when BQ_PROFILE=1.  It answers
 * "how much of a decode step is round-trip tax": submit is CPU time inside
 * bq_vk_ptq_submit, span is the time between returning from submit and calling
 * wait (the CPU's own rows fill it), and wait is what is left over. */
static double g_vk_t_sub, g_vk_t_span, g_vk_t_wait;
static long long g_vk_n_sub, g_vk_n_wait;

static double vk_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int vk_stats_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("BQ_PROFILE"); on = (e != NULL && e[0] == '1'); }
    return on;
}

static void vk_note(const char *what, VkResult r) {
    if (getenv("BQ_VK_DEBUG") != NULL) fprintf(stderr, "vk: %s failed (%d)\n", what, (int)r);
}

/* ---------------------------------------------------------------- */
/* buffers                                                          */
/* ---------------------------------------------------------------- */

static void *vk_buf_alloc_mt(bq_vk_t *vk, size_t bytes, bq_vk_buf_t **out,
                             uint32_t mem_type, int noncoherent) {
    bq_vk_buf_t *b;
    VkBufferCreateInfo bci;
    VkMemoryRequirements mr;
    VkMemoryAllocateInfo mai;
    VkResult r;
    void *ptr = NULL;

    if (vk == NULL || !vk->ready || bytes == 0) return NULL;
    b = (bq_vk_buf_t *)calloc(1, sizeof(*b));
    if (b == NULL) return NULL;

    bci = (VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = (VkDeviceSize)bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if ((r = vkCreateBuffer(vk->dev, &bci, NULL, &b->buf)) != VK_SUCCESS) {
        vk_note("vkCreateBuffer", r);
        free(b);
        return NULL;
    }
    vkGetBufferMemoryRequirements(vk->dev, b->buf, &mr);

    mai = (VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mem_type;
    if ((r = vkAllocateMemory(vk->dev, &mai, NULL, &b->mem)) != VK_SUCCESS) {
        vk_note("vkAllocateMemory", r);
        vkDestroyBuffer(vk->dev, b->buf, NULL);
        free(b);
        return NULL;
    }
    if ((r = vkBindBufferMemory(vk->dev, b->buf, b->mem, 0)) != VK_SUCCESS) {
        vk_note("vkBindBufferMemory", r);
        vkFreeMemory(vk->dev, b->mem, NULL);
        vkDestroyBuffer(vk->dev, b->buf, NULL);
        free(b);
        return NULL;
    }
    if ((r = vkMapMemory(vk->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &ptr)) != VK_SUCCESS) {
        vk_note("vkMapMemory", r);
        vkFreeMemory(vk->dev, b->mem, NULL);
        vkDestroyBuffer(vk->dev, b->buf, NULL);
        free(b);
        return NULL;
    }
    b->size = mr.size;
    b->ptr = ptr;
    b->noncoherent = noncoherent;
    if (out != NULL) *out = b;
    if (vk->n_reg < (int)(sizeof(vk->reg) / sizeof(vk->reg[0]))) {
        vk->reg[vk->n_reg].ptr = ptr;
        vk->reg[vk->n_reg].size = (size_t)mr.size;
        vk->reg[vk->n_reg].buf = b;
        ++vk->n_reg;
    }
    return ptr;
}

void *bq_vk_buf_alloc(bq_vk_t *vk, size_t bytes, bq_vk_buf_t **out) {
    return vk_buf_alloc_mt(vk, bytes, out, vk->mem_type, 0);
}

/* Weights stay read-only once loaded, so they can live in the cached host
 * visible heap even when that heap is not coherent: one flush after the copy
 * is all the GPU needs, and the CPU keeps full-speed load/store access to the
 * pages it streams row by row.  On a Mali device the plain coherent type is
 * uncached and reading it costs about eight times the throughput, which is
 * what this exists to avoid. */
void *bq_vk_buf_alloc_cached(bq_vk_t *vk, size_t bytes, bq_vk_buf_t **out) {
    if (vk == NULL) return NULL;
    return vk_buf_alloc_mt(vk, bytes, out, vk->mem_type_w,
                           vk->mem_type_w_coherent ? 0 : 1);
}

void bq_vk_buf_flush(bq_vk_t *vk, bq_vk_buf_t *buf) {
    VkMappedMemoryRange mr;
    if (vk == NULL || buf == NULL || !buf->noncoherent) return;
    mr = (VkMappedMemoryRange){ .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    mr.memory = buf->mem;
    mr.offset = 0;
    mr.size = VK_WHOLE_SIZE;
    (void)vkFlushMappedMemoryRanges(vk->dev, 1, &mr);
}

/* Finds the allocation that contains `ptr`, so callers can keep passing plain
 * CPU pointers (all Vulkan memory here is host visible). */
bq_vk_buf_t *bq_vk_buf_lookup(const bq_vk_t *vk, const void *ptr) {
    int i;
    if (vk == NULL || ptr == NULL) return NULL;
    for (i = 0; i < vk->n_reg; ++i) {
        const char *base = (const char *)vk->reg[i].ptr;
        const char *p = (const char *)ptr;
        if (p >= base && p < base + vk->reg[i].size) return vk->reg[i].buf;
    }
    return NULL;
}

static size_t vk_reg_offset(const bq_vk_t *vk, const void *ptr) {
    int i;
    for (i = 0; i < vk->n_reg; ++i) {
        const char *base = (const char *)vk->reg[i].ptr;
        const char *p = (const char *)ptr;
        if (p >= base && p < base + vk->reg[i].size) return (size_t)(p - base);
    }
    return 0;
}

void bq_vk_buf_free(bq_vk_t *vk, bq_vk_buf_t *buf) {
    int i;
    if (vk == NULL || buf == NULL) return;
    for (i = 0; i < vk->n_reg; ++i) {
        if (vk->reg[i].buf == buf) {
            vk->reg[i] = vk->reg[vk->n_reg - 1];
            --vk->n_reg;
            break;
        }
    }
    if (buf->ptr != NULL) vkUnmapMemory(vk->dev, buf->mem);
    vkFreeMemory(vk->dev, buf->mem, NULL);
    vkDestroyBuffer(vk->dev, buf->buf, NULL);
    free(buf);
}

/* Grows a scratch buffer to at least `bytes`; returns the CPU pointer. */
static void *vk_scratch(bq_vk_t *vk, bq_vk_buf_t **slot, size_t bytes) {
    if (*slot != NULL && (*slot)->size >= bytes) return (*slot)->ptr;
    bq_vk_buf_free(vk, *slot);
    *slot = NULL;
    return bq_vk_buf_alloc(vk, bytes, slot);
}

/* ---------------------------------------------------------------- */
/* context                                                          */
/* ---------------------------------------------------------------- */

static int vk_pick_memory(bq_vk_t *vk) {
    VkPhysicalDeviceMemoryProperties mp;
    uint32_t i;
    int best = -1, best_score = -1;

    vkGetPhysicalDeviceMemoryProperties(vk->pd, &mp);
    for (i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        int score;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
        if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) continue;
        score = (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 2 : 1;
        if (score > best_score) {
            best_score = score;
            best = (int)i;
        }
    }
    if (best < 0) return -1;
    vk->mem_type = (uint32_t)best;
    /* Second choice for weights: any host-visible memory that is cached,
     * coherent or not (a flush follows the one copy made into it). */
    {
        int wbest = -1, wscore = -1;
        for (i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            int score;
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
            if ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) == 0) continue;
            score = (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 2 : 1;
            if (score > wscore) { wscore = score; wbest = (int)i; }
        }
        if (wbest < 0) { wbest = best; wscore = 0; }
        vk->mem_type_w = (uint32_t)wbest;
        vk->mem_type_w_coherent = (wscore == 2);
    }
    if (getenv("BQ_VK_DEBUG") != NULL) {
        uint32_t k;
        for (k = 0; k < mp.memoryTypeCount; ++k)
            fprintf(stderr, "vk: mem type %u heap %u flags 0x%x%s\n", k,
                    (unsigned)mp.memoryTypes[k].heapIndex,
                    (unsigned)mp.memoryTypes[k].propertyFlags,
                    (int)k == best ? "  <- chosen"
                                   : ((int)k == (int)vk->mem_type_w ? "  <- weights" : ""));
        for (k = 0; k < mp.memoryHeapCount; ++k)
            fprintf(stderr, "vk: heap %u size %.2f GB flags 0x%x\n", k,
                    (double)mp.memoryHeaps[k].size / 1e9,
                    (unsigned)mp.memoryHeaps[k].flags);
    }
    return 0;
}

int bq_vk_ready(const bq_vk_t *vk) { return vk != NULL && vk->ready; }

static void vk_teardown(bq_vk_t *vk) {
    if (vk == NULL) return;
    if (vk_stats_on() && g_vk_n_sub > 0) {
        fprintf(stderr, "[vkprof] %lld submits, %lld waits: submit %.3f s, span %.3f s, "
                        "wait %.3f s; %.1f us/submit, %.1f us/wait\n",
                g_vk_n_sub, g_vk_n_wait, g_vk_t_sub, g_vk_t_span, g_vk_t_wait,
                1e6 * g_vk_t_sub / (double)g_vk_n_sub,
                g_vk_n_wait ? 1e6 * g_vk_t_wait / (double)g_vk_n_wait : 0.0);
    }
    if (vk->dev != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vk->dev);
        bq_vk_buf_free(vk, vk->xa);
        bq_vk_buf_free(vk, vk->xb);
        bq_vk_buf_free(vk, vk->yb);
        bq_vk_buf_free(vk, vk->lut);
        bq_vk_buf_free(vk, vk->xm);
        bq_vk_buf_free(vk, vk->ymm);
        bq_vk_buf_free(vk, vk->luti);
        bq_vk_buf_free(vk, vk->xtm);
        bq_vk_buf_free(vk, vk->ytm);
        bq_vk_buf_free(vk, vk->lutt);
        if (vk->pipe_tq2m != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->pipe_tq2m, NULL);
        if (vk->tmpl != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->dev, vk->tmpl, NULL);
        if (vk->tdsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->dev, vk->tdsl, NULL);
        if (vk->pipe_mm != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->pipe_mm, NULL);
        if (vk->mpl != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->dev, vk->mpl, NULL);
        if (vk->pipe != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->pipe, NULL);
        if (vk->ascores != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->ascores, NULL);
        if (vk->asoft != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->asoft, NULL);
        if (vk->aav != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->aav, NULL);
        if (vk->pelt != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->pelt, NULL);
        if (vk->pgdn != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->pgdn, NULL);
        if (vk->pconv != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->pconv, NULL);
        if (vk->psamp != VK_NULL_HANDLE) vkDestroyPipeline(vk->dev, vk->psamp, NULL);
        if (vk->spl != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->dev, vk->spl, NULL);
        if (vk->sdsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->dev, vk->sdsl, NULL);
        if (vk->apl != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->dev, vk->apl, NULL);
        if (vk->adsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->dev, vk->adsl, NULL);
        if (vk->pl != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->dev, vk->pl, NULL);
        if (vk->dpool != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk->dev, vk->dpool, NULL);
        if (vk->dsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->dev, vk->dsl, NULL);
        if (vk->fence != VK_NULL_HANDLE) vkDestroyFence(vk->dev, vk->fence, NULL);
        if (vk->cpool != VK_NULL_HANDLE) vkDestroyCommandPool(vk->dev, vk->cpool, NULL);
        vkDestroyDevice(vk->dev, NULL);
    }
    if (vk->inst != VK_NULL_HANDLE) vkDestroyInstance(vk->inst, NULL);
    free(vk);
}

/* Builds the three attention pipelines.  Failure is not fatal: the matvec
 * path keeps working and bq_vk_attn() simply reports "unavailable". */
#if defined(BQ_VK_HAVE_ATTN_SPV)
static int vk_setup_attn(bq_vk_t *vk) {
    VkDescriptorSetLayoutBinding b[3];
    VkDescriptorSetLayoutCreateInfo dlci;
    VkPushConstantRange pcr;
    VkPipelineLayoutCreateInfo plci;
    const struct { const char *name; const uint8_t *code; size_t len; VkPipeline *slot; } sh[3] = {
        { "scores",  bq_vk_attn_scores_spv,  bq_vk_attn_scores_spv_len,  &vk->ascores },
        { "softmax", bq_vk_attn_softmax_spv, bq_vk_attn_softmax_spv_len, &vk->asoft },
        { "av",      bq_vk_attn_av_spv,      bq_vk_attn_av_spv_len,      &vk->aav },
    };
    int i, made = 0;
    VkResult r;

    for (i = 0; i < 3; ++i) {
        b[i] = (VkDescriptorSetLayoutBinding){ 0 };
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dlci = (VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 3;
    dlci.pBindings = b;
    r = vkCreateDescriptorSetLayout(vk->dev, &dlci, NULL, &vk->adsl);
    if (r != VK_SUCCESS) { vk_note("attn descriptor layout", r); vk->adsl = VK_NULL_HANDLE; return -1; }

    pcr = (VkPushConstantRange){ 0 };
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 32;
    plci = (VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &vk->adsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    r = vkCreatePipelineLayout(vk->dev, &plci, NULL, &vk->apl);
    if (r != VK_SUCCESS) { vk_note("attn pipeline layout", r); vk->apl = VK_NULL_HANDLE; return -1; }

    for (i = 0; i < 3; ++i) {
        VkShaderModule sm = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        VkComputePipelineCreateInfo cpci;
        smci.codeSize = sh[i].len;
        smci.pCode = (const uint32_t *)sh[i].code;
        r = vkCreateShaderModule(vk->dev, &smci, NULL, &sm);
        if (r != VK_SUCCESS) { vk_note("attn shader module", r); return -1; }
        cpci = (VkComputePipelineCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        cpci.layout = vk->apl;
        r = vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL, sh[i].slot);
        vkDestroyShaderModule(vk->dev, sm, NULL);
        if (r != VK_SUCCESS) {
            if (getenv("BQ_VK_DEBUG") != NULL)
                fprintf(stderr, "vk: attn '%s' pipeline failed (%d)\n", sh[i].name, (int)r);
            *sh[i].slot = VK_NULL_HANDLE;
            continue;
        }
        ++made;
    }
    return made > 0 ? 0 : -1;
}
#else
static int vk_setup_attn(bq_vk_t *vk) { (void)vk; return -1; }
#endif

bq_vk_t *bq_vk_create(void) {
    bq_vk_t *vk = (bq_vk_t *)calloc(1, sizeof(*vk));
    VkApplicationInfo ai;
    VkInstanceCreateInfo ici;
    VkPhysicalDevice *pds = NULL;
    uint32_t npd = 0, i;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci;
    VkDeviceCreateInfo dci;
    VkCommandPoolCreateInfo cpi;
    VkCommandBufferAllocateInfo cbai;
    VkFenceCreateInfo fci;
    VkDescriptorSetLayoutBinding binds[5];
    VkDescriptorSetLayoutCreateInfo dlci;
    VkDescriptorPoolSize dps[1];
    VkDescriptorPoolCreateInfo dpci;
    VkPushConstantRange pcr;
    VkPipelineLayoutCreateInfo plci;
    VkShaderModuleCreateInfo smci;
    VkShaderModule sm = VK_NULL_HANDLE;
    VkComputePipelineCreateInfo cpci;
    VkResult r;
    int fam = -1;

    if (vk == NULL) return NULL;

    ai = (VkApplicationInfo){ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.pApplicationName = "bitnet-bonsai";
    ai.apiVersion = VK_API_VERSION_1_2;
    ici = (VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    if ((r = vkCreateInstance(&ici, NULL, &vk->inst)) != VK_SUCCESS) {
        vk_note("vkCreateInstance", r);
        free(vk);
        return NULL;
    }

    if ((r = vkEnumeratePhysicalDevices(vk->inst, &npd, NULL)) != VK_SUCCESS || npd == 0) {
        vk_note("vkEnumeratePhysicalDevices", r);
        vk_teardown(vk);
        return NULL;
    }
    pds = (VkPhysicalDevice *)calloc(npd, sizeof(*pds));
    if (pds == NULL || vkEnumeratePhysicalDevices(vk->inst, &npd, pds) != VK_SUCCESS) {
        free(pds);
        vk_teardown(vk);
        return NULL;
    }

    for (i = 0; i < npd && fam < 0; ++i) {
        uint32_t nfam = 0, f;
        VkQueueFamilyProperties *qfp;
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &nfam, NULL);
        qfp = (VkQueueFamilyProperties *)calloc(nfam, sizeof(*qfp));
        if (qfp == NULL) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &nfam, qfp);
        for (f = 0; f < nfam; ++f) {
            if ((qfp[f].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfp[f].queueCount > 0) {
                vk->pd = pds[i];
                fam = (int)f;
                break;
            }
        }
        free(qfp);
    }
    free(pds);
    if (fam < 0) {
        vk_teardown(vk);
        return NULL;
    }
    vk->qfam = (uint32_t)fam;

    /* The instance is created at Vulkan 1.2, so this struct is core even on a
     * 1.1 device; a driver that ignores the chain leaves driverID at 0 and
     * then the staged kernel is not offered. */
    {
        VkPhysicalDeviceDriverPropertiesKHR dp =
            { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR };
        VkPhysicalDeviceProperties2 p2 =
            { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dp };
        vkGetPhysicalDeviceProperties2(vk->pd, &p2);
        vk->driver_id = (uint32_t)dp.driverID;
        if (getenv("BQ_VK_DEBUG") != NULL)
            fprintf(stderr, "vk: driver '%s' id %u\n", dp.driverName, (unsigned)dp.driverID);
    }

    prio = 1.0f;
    qci = (VkDeviceQueueCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = vk->qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    dci = (VkDeviceCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if ((r = vkCreateDevice(vk->pd, &dci, NULL, &vk->dev)) != VK_SUCCESS) {
        vk_note("vkCreateDevice", r);
        vk->dev = VK_NULL_HANDLE;
        vk_teardown(vk);
        return NULL;
    }
    vkGetDeviceQueue(vk->dev, vk->qfam, 0, &vk->queue);
    if (vk_pick_memory(vk) != 0) {
        vk_teardown(vk);
        return NULL;
    }

    cpi = (VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = vk->qfam;
    r = vkCreateCommandPool(vk->dev, &cpi, NULL, &vk->cpool);
    if (r != VK_SUCCESS) { vk_note("vkCreateCommandPool", r); vk_teardown(vk); return NULL; }
    cbai = (VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = vk->cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(vk->dev, &cbai, &vk->cb);
    if (r != VK_SUCCESS) { vk_note("vkAllocateCommandBuffers", r); vk_teardown(vk); return NULL; }
    fci = (VkFenceCreateInfo){ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    r = vkCreateFence(vk->dev, &fci, NULL, &vk->fence);
    if (r != VK_SUCCESS) { vk_note("vkCreateFence", r); vk_teardown(vk); return NULL; }

    for (i = 0; i < 5; ++i) {
        binds[i] = (VkDescriptorSetLayoutBinding){ 0 };
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dlci = (VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 5;
    dlci.pBindings = binds;
    r = vkCreateDescriptorSetLayout(vk->dev, &dlci, NULL, &vk->dsl);
    if (r != VK_SUCCESS) { vk_note("vkCreateDescriptorSetLayout", r); vk_teardown(vk); return NULL; }

    dps[0] = (VkDescriptorPoolSize){ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, BQ_VK_MAX_SETS * 5 };
    dpci = (VkDescriptorPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = BQ_VK_MAX_SETS;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = dps;
    r = vkCreateDescriptorPool(vk->dev, &dpci, NULL, &vk->dpool);
    if (r != VK_SUCCESS) { vk_note("vkCreateDescriptorPool", r); vk_teardown(vk); return NULL; }

    pcr = (VkPushConstantRange){ 0 };
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 16;
    plci = (VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &vk->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    r = vkCreatePipelineLayout(vk->dev, &plci, NULL, &vk->pl);
    if (r != VK_SUCCESS) { vk_note("vkCreatePipelineLayout", r); vk_teardown(vk); return NULL; }

    {
        const uint8_t *code = bq_vk_ptq_matvec_spv;
        size_t code_len = bq_vk_ptq_matvec_spv_len;
#if defined(BQ_VK_HAVE_LUT_SPV)
        const char *e = getenv("BQ_VK_LUT");
        if (e != NULL && e[0] == '1') {
            code = bq_vk_ptq_matvec_lut_spv;
            code_len = bq_vk_ptq_matvec_lut_spv_len;
        }
#endif
        smci = (VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = code_len;
        smci.pCode = (const uint32_t *)code;
    }
    r = vkCreateShaderModule(vk->dev, &smci, NULL, &sm);
    if (r != VK_SUCCESS) { vk_note("vkCreateShaderModule", r); vk_teardown(vk); return NULL; }

    cpci = (VkComputePipelineCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = vk->pl;
    r = vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &vk->pipe);
    vkDestroyShaderModule(vk->dev, sm, NULL);
    if (r != VK_SUCCESS) { vk_note("vkCreateComputePipelines", r); vk_teardown(vk); return NULL; }

#if defined(BQ_VK_HAVE_SMEM_SPV)
    /* Only the vendored Qualcomm driver is known to survive a workgroup
     * barrier here; Mesa's Turnip locks the GPU up and needs a reboot, so it
     * has to be asked for explicitly with BQ_VK_SMEM=1. */
    {
        const char *e = getenv("BQ_VK_SMEM");
        int want = (vk->driver_id == 8u /* VK_DRIVER_ID_QUALCOMM_PROPRIETARY */) ||
                   (e != NULL && e[0] == '1');
        if (e != NULL && e[0] == '0') want = 0;
        if (want) {
            smci = (VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            smci.codeSize = bq_vk_ptq_matvec_smem_spv_len;
            smci.pCode = (const uint32_t *)bq_vk_ptq_matvec_smem_spv;
            if (vkCreateShaderModule(vk->dev, &smci, NULL, &sm) == VK_SUCCESS) {
                cpci.stage.module = sm;
                if (vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                             &vk->pipe_smem) != VK_SUCCESS)
                    vk->pipe_smem = VK_NULL_HANDLE;
                vkDestroyShaderModule(vk->dev, sm, NULL);
            }
            if (getenv("BQ_VK_DEBUG") != NULL)
                fprintf(stderr, "vk: staged matvec %s\n",
                        vk->pipe_smem != VK_NULL_HANDLE ? "enabled" : "unavailable");
        }
    }
#endif

#if defined(BQ_VK_HAVE_TQ2_SPV)
    /* Failure here is not fatal: the caller keeps its CPU TQ2_0 path. */
    smci = (VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = bq_vk_tq2_matvec_spv_len;
    smci.pCode = (const uint32_t *)bq_vk_tq2_matvec_spv;
    if (vkCreateShaderModule(vk->dev, &smci, NULL, &sm) == VK_SUCCESS) {
        cpci.stage.module = sm;
        if (vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                     &vk->pipe_tq2) != VK_SUCCESS)
            vk->pipe_tq2 = VK_NULL_HANDLE;
        vkDestroyShaderModule(vk->dev, sm, NULL);
    }
    if (getenv("BQ_VK_DEBUG") != NULL)
        fprintf(stderr, "vk: tq2 matvec %s\n",
                vk->pipe_tq2 != VK_NULL_HANDLE ? "enabled" : "unavailable");
#endif

#if defined(BQ_VK_HAVE_MATM_SPV)
    /* Matmat pipeline: 16 bytes of dims plus 16 activation scales, so it gets
     * its own pipeline layout rather than the matvec one's 16 bytes. */
    {
        pcr.size = 16 + 16 * sizeof(float);
        plci = (VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &vk->dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(vk->dev, &plci, NULL, &vk->mpl) == VK_SUCCESS) {
            smci = (VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            smci.codeSize = bq_vk_ptq_matmat_spv_len;
            smci.pCode = (const uint32_t *)bq_vk_ptq_matmat_spv;
            if (vkCreateShaderModule(vk->dev, &smci, NULL, &sm) == VK_SUCCESS) {
                cpci.stage.module = sm;
                cpci.layout = vk->mpl;
                if (vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                             &vk->pipe_mm) != VK_SUCCESS)
                    vk->pipe_mm = VK_NULL_HANDLE;
                vkDestroyShaderModule(vk->dev, sm, NULL);
            }
#if defined(BQ_VK_HAVE_MMG_SPV)
            /* Same layout, same bindings, only the barrier schedule differs. */
            smci.codeSize = bq_vk_ptq_matmat_g_spv_len;
            smci.pCode = (const uint32_t *)bq_vk_ptq_matmat_g_spv;
            if (vkCreateShaderModule(vk->dev, &smci, NULL, &sm) == VK_SUCCESS) {
                cpci.stage.module = sm;
                cpci.layout = vk->mpl;
                if (vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                             &vk->pipe_mmg) != VK_SUCCESS)
                    vk->pipe_mmg = VK_NULL_HANDLE;
                vkDestroyShaderModule(vk->dev, sm, NULL);
            }
#endif
        }
        if (getenv("BQ_VK_DEBUG") != NULL)
            fprintf(stderr, "vk: matmat %s (global-read variant %s)\n",
                    vk->pipe_mm != VK_NULL_HANDLE ? "enabled" : "unavailable",
                    vk->pipe_mmg != VK_NULL_HANDLE ? "enabled" : "unavailable");
    }
#endif

#if defined(BQ_VK_HAVE_TQ2M_SPV)
    /* TQ2_0 matmat: its own three-binding layout, because the TQ2 shaders
     * read the weight words raw and take no trit tables. */
    {
        VkDescriptorSetLayoutBinding tb[4];
        for (i = 0; i < 4; ++i) {
            tb[i] = (VkDescriptorSetLayoutBinding){ 0 };
            tb[i].binding = i;
            tb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            tb[i].descriptorCount = 1;
            tb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        dlci = (VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dlci.bindingCount = 4;
        dlci.pBindings = tb;
        if (vkCreateDescriptorSetLayout(vk->dev, &dlci, NULL, &vk->tdsl) == VK_SUCCESS) {
            pcr = (VkPushConstantRange){ 0 };
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.size = 20 + 8 * sizeof(float);
            plci = (VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &vk->tdsl;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcr;
            if (vkCreatePipelineLayout(vk->dev, &plci, NULL, &vk->tmpl) == VK_SUCCESS) {
                smci = (VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                smci.codeSize = bq_vk_tq2_matmat_spv_len;
                smci.pCode = (const uint32_t *)bq_vk_tq2_matmat_spv;
                if (vkCreateShaderModule(vk->dev, &smci, NULL, &sm) == VK_SUCCESS) {
                    cpci.stage.module = sm;
                    cpci.layout = vk->tmpl;
                    if (vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                                 &vk->pipe_tq2m) != VK_SUCCESS)
                        vk->pipe_tq2m = VK_NULL_HANDLE;
                    vkDestroyShaderModule(vk->dev, sm, NULL);
                }
            }
        }
        if (getenv("BQ_VK_DEBUG") != NULL)
            fprintf(stderr, "vk: tq2 matmat %s\n",
                    vk->pipe_tq2m != VK_NULL_HANDLE ? "enabled" : "unavailable");
    }
#endif
    vk->ready = 1;
    (void)vk_setup_attn(vk);
#if defined(BQ_VK_HAVE_STAGE_SPV)
    (void)vk_setup_stages(vk);
#endif

    /* Trit lookup table.  lut[b*4 + 0..3] are the four low digits of packed
     * byte b and lut[1024 + b] is the fifth; the arithmetic kernel ignores the
     * second half, the BQ_VK_LUT kernel needs it. */
    {
        float *lt = (float *)bq_vk_buf_alloc(vk, 5120, &vk->lut);
        unsigned b;
        if (lt == NULL) { vk_note("lut buffer", VK_ERROR_OUT_OF_DEVICE_MEMORY); vk_teardown(vk); return NULL; }
        for (b = 0; b < 256u; ++b) {
            const unsigned p3[5] = { 1u, 3u, 9u, 27u, 81u };
            int k;
            for (k = 0; k < 4; ++k)
                lt[b * 4 + k] = (float)(int)((((b * p3[k]) & 0xFFu) * 3u) >> 8) - 1.0f;
            lt[1024 + b] = (float)(int)((((b * p3[4]) & 0xFFu) * 3u) >> 8) - 1.0f;
        }
    }
#if defined(BQ_VK_HAVE_MATM_SPV)
    /* Integer trit tables for the DP4A decoder: lt4[b] is the four low digits
     * of packed byte b as an int8x4, lt5[b] the fifth as a signed byte. */
    {
        uint32_t *lt = (uint32_t *)bq_vk_buf_alloc(vk, 2048, &vk->luti);
        unsigned b;
        if (lt != NULL) {
            for (b = 0; b < 256u; ++b) {
                const unsigned p3[5] = { 1u, 3u, 9u, 27u, 81u };
                unsigned v = 0, k;
                for (k = 0; k < 4; ++k) {
                    unsigned t = ((((b * p3[k]) & 0xFFu) * 3u) >> 8) - 1u;
                    v |= (t & 0xFFu) << (8u * k);
                }
                lt[b] = v;
                lt[256 + b] = ((((b * 81u) & 0xFFu) * 3u) >> 8) - 1u & 0xFFu;
            }
        }
    }
#endif
#if defined(BQ_VK_HAVE_TQ2M_SPV)
    /* lt2[b] is the four 2-bit codes of byte b minus one, as an int8x4: the
     * TQ2 group order is the stored byte order, so one lookup decodes the four
     * weights an int8 activation word scores. */
    {
        uint32_t *lt = (uint32_t *)bq_vk_buf_alloc(vk, 1024, &vk->lutt);
        unsigned b;
        if (lt != NULL)
            for (b = 0; b < 256u; ++b) {
                unsigned k, v = 0;
                for (k = 0; k < 4u; ++k)
                    v |= (((b >> (2u * k)) & 3u) - 1u & 0xFFu) << (8u * k);
                lt[b] = v;
            }
    }
#endif
    return vk;
}

void bq_vk_destroy(bq_vk_t *vk) {
    if (vk == NULL) return;
    vk_teardown(vk);
}

/* ---------------------------------------------------------------- */
/* PTQ1_0 matvec                                                    */
/* ---------------------------------------------------------------- */

/* Packs x into the split activation layout the shader expects:
 *   xa[b*26 + m] = vec4(x[b*128 + m], x[b*128 + 16 + m], +32, +48)
 *   xb[b*24 + m] = x[b*128 + 64 + m]                 (m < 16)
 *   xa[b*26 + 16 + k] = vec4(x[+80+k], +88, +96, +104), xb[b*24+16+k] = x[+112+k]
 *   xa[b*26 + 24 + h] = vec4(x[+120+h], +122+h, +124+h, +126+h)
 */
static void vk_pack_x(float *xa, float *xb, const float *x, int nb) {
    int b;
    for (b = 0; b < nb; ++b) {
        const float *x0 = x + (size_t)b * 128;
        float *va = xa + (size_t)b * 104;
        float *vb = xb + (size_t)b * 24;
        int m, d;
        for (m = 0; m < 16; ++m) {
            for (d = 0; d < 4; ++d) va[m * 4 + d] = x0[d * 16 + m];
            vb[m] = x0[64 + m];
        }
        for (m = 0; m < 8; ++m) {
            for (d = 0; d < 4; ++d) va[(16 + m) * 4 + d] = x0[80 + d * 8 + m];
            vb[16 + m] = x0[112 + m];
        }
        for (m = 0; m < 2; ++m) {
            for (d = 0; d < 4; ++d) va[(24 + m) * 4 + d] = x0[120 + 2 * d + m];
        }
    }
}

/* Binds one row window of the matrix and records a single dispatch over it.
 * yoff is the byte offset into the result buffer where this window's rows go,
 * so windows can be handed out in any order. */
static int vk_emit_rows(bq_vk_t *vk, VkPipeline pipe, const bq_vk_buf_t *w, size_t stride,
                        int nb, int base, int n, VkDeviceSize yoff, uint32_t groups) {
    VkDescriptorSetAllocateInfo dsai;
    VkDescriptorBufferInfo dbi[5];
    VkWriteDescriptorSet wr[5];
    VkDescriptorSet ds;
    uint32_t pc[4];
    VkResult r;
    int i;

#if defined(BQ_VK_HAVE_SMEM_SPV)
    if (pipe == vk->pipe && vk->pipe_smem != VK_NULL_HANDLE) {
        const char *e = getenv("BQ_VK_SMEM");
        int force = (e != NULL && e[0] == '1');
        int never = (e != NULL && e[0] == '0');
        if (!never && (force || n >= BQ_VK_SMEM_MIN_ROWS)) pipe = vk->pipe_smem;
    }
#endif
    dsai = (VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = vk->dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &vk->dsl;
    if ((r = vkAllocateDescriptorSets(vk->dev, &dsai, &ds)) != VK_SUCCESS) {
        vk_note("vkAllocateDescriptorSets", r);
        return -1;
    }
    dbi[0] = (VkDescriptorBufferInfo){ w->buf, (VkDeviceSize)base * stride,
                                       (VkDeviceSize)n * stride };
    dbi[1] = (VkDescriptorBufferInfo){ vk->xa->buf, 0, VK_WHOLE_SIZE };
    dbi[2] = (VkDescriptorBufferInfo){ vk->yb->buf, yoff, (VkDeviceSize)n * sizeof(float) };
    dbi[3] = (VkDescriptorBufferInfo){ vk->xb->buf, 0, VK_WHOLE_SIZE };
    dbi[4] = (VkDescriptorBufferInfo){ vk->lut->buf, 0, VK_WHOLE_SIZE };
    for (i = 0; i < 5; ++i) {
        wr[i] = (VkWriteDescriptorSet){ 0 };
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = ds;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(vk->dev, 5, wr, 0, NULL);

    pc[0] = (uint32_t)nb;
    memcpy(&pc[1], &(float){ 1.0f }, sizeof(float));
    pc[2] = (uint32_t)n;
    pc[3] = 0;

    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(vk->cb, vk->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(vk->cb, groups, 1, 1);
    return 0;
}

/* ---- batched submission ---------------------------------------------
 * One decode step issues several matvecs that share an activation and do not
 * depend on each other (q/k/v, gate/up, alpha/beta).  Measured on the Adreno
 * parts this targets, one submit-plus-wait round trip costs about 0.6 ms even
 * when it carries a single 128-row window, and at model window sizes that is
 * most of the GPU's timeline.  Writing a whole group's dispatches into one
 * command buffer and submitting once amortises the round trip over the group;
 * the caller still runs its CPU rows for the whole group while the batch is
 * in flight, so the overlap is unchanged. */
int bq_vk_ptq_batch_begin(bq_vk_t *vk, const float *x, int n_in, int max_rows) {
    VkCommandBufferBeginInfo cbbi;
    float *xa, *xb;
    int nb = n_in / 128;
    VkResult r;

    if (vk == NULL || !vk->ready || x == NULL || nb <= 0 || max_rows <= 0) return -1;
    if (vk->busy && bq_vk_wait(vk) != 0) return -1;
    xa = (float *)vk_scratch(vk, &vk->xa, (size_t)nb * 104 * sizeof(float));
    xb = (float *)vk_scratch(vk, &vk->xb, (size_t)nb * 24 * sizeof(float));
    if (xa == NULL || xb == NULL) return -1;
    /* The output buffer must be sized now: descriptor sets written by the
     * first segment must not be invalidated by a later segment regrowing it. */
    if (vk_scratch(vk, &vk->yb, (size_t)max_rows * sizeof(float)) == NULL) return -1;
    vk_pack_x(xa, xb, x, nb);

    vkResetDescriptorPool(vk->dev, vk->dpool, 0);
    vkResetCommandBuffer(vk->cb, 0);
    cbbi = (VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(vk->cb, &cbbi)) != VK_SUCCESS) {
        vk_note("vkBeginCommandBuffer", r);
        return -1;
    }
    vk->batch_nb = nb;
    vk->nseg = 0;
    vk->seg_rows = 0;
    vk->t_sub_start = vk_stats_on() ? vk_now() : 0.0;
    return 0;
}

/* Appends rows [row0,row1) of `w` to the open batch; all segments of a batch
 * share the activation packed by bq_vk_ptq_batch_begin. */
int bq_vk_ptq_batch_add(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                        int row0, int row1, float *y) {
    int nb, rows = row1 - row0, done = 0;
    VkDeviceSize max_rows;

    if (vk == NULL || !vk->ready || w == NULL || rows <= 0) return -1;
    if (vk->nseg >= BQ_VK_MAX_SEG) return -1;
    nb = vk->batch_nb;
    max_rows = BQ_VK_MAX_BIND / (VkDeviceSize)stride;
    if (max_rows < 1) return -1;
    while (done < rows) {
        int n = rows - done;
        if ((VkDeviceSize)n > max_rows) n = (int)max_rows;
        if (vk_emit_rows(vk, vk->pipe, w, stride, nb, row0 + done, n,
                         (VkDeviceSize)(vk->seg_rows + (VkDeviceSize)done) * sizeof(float),
                         (uint32_t)((n + BQ_VK_LOCAL_SIZE - 1) / BQ_VK_LOCAL_SIZE)) != 0)
            return -1;
        done += n;
    }
    vk->seg[vk->nseg].y = y;
    vk->seg[vk->nseg].row0 = row0;
    vk->seg[vk->nseg].rows = rows;
    vk->seg[vk->nseg].yoff = (size_t)vk->seg_rows;
    vk->seg_rows += (VkDeviceSize)rows;
    ++vk->nseg;
    return 0;
}

int bq_vk_ptq_batch_end(bq_vk_t *vk) {
    VkSubmitInfo si;
    VkResult r;

    if (vk == NULL || !vk->ready || vk->nseg <= 0) return -1;
    if ((r = vkEndCommandBuffer(vk->cb)) != VK_SUCCESS) {
        vk_note("vkEndCommandBuffer", r);
        return -1;
    }
    vkResetFences(vk->dev, 1, &vk->fence);
    si = (VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk->cb;
    if ((r = vkQueueSubmit(vk->queue, 1, &si, vk->fence)) != VK_SUCCESS) {
        vk_note("vkQueueSubmit", r);
        return -1;
    }
    vk->busy = 1;
    if (vk->t_sub_start != 0.0) {
        double tn = vk_now();
        g_vk_t_sub += tn - vk->t_sub_start;
        vk->t_sub_end = tn;
        vk->t_sub_start = 0.0;
        ++g_vk_n_sub;
    }
    return 0;
}

int bq_vk_ptq_submit(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                     int n_in, int row0, int row1, const float *x, float *y) {
    if (bq_vk_ptq_batch_begin(vk, x, n_in, row1 - row0) != 0) return -1;
    if (bq_vk_ptq_batch_add(vk, w, stride, row0, row1, y) != 0) return -1;
    return bq_vk_ptq_batch_end(vk);
}

/* ---------------------------------------------------------------- */
/* TQ2_0 / I2_S matvec                                              */
/* ---------------------------------------------------------------- */

int bq_vk_tq2_available(const bq_vk_t *vk) {
#if defined(BQ_VK_HAVE_TQ2_SPV)
    return vk != NULL && vk->ready && vk->pipe_tq2 != VK_NULL_HANDLE;
#else
    (void)vk;
    return 0;
#endif
}

#if defined(BQ_VK_HAVE_TQ2_SPV)
/* One window of rows [row0, row1) of the matrix bound in full. */
static int vk_emit_tq2_rows(bq_vk_t *vk, const bq_vk_buf_t *w,
                            int nb, int row0, int n, VkDeviceSize yoff, uint32_t groups) {
    VkDescriptorSetAllocateInfo dsai;
    VkDescriptorBufferInfo dbi[5];
    VkWriteDescriptorSet wr[5];
    VkDescriptorSet ds;
    uint32_t pc[4];
    VkResult r;
    int i;

    dsai = (VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = vk->dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &vk->dsl;
    if ((r = vkAllocateDescriptorSets(vk->dev, &dsai, &ds)) != VK_SUCCESS) {
        vk_note("vkAllocateDescriptorSets(tq2)", r);
        return -1;
    }
    /* The whole matrix is bound and the window is carried by row_base: a
     * 66-byte block stride leaves row starts at arbitrary byte offsets, so a
     * descriptor range could not be aligned to the driver's requirement. */
    dbi[0] = (VkDescriptorBufferInfo){ w->buf, 0, VK_WHOLE_SIZE };
    dbi[1] = (VkDescriptorBufferInfo){ vk->xt->buf, 0, VK_WHOLE_SIZE };
    dbi[2] = (VkDescriptorBufferInfo){ vk->yb->buf, yoff, (VkDeviceSize)n * sizeof(float) };
    dbi[3] = (VkDescriptorBufferInfo){ vk->lut->buf, 0, VK_WHOLE_SIZE };
    dbi[4] = (VkDescriptorBufferInfo){ vk->lut->buf, 0, VK_WHOLE_SIZE };
    for (i = 0; i < 5; ++i) {
        wr[i] = (VkWriteDescriptorSet){ 0 };
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = ds;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(vk->dev, 5, wr, 0, NULL);

    pc[0] = (uint32_t)nb;
    pc[1] = (uint32_t)n;
    pc[2] = (uint32_t)row0;
    pc[3] = 0;

    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_tq2);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(vk->cb, vk->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(vk->cb, groups, 1, 1);
    return 0;
}
#endif

int bq_vk_tq2_batch_begin(bq_vk_t *vk, const float *x, int n_in, int max_rows) {
    VkCommandBufferBeginInfo cbbi;
    float *xt;
    VkResult r;

    if (vk == NULL || !vk->ready || x == NULL || n_in <= 0 || max_rows <= 0) return -1;
#if !defined(BQ_VK_HAVE_TQ2_SPV)
    return -1;
#endif
    if (vk->busy && bq_vk_wait(vk) != 0) return -1;
    /* The activation is read straight as consecutive floats: the I2_S layout
     * indexes x linearly within each 64-weight sub-block, so none of the
     * split packing the PTQ1_0 kernel needs applies here. */
    xt = (float *)vk_scratch(vk, &vk->xt, (size_t)n_in * sizeof(float));
    if (xt == NULL) return -1;
    if (vk_scratch(vk, &vk->yb, (size_t)max_rows * sizeof(float)) == NULL) return -1;
    memcpy(xt, x, (size_t)n_in * sizeof(float));

    vkResetDescriptorPool(vk->dev, vk->dpool, 0);
    vkResetCommandBuffer(vk->cb, 0);
    cbbi = (VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(vk->cb, &cbbi)) != VK_SUCCESS) {
        vk_note("vkBeginCommandBuffer(tq2)", r);
        return -1;
    }
    vk->nseg = 0;
    vk->seg_rows = 0;
    vk->t_sub_start = vk_stats_on() ? vk_now() : 0.0;
    return 0;
}

int bq_vk_tq2_batch_add(bq_vk_t *vk, const bq_vk_buf_t *packed,
                        int n_in, int row0, int row1, float *y) {
#if defined(BQ_VK_HAVE_TQ2_SPV)
    int nb, rows = row1 - row0;
    VkDeviceSize max_rows, off = 0;

    if (vk == NULL || !vk->ready || packed == NULL || rows <= 0) return -1;
    if ((n_in % 256) != 0) return -1;
    if (vk->nseg >= BQ_VK_MAX_SEG) return -1;
    nb = n_in / 256;
    if (nb > 256) return -1;   /* push-constant block count stays small */
    /* One dispatch covers the window; only a window whose result slice would
     * exceed the 128 MiB descriptor range is split, which no tensor this
     * targets comes close to. */
    max_rows = BQ_VK_MAX_BIND / sizeof(float);
    while (off < (VkDeviceSize)rows) {
        int n = rows - (int)off;
        if ((VkDeviceSize)n > max_rows) n = (int)max_rows;
        if (vk_emit_tq2_rows(vk, packed, nb, row0 + (int)off, n,
                             (VkDeviceSize)(vk->seg_rows + off) * sizeof(float),
                             (uint32_t)((n + BQ_VK_LOCAL_SIZE - 1) / BQ_VK_LOCAL_SIZE)) != 0)
            return -1;
        off += (VkDeviceSize)n;
    }
    vk->seg[vk->nseg].y = y;
    vk->seg[vk->nseg].row0 = row0;
    vk->seg[vk->nseg].rows = rows;
    vk->seg[vk->nseg].yoff = (size_t)vk->seg_rows;
    vk->seg_rows += (VkDeviceSize)rows;
    ++vk->nseg;
    return 0;
#else
    (void)vk; (void)packed; (void)n_in; (void)row0; (void)row1; (void)y;
    return -1;
#endif
}

int bq_vk_tq2_batch_end(bq_vk_t *vk) {
    /* Same submit-and-record bookkeeping as the PTQ batch. */
    return bq_vk_ptq_batch_end(vk);
}

/* ---------------------------------------------------------------- */
/* TQ2_0 matmat                                                     */
/* ---------------------------------------------------------------- */

int bq_vk_tq2_matmat_available(const bq_vk_t *vk) {
#if defined(BQ_VK_HAVE_TQ2M_SPV)
    return vk != NULL && vk->ready && vk->pipe_tq2m != VK_NULL_HANDLE;
#else
    (void)vk;
    return 0;
#endif
}

#if defined(BQ_VK_HAVE_TQ2M_SPV)
#define BQ_VK_TQ2M_LS 64

/* One window of rows, all B activation rows inside the dispatch. */
static int vk_emit_tq2m_rows(bq_vk_t *vk, const bq_vk_buf_t *w, int nb,
                             int row0, int n, VkDeviceSize yoff) {
    VkDescriptorSetAllocateInfo dsai;
    VkDescriptorBufferInfo dbi[4];
    VkWriteDescriptorSet wr[4];
    VkDescriptorSet ds;
    uint32_t pc[5 + 8];
    VkResult r;
    int i;

    dsai = (VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = vk->dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &vk->tdsl;
    if ((r = vkAllocateDescriptorSets(vk->dev, &dsai, &ds)) != VK_SUCCESS) {
        vk_note("vkAllocateDescriptorSets(tq2m)", r);
        return -1;
    }
    /* Bound from row 0 and windowed by row_base, as in the TQ2 matvec: a
     * 66-byte row stride gives no descriptor-offset alignment. */
    dbi[0] = (VkDescriptorBufferInfo){ w->buf, 0, VK_WHOLE_SIZE };
    dbi[1] = (VkDescriptorBufferInfo){ vk->xtm->buf, 0, VK_WHOLE_SIZE };
    dbi[2] = (VkDescriptorBufferInfo){ vk->ytm->buf, yoff,
                                       (VkDeviceSize)vk->tq2m_B * (VkDeviceSize)n * sizeof(float) };
    dbi[3] = (VkDescriptorBufferInfo){ vk->lutt->buf, 0, VK_WHOLE_SIZE };
    for (i = 0; i < 4; ++i) {
        wr[i] = (VkWriteDescriptorSet){ 0 };
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = ds;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(vk->dev, 4, wr, 0, NULL);

    pc[0] = (uint32_t)nb;
    pc[1] = (uint32_t)n;
    pc[2] = (uint32_t)row0;
    pc[3] = (uint32_t)vk->tq2m_B;
    pc[4] = (uint32_t)(vk->tq2m_nin / 4);   /* uints per activation row */
    memcpy(pc + 5, vk->tq2m_scale, 8 * sizeof(float));

    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_tq2m);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->tmpl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(vk->cb, vk->tmpl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       (uint32_t)sizeof(pc), pc);
    vkCmdDispatch(vk->cb, (uint32_t)((n + BQ_VK_TQ2M_LS - 1) / BQ_VK_TQ2M_LS), 1, 1);
    return 0;
}
#endif

int bq_vk_tq2_matmat_begin(bq_vk_t *vk, const int8_t *xq, const float *xscale,
                           int n_in, int B, int max_rows) {
    VkCommandBufferBeginInfo cbbi;
    VkResult r;

    if (vk == NULL || !vk->ready || xq == NULL || xscale == NULL || n_in <= 0 ||
        (n_in % 256) != 0 || B < 1 || B > 8 || max_rows <= 0) return -1;
#if !defined(BQ_VK_HAVE_TQ2M_SPV)
    return -1;
#endif
    if (vk->busy && bq_vk_wait(vk) != 0) return -1;
    /* Eight rows of activation room whatever B is, so a later begin() with a
     * larger B does not have to grow buffers a recorded set already refers
     * to; only rows < B are ever read. */
    if (vk_scratch(vk, &vk->xtm, (size_t)n_in * 8u) == NULL) return -1;
    if (vk_scratch(vk, &vk->ytm, (size_t)max_rows * (size_t)B * sizeof(float)) == NULL)
        return -1;
    if (vk->lutt == NULL) return -1;
    /* A TQ2_0 code byte holds the four weights 32 positions apart, so the
     * activation has to be transposed into the same group order: word j of a
     * 128-weight half carries lanes d = x[32d + j].  A byte permute per row,
     * against the n_in MACs the shader then does per row. */
    {
        int b, k, j, d;
        int8_t *t0 = (int8_t *)vk->xtm->ptr;
        for (b = 0; b < B; ++b) {
            const int8_t *s = xq + (size_t)b * (size_t)n_in;
            int8_t *t = t0 + (size_t)b * (size_t)n_in;
            for (k = 0; k < n_in; k += 128)
                for (j = 0; j < 32; ++j)
                    for (d = 0; d < 4; ++d)
                        t[k + 4 * j + d] = s[k + 32 * d + j];
        }
    }
    memcpy(vk->tq2m_scale, xscale, (size_t)B * sizeof(float));
    vk->tm_cap_floats = (size_t)max_rows * (size_t)B;
    vk->tq2m_B = B;
    vk->tq2m_nin = n_in;
    vk->tnseg = 0;
    vk->tseg_floats = 0;

    vkResetDescriptorPool(vk->dev, vk->dpool, 0);
    vkResetCommandBuffer(vk->cb, 0);
    cbbi = (VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(vk->cb, &cbbi)) != VK_SUCCESS) {
        vk_note("vkBeginCommandBuffer(tq2m)", r);
        return -1;
    }
    return 0;
}

int bq_vk_tq2_matmat_add(bq_vk_t *vk, const bq_vk_buf_t *packed, int wrow0,
                         int row0, int row1, int ystride, float *y) {
    int nb, rows = row1 - row0;

    if (vk == NULL || !vk->ready || packed == NULL || y == NULL || rows <= 0) return -1;
    if (vk->tnseg >= BQ_VK_MAX_SEG) return -1;
    if (vk->tq2m_nin <= 0 || (vk->tq2m_nin % 256) != 0) return -1;
    nb = vk->tq2m_nin / 256;
    if (nb > 256) return -1;
    if (vk->tseg_floats + (size_t)vk->tq2m_B * (size_t)rows > vk->tm_cap_floats) return -1;
#if defined(BQ_VK_HAVE_TQ2M_SPV)
    if (vk_emit_tq2m_rows(vk, packed, nb, wrow0, rows,
                          (VkDeviceSize)vk->tseg_floats * sizeof(float)) != 0)
        return -1;
#endif
    {
        struct bq_vk_tseg *ts = &vk->tseg[vk->tnseg];
        ts->y = y;
        ts->row0 = row0;
        ts->rows = rows;
        ts->yoff = vk->tseg_floats;
        ts->ystride = ystride;
        vk->tseg_floats += (size_t)vk->tq2m_B * (size_t)rows;
        ++vk->tnseg;
    }
    return 0;
}

int bq_vk_tq2_matmat_end(bq_vk_t *vk) {
    VkSubmitInfo si;
    VkResult r;

    if (vk == NULL || !vk->ready || vk->tnseg <= 0) return -1;
    if ((r = vkEndCommandBuffer(vk->cb)) != VK_SUCCESS) {
        vk_note("vkEndCommandBuffer(tq2m)", r);
        return -1;
    }
    vkResetFences(vk->dev, 1, &vk->fence);
    si = (VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk->cb;
    if ((r = vkQueueSubmit(vk->queue, 1, &si, vk->fence)) != VK_SUCCESS) {
        vk_note("vkQueueSubmit(tq2m)", r);
        return -1;
    }
    vk->busy = 1;
    vk->t_sub_end = vk_stats_on() ? vk_now() : 0.0;
    ++g_vk_n_sub;
    return 0;
}

int bq_vk_wait(bq_vk_t *vk) {
    VkResult r;
    double t0 = vk_stats_on() ? vk_now() : 0.0;
    int i;

    if (vk == NULL || !vk->ready) return -1;
    if (!vk->busy) return 0;
    if ((r = vkWaitForFences(vk->dev, 1, &vk->fence, VK_TRUE, UINT64_MAX)) != VK_SUCCESS) {
        vk_note("vkWaitForFences", r);
        vk->busy = 0;
        return -1;
    }
    if (t0 != 0.0 && vk->t_sub_end != 0.0) { g_vk_t_span += t0 - vk->t_sub_end; vk->t_sub_end = 0.0; }
    for (i = 0; i < vk->nseg; ++i) {
        const float *src = (const float *)vk->yb->ptr + vk->seg[i].yoff;
        memcpy(vk->seg[i].y + vk->seg[i].row0, src, (size_t)vk->seg[i].rows * sizeof(float));
    }
    for (i = 0; i < vk->mnseg; ++i) {
        const struct bq_vk_mseg *ms = &vk->mseg[i];
        const float *src = (const float *)vk->ymm->ptr + ms->yoff;
        int b;
        for (b = 0; b < vk->mm_B; ++b)
            memcpy(ms->y + (size_t)b * (size_t)ms->ystride + ms->row0,
                   src + (size_t)b * (size_t)ms->rows, (size_t)ms->rows * sizeof(float));
    }
    for (i = 0; i < vk->tnseg; ++i) {
        const struct bq_vk_tseg *ts = &vk->tseg[i];
        const float *src = (const float *)vk->ytm->ptr + ts->yoff;
        int b;
        for (b = 0; b < vk->tq2m_B; ++b)
            memcpy(ts->y + (size_t)b * (size_t)ts->ystride + ts->row0,
                   src + (size_t)b * (size_t)ts->rows, (size_t)ts->rows * sizeof(float));
    }
    vk->nseg = 0;
    vk->seg_rows = 0;
    vk->mnseg = 0;
    vk->mseg_floats = 0;
    vk->tnseg = 0;
    vk->tseg_floats = 0;
    vk->busy = 0;
    if (t0 != 0.0) { g_vk_t_wait += vk_now() - t0; ++g_vk_n_wait; }
    return 0;
}

/* ---------------------------------------------------------------- */
/* PTQ1_0 matmat                                                    */
/* ---------------------------------------------------------------- */

int bq_vk_ptq_matmat_available(const bq_vk_t *vk) {
#if defined(BQ_VK_HAVE_MATM_SPV)
    return vk != NULL && vk->ready && vk->pipe_mm != VK_NULL_HANDLE;
#else
    (void)vk;
    return 0;
#endif
}

#if defined(BQ_VK_HAVE_MATM_SPV)
#ifndef BQ_VK_MM_LS
#define BQ_VK_MM_LS 64
#endif

/* The barrier-free matmat is the default: it measured 19.8 G weights/s at
 * B=8 against 15.7 for the staged one on an Adreno 740, bit-identical, and
 * the two workgroup barriers per 128-weight block were the whole difference.
 * BQ_VK_MM_G=0 still selects the staged kernel as the A/B reference. */
static int mm_global(void) {
    static int g = -1;
    if (g < 0) {
        const char *e = getenv("BQ_VK_MM_G");
        g = (e == NULL || e[0] != '0');
    }
    return g;
}

/* One window of rows, all B activation rows inside the dispatch. */
static int vk_emit_mm_rows(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                           int base, int n, VkDeviceSize yoff) {
    VkDescriptorSetAllocateInfo dsai;
    VkDescriptorBufferInfo dbi[5];
    VkWriteDescriptorSet wr[5];
    VkDescriptorSet ds;
    uint32_t pc[4 + 16];
    VkResult r;
    int i;

    dsai = (VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = vk->dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &vk->dsl;
    if ((r = vkAllocateDescriptorSets(vk->dev, &dsai, &ds)) != VK_SUCCESS) {
        vk_note("vkAllocateDescriptorSets(matmat)", r);
        return -1;
    }
    dbi[0] = (VkDescriptorBufferInfo){ w->buf, (VkDeviceSize)base * stride,
                                       (VkDeviceSize)n * stride };
    dbi[1] = (VkDescriptorBufferInfo){ vk->xm->buf, 0, VK_WHOLE_SIZE };
    dbi[2] = (VkDescriptorBufferInfo){ vk->ymm->buf, yoff,
                                       (VkDeviceSize)vk->mm_B * (VkDeviceSize)n * sizeof(float) };
    dbi[3] = (VkDescriptorBufferInfo){ vk->luti->buf, 0, VK_WHOLE_SIZE };
    dbi[4] = (VkDescriptorBufferInfo){ vk->luti->buf, 0, VK_WHOLE_SIZE };
    for (i = 0; i < 5; ++i) {
        wr[i] = (VkWriteDescriptorSet){ 0 };
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = ds;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(vk->dev, 5, wr, 0, NULL);

    pc[0] = (uint32_t)(vk->mm_nin / 128);
    pc[1] = (uint32_t)n;
    pc[2] = (uint32_t)vk->mm_B;
    pc[3] = (uint32_t)(vk->mm_nin / 4);
    memcpy(pc + 4, vk->mm_scale, sizeof(vk->mm_scale));

    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      mm_global() && vk->pipe_mmg != VK_NULL_HANDLE ? vk->pipe_mmg
                                                                   : vk->pipe_mm);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->mpl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(vk->cb, vk->mpl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       (uint32_t)sizeof(pc), pc);
    vkCmdDispatch(vk->cb, (uint32_t)((n + BQ_VK_MM_LS - 1) / BQ_VK_MM_LS), 1, 1);
    return 0;
}
#endif

int bq_vk_ptq_matmat_begin(bq_vk_t *vk, const int8_t *xq, const float *xscale,
                           int n_in, int B, int max_rows) {
    VkCommandBufferBeginInfo cbbi;
    VkResult r;

    if (vk == NULL || !vk->ready || xq == NULL || n_in <= 0 || n_in % 128 != 0 ||
        B < 1 || B > 16 || max_rows <= 0) return -1;
    if (vk->busy && bq_vk_wait(vk) != 0) return -1;
    /* The shader is instantiated for a fixed row count per dispatch, so the
     * rows past B have to read zeros instead of whatever the scratch held. */
    if (vk_scratch(vk, &vk->xm, (size_t)n_in * 16) == NULL) return -1;
    if (vk_scratch(vk, &vk->ymm, (size_t)max_rows * (size_t)B * sizeof(float)) == NULL)
        return -1;
    vk->mm_cap_floats = (size_t)max_rows * (size_t)B;
    memcpy(vk->xm->ptr, xq, (size_t)n_in * (size_t)B);
    if (B < 16)
        memset((int8_t *)vk->xm->ptr + (size_t)n_in * (size_t)B, 0,
               (size_t)n_in * (size_t)(16 - B));
    memcpy(vk->mm_scale, xscale, (size_t)B * sizeof(float));
    vk->mm_B = B;
    vk->mm_nin = n_in;
    vk->mnseg = 0;
    vk->mseg_floats = 0;

    vkResetDescriptorPool(vk->dev, vk->dpool, 0);
    vkResetCommandBuffer(vk->cb, 0);
    cbbi = (VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(vk->cb, &cbbi)) != VK_SUCCESS) {
        vk_note("vkBeginCommandBuffer(matmat)", r);
        return -1;
    }
    vk->t_sub_start = 0.0;
    return 0;
}

int bq_vk_ptq_matmat_add(bq_vk_t *vk, const bq_vk_buf_t *w, size_t stride,
                         int row0, int row1, int ystride, float *y) {
    int rows = row1 - row0;

    if (vk == NULL || !vk->ready || w == NULL || y == NULL || rows <= 0) return -1;
    if (vk->mnseg >= BQ_VK_MAX_SEG) return -1;
    /* the whole matrix is bound as one descriptor range */
    if ((VkDeviceSize)rows * stride > BQ_VK_MAX_BIND) return -1;
    /* Growing the result buffer here would invalidate descriptor sets this
     * command buffer already refers to, so begin() reserved the room. */
    if (vk->mseg_floats + (size_t)vk->mm_B * (size_t)rows > vk->mm_cap_floats) return -1;
    if (vk->xm == NULL || vk->ymm == NULL || vk->luti == NULL) return -1;
#if defined(BQ_VK_HAVE_MATM_SPV)
    if (vk_emit_mm_rows(vk, w, stride, row0, rows,
                        (VkDeviceSize)vk->mseg_floats * sizeof(float)) != 0)
        return -1;
#endif
    {
        struct bq_vk_mseg *ms = &vk->mseg[vk->mnseg];
        ms->y = y;
        ms->row0 = row0;
        ms->rows = rows;
        ms->yoff = vk->mseg_floats;
        ms->ystride = ystride;
        vk->mseg_floats += (size_t)vk->mm_B * (size_t)rows;
        ++vk->mnseg;
    }
    return 0;
}

int bq_vk_ptq_matmat_end(bq_vk_t *vk) {
    VkSubmitInfo si;
    VkResult r;

    if (vk == NULL || !vk->ready || vk->mnseg <= 0) return -1;
    if ((r = vkEndCommandBuffer(vk->cb)) != VK_SUCCESS) {
        vk_note("vkEndCommandBuffer(matmat)", r);
        return -1;
    }
    vkResetFences(vk->dev, 1, &vk->fence);
    si = (VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk->cb;
    if ((r = vkQueueSubmit(vk->queue, 1, &si, vk->fence)) != VK_SUCCESS) {
        vk_note("vkQueueSubmit(matmat)", r);
        return -1;
    }
    vk->busy = 1;
    vk->t_sub_end = vk_stats_on() ? vk_now() : 0.0;
    ++g_vk_n_sub;
    return 0;
}

/* Decode-step attention: scores -> softmax -> weighted sum over the cache, all
 * in one submission.  The three stages are separate dispatches because each
 * needs the previous stage's whole output; none of them needs shared memory or
 * a barrier, which this driver handles badly. */
int bq_vk_attn(bq_vk_t *vk, const float *q, const float *k, const float *v,
               int n_head, int nkv, int hd, int nt, int rep, float scale, float *out) {
    bq_vk_buf_t *qb, *kb, *vb;
    VkDeviceSize qo, ko, vo;
    VkCommandBufferBeginInfo cbbi;
    VkDescriptorSetAllocateInfo dsai;
    VkDescriptorBufferInfo dbi[3];
    VkWriteDescriptorSet wr[3];
    VkDescriptorSet ds[3];
    VkDescriptorSetLayout layouts[3];
    VkSubmitInfo si;
    uint32_t pc[8];
    VkResult r;
    int i, j;

    if (vk == NULL || !vk->ready || vk->ascores == VK_NULL_HANDLE ||
        vk->asoft == VK_NULL_HANDLE || vk->aav == VK_NULL_HANDLE || out == NULL) return -1;
    if (n_head <= 0 || nkv <= 0 || hd <= 0 || nt <= 0 || rep <= 0) return -1;
    qb = bq_vk_buf_lookup(vk, q);
    kb = bq_vk_buf_lookup(vk, k);
    vb = bq_vk_buf_lookup(vk, v);
    if (qb == NULL || kb == NULL || vb == NULL) return -1;
    qo = (VkDeviceSize)vk_reg_offset(vk, q);
    ko = (VkDeviceSize)vk_reg_offset(vk, k);
    vo = (VkDeviceSize)vk_reg_offset(vk, v);

    if (vk->busy && bq_vk_wait(vk) != 0) return -1;

    if (vk_scratch(vk, &vk->as, (size_t)n_head * (size_t)nt * sizeof(float)) == NULL) return -1;
    if (vk_scratch(vk, &vk->aout, (size_t)n_head * (size_t)hd * sizeof(float)) == NULL) return -1;

    vkResetDescriptorPool(vk->dev, vk->dpool, 0);
    vkResetCommandBuffer(vk->cb, 0);
    cbbi = (VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(vk->cb, &cbbi)) != VK_SUCCESS) {
        vk_note("vkBeginCommandBuffer", r);
        return -1;
    }

    layouts[0] = vk->adsl;
    layouts[1] = vk->adsl;
    layouts[2] = vk->adsl;
    dsai = (VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = vk->dpool;
    dsai.descriptorSetCount = 3;
    dsai.pSetLayouts = layouts;
    if ((r = vkAllocateDescriptorSets(vk->dev, &dsai, ds)) != VK_SUCCESS) {
        vk_note("attn vkAllocateDescriptorSets", r);
        return -1;
    }

    /* binding order per stage: (reads) ..., output */
    for (i = 0; i < 3; ++i) {
        dbi[0] = (VkDescriptorBufferInfo){ i == 0 ? qb->buf : vk->as->buf, i == 0 ? qo : 0, VK_WHOLE_SIZE };
        dbi[1] = (VkDescriptorBufferInfo){ i == 0 ? kb->buf : (i == 2 ? vb->buf : vk->as->buf),
                                           i == 0 ? ko : (i == 2 ? vo : 0), VK_WHOLE_SIZE };
        dbi[2] = (VkDescriptorBufferInfo){ i == 0 ? vk->as->buf : (i == 1 ? vk->as->buf : vk->aout->buf),
                                           0, VK_WHOLE_SIZE };
        for (j = 0; j < 3; ++j) {
            wr[j] = (VkWriteDescriptorSet){ 0 };
            wr[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[j].dstSet = ds[i];
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &dbi[j];
        }
        vkUpdateDescriptorSets(vk->dev, 3, wr, 0, NULL);
    }

    /* stage 1: scores */
    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->ascores);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->apl, 0, 1, &ds[0], 0, NULL);
    pc[0] = (uint32_t)n_head; pc[1] = (uint32_t)nkv; pc[2] = (uint32_t)hd; pc[3] = (uint32_t)nt;
    memcpy(&pc[4], &scale, sizeof(float));
    pc[5] = (uint32_t)rep; pc[6] = 0; pc[7] = 0;
    vkCmdPushConstants(vk->cb, vk->apl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
    vkCmdDispatch(vk->cb, (uint32_t)(((size_t)n_head * nt + 127) / 128), 1, 1);

    /* stage 2: softmax per head, in place */
    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->asoft);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->apl, 0, 1, &ds[1], 0, NULL);
    memset(pc, 0, sizeof(pc));
    pc[0] = (uint32_t)nt;
    vkCmdPushConstants(vk->cb, vk->apl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
    vkCmdDispatch(vk->cb, (uint32_t)n_head, 1, 1);

    /* stage 3: p * V */
    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->aav);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->apl, 0, 1, &ds[2], 0, NULL);
    memset(pc, 0, sizeof(pc));
    pc[0] = (uint32_t)n_head; pc[1] = (uint32_t)nkv; pc[2] = (uint32_t)hd; pc[3] = (uint32_t)nt;
    pc[4] = (uint32_t)rep;
    vkCmdPushConstants(vk->cb, vk->apl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
    vkCmdDispatch(vk->cb, (uint32_t)(((size_t)n_head * hd + 127) / 128), 1, 1);

    if ((r = vkEndCommandBuffer(vk->cb)) != VK_SUCCESS) {
        vk_note("attn vkEndCommandBuffer", r);
        return -1;
    }
    vkResetFences(vk->dev, 1, &vk->fence);
    si = (VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk->cb;
    if ((r = vkQueueSubmit(vk->queue, 1, &si, vk->fence)) != VK_SUCCESS) {
        vk_note("attn vkQueueSubmit", r);
        return -1;
    }
    if ((r = vkWaitForFences(vk->dev, 1, &vk->fence, VK_TRUE, UINT64_MAX)) != VK_SUCCESS) {
        vk_note("attn vkWaitForFences", r);
        return -1;
    }
    memcpy(out, vk->aout->ptr, (size_t)n_head * (size_t)hd * sizeof(float));
    return 0;
}

/* ---------------------------------------------------------------- */
/* staged ops: normalisation, elementwise, conv, gdn, sampling       */
/* ---------------------------------------------------------------- */

#if defined(BQ_VK_HAVE_STAGE_SPV)

/* A single descriptor layout with seven storage-buffer bindings serves every
 * stage: elt uses four, conv three, gdn seven, sample two.  Slots a stage does
 * not use are bound to the lookup table so the set is always complete. */
static int vk_setup_stages(bq_vk_t *vk) {
    VkDescriptorSetLayoutBinding b[7];
    VkDescriptorSetLayoutCreateInfo dlci;
    VkPushConstantRange pcr;
    VkPipelineLayoutCreateInfo plci;
    const struct { const char *name; const uint8_t *code; size_t len; VkPipeline *slot; } sh[4] = {
        { "elt",    bq_vk_elt_spv,    bq_vk_elt_spv_len,    &vk->pelt  },
        { "gdn",    bq_vk_gdn_spv,    bq_vk_gdn_spv_len,    &vk->pgdn  },
        { "conv1d", bq_vk_conv1d_spv, bq_vk_conv1d_spv_len, &vk->pconv },
        { "sample", bq_vk_sample_spv, bq_vk_sample_spv_len, &vk->psamp },
    };
    int i, made = 0;
    VkResult r;

    for (i = 0; i < 7; ++i) {
        b[i] = (VkDescriptorSetLayoutBinding){ 0 };
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dlci = (VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 7;
    dlci.pBindings = b;
    r = vkCreateDescriptorSetLayout(vk->dev, &dlci, NULL, &vk->sdsl);
    if (r != VK_SUCCESS) { vk_note("stage descriptor layout", r); vk->sdsl = VK_NULL_HANDLE; return -1; }

    pcr = (VkPushConstantRange){ 0 };
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 32;
    plci = (VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &vk->sdsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    r = vkCreatePipelineLayout(vk->dev, &plci, NULL, &vk->spl);
    if (r != VK_SUCCESS) { vk_note("stage pipeline layout", r); vk->spl = VK_NULL_HANDLE; return -1; }

    for (i = 0; i < 4; ++i) {
        VkShaderModule sm = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        VkComputePipelineCreateInfo cpci;
        smci.codeSize = sh[i].len;
        smci.pCode = (const uint32_t *)sh[i].code;
        r = vkCreateShaderModule(vk->dev, &smci, NULL, &sm);
        if (r != VK_SUCCESS) { vk_note("stage shader module", r); return -1; }
        cpci = (VkComputePipelineCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        cpci.layout = vk->spl;
        r = vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL, sh[i].slot);
        vkDestroyShaderModule(vk->dev, sm, NULL);
        /* A driver may refuse one stage while accepting the others; the CPU
         * path covers just that stage, so keep going instead of discarding
         * the pipelines that did get created. */
        if (r != VK_SUCCESS) {
            if (getenv("BQ_VK_DEBUG") != NULL)
                fprintf(stderr, "vk: stage '%s' pipeline failed (%d)\n", sh[i].name, (int)r);
            *sh[i].slot = VK_NULL_HANDLE;
            continue;
        }
        ++made;
    }
    return made > 0 ? 0 : -1;
}

static bq_vk_buf_t *vk_scratch_buf(bq_vk_t *vk, bq_vk_buf_t **slot, size_t bytes) {
    if (*slot != NULL && (*slot)->size >= bytes) return *slot;
    bq_vk_buf_free(vk, *slot);
    *slot = NULL;
    if (bq_vk_buf_alloc(vk, bytes, slot) == NULL) return NULL;
    return *slot;
}

/* Records one dispatch and waits for it: bind seven buffers, push 32 bytes of
 * constants, run `groups` workgroups of a single invocation each. */
static int vk_run1(bq_vk_t *vk, VkPipeline pipe, bq_vk_buf_t *const bufs[7],
                   const uint32_t pc[8], uint32_t groups) {
    VkDescriptorBufferInfo dbi[7];
    VkWriteDescriptorSet wr[7];
    VkDescriptorSetAllocateInfo dsai;
    VkDescriptorSet ds;
    VkCommandBufferBeginInfo cbbi;
    VkSubmitInfo si;
    VkResult r;
    int i;

    if (groups == 0) return 0;
    if (vk->busy && bq_vk_wait(vk) != 0) return -1;

    vkResetDescriptorPool(vk->dev, vk->dpool, 0);
    vkResetCommandBuffer(vk->cb, 0);
    cbbi = (VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(vk->cb, &cbbi)) != VK_SUCCESS) {
        vk_note("stage vkBeginCommandBuffer", r);
        return -1;
    }

    dsai = (VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = vk->dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &vk->sdsl;
    if ((r = vkAllocateDescriptorSets(vk->dev, &dsai, &ds)) != VK_SUCCESS) {
        vk_note("stage vkAllocateDescriptorSets", r);
        return -1;
    }
    for (i = 0; i < 7; ++i) {
        const bq_vk_buf_t *bf = bufs[i] != NULL ? bufs[i] : vk->lut;
        dbi[i] = (VkDescriptorBufferInfo){ bf->buf, 0, VK_WHOLE_SIZE };
        wr[i] = (VkWriteDescriptorSet){ 0 };
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = ds;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(vk->dev, 7, wr, 0, NULL);

    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->spl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(vk->cb, vk->spl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
    vkCmdDispatch(vk->cb, groups, 1, 1);

    if ((r = vkEndCommandBuffer(vk->cb)) != VK_SUCCESS) {
        vk_note("stage vkEndCommandBuffer", r);
        return -1;
    }
    vkResetFences(vk->dev, 1, &vk->fence);
    si = (VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk->cb;
    if ((r = vkQueueSubmit(vk->queue, 1, &si, vk->fence)) != VK_SUCCESS) {
        vk_note("stage vkQueueSubmit", r);
        return -1;
    }
    if ((r = vkWaitForFences(vk->dev, 1, &vk->fence, VK_TRUE, UINT64_MAX)) != VK_SUCCESS) {
        vk_note("stage vkWaitForFences", r);
        return -1;
    }
    return 0;
}

#endif /* BQ_VK_HAVE_STAGE_SPV */

int bq_vk_elt(bq_vk_t *vk, int mode, const float *a, const float *b, const float *w,
              int n, int nrow, float eps, float *out) {
#if defined(BQ_VK_HAVE_STAGE_SPV)
    const size_t bytes = (size_t)n * (size_t)nrow * sizeof(float);
    bq_vk_buf_t *bufs[7] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    uint32_t pc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    const int needs_b = (mode == BQ_VK_ELT_SWIGLU || mode == BQ_VK_ELT_SIGMOID ||
                         mode == BQ_VK_ELT_ADD || mode == BQ_VK_ELT_RMS_GATE);
    const int needs_w = (mode == BQ_VK_ELT_RMS || mode == BQ_VK_ELT_RMS_GATE ||
                         mode == BQ_VK_ELT_SCALE);

    if (vk == NULL || !vk->ready || vk->pelt == VK_NULL_HANDLE) return -1;
    if (a == NULL || out == NULL || n <= 0 || nrow <= 0) return -1;
    if ((bufs[0] = vk_scratch_buf(vk, &vk->st[0], bytes)) == NULL) return -1;
    if ((bufs[3] = vk_scratch_buf(vk, &vk->st[3], bytes)) == NULL) return -1;
    if (needs_b && b != NULL) {
        if ((bufs[1] = vk_scratch_buf(vk, &vk->st[1], bytes)) == NULL) return -1;
        memcpy(bufs[1]->ptr, b, bytes);
    }
    if (needs_w && w != NULL) {
        if ((bufs[2] = vk_scratch_buf(vk, &vk->st[2], (size_t)n * sizeof(float))) == NULL) return -1;
        memcpy(bufs[2]->ptr, w, (size_t)n * sizeof(float));
    }
    memcpy(bufs[0]->ptr, a, bytes);

    pc[0] = (uint32_t)mode;
    pc[1] = (uint32_t)n;
    pc[2] = (uint32_t)nrow;
    pc[3] = (needs_w && w != NULL) ? 1u : 0u;
    memcpy(&pc[4], &eps, sizeof(float));
    if (vk_run1(vk, vk->pelt, bufs, pc, (uint32_t)nrow) != 0) return -1;
    memcpy(out, bufs[3]->ptr, bytes);
    return 0;
#else
    (void)vk; (void)mode; (void)a; (void)b; (void)w; (void)n; (void)nrow;
    (void)eps; (void)out;
    return -1;
#endif
}

int bq_vk_gdn(bq_vk_t *vk, int slot, const float *q, const float *k, const float *v,
              const float *decay, const float *beta, int nhead, int nkv, int S, float *out) {
#if defined(BQ_VK_HAVE_STAGE_SPV)
    const size_t st_bytes = (size_t)nhead * (size_t)S * (size_t)S * sizeof(float);
    bq_vk_buf_t *bufs[7] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    uint32_t pc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float scale;
    int i;

    if (vk == NULL || !vk->ready || vk->pgdn == VK_NULL_HANDLE) return -1;
    if (slot < 0 || slot >= 64 || nhead <= 0 || nkv <= 0 || S <= 0) return -1;
    if ((vk->gdn_state[slot] = vk_scratch_buf(vk, &vk->gdn_state[slot], st_bytes)) == NULL)
        return -1;
    if (vk->gdn_bytes[slot] != st_bytes) {
        memset(vk->gdn_state[slot]->ptr, 0, st_bytes);
        vk->gdn_bytes[slot] = st_bytes;
    }
    bufs[0] = vk->gdn_state[slot];

    if ((bufs[1] = vk_scratch_buf(vk, &vk->st[0], (size_t)nkv * S * sizeof(float))) == NULL) return -1;
    if ((bufs[2] = vk_scratch_buf(vk, &vk->st[1], (size_t)nkv * S * sizeof(float))) == NULL) return -1;
    if ((bufs[3] = vk_scratch_buf(vk, &vk->st[2], (size_t)nhead * S * sizeof(float))) == NULL) return -1;
    if ((bufs[4] = vk_scratch_buf(vk, &vk->st[3], (size_t)nhead * S * sizeof(float))) == NULL) return -1;
    if ((bufs[5] = vk_scratch_buf(vk, &vk->st[4], (size_t)nhead * sizeof(float))) == NULL) return -1;
    if ((bufs[6] = vk_scratch_buf(vk, &vk->st[5], (size_t)nhead * sizeof(float))) == NULL) return -1;

    memcpy(bufs[1]->ptr, q, (size_t)nkv * S * sizeof(float));
    memcpy(bufs[2]->ptr, k, (size_t)nkv * S * sizeof(float));
    memcpy(bufs[3]->ptr, v, (size_t)nhead * S * sizeof(float));
    memcpy(bufs[5]->ptr, decay, (size_t)nhead * sizeof(float));
    memcpy(bufs[6]->ptr, beta, (size_t)nhead * sizeof(float));

    scale = 1.0f / sqrtf((float)S);
    pc[0] = (uint32_t)S;
    pc[1] = (uint32_t)nhead;
    pc[2] = (uint32_t)nkv;
    memcpy(&pc[3], &scale, sizeof(float));
    if (vk_run1(vk, vk->pgdn, bufs, pc, (uint32_t)nhead) != 0) return -1;
    memcpy(out, bufs[4]->ptr, (size_t)nhead * S * sizeof(float));
    (void)i;
    return 0;
#else
    (void)vk; (void)slot; (void)q; (void)k; (void)v; (void)decay; (void)beta;
    (void)nhead; (void)nkv; (void)S; (void)out;
    return -1;
#endif
}

int bq_vk_conv(bq_vk_t *vk, const float *w, const float *hist, int K, int n_ch, float *out) {
#if defined(BQ_VK_HAVE_STAGE_SPV)
    bq_vk_buf_t *bufs[7] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    uint32_t pc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    if (vk == NULL || !vk->ready || vk->pconv == VK_NULL_HANDLE) return -1;
    if (w == NULL || hist == NULL || out == NULL || K <= 0 || n_ch <= 0) return -1;
    if ((bufs[0] = vk_scratch_buf(vk, &vk->st[0], (size_t)K * n_ch * sizeof(float))) == NULL) return -1;
    if ((bufs[1] = vk_scratch_buf(vk, &vk->st[1], (size_t)K * n_ch * sizeof(float))) == NULL) return -1;
    if ((bufs[2] = vk_scratch_buf(vk, &vk->st[2], (size_t)n_ch * sizeof(float))) == NULL) return -1;
    memcpy(bufs[0]->ptr, w, (size_t)K * n_ch * sizeof(float));
    memcpy(bufs[1]->ptr, hist, (size_t)K * n_ch * sizeof(float));
    pc[0] = (uint32_t)K;
    pc[1] = (uint32_t)n_ch;
    if (vk_run1(vk, vk->pconv, bufs, pc, (uint32_t)n_ch) != 0) return -1;
    memcpy(out, bufs[2]->ptr, (size_t)n_ch * sizeof(float));
    return 0;
#else
    (void)vk; (void)w; (void)hist; (void)K; (void)n_ch; (void)out;
    return -1;
#endif
}

/* Must equal KMAX in src/vk/sample.comp: the shader keeps its top-k list in
 * the second storage buffer, 2 + 2*KMAX uints wide. */
#define BQ_VK_SAMPLE_KMAX 64

int bq_vk_sample(bq_vk_t *vk, const float *logits, int n, int k, unsigned seed,
                 int greedy, float temperature, float top_p, uint32_t *token) {
#if defined(BQ_VK_HAVE_STAGE_SPV)
    bq_vk_buf_t *bufs[7] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    uint32_t pc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    if (vk == NULL || !vk->ready || vk->psamp == VK_NULL_HANDLE) return -1;
    if (logits == NULL || token == NULL || n <= 0) return -1;
    if ((bufs[0] = vk_scratch_buf(vk, &vk->st[0], (size_t)n * sizeof(float))) == NULL) return -1;
    /* token, then the top-k value/most-significant-bits pair of tables; the
     * sizes must match KMAX and the layout comment in sample.comp */
    if ((bufs[1] = vk_scratch_buf(vk, &vk->st[1],
                                  (2u + 2u * BQ_VK_SAMPLE_KMAX) * sizeof(uint32_t))) == NULL)
        return -1;
    memcpy(bufs[0]->ptr, logits, (size_t)n * sizeof(float));
    pc[0] = (uint32_t)n;
    pc[1] = (uint32_t)k;
    pc[2] = (uint32_t)seed;
    pc[3] = greedy ? 1u : 0u;
    memcpy(&pc[4], &temperature, sizeof(float));
    memcpy(&pc[5], &top_p, sizeof(float));
    if (vk_run1(vk, vk->psamp, bufs, pc, 1) != 0) return -1;
    memcpy(token, bufs[1]->ptr, sizeof(uint32_t));
    return 0;
#else
    (void)vk; (void)logits; (void)n; (void)k; (void)seed; (void)greedy;
    (void)temperature; (void)top_p; (void)token;
    return -1;
#endif
}

#endif /* BITNET_ENABLE_VULKAN */
