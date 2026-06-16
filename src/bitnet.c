#include "bitnet.h"

#include "bitnet_internal.h"
#include "gguf.h"
#include "ops.h"
#include "quant_q4k.h"
#include "quant_q6k.h"
#include "quant_tq2_0.h"
#include "sampler.h"
#include "tensor.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "tokenizer.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
#if defined(__ANDROID__)
#include <sys/syscall.h>
#endif



#if defined(__GNUC__) || defined(__clang__)
#define BITNET_MAYBE_UNUSED __attribute__((unused))
#else
#define BITNET_MAYBE_UNUSED
#endif

#define BITNET_MAX_TQ2_BLOCKS 128
#define BITNET_LORA_MAX_RANK 256
#define BITNET_LORA_OUTPUT_BLOCK_INDEX UINT32_MAX

typedef enum bitnet_lora_layer {
    BITNET_LORA_LAYER_ATTN_Q = 0,
    BITNET_LORA_LAYER_ATTN_K = 1,
    BITNET_LORA_LAYER_ATTN_V = 2,
    BITNET_LORA_LAYER_ATTN_OUTPUT = 3,
    BITNET_LORA_LAYER_FFN_GATE = 4,
    BITNET_LORA_LAYER_FFN_UP = 5,
    BITNET_LORA_LAYER_FFN_DOWN = 6,
    BITNET_LORA_LAYER_OUTPUT = 7,
    BITNET_LORA_LAYER_COUNT = 8
} bitnet_lora_layer_t;

typedef struct bitnet_lora_tensor {
    uint32_t block_index;
    uint32_t layer_id;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t rank;
    float scale;
    float *a;
    float *b;
    uint32_t sparse_row_count;
    uint32_t *sparse_rows;
} bitnet_lora_tensor_t;

typedef struct bitnet_lora_adapter {
    char *path;
    float scale;
    uint32_t tensor_count;
    bitnet_lora_tensor_t *tensors;
} bitnet_lora_adapter_t;

typedef struct bitnet_block_scale_cache {
    float *attn_q;
    float *attn_k;
    float *attn_v;
    float *attn_output;
    float *ffn_gate;
    float *ffn_up;
    float *ffn_down;
} bitnet_block_scale_cache_t;

typedef struct bitnet_block_tl1_weights {
    uint8_t *attn_q;
    uint8_t *attn_k;
    uint8_t *attn_v;
    uint8_t *attn_output;
    uint8_t *ffn_gate;
    uint8_t *ffn_up;
    uint8_t *ffn_down;
} bitnet_block_tl1_weights_t;

typedef struct bitnet_tq2_tl1_cache {
    bitnet_block_tl1_weights_t *blocks;
} bitnet_tq2_tl1_cache_t;

typedef struct bitnet_block_i2s_weights {
    uint8_t *attn_q;
    uint8_t *attn_k;
    uint8_t *attn_v;
    uint8_t *attn_output;
    uint8_t *ffn_gate;
    uint8_t *ffn_up;
    uint8_t *ffn_down;
} bitnet_block_i2s_weights_t;

typedef struct bitnet_tq2_i2s_cache {
    bitnet_block_i2s_weights_t *blocks;
    float **scales;      /* per-block, per-tensor scales [block_count][7 tensors] */
    int32_t **bsums;     /* per-block bsums [block_count][2: hidden + ffn] */
} bitnet_tq2_i2s_cache_t;

typedef struct bitnet_tq2_scale_cache {
    bitnet_block_scale_cache_t *blocks;
} bitnet_tq2_scale_cache_t;

struct bitnet_model {
    char *model_path;
    gguf_file_t gguf;
    bitnet_tokenizer_t *tokenizer;
    bitnet_tensor_cache_t tensor_cache;
    bitnet_tq2_scale_cache_t scale_cache;
    bitnet_tq2_tl1_cache_t tl1_cache;
    bitnet_tq2_i2s_cache_t i2s_cache;
    int8_t *output_q8;
    float *output_q8_scales;
    int8_t *output_q8_scales_i8;
    float *output_q8_d;
    uint32_t token_embd_type;
    int output_is_tied_token_embd;
    int output_q8_blockscale;

    uint32_t embedding_length;
    uint32_t block_count;
    uint32_t head_count;
    uint32_t head_count_kv;
    uint32_t attention_key_length;
    uint32_t attention_value_length;
    uint32_t feed_forward_length;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t rope_dimension_count;
    int is_minicpm;
    float embedding_scale;
    float residual_scale;
    float logit_scale;
    bitnet_lora_adapter_t *lora;
};

struct bitnet_context {
    bitnet_model_t *model;
    int max_tokens;
    int n_pos; /* current position in KV cache, persists across eval calls */
    int kv_cache_type;

    float *key_cache;
    float *value_cache;
    int8_t *key_cache_q8;
    int8_t *value_cache_q8;
    float *key_cache_scales;
    float *value_cache_scales;
    float *logits;
    float *last_hidden;

    /* pre-allocated eval buffers */
    float *hidden;
    float *q;
    float *k;
    float *v;
    float *attn_buffer;
    float *gate;
    float *up;
    float *down;
    float *scores;
    float *tmp_out;
    int8_t *attn_q8;
    float *attn_q8_scales;
    float *tq2_lut;
    int8_t *tq2_qhidden;
    int8_t *tq2_qffn;
    int8_t *tq2_tl1_lut;
    float *rope_cos;
    float *rope_sin;
    size_t tq2_lut_count;
    size_t tq2_tl1_lut_size;
};

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define BITNET_USE_TQ2_NEON_ATTN 1
#define BITNET_USE_TQ2_NEON_FFN 1
#define BITNET_USE_Q6K_NEON_OUTPUT 1
#define BITNET_USE_TQ2_TL1_LUT 0
#define BITNET_USE_TQ2_I2S 1
#else
#define BITNET_USE_TQ2_NEON_ATTN 0
#define BITNET_USE_TQ2_NEON_FFN 0
#define BITNET_USE_Q6K_NEON_OUTPUT 0
#define BITNET_USE_TQ2_TL1_LUT 0
#define BITNET_USE_TQ2_I2S 0
#endif

static uint32_t read_config_u32(const gguf_file_t *file, const char *key) {
    const gguf_metadata_t *entry = gguf_find_metadata(file, key);
    if (entry == NULL || entry->type != GGUF_TYPE_UINT32) return 0;
    return (uint32_t)entry->value.u64;
}

static float read_config_f32_default(const gguf_file_t *file, const char *key, float default_value) {
    const gguf_metadata_t *entry = gguf_find_metadata(file, key);
    if (entry == NULL) return default_value;
    if (entry->type == GGUF_TYPE_FLOAT32 || entry->type == GGUF_TYPE_FLOAT64) {
        return (float)entry->value.f64;
    }
    return default_value;
}

static float bitnet_fp16_to_fp32(uint16_t h) {
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
        {
            float value = ldexpf((float)mant / 1024.0f, -14);
            return sign ? -value : value;
        }
    }

    if (exp == 31) {
        uint32_t raw = (sign << 31) | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &raw, sizeof(f));
        return f;
    }

    exp = exp - 15u + 127u;
    mant <<= 13;

    {
        uint32_t raw = (sign << 31) | (exp << 23) | mant;
        float f;
        memcpy(&f, &raw, sizeof(f));
        return f;
    }
}

static int bitnet_f16_embedding_lookup(const void *data, int token_id,
                                       int embedding_length, int vocab_size,
                                       float *out) {
    const uint16_t *row = NULL;

    if (data == NULL || out == NULL || embedding_length <= 0 ||
        vocab_size <= 0 || token_id < 0 || token_id >= vocab_size) {
        return -1;
    }

    row = (const uint16_t *)data + (size_t)token_id * (size_t)embedding_length;
    for (int i = 0; i < embedding_length; ++i) {
        out[i] = bitnet_fp16_to_fp32(row[i]);
    }
    return 0;
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int bitnet_choose_thread_count(void) {
    const char *env = getenv("BITNET_NUM_THREADS");
    long n = 0;

    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            n = parsed;
        }
    }

    if (n <= 0) {
        n = 3;
    }
    if (n < 1) n = 1;
    if (n > 8) n = 8;
    return (int)n;
}

#if defined(__ANDROID__)
static int bitnet_env_is_disabled(const char *value) {
    return value != NULL &&
           (strcmp(value, "0") == 0 ||
            strcmp(value, "off") == 0 ||
            strcmp(value, "false") == 0 ||
            strcmp(value, "none") == 0);
}

static unsigned long long bitnet_parse_affinity_mask(const char *value) {
    char *end = NULL;
    unsigned long long mask = 0;

    if (value == NULL || value[0] == '\0') {
        return 0;
    }

    mask = strtoull(value, &end, 0);
    if (end != NULL && *end == '\0' && end != value) {
        return mask;
    }

    end = NULL;
    mask = strtoull(value, &end, 16);
    if (end != NULL && *end == '\0' && end != value) {
        return mask;
    }

    return 0;
}

static void bitnet_apply_android_affinity_for_model(uint32_t embedding_length) {
    const char *env = getenv("BITNET_CPU_AFFINITY_MASK");
    unsigned long long desired_mask = 0;
    unsigned long long current_mask = 0;
    int has_explicit_mask = env != NULL && env[0] != '\0';
    long cpu_count = sysconf(_SC_NPROCESSORS_CONF);

    if (bitnet_env_is_disabled(env)) {
        return;
    }

    desired_mask = bitnet_parse_affinity_mask(env);
    if (desired_mask == 0 && has_explicit_mask) {
        return;
    }
    if (desired_mask == 0) {
        if (cpu_count < 8) {
            return;
        }
        /* 865 testing: 0.5B likes CPU3-7; 1B likes CPU4-7; larger models stay on scheduler defaults. */
        if (embedding_length == 1024u) {
            desired_mask = 0xF8ULL;
        } else if (embedding_length == 2048u) {
            desired_mask = 0xF0ULL;
        } else {
            return;
        }
    }

#if defined(__NR_sched_getaffinity)
    (void)syscall(__NR_sched_getaffinity, 0, sizeof(current_mask), &current_mask);
    if (current_mask != 0) {
        desired_mask &= current_mask;
    }
#endif
    if (desired_mask == 0) {
        return;
    }

#if defined(__NR_sched_setaffinity)
    (void)syscall(__NR_sched_setaffinity, 0, sizeof(desired_mask), &desired_mask);
#endif
}
#else
static void bitnet_apply_android_affinity_for_model(uint32_t embedding_length) {
    (void)embedding_length;
}
#endif

static size_t bitnet_kv_dim_for_model(const bitnet_model_t *model) {
    uint32_t head_dim = 0;

    if (model == NULL || model->head_count == 0) {
        return 0;
    }

    head_dim = model->attention_key_length != 0 ?
               model->attention_key_length :
               model->embedding_length / model->head_count;
    return (size_t)model->head_count_kv * (size_t)head_dim;
}

static size_t bitnet_attention_dim_for_model(const bitnet_model_t *model) {
    uint32_t head_dim = 0;

    if (model == NULL || model->head_count == 0) {
        return 0;
    }

    head_dim = model->attention_key_length != 0 ?
               model->attention_key_length :
               model->embedding_length / model->head_count;
    return (size_t)model->head_count * (size_t)head_dim;
}

static char *bitnet_strdup_local(const char *text) {
    size_t len = 0;
    char *out = NULL;

    if (text == NULL) return NULL;
    len = strlen(text);
    out = (char *)malloc(len + 1);
    if (out == NULL) return NULL;
    memcpy(out, text, len + 1);
    return out;
}

static void bitnet_free_lora_adapter(bitnet_lora_adapter_t *adapter) {
    if (adapter == NULL) return;
    if (adapter->tensors != NULL) {
        for (uint32_t i = 0; i < adapter->tensor_count; ++i) {
            free(adapter->tensors[i].a);
            free(adapter->tensors[i].b);
            free(adapter->tensors[i].sparse_rows);
        }
        free(adapter->tensors);
    }
    free(adapter->path);
    free(adapter);
}

static int bitnet_lora_read_exact(FILE *f, void *dst, size_t bytes) {
    if (f == NULL || dst == NULL) return -1;
    return fread(dst, 1, bytes, f) == bytes ? 0 : -1;
}

static int bitnet_lora_read_u32(FILE *f, uint32_t *value) {
    return bitnet_lora_read_exact(f, value, sizeof(*value));
}

static int bitnet_lora_read_f32(FILE *f, float *value) {
    return bitnet_lora_read_exact(f, value, sizeof(*value));
}

static int bitnet_lora_layer_dims(const bitnet_model_t *model,
                                  uint32_t layer_id,
                                  uint32_t *in_dim,
                                  uint32_t *out_dim) {
    uint32_t emb_dim = 0;
    uint32_t q_dim = 0;
    uint32_t kv_dim = 0;
    uint32_t ffn_dim = 0;

    if (model == NULL || in_dim == NULL || out_dim == NULL ||
        layer_id >= BITNET_LORA_LAYER_COUNT) {
        return -1;
    }

    emb_dim = model->embedding_length;
    q_dim = (uint32_t)bitnet_attention_dim_for_model(model);
    kv_dim = (uint32_t)bitnet_kv_dim_for_model(model);
    ffn_dim = model->feed_forward_length;

    switch ((bitnet_lora_layer_t)layer_id) {
    case BITNET_LORA_LAYER_ATTN_Q:
        *in_dim = emb_dim;
        *out_dim = q_dim;
        return 0;
    case BITNET_LORA_LAYER_ATTN_K:
    case BITNET_LORA_LAYER_ATTN_V:
        *in_dim = emb_dim;
        *out_dim = kv_dim;
        return 0;
    case BITNET_LORA_LAYER_ATTN_OUTPUT:
        *in_dim = q_dim;
        *out_dim = emb_dim;
        return 0;
    case BITNET_LORA_LAYER_FFN_GATE:
    case BITNET_LORA_LAYER_FFN_UP:
        *in_dim = emb_dim;
        *out_dim = ffn_dim;
        return 0;
    case BITNET_LORA_LAYER_FFN_DOWN:
        *in_dim = ffn_dim;
        *out_dim = emb_dim;
        return 0;
    case BITNET_LORA_LAYER_OUTPUT:
        *in_dim = emb_dim;
        *out_dim = model->vocab_size;
        return 0;
    default:
        return -1;
    }
}

static int bitnet_lora_mul_overflows(size_t a, size_t b) {
    return a != 0 && b > ((size_t)-1) / a;
}

static int bitnet_lora_build_sparse_rows(bitnet_lora_tensor_t *tensor) {
    uint32_t count = 0;

    if (tensor == NULL || tensor->b == NULL || tensor->out_dim == 0 || tensor->rank == 0) {
        return -1;
    }

    for (uint32_t o = 0; o < tensor->out_dim; ++o) {
        const float *row = tensor->b + (size_t)o * (size_t)tensor->rank;
        for (uint32_t r = 0; r < tensor->rank; ++r) {
            if (row[r] != 0.0f) {
                ++count;
                break;
            }
        }
    }
    if (count == 0 || count == tensor->out_dim) {
        return 0;
    }

    tensor->sparse_rows = (uint32_t *)malloc((size_t)count * sizeof(*tensor->sparse_rows));
    if (tensor->sparse_rows == NULL) return -1;
    tensor->sparse_row_count = count;

    count = 0;
    for (uint32_t o = 0; o < tensor->out_dim; ++o) {
        const float *row = tensor->b + (size_t)o * (size_t)tensor->rank;
        for (uint32_t r = 0; r < tensor->rank; ++r) {
            if (row[r] != 0.0f) {
                tensor->sparse_rows[count++] = o;
                break;
            }
        }
    }
    return 0;
}

int bitnet_load_lora(bitnet_model_t *model, const char *path, float scale) {
    static const char expected_magic[8] = {'B', 'N', 'L', 'O', 'R', 'A', '1', '\0'};
    FILE *f = NULL;
    bitnet_lora_adapter_t *adapter = NULL;
    char magic[8];
    uint32_t version = 0;
    uint32_t tensor_count = 0;
    uint32_t persona_len = 0;
    float file_scale = 1.0f;
    int rc = -1;

    if (model == NULL || path == NULL || path[0] == '\0') {
        return -1;
    }

    f = fopen(path, "rb");
    if (f == NULL) {
        return -2;
    }

    if (bitnet_lora_read_exact(f, magic, sizeof(magic)) != 0 ||
        memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        bitnet_lora_read_u32(f, &version) != 0 ||
        bitnet_lora_read_u32(f, &tensor_count) != 0 ||
        bitnet_lora_read_f32(f, &file_scale) != 0 ||
        bitnet_lora_read_u32(f, &persona_len) != 0) {
        goto cleanup;
    }
    if (version != 1 || tensor_count > 4096u || persona_len > 65536u ||
        !isfinite(file_scale) || !isfinite(scale)) {
        goto cleanup;
    }

    adapter = (bitnet_lora_adapter_t *)calloc(1, sizeof(*adapter));
    if (adapter == NULL) goto cleanup;
    adapter->path = bitnet_strdup_local(path);
    adapter->scale = file_scale * scale;
    adapter->tensor_count = tensor_count;
    if (adapter->path == NULL) goto cleanup;

    if (persona_len > 0) {
        char discard[512];
        uint32_t remaining = persona_len;
        while (remaining > 0) {
            size_t chunk = remaining > sizeof(discard) ? sizeof(discard) : (size_t)remaining;
            if (bitnet_lora_read_exact(f, discard, chunk) != 0) {
                goto cleanup;
            }
            remaining -= (uint32_t)chunk;
        }
    }

    if (tensor_count > 0) {
        adapter->tensors = (bitnet_lora_tensor_t *)calloc(tensor_count, sizeof(*adapter->tensors));
        if (adapter->tensors == NULL) goto cleanup;
    }

    for (uint32_t i = 0; i < tensor_count; ++i) {
        bitnet_lora_tensor_t *tensor = &adapter->tensors[i];
        uint32_t expected_in = 0;
        uint32_t expected_out = 0;
        size_t a_count = 0;
        size_t b_count = 0;

        if (bitnet_lora_read_u32(f, &tensor->block_index) != 0 ||
            bitnet_lora_read_u32(f, &tensor->layer_id) != 0 ||
            bitnet_lora_read_u32(f, &tensor->in_dim) != 0 ||
            bitnet_lora_read_u32(f, &tensor->out_dim) != 0 ||
            bitnet_lora_read_u32(f, &tensor->rank) != 0 ||
            bitnet_lora_read_f32(f, &tensor->scale) != 0) {
            goto cleanup;
        }

        if (((tensor->layer_id == BITNET_LORA_LAYER_OUTPUT &&
              tensor->block_index != BITNET_LORA_OUTPUT_BLOCK_INDEX) ||
             (tensor->layer_id != BITNET_LORA_LAYER_OUTPUT &&
              tensor->block_index >= model->block_count)) ||
            tensor->rank == 0 || tensor->rank > BITNET_LORA_MAX_RANK ||
            !isfinite(tensor->scale) ||
            bitnet_lora_layer_dims(model, tensor->layer_id,
                                   &expected_in, &expected_out) != 0 ||
            tensor->in_dim != expected_in ||
            tensor->out_dim != expected_out ||
            bitnet_lora_mul_overflows((size_t)tensor->rank, (size_t)tensor->in_dim) ||
            bitnet_lora_mul_overflows((size_t)tensor->out_dim, (size_t)tensor->rank)) {
            goto cleanup;
        }

        a_count = (size_t)tensor->rank * (size_t)tensor->in_dim;
        b_count = (size_t)tensor->out_dim * (size_t)tensor->rank;
        tensor->a = (float *)malloc(a_count * sizeof(*tensor->a));
        tensor->b = (float *)malloc(b_count * sizeof(*tensor->b));
        if (tensor->a == NULL || tensor->b == NULL) goto cleanup;
        if (bitnet_lora_read_exact(f, tensor->a, a_count * sizeof(*tensor->a)) != 0 ||
            bitnet_lora_read_exact(f, tensor->b, b_count * sizeof(*tensor->b)) != 0) {
            goto cleanup;
        }
        if (bitnet_lora_build_sparse_rows(tensor) != 0) {
            goto cleanup;
        }
    }

    {
        int trailing = fgetc(f);
        if (trailing != EOF) goto cleanup;
    }

    bitnet_free_lora_adapter(model->lora);
    model->lora = adapter;
    adapter = NULL;
    rc = 0;

cleanup:
    if (f != NULL) fclose(f);
    bitnet_free_lora_adapter(adapter);
    return rc;
}

int bitnet_lora_count(const bitnet_model_t *model) {
    if (model == NULL || model->lora == NULL) return 0;
    if (model->lora->tensor_count > (uint32_t)INT_MAX) return INT_MAX;
    return (int)model->lora->tensor_count;
}

int bitnet_embedding_length(const bitnet_model_t *model) {
    if (model == NULL || model->embedding_length > (uint32_t)INT_MAX) return -1;
    return (int)model->embedding_length;
}

int bitnet_vocab_size(const bitnet_model_t *model) {
    if (model == NULL || model->vocab_size > (uint32_t)INT_MAX) return -1;
    return (int)model->vocab_size;
}

static int bitnet_apply_lora(const bitnet_model_t *model,
                             int block_index,
                             bitnet_lora_layer_t layer_id,
                             const float *input,
                             float *output) {
    bitnet_lora_adapter_t *adapter = NULL;
    float tmp[BITNET_LORA_MAX_RANK];

    if (model == NULL || model->lora == NULL) return 0;
    if (input == NULL || output == NULL ||
        (block_index < 0 && layer_id != BITNET_LORA_LAYER_OUTPUT)) {
        return -1;
    }

    adapter = model->lora;
    for (uint32_t ti = 0; ti < adapter->tensor_count; ++ti) {
        const bitnet_lora_tensor_t *tensor = &adapter->tensors[ti];
        float combined_scale = 0.0f;

        uint32_t expected_block = layer_id == BITNET_LORA_LAYER_OUTPUT ?
            BITNET_LORA_OUTPUT_BLOCK_INDEX : (uint32_t)block_index;

        if (tensor->block_index != expected_block || tensor->layer_id != (uint32_t)layer_id) {
            continue;
        }
        combined_scale = adapter->scale * tensor->scale;
        if (combined_scale == 0.0f) continue;

        for (uint32_t r = 0; r < tensor->rank; ++r) {
            const float *a_row = tensor->a + (size_t)r * (size_t)tensor->in_dim;
            float acc = 0.0f;
            uint32_t i = 0;
#if defined(__ARM_NEON)
            float32x4_t vacc = vdupq_n_f32(0.0f);
            for (; i + 3 < tensor->in_dim; i += 4) {
                float32x4_t av = vld1q_f32(a_row + i);
                float32x4_t xv = vld1q_f32(input + i);
                vacc = vfmaq_f32(vacc, av, xv);
            }
            acc = vaddvq_f32(vacc);
#endif
            for (; i < tensor->in_dim; ++i) {
                acc += a_row[i] * input[i];
            }
            tmp[r] = acc;
        }

        if (tensor->sparse_rows != NULL) {
            for (uint32_t si = 0; si < tensor->sparse_row_count; ++si) {
                uint32_t o = tensor->sparse_rows[si];
                const float *b_row = tensor->b + (size_t)o * (size_t)tensor->rank;
                float acc = 0.0f;
                for (uint32_t r = 0; r < tensor->rank; ++r) {
                    acc += b_row[r] * tmp[r];
                }
                output[o] += combined_scale * acc;
            }
        } else {
            for (uint32_t o = 0; o < tensor->out_dim; ++o) {
                const float *b_row = tensor->b + (size_t)o * (size_t)tensor->rank;
                float acc = 0.0f;
                for (uint32_t r = 0; r < tensor->rank; ++r) {
                    acc += b_row[r] * tmp[r];
                }
                output[o] += combined_scale * acc;
            }
        }
    }

    return 0;
}

static size_t bitnet_kv_element_count(const bitnet_context_t *ctx) {
    if (ctx == NULL || ctx->model == NULL) {
        return 0;
    }
    return (size_t)ctx->model->block_count * (size_t)ctx->max_tokens *
           bitnet_kv_dim_for_model(ctx->model);
}

static size_t bitnet_kv_scale_count(const bitnet_context_t *ctx) {
    if (ctx == NULL || ctx->model == NULL) {
        return 0;
    }
    return (size_t)ctx->model->block_count * (size_t)ctx->max_tokens *
           (size_t)ctx->model->head_count_kv;
}

static float bitnet_quantize_f32_to_i8(const float *src, int n, int8_t *dst) {
    float max_abs = 0.0f;

    for (int i = 0; i < n; ++i) {
        float v = fabsf(src[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }

    if (max_abs == 0.0f) {
        memset(dst, 0, (size_t)n * sizeof(*dst));
        return 0.0f;
    }

    {
        float scale = max_abs / 127.0f;
        float inv_scale = 127.0f / max_abs;
        for (int i = 0; i < n; ++i) {
            float scaled = src[i] * inv_scale;
            int q = (int)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            dst[i] = (int8_t)q;
        }
        return scale;
    }
}

static int bitnet_rms_norm_quant_tq2_i8(float *dst, const float *src, const float *weight,
                                        int n, int8_t *qvec, float *scale,
                                        int32_t *block_bsums) {
    float sum = 0.0f;
    float max_abs = 0.0f;
    int i = 0;

    if (dst == NULL || src == NULL || weight == NULL || qvec == NULL ||
        scale == NULL || n <= 0) {
        return -1;
    }

#if defined(__ARM_NEON)
    {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (; i + 3 < n; i += 4) {
            float32x4_t v = vld1q_f32(src + i);
            sum_vec = vfmaq_f32(sum_vec, v, v);
        }
        sum = vaddvq_f32(sum_vec);
    }
#endif
    for (; i < n; ++i) {
        sum += src[i] * src[i];
    }

    const float inv_rms = 1.0f / sqrtf(sum / (float)n + 1e-6f);

    i = 0;
#if defined(__ARM_NEON)
    {
        float32x4_t max_vec = vdupq_n_f32(0.0f);
        float32x4_t inv = vdupq_n_f32(inv_rms);
        for (; i + 3 < n; i += 4) {
            float32x4_t v = vld1q_f32(src + i);
            float32x4_t w = vld1q_f32(weight + i);
            float32x4_t out = vmulq_f32(vmulq_f32(v, inv), w);
            vst1q_f32(dst + i, out);
            max_vec = vmaxq_f32(max_vec, vabsq_f32(out));
        }
        max_abs = vmaxvq_f32(max_vec);
    }
#endif
    for (; i < n; ++i) {
        float out = src[i] * inv_rms * weight[i];
        float a = fabsf(out);
        dst[i] = out;
        if (a > max_abs) max_abs = a;
    }

    return bitnet_tq2_0_quantize_vec_i8_known_max(dst, n, qvec, scale,
                                                  block_bsums, max_abs);
}

static void bitnet_scale_vector(float *x, float scale, int n) {
    int i = 0;
    if (x == NULL || n <= 0 || scale == 1.0f) return;
#if defined(__ARM_NEON)
    {
        float32x4_t sv = vdupq_n_f32(scale);
        for (; i + 3 < n; i += 4) {
            vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), sv));
        }
    }
#endif
    for (; i < n; ++i) {
        x[i] *= scale;
    }
}

static void bitnet_residual_add_scaled(float *out, const float *a, const float *b,
                                       float b_scale, int n) {
    int i = 0;
    if (b_scale == 1.0f) {
        bitnet_residual_add(out, a, b, n);
        return;
    }
#if defined(__ARM_NEON)
    {
        float32x4_t sv = vdupq_n_f32(b_scale);
        for (; i + 3 < n; i += 4) {
            float32x4_t av = vld1q_f32(a + i);
            float32x4_t bv = vld1q_f32(b + i);
            vst1q_f32(out + i, vfmaq_f32(av, bv, sv));
        }
    }
#endif
    for (; i < n; ++i) {
        out[i] = a[i] + b[i] * b_scale;
    }
}

static int bitnet_dot_i8(const int8_t *a, const int8_t *b, int n) {
    int sum = 0;
    int i = 0;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    {
        int32x4_t acc = vdupq_n_s32(0);
        for (; i + 15 < n; i += 16) {
            int8x16_t av = vld1q_s8(a + i);
            int8x16_t bv = vld1q_s8(b + i);
            acc = vdotq_s32(acc, av, bv);
        }
        sum = vaddvq_s32(acc);
    }
#endif

    for (; i < n; ++i) {
        sum += (int)a[i] * (int)b[i];
    }
    return sum;
}

static void bitnet_accum_i8_scaled(float *dst, const int8_t *src, float scale, int n) {
    int i = 0;

#if defined(__ARM_NEON)
    {
        float32x4_t scale_v = vdupq_n_f32(scale);
        for (; i + 15 < n; i += 16) {
            int8x16_t src_i8 = vld1q_s8(src + i);
            int16x8_t lo_i16 = vmovl_s8(vget_low_s8(src_i8));
            int16x8_t hi_i16 = vmovl_s8(vget_high_s8(src_i8));
            int32x4_t v0_i32 = vmovl_s16(vget_low_s16(lo_i16));
            int32x4_t v1_i32 = vmovl_s16(vget_high_s16(lo_i16));
            int32x4_t v2_i32 = vmovl_s16(vget_low_s16(hi_i16));
            int32x4_t v3_i32 = vmovl_s16(vget_high_s16(hi_i16));
            float32x4_t d0 = vld1q_f32(dst + i);
            float32x4_t d1 = vld1q_f32(dst + i + 4);
            float32x4_t d2 = vld1q_f32(dst + i + 8);
            float32x4_t d3 = vld1q_f32(dst + i + 12);

            vst1q_f32(dst + i, vfmaq_f32(d0, vcvtq_f32_s32(v0_i32), scale_v));
            vst1q_f32(dst + i + 4, vfmaq_f32(d1, vcvtq_f32_s32(v1_i32), scale_v));
            vst1q_f32(dst + i + 8, vfmaq_f32(d2, vcvtq_f32_s32(v2_i32), scale_v));
            vst1q_f32(dst + i + 12, vfmaq_f32(d3, vcvtq_f32_s32(v3_i32), scale_v));
        }
    }
#endif

    for (; i < n; ++i) {
        dst[i] += scale * (float)src[i];
    }
}

static int bitnet_allocate_q8_kv_cache(bitnet_context_t *ctx) {
    size_t kv_count = bitnet_kv_element_count(ctx);
    size_t scale_count = bitnet_kv_scale_count(ctx);
    size_t q_count = 0;

    if (ctx == NULL || ctx->model == NULL) {
        return -1;
    }

    q_count = bitnet_attention_dim_for_model(ctx->model);
    ctx->key_cache_q8 = (int8_t *)calloc(kv_count, sizeof(*ctx->key_cache_q8));
    ctx->value_cache_q8 = (int8_t *)calloc(kv_count, sizeof(*ctx->value_cache_q8));
    ctx->key_cache_scales = (float *)calloc(scale_count, sizeof(*ctx->key_cache_scales));
    ctx->value_cache_scales = (float *)calloc(scale_count, sizeof(*ctx->value_cache_scales));
    ctx->attn_q8 = (int8_t *)calloc(q_count, sizeof(*ctx->attn_q8));
    ctx->attn_q8_scales = (float *)calloc((size_t)ctx->model->head_count, sizeof(*ctx->attn_q8_scales));

    if (ctx->key_cache_q8 == NULL || ctx->value_cache_q8 == NULL ||
        ctx->key_cache_scales == NULL || ctx->value_cache_scales == NULL ||
        ctx->attn_q8 == NULL || ctx->attn_q8_scales == NULL) {
        free(ctx->key_cache_q8);
        free(ctx->value_cache_q8);
        free(ctx->key_cache_scales);
        free(ctx->value_cache_scales);
        free(ctx->attn_q8);
        free(ctx->attn_q8_scales);
        ctx->key_cache_q8 = NULL;
        ctx->value_cache_q8 = NULL;
        ctx->key_cache_scales = NULL;
        ctx->value_cache_scales = NULL;
        ctx->attn_q8 = NULL;
        ctx->attn_q8_scales = NULL;
        return -1;
    }

    return 0;
}

static void bitnet_free_q8_kv_cache(bitnet_context_t *ctx) {
    if (ctx == NULL) return;
    free(ctx->key_cache_q8);
    free(ctx->value_cache_q8);
    free(ctx->key_cache_scales);
    free(ctx->value_cache_scales);
    free(ctx->attn_q8);
    free(ctx->attn_q8_scales);
    ctx->key_cache_q8 = NULL;
    ctx->value_cache_q8 = NULL;
    ctx->key_cache_scales = NULL;
    ctx->value_cache_scales = NULL;
    ctx->attn_q8 = NULL;
    ctx->attn_q8_scales = NULL;
}

typedef struct {
    pthread_t threads[8];
    int n_threads;
    int initialized;
    atomic_uint generation;
    atomic_int completed;
    atomic_int next_row;
    atomic_int stop;
    atomic_int park;

    const uint8_t *output_data;
    size_t output_col_size;
    int output_blocks_per_col;
    int vocab_size;
    const float *hidden;
    const int8_t *qhidden;
    float hidden_scale;
    int use_i8_output;
    const int8_t *output_q8;
    const float *output_q8_scales;
    const int8_t *output_q8_scales_i8;
    const float *output_q8_d;
    int chunk_rows;
    float *logits;
} bitnet_output_pool_t;

static bitnet_output_pool_t g_output_pool = {
    {0},
    0,
    0,
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(0),
    NULL,
    0,
    0,
    0,
    NULL,
    NULL,
    0.0f,
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    0,
    NULL,
};
static pthread_once_t g_output_pool_once = PTHREAD_ONCE_INIT;

#define BITNET_OUTPUT_SPIN_ITERS 5000
#if defined(__ANDROID__)
#define BITNET_OUTPUT_PARK_USLEEP 1
#else
#define BITNET_OUTPUT_PARK_USLEEP 100
#endif
#ifndef BITNET_DISPATCHER_RUNS_WORK
#define BITNET_DISPATCHER_RUNS_WORK 1
#endif

#define BITNET_OUTPUT_SPIN_WAIT(cond) do { \
    int _spin = 0; \
    while (!(cond)) { \
        if (++_spin < BITNET_OUTPUT_SPIN_ITERS) { \
            __asm__ __volatile__("yield" ::: "memory"); \
        } else { \
            usleep(1); \
            _spin = 0; \
        } \
    } \
} while (0)

static void bitnet_output_compute_rows(const uint8_t *output_data,
                                       size_t output_col_size,
                                       int output_blocks_per_col,
                                       int vocab_begin,
                                       int vocab_end,
                                       const float *hidden,
                                       float *logits) {
    for (int j = vocab_begin; j < vocab_end; ++j) {
        float sum = 0.0f;
        for (int b = 0; b < output_blocks_per_col; ++b) {
            const bitnet_q6k_block_t *block =
                (const bitnet_q6k_block_t *)(output_data +
                    (size_t)j * output_col_size +
                    (size_t)b * BITNET_Q6K_BLOCK_SIZE);
            float block_dot = 0.0f;
            if (bitnet_q6k_dot_product(block,
                    hidden + (size_t)b * BITNET_Q6K_QK,
                    BITNET_Q6K_QK, &block_dot) != 0) {
                block_dot = 0.0f;
            }
            sum += block_dot;
        }
        logits[j] = sum;
    }
}

static void bitnet_output_compute_rows_i8_neon(const uint8_t *output_data,
                                               size_t output_col_size,
                                               int output_blocks_per_col,
                                               int vocab_begin,
                                               int vocab_end,
                                               const int8_t *qhidden,
                                               float hidden_scale,
                                               float *logits) {
    for (int j = vocab_begin; j < vocab_end; ++j) {
        float sum = 0.0f;
        for (int b = 0; b < output_blocks_per_col; ++b) {
            const bitnet_q6k_block_t *block =
                (const bitnet_q6k_block_t *)(output_data +
                    (size_t)j * output_col_size +
                    (size_t)b * BITNET_Q6K_BLOCK_SIZE);
            float block_dot = 0.0f;
            if (bitnet_q6k_dot_product_i8_neon(block,
                    qhidden + (size_t)b * BITNET_Q6K_QK,
                    hidden_scale, BITNET_Q6K_QK, &block_dot) != 0) {
                block_dot = 0.0f;
            }
            sum += block_dot;
        }
        logits[j] = sum;
    }
}

static void bitnet_output_compute_rows_q8_neon(const int8_t *output_q8,
                                               const float *output_q8_scales,
                                               int output_blocks_per_col,
                                               int vocab_begin,
                                               int vocab_end,
                                               const int8_t *qhidden,
                                               float hidden_scale,
                                               float *logits) {
    const int emb_dim = output_blocks_per_col * BITNET_Q6K_QK;
    const int scale_stride = output_blocks_per_col * 16;

    int j = vocab_begin;
    for (; j + 7 < vocab_end; j += 8) {
        if (bitnet_q6k_dot_product_q8_8_neon(
                output_q8 + (size_t)j * (size_t)emb_dim,
                output_q8_scales + (size_t)j * (size_t)scale_stride,
                emb_dim, scale_stride, output_blocks_per_col,
                qhidden, hidden_scale, logits + j) != 0) {
            for (int k = 0; k < 8; ++k) {
                logits[j + k] = 0.0f;
            }
        }
    }
    for (; j + 3 < vocab_end; j += 4) {
        if (bitnet_q6k_dot_product_q8_4_neon(
                output_q8 + (size_t)j * (size_t)emb_dim,
                output_q8_scales + (size_t)j * (size_t)scale_stride,
                emb_dim, scale_stride, output_blocks_per_col,
                qhidden, hidden_scale, logits + j) != 0) {
            logits[j + 0] = 0.0f;
            logits[j + 1] = 0.0f;
            logits[j + 2] = 0.0f;
            logits[j + 3] = 0.0f;
        }
    }
    for (; j < vocab_end; ++j) {
        if (bitnet_q6k_dot_product_q8_neon(
                output_q8 + (size_t)j * (size_t)emb_dim,
                output_q8_scales + (size_t)j * (size_t)output_blocks_per_col * 16u,
                output_blocks_per_col, qhidden, hidden_scale, &logits[j]) != 0) {
            logits[j] = 0.0f;
        }
    }
}

static void bitnet_output_compute_rows_q8_compact_neon(const int8_t *output_q8,
                                                       const int8_t *output_q8_scales_i8,
                                                       const float *output_q8_d,
                                                       int output_blocks_per_col,
                                                       int vocab_begin,
                                                       int vocab_end,
                                                       const int8_t *qhidden,
                                                       float hidden_scale,
                                                       float *logits) {
    const int emb_dim = output_blocks_per_col * BITNET_Q6K_QK;
    const int scale_stride = output_blocks_per_col * 16;
    const int d_stride = output_blocks_per_col;

    int j = vocab_begin;
    for (; j + 7 < vocab_end; j += 8) {
#if defined(__GNUC__) || defined(__clang__)
        if (j + 23 < vocab_end) {
            __builtin_prefetch(output_q8 + (size_t)(j + 16) * (size_t)emb_dim, 0, 3);
            __builtin_prefetch(output_q8_scales_i8 + (size_t)(j + 16) * (size_t)scale_stride, 0, 3);
            __builtin_prefetch(output_q8_d + (size_t)(j + 16) * (size_t)d_stride, 0, 3);
        }
#endif
        if (bitnet_q6k_dot_product_q8_compact_8_neon(
                output_q8 + (size_t)j * (size_t)emb_dim,
                output_q8_scales_i8 + (size_t)j * (size_t)scale_stride,
                output_q8_d + (size_t)j * (size_t)d_stride,
                emb_dim, scale_stride, d_stride,
                output_blocks_per_col, qhidden, hidden_scale, logits + j) != 0) {
            for (int k = 0; k < 8; ++k) {
                logits[j + k] = 0.0f;
            }
        }
    }
    for (; j + 3 < vocab_end; j += 4) {
        if (bitnet_q6k_dot_product_q8_compact_4_neon(
                output_q8 + (size_t)j * (size_t)emb_dim,
                output_q8_scales_i8 + (size_t)j * (size_t)scale_stride,
                output_q8_d + (size_t)j * (size_t)d_stride,
                emb_dim, scale_stride, d_stride,
                output_blocks_per_col, qhidden, hidden_scale, logits + j) != 0) {
            logits[j + 0] = 0.0f;
            logits[j + 1] = 0.0f;
            logits[j + 2] = 0.0f;
            logits[j + 3] = 0.0f;
        }
    }
    for (; j < vocab_end; ++j) {
        if (bitnet_q6k_dot_product_q8_compact_neon(
                output_q8 + (size_t)j * (size_t)emb_dim,
                output_q8_scales_i8 + (size_t)j * (size_t)scale_stride,
                output_q8_d + (size_t)j * (size_t)d_stride,
                output_blocks_per_col, qhidden, hidden_scale, &logits[j]) != 0) {
            logits[j] = 0.0f;
        }
    }
}

static void bitnet_output_compute_rows_q8_blockscale_neon(const int8_t *output_q8,
                                                          const float *output_q8_d,
                                                          int output_blocks_per_col,
                                                          int vocab_begin,
                                                          int vocab_end,
                                                          const int8_t *qhidden,
                                                          float hidden_scale,
                                                          float *logits) {
    const int emb_dim = output_blocks_per_col * BITNET_Q6K_QK;
    const int d_stride = output_blocks_per_col;

    int j = vocab_begin;
    for (; j + 7 < vocab_end; j += 8) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        float32x4_t sum0_v = vdupq_n_f32(0.0f);
        float32x4_t sum1_v = vdupq_n_f32(0.0f);
        float32x4_t sum2_v = vdupq_n_f32(0.0f);
        float32x4_t sum3_v = vdupq_n_f32(0.0f);
        float32x4_t sum4_v = vdupq_n_f32(0.0f);
        float32x4_t sum5_v = vdupq_n_f32(0.0f);
        float32x4_t sum6_v = vdupq_n_f32(0.0f);
        float32x4_t sum7_v = vdupq_n_f32(0.0f);

        const int8_t *row0 = output_q8 + (size_t)j * (size_t)emb_dim;
        const int8_t *row1 = row0 + (size_t)emb_dim;
        const int8_t *row2 = row1 + (size_t)emb_dim;
        const int8_t *row3 = row2 + (size_t)emb_dim;
        const int8_t *row4 = row3 + (size_t)emb_dim;
        const int8_t *row5 = row4 + (size_t)emb_dim;
        const int8_t *row6 = row5 + (size_t)emb_dim;
        const int8_t *row7 = row6 + (size_t)emb_dim;
        const float *d0 = output_q8_d + (size_t)j * (size_t)d_stride;
        const float *d1 = d0 + (size_t)d_stride;
        const float *d2 = d1 + (size_t)d_stride;
        const float *d3 = d2 + (size_t)d_stride;
        const float *d4 = d3 + (size_t)d_stride;
        const float *d5 = d4 + (size_t)d_stride;
        const float *d6 = d5 + (size_t)d_stride;
        const float *d7 = d6 + (size_t)d_stride;

        for (int b = 0; b < output_blocks_per_col; ++b) {
            const int8_t *v = qhidden + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q0 = row0 + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q1 = row1 + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q2 = row2 + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q3 = row3 + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q4 = row4 + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q5 = row5 + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q6 = row6 + (size_t)b * BITNET_Q6K_QK;
            const int8_t *q7 = row7 + (size_t)b * BITNET_Q6K_QK;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0);
            int32x4_t acc7 = vdupq_n_s32(0);

            for (int g = 0; g < 16; ++g) {
                const int off = g * 16;
                const int8x16_t vv = vld1q_s8(v + off);
                acc0 = vdotq_s32(acc0, vld1q_s8(q0 + off), vv);
                acc1 = vdotq_s32(acc1, vld1q_s8(q1 + off), vv);
                acc2 = vdotq_s32(acc2, vld1q_s8(q2 + off), vv);
                acc3 = vdotq_s32(acc3, vld1q_s8(q3 + off), vv);
                acc4 = vdotq_s32(acc4, vld1q_s8(q4 + off), vv);
                acc5 = vdotq_s32(acc5, vld1q_s8(q5 + off), vv);
                acc6 = vdotq_s32(acc6, vld1q_s8(q6 + off), vv);
                acc7 = vdotq_s32(acc7, vld1q_s8(q7 + off), vv);
            }

            sum0_v = vfmaq_n_f32(sum0_v, vcvtq_f32_s32(acc0), d0[b]);
            sum1_v = vfmaq_n_f32(sum1_v, vcvtq_f32_s32(acc1), d1[b]);
            sum2_v = vfmaq_n_f32(sum2_v, vcvtq_f32_s32(acc2), d2[b]);
            sum3_v = vfmaq_n_f32(sum3_v, vcvtq_f32_s32(acc3), d3[b]);
            sum4_v = vfmaq_n_f32(sum4_v, vcvtq_f32_s32(acc4), d4[b]);
            sum5_v = vfmaq_n_f32(sum5_v, vcvtq_f32_s32(acc5), d5[b]);
            sum6_v = vfmaq_n_f32(sum6_v, vcvtq_f32_s32(acc6), d6[b]);
            sum7_v = vfmaq_n_f32(sum7_v, vcvtq_f32_s32(acc7), d7[b]);
        }

        logits[j + 0] = vaddvq_f32(sum0_v) * hidden_scale;
        logits[j + 1] = vaddvq_f32(sum1_v) * hidden_scale;
        logits[j + 2] = vaddvq_f32(sum2_v) * hidden_scale;
        logits[j + 3] = vaddvq_f32(sum3_v) * hidden_scale;
        logits[j + 4] = vaddvq_f32(sum4_v) * hidden_scale;
        logits[j + 5] = vaddvq_f32(sum5_v) * hidden_scale;
        logits[j + 6] = vaddvq_f32(sum6_v) * hidden_scale;
        logits[j + 7] = vaddvq_f32(sum7_v) * hidden_scale;
#else
        for (int row = 0; row < 8; ++row) {
            float sum = 0.0f;
            const int8_t *q = output_q8 + (size_t)(j + row) * (size_t)emb_dim;
            const float *d = output_q8_d + (size_t)(j + row) * (size_t)d_stride;
            for (int b = 0; b < output_blocks_per_col; ++b) {
                int32_t acc = 0;
                for (int i = 0; i < BITNET_Q6K_QK; ++i) {
                    int idx = b * BITNET_Q6K_QK + i;
                    acc += (int32_t)q[idx] * (int32_t)qhidden[idx];
                }
                sum += (float)acc * d[b];
            }
            logits[j + row] = sum * hidden_scale;
        }
#endif
    }

    for (; j < vocab_end; ++j) {
        float sum = 0.0f;
        const int8_t *q = output_q8 + (size_t)j * (size_t)emb_dim;
        const float *d = output_q8_d + (size_t)j * (size_t)d_stride;
        for (int b = 0; b < output_blocks_per_col; ++b) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t acc = vdupq_n_s32(0);
            const int8_t *qb = q + (size_t)b * BITNET_Q6K_QK;
            const int8_t *vb = qhidden + (size_t)b * BITNET_Q6K_QK;
            for (int g = 0; g < 16; ++g) {
                const int off = g * 16;
                acc = vdotq_s32(acc, vld1q_s8(qb + off), vld1q_s8(vb + off));
            }
            sum += (float)vaddvq_s32(acc) * d[b];
#else
            int32_t acc = 0;
            for (int i = 0; i < BITNET_Q6K_QK; ++i) {
                int idx = b * BITNET_Q6K_QK + i;
                acc += (int32_t)q[idx] * (int32_t)qhidden[idx];
            }
            sum += (float)acc * d[b];
#endif
        }
        logits[j] = sum * hidden_scale;
    }
}

static int bitnet_choose_output_chunk_rows_with_default(int default_rows) {
    static int cached_env = 0;
    const char *env = NULL;
    long n = 0;

    env = getenv("BITNET_OUTPUT_CHUNK_ROWS");
    if (env != NULL && env[0] != '\0') {
        if (cached_env > 0) {
            return cached_env;
        }
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            n = parsed;
        }
        if (n <= 0) n = default_rows;
        if (n < 64) n = 64;
        if (n > 8192) n = 8192;
        cached_env = (int)n;
        return cached_env;
    }

    if (n <= 0) n = default_rows;
    if (n < 64) n = 64;
    if (n > 8192) n = 8192;
    return (int)n;
}

static int bitnet_choose_output_chunk_rows(void) {
    return bitnet_choose_output_chunk_rows_with_default(1024);
}

static int bitnet_choose_q8_compact_output_chunk_rows(int output_blocks_per_col) {
#if defined(__ANDROID__)
    if (output_blocks_per_col == 10 || output_blocks_per_col == 16) {
        return bitnet_choose_output_chunk_rows_with_default(64);
    }
#else
    (void)output_blocks_per_col;
#endif
    return bitnet_choose_output_chunk_rows();
}

static void bitnet_output_pool_run_chunks(void) {
    for (;;) {
        const int chunk_rows = g_output_pool.chunk_rows > 0 ?
                               g_output_pool.chunk_rows : 1024;
        int begin = atomic_fetch_add_explicit(&g_output_pool.next_row,
                                              chunk_rows,
                                              memory_order_relaxed);
        int end = begin + chunk_rows;
        if (begin >= g_output_pool.vocab_size) break;
        if (end > g_output_pool.vocab_size) end = g_output_pool.vocab_size;

        if (g_output_pool.use_i8_output == 3) {
            bitnet_output_compute_rows_q8_compact_neon(g_output_pool.output_q8,
                                                       g_output_pool.output_q8_scales_i8,
                                                       g_output_pool.output_q8_d,
                                                       g_output_pool.output_blocks_per_col,
                                                       begin, end,
                                                       g_output_pool.qhidden,
                                                       g_output_pool.hidden_scale,
                                                       g_output_pool.logits);
        } else if (g_output_pool.use_i8_output == 4) {
            bitnet_output_compute_rows_q8_blockscale_neon(g_output_pool.output_q8,
                                                          g_output_pool.output_q8_d,
                                                          g_output_pool.output_blocks_per_col,
                                                          begin, end,
                                                          g_output_pool.qhidden,
                                                          g_output_pool.hidden_scale,
                                                          g_output_pool.logits);
        } else if (g_output_pool.use_i8_output == 2) {
            bitnet_output_compute_rows_q8_neon(g_output_pool.output_q8,
                                               g_output_pool.output_q8_scales,
                                               g_output_pool.output_blocks_per_col,
                                               begin, end,
                                               g_output_pool.qhidden,
                                               g_output_pool.hidden_scale,
                                               g_output_pool.logits);
        } else if (g_output_pool.use_i8_output) {
            bitnet_output_compute_rows_i8_neon(g_output_pool.output_data,
                                               g_output_pool.output_col_size,
                                               g_output_pool.output_blocks_per_col,
                                               begin, end,
                                               g_output_pool.qhidden,
                                               g_output_pool.hidden_scale,
                                               g_output_pool.logits);
        } else {
            bitnet_output_compute_rows(g_output_pool.output_data,
                                       g_output_pool.output_col_size,
                                       g_output_pool.output_blocks_per_col,
                                       begin, end,
                                       g_output_pool.hidden,
                                       g_output_pool.logits);
        }
    }
}

static void *bitnet_output_pool_worker_main(void *ptr) {
    unsigned seen_generation = 0;
    (void)ptr;

#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    for (;;) {
        int spin_count = 0;
        while (atomic_load_explicit(&g_output_pool.generation, memory_order_acquire) == seen_generation) {
            if (atomic_load_explicit(&g_output_pool.stop, memory_order_acquire)) {
                return NULL;
            }
            if (atomic_load_explicit(&g_output_pool.park, memory_order_acquire)) {
                usleep(BITNET_OUTPUT_PARK_USLEEP);
                spin_count = 0;
                continue;
            }
            if (++spin_count < BITNET_OUTPUT_SPIN_ITERS) {
                __asm__ __volatile__("yield" ::: "memory");
            } else {
                usleep(1);
                spin_count = 0;
            }
        }
        seen_generation = atomic_load_explicit(&g_output_pool.generation, memory_order_acquire);

        bitnet_output_pool_run_chunks();

        atomic_fetch_add_explicit(&g_output_pool.completed, 1, memory_order_release);
    }
}

static void init_output_pool_once(void) {
    int n_threads = bitnet_choose_thread_count();

    if (n_threads <= 1) {
        g_output_pool.initialized = 1;
        return;
    }

    for (int i = 0; i < n_threads; ++i) {
        if (pthread_create(&g_output_pool.threads[i], NULL,
                           bitnet_output_pool_worker_main, NULL) != 0) {
            break;
        }
        ++g_output_pool.n_threads;
    }

    atomic_store_explicit(&g_output_pool.completed, g_output_pool.n_threads, memory_order_release);
    g_output_pool.initialized = 1;
}

static int BITNET_MAYBE_UNUSED bitnet_output_projection_parallel(const uint8_t *output_data,
                                                                 size_t output_col_size,
                                                                 int output_blocks_per_col,
                                                                 int vocab_size,
                                                                 const float *hidden,
                                                                 float *logits) {
    (void)pthread_once(&g_output_pool_once, init_output_pool_once);

    if (!g_output_pool.initialized || g_output_pool.n_threads <= 1 || vocab_size < 64) {
        bitnet_output_compute_rows(output_data, output_col_size, output_blocks_per_col,
                                   0, vocab_size, hidden, logits);
        return 0;
    }

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);

    g_output_pool.output_data = output_data;
    g_output_pool.output_col_size = output_col_size;
    g_output_pool.output_blocks_per_col = output_blocks_per_col;
    g_output_pool.vocab_size = vocab_size;
    g_output_pool.hidden = hidden;
    g_output_pool.qhidden = NULL;
    g_output_pool.hidden_scale = 0.0f;
    g_output_pool.use_i8_output = 0;
    g_output_pool.logits = logits;
    g_output_pool.output_q8 = NULL;
    g_output_pool.output_q8_scales = NULL;
    g_output_pool.output_q8_scales_i8 = NULL;
    g_output_pool.output_q8_d = NULL;
    g_output_pool.chunk_rows = bitnet_choose_output_chunk_rows();
    atomic_store_explicit(&g_output_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_output_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_output_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_output_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    bitnet_output_pool_run_chunks();
#endif

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);
    atomic_store_explicit(&g_output_pool.park, 1, memory_order_release);

    return 0;
}

static int BITNET_MAYBE_UNUSED bitnet_output_projection_i8_neon_parallel(const uint8_t *output_data,
                                                                         size_t output_col_size,
                                                                         int output_blocks_per_col,
                                                                         int vocab_size,
                                                                         const int8_t *qhidden,
                                                                         float hidden_scale,
                                                                         float *logits) {
    (void)pthread_once(&g_output_pool_once, init_output_pool_once);

    if (!g_output_pool.initialized || g_output_pool.n_threads <= 1 || vocab_size < 64) {
        bitnet_output_compute_rows_i8_neon(output_data, output_col_size, output_blocks_per_col,
                                           0, vocab_size, qhidden, hidden_scale, logits);
        return 0;
    }

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);

    g_output_pool.output_data = output_data;
    g_output_pool.output_col_size = output_col_size;
    g_output_pool.output_blocks_per_col = output_blocks_per_col;
    g_output_pool.vocab_size = vocab_size;
    g_output_pool.hidden = NULL;
    g_output_pool.qhidden = qhidden;
    g_output_pool.hidden_scale = hidden_scale;
    g_output_pool.use_i8_output = 1;
    g_output_pool.output_q8 = NULL;
    g_output_pool.output_q8_scales = NULL;
    g_output_pool.output_q8_scales_i8 = NULL;
    g_output_pool.output_q8_d = NULL;
    g_output_pool.logits = logits;
    g_output_pool.chunk_rows = bitnet_choose_output_chunk_rows();
    atomic_store_explicit(&g_output_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_output_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_output_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_output_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    bitnet_output_pool_run_chunks();
#endif

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);
    atomic_store_explicit(&g_output_pool.park, 1, memory_order_release);

    return 0;
}

static int BITNET_MAYBE_UNUSED bitnet_output_projection_q8_neon_parallel(const int8_t *output_q8,
                                                                         const float *output_q8_scales,
                                                                         int output_blocks_per_col,
                                                                         int vocab_size,
                                                                         const int8_t *qhidden,
                                                                         float hidden_scale,
                                                                         float *logits) {
    (void)pthread_once(&g_output_pool_once, init_output_pool_once);

    if (!g_output_pool.initialized || g_output_pool.n_threads <= 1 || vocab_size < 64) {
        bitnet_output_compute_rows_q8_neon(output_q8, output_q8_scales,
                                           output_blocks_per_col, 0, vocab_size,
                                           qhidden, hidden_scale, logits);
        return 0;
    }

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);

    g_output_pool.output_data = NULL;
    g_output_pool.output_col_size = 0;
    g_output_pool.output_blocks_per_col = output_blocks_per_col;
    g_output_pool.vocab_size = vocab_size;
    g_output_pool.hidden = NULL;
    g_output_pool.qhidden = qhidden;
    g_output_pool.hidden_scale = hidden_scale;
    g_output_pool.use_i8_output = 2;
    g_output_pool.output_q8 = output_q8;
    g_output_pool.output_q8_scales = output_q8_scales;
    g_output_pool.output_q8_scales_i8 = NULL;
    g_output_pool.output_q8_d = NULL;
    g_output_pool.logits = logits;
    g_output_pool.chunk_rows = bitnet_choose_output_chunk_rows();
    atomic_store_explicit(&g_output_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_output_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_output_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_output_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    bitnet_output_pool_run_chunks();
#endif

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);
    atomic_store_explicit(&g_output_pool.park, 1, memory_order_release);

    return 0;
}

static int BITNET_MAYBE_UNUSED bitnet_output_projection_q8_compact_neon_parallel(const int8_t *output_q8,
                                                                                 const int8_t *output_q8_scales_i8,
                                                                                 const float *output_q8_d,
                                                                                 int output_blocks_per_col,
                                                                                 int vocab_size,
                                                                                 const int8_t *qhidden,
                                                                                 float hidden_scale,
                                                                                 float *logits) {
    (void)pthread_once(&g_output_pool_once, init_output_pool_once);

    if (!g_output_pool.initialized || g_output_pool.n_threads <= 1 || vocab_size < 64) {
        bitnet_output_compute_rows_q8_compact_neon(output_q8, output_q8_scales_i8,
                                                   output_q8_d, output_blocks_per_col,
                                                   0, vocab_size, qhidden, hidden_scale, logits);
        return 0;
    }

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);

    g_output_pool.output_data = NULL;
    g_output_pool.output_col_size = 0;
    g_output_pool.output_blocks_per_col = output_blocks_per_col;
    g_output_pool.vocab_size = vocab_size;
    g_output_pool.hidden = NULL;
    g_output_pool.qhidden = qhidden;
    g_output_pool.hidden_scale = hidden_scale;
    g_output_pool.use_i8_output = 3;
    g_output_pool.output_q8 = output_q8;
    g_output_pool.output_q8_scales = NULL;
    g_output_pool.output_q8_scales_i8 = output_q8_scales_i8;
    g_output_pool.output_q8_d = output_q8_d;
    g_output_pool.logits = logits;
    g_output_pool.chunk_rows = bitnet_choose_q8_compact_output_chunk_rows(output_blocks_per_col);
    atomic_store_explicit(&g_output_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_output_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_output_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_output_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    bitnet_output_pool_run_chunks();
#endif

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);
    atomic_store_explicit(&g_output_pool.park, 1, memory_order_release);

    return 0;
}

static int BITNET_MAYBE_UNUSED bitnet_output_projection_q8_blockscale_neon_parallel(const int8_t *output_q8,
                                                                                    const float *output_q8_d,
                                                                                    int output_blocks_per_col,
                                                                                    int vocab_size,
                                                                                    const int8_t *qhidden,
                                                                                    float hidden_scale,
                                                                                    float *logits) {
    (void)pthread_once(&g_output_pool_once, init_output_pool_once);

    if (!g_output_pool.initialized || g_output_pool.n_threads <= 1 || vocab_size < 64) {
        bitnet_output_compute_rows_q8_blockscale_neon(output_q8, output_q8_d,
                                                      output_blocks_per_col, 0, vocab_size,
                                                      qhidden, hidden_scale, logits);
        return 0;
    }

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);

    g_output_pool.output_data = NULL;
    g_output_pool.output_col_size = 0;
    g_output_pool.output_blocks_per_col = output_blocks_per_col;
    g_output_pool.vocab_size = vocab_size;
    g_output_pool.hidden = NULL;
    g_output_pool.qhidden = qhidden;
    g_output_pool.hidden_scale = hidden_scale;
    g_output_pool.use_i8_output = 4;
    g_output_pool.output_q8 = output_q8;
    g_output_pool.output_q8_scales = NULL;
    g_output_pool.output_q8_scales_i8 = NULL;
    g_output_pool.output_q8_d = output_q8_d;
    g_output_pool.logits = logits;
    /* 0.5B tied output has many short rows; smaller chunks balance better on 865. */
    g_output_pool.chunk_rows = bitnet_choose_output_chunk_rows_with_default(64);
    atomic_store_explicit(&g_output_pool.park, 0, memory_order_release);
    atomic_store_explicit(&g_output_pool.next_row, 0, memory_order_relaxed);
    atomic_store_explicit(&g_output_pool.completed, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_output_pool.generation, 1, memory_order_release);

#if BITNET_DISPATCHER_RUNS_WORK
    bitnet_output_pool_run_chunks();
#endif

    BITNET_OUTPUT_SPIN_WAIT(
        atomic_load_explicit(&g_output_pool.completed, memory_order_acquire) >= g_output_pool.n_threads);
    atomic_store_explicit(&g_output_pool.park, 1, memory_order_release);

    return 0;
}

static int metadata_string_equals(const gguf_file_t *file, const char *key, const char *expected) {
    const gguf_metadata_t *entry = gguf_find_metadata(file, key);
    return entry != NULL && entry->type == GGUF_TYPE_STRING && entry->string_value != NULL && strcmp(entry->string_value, expected) == 0;
}

static int metadata_u32_equals(const gguf_file_t *file, const char *key, uint32_t expected) {
    const gguf_metadata_t *entry = gguf_find_metadata(file, key);
    return entry != NULL && entry->type == GGUF_TYPE_UINT32 && entry->value.u64 == expected;
}

static int tensor_matches(const gguf_file_t *file, const char *name, uint32_t type, const uint64_t *dims, uint32_t n_dims) {
    const gguf_tensor_t *tensor = gguf_find_tensor(file, name);
    return tensor != NULL && tensor->type == type && gguf_tensor_has_shape(tensor, dims, n_dims);
}

typedef struct bitnet_model_config {
    uint32_t embedding_length;
    uint32_t block_count;
    uint32_t head_count;
    uint32_t head_count_kv;
    uint32_t attention_key_length;
    uint32_t attention_value_length;
    uint32_t feed_forward_length;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t rope_dimension_count;
    int is_minicpm;
    float embedding_scale;
    float residual_scale;
    float logit_scale;
} bitnet_model_config_t;

static const char *model_config_prefix(const gguf_file_t *file) {
    const gguf_metadata_t *entry = gguf_find_metadata(file, "general.architecture");
    if (entry == NULL || entry->type != GGUF_TYPE_STRING || entry->string_value == NULL) {
        return NULL;
    }
    if (strcmp(entry->string_value, "llama") == 0) {
        return "llama";
    }
    if (strcmp(entry->string_value, "minicpm") == 0) {
        return "minicpm";
    }
    return NULL;
}

static uint32_t read_arch_config_u32(const gguf_file_t *file, const char *prefix, const char *suffix) {
    char key[96];
    if (prefix == NULL || suffix == NULL) return 0;
    (void)snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
    return read_config_u32(file, key);
}

static float read_arch_config_f32_default(const gguf_file_t *file, const char *prefix,
                                          const char *suffix, float default_value) {
    char key[96];
    if (prefix == NULL || suffix == NULL) return default_value;
    (void)snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
    return read_config_f32_default(file, key, default_value);
}

static uint32_t tensor_dim_u32(const gguf_file_t *file, const char *name, uint32_t dim) {
    const gguf_tensor_t *tensor = gguf_find_tensor(file, name);
    if (tensor == NULL || dim >= tensor->n_dims || tensor->dims[dim] > UINT32_MAX) {
        return 0;
    }
    return (uint32_t)tensor->dims[dim];
}

static int read_model_config(const gguf_file_t *file, bitnet_model_config_t *cfg) {
    const char *prefix = NULL;
    uint32_t q_dim = 0;
    uint32_t k_dim = 0;
    uint32_t v_dim = 0;

    if (file == NULL || cfg == NULL) return -1;
    memset(cfg, 0, sizeof(*cfg));

    prefix = model_config_prefix(file);
    if (prefix == NULL) return -2;
    cfg->is_minicpm = strcmp(prefix, "minicpm") == 0;

    cfg->embedding_length = read_arch_config_u32(file, prefix, "embedding_length");
    cfg->block_count = read_arch_config_u32(file, prefix, "block_count");
    cfg->feed_forward_length = read_arch_config_u32(file, prefix, "feed_forward_length");
    cfg->head_count = read_arch_config_u32(file, prefix, "attention.head_count");
    cfg->head_count_kv = read_arch_config_u32(file, prefix, "attention.head_count_kv");
    cfg->context_length = read_arch_config_u32(file, prefix, "context_length");
    cfg->vocab_size = read_arch_config_u32(file, prefix, "vocab_size");
    cfg->rope_dimension_count = read_arch_config_u32(file, prefix, "rope.dimension_count");
    cfg->attention_key_length = read_arch_config_u32(file, prefix, "attention.key_length");
    cfg->attention_value_length = read_arch_config_u32(file, prefix, "attention.value_length");
    cfg->embedding_scale = read_arch_config_f32_default(file, prefix, "embedding_scale", 1.0f);
    cfg->residual_scale = read_arch_config_f32_default(file, prefix, "residual_scale", 1.0f);
    cfg->logit_scale = read_arch_config_f32_default(file, prefix, "logit_scale", 1.0f);

    if (cfg->embedding_length == 0) {
        cfg->embedding_length = tensor_dim_u32(file, "output.weight", 0);
        if (cfg->embedding_length == 0) {
            cfg->embedding_length = tensor_dim_u32(file, "token_embd.weight", 0);
        }
    }
    if (cfg->vocab_size == 0) {
        cfg->vocab_size = tensor_dim_u32(file, "output.weight", 1);
        if (cfg->vocab_size == 0) {
            cfg->vocab_size = tensor_dim_u32(file, "token_embd.weight", 1);
        }
    }

    q_dim = tensor_dim_u32(file, "blk.0.attn_q.weight", 1);
    k_dim = tensor_dim_u32(file, "blk.0.attn_k.weight", 1);
    v_dim = tensor_dim_u32(file, "blk.0.attn_v.weight", 1);

    if (cfg->attention_key_length == 0 && cfg->head_count != 0 && q_dim % cfg->head_count == 0) {
        cfg->attention_key_length = q_dim / cfg->head_count;
    }
    if (cfg->attention_value_length == 0 && cfg->head_count_kv != 0 && v_dim % cfg->head_count_kv == 0) {
        cfg->attention_value_length = v_dim / cfg->head_count_kv;
    }
    if (cfg->attention_value_length == 0 && cfg->head_count_kv != 0 && k_dim % cfg->head_count_kv == 0) {
        cfg->attention_value_length = k_dim / cfg->head_count_kv;
    }
    if (cfg->rope_dimension_count == 0) {
        cfg->rope_dimension_count = cfg->attention_key_length;
    }

    return 0;
}

static float *build_tensor_scales(const gguf_file_t *file, const gguf_tensor_t *tensor,
                                  int out_dim, int in_dim) {
    const void *weight = gguf_get_tensor_ptr(file, tensor);
    size_t count = bitnet_tq2_0_scale_float_count(out_dim, in_dim);
    float *scales = NULL;

    if (weight == NULL || count == 0) {
        return NULL;
    }

    scales = (float *)malloc(count * sizeof(*scales));
    if (scales == NULL) {
        return NULL;
    }

    if (bitnet_tq2_0_build_scales(weight, out_dim, in_dim, scales, count) != 0) {
        free(scales);
        return NULL;
    }

    return scales;
}

static void free_tq2_scale_cache(bitnet_tq2_scale_cache_t *cache, uint32_t block_count) {
    if (cache == NULL) return;
    if (cache->blocks != NULL) {
        for (uint32_t i = 0; i < block_count; ++i) {
            free(cache->blocks[i].attn_q);
            free(cache->blocks[i].attn_k);
            free(cache->blocks[i].attn_v);
            free(cache->blocks[i].attn_output);
            free(cache->blocks[i].ffn_gate);
            free(cache->blocks[i].ffn_up);
            free(cache->blocks[i].ffn_down);
        }
        free(cache->blocks);
    }
    cache->blocks = NULL;
}

static void free_tq2_tl1_cache(bitnet_tq2_tl1_cache_t *cache, uint32_t block_count) {
    if (cache == NULL) return;
    if (cache->blocks != NULL) {
        for (uint32_t i = 0; i < block_count; ++i) {
            free(cache->blocks[i].attn_q);
            free(cache->blocks[i].attn_k);
            free(cache->blocks[i].attn_v);
            free(cache->blocks[i].attn_output);
            free(cache->blocks[i].ffn_gate);
            free(cache->blocks[i].ffn_up);
            free(cache->blocks[i].ffn_down);
        }
        free(cache->blocks);
    }
    cache->blocks = NULL;
}

static int build_tensor_i2s(const gguf_file_t *file, const gguf_tensor_t *tensor,
                              int out_dim, int in_dim,
                              uint8_t **packed_out, float **scales_out, int32_t **bsums_out) {
    const void *weight = gguf_get_tensor_ptr(file, tensor);
    size_t packed_size = bitnet_tq2_0_i2s_packed_size(out_dim, in_dim);
    int out_dim_padded = (out_dim + 3) & ~3;
    int n_groups = out_dim_padded / 4;
    int sub_blocks_per_group = in_dim / 64; /* QK_I2S = 64 */
    size_t scales_count = (size_t)n_groups * (size_t)sub_blocks_per_group * 4;
    size_t bsums_count = scales_count;
    uint8_t *packed = NULL;
    float *scales = NULL;
    int32_t *bsums = NULL;

    if (weight == NULL || packed_size == 0) {
        return -1;
    }

    packed = (uint8_t *)malloc(packed_size);
    if (packed == NULL) {
        return -1;
    }

    scales = (float *)malloc(scales_count * sizeof(float));
    if (scales == NULL) {
        free(packed);
        return -1;
    }

    bsums = (int32_t *)malloc(bsums_count * sizeof(int32_t));
    if (bsums == NULL) {
        free(scales);
        free(packed);
        return -1;
    }

    if (bitnet_tq2_0_reorder_to_i2s(weight, out_dim, in_dim, packed, scales, bsums) != 0) {
        free(bsums);
        free(scales);
        free(packed);
        return -1;
    }

    *packed_out = packed;
    *scales_out = scales;
    *bsums_out = bsums;
    return 0;
}

static void free_tq2_i2s_cache(bitnet_tq2_i2s_cache_t *cache, uint32_t block_count) {
    if (cache == NULL) return;
    if (cache->blocks != NULL) {
        for (uint32_t i = 0; i < block_count; ++i) {
            free(cache->blocks[i].attn_q);
            free(cache->blocks[i].attn_k);
            free(cache->blocks[i].attn_v);
            free(cache->blocks[i].attn_output);
            free(cache->blocks[i].ffn_gate);
            free(cache->blocks[i].ffn_up);
            free(cache->blocks[i].ffn_down);
        }
        free(cache->blocks);
        cache->blocks = NULL;
    }
    if (cache->scales != NULL) {
        for (uint32_t i = 0; i < block_count * 7; ++i) {
            free(cache->scales[i]);
        }
        free(cache->scales);
        cache->scales = NULL;
    }
    if (cache->bsums != NULL) {
        for (uint32_t i = 0; i < block_count * 7; ++i) {
            free(cache->bsums[i]);
        }
        free(cache->bsums);
        cache->bsums = NULL;
    }
}

static void free_output_q6k_cache(bitnet_model_t *model) {
    if (model == NULL) return;
    free(model->output_q8);
    free(model->output_q8_scales);
    free(model->output_q8_scales_i8);
    free(model->output_q8_d);
    model->output_q8 = NULL;
    model->output_q8_scales = NULL;
    model->output_q8_scales_i8 = NULL;
    model->output_q8_d = NULL;
    model->output_q8_blockscale = 0;
}

static int BITNET_MAYBE_UNUSED build_output_q6k_cache(bitnet_model_t *model) {
    int emb_dim = 0;
    int vocab_size = 0;
    int blocks_per_row = 0;
    const void *weight = NULL;

    if (model == NULL || model->tensor_cache.output == NULL) {
        return -1;
    }

    emb_dim = (int)model->embedding_length;
    vocab_size = (int)model->vocab_size;
    if (emb_dim <= 0 || vocab_size <= 0 || emb_dim % BITNET_Q6K_QK != 0) {
        return -1;
    }

    blocks_per_row = emb_dim / BITNET_Q6K_QK;
    weight = gguf_get_tensor_ptr(&model->gguf, model->tensor_cache.output);
    if (weight == NULL) {
        return -1;
    }

    model->output_q8 = (int8_t *)malloc((size_t)vocab_size * (size_t)emb_dim * sizeof(*model->output_q8));
    model->output_q8_scales_i8 = (int8_t *)malloc((size_t)vocab_size * (size_t)blocks_per_row * 16u *
                                                  sizeof(*model->output_q8_scales_i8));
    model->output_q8_d = (float *)malloc((size_t)vocab_size * (size_t)blocks_per_row *
                                         sizeof(*model->output_q8_d));
    if (model->output_q8 == NULL || model->output_q8_scales_i8 == NULL || model->output_q8_d == NULL) {
        free_output_q6k_cache(model);
        return -1;
    }

    if (bitnet_q6k_expand_to_q8_compact(weight, vocab_size, emb_dim,
                                        model->output_q8,
                                        model->output_q8_scales_i8,
                                        model->output_q8_d) != 0) {
        free_output_q6k_cache(model);
        return -1;
    }

    return 0;
}

static int BITNET_MAYBE_UNUSED build_output_f16_tied_cache(bitnet_model_t *model) {
    int emb_dim = 0;
    int vocab_size = 0;
    int blocks_per_row = 0;
    const uint16_t *weight = NULL;

    if (model == NULL || !model->output_is_tied_token_embd ||
        model->tensor_cache.token_embd == NULL) {
        return -1;
    }

    emb_dim = (int)model->embedding_length;
    vocab_size = (int)model->vocab_size;
    if (emb_dim <= 0 || vocab_size <= 0 || emb_dim % BITNET_Q6K_QK != 0) {
        return -1;
    }

    weight = (const uint16_t *)gguf_get_tensor_ptr(&model->gguf, model->tensor_cache.token_embd);
    if (weight == NULL) {
        return -1;
    }

    blocks_per_row = emb_dim / BITNET_Q6K_QK;
    model->output_q8 = (int8_t *)malloc((size_t)vocab_size * (size_t)emb_dim * sizeof(*model->output_q8));
    model->output_q8_d = (float *)malloc((size_t)vocab_size * (size_t)blocks_per_row *
                                         sizeof(*model->output_q8_d));
    if (model->output_q8 == NULL || model->output_q8_d == NULL) {
        free_output_q6k_cache(model);
        return -1;
    }
    model->output_q8_blockscale = 1;

    for (int row = 0; row < vocab_size; ++row) {
        const uint16_t *src_row = weight + (size_t)row * (size_t)emb_dim;
        int8_t *dst_row = model->output_q8 + (size_t)row * (size_t)emb_dim;
        float *d_row = model->output_q8_d + (size_t)row * (size_t)blocks_per_row;

        for (int b = 0; b < blocks_per_row; ++b) {
            const uint16_t *src_block = src_row + (size_t)b * BITNET_Q6K_QK;
            int8_t *dst_block = dst_row + (size_t)b * BITNET_Q6K_QK;
            float max_abs = 0.0f;

            for (int i = 0; i < BITNET_Q6K_QK; ++i) {
                float a = fabsf(bitnet_fp16_to_fp32(src_block[i]));
                if (a > max_abs) max_abs = a;
            }

            if (max_abs == 0.0f) {
                memset(dst_block, 0, BITNET_Q6K_QK * sizeof(*dst_block));
                d_row[b] = 0.0f;
                continue;
            }

            d_row[b] = max_abs / 127.0f;
            {
                float inv = 127.0f / max_abs;
                for (int i = 0; i < BITNET_Q6K_QK; ++i) {
                    float scaled = bitnet_fp16_to_fp32(src_block[i]) * inv;
                    int q = (int)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    dst_block[i] = (int8_t)q;
                }
            }
        }
    }

    return 0;
}

static int BITNET_MAYBE_UNUSED build_tq2_i2s_cache(bitnet_model_t *model) {
    int emb_dim = 0;
    int q_dim = 0;
    int kv_dim = 0;
    int attn_dim = 0;
    int ffn_dim = 0;

    if (model == NULL || model->block_count == 0) {
        return -1;
    }

    emb_dim = (int)model->embedding_length;
    q_dim = (int)bitnet_attention_dim_for_model(model);
    kv_dim = (int)bitnet_kv_dim_for_model(model);
    attn_dim = q_dim;
    ffn_dim = (int)model->feed_forward_length;

    model->i2s_cache.blocks = (bitnet_block_i2s_weights_t *)calloc(model->block_count,
                                                                     sizeof(*model->i2s_cache.blocks));
    if (model->i2s_cache.blocks == NULL) {
        return -1;
    }

    model->i2s_cache.scales = (float **)calloc(model->block_count * 7, sizeof(float *));
    if (model->i2s_cache.scales == NULL) {
        free(model->i2s_cache.blocks);
        model->i2s_cache.blocks = NULL;
        return -1;
    }

    model->i2s_cache.bsums = (int32_t **)calloc(model->block_count * 7, sizeof(int32_t *));
    if (model->i2s_cache.bsums == NULL) {
        free(model->i2s_cache.scales);
        model->i2s_cache.scales = NULL;
        free(model->i2s_cache.blocks);
        model->i2s_cache.blocks = NULL;
        return -1;
    }

    for (uint32_t i = 0; i < model->block_count; ++i) {
        const bitnet_block_tensors_t *bt = &model->tensor_cache.blocks[i];
        bitnet_block_i2s_weights_t *bw = &model->i2s_cache.blocks[i];
        float **sc = model->i2s_cache.scales + i * 7;
        int32_t **bs = model->i2s_cache.bsums + i * 7;

        /* tensor 0: attn_q */
        if (build_tensor_i2s(&model->gguf, bt->attn_q, q_dim, emb_dim,
                              &bw->attn_q, &sc[0], &bs[0]) != 0) {
            free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
            return -1;
        }
        /* tensor 1: attn_k */
        if (build_tensor_i2s(&model->gguf, bt->attn_k, kv_dim, emb_dim,
                              &bw->attn_k, &sc[1], &bs[1]) != 0) {
            free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
            return -1;
        }
        /* tensor 2: attn_v */
        if (build_tensor_i2s(&model->gguf, bt->attn_v, kv_dim, emb_dim,
                              &bw->attn_v, &sc[2], &bs[2]) != 0) {
            free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
            return -1;
        }
        /* tensor 3: attn_output */
        if (build_tensor_i2s(&model->gguf, bt->attn_output, emb_dim, attn_dim,
                              &bw->attn_output, &sc[3], &bs[3]) != 0) {
            free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
            return -1;
        }
        /* tensor 4: ffn_gate */
        if (build_tensor_i2s(&model->gguf, bt->ffn_gate, ffn_dim, emb_dim,
                              &bw->ffn_gate, &sc[4], &bs[4]) != 0) {
            free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
            return -1;
        }
        /* tensor 5: ffn_up */
        if (build_tensor_i2s(&model->gguf, bt->ffn_up, ffn_dim, emb_dim,
                              &bw->ffn_up, &sc[5], &bs[5]) != 0) {
            free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
            return -1;
        }
        /* tensor 6: ffn_down */
        if (build_tensor_i2s(&model->gguf, bt->ffn_down, emb_dim, ffn_dim,
                              &bw->ffn_down, &sc[6], &bs[6]) != 0) {
            free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
            return -1;
        }
    }

    return 0;
}

static uint8_t *build_tensor_tl1(const gguf_file_t *file, const gguf_tensor_t *tensor,
                                   int out_dim, int in_dim) {
    const void *weight = gguf_get_tensor_ptr(file, tensor);
    size_t reordered_size = (size_t)out_dim * (size_t)(in_dim / BITNET_TQ2_0_QK) * BITNET_TQ2_0_QS_SIZE;
    uint8_t *reordered = NULL;

    if (weight == NULL || reordered_size == 0) {
        return NULL;
    }

    reordered = (uint8_t *)malloc(reordered_size);
    if (reordered == NULL) {
        return NULL;
    }

    if (bitnet_tq2_0_reorder_to_tl1(weight, out_dim, in_dim, reordered) != 0) {
        free(reordered);
        return NULL;
    }

    return reordered;
}

static int build_tq2_scale_cache(bitnet_model_t *model) {
    int emb_dim = 0;
    int q_dim = 0;
    int kv_dim = 0;
    int attn_dim = 0;
    int ffn_dim = 0;

    if (model == NULL || model->block_count == 0) {
        return -1;
    }

    emb_dim = (int)model->embedding_length;
    q_dim = (int)bitnet_attention_dim_for_model(model);
    kv_dim = (int)bitnet_kv_dim_for_model(model);
    attn_dim = q_dim;
    ffn_dim = (int)model->feed_forward_length;

    model->scale_cache.blocks = (bitnet_block_scale_cache_t *)calloc(model->block_count,
                                                                     sizeof(*model->scale_cache.blocks));
    if (model->scale_cache.blocks == NULL) {
        return -1;
    }

    for (uint32_t i = 0; i < model->block_count; ++i) {
        const bitnet_block_tensors_t *bt = &model->tensor_cache.blocks[i];
        bitnet_block_scale_cache_t *bs = &model->scale_cache.blocks[i];

        bs->attn_q = build_tensor_scales(&model->gguf, bt->attn_q, q_dim, emb_dim);
        bs->attn_k = build_tensor_scales(&model->gguf, bt->attn_k, kv_dim, emb_dim);
        bs->attn_v = build_tensor_scales(&model->gguf, bt->attn_v, kv_dim, emb_dim);
        bs->attn_output = build_tensor_scales(&model->gguf, bt->attn_output, emb_dim, attn_dim);
        bs->ffn_gate = build_tensor_scales(&model->gguf, bt->ffn_gate, ffn_dim, emb_dim);
        bs->ffn_up = build_tensor_scales(&model->gguf, bt->ffn_up, ffn_dim, emb_dim);
        bs->ffn_down = build_tensor_scales(&model->gguf, bt->ffn_down, emb_dim, ffn_dim);

        if (bs->attn_q == NULL || bs->attn_k == NULL || bs->attn_v == NULL ||
            bs->attn_output == NULL || bs->ffn_gate == NULL || bs->ffn_up == NULL ||
            bs->ffn_down == NULL) {
            free_tq2_scale_cache(&model->scale_cache, model->block_count);
            return -1;
        }
    }

    return 0;
}

static int BITNET_MAYBE_UNUSED build_tq2_tl1_cache(bitnet_model_t *model) {
    int emb_dim = 0;
    int q_dim = 0;
    int kv_dim = 0;
    int attn_dim = 0;
    int ffn_dim = 0;

    if (model == NULL || model->block_count == 0) {
        return -1;
    }

    emb_dim = (int)model->embedding_length;
    q_dim = (int)bitnet_attention_dim_for_model(model);
    kv_dim = (int)bitnet_kv_dim_for_model(model);
    attn_dim = q_dim;
    ffn_dim = (int)model->feed_forward_length;

    model->tl1_cache.blocks = (bitnet_block_tl1_weights_t *)calloc(model->block_count,
                                                                     sizeof(*model->tl1_cache.blocks));
    if (model->tl1_cache.blocks == NULL) {
        return -1;
    }

    for (uint32_t i = 0; i < model->block_count; ++i) {
        const bitnet_block_tensors_t *bt = &model->tensor_cache.blocks[i];
        bitnet_block_tl1_weights_t *tw = &model->tl1_cache.blocks[i];

        tw->attn_q = build_tensor_tl1(&model->gguf, bt->attn_q, q_dim, emb_dim);
        tw->attn_k = build_tensor_tl1(&model->gguf, bt->attn_k, kv_dim, emb_dim);
        tw->attn_v = build_tensor_tl1(&model->gguf, bt->attn_v, kv_dim, emb_dim);
        tw->attn_output = build_tensor_tl1(&model->gguf, bt->attn_output, emb_dim, attn_dim);
        tw->ffn_gate = build_tensor_tl1(&model->gguf, bt->ffn_gate, ffn_dim, emb_dim);
        tw->ffn_up = build_tensor_tl1(&model->gguf, bt->ffn_up, ffn_dim, emb_dim);
        tw->ffn_down = build_tensor_tl1(&model->gguf, bt->ffn_down, emb_dim, ffn_dim);

        if (tw->attn_q == NULL || tw->attn_k == NULL || tw->attn_v == NULL ||
            tw->attn_output == NULL || tw->ffn_gate == NULL || tw->ffn_up == NULL ||
            tw->ffn_down == NULL) {
            free_tq2_tl1_cache(&model->tl1_cache, model->block_count);
            return -1;
        }
    }

    return 0;
}

int bitnet_validate_target_model(const gguf_file_t *file) {
    bitnet_model_config_t cfg;
    uint32_t q_dim = 0;
    uint32_t kv_dim = 0;

    if (file == NULL) return -1;
    if (model_config_prefix(file) == NULL) return -2;
    if (!metadata_string_equals(file, "tokenizer.ggml.model", BITNET_TARGET_TOKENIZER_MODEL)) return -3;
    if (!metadata_u32_equals(file, "general.file_type", BITNET_TARGET_FILE_TYPE)) return -4;

    if (read_model_config(file, &cfg) != 0 ||
        cfg.embedding_length == 0 || cfg.block_count == 0 || cfg.feed_forward_length == 0 ||
        cfg.head_count == 0 || cfg.head_count_kv == 0 || cfg.vocab_size == 0 ||
        cfg.rope_dimension_count == 0 || cfg.attention_key_length == 0 ||
        cfg.attention_value_length == 0) {
        return -5;
    }
    if (cfg.attention_key_length != cfg.attention_value_length ||
        cfg.rope_dimension_count > cfg.attention_key_length) {
        return -6;
    }
    if ((cfg.embedding_length % BITNET_Q4K_QK) != 0 ||
        (cfg.embedding_length % BITNET_Q6K_QK) != 0 ||
        (cfg.embedding_length % BITNET_TQ2_0_QK) != 0 ||
        (cfg.feed_forward_length % BITNET_TQ2_0_QK) != 0 ||
        (cfg.attention_key_length % 16u) != 0) {
        return -7;
    }

    q_dim = cfg.head_count * cfg.attention_key_length;
    kv_dim = cfg.head_count_kv * cfg.attention_key_length;

    {
        const gguf_tensor_t *token_embd = gguf_find_tensor(file, "token_embd.weight");
        const gguf_tensor_t *output = gguf_find_tensor(file, "output.weight");
        const uint64_t token_dims[] = {cfg.embedding_length, cfg.vocab_size};
        const uint64_t output_dims[] = {cfg.embedding_length, cfg.vocab_size};
        const uint64_t norm_dims[] = {cfg.embedding_length};
        const uint64_t attn_q_dims[] = {cfg.embedding_length, q_dim};
        const uint64_t attn_kv_dims[] = {cfg.embedding_length, kv_dim};
        const uint64_t attn_output_dims[] = {q_dim, cfg.embedding_length};
        const uint64_t ffn_down_dims[] = {cfg.feed_forward_length, cfg.embedding_length};
        const uint64_t ffn_up_dims[] = {cfg.embedding_length, cfg.feed_forward_length};

        if (token_embd == NULL || !gguf_tensor_has_shape(token_embd, token_dims, 2)) return -19;
        if (token_embd->type != BITNET_TARGET_TENSOR_TYPE_Q4_K &&
            token_embd->type != BITNET_TARGET_TENSOR_TYPE_F16) return -19;

        if (output != NULL) {
            if (token_embd->type != BITNET_TARGET_TENSOR_TYPE_Q4_K) return -20;
            if (!gguf_tensor_has_shape(output, output_dims, 2) ||
                output->type != BITNET_TARGET_TENSOR_TYPE_Q6_K) return -20;
        } else {
            if (token_embd->type != BITNET_TARGET_TENSOR_TYPE_F16) return -20;
        }
        if (!tensor_matches(file, "output_norm.weight", BITNET_TARGET_TENSOR_TYPE_F32, norm_dims, 1)) return -21;
        if (!tensor_matches(file, "blk.0.attn_q.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, attn_q_dims, 2)) return -22;
        if (!tensor_matches(file, "blk.0.attn_k.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, attn_kv_dims, 2)) return -23;
        if (!tensor_matches(file, "blk.0.attn_v.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, attn_kv_dims, 2)) return -24;
        if (!tensor_matches(file, "blk.0.attn_output.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, attn_output_dims, 2)) return -25;
        if (!tensor_matches(file, "blk.0.ffn_down.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, ffn_down_dims, 2)) return -26;
        if (!tensor_matches(file, "blk.0.ffn_gate.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, ffn_up_dims, 2)) return -27;
        if (!tensor_matches(file, "blk.0.ffn_up.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, ffn_up_dims, 2)) return -28;
        if (!tensor_matches(file, "blk.0.attn_norm.weight", BITNET_TARGET_TENSOR_TYPE_F32, norm_dims, 1)) return -29;
        if (!tensor_matches(file, "blk.0.ffn_norm.weight", BITNET_TARGET_TENSOR_TYPE_F32, norm_dims, 1)) return -30;
    }

    return 0;
}

/* ---- tensor cache: pre-resolve all tensor names at load time ---- */

int bitnet_build_tensor_cache(const gguf_file_t *file, uint32_t block_count, bitnet_tensor_cache_t *cache) {
    uint32_t i = 0;

    if (file == NULL || cache == NULL || block_count == 0) return -1;

    cache->token_embd = gguf_find_tensor(file, "token_embd.weight");
    cache->output = gguf_find_tensor(file, "output.weight");
    cache->output_norm = gguf_find_tensor(file, "output_norm.weight");

    if (cache->token_embd == NULL || cache->output_norm == NULL) {
        return -1;
    }
    if (cache->output == NULL && cache->token_embd->type != BITNET_TARGET_TENSOR_TYPE_F16) {
        return -1;
    }

    cache->blocks = (bitnet_block_tensors_t *)calloc(block_count, sizeof(*cache->blocks));
    if (cache->blocks == NULL) return -1;

    for (i = 0; i < block_count; ++i) {
        bitnet_block_tensors_t *bt = &cache->blocks[i];
        char tname[64];

        (void)snprintf(tname, sizeof(tname), "blk.%u.attn_norm.weight", i);
        bt->attn_norm = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.attn_q.weight", i);
        bt->attn_q = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.attn_k.weight", i);
        bt->attn_k = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.attn_v.weight", i);
        bt->attn_v = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.attn_output.weight", i);
        bt->attn_output = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.ffn_norm.weight", i);
        bt->ffn_norm = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.ffn_gate.weight", i);
        bt->ffn_gate = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.ffn_up.weight", i);
        bt->ffn_up = gguf_find_tensor(file, tname);

        (void)snprintf(tname, sizeof(tname), "blk.%u.ffn_down.weight", i);
        bt->ffn_down = gguf_find_tensor(file, tname);

        if (bt->attn_norm == NULL || bt->attn_q == NULL || bt->attn_k == NULL ||
            bt->attn_v == NULL || bt->attn_output == NULL || bt->ffn_norm == NULL ||
            bt->ffn_gate == NULL || bt->ffn_up == NULL || bt->ffn_down == NULL) {
            bitnet_free_tensor_cache(cache);
            return -1;
        }
    }

    return 0;
}

void bitnet_free_tensor_cache(bitnet_tensor_cache_t *cache) {
    if (cache == NULL) return;
    free(cache->blocks);
    cache->blocks = NULL;
    cache->token_embd = NULL;
    cache->output = NULL;
    cache->output_norm = NULL;
}

/* ---- model load / context create ---- */

bitnet_model_t *bitnet_load_model(const char *path) {
    bitnet_model_t *model = NULL;
    bitnet_model_config_t cfg;
    size_t path_len = 0;

    if (path == NULL) {
        return NULL;
    }

    model = (bitnet_model_t *)calloc(1, sizeof(*model));
    if (model == NULL) {
        return NULL;
    }

    path_len = strlen(path) + 1;
    model->model_path = (char *)malloc(path_len);
    if (model->model_path == NULL) {
        free(model);
        return NULL;
    }
    (void)memcpy(model->model_path, path, path_len);

    if (gguf_open(path, &model->gguf) != 0) {
        bitnet_free_model(model);
        return NULL;
    }

    if (bitnet_validate_target_model(&model->gguf) != 0) {
        bitnet_free_model(model);
        return NULL;
    }

    if (bitnet_tokenizer_load(&model->tokenizer, path) != 0) {
        bitnet_free_model(model);
        return NULL;
    }

    if (read_model_config(&model->gguf, &cfg) != 0) {
        bitnet_free_model(model);
        return NULL;
    }
    bitnet_apply_android_affinity_for_model(cfg.embedding_length);

    model->embedding_length = cfg.embedding_length;
    model->block_count = cfg.block_count;
    model->head_count = cfg.head_count;
    model->head_count_kv = cfg.head_count_kv;
    model->attention_key_length = cfg.attention_key_length;
    model->attention_value_length = cfg.attention_value_length;
    model->feed_forward_length = cfg.feed_forward_length;
    model->context_length = cfg.context_length;
    model->vocab_size = cfg.vocab_size;
    model->rope_dimension_count = cfg.rope_dimension_count;
    model->is_minicpm = cfg.is_minicpm;
    model->embedding_scale = cfg.embedding_scale;
    model->residual_scale = cfg.residual_scale;
    model->logit_scale = cfg.logit_scale;

    if (bitnet_build_tensor_cache(&model->gguf, model->block_count, &model->tensor_cache) != 0) {
        bitnet_free_model(model);
        return NULL;
    }
    model->token_embd_type = model->tensor_cache.token_embd->type;
    model->output_is_tied_token_embd =
        model->tensor_cache.output == NULL &&
        model->token_embd_type == BITNET_TARGET_TENSOR_TYPE_F16;

    if (build_tq2_scale_cache(model) != 0) {
        bitnet_free_model(model);
        return NULL;
    }

#if BITNET_USE_TQ2_TL1_LUT
    if (build_tq2_tl1_cache(model) != 0) {
        bitnet_free_model(model);
        return NULL;
    }
#endif

#if BITNET_USE_TQ2_I2S
    if (build_tq2_i2s_cache(model) != 0) {
        bitnet_free_model(model);
        return NULL;
    }
#endif

#if BITNET_USE_Q6K_NEON_OUTPUT
    if (model->output_is_tied_token_embd) {
        if (build_output_f16_tied_cache(model) != 0) {
            bitnet_free_model(model);
            return NULL;
        }
    } else {
        (void)build_output_q6k_cache(model);
    }
#endif

    return model;
}

bitnet_context_t *bitnet_create_context(bitnet_model_t *model, int max_tokens) {
    bitnet_context_t *ctx = NULL;
    size_t kv_dim = 0;
    size_t kv_size = 0;
    int emb_dim = 0;
    int q_dim = 0;
    int head_dim = 0;
    int n_kv_heads = 0;
    int attn_dim = 0;
    int ffn_dim = 0;
    size_t tq2_lut_count = 0;
    size_t tq2_tl1_lut_size_hidden = 0;
    size_t tq2_tl1_lut_size_ffn = 0;
    size_t rope_table_count = 0;

    if (model == NULL || max_tokens <= 0) {
        return NULL;
    }

    emb_dim = (int)model->embedding_length;
    n_kv_heads = (int)model->head_count_kv;
    head_dim = (int)model->attention_key_length;
    q_dim = (int)bitnet_attention_dim_for_model(model);
    attn_dim = q_dim;
    ffn_dim = (int)model->feed_forward_length;
    kv_dim = (size_t)n_kv_heads * head_dim;
    kv_size = (size_t)model->block_count * (size_t)max_tokens * kv_dim;
    tq2_lut_count = bitnet_tq2_0_lut_float_count(ffn_dim);
    tq2_tl1_lut_size_hidden = bitnet_tq2_0_tl1_lut_size(emb_dim);
    tq2_tl1_lut_size_ffn = bitnet_tq2_0_tl1_lut_size(ffn_dim);
    rope_table_count = (size_t)max_tokens * (size_t)(model->rope_dimension_count / 2u);

    ctx = (bitnet_context_t *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->model = model;
    ctx->max_tokens = max_tokens;
    ctx->kv_cache_type = BITNET_KV_CACHE_F32;

    ctx->key_cache = (float *)calloc(kv_size, sizeof(float));
    ctx->value_cache = (float *)calloc(kv_size, sizeof(float));
    ctx->logits = (float *)calloc((size_t)model->vocab_size, sizeof(float));
    ctx->last_hidden = (float *)calloc((size_t)emb_dim, sizeof(float));

    /* pre-allocate eval buffers */
    ctx->hidden = (float *)calloc((size_t)emb_dim, sizeof(float));
    ctx->q = (float *)calloc((size_t)q_dim, sizeof(float));
    ctx->k = (float *)calloc(kv_dim, sizeof(float));
    ctx->v = (float *)calloc(kv_dim, sizeof(float));
    ctx->attn_buffer = (float *)calloc((size_t)attn_dim, sizeof(float));
    ctx->gate = (float *)calloc((size_t)ffn_dim, sizeof(float));
    ctx->up = (float *)calloc((size_t)ffn_dim, sizeof(float));
    ctx->down = (float *)calloc((size_t)ffn_dim, sizeof(float));
    ctx->scores = (float *)calloc((size_t)max_tokens * (size_t)model->head_count, sizeof(float));
    ctx->tmp_out = (float *)calloc((size_t)emb_dim, sizeof(float));
    ctx->tq2_lut = (float *)calloc(tq2_lut_count, sizeof(float));
    ctx->tq2_qhidden = (int8_t *)calloc((size_t)(q_dim > emb_dim ? q_dim : emb_dim),
                                        sizeof(*ctx->tq2_qhidden));
    ctx->tq2_qffn = (int8_t *)calloc((size_t)ffn_dim, sizeof(*ctx->tq2_qffn));
    /* TL1 LUT: allocate for the larger of hidden/ffn dimension */
    {
        size_t max_tl1 = tq2_tl1_lut_size_hidden > tq2_tl1_lut_size_ffn ?
                         tq2_tl1_lut_size_hidden : tq2_tl1_lut_size_ffn;
        ctx->tq2_tl1_lut = (int8_t *)calloc(max_tl1, sizeof(*ctx->tq2_tl1_lut));
        ctx->tq2_tl1_lut_size = max_tl1;
    }
    ctx->rope_cos = (float *)calloc(rope_table_count, sizeof(float));
    ctx->rope_sin = (float *)calloc(rope_table_count, sizeof(float));
    ctx->tq2_lut_count = tq2_lut_count;

    if (ctx->key_cache == NULL || ctx->value_cache == NULL || ctx->logits == NULL ||
        ctx->last_hidden == NULL ||
        ctx->hidden == NULL || ctx->q == NULL || ctx->k == NULL || ctx->v == NULL ||
        ctx->attn_buffer == NULL || ctx->gate == NULL || ctx->up == NULL ||
        ctx->down == NULL || ctx->scores == NULL || ctx->tmp_out == NULL ||
        ctx->tq2_lut == NULL || ctx->tq2_qhidden == NULL || ctx->tq2_qffn == NULL ||
        ctx->tq2_tl1_lut == NULL || ctx->rope_cos == NULL || ctx->rope_sin == NULL) {
        bitnet_free_context(ctx);
        return NULL;
    }

    for (int pos = 0; pos < max_tokens; ++pos) {
        for (int j = 0; j < (int)model->rope_dimension_count / 2; ++j) {
            float theta = 1.0f / powf(10000.0f, (float)(2 * j) / (float)head_dim);
            size_t idx = (size_t)pos * (size_t)(model->rope_dimension_count / 2u) + (size_t)j;
            ctx->rope_cos[idx] = cosf((float)pos * theta);
            ctx->rope_sin[idx] = sinf((float)pos * theta);
        }
    }

    return ctx;
}

int bitnet_tokenize_ex(bitnet_model_t *model, const char *text, int *tokens, int max_tokens, int add_bos) {
    int n = 0;
    if (model == NULL || model->tokenizer == NULL) {
        return -1;
    }
    if (add_bos && bitnet_tokenizer_add_bos(model->tokenizer) && max_tokens > 0) {
        int bos = bitnet_tokenizer_bos_id(model->tokenizer);
        if (bos >= 0) {
            tokens[0] = bos;
            n = 1;
        }
    }
    {
        int enc = bitnet_tokenizer_encode(model->tokenizer, text, tokens + n, max_tokens - n);
        if (enc < 0) return enc;
        n += enc;
    }
    return n;
}

int bitnet_tokenize(bitnet_model_t *model, const char *text, int *tokens, int max_tokens) {
    return bitnet_tokenize_ex(model, text, tokens, max_tokens, 1);
}

int bitnet_context_pos(const bitnet_context_t *ctx) {
    if (ctx == NULL) return -1;
    return ctx->n_pos;
}

int bitnet_context_capacity(const bitnet_context_t *ctx) {
    if (ctx == NULL) return -1;
    return ctx->max_tokens;
}

int bitnet_reset_context(bitnet_context_t *ctx) {
    if (ctx == NULL) return -1;
    ctx->n_pos = 0;
    if (ctx->logits != NULL && ctx->model != NULL) {
        memset(ctx->logits, 0, (size_t)ctx->model->vocab_size * sizeof(*ctx->logits));
    }
    return 0;
}

int bitnet_rewind_context(bitnet_context_t *ctx, int n_pos) {
    if (ctx == NULL || n_pos < 0 || n_pos > ctx->n_pos) {
        return -1;
    }
    ctx->n_pos = n_pos;
    return 0;
}

int bitnet_get_kv_cache_type(const bitnet_context_t *ctx) {
    if (ctx == NULL) return -1;
    return ctx->kv_cache_type;
}

int bitnet_set_kv_cache_type(bitnet_context_t *ctx, int cache_type) {
    size_t kv_count = 0;

    if (ctx == NULL || ctx->model == NULL) {
        return -1;
    }
    if (cache_type != BITNET_KV_CACHE_F32 && cache_type != BITNET_KV_CACHE_Q8) {
        return -1;
    }
    if (ctx->n_pos != 0) {
        return -2;
    }
    if (ctx->kv_cache_type == cache_type) {
        return 0;
    }

    kv_count = bitnet_kv_element_count(ctx);
    if (cache_type == BITNET_KV_CACHE_Q8) {
        if (bitnet_allocate_q8_kv_cache(ctx) != 0) {
            return -1;
        }
        free(ctx->key_cache);
        free(ctx->value_cache);
        ctx->key_cache = NULL;
        ctx->value_cache = NULL;
    } else {
        ctx->key_cache = (float *)calloc(kv_count, sizeof(*ctx->key_cache));
        ctx->value_cache = (float *)calloc(kv_count, sizeof(*ctx->value_cache));
        if (ctx->key_cache == NULL || ctx->value_cache == NULL) {
            free(ctx->key_cache);
            free(ctx->value_cache);
            ctx->key_cache = NULL;
            ctx->value_cache = NULL;
            return -1;
        }
        bitnet_free_q8_kv_cache(ctx);
    }

    ctx->kv_cache_type = cache_type;
    return 0;
}

unsigned long long bitnet_kv_cache_bytes(const bitnet_context_t *ctx) {
    size_t kv_count = bitnet_kv_element_count(ctx);
    size_t scale_count = bitnet_kv_scale_count(ctx);

    if (ctx == NULL) {
        return 0;
    }

    if (ctx->kv_cache_type == BITNET_KV_CACHE_Q8) {
        return (unsigned long long)(2u * kv_count * sizeof(int8_t) +
                                    2u * scale_count * sizeof(float));
    }
    return (unsigned long long)(2u * kv_count * sizeof(float));
}

int bitnet_decode_token(bitnet_model_t *model, int token, char *out, int out_size) {
    if (model == NULL || model->tokenizer == NULL) {
        return -1;
    }
    return bitnet_tokenizer_decode(model->tokenizer, token, out, out_size);
}

int bitnet_eos_token(bitnet_model_t *model) {
    if (model == NULL || model->tokenizer == NULL) {
        return -1;
    }
    return bitnet_tokenizer_eos_id(model->tokenizer);
}

int bitnet_pad_token(bitnet_model_t *model) {
    if (model == NULL || model->tokenizer == NULL) {
        return -1;
    }
    return bitnet_tokenizer_pad_id(model->tokenizer);
}

int bitnet_token_is_eog(bitnet_model_t *model, int token) {
    int eos = bitnet_eos_token(model);
    int pad = bitnet_pad_token(model);
    return (eos >= 0 && token == eos) || (pad >= 0 && token == pad);
}

static int bitnet_recent_contains_token(const int *tokens, int n_tokens, int token) {
    if (tokens == NULL || n_tokens <= 0 || token < 0) {
        return 0;
    }
    for (int i = 0; i < n_tokens; ++i) {
        if (tokens[i] == token) {
            return 1;
        }
    }
    return 0;
}

void bitnet_apply_repetition_penalty(float *logits,
                                     int vocab_size,
                                     const int *tokens,
                                     int n_tokens,
                                     float penalty) {
    if (logits == NULL || vocab_size <= 0 || tokens == NULL || n_tokens <= 0 || penalty <= 1.0f) {
        return;
    }

    for (int i = 0; i < n_tokens; ++i) {
        int token = tokens[i];
        if (token < 0 || token >= vocab_size) {
            continue;
        }
        if (bitnet_recent_contains_token(tokens, i, token)) {
            continue;
        }
        if (logits[token] >= 0.0f) {
            logits[token] /= penalty;
        } else {
            logits[token] *= penalty;
        }
    }
}

/* ---- RoPE ---- */

static void BITNET_MAYBE_UNUSED rope_apply_cached(float *x, int n_heads, int head_dim, int rope_dim,
                                                  const float *rope_cos, const float *rope_sin) {
    if (x == NULL || rope_cos == NULL || rope_sin == NULL ||
        n_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
        return;
    }

    for (int h = 0; h < n_heads; ++h) {
        for (int j = 0; j < rope_dim / 2; ++j) {
            int idx0 = h * head_dim + 2 * j;
            int idx1 = h * head_dim + 2 * j + 1;
            float c = rope_cos[j];
            float s = rope_sin[j];
            float x0 = x[idx0];
            float x1 = x[idx1];
            x[idx0] = x0 * c - x1 * s;
            x[idx1] = x0 * s + x1 * c;
        }
    }
}

/* ---- bitnet_eval: full 28-block llama transformer forward pass (mmap) ---- */

int bitnet_eval(bitnet_context_t *ctx, const int *tokens, int n_tokens) {
    bitnet_model_t *model = NULL;
    const bitnet_tensor_cache_t *cache = NULL;
    float *hidden = NULL;
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *attn_buffer = NULL;
    float *gate = NULL;
    float *up = NULL;
    float *down = NULL;
    float *scores = NULL;
    float *tmp_out = NULL;
    float *tq2_lut BITNET_MAYBE_UNUSED = NULL;
    int8_t *tq2_qhidden BITNET_MAYBE_UNUSED = NULL;
    int8_t *tq2_qffn BITNET_MAYBE_UNUSED = NULL;
    int error_ret = -1;
    int emb_dim = 0;
    int q_dim = 0;
    int head_dim = 0;
    int kv_dim = 0;
    int attn_dim = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int ffn_dim = 0;
    int vocab_size = 0;
    int rope_dim = 0;
    int token_idx = 0;
    int block_idx = 0;
    int profile_eval = 0;
    double profile_start = 0.0;
    double profile_output_start = 0.0;
    double profile_output_end = 0.0;
    double profile_qkv_sec = 0.0;
    double profile_attn_sec = 0.0;
    double profile_attn_out_sec = 0.0;
    double profile_gate_up_sec = 0.0;
    double profile_ffn_down_sec = 0.0;
    double profile_lut_build_sec = 0.0;
    float tq2_hidden_scale BITNET_MAYBE_UNUSED = 0.0f;
    float tq2_ffn_scale BITNET_MAYBE_UNUSED = 0.0f;
    float tq2_ffn_max_abs BITNET_MAYBE_UNUSED = 0.0f;

    if (ctx == NULL || ctx->model == NULL || tokens == NULL || n_tokens <= 0) {
        return -1;
    }
    if (ctx->n_pos < 0 || n_tokens > ctx->max_tokens - ctx->n_pos) {
        return -2;
    }
    if ((ctx->kv_cache_type == BITNET_KV_CACHE_F32 &&
         (ctx->key_cache == NULL || ctx->value_cache == NULL)) ||
        (ctx->kv_cache_type == BITNET_KV_CACHE_Q8 &&
         (ctx->key_cache_q8 == NULL || ctx->value_cache_q8 == NULL ||
          ctx->key_cache_scales == NULL || ctx->value_cache_scales == NULL ||
          ctx->attn_q8 == NULL || ctx->attn_q8_scales == NULL))) {
        return -1;
    }

    profile_eval = getenv("BITNET_PROFILE_EVAL") != NULL;
    if (profile_eval) {
        profile_start = monotonic_seconds();
    }

    model = ctx->model;
    cache = &model->tensor_cache;

    emb_dim = (int)model->embedding_length;
    n_heads = (int)model->head_count;
    n_kv_heads = (int)model->head_count_kv;
    ffn_dim = (int)model->feed_forward_length;
    vocab_size = (int)model->vocab_size;
    rope_dim = (int)model->rope_dimension_count;
    head_dim = (int)model->attention_key_length;
    q_dim = (int)bitnet_attention_dim_for_model(model);
    kv_dim = (int)bitnet_kv_dim_for_model(model);
    attn_dim = q_dim;

    /* use pre-allocated buffers */
    hidden = ctx->hidden;
    q = ctx->q;
    k = ctx->k;
    v = ctx->v;
    attn_buffer = ctx->attn_buffer;
    gate = ctx->gate;
    up = ctx->up;
    down = ctx->down;
    scores = ctx->scores;
    tmp_out = ctx->tmp_out;
    tq2_lut = ctx->tq2_lut;
    tq2_qhidden = ctx->tq2_qhidden;
    tq2_qffn = ctx->tq2_qffn;
    /* Per-block bsums arrays: 8B FFN uses 16384/256 = 64 blocks. */
    int32_t tq2_hidden_block_bsums[BITNET_MAX_TQ2_BLOCKS] BITNET_MAYBE_UNUSED;
    int32_t tq2_ffn_block_bsums[BITNET_MAX_TQ2_BLOCKS] BITNET_MAYBE_UNUSED;

    /* ---- process tokens sequentially ---- */
    for (token_idx = 0; token_idx < n_tokens; ++token_idx) {
        int token = tokens[token_idx];

        /* --- embedding lookup via mmap --- */
        {
            const uint8_t *emb_base = (const uint8_t *)gguf_get_tensor_ptr(&model->gguf, cache->token_embd);

            if (emb_base == NULL) goto cleanup;
            if (model->token_embd_type == BITNET_TARGET_TENSOR_TYPE_F16) {
                if (bitnet_f16_embedding_lookup(emb_base, token,
                                                emb_dim, vocab_size,
                                                hidden) != 0) {
                    goto cleanup;
                }
            } else {
                size_t blocks_per_row = (size_t)emb_dim / BITNET_Q4K_QK;
                size_t emb_row_size = blocks_per_row * BITNET_Q4K_BLOCK_SIZE;
                const uint8_t *emb_row = emb_base + (size_t)token * emb_row_size;

                if (bitnet_q4k_embedding_lookup(emb_row, 0,
                                                emb_dim, vocab_size,
                                                hidden, (size_t)emb_dim * sizeof(float)) != 0) {
                    goto cleanup;
                }
            }
            if (model->is_minicpm) {
                bitnet_scale_vector(hidden, model->embedding_scale, emb_dim);
            }
        }

        /* --- run through all 28 blocks --- */
        for (block_idx = 0; block_idx < (int)model->block_count; ++block_idx) {
            const bitnet_block_tensors_t *bt = &cache->blocks[block_idx];
#if !BITNET_USE_TQ2_I2S
            const bitnet_block_scale_cache_t *bs = &model->scale_cache.blocks[block_idx];
#else
            const bitnet_block_i2s_weights_t *bw = &model->i2s_cache.blocks[block_idx];
#endif

            /* ===== Block step 1: attn RMSNorm ===== */
            {
                const float *norm_w = (const float *)gguf_get_tensor_ptr(&model->gguf, bt->attn_norm);
                if (norm_w == NULL) goto cleanup;
                /* rms_norm into tmp_out, then swap pointers:
                 * tmp_out = norm(hidden * norm_w), hidden unchanged (residual) */
#if BITNET_USE_TQ2_I2S
                if (bitnet_rms_norm_quant_tq2_i8(tmp_out, hidden, norm_w, emb_dim,
                                                  tq2_qhidden, &tq2_hidden_scale,
                                                  NULL) != 0) {
                    goto cleanup;
                }
#else
                bitnet_rms_norm_inplace(tmp_out, hidden, norm_w, emb_dim);
#endif
                { float *swap_tmp = hidden; hidden = tmp_out; tmp_out = swap_tmp; }
            }

            /* ===== Block step 2a/2b/2c: q/k/v projection ===== */
            double profile_step_start = profile_eval ? monotonic_seconds() : 0.0;
#if BITNET_USE_TQ2_I2S
            /* Quantized together with attn RMSNorm above. */
#elif BITNET_USE_TQ2_TL1_LUT
            {
                float tl1_vec_scale = 0.0f;
                const bitnet_block_tl1_weights_t *tw = &model->tl1_cache.blocks[block_idx];
                if (bitnet_tq2_0_tl1_preprocessor(hidden, emb_dim,
                                                    &tl1_vec_scale,
                                                    ctx->tq2_tl1_lut, ctx->tq2_tl1_lut_size) != 0) {
                    goto cleanup;
                }
                if (bitnet_tq2_0_tl1_qgemm(tw->attn_q, bs->attn_q,
                                              tq2_hidden_block_bsums,
                                              q_dim, emb_dim,
                                              ctx->tq2_tl1_lut, tl1_vec_scale, q) != 0) {
                    goto cleanup;
                }
                if (bitnet_tq2_0_tl1_qgemm(tw->attn_k, bs->attn_k,
                                              tq2_hidden_block_bsums,
                                              kv_dim, emb_dim,
                                              ctx->tq2_tl1_lut, tl1_vec_scale, k) != 0) {
                    goto cleanup;
                }
                if (bitnet_tq2_0_tl1_qgemm(tw->attn_v, bs->attn_v,
                                              tq2_hidden_block_bsums,
                                              kv_dim, emb_dim,
                                              ctx->tq2_tl1_lut, tl1_vec_scale, v) != 0) {
                    goto cleanup;
                }
            }
#elif BITNET_USE_TQ2_NEON_ATTN
            if (bitnet_tq2_0_quantize_vec_i8(hidden, emb_dim,
                                             tq2_qhidden, &tq2_hidden_scale, tq2_hidden_block_bsums) != 0) {
                goto cleanup;
            }
#else
            double profile_lut_start = profile_eval ? profile_step_start : 0.0;
            if (bitnet_tq2_0_build_vec_lut(hidden, emb_dim,
                                           tq2_lut, ctx->tq2_lut_count) != 0) {
                goto cleanup;
            }
            if (profile_eval) {
                profile_lut_build_sec += monotonic_seconds() - profile_lut_start;
            }
#endif
            {
#if BITNET_USE_TQ2_I2S
                if (bitnet_tq2_0_matmul_i2s_qkv_parallel(bw->attn_q,
                                                   model->i2s_cache.scales[block_idx * 7 + 0],
                                                   bw->attn_k,
                                                   model->i2s_cache.scales[block_idx * 7 + 1],
                                                   bw->attn_v,
                                                   model->i2s_cache.scales[block_idx * 7 + 2],
                                                   NULL,
                                                   q_dim, kv_dim, emb_dim,
                                                   tq2_qhidden, tq2_hidden_scale,
                                                   q, k, v) != 0) {
                    goto cleanup;
                }
#else
                const void *w = gguf_get_tensor_ptr(&model->gguf, bt->attn_q);
                if (w == NULL) goto cleanup;
#if BITNET_USE_TQ2_NEON_ATTN
                if (bitnet_tq2_0_matmul_vector_i8_neon_scales(w, bs->attn_q,
                                                               q_dim, emb_dim,
                                                               tq2_qhidden, tq2_hidden_scale,
                                                               tq2_hidden_block_bsums, q) != 0) {
                    goto cleanup;
                }
#else
                (void)bitnet_tq2_0_matmul_vector_lut_scales(w, bs->attn_q,
                                                             q_dim, emb_dim,
                                                             tq2_lut, q);
#endif
#endif /* BITNET_USE_TQ2_I2S */
            }

            /* ===== Block step 2b: k projection ===== */
            {
#if BITNET_USE_TQ2_I2S
                /* Already computed together with Q above. */
#else
                const void *w = gguf_get_tensor_ptr(&model->gguf, bt->attn_k);
                if (w == NULL) goto cleanup;
#if BITNET_USE_TQ2_NEON_ATTN
                if (bitnet_tq2_0_matmul_vector_i8_neon_scales(w, bs->attn_k,
                                                               kv_dim, emb_dim,
                                                               tq2_qhidden, tq2_hidden_scale,
                                                               tq2_hidden_block_bsums, k) != 0) {
                    goto cleanup;
                }
#else
                (void)bitnet_tq2_0_matmul_vector_lut_scales(w, bs->attn_k,
                                                             kv_dim, emb_dim,
                                                             tq2_lut, k);
#endif
#endif /* BITNET_USE_TQ2_I2S */
            }

            /* ===== Block step 2c: v projection ===== */
            {
#if BITNET_USE_TQ2_I2S
                /* Already computed together with Q above. */
#else
                const void *w = gguf_get_tensor_ptr(&model->gguf, bt->attn_v);
                if (w == NULL) goto cleanup;
#if BITNET_USE_TQ2_NEON_ATTN
                if (bitnet_tq2_0_matmul_vector_i8_neon_scales(w, bs->attn_v,
                                                               kv_dim, emb_dim,
                                                               tq2_qhidden, tq2_hidden_scale,
                                                               tq2_hidden_block_bsums, v) != 0) {
                    goto cleanup;
                }
#else
                (void)bitnet_tq2_0_matmul_vector_lut_scales(w, bs->attn_v,
                                                             kv_dim, emb_dim,
                                                             tq2_lut, v);
#endif
#endif /* BITNET_USE_TQ2_I2S */
            }
            if (bitnet_apply_lora(model, block_idx, BITNET_LORA_LAYER_ATTN_Q, hidden, q) != 0 ||
                bitnet_apply_lora(model, block_idx, BITNET_LORA_LAYER_ATTN_K, hidden, k) != 0 ||
                bitnet_apply_lora(model, block_idx, BITNET_LORA_LAYER_ATTN_V, hidden, v) != 0) {
                goto cleanup;
            }
            if (profile_eval) {
                profile_qkv_sec += monotonic_seconds() - profile_step_start;
            }

            /* ===== Block step 2d: RoPE on q and k ===== */
            {
                size_t rope_offset = (size_t)ctx->n_pos * (size_t)(rope_dim / 2);
                bitnet_rope_apply(q, n_heads, head_dim, rope_dim,
                                  ctx->rope_cos + rope_offset,
                                  ctx->rope_sin + rope_offset);
                bitnet_rope_apply(k, n_kv_heads, head_dim, rope_dim,
                                  ctx->rope_cos + rope_offset,
                                  ctx->rope_sin + rope_offset);
            }

            /* ===== Block step 2e: KV cache store ===== */
            {
                size_t cache_offset = ((size_t)block_idx * (size_t)ctx->max_tokens + (size_t)ctx->n_pos) * (size_t)kv_dim;
                if (ctx->kv_cache_type == BITNET_KV_CACHE_Q8) {
                    size_t scale_offset = ((size_t)block_idx * (size_t)ctx->max_tokens + (size_t)ctx->n_pos) *
                                          (size_t)n_kv_heads;
                    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
                        size_t head_offset = (size_t)kv_h * (size_t)head_dim;
                        ctx->key_cache_scales[scale_offset + (size_t)kv_h] =
                            bitnet_quantize_f32_to_i8(k + head_offset, head_dim,
                                                       ctx->key_cache_q8 + cache_offset + head_offset);
                        ctx->value_cache_scales[scale_offset + (size_t)kv_h] =
                            bitnet_quantize_f32_to_i8(v + head_offset, head_dim,
                                                       ctx->value_cache_q8 + cache_offset + head_offset);
                    }
                } else {
                    memcpy(ctx->key_cache + cache_offset, k, kv_dim * sizeof(float));
                    memcpy(ctx->value_cache + cache_offset, v, kv_dim * sizeof(float));
                }
            }

            /* ===== Block step 2f: attention (GQA: reuse scores for shared kv heads) ===== */
            profile_step_start = profile_eval ? monotonic_seconds() : 0.0;
            {
                int n_positions = ctx->n_pos + 1;
                float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);
                int gqa_ratio = n_heads / n_kv_heads;

                if (ctx->kv_cache_type == BITNET_KV_CACHE_Q8) {
                    for (int query_head = 0; query_head < n_heads; ++query_head) {
                        ctx->attn_q8_scales[query_head] =
                            bitnet_quantize_f32_to_i8(q + (size_t)query_head * (size_t)head_dim,
                                                       head_dim,
                                                       ctx->attn_q8 + (size_t)query_head * (size_t)head_dim);
                    }

                    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
                        for (int past_pos = 0; past_pos < n_positions; ++past_pos) {
                            size_t pos_base = (size_t)block_idx * (size_t)ctx->max_tokens + (size_t)past_pos;
                            size_t k_offset = pos_base * (size_t)kv_dim +
                                              (size_t)kv_h * (size_t)head_dim;
                            float k_scale = ctx->key_cache_scales[pos_base * (size_t)n_kv_heads + (size_t)kv_h];

                            for (int q_in_group = 0; q_in_group < gqa_ratio; ++q_in_group) {
                                int query_head = kv_h * gqa_ratio + q_in_group;
                                const int8_t *q_head = ctx->attn_q8 + (size_t)query_head * (size_t)head_dim;
                                float q_scale = ctx->attn_q8_scales[query_head];
                                int dot_i32 = bitnet_dot_i8(q_head, ctx->key_cache_q8 + k_offset, head_dim);
                                scores[query_head * n_positions + past_pos] =
                                    (float)dot_i32 * q_scale * k_scale * inv_sqrt_head_dim;
                            }
                        }
                    }
                } else {
                    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
                        for (int past_pos = 0; past_pos < n_positions; ++past_pos) {
                            size_t k_offset = ((size_t)block_idx * (size_t)ctx->max_tokens + (size_t)past_pos) * (size_t)kv_dim +
                                              (size_t)kv_h * (size_t)head_dim;

                            for (int q_in_group = 0; q_in_group < gqa_ratio; ++q_in_group) {
                                int query_head = kv_h * gqa_ratio + q_in_group;
                                float *q_head = q + query_head * head_dim;
                                float dot = 0.0f;
                                int d = 0;
#if defined(__ARM_NEON)
                                float32x4_t acc = vdupq_n_f32(0.0f);
                                for (d = 0; d + 3 < head_dim; d += 4) {
                                    float32x4_t qv = vld1q_f32(q_head + d);
                                    float32x4_t kv = vld1q_f32(ctx->key_cache + k_offset + d);
                                    acc = vfmaq_f32(acc, qv, kv);
                                }
                                dot = vaddvq_f32(acc);
#endif
                                for (; d < head_dim; ++d) {
                                    dot += q_head[d] * ctx->key_cache[k_offset + d];
                                }
                                scores[query_head * n_positions + past_pos] = dot * inv_sqrt_head_dim;
                            }
                        }
                    }
                }

                /* Softmax per query head + weighted sum of values */
                for (int query_head = 0; query_head < n_heads; ++query_head) {
                    int kv_h = query_head / gqa_ratio;
                    float *head_scores = scores + query_head * n_positions;
                    bitnet_softmax(head_scores, n_positions);

                    float *attn_head = attn_buffer + query_head * head_dim;
                    memset(attn_head, 0, (size_t)head_dim * sizeof(float));
                    for (int past_pos = 0; past_pos < n_positions; ++past_pos) {
                        float w = head_scores[past_pos];
                        size_t v_offset = ((size_t)block_idx * (size_t)ctx->max_tokens + (size_t)past_pos) * (size_t)kv_dim +
                                          (size_t)kv_h * (size_t)head_dim;
                        int d = 0;
                        if (ctx->kv_cache_type == BITNET_KV_CACHE_Q8) {
                            size_t pos_base = (size_t)block_idx * (size_t)ctx->max_tokens + (size_t)past_pos;
                            float ws = w * ctx->value_cache_scales[pos_base * (size_t)n_kv_heads + (size_t)kv_h];
                            const int8_t *v_q8 = ctx->value_cache_q8 + v_offset;
                            bitnet_accum_i8_scaled(attn_head, v_q8, ws, head_dim);
                            continue;
                        }
#if defined(__ARM_NEON)
                        float32x4_t wv = vdupq_n_f32(w);
                        for (d = 0; d + 3 < head_dim; d += 4) {
                            float32x4_t vv = vld1q_f32(ctx->value_cache + v_offset + d);
                            float32x4_t av = vld1q_f32(attn_head + d);
                            vst1q_f32(attn_head + d, vfmaq_f32(av, wv, vv));
                        }
#endif
                        for (; d < head_dim; ++d) {
                            attn_head[d] += w * ctx->value_cache[v_offset + d];
                        }
                    }
                }
            }
            if (profile_eval) {
                profile_attn_sec += monotonic_seconds() - profile_step_start;
            }

            /* ===== Block step 3: attn_output projection ===== */
            profile_step_start = profile_eval ? monotonic_seconds() : 0.0;
#if BITNET_USE_TQ2_I2S
            if (bitnet_tq2_0_quantize_vec_i8(attn_buffer, attn_dim,
                                             tq2_qhidden, &tq2_hidden_scale, NULL) != 0) {
                goto cleanup;
            }
#elif BITNET_USE_TQ2_TL1_LUT
            {
                float tl1_vec_scale = 0.0f;
                const bitnet_block_tl1_weights_t *tw = &model->tl1_cache.blocks[block_idx];
                if (bitnet_tq2_0_tl1_preprocessor(attn_buffer, attn_dim,
                                                    &tl1_vec_scale,
                                                    ctx->tq2_tl1_lut, ctx->tq2_tl1_lut_size) != 0) {
                    goto cleanup;
                }
                if (bitnet_tq2_0_tl1_qgemm(tw->attn_output, bs->attn_output,
                                              tq2_hidden_block_bsums,
                                              emb_dim, attn_dim,
                                              ctx->tq2_tl1_lut, tl1_vec_scale, down) != 0) {
                    goto cleanup;
                }
            }
#elif BITNET_USE_TQ2_NEON_ATTN
            if (bitnet_tq2_0_quantize_vec_i8(attn_buffer, attn_dim,
                                             tq2_qhidden, &tq2_hidden_scale, tq2_hidden_block_bsums) != 0) {
                goto cleanup;
            }
#else
            profile_lut_start = profile_eval ? profile_step_start : 0.0;
            if (bitnet_tq2_0_build_vec_lut(attn_buffer, attn_dim,
                                           tq2_lut, ctx->tq2_lut_count) != 0) {
                goto cleanup;
            }
            if (profile_eval) {
                profile_lut_build_sec += monotonic_seconds() - profile_lut_start;
            }
#endif
            {
#if BITNET_USE_TQ2_I2S
                if (bitnet_tq2_0_matmul_i2s_neon_parallel(bw->attn_output,
                                                   model->i2s_cache.scales[block_idx * 7 + 3],
                                                   NULL,
                                                   emb_dim, attn_dim,
                                                   tq2_qhidden, tq2_hidden_scale, down) != 0) {
                    goto cleanup;
                }
#else
                const void *w = gguf_get_tensor_ptr(&model->gguf, bt->attn_output);
                if (w == NULL) goto cleanup;
                /* Use 'down' as temp output buffer (not yet used in this block) */
#if BITNET_USE_TQ2_NEON_ATTN
                if (bitnet_tq2_0_matmul_vector_i8_neon_scales(w, bs->attn_output,
                                                               emb_dim, attn_dim,
                                                               tq2_qhidden, tq2_hidden_scale,
                                                               tq2_hidden_block_bsums, down) != 0) {
                    goto cleanup;
                }
#else
                (void)bitnet_tq2_0_matmul_vector_lut_scales(w, bs->attn_output,
                                                             emb_dim, attn_dim,
                                                             tq2_lut, down);
#endif
#endif /* BITNET_USE_TQ2_I2S */
                if (bitnet_apply_lora(model, block_idx, BITNET_LORA_LAYER_ATTN_OUTPUT,
                                      attn_buffer, down) != 0) {
                    goto cleanup;
                }
                /* residual add: use saved residual (in tmp_out) + attention output */
                bitnet_residual_add_scaled(hidden, tmp_out, down,
                                           model->is_minicpm ? model->residual_scale : 1.0f,
                                           emb_dim);
            }
            if (profile_eval) {
                profile_attn_out_sec += monotonic_seconds() - profile_step_start;
            }

            /* ===== Block step 4: ffn RMSNorm ===== */
            {
                const float *norm_w = (const float *)gguf_get_tensor_ptr(&model->gguf, bt->ffn_norm);
                if (norm_w == NULL) goto cleanup;
                /* rms_norm into tmp_out, then swap pointers:
                 * tmp_out = norm(hidden * norm_w), hidden unchanged (residual) */
#if BITNET_USE_TQ2_I2S
                if (bitnet_rms_norm_quant_tq2_i8(tmp_out, hidden, norm_w, emb_dim,
                                                  tq2_qhidden, &tq2_hidden_scale,
                                                  NULL) != 0) {
                    goto cleanup;
                }
#else
                bitnet_rms_norm_inplace(tmp_out, hidden, norm_w, emb_dim);
#endif
                { float *swap_tmp = hidden; hidden = tmp_out; tmp_out = swap_tmp; }
            }

            /* ===== Block step 5a: ffn_gate projection ===== */
            profile_step_start = profile_eval ? monotonic_seconds() : 0.0;
            {
#if BITNET_USE_TQ2_I2S
                /* Quantized together with FFN RMSNorm above. */
                if (ffn_dim >= 8192) {
                    if (bitnet_tq2_0_matmul_i2s_neon_parallel(bw->ffn_gate,
                                                       model->i2s_cache.scales[block_idx * 7 + 4],
                                                       NULL,
                                                       ffn_dim, emb_dim,
                                                       tq2_qhidden, tq2_hidden_scale,
                                                       gate) != 0) {
                        goto cleanup;
                    }
                    if (bitnet_tq2_0_matmul_i2s_neon_parallel(bw->ffn_up,
                                                       model->i2s_cache.scales[block_idx * 7 + 5],
                                                       NULL,
                                                       ffn_dim, emb_dim,
                                                       tq2_qhidden, tq2_hidden_scale,
                                                       up) != 0) {
                        goto cleanup;
                    }
                } else {
                    if (bitnet_tq2_0_matmul_i2s_neon_pair_parallel(bw->ffn_gate,
                                                            model->i2s_cache.scales[block_idx * 7 + 4],
                                                            bw->ffn_up,
                                                            model->i2s_cache.scales[block_idx * 7 + 5],
                                                            NULL,
                                                            ffn_dim, emb_dim,
                                                            tq2_qhidden, tq2_hidden_scale,
                                                            gate, up) != 0) {
                        goto cleanup;
                    }
                }
#elif BITNET_USE_TQ2_TL1_LUT
                {
                    float tl1_vec_scale = 0.0f;
                    const bitnet_block_tl1_weights_t *tw = &model->tl1_cache.blocks[block_idx];
                    if (bitnet_tq2_0_tl1_preprocessor(hidden, emb_dim,
                                                        &tl1_vec_scale,
                                                        ctx->tq2_tl1_lut, ctx->tq2_tl1_lut_size) != 0) {
                        goto cleanup;
                    }
                    if (bitnet_tq2_0_tl1_qgemm_pair(tw->ffn_gate, bs->ffn_gate,
                                                       tw->ffn_up, bs->ffn_up,
                                                       tq2_hidden_block_bsums,
                                                       ffn_dim, emb_dim,
                                                       ctx->tq2_tl1_lut, tl1_vec_scale,
                                                       gate, up) != 0) {
                        goto cleanup;
                    }
                }
#elif BITNET_USE_TQ2_NEON_FFN
                {
                    const void *w_gate = gguf_get_tensor_ptr(&model->gguf, bt->ffn_gate);
                    const void *w_up = gguf_get_tensor_ptr(&model->gguf, bt->ffn_up);
                    if (w_gate == NULL || w_up == NULL) goto cleanup;
                    if (bitnet_tq2_0_quantize_vec_i8(hidden, emb_dim,
                                                     tq2_qhidden, &tq2_hidden_scale, tq2_hidden_block_bsums) != 0) {
                        goto cleanup;
                    }
                    if (bitnet_tq2_0_matmul_vector_i8_neon_pair_scales(w_gate, bs->ffn_gate,
                                                                        w_up, bs->ffn_up,
                                                                        ffn_dim, emb_dim,
                                                                        tq2_qhidden, tq2_hidden_scale,
                                                                        tq2_hidden_block_bsums, gate, up) != 0) {
                        goto cleanup;
                    }
                }
#else
                {
                    const void *w_gate = gguf_get_tensor_ptr(&model->gguf, bt->ffn_gate);
                    const void *w_up = gguf_get_tensor_ptr(&model->gguf, bt->ffn_up);
                    if (w_gate == NULL || w_up == NULL) goto cleanup;
                    profile_lut_start = profile_eval ? profile_step_start : 0.0;
                    if (bitnet_tq2_0_build_vec_lut(hidden, emb_dim,
                                                   tq2_lut, ctx->tq2_lut_count) != 0) {
                        goto cleanup;
                    }
                    if (profile_eval) {
                        profile_lut_build_sec += monotonic_seconds() - profile_lut_start;
                    }
                    (void)bitnet_tq2_0_matmul_vector_lut_pair_scales(w_gate, bs->ffn_gate,
                                                                      w_up, bs->ffn_up,
                                                                      ffn_dim, emb_dim,
                                                                      tq2_lut, gate, up);
                }
#endif
            }
            if (bitnet_apply_lora(model, block_idx, BITNET_LORA_LAYER_FFN_GATE, hidden, gate) != 0 ||
                bitnet_apply_lora(model, block_idx, BITNET_LORA_LAYER_FFN_UP, hidden, up) != 0) {
                goto cleanup;
            }
            if (profile_eval) {
                profile_gate_up_sec += monotonic_seconds() - profile_step_start;
            }

            /* ===== Block step 5c: SiLU(gate) and element-wise merge ===== */
            tq2_ffn_max_abs = bitnet_silu_mul_max_abs(gate, up, ffn_dim);

            /* ===== Block step 5d: ffn_down projection ===== */
            profile_step_start = profile_eval ? monotonic_seconds() : 0.0;
            {
#if BITNET_USE_TQ2_I2S
                if (bitnet_tq2_0_quantize_vec_i8_known_max(gate, ffn_dim,
                                                           tq2_qffn, &tq2_ffn_scale,
                                                           NULL,
                                                           tq2_ffn_max_abs) != 0) {
                    goto cleanup;
                }
                if (bitnet_tq2_0_matmul_i2s_neon_parallel(bw->ffn_down,
                                                   model->i2s_cache.scales[block_idx * 7 + 6],
                                                   NULL,
                                                   emb_dim, ffn_dim,
                                                   tq2_qffn, tq2_ffn_scale, down) != 0) {
                    goto cleanup;
                }
#elif BITNET_USE_TQ2_TL1_LUT
                {
                    float tl1_vec_scale = 0.0f;
                    const bitnet_block_tl1_weights_t *tw = &model->tl1_cache.blocks[block_idx];
                    if (bitnet_tq2_0_tl1_preprocessor(gate, ffn_dim,
                                                        &tl1_vec_scale,
                                                        ctx->tq2_tl1_lut, ctx->tq2_tl1_lut_size) != 0) {
                        goto cleanup;
                    }
                    if (bitnet_tq2_0_tl1_qgemm(tw->ffn_down, bs->ffn_down,
                                                  tq2_ffn_block_bsums,
                                                  emb_dim, ffn_dim,
                                                  ctx->tq2_tl1_lut, tl1_vec_scale, down) != 0) {
                        goto cleanup;
                    }
                }
#elif BITNET_USE_TQ2_NEON_FFN
                {
                    const void *w = gguf_get_tensor_ptr(&model->gguf, bt->ffn_down);
                    if (w == NULL) goto cleanup;
                    if (bitnet_tq2_0_quantize_vec_i8(gate, ffn_dim,
                                                     tq2_qffn, &tq2_ffn_scale, tq2_ffn_block_bsums) != 0) {
                        goto cleanup;
                    }
                    if (bitnet_tq2_0_matmul_vector_i8_neon_scales(w, bs->ffn_down,
                                                                   emb_dim, ffn_dim,
                                                                   tq2_qffn, tq2_ffn_scale,
                                                                   tq2_ffn_block_bsums, down) != 0) {
                        goto cleanup;
                    }
                }
#else
                {
                    const void *w = gguf_get_tensor_ptr(&model->gguf, bt->ffn_down);
                    if (w == NULL) goto cleanup;
                    profile_lut_start = profile_eval ? profile_step_start : 0.0;
                    if (bitnet_tq2_0_build_vec_lut(gate, ffn_dim,
                                                   tq2_lut, ctx->tq2_lut_count) != 0) {
                        goto cleanup;
                    }
                    if (profile_eval) {
                        profile_lut_build_sec += monotonic_seconds() - profile_lut_start;
                    }
                    (void)bitnet_tq2_0_matmul_vector_lut_scales(w, bs->ffn_down,
                                                                 emb_dim, ffn_dim,
                                                                 tq2_lut, down);
                }
#endif
            }
            if (bitnet_apply_lora(model, block_idx, BITNET_LORA_LAYER_FFN_DOWN, gate, down) != 0) {
                goto cleanup;
            }
            if (profile_eval) {
                profile_ffn_down_sec += monotonic_seconds() - profile_step_start;
            }

            /* ===== Block step 5e: residual add ===== */
            bitnet_residual_add_scaled(hidden, tmp_out, down,
                                       model->is_minicpm ? model->residual_scale : 1.0f,
                                       emb_dim);

        }

        /* advance position for next token */
        ++ctx->n_pos;
    }

    /* ---- final output_norm RMSNorm ---- */
    {
        const float *norm_w = (const float *)gguf_get_tensor_ptr(&model->gguf, cache->output_norm);
        if (norm_w == NULL) goto cleanup;
        bitnet_rms_norm(hidden, norm_w, emb_dim);
        if (ctx->last_hidden != NULL) {
            memcpy(ctx->last_hidden, hidden, (size_t)emb_dim * sizeof(*ctx->last_hidden));
        }
    }

    /* ---- output projection via Q6_K (mmap) ---- */
    if (profile_eval) {
        profile_output_start = monotonic_seconds();
    }
#if BITNET_USE_TQ2_I2S
    bitnet_tq2_0_park_workers();
#endif
    {
        size_t output_blocks_per_col = (size_t)emb_dim / BITNET_Q6K_QK;
        size_t output_col_size = output_blocks_per_col * BITNET_Q6K_BLOCK_SIZE;
        const uint8_t *output_data = cache->output != NULL ?
            (const uint8_t *)gguf_get_tensor_ptr(&model->gguf, cache->output) : NULL;

#if BITNET_USE_Q6K_NEON_OUTPUT
        if (bitnet_tq2_0_quantize_vec_i8(hidden, emb_dim,
                                         tq2_qhidden, &tq2_hidden_scale, NULL) != 0) {
            goto cleanup;
        }
        if (model->is_minicpm && model->logit_scale != 0.0f) {
            tq2_hidden_scale /= model->logit_scale;
        }
        if (model->output_q8_blockscale &&
            model->output_q8 != NULL &&
            model->output_q8_d != NULL) {
            if (bitnet_output_projection_q8_blockscale_neon_parallel(model->output_q8,
                                                                      model->output_q8_d,
                                                                      (int)output_blocks_per_col,
                                                                      vocab_size, tq2_qhidden,
                                                                      tq2_hidden_scale, ctx->logits) != 0) {
                goto cleanup;
            }
        } else if (model->output_q8 != NULL &&
            model->output_q8_scales_i8 != NULL &&
            model->output_q8_d != NULL) {
            if (bitnet_output_projection_q8_compact_neon_parallel(model->output_q8,
                                                                   model->output_q8_scales_i8,
                                                                   model->output_q8_d,
                                                                   (int)output_blocks_per_col,
                                                                   vocab_size, tq2_qhidden,
                                                                   tq2_hidden_scale, ctx->logits) != 0) {
                goto cleanup;
            }
        } else if (model->output_q8 != NULL && model->output_q8_scales != NULL) {
            if (bitnet_output_projection_q8_neon_parallel(model->output_q8,
                                                          model->output_q8_scales,
                                                          (int)output_blocks_per_col,
                                                          vocab_size, tq2_qhidden,
                                                          tq2_hidden_scale, ctx->logits) != 0) {
                goto cleanup;
            }
        } else {
            if (output_data == NULL) goto cleanup;
            if (bitnet_output_projection_i8_neon_parallel(output_data, output_col_size,
                                                          (int)output_blocks_per_col,
                                                          vocab_size, tq2_qhidden,
                                                          tq2_hidden_scale, ctx->logits) != 0) {
                goto cleanup;
            }
        }
#else
        if (output_data == NULL) goto cleanup;
        if (bitnet_output_projection_parallel(output_data, output_col_size,
                                              (int)output_blocks_per_col,
                                              vocab_size, hidden, ctx->logits) != 0) {
            goto cleanup;
        }
#endif
    }
    if (bitnet_apply_lora(model, -1, BITNET_LORA_LAYER_OUTPUT, hidden, ctx->logits) != 0) {
        goto cleanup;
    }
    if (profile_eval) {
        profile_output_end = monotonic_seconds();
        fprintf(stderr,
                "bitnet_profile eval_tokens=%d transformer_sec=%.6f output_sec=%.6f total_sec=%.6f qkv_sec=%.6f attn_sec=%.6f attn_out_sec=%.6f gate_up_sec=%.6f ffn_down_sec=%.6f lut_build_sec=%.6f other_sec=%.6f\n",
                n_tokens,
                profile_output_start - profile_start,
                profile_output_end - profile_output_start,
                profile_output_end - profile_start,
                profile_qkv_sec,
                profile_attn_sec,
                profile_attn_out_sec,
                profile_gate_up_sec,
                profile_ffn_down_sec,
                profile_lut_build_sec,
                (profile_output_start - profile_start) -
                    (profile_qkv_sec + profile_attn_sec + profile_attn_out_sec +
                     profile_gate_up_sec + profile_ffn_down_sec));
    }

    error_ret = 0;

cleanup:
    if (error_ret != 0) {
        int i = 0;
        for (i = 0; i < vocab_size; ++i) {
            ctx->logits[i] = 0.0f;
        }
    }
    return error_ret;
}

int bitnet_sample_greedy(bitnet_context_t *ctx) {
    if (ctx == NULL) return -1;
    return bitnet_sample_greedy_impl((int)ctx->model->vocab_size, ctx->logits);
}

int bitnet_sample_greedy_repetition_penalty(bitnet_context_t *ctx,
                                            const int *tokens,
                                            int n_tokens,
                                            float penalty) {
    int vocab_size = 0;
    int best_idx = -1;
    float best_val = -INFINITY;

    if (ctx == NULL || ctx->model == NULL || ctx->logits == NULL) {
        return -1;
    }
    if (tokens == NULL || n_tokens <= 0 || penalty <= 1.0f) {
        return bitnet_sample_greedy(ctx);
    }

    vocab_size = (int)ctx->model->vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        float val = ctx->logits[i];
        if (!bitnet_token_is_eog(ctx->model, i) && bitnet_recent_contains_token(tokens, n_tokens, i)) {
            val = val >= 0.0f ? val / penalty : val * penalty;
        }
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

const float *bitnet_get_logits(const bitnet_context_t *ctx) {
    if (ctx == NULL) return NULL;
    return ctx->logits;
}

const float *bitnet_get_last_hidden(const bitnet_context_t *ctx) {
    if (ctx == NULL) return NULL;
    return ctx->last_hidden;
}

void bitnet_free_context(bitnet_context_t *ctx) {
    if (ctx == NULL) return;
    free(ctx->key_cache);
    free(ctx->value_cache);
    bitnet_free_q8_kv_cache(ctx);
    free(ctx->logits);
    free(ctx->last_hidden);
    free(ctx->hidden);
    free(ctx->q);
    free(ctx->k);
    free(ctx->v);
    free(ctx->attn_buffer);
    free(ctx->gate);
    free(ctx->up);
    free(ctx->down);
    free(ctx->scores);
    free(ctx->tmp_out);
    free(ctx->tq2_lut);
    free(ctx->tq2_qhidden);
    free(ctx->tq2_qffn);
    free(ctx->tq2_tl1_lut);
    free(ctx->rope_cos);
    free(ctx->rope_sin);
    free(ctx);
}

void bitnet_free_model(bitnet_model_t *model) {
    if (model == NULL) return;
    free_output_q6k_cache(model);
    free_tq2_i2s_cache(&model->i2s_cache, model->block_count);
    free_tq2_scale_cache(&model->scale_cache, model->block_count);
    free_tq2_tl1_cache(&model->tl1_cache, model->block_count);
    bitnet_free_tensor_cache(&model->tensor_cache);
    bitnet_free_lora_adapter(model->lora);
    gguf_close(&model->gguf);
    bitnet_tokenizer_free(model->tokenizer);
    free(model->model_path);
    free(model);
}
