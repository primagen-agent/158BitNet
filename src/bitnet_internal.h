#ifndef BITNET_INTERNAL_H
#define BITNET_INTERNAL_H

#include "gguf.h"

#define BITNET_TARGET_MODEL_FILE "bitcpm4-1b-tq2_0.gguf"
#define BITNET_TARGET_ARCHITECTURE "llama"
#define BITNET_TARGET_TOKENIZER_MODEL "llama"
#define BITNET_TARGET_FILE_TYPE 37u
#define BITNET_TARGET_EMBEDDING_LENGTH 2048u
#define BITNET_TARGET_BLOCK_COUNT 28u
#define BITNET_TARGET_CONTEXT_LENGTH 32768u
#define BITNET_TARGET_FEED_FORWARD_LENGTH 6144u
#define BITNET_TARGET_ATTENTION_HEAD_COUNT 16u
#define BITNET_TARGET_ATTENTION_HEAD_COUNT_KV 2u
#define BITNET_TARGET_VOCAB_SIZE 73448u
#define BITNET_TARGET_ROPE_DIMENSION_COUNT 128u
#define BITNET_TARGET_TENSOR_TYPE_TQ2_0 35u
#define BITNET_TARGET_TENSOR_TYPE_Q6_K 14u
#define BITNET_TARGET_TENSOR_TYPE_Q4_K 12u
#define BITNET_TARGET_TENSOR_TYPE_F16 1u
#define BITNET_TARGET_TENSOR_TYPE_F32 0u

typedef struct bitnet_block_tensors {
    const gguf_tensor_t *attn_norm;
    const gguf_tensor_t *attn_q;
    const gguf_tensor_t *attn_k;
    const gguf_tensor_t *attn_v;
    const gguf_tensor_t *attn_output;
    const gguf_tensor_t *ffn_norm;
    const gguf_tensor_t *ffn_gate;
    const gguf_tensor_t *ffn_up;
    const gguf_tensor_t *ffn_down;
} bitnet_block_tensors_t;

typedef struct bitnet_tensor_cache {
    const gguf_tensor_t *token_embd;
    const gguf_tensor_t *output;
    const gguf_tensor_t *output_norm;
    bitnet_block_tensors_t *blocks;
} bitnet_tensor_cache_t;

int bitnet_validate_target_model(const gguf_file_t *file);
int bitnet_build_tensor_cache(const gguf_file_t *file, uint32_t block_count, bitnet_tensor_cache_t *cache);
void bitnet_free_tensor_cache(bitnet_tensor_cache_t *cache);

#endif
