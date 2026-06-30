#include "gguf.h"
#include "tensor.h"
#include "bitnet_internal.h"

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

static int expect_u32_metadata(const gguf_file_t *file, const char *key, unsigned int expected, int base) {
    const gguf_metadata_t *entry = gguf_find_metadata(file, key);
    if (entry == 0) return base;
    if (entry->type != GGUF_TYPE_UINT32) return base + 1;
    if ((unsigned int)entry->value.u64 != expected) return base + 2;
    return 0;
}

static int expect_string_metadata(const gguf_file_t *file, const char *key, const char *expected) {
    const gguf_metadata_t *entry = gguf_find_metadata(file, key);
    const char *actual = 0;
    const char *a = 0;
    const char *b = 0;
    if (entry == 0) return 110;
    if (entry->type != GGUF_TYPE_STRING) return 111;
    actual = entry->string_value;
    if (actual == 0) return 112;
    a = actual;
    b = expected;
    while (*a != '\0' && *b != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b ? 0 : 113;
}

static int expect_tensor(const gguf_file_t *file, const char *name, unsigned int type, const uint64_t *dims, uint32_t n_dims) {
    const gguf_tensor_t *tensor = gguf_find_tensor(file, name);
    if (tensor == 0) return 120;
    if (tensor->type != type) return 121;
    if (!gguf_tensor_has_shape(tensor, dims, n_dims)) return 122;
    return 0;
}

static int test_i2s_model_metadata(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/ggml-model-i2_s.gguf";
    const uint64_t token_dims[] = {2560, 128256};
    const uint64_t norm_dims[] = {2560};
    const uint64_t attn_q_dims[] = {2560, 2560};
    const uint64_t attn_k_dims[] = {2560, 640};
    const uint64_t ffn_down_dims[] = {6912, 2560};
    gguf_file_t file;
    int rc = gguf_open(model_path, &file);
    if (rc != 0) return 0;

    rc = expect_string_metadata(&file, "general.architecture", "bitnet-b1.58");
    if (rc != 0) { gguf_close(&file); return rc + 200; }
    rc = expect_string_metadata(&file, "tokenizer.ggml.model", "gpt2");
    if (rc != 0) { gguf_close(&file); return rc + 210; }
    rc = expect_u32_metadata(&file, "general.file_type", BITNET_BITNET_B158_FILE_TYPE_I2_S, 220);
    if (rc != 0) { gguf_close(&file); return rc; }
    rc = expect_u32_metadata(&file, "bitnet-b1.58.embedding_length", 2560, 230);
    if (rc != 0) { gguf_close(&file); return rc; }
    rc = expect_u32_metadata(&file, "bitnet-b1.58.block_count", 30, 240);
    if (rc != 0) { gguf_close(&file); return rc; }

    rc = expect_tensor(&file, "token_embd.weight", BITNET_TARGET_TENSOR_TYPE_F16, token_dims, 2);
    if (rc != 0) { gguf_close(&file); return rc + 250; }
    rc = expect_tensor(&file, "output_norm.weight", BITNET_TARGET_TENSOR_TYPE_F32, norm_dims, 1);
    if (rc != 0) { gguf_close(&file); return rc + 260; }
    rc = expect_tensor(&file, "blk.0.attn_q.weight", BITNET_TARGET_TENSOR_TYPE_I2_S, attn_q_dims, 2);
    if (rc != 0) { gguf_close(&file); return rc + 270; }
    rc = expect_tensor(&file, "blk.0.attn_k.weight", BITNET_TARGET_TENSOR_TYPE_I2_S, attn_k_dims, 2);
    if (rc != 0) { gguf_close(&file); return rc + 280; }
    rc = expect_tensor(&file, "blk.0.ffn_down.weight", BITNET_TARGET_TENSOR_TYPE_I2_S, ffn_down_dims, 2);
    if (rc != 0) { gguf_close(&file); return rc + 290; }

    rc = bitnet_validate_target_model(&file);
    gguf_close(&file);
    return rc == 0 ? 0 : rc + 300;
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    const uint64_t output_dims[] = {2048, 73448};
    const uint64_t norm_dims[] = {2048};
    const uint64_t attn_q_dims[] = {2048, 2048};
    const uint64_t attn_k_dims[] = {2048, 256};
    const uint64_t ffn_down_dims[] = {6144, 2048};
    gguf_file_t file;
    int rc = gguf_open(model_path, &file);
    if (rc != 0) return 1;
    if (file.version == 0) return 2;
    if (file.tensor_count == 0) return 3;
    if (file.metadata_count == 0) return 4;

    rc = expect_string_metadata(&file, "general.architecture", BITNET_TARGET_ARCHITECTURE);
    if (rc != 0) return rc;
    rc = expect_string_metadata(&file, "tokenizer.ggml.model", BITNET_TARGET_TOKENIZER_MODEL);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "general.file_type", BITNET_TARGET_FILE_TYPE, 100);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "llama.embedding_length", BITNET_TARGET_EMBEDDING_LENGTH, 110);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "llama.block_count", BITNET_TARGET_BLOCK_COUNT, 120);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "llama.context_length", BITNET_TARGET_CONTEXT_LENGTH, 130);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "llama.feed_forward_length", BITNET_TARGET_FEED_FORWARD_LENGTH, 140);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "llama.attention.head_count", BITNET_TARGET_ATTENTION_HEAD_COUNT, 150);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "llama.attention.head_count_kv", BITNET_TARGET_ATTENTION_HEAD_COUNT_KV, 160);
    if (rc != 0) return rc;
    rc = expect_u32_metadata(&file, "llama.vocab_size", BITNET_TARGET_VOCAB_SIZE, 170);
    if (rc != 0) return rc;

    rc = expect_tensor(&file, "output.weight", BITNET_TARGET_TENSOR_TYPE_Q6_K, output_dims, 2);
    if (rc != 0) return rc;
    rc = expect_tensor(&file, "output_norm.weight", BITNET_TARGET_TENSOR_TYPE_F32, norm_dims, 1);
    if (rc != 0) return rc;
    rc = expect_tensor(&file, "blk.0.attn_q.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, attn_q_dims, 2);
    if (rc != 0) return rc;
    rc = expect_tensor(&file, "blk.0.attn_k.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, attn_k_dims, 2);
    if (rc != 0) return rc;
    rc = expect_tensor(&file, "blk.0.ffn_down.weight", BITNET_TARGET_TENSOR_TYPE_TQ2_0, ffn_down_dims, 2);
    if (rc != 0) return rc;

    rc = bitnet_validate_target_model(&file);
    if (rc != 0) return rc;

    gguf_close(&file);
    rc = test_i2s_model_metadata();
    if (rc != 0) return rc;
    return 0;
}
