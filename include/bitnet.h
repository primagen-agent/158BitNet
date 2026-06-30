#ifndef BITNET_H
#define BITNET_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitnet_model bitnet_model_t;
typedef struct bitnet_context bitnet_context_t;

#define BITNET_KV_CACHE_F32 0
#define BITNET_KV_CACHE_Q8 1

bitnet_model_t *bitnet_load_model(const char *path);
bitnet_context_t *bitnet_create_context(bitnet_model_t *model, int max_tokens);
int bitnet_load_lora(bitnet_model_t *model, const char *path, float scale);
int bitnet_lora_count(const bitnet_model_t *model);
int bitnet_embedding_length(const bitnet_model_t *model);
int bitnet_vocab_size(const bitnet_model_t *model);
int bitnet_chat_template_kind(const bitnet_model_t *model);
const float *bitnet_get_last_hidden(const bitnet_context_t *ctx);

int bitnet_tokenize(bitnet_model_t *model, const char *text, int *tokens, int max_tokens);
int bitnet_tokenize_ex(bitnet_model_t *model, const char *text, int *tokens, int max_tokens, int add_bos);
int bitnet_decode_token(bitnet_model_t *model, int token, char *out, int out_size);
int bitnet_eos_token(bitnet_model_t *model);
int bitnet_pad_token(bitnet_model_t *model);
int bitnet_token_is_eog(bitnet_model_t *model, int token);

int bitnet_eval(bitnet_context_t *ctx, const int *tokens, int n_tokens);
int bitnet_sample_greedy(bitnet_context_t *ctx);
int bitnet_sample_greedy_repetition_penalty(bitnet_context_t *ctx,
                                            const int *tokens,
                                            int n_tokens,
                                            float penalty);
const float *bitnet_get_logits(const bitnet_context_t *ctx);
void bitnet_apply_repetition_penalty(float *logits,
                                     int vocab_size,
                                     const int *tokens,
                                     int n_tokens,
                                     float penalty);

int bitnet_context_pos(const bitnet_context_t *ctx);
int bitnet_context_capacity(const bitnet_context_t *ctx);
int bitnet_reset_context(bitnet_context_t *ctx);
int bitnet_rewind_context(bitnet_context_t *ctx, int n_pos);
int bitnet_set_kv_cache_type(bitnet_context_t *ctx, int cache_type);
int bitnet_get_kv_cache_type(const bitnet_context_t *ctx);
unsigned long long bitnet_kv_cache_bytes(const bitnet_context_t *ctx);

void bitnet_free_context(bitnet_context_t *ctx);
void bitnet_free_model(bitnet_model_t *model);

#ifdef __cplusplus
}
#endif

#endif
