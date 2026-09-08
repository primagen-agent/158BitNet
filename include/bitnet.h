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
/* Mean of output-normalized hidden rows from the most recent bitnet_eval,
 * plus its final row. This matches the portable memory-controller pooling
 * used during training. */
const float *bitnet_get_last_pooled_hidden(const bitnet_context_t *ctx);
/* Output-normalized hidden rows from the most recent bitnet_eval, laid out as
 * [bitnet_last_eval_hidden_count(ctx)][bitnet_embedding_length(model)]. */
const float *bitnet_get_last_eval_hidden(const bitnet_context_t *ctx);
int bitnet_last_eval_hidden_count(const bitnet_context_t *ctx);

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

/* Optional Metis memory model (side-car, no-op when not loaded).
 *
 * bitnet_load_memory_model loads a .bnmem file onto the model (validates
 * d_model/block_count against the backbone; replaces any previously loaded
 * memory model). bitnet_context_attach_memory binds a loaded memory model to
 * a context: it allocates the per-context memory state (M/S, initially zero,
 * inactive) and the commit-capture buffer. The model MUST be the same
 * bitnet_model_t the context was created from (the context sizes its memory
 * buffers from its own model's dimensions). A context without an attached
 * memory model evaluates exactly as before (bit-exact, one null-check per
 * block).
 *
 * Memory semantics: while attached, every bitnet_eval APPENDS each token's
 * attn-normed hidden state at the deepest memory layer to a commit-capture
 * buffer (an accumulation budget of min(max_tokens, 4096) rows per commit
 * window). bitnet_memory_commit runs the GDU commit over ALL states
 * accumulated since the previous commit — prefill plus every decode step of
 * an exchange, committed once at end of generation — and activates the
 * memory fusion in subsequent evals (state-gated bypass: until the first
 * commit the memory path is a strict no-op). Long windows are committed in
 * <=1024-row slices (GDU state carries across slices). An eval that would
 * push the window past the buffer capacity returns -3 and computes nothing.
 * bitnet_memory_reset zeroes the memory state and deactivates the fusion;
 * bitnet_reset_context (KV rewind) intentionally does NOT touch memory state.
 *
 * Returns: 0 on success; bitnet_load_memory_model -1 on load/validation
 * failure; bitnet_context_attach_memory -1 when the model has no memory
 * model loaded or allocation fails; bitnet_memory_commit -1 when no memory
 * model is attached or the last eval saved no states. */
int  bitnet_load_memory_model(bitnet_model_t *model, const char *path);
int  bitnet_context_attach_memory(bitnet_context_t *ctx, const bitnet_model_t *model);
void bitnet_memory_reset(bitnet_context_t *ctx);
int  bitnet_memory_commit(bitnet_context_t *ctx);
int  bitnet_memory_active(const bitnet_context_t *ctx);
/* Export/import the committed M/S state for one attached context. Pending
 * capture rows are intentionally excluded. Import validates the snapshot
 * against the attached memory model and leaves the current state unchanged
 * on failure. Returns 0 on success, -1 on error. */
int  bitnet_memory_export(const bitnet_context_t *ctx, const char *path);
int  bitnet_memory_import(bitnet_context_t *ctx, const char *path);
/* Discard the pending commit-capture rows (most recent evals) WITHOUT
 * writing them to memory. Reference protocol: a memory session commits
 * each user message right after its prefill and DISCARDS the assistant
 * ack's decode rows, so ack text never enters the memory state. */
void bitnet_memory_discard_captured(bitnet_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
