#ifndef TOKENIZER_H
#define TOKENIZER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitnet_tokenizer bitnet_tokenizer_t;

int bitnet_tokenizer_load(bitnet_tokenizer_t **tokenizer, const char *gguf_path);
void bitnet_tokenizer_free(bitnet_tokenizer_t *tokenizer);

int bitnet_tokenizer_encode(bitnet_tokenizer_t *tokenizer, const char *text, int *tokens, int max_tokens);
int bitnet_tokenizer_decode(bitnet_tokenizer_t *tokenizer, int token, char *out, int out_size);
int bitnet_tokenizer_bos_id(const bitnet_tokenizer_t *tokenizer);
int bitnet_tokenizer_eos_id(const bitnet_tokenizer_t *tokenizer);
int bitnet_tokenizer_pad_id(const bitnet_tokenizer_t *tokenizer);
int bitnet_tokenizer_add_bos(const bitnet_tokenizer_t *tokenizer);

#ifdef __cplusplus
}
#endif

#endif
