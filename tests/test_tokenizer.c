#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    bitnet_tokenizer_t *tokenizer = NULL;
    int tokens[64];
    char buf[64];
    int n_tokens = 0;
    int rc = 0;

    if (bitnet_tokenizer_load(&tokenizer, model_path) != 0) return 1;
    if (tokenizer == NULL) return 2;

    n_tokens = bitnet_tokenizer_encode(tokenizer, "hello world", tokens, 64);
    if (n_tokens <= 0) return 3;

    rc = bitnet_tokenizer_decode(tokenizer, tokens[0], buf, sizeof(buf));
    if (rc <= 0) return 4;

    bitnet_tokenizer_free(tokenizer);
    return 0;
}
