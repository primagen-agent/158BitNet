#include "tokenizer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    static const char i2s_model_path[] = BITNET_SOURCE_DIR "/models/ggml-model-i2_s.gguf";
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

    /* An unmatched raw byte must use the SentencePiece <0xHH> fallback token,
     * not collapse to <unk>.  The leading-space token may also be emitted. */
    {
        const char raw_byte[] = { 1, 0 };
        int found_raw_byte = 0;
        n_tokens = bitnet_tokenizer_encode(tokenizer, raw_byte, tokens, 64);
        if (n_tokens <= 0) return 11;
        for (int i = 0; i < n_tokens; ++i) {
            rc = bitnet_tokenizer_decode(tokenizer, tokens[i], buf, sizeof(buf));
            if (rc == 1 && (unsigned char)buf[0] == 1u) {
                found_raw_byte = 1;
            }
        }
        if (!found_raw_byte) return 12;
    }

    bitnet_tokenizer_free(tokenizer);

    if (bitnet_tokenizer_load(&tokenizer, i2s_model_path) == 0) {
        char roundtrip[128];
        size_t offset = 0;

        n_tokens = bitnet_tokenizer_encode(tokenizer, "hello world", tokens, 64);
        if (n_tokens <= 0) return 5;

        roundtrip[0] = '\0';
        for (int i = 0; i < n_tokens; ++i) {
            rc = bitnet_tokenizer_decode(tokenizer, tokens[i], buf, sizeof(buf));
            if (rc < 0) return 6;
            if (offset + (size_t)rc >= sizeof(roundtrip)) return 7;
            memcpy(roundtrip + offset, buf, (size_t)rc);
            offset += (size_t)rc;
            roundtrip[offset] = '\0';
        }
        if (strcmp(roundtrip, "hello world") != 0) return 8;

        n_tokens = bitnet_tokenizer_encode(tokenizer, "<|end_of_text|>", tokens, 64);
        if (n_tokens != 1) return 9;
        if (tokens[0] != 128001) return 10;

        bitnet_tokenizer_free(tokenizer);
    }
    return 0;
}
