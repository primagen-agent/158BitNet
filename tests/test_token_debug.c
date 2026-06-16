#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *model_path = "models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = NULL;
    int tokens[128];
    int n = 0;
    int i = 0;

    if (argc > 1) model_path = argv[1];

    model = bitnet_load_model(model_path);
    if (model == NULL) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Test 1: English */
    {
        const char *text = "The capital of France is";
        n = bitnet_tokenize(model, text, tokens, 128);
        printf("Text: \"%s\"\n", text);
        printf("Tokens (%d):", n);
        for (i = 0; i < n; ++i) {
            char decoded[64];
            bitnet_decode_token(model, tokens[i], decoded, sizeof(decoded));
            printf(" %d[%s]", tokens[i], decoded);
        }
        printf("\n\n");
    }

    /* Test 2: English story */
    {
        const char *text = "Once upon a time";
        n = bitnet_tokenize(model, text, tokens, 128);
        printf("Text: \"%s\"\n", text);
        printf("Tokens (%d):", n);
        for (i = 0; i < n; ++i) {
            char decoded[64];
            bitnet_decode_token(model, tokens[i], decoded, sizeof(decoded));
            printf(" %d[%s]", tokens[i], decoded);
        }
        printf("\n\n");
    }

    /* Test 3: Chinese */
    {
        const char *text = "人工智能是";
        n = bitnet_tokenize(model, text, tokens, 128);
        printf("Text: \"%s\"\n", text);
        printf("Tokens (%d):", n);
        for (i = 0; i < n; ++i) {
            char decoded[64];
            bitnet_decode_token(model, tokens[i], decoded, sizeof(decoded));
            printf(" %d[%s]", tokens[i], decoded);
        }
        printf("\n\n");
    }

    /* Test 4: simple word */
    {
        const char *text = "hello";
        n = bitnet_tokenize(model, text, tokens, 128);
        printf("Text: \"%s\"\n", text);
        printf("Tokens (%d):", n);
        for (i = 0; i < n; ++i) {
            char decoded[64];
            bitnet_decode_token(model, tokens[i], decoded, sizeof(decoded));
            printf(" %d[%s]", tokens[i], decoded);
        }
        printf("\n\n");
    }

    /* Test 5: single space-prefixed word */
    {
        const char *text = " Paris";
        n = bitnet_tokenize(model, text, tokens, 128);
        printf("Text: \"%s\"\n", text);
        printf("Tokens (%d):", n);
        for (i = 0; i < n; ++i) {
            char decoded[64];
            bitnet_decode_token(model, tokens[i], decoded, sizeof(decoded));
            printf(" %d[%s]", tokens[i], decoded);
        }
        printf("\n\n");
    }

    bitnet_free_model(model);
    return 0;
}
