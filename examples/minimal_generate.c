#include "bitnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GENERATE 4
#define MIN_CONTEXT_TOKENS 128
#define DEFAULT_MAX_CONTEXT_TOKENS 2048
#define DEFAULT_REPEAT_LAST_N 64
#define DEFAULT_REPEAT_PENALTY 1.1f

static int env_int_or_default(const char *name, int fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    long parsed = 0;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    parsed = strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > 32768) {
        return fallback;
    }
    return (int)parsed;
}

static float env_float_or_default(const char *name, float fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    float parsed = 0.0f;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    parsed = strtof(value, &end);
    if (end == value || parsed < 1.0f || parsed > 10.0f) {
        return fallback;
    }
    return parsed;
}

typedef struct utf8_stream_state {
    char pending[4];
    int pending_len;
    int pending_expected;
} utf8_stream_state_t;

static int utf8_expected_len(unsigned char b) {
    if (b < 0x80u) return 1;
    if (b >= 0xC2u && b <= 0xDFu) return 2;
    if (b >= 0xE0u && b <= 0xEFu) return 3;
    if (b >= 0xF0u && b <= 0xF4u) return 4;
    return 0;
}

static int utf8_is_cont(unsigned char b) {
    return (b & 0xC0u) == 0x80u;
}

static void print_decoded_utf8(utf8_stream_state_t *state, const char *bytes, int byte_len) {
    if (state == NULL || bytes == NULL || byte_len <= 0) return;

    for (int i = 0; i < byte_len; ++i) {
        unsigned char b = (unsigned char)bytes[i];

        if (state->pending_len == 0) {
            int expected = utf8_expected_len(b);
            if (expected == 1) {
                putchar((int)b);
            } else if (expected > 1) {
                state->pending[0] = (char)b;
                state->pending_len = 1;
                state->pending_expected = expected;
            }
        } else if (utf8_is_cont(b)) {
            state->pending[state->pending_len++] = (char)b;
            if (state->pending_len == state->pending_expected) {
                fwrite(state->pending, 1, (size_t)state->pending_expected, stdout);
                state->pending_len = 0;
                state->pending_expected = 0;
            }
        } else {
            state->pending_len = 0;
            state->pending_expected = 0;
            --i;
        }
    }
}

static int read_le_u32(FILE *f, unsigned int *out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *out = (unsigned int)b[0] | ((unsigned int)b[1] << 8) |
           ((unsigned int)b[2] << 16) | ((unsigned int)b[3] << 24);
    return 1;
}

/* Display-only banner: re-read the .bnmem header (the file was just
 * validated by bitnet_load_memory_model) to list its memory layer ids.
 * v1/v2 layout: 8-byte magic, u32 version (1 or 2), u32 n_layers, then u32
 * layer_ids[4] (0xFFFFFFFF = unused slot), all little-endian.
 * v3/v4 layout: magic "BNMEM3"/"BNMEM4", u32 version, u32 n_layers, then
 * u32 layer_ids[32]. Range-compressed display for the 32-layer full-path
 * file; v4 files append " v4" to flag reference-trained direct-use models. */
static void print_memory_model_banner(const char *path) {
    enum { kMaxLayers = 32 };
    unsigned char magic[8];
    unsigned int version = 0, n_layers = 0;
    unsigned int ids[kMaxLayers];
    FILE *f = fopen(path, "rb");
    int ok = (f != NULL);

    if (ok) ok = (fread(magic, 1, sizeof magic, f) == (size_t)sizeof magic);
    if (ok) ok = read_le_u32(f, &version);
    if (ok) ok = (version >= 1u && version <= 4u);
    if (ok) ok = read_le_u32(f, &n_layers);
    const int slots = version >= 3u ? 32 : 4;
    for (int i = 0; i < kMaxLayers; ++i) {
        ids[i] = 0xFFFFFFFFu;
        if (ok && i < slots) ok = read_le_u32(f, &ids[i]);
    }
    if (f != NULL) fclose(f);

    fprintf(stderr, "[bitnet] memory model: %s%s layers=[", path,
            version == 4u ? " v4" : "");
    if (ok && n_layers >= 1u && n_layers <= (unsigned int)slots) {
        if (version >= 3u && n_layers > 6) {
            fprintf(stderr, "%u..%u x%u", ids[0], ids[n_layers - 1],
                    n_layers);
        } else {
            for (unsigned int i = 0; i < n_layers; ++i) {
                fprintf(stderr, "%s%u", i > 0 ? "," : "", ids[i]);
            }
        }
    }
    fprintf(stderr, "]\n");
}

int main(int argc, char **argv) {
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int *prompt_tokens = NULL;
    int *history_tokens = NULL;
    int n_prompt_tokens = 0;
    int history_count = 0;
    int context_tokens = 0;
    int max_context_tokens = env_int_or_default("BITNET_MAX_CONTEXT", DEFAULT_MAX_CONTEXT_TOKENS);
    int repeat_last_n = env_int_or_default("BITNET_REPEAT_LAST_N", DEFAULT_REPEAT_LAST_N);
    float repeat_penalty = env_float_or_default("BITNET_REPEAT_PENALTY", DEFAULT_REPEAT_PENALTY);
    const char *memory_model_path = NULL;
    char decoded[256];
    utf8_stream_state_t utf8_state;
    int i = 0;

    memset(&utf8_state, 0, sizeof(utf8_state));

    /* Pre-pass: pull "--memory-model PATH" out of argv, shifting the rest
     * down so the positional parsing below is unchanged. */
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--memory-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--memory-model requires a path argument\n");
                return 1;
            }
            memory_model_path = argv[i + 1];
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2;
            --i;
        }
    }

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt> [num_tokens] [--memory-model PATH]\n", argv[0]);
        return 1;
    }

    int num_tokens = MAX_GENERATE;
    if (argc >= 4) {
        num_tokens = atoi(argv[3]);
        if (num_tokens <= 0) num_tokens = MAX_GENERATE;
    }

    model = bitnet_load_model(argv[1]);
    if (model == NULL) {
        fprintf(stderr, "Failed to load model from: %s\n", argv[1]);
        return 1;
    }

    if (memory_model_path != NULL) {
        if (bitnet_load_memory_model(model, memory_model_path) != 0) {
            fprintf(stderr, "Failed to load memory model: %s\n", memory_model_path);
            bitnet_free_model(model);
            return 1;
        }
        print_memory_model_banner(memory_model_path);
    }

    if (max_context_tokens < MIN_CONTEXT_TOKENS) {
        max_context_tokens = MIN_CONTEXT_TOKENS;
    }

    prompt_tokens = (int *)calloc((size_t)max_context_tokens, sizeof(*prompt_tokens));
    if (prompt_tokens == NULL) {
        fprintf(stderr, "Failed to allocate prompt buffer\n");
        bitnet_free_model(model);
        return 1;
    }

    n_prompt_tokens = bitnet_tokenize(model, argv[2], prompt_tokens, max_context_tokens);
    if (n_prompt_tokens <= 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }
    if (n_prompt_tokens >= max_context_tokens - 1) {
        fprintf(stderr, "Prompt is too long for BITNET_MAX_CONTEXT=%d\n", max_context_tokens);
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }

    if (n_prompt_tokens + num_tokens + 1 > max_context_tokens) {
        int clamped = max_context_tokens - n_prompt_tokens - 1;
        if (clamped <= 0) {
            fprintf(stderr, "No room left for generation in context\n");
            free(prompt_tokens);
            bitnet_free_model(model);
            return 1;
        }
        fprintf(stderr, "Clamping generation from %d to %d tokens for context size %d\n",
                num_tokens, clamped, max_context_tokens);
        num_tokens = clamped;
    }

    context_tokens = n_prompt_tokens + num_tokens + 1;
    if (context_tokens < MIN_CONTEXT_TOKENS) {
        context_tokens = MIN_CONTEXT_TOKENS;
    }

    ctx = bitnet_create_context(model, context_tokens);
    if (ctx == NULL) {
        fprintf(stderr, "Failed to create context\n");
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }

    if (memory_model_path != NULL) {
        if (bitnet_context_attach_memory(ctx, model) != 0) {
            fprintf(stderr, "Failed to attach memory model\n");
            bitnet_free_context(ctx);
            free(prompt_tokens);
            bitnet_free_model(model);
            return 1;
        }
    }

    history_tokens = (int *)calloc((size_t)context_tokens, sizeof(*history_tokens));
    if (history_tokens == NULL) {
        fprintf(stderr, "Failed to allocate generation history\n");
        bitnet_free_context(ctx);
        free(prompt_tokens);
        bitnet_free_model(model);
        return 1;
    }
    memcpy(history_tokens, prompt_tokens, (size_t)n_prompt_tokens * sizeof(*history_tokens));
    history_count = n_prompt_tokens;

    printf("Prompt tokens (%d):", n_prompt_tokens);
    for (i = 0; i < n_prompt_tokens; ++i) {
        printf(" %d", prompt_tokens[i]);
    }
    printf("\n");

    if (bitnet_eval(ctx, prompt_tokens, n_prompt_tokens) != 0) {
        fprintf(stderr, "Eval failed\n");
        free(history_tokens);
        free(prompt_tokens);
        bitnet_free_context(ctx);
        bitnet_free_model(model);
        return 1;
    }

    /* Commit the prompt's hidden states into the memory M/S store; reads in
     * the decode loop below participate because active=1 from here on. */
    if (memory_model_path != NULL) {
        if (bitnet_memory_commit(ctx) != 0) {
            fprintf(stderr, "Failed to commit memory model states\n");
            free(history_tokens);
            free(prompt_tokens);
            bitnet_free_context(ctx);
            bitnet_free_model(model);
            return 1;
        }
    }

    for (i = 0; i < num_tokens; ++i) {
        int recent_count = history_count < repeat_last_n ? history_count : repeat_last_n;
        const int *recent = history_tokens + history_count - recent_count;
        int next_token = bitnet_sample_greedy_repetition_penalty(ctx, recent, recent_count, repeat_penalty);
        if (next_token < 0) {
            fprintf(stderr, "Sample failed\n");
            break;
        }
        if (bitnet_token_is_eog(model, next_token)) {
            break;
        }

        {
            int decoded_len = bitnet_decode_token(model, next_token, decoded, (int)sizeof(decoded));
            if (decoded_len > 0) {
                print_decoded_utf8(&utf8_state, decoded, decoded_len);
            }
            fflush(stdout);
        }

        if (history_count < context_tokens) {
            history_tokens[history_count++] = next_token;
        }
        if (bitnet_eval(ctx, &next_token, 1) != 0) {
            fprintf(stderr, "Eval failed for token %d\n", next_token);
            break;
        }
    }

    printf("\n");

    free(history_tokens);
    free(prompt_tokens);
    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return 0;
}
