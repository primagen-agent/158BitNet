#include "tokenizer.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

typedef enum bitnet_tokenizer_model {
    BITNET_TOKENIZER_MODEL_LLAMA = 0,
    BITNET_TOKENIZER_MODEL_GPT2 = 1
} bitnet_tokenizer_model_t;

typedef struct bitnet_token_index_entry {
    const char *key;
    size_t len;
    int id;
} bitnet_token_index_entry_t;

typedef struct bitnet_merge_entry {
    uint64_t key;
    int rank;
    int token_id;
} bitnet_merge_entry_t;

struct bitnet_tokenizer {
    bitnet_tokenizer_model_t model;
    size_t vocab_size;
    char **tokens;
    float *scores;
    int *token_types;
    bitnet_token_index_entry_t *token_index;
    size_t token_index_capacity;
    bitnet_merge_entry_t *merge_table;
    size_t merge_table_capacity;
    int bos_token_id;
    int eos_token_id;
    int unk_token_id;
    int pad_token_id;
    int add_bos_token;
};

static uint64_t fnv1a_hash_bytes(const char *s, size_t len) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static size_t next_power_of_two_size(size_t n) {
    size_t p = 1;
    while (p < n && p <= (SIZE_MAX >> 1)) {
        p <<= 1;
    }
    return p;
}

static int token_index_init(bitnet_tokenizer_t *tokenizer) {
    size_t capacity = 0;

    if (tokenizer == NULL || tokenizer->tokens == NULL || tokenizer->vocab_size == 0) {
        return -1;
    }

    capacity = next_power_of_two_size(tokenizer->vocab_size * 2u + 1u);
    tokenizer->token_index = (bitnet_token_index_entry_t *)calloc(capacity, sizeof(*tokenizer->token_index));
    if (tokenizer->token_index == NULL) {
        return -1;
    }
    tokenizer->token_index_capacity = capacity;

    for (size_t i = 0; i < tokenizer->vocab_size; ++i) {
        const char *tok = tokenizer->tokens[i];
        size_t len = tok != NULL ? strlen(tok) : 0;
        uint64_t hash = fnv1a_hash_bytes(tok != NULL ? tok : "", len);
        size_t mask = capacity - 1u;
        size_t slot = (size_t)hash & mask;

        if (tok == NULL) {
            continue;
        }
        while (tokenizer->token_index[slot].key != NULL) {
            slot = (slot + 1u) & mask;
        }
        tokenizer->token_index[slot].key = tok;
        tokenizer->token_index[slot].len = len;
        tokenizer->token_index[slot].id = (int)i;
    }

    return 0;
}

static int token_index_find(const bitnet_tokenizer_t *tokenizer, const char *str, size_t len) {
    size_t mask = 0;
    size_t slot = 0;

    if (tokenizer == NULL || tokenizer->token_index == NULL ||
        tokenizer->token_index_capacity == 0 || str == NULL) {
        return -1;
    }

    mask = tokenizer->token_index_capacity - 1u;
    slot = (size_t)fnv1a_hash_bytes(str, len) & mask;
    while (tokenizer->token_index[slot].key != NULL) {
        const bitnet_token_index_entry_t *entry = &tokenizer->token_index[slot];
        if (entry->len == len && memcmp(entry->key, str, len) == 0) {
            return entry->id;
        }
        slot = (slot + 1u) & mask;
    }
    return -1;
}

static uint64_t merge_key(int left, int right) {
    return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
}

static int merge_table_insert(bitnet_tokenizer_t *tokenizer,
                              int left, int right, int rank, int token_id) {
    size_t mask = tokenizer->merge_table_capacity - 1u;
    uint64_t key = merge_key(left, right);
    size_t slot = (size_t)(key ^ (key >> 33)) & mask;

    while (tokenizer->merge_table[slot].token_id >= 0) {
        if (tokenizer->merge_table[slot].key == key) {
            if (rank < tokenizer->merge_table[slot].rank) {
                tokenizer->merge_table[slot].rank = rank;
                tokenizer->merge_table[slot].token_id = token_id;
            }
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    tokenizer->merge_table[slot].key = key;
    tokenizer->merge_table[slot].rank = rank;
    tokenizer->merge_table[slot].token_id = token_id;
    return 0;
}

static int merge_table_find(const bitnet_tokenizer_t *tokenizer,
                            int left, int right, int *rank, int *token_id) {
    size_t mask = 0;
    size_t slot = 0;
    uint64_t key = 0;

    if (tokenizer == NULL || tokenizer->merge_table == NULL ||
        tokenizer->merge_table_capacity == 0) {
        return 0;
    }

    key = merge_key(left, right);
    mask = tokenizer->merge_table_capacity - 1u;
    slot = (size_t)(key ^ (key >> 33)) & mask;
    while (tokenizer->merge_table[slot].token_id >= 0) {
        if (tokenizer->merge_table[slot].key == key) {
            if (rank != NULL) *rank = tokenizer->merge_table[slot].rank;
            if (token_id != NULL) *token_id = tokenizer->merge_table[slot].token_id;
            return 1;
        }
        slot = (slot + 1u) & mask;
    }
    return 0;
}

static int build_gpt2_merge_table(bitnet_tokenizer_t *tokenizer,
                                  const gguf_metadata_t *merges_meta) {
    size_t capacity = 0;

    if (tokenizer == NULL || merges_meta == NULL ||
        merges_meta->type != GGUF_TYPE_ARRAY ||
        merges_meta->array_type != GGUF_TYPE_STRING ||
        merges_meta->array_count == 0) {
        return 0;
    }

    capacity = next_power_of_two_size((size_t)merges_meta->array_count * 2u + 1u);
    tokenizer->merge_table = (bitnet_merge_entry_t *)malloc(capacity * sizeof(*tokenizer->merge_table));
    if (tokenizer->merge_table == NULL) {
        return -1;
    }
    tokenizer->merge_table_capacity = capacity;
    for (size_t i = 0; i < capacity; ++i) {
        tokenizer->merge_table[i].key = 0;
        tokenizer->merge_table[i].rank = INT_MAX;
        tokenizer->merge_table[i].token_id = -1;
    }

    for (uint64_t rank = 0; rank < merges_meta->array_count; ++rank) {
        const char *merge = merges_meta->array_strings[rank];
        const char *sep = NULL;
        size_t left_len = 0;
        size_t right_len = 0;
        int left_id = -1;
        int right_id = -1;
        int merged_id = -1;
        char stack_buf[512];
        char *merged = stack_buf;

        if (merge == NULL) {
            continue;
        }
        sep = strchr(merge, ' ');
        if (sep == NULL || sep == merge || sep[1] == '\0') {
            continue;
        }

        left_len = (size_t)(sep - merge);
        right_len = strlen(sep + 1);
        left_id = token_index_find(tokenizer, merge, left_len);
        right_id = token_index_find(tokenizer, sep + 1, right_len);
        if (left_id < 0 || right_id < 0) {
            continue;
        }

        if (left_len + right_len + 1u > sizeof(stack_buf)) {
            merged = (char *)malloc(left_len + right_len + 1u);
            if (merged == NULL) {
                return -1;
            }
        }
        memcpy(merged, merge, left_len);
        memcpy(merged + left_len, sep + 1, right_len);
        merged[left_len + right_len] = '\0';

        merged_id = token_index_find(tokenizer, merged, left_len + right_len);
        if (merged != stack_buf) {
            free(merged);
        }
        if (merged_id < 0) {
            continue;
        }

        (void)merge_table_insert(tokenizer, left_id, right_id, (int)rank, merged_id);
    }

    return 0;
}

int bitnet_tokenizer_load(bitnet_tokenizer_t **out, const char *gguf_path) {
    gguf_file_t file;
    const gguf_metadata_t *tokens_meta = NULL;
    const gguf_metadata_t *scores_meta = NULL;
    const gguf_metadata_t *types_meta = NULL;
    const gguf_metadata_t *model_meta = NULL;
    const gguf_metadata_t *merges_meta = NULL;
    const gguf_metadata_t *bos_entry = NULL;
    const gguf_metadata_t *eos_entry = NULL;
    const gguf_metadata_t *unk_entry = NULL;
    const gguf_metadata_t *pad_entry = NULL;
    gguf_metadata_t *tokens_mutable = NULL;
    gguf_metadata_t *scores_mutable = NULL;
    gguf_metadata_t *types_mutable = NULL;
    bitnet_tokenizer_t *tokenizer = NULL;
    size_t vocab_size = 0;

    if (out == NULL || gguf_path == NULL) {
        return -1;
    }

    *out = NULL;

    if (gguf_open(gguf_path, &file) != 0) {
        return -1;
    }

    tokens_meta = gguf_find_metadata(&file, "tokenizer.ggml.tokens");
    if (tokens_meta == NULL || tokens_meta->type != GGUF_TYPE_ARRAY || tokens_meta->array_type != GGUF_TYPE_STRING) {
        gguf_close(&file);
        return -1;
    }

    vocab_size = (size_t)tokens_meta->array_count;
    if (vocab_size == 0) {
        gguf_close(&file);
        return -1;
    }

    tokenizer = (bitnet_tokenizer_t *)calloc(1, sizeof(*tokenizer));
    if (tokenizer == NULL) {
        gguf_close(&file);
        return -1;
    }

    tokenizer->vocab_size = vocab_size;
    tokenizer->bos_token_id = -1;
    tokenizer->eos_token_id = -1;
    tokenizer->unk_token_id = -1;
    tokenizer->pad_token_id = -1;

    model_meta = gguf_find_metadata(&file, "tokenizer.ggml.model");
    if (model_meta != NULL && model_meta->type == GGUF_TYPE_STRING &&
        model_meta->string_value != NULL &&
        strcmp(model_meta->string_value, "gpt2") == 0) {
        tokenizer->model = BITNET_TOKENIZER_MODEL_GPT2;
    } else {
        tokenizer->model = BITNET_TOKENIZER_MODEL_LLAMA;
    }

    /* Steal tokens string array */
    tokens_mutable = (gguf_metadata_t *)tokens_meta;
    tokenizer->tokens = tokens_mutable->array_strings;
    tokens_mutable->array_strings = NULL;

    if (token_index_init(tokenizer) != 0) {
        bitnet_tokenizer_free(tokenizer);
        gguf_close(&file);
        return -1;
    }

    /* Steal scores float array */
    scores_meta = gguf_find_metadata(&file, "tokenizer.ggml.scores");
    if (scores_meta != NULL && scores_meta->type == GGUF_TYPE_ARRAY && scores_meta->array_type == GGUF_TYPE_FLOAT32) {
        scores_mutable = (gguf_metadata_t *)scores_meta;
        tokenizer->scores = scores_mutable->array_floats;
        scores_mutable->array_floats = NULL;
    }

    /* Steal token_type int array */
    types_meta = gguf_find_metadata(&file, "tokenizer.ggml.token_type");
    if (types_meta != NULL && types_meta->type == GGUF_TYPE_ARRAY && types_meta->array_type == GGUF_TYPE_INT32) {
        types_mutable = (gguf_metadata_t *)types_meta;
        tokenizer->token_types = types_mutable->array_ints;
        types_mutable->array_ints = NULL;
    }

    bos_entry = gguf_find_metadata(&file, "tokenizer.ggml.bos_token_id");
    eos_entry = gguf_find_metadata(&file, "tokenizer.ggml.eos_token_id");
    unk_entry = gguf_find_metadata(&file, "tokenizer.ggml.unknown_token_id");
    pad_entry = gguf_find_metadata(&file, "tokenizer.ggml.padding_token_id");

    if (bos_entry != NULL && bos_entry->type == GGUF_TYPE_UINT32) tokenizer->bos_token_id = (int)bos_entry->value.u64;
    if (eos_entry != NULL && eos_entry->type == GGUF_TYPE_UINT32) tokenizer->eos_token_id = (int)eos_entry->value.u64;
    if (unk_entry != NULL && unk_entry->type == GGUF_TYPE_UINT32) tokenizer->unk_token_id = (int)unk_entry->value.u64;
    if (pad_entry != NULL && pad_entry->type == GGUF_TYPE_UINT32) tokenizer->pad_token_id = (int)pad_entry->value.u64;

    {
        const gguf_metadata_t *add_bos_entry = gguf_find_metadata(&file, "tokenizer.ggml.add_bos_token");
        if (add_bos_entry != NULL && add_bos_entry->type == GGUF_TYPE_BOOL) {
            tokenizer->add_bos_token = (int)add_bos_entry->value.u64;
        }
    }

    if (tokenizer->model == BITNET_TOKENIZER_MODEL_GPT2) {
        merges_meta = gguf_find_metadata(&file, "tokenizer.ggml.merges");
        if (build_gpt2_merge_table(tokenizer, merges_meta) != 0) {
            bitnet_tokenizer_free(tokenizer);
            gguf_close(&file);
            return -1;
        }
    }

    gguf_close(&file);
    *out = tokenizer;
    return 0;
}

void bitnet_tokenizer_free(bitnet_tokenizer_t *tokenizer) {
    size_t i = 0;
    if (tokenizer == NULL) return;

    if (tokenizer->tokens != NULL) {
        for (i = 0; i < tokenizer->vocab_size; ++i) {
            free(tokenizer->tokens[i]);
        }
        free(tokenizer->tokens);
    }

    free(tokenizer->scores);
    free(tokenizer->token_types);
    free(tokenizer->token_index);
    free(tokenizer->merge_table);
    free(tokenizer);
}

static int token_str_len(const char *s) {
    const char *p = s;
    while (*p != '\0') ++p;
    return (int)(p - s);
}

/* ---- BPE Encoder ---- */

/* Find token ID by exact string match. Returns -1 if not found. */
static int find_token_id(const bitnet_tokenizer_t *tokenizer, const char *str, size_t len) {
    return token_index_find(tokenizer, str, len);
}

/* Check if a byte value has a <0xHH> fallback token. Returns token ID or -1. */
static int byte_fallback_token(const bitnet_tokenizer_t *tokenizer, uint8_t byte_val) {
    char buf[5];
    (void)snprintf(buf, sizeof(buf), "<0x%02X>", byte_val);
    return find_token_id(tokenizer, buf, 4);
}

static int gpt2_byte_to_codepoint(unsigned int byte_val) {
    if ((byte_val >= 33u && byte_val <= 126u) ||
        (byte_val >= 161u && byte_val <= 172u) ||
        (byte_val >= 174u && byte_val <= 255u)) {
        return (int)byte_val;
    }

    {
        unsigned int n = 0;
        for (unsigned int b = 0; b < 256u; ++b) {
            int direct = (b >= 33u && b <= 126u) ||
                         (b >= 161u && b <= 172u) ||
                         (b >= 174u && b <= 255u);
            if (!direct) {
                if (b == byte_val) {
                    return (int)(256u + n);
                }
                ++n;
            }
        }
    }

    return -1;
}

static int gpt2_codepoint_to_byte(unsigned int codepoint) {
    if ((codepoint >= 33u && codepoint <= 126u) ||
        (codepoint >= 161u && codepoint <= 172u) ||
        (codepoint >= 174u && codepoint <= 255u)) {
        return (int)codepoint;
    }

    {
        unsigned int n = 0;
        for (unsigned int b = 0; b < 256u; ++b) {
            int direct = (b >= 33u && b <= 126u) ||
                         (b >= 161u && b <= 172u) ||
                         (b >= 174u && b <= 255u);
            if (!direct) {
                if (codepoint == 256u + n) {
                    return (int)b;
                }
                ++n;
            }
        }
    }

    return -1;
}

static int utf8_encode_codepoint(unsigned int cp, char out[4]) {
    if (cp <= 0x7Fu) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FFu) {
        out[0] = (char)(0xC0u | (cp >> 6));
        out[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp <= 0xFFFFu) {
        out[0] = (char)(0xE0u | (cp >> 12));
        out[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (char)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    if (cp <= 0x10FFFFu) {
        out[0] = (char)(0xF0u | (cp >> 18));
        out[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
        out[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[3] = (char)(0x80u | (cp & 0x3Fu));
        return 4;
    }
    return -1;
}

static int utf8_decode_codepoint(const unsigned char *src, const unsigned char *end,
                                 unsigned int *cp, int *len) {
    unsigned char c = 0;

    if (src == NULL || src >= end || cp == NULL || len == NULL) {
        return -1;
    }

    c = src[0];
    if (c < 0x80u) {
        *cp = c;
        *len = 1;
        return 0;
    }
    if ((c & 0xE0u) == 0xC0u && src + 1 < end) {
        *cp = ((unsigned int)(c & 0x1Fu) << 6) |
              (unsigned int)(src[1] & 0x3Fu);
        *len = 2;
        return 0;
    }
    if ((c & 0xF0u) == 0xE0u && src + 2 < end) {
        *cp = ((unsigned int)(c & 0x0Fu) << 12) |
              ((unsigned int)(src[1] & 0x3Fu) << 6) |
              (unsigned int)(src[2] & 0x3Fu);
        *len = 3;
        return 0;
    }
    if ((c & 0xF8u) == 0xF0u && src + 3 < end) {
        *cp = ((unsigned int)(c & 0x07u) << 18) |
              ((unsigned int)(src[1] & 0x3Fu) << 12) |
              ((unsigned int)(src[2] & 0x3Fu) << 6) |
              (unsigned int)(src[3] & 0x3Fu);
        *len = 4;
        return 0;
    }

    return -1;
}

static int gpt2_encode(bitnet_tokenizer_t *tokenizer, const char *text,
                       int *tokens, int max_tokens) {
    const unsigned char *bytes = (const unsigned char *)text;
    size_t text_len = strlen(text);
    int *piece_ids = NULL;
    int n_pieces = 0;
    int out_count = 0;

    piece_ids = (int *)calloc(text_len + 1u, sizeof(*piece_ids));
    if (piece_ids == NULL) {
        return -1;
    }

    for (size_t i = 0; i < text_len;) {
        int special_id = -1;
        size_t special_len = 0;

        if (bytes[i] == '<') {
            for (size_t try_len = text_len - i; try_len > 0; --try_len) {
                if (try_len >= 4 &&
                    bytes[i + 1] == '|' &&
                    bytes[i + try_len - 2] == '|' &&
                    bytes[i + try_len - 1] == '>') {
                    special_id = find_token_id(tokenizer, (const char *)bytes + i, try_len);
                    if (special_id >= 0) {
                        special_len = try_len;
                        break;
                    }
                }
            }
        }
        if (special_id >= 0 && special_len > 0) {
            piece_ids[n_pieces++] = special_id;
            i += special_len;
            continue;
        }

        int cp = gpt2_byte_to_codepoint(bytes[i]);
        char utf8[4];
        int len = cp >= 0 ? utf8_encode_codepoint((unsigned int)cp, utf8) : -1;
        int id = len > 0 ? find_token_id(tokenizer, utf8, (size_t)len) : -1;

        if (id < 0) {
            id = byte_fallback_token(tokenizer, bytes[i]);
        }
        if (id < 0) {
            id = tokenizer->unk_token_id;
        }
        if (id >= 0) {
            piece_ids[n_pieces++] = id;
        }
        ++i;
    }

    while (n_pieces > 1) {
        int best_idx = -1;
        int best_rank = INT_MAX;
        int best_token = -1;

        for (int i = 0; i + 1 < n_pieces; ++i) {
            int rank = INT_MAX;
            int merged_id = -1;
            if (merge_table_find(tokenizer, piece_ids[i], piece_ids[i + 1],
                                 &rank, &merged_id) &&
                rank < best_rank) {
                best_idx = i;
                best_rank = rank;
                best_token = merged_id;
            }
        }

        if (best_idx < 0 || best_token < 0) {
            break;
        }

        piece_ids[best_idx] = best_token;
        for (int i = best_idx + 1; i < n_pieces - 1; ++i) {
            piece_ids[i] = piece_ids[i + 1];
        }
        --n_pieces;
    }

    for (int i = 0; i < n_pieces && out_count < max_tokens; ++i) {
        tokens[out_count++] = piece_ids[i];
    }

    free(piece_ids);
    return out_count;
}

/* BPE encode: start with character-level tokens, then iteratively merge
 * the highest-scoring adjacent pair until no more merges are possible.
 *
 * This follows the SentencePiece BPE algorithm used by llama.cpp. */
int bitnet_tokenizer_encode(bitnet_tokenizer_t *tokenizer, const char *text, int *tokens, int max_tokens) {
    int n = 0;
    const unsigned char *pos = NULL;
    int *piece_ids = NULL;
    int n_pieces = 0;
    int i = 0;
    char *spaced = NULL;

    if (tokenizer == NULL || text == NULL || tokens == NULL || max_tokens <= 0) {
        return -1;
    }

    if (tokenizer->model == BITNET_TOKENIZER_MODEL_GPT2) {
        return gpt2_encode(tokenizer, text, tokens, max_tokens);
    }

    /* Step 1: Convert input text to initial piece tokens.
     * For SentencePiece/llama tokenizer:
     * - A space is prepended to the text (SentencePiece convention: treat
     *   the beginning of text as if it follows a space)
     * - Spaces are converted to ▁ (U+2581, 0xE2 0x96 0x81 in UTF-8)
     * - Each character/byte is matched to a token
     * - Unmatched bytes use <0xHH> byte fallback tokens */
    {
        size_t text_len = strlen(text);
        /* +1 for prepended space */
        piece_ids = (int *)calloc((text_len + 1) * 4 + 1, sizeof(int));
        if (piece_ids == NULL) return -1;
    }

    /* Prepend a space to the text (SentencePiece convention) */
    {
        size_t text_len = strlen(text);
        spaced = (char *)calloc(text_len + 2, 1);
        if (spaced == NULL) { free(piece_ids); return -1; }
        spaced[0] = ' ';
        memcpy(spaced + 1, text, text_len);
    }

    pos = (const unsigned char *)spaced;
    n_pieces = 0;

    while (*pos != '\0') {
        int found = 0;

        /* Try to match the longest token starting at current position.
         * SentencePiece uses ▁ (LOWER ONE EIGHTH BLOCK U+2581) as space prefix.
         * We need to handle the space→▁ conversion. */
        {
            /* Build a temporary buffer with ▁ substitution for matching */
            const unsigned char *start = pos;
            size_t remaining = strlen((const char *)pos);

            /* Try longest match first */
            size_t try_len = remaining;
            for (try_len = remaining; try_len >= 1; --try_len) {
                /* Build candidate string with space→▁ substitution */
                char candidate[256];
                size_t candidate_len = 0;
                size_t j = 0;

                if (try_len > sizeof(candidate) - 1) continue;

                for (j = 0; j < try_len; ++j) {
                    if (start[j] == ' ') {
                        /* Space becomes ▁ (3 bytes UTF-8) */
                        if (candidate_len + 3 > sizeof(candidate) - 1) break;
                        candidate[candidate_len++] = (char)0xE2;
                        candidate[candidate_len++] = (char)0x96;
                        candidate[candidate_len++] = (char)0x81;
                    } else {
                        candidate[candidate_len++] = (char)start[j];
                    }
                }

                if (j < try_len) continue; /* candidate buffer overflow */

                {
                    int id = find_token_id(tokenizer, candidate, candidate_len);
                    if (id >= 0) {
                        piece_ids[n_pieces++] = id;
                        pos += try_len;
                        found = 1;
                        break;
                    }
                }
            }
        }

        if (!found) {
            /* Byte fallback: try <0xHH> token for this byte */
            int bf_id = byte_fallback_token(tokenizer, *pos);
            if (bf_id >= 0) {
                piece_ids[n_pieces++] = bf_id;
                ++pos;
            } else if (tokenizer->unk_token_id >= 0) {
                piece_ids[n_pieces++] = tokenizer->unk_token_id;
                ++pos;
            } else {
                /* Skip unrecognized byte */
                ++pos;
            }
        }
    }

    /* Step 2: BPE merge loop.
     * Repeatedly find the adjacent pair with the highest merge score
     * and merge them into a single token. */
    if (tokenizer->scores != NULL) {
        while (n_pieces > 1) {
            /* Find the best merge: look for adjacent pair (piece_ids[i], piece_ids[i+1])
             * where the concatenation of their token strings exists as a token in vocab,
             * and that merged token has the highest score. */
            int best_score_idx = -1;
            float best_score = 0.0f;
            int best_merged_id = -1;

            for (i = 0; i < n_pieces - 1; ++i) {
                /* Concatenate token strings for pieces i and i+1 */
                const char *str_a = tokenizer->tokens[piece_ids[i]];
                const char *str_b = tokenizer->tokens[piece_ids[i + 1]];
                size_t len_a = strlen(str_a);
                size_t len_b = strlen(str_b);
                char merged[512];

                if (len_a + len_b >= sizeof(merged)) continue;

                memcpy(merged, str_a, len_a);
                memcpy(merged + len_a, str_b, len_b);

                {
                    int merged_id = find_token_id(tokenizer, merged, len_a + len_b);
                    if (merged_id >= 0) {
                        float score = tokenizer->scores[merged_id];
                        if (best_score_idx < 0 || score > best_score) {
                            best_score_idx = i;
                            best_score = score;
                            best_merged_id = merged_id;
                        }
                    }
                }
            }

            if (best_score_idx < 0) {
                break; /* No more merges possible */
            }

            /* Apply the merge: replace pieces[best_score_idx] and pieces[best_score_idx+1]
             * with the merged token */
            piece_ids[best_score_idx] = best_merged_id;
            for (i = best_score_idx + 1; i < n_pieces - 1; ++i) {
                piece_ids[i] = piece_ids[i + 1];
            }
            --n_pieces;
        }
    }

    /* Step 3: Copy results to output */
    for (i = 0; i < n_pieces && n < max_tokens; ++i) {
        tokens[n++] = piece_ids[i];
    }

    free(piece_ids);
    free(spaced);
    return n;
}

int bitnet_tokenizer_decode(bitnet_tokenizer_t *tokenizer, int token, char *out, int out_size) {
    const char *raw = NULL;
    int raw_len = 0;
    int out_pos = 0;

    if (tokenizer == NULL || out == NULL || out_size <= 0) return -1;
    if (token < 0 || (size_t)token >= tokenizer->vocab_size) return -1;

    raw = tokenizer->tokens[token];
    if (raw == NULL) return -1;

    raw_len = token_str_len(raw);

    if (raw_len == 0) {
        if (out_size > 0) out[0] = '\0';
        return 0;
    }

    if (tokenizer->model == BITNET_TOKENIZER_MODEL_GPT2) {
        const unsigned char *src = (const unsigned char *)raw;
        const unsigned char *src_end = src + raw_len;

        if (raw[0] == '<' && raw[raw_len - 1] == '>') {
            out[0] = '\0';
            return 0;
        }

        while (src < src_end && out_pos < out_size - 1) {
            unsigned int cp = 0;
            int cp_len = 0;
            int byte_val = -1;

            if (utf8_decode_codepoint(src, src_end, &cp, &cp_len) != 0) {
                out[out_pos++] = (char)*src++;
                continue;
            }

            byte_val = gpt2_codepoint_to_byte(cp);
            if (byte_val >= 0) {
                out[out_pos++] = (char)byte_val;
            } else {
                for (int i = 0; i < cp_len && out_pos < out_size - 1; ++i) {
                    out[out_pos++] = (char)src[i];
                }
            }
            src += cp_len;
        }

        out[out_pos] = '\0';
        return out_pos;
    }

    /* Convert ▁ (U+2581, 0xE2 0x96 0x81) back to space during decode */
    {
        const unsigned char *src = (const unsigned char *)raw;
        const unsigned char *src_end = src + raw_len;

        while (src < src_end && out_pos < out_size - 1) {
            if (src + 2 < src_end && src[0] == 0xE2 && src[1] == 0x96 && src[2] == 0x81) {
                out[out_pos++] = ' ';
                src += 3;
            } else {
                out[out_pos++] = (char)*src;
                ++src;
            }
        }
    }

    out[out_pos] = '\0';
    return out_pos;
}

int bitnet_tokenizer_bos_id(const bitnet_tokenizer_t *tokenizer) {
    if (tokenizer == NULL) return -1;
    return tokenizer->bos_token_id;
}

int bitnet_tokenizer_eos_id(const bitnet_tokenizer_t *tokenizer) {
    if (tokenizer == NULL) return -1;
    return tokenizer->eos_token_id;
}

int bitnet_tokenizer_pad_id(const bitnet_tokenizer_t *tokenizer) {
    if (tokenizer == NULL) return -1;
    return tokenizer->pad_token_id;
}

int bitnet_tokenizer_add_bos(const bitnet_tokenizer_t *tokenizer) {
    if (tokenizer == NULL) return 0;
    return tokenizer->add_bos_token;
}
