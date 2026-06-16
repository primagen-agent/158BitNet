#include "tokenizer.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct bitnet_tokenizer {
    size_t vocab_size;
    char **tokens;
    float *scores;
    int *token_types;
    int bos_token_id;
    int eos_token_id;
    int unk_token_id;
    int pad_token_id;
    int add_bos_token;
};

int bitnet_tokenizer_load(bitnet_tokenizer_t **out, const char *gguf_path) {
    gguf_file_t file;
    const gguf_metadata_t *tokens_meta = NULL;
    const gguf_metadata_t *scores_meta = NULL;
    const gguf_metadata_t *types_meta = NULL;
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

    /* Steal tokens string array */
    tokens_mutable = (gguf_metadata_t *)tokens_meta;
    tokenizer->tokens = tokens_mutable->array_strings;
    tokens_mutable->array_strings = NULL;

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
    size_t i = 0;
    for (i = 0; i < tokenizer->vocab_size; ++i) {
        const char *tok = tokenizer->tokens[i];
        if (tok == NULL) continue;
        size_t tok_len = strlen(tok);
        if (tok_len == len && memcmp(tok, str, len) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/* Check if a byte value has a <0xHH> fallback token. Returns token ID or -1. */
static int byte_fallback_token(const bitnet_tokenizer_t *tokenizer, uint8_t byte_val) {
    char buf[5];
    (void)snprintf(buf, sizeof(buf), "<0x%02X>", byte_val);
    return find_token_id(tokenizer, buf, 4);
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
