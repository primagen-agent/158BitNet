#include "retrieval_index.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct ranked_record {
    float score;
    size_t index;
} ranked_record_t;

static int compare_desc(const void *left, const void *right) {
    const ranked_record_t *a = left;
    const ranked_record_t *b = right;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->index > b->index ? -1 : (a->index < b->index ? 1 : 0);
}

static int stop_word(const char *word) {
    static const char *const words[] = {
        "the","a","an","is","what","which","about","memory","evidence",
        "retrieve","regarding","for","and","does","conversation","say",
        "associated","with","at","point","did","source","subject",
        "predicate","value","time","question","answer","using","only",
        "this","that"};
    for (size_t i = 0; i < sizeof words / sizeof words[0]; ++i)
        if (strcmp(word, words[i]) == 0) return 1;
    return 0;
}

static int contains_word(const char *text, const char *word) {
    size_t wanted = strlen(word);
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        while (*p && !isalnum(*p)) ++p;
        const unsigned char *start = p;
        while (*p && isalnum(*p)) ++p;
        if ((size_t)(p - start) == wanted) {
            size_t i;
            for (i = 0; i < wanted; ++i)
                if (tolower(start[i]) != (unsigned char)word[i]) break;
            if (i == wanted) return 1;
        }
    }
    return 0;
}

static int lexical_overlap(const char *query, const char *record) {
    const unsigned char *p = (const unsigned char *)query;
    char seen[64][64];
    int seen_count = 0, score = 0;
    while (*p && seen_count < 64) {
        while (*p && !isalnum(*p)) ++p;
        const unsigned char *start = p;
        while (*p && isalnum(*p)) ++p;
        size_t length = (size_t)(p - start);
        if (length < 2 || length >= 64) continue;
        char word[64];
        for (size_t i = 0; i < length; ++i)
            word[i] = (char)tolower(start[i]);
        word[length] = '\0';
        if (stop_word(word)) continue;
        int duplicate = 0;
        for (int i = 0; i < seen_count; ++i)
            duplicate |= strcmp(seen[i], word) == 0;
        if (duplicate) continue;
        memcpy(seen[seen_count++], word, length + 1);
        score += contains_word(record, word);
    }
    return score;
}

void metis_retrieval_index_init(metis_retrieval_index_t *index) {
    if (index) memset(index, 0, sizeof *index);
}

void metis_retrieval_index_clear(metis_retrieval_index_t *index) {
    if (!index) return;
    for (size_t i = 0; i < index->capacity; ++i) {
        free(index->global_keys ? index->global_keys[i] : NULL);
        free(index->token_keys ? index->token_keys[i] : NULL);
    }
    free(index->token_counts);
    free(index->global_keys);
    free(index->token_keys);
    memset(index, 0, sizeof *index);
}

int metis_retrieval_index_configure(
    metis_retrieval_index_t *index, int rank) {
    if (!index || rank < 1 || rank > 1024) return -1;
    if (index->count && index->rank != rank) return -1;
    index->rank = rank;
    return 0;
}

static int reserve(metis_retrieval_index_t *index, size_t needed) {
    if (needed <= index->capacity) return 0;
    size_t capacity = index->capacity ? index->capacity : 16;
    while (capacity < needed) capacity *= 2;
    size_t *counts = calloc(capacity, sizeof *counts);
    float **globals = calloc(capacity, sizeof *globals);
    float **tokens = calloc(capacity, sizeof *tokens);
    if (!counts || !globals || !tokens) {
        free(counts); free(globals); free(tokens); return -1;
    }
    if (index->capacity) {
        memcpy(counts, index->token_counts,
               index->capacity * sizeof *counts);
        memcpy(globals, index->global_keys,
               index->capacity * sizeof *globals);
        memcpy(tokens, index->token_keys,
               index->capacity * sizeof *tokens);
    }
    free(index->token_counts);
    free(index->global_keys);
    free(index->token_keys);
    index->token_counts = counts;
    index->global_keys = globals;
    index->token_keys = tokens;
    index->capacity = capacity;
    return 0;
}

int metis_retrieval_index_set(
    metis_retrieval_index_t *index, size_t record_index,
    const float *global_key, const float *token_keys, size_t token_count) {
    size_t global_bytes, token_bytes;
    float *global_copy, *token_copy;
    if (!index || index->rank < 1 || !global_key || !token_keys ||
        token_count < 1 || token_count > 512 ||
        reserve(index, record_index + 1) != 0) return -1;
    global_bytes = (size_t)index->rank * sizeof(float);
    token_bytes = token_count * (size_t)index->rank * sizeof(float);
    global_copy = malloc(global_bytes);
    token_copy = malloc(token_bytes);
    if (!global_copy || !token_copy) {
        free(global_copy); free(token_copy); return -1;
    }
    memcpy(global_copy, global_key, global_bytes);
    memcpy(token_copy, token_keys, token_bytes);
    free(index->global_keys[record_index]);
    free(index->token_keys[record_index]);
    index->global_keys[record_index] = global_copy;
    index->token_keys[record_index] = token_copy;
    index->token_counts[record_index] = token_count;
    if (index->count <= record_index) index->count = record_index + 1;
    return 0;
}

int metis_retrieval_index_complete(
    const metis_retrieval_index_t *index, size_t record_count) {
    if (!index || index->rank < 1 || index->count < record_count) return 0;
    for (size_t i = 0; i < record_count; ++i)
        if (!index->global_keys[i] || !index->token_keys[i] ||
            !index->token_counts[i]) return 0;
    return 1;
}

int metis_retrieval_index_rank(
    const metis_retrieval_index_t *index,
    const metis_memory_retriever_t *retriever,
    const char *const *records, const float *priorities,
    size_t record_count, const char *query, float priority_weight,
    const float *query_global, const float *query_tokens,
    size_t query_token_count, size_t *indices, int max_results) {
    ranked_record_t *ranked;
    if (!retriever || !records || !query || !query_global || !query_tokens ||
        !query_token_count || !indices || max_results < 1 ||
        !isfinite(priority_weight) ||
        priority_weight < 0.0f || priority_weight >= 0.5f ||
        !metis_retrieval_index_complete(index, record_count)) return -1;
    if ((size_t)max_results > record_count) max_results = (int)record_count;
    ranked = calloc(record_count, sizeof *ranked);
    if (!ranked) return -1;
    for (size_t i = 0; i < record_count; ++i) {
        float semantic = metis_memory_retriever_score(
            retriever, query_global, query_tokens, query_token_count,
            index->global_keys[i], index->token_keys[i],
            index->token_counts[i]);
        float residual = metis_memory_retriever_residual(
            retriever, semantic);
        residual += priority_weight * (
            priorities == NULL ? 0.0f : priorities[i]);
        residual = fminf(
            retriever->max_residual,
            fmaxf(-retriever->max_residual, residual));
        ranked[i].score =
            (float)lexical_overlap(query, records[i]) + residual;
        ranked[i].index = i;
    }
    qsort(ranked, record_count, sizeof *ranked, compare_desc);
    for (int i = 0; i < max_results; ++i) indices[i] = ranked[i].index;
    free(ranked);
    return max_results;
}
