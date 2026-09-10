#include "metis/retrieval_index.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    metis_retrieval_index_t index;
    metis_memory_retriever_t retriever;
    const char *records[2] = {
        "Alice moved to Kyoto.", "Bob likes jazz."};
    const float priorities[2] = {1.0f, 0.0f};
    float global0[2] = {1, 0}, global1[2] = {0, 1};
    float token0[2] = {1, 0}, token1[2] = {0, 1};
    float query_global[2] = {1, 0}, query_tokens[2] = {1, 0};
    size_t ranked[2];
    memset(&retriever, 0, sizeof retriever);
    retriever.rank = 2;
    retriever.late_weight = 0.5f;
    retriever.max_residual = 0.49f;
    metis_retrieval_index_init(&index);
    if (metis_retrieval_index_configure(&index, 2) ||
        metis_retrieval_index_set(&index, 0, global0, token0, 1) ||
        metis_retrieval_index_set(&index, 1, global1, token1, 1) ||
        !metis_retrieval_index_complete(&index, 2) ||
        metis_retrieval_index_rank(
            &index, &retriever, records, priorities, 2,
            "Where did Alice move?", 0.25f,
            query_global, query_tokens, 1, ranked, 2) != 2 ||
        ranked[0] != 0) {
        metis_retrieval_index_clear(&index);
        return 1;
    }
    {
        const char *ties[2] = {
            "Alice likes tea.", "Alice likes jazz."};
        float same_global[2] = {1, 0};
        float same_token[2] = {1, 0};
        if (metis_retrieval_index_set(
                &index, 1, same_global, same_token, 1) ||
            metis_retrieval_index_rank(
                &index, &retriever, ties, priorities, 2,
                "What does Alice like?", 0.25f,
                query_global, query_tokens, 1, ranked, 2) != 2 ||
            ranked[0] != 0) {
            metis_retrieval_index_clear(&index);
            return 1;
        }
    }
    metis_retrieval_index_clear(&index);
    puts("test_retrieval_index: OK");
    return 0;
}
