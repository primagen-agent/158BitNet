#include "sampler.h"

int bitnet_sample_greedy_impl(int vocab_size, const float *logits) {
    int i = 0;
    int best = 0;
    float best_score = logits[0];

    for (i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best = i;
        }
    }

    return best;
}
