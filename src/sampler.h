#ifndef SAMPLER_H
#define SAMPLER_H

#ifdef __cplusplus
extern "C" {
#endif

int bitnet_sample_greedy_impl(int vocab_size, const float *logits);

#ifdef __cplusplus
}
#endif

#endif
