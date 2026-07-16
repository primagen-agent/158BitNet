#include "bitnet.h"

#include <pthread.h>
#include <stdio.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

typedef struct eval_job {
    bitnet_context_t *ctx;
    const int *tokens;
    int n_tokens;
    int rc;
    int next_token;
} eval_job_t;

static void *run_eval(void *opaque) {
    eval_job_t *job = (eval_job_t *)opaque;
    job->rc = bitnet_eval(job->ctx, job->tokens, job->n_tokens);
    job->next_token = job->rc == 0 ? bitnet_sample_greedy(job->ctx) : -1;
    return NULL;
}

int main(void) {
    static const char path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = bitnet_load_model(path);
    bitnet_context_t *ctx_a = NULL;
    bitnet_context_t *ctx_b = NULL;
    pthread_t a, b;
    int tokens[32];
    int n_tokens;
    eval_job_t job_a, job_b;

    if (model == NULL) return 1;
    ctx_a = bitnet_create_context(model, 64);
    ctx_b = bitnet_create_context(model, 64);
    if (ctx_a == NULL || ctx_b == NULL) return 2;
    n_tokens = bitnet_tokenize(model, "The capital of France is", tokens, 32);
    if (n_tokens <= 0) return 3;
    job_a = (eval_job_t){ctx_a, tokens, n_tokens, -1, -1};
    job_b = (eval_job_t){ctx_b, tokens, n_tokens, -1, -1};
    if (pthread_create(&a, NULL, run_eval, &job_a) != 0 ||
        pthread_create(&b, NULL, run_eval, &job_b) != 0) return 4;
    (void)pthread_join(a, NULL);
    (void)pthread_join(b, NULL);
    if (job_a.rc != 0 || job_b.rc != 0 || job_a.next_token != job_b.next_token) return 5;

    bitnet_free_context(ctx_a);
    bitnet_free_context(ctx_b);
    bitnet_free_model(model);
    return 0;
}
