#include "bitnet.h"

#include <math.h>
#include <stdio.h>

#ifndef BITNET_SOURCE_DIR
#define BITNET_SOURCE_DIR "."
#endif

static int fail(const char *msg) {
    fprintf(stderr, "FAIL %s\n", msg);
    return 1;
}

static int close_float(float a, float b) {
    return fabsf(a - b) < 1e-5f;
}

int main(void) {
    static const char model_path[] = BITNET_SOURCE_DIR "/models/bitcpm4-1b-tq2_0.gguf";
    bitnet_model_t *model = NULL;
    int eos = -1;
    int pad = -1;

    {
        float logits[] = {0.0f, 6.0f, 5.0f, -2.0f, 4.0f};
        int recent[] = {1, 3, 1, -1, 99};
        bitnet_apply_repetition_penalty(logits, 5, recent, 5, 2.0f);
        if (!close_float(logits[1], 3.0f)) return fail("positive logit penalty");
        if (!close_float(logits[3], -4.0f)) return fail("negative logit penalty");
        if (!close_float(logits[2], 5.0f)) return fail("untouched logit");
    }

    {
        float logits[] = {0.0f, 6.0f};
        int recent[] = {1};
        bitnet_apply_repetition_penalty(logits, 2, recent, 1, 1.0f);
        if (!close_float(logits[1], 6.0f)) return fail("penalty one unchanged");
    }

    model = bitnet_load_model(model_path);
    if (model == NULL) return fail("load model");

    eos = bitnet_eos_token(model);
    pad = bitnet_pad_token(model);
    if (eos < 0) return fail("eos token");
    if (pad < 0) return fail("pad token");
    if (!bitnet_token_is_eog(model, eos)) return fail("eos is eog");
    if (!bitnet_token_is_eog(model, pad)) return fail("pad is eog");
    if (bitnet_token_is_eog(model, 1)) return fail("bos is not eog");

    bitnet_free_model(model);
    return 0;
}
