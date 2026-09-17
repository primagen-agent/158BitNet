#include "metis/resident_identity.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float worst;
static int compare(FILE *f, const float *got, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float expected;
        if (fread(&expected, 4, 1, f) != 1)
            return -1;
        float error = fabsf(expected - got[i]);
        worst = fmaxf(worst, error);
        if (!isfinite(error) || error > 0.002f) {
            fprintf(stderr, "parity mismatch %zu: %.9g vs %.9g\n", i, got[i], expected);
            return -1;
        }
    }
    return 0;
}
int main(int argc, char **argv) {
    if (argc != 4)
        return 2;
    resident_model_t *m = resident_model_load(argv[1], argv[2]);
    if (!m)
        return 3;
    FILE *f = fopen(argv[3], "rb");
    char magic[8];
    uint32_t worlds;
    if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "RSPAR001", 8) ||
        fread(&worlds, 4, 1, f) != 1 || worlds > 32)
        return 4;
    for (uint32_t w = 0; w < worlds; w++) {
        resident_state_t state = {0};
        uint32_t n, nq;
        if (fread(&n, 4, 1, f) != 1 || n > RESIDENT_MAX_EVENTS || fread(&nq, 4, 1, f) != 1 ||
            nq > 128)
            return 5;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t t;
            if (fread(&t, 4, 1, f) != 1 || t < 2 || t > 128)
                return 5;
            float *x = malloc(t * 2048 * sizeof(float));
            if (!x || fread(x, 2048 * sizeof(float), t, f) != t)
                return 5;
            if (resident_write(m, x, t, &state.slots[i]))
                return 6;
            state.count++;
            free(x);
            if (compare(f, state.slots[i].data, t * RESIDENT_STRIDE))
                return 7;
        }
        for (uint32_t i = 0; i < nq; i++) {
            uint32_t t;
            if (fread(&t, 4, 1, f) != 1 || t < 2 || t > 128)
                return 5;
            float *x = malloc(t * 2048 * sizeof(float));
            float scores[32], counts[5];
            if (!x || fread(x, 2048 * sizeof(float), t, f) != t)
                return 5;
            if (resident_read(m, &state, x, t, scores, counts) || compare(f, scores, n) ||
                compare(f, counts, 5))
                return 8;
            free(x);
        }
        resident_state_clear(&state);
    }
    if (fgetc(f) != EOF)
        return 9;
    fclose(f);
    resident_model_free(m);
    printf("resident parity passed: max_abs_error=%.9g\n", worst);
    return 0;
}
