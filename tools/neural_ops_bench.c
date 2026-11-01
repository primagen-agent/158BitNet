#define _POSIX_C_SOURCE 200809L
#include "neural_slots.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* Deterministic normalized math fixtures and actual model weights. This tool
 * measures NN operators only: no backbone forward, text, training or scoring. */
static double seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!frequency.QuadPart)
        QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}
static unsigned random_state = 8317;
static float random_value(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return ((float)(random_state & 65535) / 32767.5f - 1.f) / sqrtf(NS_HIDDEN);
}
static volatile float checksum;
static int dump(FILE *f, const float *p, size_t n) {
    return fwrite(p, sizeof(float), n, f) != n;
}
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: neural_ops_bench MODEL GGUF OUTPUT_JSON OUTPUT_FLOATS\n");
        return 2;
    }
    ns_model_t *m = ns_model_load(argv[1], argv[2]);
    FILE *json = fopen(argv[3], "w"), *out = fopen(argv[4], "wb");
    if (!m || !json || !out)
        return 3;
    const unsigned counts[] = {16, 64, 256};
    fprintf(json, "{\"backbone_forward\":false,\"fixture_seed\":8317,\"cases\":[");
    for (unsigned test = 0; test < 3; test++) {
        size_t n = counts[test];
        ns_features_t x = {.count = n, .word_count = n};
        x.rows = malloc(n * NS_INPUT * sizeof(float));
        x.word_rows = x.rows;
        if (!x.rows)
            return 4;
        for (size_t i = 0; i < n * NS_INPUT; i++)
            x.rows[i] = random_value();
        ns_slot_t slot = {0};
        float q[NS_QUERY], id[NS_IDENTITY], initial[NS_QUERY], advanced[NS_QUERY],
              scope_q[NS_QUERY], scope_id[NS_IDENTITY];
        float *begins = malloc(n * sizeof(float)), *ends = malloc(n * sizeof(float));
        if (!x.rows || !begins || !ends || ns_prepare(m, &x, &slot) || ns_query(m, &x, q, id) ||
            ns_scope_query(m, &x, scope_q, scope_id))
            return 4;
        const int repeats = 8, bank = 128;
        double elapsed[7];
        double t = seconds();
        for (int r = 0; r < repeats; r++)
            if (ns_prepare(m, &x, &slot))
                return 4;
        elapsed[0] = (seconds() - t) / repeats;
        t = seconds();
        for (int r = 0; r < repeats; r++)
            if (ns_query(m, &x, q, id))
                return 4;
        elapsed[1] = (seconds() - t) / repeats;
        t = seconds();
        for (int r = 0; r < repeats; r++)
            if (ns_plan_boundary_logits(m, slot.encoded, x.rows, n, begins, ends))
                return 4;
        elapsed[2] = (seconds() - t) / repeats;
        t = seconds();
        for (int r = 0; r < repeats; r++)
            for (int s = 0; s < bank; s++) {
                slot.status = s % 3 == 0 ? NS_CURRENT : (s % 3 == 1 ? NS_SUPERSEDED : NS_TENTATIVE);
                float score = ns_score(m, q, id, &slot, s, bank, 0);
                if (!isfinite(score))
                    return 4;
                checksum += score;
            }
        elapsed[3] = (seconds() - t) / repeats;
        for (int j = 0; j < NS_QUERY; j++)
            initial[j] = .5f * q[j] + .03f * (j % 5);
        t = seconds();
        for (int r = 0; r < repeats; r++) {
            memcpy(advanced, q, sizeof q);
            if (ns_advance_query(m, initial, advanced))
                return 4;
        }
        elapsed[4] = (seconds() - t) / repeats;
        t = seconds();
        for (int r = 0; r < repeats; r++)
            if (ns_scope_query(m, &x, scope_q, scope_id))
                return 4;
        elapsed[5] = (seconds() - t) / repeats;
        t = seconds();
        for (int r = 0; r < repeats; r++)
            for (int s = 0; s < bank; s++) {
                float score = ns_score(m, scope_q, scope_id, &slot, s, bank, 1);
                if (!isfinite(score))
                    return 4;
                checksum += score;
            }
        elapsed[6] = (seconds() - t) / repeats;
        slot.status = NS_CURRENT;
        float scores[2] = {ns_score(m, q, id, &slot, 0, bank, 0),
                           ns_score(m, scope_q, scope_id, &slot, 0, bank, 1)};
        if (dump(out, slot.encoded, n * NS_WIDTH) || dump(out, slot.identity, NS_IDENTITY) ||
            dump(out, q, NS_QUERY) || dump(out, id, NS_IDENTITY) || dump(out, begins, n) ||
            dump(out, ends, n) || dump(out, scores, 2) || dump(out, advanced, NS_QUERY) ||
            dump(out, scope_q, NS_QUERY) || dump(out, scope_id, NS_IDENTITY))
            return 5;
        fprintf(json, "%s{\"tokens\":%zu,\"word_rows\":%zu,\"bank_slots\":%d,"
                      "\"prepare_ms\":%.6f,\"query_ms\":%.6f,\"boundary_ms\":%.6f,"
                      "\"read_bank_ms\":%.6f,\"read_update_ms\":%.6f,"
                      "\"scope_query_ms\":%.6f,\"scope_bank_ms\":%.6f}",
                test ? "," : "", n, n, bank, elapsed[0] * 1000, elapsed[1] * 1000,
                elapsed[2] * 1000, elapsed[3] * 1000, elapsed[4] * 1000,
                elapsed[5] * 1000, elapsed[6] * 1000);
        free(slot.encoded);
        free(x.rows);
        free(begins);
        free(ends);
    }
    fprintf(json, "],\"checksum\":%.6f}\n", (double)checksum);
    fclose(json);
    fclose(out);
    ns_model_free(m);
    return 0;
}
