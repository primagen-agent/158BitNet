/* Batch writer transport for evaluation. Only current/context texts enter the
 * model. Uses the serving writer verbatim; no QA, gold address or action input. */
#include "neural_slots.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int read_u32(FILE *f, uint32_t *value) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4)
        return -1;
    *value = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return 0;
}
static char *read_text(FILE *f) {
    uint32_t bytes;
    if (read_u32(f, &bytes) || bytes > 1024 * 1024)
        return NULL;
    char *text = calloc((size_t)bytes + 1, 1);
    if (!text || fread(text, 1, bytes, f) != bytes || memchr(text, 0, bytes)) {
        free(text);
        return NULL;
    }
    return text;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "Usage: neural_slot_ingest GGUF BNSMODEL INPUT SNAPSHOT [--independent]\n");
        return 2;
    }
    int independent = argc == 6 && !strcmp(argv[5], "--independent");
    if (argc == 6 && !independent)
        return 2;
    FILE *input = fopen(argv[3], "rb");
    char magic[8];
    uint32_t count;
    if (!input || fread(magic, 1, 8, input) != 8 || memcmp(magic, "BNSOBS01", 8) ||
        read_u32(input, &count) || count > 100000) {
        if (input)
            fclose(input);
        return 3;
    }
    /* Refuse accidental continuation of a partial bank. The evaluator owns a
     * fresh output directory for each attempt. */
    FILE *existing = fopen(argv[4], "rb");
    if (existing) {
        fclose(existing);
        fclose(input);
        return 4;
    }
    bitnet_model_t *backbone = bitnet_load_model(argv[1]);
    ns_model_t *model = ns_model_load(argv[2], argv[1]);
    ns_bank_t bank = {0};
    int rc = 1;
    if (!backbone || !model)
        goto done;
    for (uint32_t i = 0; i < count; i++) {
        if (independent)
            ns_bank_clear(&bank);
        char *current = read_text(input), *past = read_text(input);
        char *speaker = read_text(input);
        char *timestamp = read_text(input);
        char *source_id = read_text(input);
        if (!current || !past || !*current || (!speaker || !timestamp || !source_id)) {
            free(current);
            free(past);
            free(speaker);
            free(timestamp);
            free(source_id);
            goto done;
        }
        clock_t began = clock();
        ns_write_result_t result = {0};
        char *target = NULL;
        if (independent) {
            size_t n = strlen(argv[4]) + 40;
            target = malloc(n);
            if (target)
                snprintf(target, n, "%s.%u.bnslots", argv[4], i);
        }
        if (independent && !target) {
            free(current);
            free(past);
            free(speaker);
            free(timestamp);
            free(source_id);
            goto done;
        }
        ns_observation_t observation = {.body = current,
                                        .past = past,
                                        .speaker = speaker,
                                        .timestamp = timestamp,
                                        .source_id = source_id};
        int failed = ns_ingest_observation(model, backbone, &bank, &observation,
                                           target ? target : argv[4], &result);
        if (independent && !failed && ns_bank_save(&bank, model, target))
            failed = 1;
        free(target);
        free(current);
        free(past);
        free(speaker);
        free(timestamp);
        free(source_id);
        printf("{\"index\":%u,\"failed\":%s,\"action\":%d,\"written\":%zu,\"updated\":%zu,"
               "\"deleted\":%zu,\"unresolved_reference\":%d,\"records\":%zu,\"cpu_s\":%.6f,"
               "\"candidates\":%zu}\n",
               i, failed ? "true" : "false", result.action, result.written, result.updated,
               result.deleted, result.unresolved, bank.count,
               (double)(clock() - began) / CLOCKS_PER_SEC, result.candidates);
        fflush(stdout);
    }
    if (fgetc(input) != EOF || ferror(input) || ns_bank_save(&bank, model, argv[4]))
        goto done;
    rc = 0;
done:
    ns_bank_clear(&bank);
    ns_model_free(model);
    bitnet_free_model(backbone);
    fclose(input);
    return rc;
}
