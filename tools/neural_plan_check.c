/* Native joint-plan transport. Input contains body and literal provenance only. */
#include "neural_slots.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char *text(FILE *f) {
    uint32_t n;
    if (fread(&n, 4, 1, f) != 1 || n > 1024 * 1024)
        return NULL;
    char *s = calloc((size_t)n + 1, 1);
    if (!s || fread(s, 1, n, f) != n || memchr(s, 0, n)) {
        free(s);
        return NULL;
    }
    return s;
}
static int string(FILE *f, const char *s) {
    uint32_t n = s ? (uint32_t)strlen(s) : 0;
    return fwrite(&n, 4, 1, f) != 1 || (n && fwrite(s, 1, n, f) != n);
}
int main(int argc, char **argv) {
    if (argc != 5)
        return 2;
    ns_model_t *model = ns_model_load(argv[1], argv[2]);
    bitnet_model_t *backbone = bitnet_load_model(argv[2]);
    FILE *in = fopen(argv[3], "rb"), *out = fopen(argv[4], "wb");
    int rc = 1;
    uint32_t n;
    if (!model || !backbone || !in || !out || fread(&n, 4, 1, in) != 1 || n > 10000 ||
        fwrite(&n, 4, 1, out) != 1)
        goto done;
    for (uint32_t i = 0; i < n; i++) {
        char *body = text(in), *past = text(in), *speaker = text(in), *timestamp = text(in),
             *source_id = text(in);
        ns_observation_t observation = {.body = body,
                                        .past = past,
                                        .speaker = speaker,
                                        .timestamp = timestamp,
                                        .source_id = source_id};
        ns_write_plan_t plan = {0};
        int failed = !body || !past || !speaker || !timestamp || !source_id ||
                     ns_plan_observation(model, backbone, &observation, &plan);
        if (!failed) {
            uint32_t count = (uint32_t)plan.count,
                     unresolved = (uint32_t)plan.interaction_unresolved;
            failed = fwrite(&count, 4, 1, out) != 1 || fwrite(&unresolved, 4, 1, out) != 1 ||
                     string(out, plan.interaction_reference);
            for (size_t j = 0; !failed && j < plan.count; j++) {
                const ns_plan_item_t *item = plan.items + j;
                uint32_t fields[4] = {(uint32_t)item->span.start, (uint32_t)item->span.end,
                                      (uint32_t)item->action, (uint32_t)item->unresolved};
                failed = fwrite(fields, 4, 4, out) != 4 || string(out, item->reference);
            }
        }
        ns_write_plan_clear(&plan);
        free(body);
        free(past);
        free(speaker);
        free(timestamp);
        free(source_id);
        if (failed)
            goto done;
    }
    if (fgetc(in) != EOF || ferror(in))
        goto done;
    rc = 0;
done:
    if (in)
        fclose(in);
    if (out && fclose(out))
        rc = 1;
    ns_model_free(model);
    bitnet_free_model(backbone);
    return rc;
}
