#include "neural_slots.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Run learned heads on a frozen feature batch for C/PyTorch parity. */
int main(int argc, char **argv) {
    if (argc != 6)
        return 2;
    ns_model_t *m = ns_model_load(argv[1], argv[2]);
    if (!m)
        return 3;
    FILE *in = fopen(argv[3], "rb"), *out = fopen(argv[4], "wb");
    FILE *sources = fopen(argv[5], "rb");
    uint32_t source_count = 0;
    int rc = 1;
    if (!sources || fread(&source_count, 4, 1, sources) != 1)
        goto done;
    char magic[8];
    uint32_t count;
    if (!in || !out || fread(magic, 1, 8, in) != 8 || memcmp(magic, "BNSFEAT1", 8) ||
        fread(&count, 4, 1, in) != 1 || !count || count > 10000)
        goto done;
    if (source_count != count)
        goto done;
    if (fwrite("BNSCHECK", 1, 8, out) != 8 || fwrite(&count, 4, 1, out) != 1)
        goto done;
    for (uint32_t r = 0; r < count; r++) {
        uint32_t n;
        if (fread(&n, 4, 1, in) != 1 || !n || n >= NS_TOKEN_LIMIT)
            goto done;
        ns_features_t x = {0};
        x.count = n;
        x.rows = malloc((size_t)n * NS_INPUT * sizeof(float));
        float q[NS_QUERY], id[NS_IDENTITY], scope_q[NS_QUERY], scope_id[NS_IDENTITY];
        ns_slot_t slot = {0};
        x.starts = malloc(n * sizeof *x.starts);
        x.ends = malloc(n * sizeof *x.ends);
        if (!x.starts || !x.ends) {
            ns_features_clear(&x);
            goto done;
        }
        for (uint32_t t = 0; t < n; t++) {
            uint32_t offsets[2];
            if (fread(offsets, 4, 2, in) != 2) {
                ns_features_clear(&x);
                goto done;
            }
            x.starts[t] = offsets[0];
            x.ends[t] = offsets[1];
        }
        char *text = NULL;
        if (!x.rows || fread(x.rows, NS_INPUT * sizeof(float), n, in) != n) {
            ns_features_clear(&x);
            goto done;
        }
        if (sources) {
            uint32_t bytes;
            if (fread(&bytes, 4, 1, sources) != 1 || bytes > 1024 * 1024) {
                ns_features_clear(&x);
                goto done;
            }
            text = calloc((size_t)bytes + 1, 1);
            if (!text || fread(text, 1, bytes, sources) != bytes || ns_features_words(&x, text)) {
                free(text);
                ns_features_clear(&x);
                goto done;
            }
        }
        if (ns_prepare(m, &x, &slot) || ns_query(m, &x, q, id) ||
            ns_scope_query(m, &x, scope_q, scope_id)) {
            free(text);
            free(slot.encoded);
            ns_features_clear(&x);
            goto done;
        }
        int kind = ns_predict_kind(m, &x);
        slot.status = NS_CURRENT;
        float scores[2] = {ns_score(m, q, id, &slot, 0, 1, 0),
                           ns_score(m, scope_q, scope_id, &slot, 0, 1, 1)};
        int failed = fwrite(&n, 4, 1, out) != 1 || fwrite(&kind, 4, 1, out) != 1 ||
                     fwrite(slot.encoded, NS_WIDTH * sizeof(float), n, out) != n ||
                     fwrite(slot.identity, 4, NS_IDENTITY, out) != NS_IDENTITY ||
                     fwrite(q, 4, NS_QUERY, out) != NS_QUERY ||
                     fwrite(id, 4, NS_IDENTITY, out) != NS_IDENTITY ||
                     fwrite(scores, 4, 2, out) != 2;
        if (!failed) {
            float initial[NS_QUERY], current[NS_QUERY];
            memcpy(current, q, sizeof current);
            for (int j = 0; j < NS_QUERY; j++)
                initial[j] = q[j] * .5f + .03f * (j % 5);
            failed = ns_advance_query(m, initial, current) ||
                     fwrite(current, 4, NS_QUERY, out) != NS_QUERY;
            float original_identity[NS_IDENTITY], current_identity[NS_IDENTITY];
            for (int j = 0; j < NS_IDENTITY; j++)
                original_identity[j] = id[j] * .5f + .01f * ((j % 7) - 3);
            for (int block = 0; block < 2; block++) {
                int first = block ? NS_LEXICAL : 0, count = block ? NS_WIDTH : NS_LEXICAL;
                float norm = 0;
                for (int j = first; j < first + count; j++)
                    norm += original_identity[j] * original_identity[j];
                norm = fmaxf(sqrtf(norm), 1e-12f);
                for (int j = first; j < first + count; j++)
                    original_identity[j] /= norm;
            }
            memcpy(current_identity, id, sizeof current_identity);
            if (!failed)
                failed = ns_advance_identity(m, initial, q, original_identity, current_identity) ||
                         fwrite(current_identity, 4, NS_IDENTITY, out) != NS_IDENTITY;
        }
        if (!failed) {
            float *begins = malloc(n * sizeof(float)), *ends = malloc(n * sizeof(float)),
                  *membership = malloc(n * sizeof(float)),
                  *body = malloc((size_t)n * NS_WIDTH * sizeof(float)),
                  *context = malloc(n * sizeof(float));
            if (body)
                for (size_t j = 0; j < (size_t)n * NS_WIDTH; j++)
                    body[j] = slot.encoded[j] * .5f + .03f * (j % 7);
            failed = !begins || !ends || !membership || !body || !context ||
                     ns_plan_source_logits(m, slot.encoded, x.rows, n, begins, ends, membership) ||
                     ns_context_membership_logits(m, slot.encoded, body, x.rows, n, context) ||
                     fwrite(begins, 4, n, out) != n || fwrite(ends, 4, n, out) != n ||
                     fwrite(membership, 4, n, out) != n || fwrite(context, 4, n, out) != n;
            free(begins);
            free(ends);
            free(membership);
            free(body);
            free(context);
        }
        if (!failed) {
            float kind_logits[3];
            failed = fwrite(scope_q, 4, NS_QUERY, out) != NS_QUERY ||
                     fwrite(scope_id, 4, NS_IDENTITY, out) != NS_IDENTITY ||
                     ns_kind_logits(m, &x, kind_logits) || fwrite(kind_logits, 4, 3, out) != 3 ||
                     fwrite(slot.read_context, 4, NS_READ_CONTEXT_WIDTH, out) != NS_READ_CONTEXT_WIDTH;
        }
        free(text);
        free(slot.encoded);
        ns_features_clear(&x);
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
    if (sources)
        fclose(sources);
    ns_model_free(m);
    return rc;
}
