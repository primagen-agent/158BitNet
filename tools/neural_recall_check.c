/* Restore actual writer snapshots in a new process, then run model activation.
 * Transport fields are snapshot path, body, past and literal provenance only. */
#include "neural_slots.h"
#include <inttypes.h>
#include <math.h>
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
typedef struct {
    FILE *file;
    uint32_t query_index;
} trace_writer_t;

static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"' || *p == '\\')
            fprintf(f, "\\%c", *p);
        else if (*p < 0x20)
            fprintf(f, "\\u%04x", *p);
        else
            fputc(*p, f);
    }
    fputc('"', f);
}
/* Evaluation diagnostic: the stored text a bank actually holds. */
static int dump_bank(FILE *f, const ns_bank_t *bank) {
    for (size_t i = 0; i < bank->count; i++) {
        const ns_slot_t *s = bank->slots[i];
        fprintf(f, "{\"id\":%" PRIu64 ",\"status\":%d,\"content\":", s->id, (int)s->status);
        json_string(f, s->content);
        fputs(",\"referent\":", f);
        json_string(f, s->referent);
        fputs(",\"speaker\":", f);
        json_string(f, s->speaker);
        fputs(",\"observed_at\":", f);
        json_string(f, s->observed_at);
        fputs(",\"source_id\":", f);
        json_string(f, s->source_id);
        fputs("}\n", f);
    }
    return ferror(f) ? -1 : 0;
}
static uint64_t fingerprint(const float *values, size_t count) {
    const unsigned char *bytes = (const unsigned char *)values;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count * sizeof(float); i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
static int write_trace(const ns_round_trace_t *round, void *user) {
    trace_writer_t *writer = user;
    FILE *f = writer->file;
    if (fprintf(f, "{\"type\":\"round\",\"query_index\":%u,\"round\":%u,"
                   "\"set_converged\":%s,\"fingerprint_algorithm\":\"fnv1a64-float32-bytes\","
                   "\"query_fingerprint\":\"%016" PRIx64 "\","
                   "\"identity_fingerprint\":\"%016" PRIx64 "\","
                   "\"original_query_fingerprint\":\"%016" PRIx64 "\","
                   "\"original_identity_fingerprint\":\"%016" PRIx64 "\",\"slots\":[",
                writer->query_index, round->round, round->stable ? "true" : "false",
                fingerprint(round->query, NS_QUERY), fingerprint(round->identity, NS_IDENTITY),
                fingerprint(round->original_query, NS_QUERY),
                fingerprint(round->original_identity, NS_IDENTITY)) < 0)
        return -1;
    for (size_t i = 0; i < round->slot_count; i++) {
        const ns_slot_trace_t *slot = round->slots + i;
        const ns_score_trace_t *s = &slot->factors;
        const float values[] = {
            s->semantic_logit, s->identity_logit, s->version_logit,
            s->intent_logits[0], s->intent_logits[1], s->intent_logits[2],
            s->semantic_log_probability, s->identity_log_probability,
            s->version_log_probability, s->intent_log_probability,
            s->joint_log_probability, s->clamped_log_probability, s->score
        };
        for (size_t j = 0; j < sizeof values / sizeof *values; j++)
            if (!isfinite(values[j]))
                return -1;
        char head_json[512] = "";
        if (slot->has_head) {
            const float *h = slot->head_features;
            for (size_t j = 0; j < NS_READ_HEAD_FEATURES; j++)
                if (!isfinite(h[j]) || !isfinite(slot->head_logit))
                    return -1;
            int written = snprintf(
                head_json, sizeof head_json,
                ",\"head_logit\":%.9g,\"head_features\":["
                "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]",
                slot->head_logit, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8],
                h[9], h[10]);
            if (written < 0 || (size_t)written >= sizeof head_json)
                return -1;
        }
        if (fprintf(f, "%s{\"id\":%" PRIu64 ",\"status\":%u,"
                       "\"semantic_logit\":%.9g,\"identity_logit\":%.9g,\"version_logit\":%.9g,"
                       "\"intent_logits\":[%.9g,%.9g,%.9g],\"intent_index\":%u,"
                       "\"semantic_log_probability\":%.9g,\"identity_log_probability\":%.9g,"
                       "\"version_log_probability\":%.9g,\"intent_log_probability\":%.9g,"
                       "\"joint_log_probability\":%.9g,\"clamped_log_probability\":%.9g,"
                       "\"final_score\":%.9g%s}",
                    i ? "," : "", slot->id, (unsigned)slot->status,
                    s->semantic_logit, s->identity_logit, s->version_logit,
                    s->intent_logits[0], s->intent_logits[1], s->intent_logits[2], s->intent_index,
                    s->semantic_log_probability, s->identity_log_probability,
                    s->version_log_probability, s->intent_log_probability,
                    s->joint_log_probability, s->clamped_log_probability, s->score,
                    head_json) < 0)
            return -1;
    }
    if (fputs("],\"activated_ids\":[", f) == EOF)
        return -1;
    for (size_t i = 0; i < round->active_count; i++)
        if (fprintf(f, "%s%" PRIu64, i ? "," : "", round->active_ids[i]) < 0)
            return -1;
    return fputs("]}\n", f) == EOF || ferror(f) ? -1 : 0;
}
int main(int argc, char **argv) {
    const char *trace_path = NULL, *read_head_path = NULL, *dump_path = NULL;
    int force_activation = 0;
    unsigned rounds = 3;
    int idx = 5;
    if (idx < argc && argv[idx][0] != '-') {
        char *end = NULL;
        unsigned long requested = strtoul(argv[idx++], &end, 10);
        if (!requested || requested > 16 || (end && *end))
            return 2;
        rounds = (unsigned)requested;
    }
    while (idx < argc) {
        if (!strcmp(argv[idx], "--trace") && idx + 1 < argc)
            trace_path = argv[++idx];
        else if (!strcmp(argv[idx], "--read-head") && idx + 1 < argc)
            read_head_path = argv[++idx];
        else if (!strcmp(argv[idx], "--force-activation"))
            force_activation = 1;
        else if (!strcmp(argv[idx], "--dump-bank") && idx + 1 < argc)
            dump_path = argv[++idx];
        else
            return 2;
        idx++;
    }
    if (argc < 5) {
        fprintf(stderr, "Usage: %s MEMORY_MODEL BACKBONE INPUT OUTPUT [ROUNDS]"
                        " [--trace JSONL] [--read-head ARTIFACT] [--force-activation]"
                        " [--dump-bank JSONL]\n"
                        "--force-activation scores the bank even where the writer left the"
                        " reference unresolved; evaluation diagnostic only.\n"
                        "--dump-bank writes the first query's stored records; evaluation"
                        " diagnostic only.\n"
                        "Trace set_converged means stable activation IDs, not support completeness.\n",
                argv[0]);
        return 2;
    }
    for (int i = 1; i <= 4; i++)
        if ((trace_path && !strcmp(trace_path, argv[i])) ||
            (read_head_path && !strcmp(read_head_path, argv[i])))
            return 2;
    ns_model_t *model = ns_model_load(argv[1], argv[2]);
    bitnet_model_t *backbone = bitnet_load_model(argv[2]);
    ns_read_head_t *head = read_head_path ? ns_read_head_load(read_head_path, model) : NULL;
    FILE *in = fopen(argv[3], "rb"), *out = fopen(argv[4], "wb");
    FILE *trace = trace_path ? fopen(trace_path, "wb") : NULL;
    int rc = 1;
    uint32_t n;
    if (!model || !backbone || (read_head_path && !head) || !in || !out ||
        (trace_path && !trace) ||
        fread(&n, 4, 1, in) != 1 || n > 10000 ||
        fwrite(&n, 4, 1, out) != 1)
        goto done;
    for (uint32_t i = 0; i < n; i++) {
        char *path = text(in), *body = text(in), *past = text(in), *speaker = text(in),
             *timestamp = text(in), *source_id = text(in), *reference = NULL, *interaction = NULL,
             *prompt = NULL;
        ns_bank_t bank = {0};
        ns_activation_t activation = {0};
        int unresolved = 0,
            failed = !path || !body || !past || !speaker || !timestamp || !source_id;
        ns_observation_t observation = {.body = body,
                                        .past = past,
                                        .speaker = speaker,
                                        .timestamp = timestamp,
                                        .source_id = source_id};
        if (!failed)
            failed = ns_bank_load(&bank, model, path);
        if (!failed && dump_path && i == 0) {
            FILE *dump = fopen(dump_path, "wb");
            failed = !dump || dump_bank(dump, &bank);
            if (dump && fclose(dump))
                failed = 1;
        }
        if (!failed) {
            interaction =
                ns_observation_interaction(model, backbone, &observation, &reference, &unresolved);
            failed = !interaction;
        }
        if (!failed && (!unresolved || force_activation)) {
            if (trace) {
                trace_writer_t writer = {.file = trace, .query_index = i};
                failed = head ? ns_activate_context_trace_head(
                                    model, head, backbone, &bank, interaction, reference, rounds,
                                    &activation, write_trace, &writer)
                              : ns_activate_context_trace(model, backbone, &bank, interaction,
                                                          reference, rounds, &activation,
                                                          write_trace, &writer);
            } else
                failed = head ? ns_activate_context_head(model, head, backbone, &bank, interaction,
                                                         reference, rounds, &activation)
                              : ns_activate_context(model, backbone, &bank, interaction, reference,
                                                    rounds, &activation);
        }
        if (!failed) {
            prompt = ns_memory_prompt_context(&bank, &activation, NULL, body, NULL, reference);
            failed = !prompt;
        }
        if (!failed) {
            uint32_t fields[3] = {(uint32_t)activation.count, (uint32_t)unresolved,
                                  (uint32_t)(unresolved || activation.complete)};
            failed =
                fwrite(fields, 4, 3, out) != 3 || string(out, reference) || string(out, prompt);
            for (size_t j = 0; !failed && j < activation.count; j++)
                failed = fwrite(&activation.ids[j], 8, 1, out) != 1;
        }
        if (!failed && trace)
            failed = fprintf(trace, "{\"type\":\"query_end\",\"query_index\":%u,"
                                    "\"unresolved\":%s,\"rounds\":%u,\"set_converged\":%s,"
                                    "\"convergence_meaning\":\"stable activation IDs, not support completeness\"}\n",
                             i, unresolved ? "true" : "false", activation.rounds,
                             activation.complete ? "true" : "false") < 0 || ferror(trace);
        free(path);
        free(body);
        free(past);
        free(speaker);
        free(timestamp);
        free(source_id);
        free(reference);
        free(interaction);
        free(prompt);
        ns_bank_clear(&bank);
        ns_activation_clear(&activation);
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
    if (trace && fclose(trace))
        rc = 1;
    ns_model_free(model);
    bitnet_free_model(backbone);
    ns_read_head_free(head);
    return rc;
}
