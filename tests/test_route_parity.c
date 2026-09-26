/* test_route_parity.c — the C route computation must reproduce the trained
 * FineSpanReader's logits (span-based evidence) from raw texts + token pieces.
 *
 * Samples are exported by python/export_route_parity_samples.py into
 * build/neural-memory-route-parity/samples.bin, together with the live
 * tokenizer's pieces, so the span reconstruction runs without a tokenizer.
 * Skips (exit 0) when samples or the bnmodel are absent — GGUF-less CI stays
 * green; regenerate samples after any route-head change and run locally.
 */
#include "neural_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; \
} } while (0)

static const char *SAMPLES = "neural-memory-route-parity/samples.bin";
static const char *MODEL = "neural-c-weights/model.bnmodel";

static unsigned char *slurp(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)size + 1);
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) { free(buf); fclose(f); return NULL; }
    fclose(f);
    buf[size] = 0;
    *size_out = (size_t)size;
    return buf;
}

int main(void) {
    setbuf(stdout, NULL);
    size_t blob_size;
    unsigned char *blob = slurp(SAMPLES, &blob_size);
    if (!blob) {
        FILE *probe = fopen(MODEL, "rb");
        if (probe) fclose(probe);
        printf("test_route_parity: SKIP (%s missing; run "
               "python/export_route_parity_samples.py)\n", SAMPLES);
        return 0;
    }
    nm_ctx_t *ctx = nm_init(MODEL);
    if (!ctx) { printf("test_route_parity: SKIP (%s missing)\n", MODEL); free(blob); return 0; }

    unsigned char *p = blob;
    CHECK(memcmp(p, "NMRP", 4) == 0); p += 4;
    int version, n_records;
    memcpy(&version, p, 4); p += 4;
    memcpy(&n_records, p, 4); p += 4;
    CHECK(version == 1);
    CHECK(n_records > 0 && n_records < 64);

    for (int rec = 0; rec < n_records; rec++) {
        int nq, ns;
        memcpy(&nq, p, 4); p += 4;
        float *qf = malloc((size_t)nq * 2048 * 4);
        memcpy(qf, p, (size_t)nq * 2048 * 4); p += (size_t)nq * 2048 * 4;
        memcpy(&ns, p, 4); p += 4;
        float *sf = malloc((size_t)ns * 2048 * 4);
        memcpy(sf, p, (size_t)ns * 2048 * 4); p += (size_t)ns * 2048 * 4;

        int qrlen; memcpy(&qrlen, p, 4); p += 4;
        char question[1024];
        CHECK(qrlen < (int)sizeof question);
        memcpy(question, p, (size_t)qrlen); question[qrlen] = 0; p += qrlen;

        int qlen; memcpy(&qlen, p, 4); p += 4;
        char query_framed[2048];
        CHECK(qlen < (int)sizeof query_framed);
        memcpy(query_framed, p, (size_t)qlen); query_framed[qlen] = 0; p += qlen;

        /* query framing parity: training frames the query as an ARRAY */
        char qframe[2100];
        nm_frame_query(question, qframe, sizeof qframe);
        if (strcmp(qframe, query_framed) != 0) {
            printf("FAIL record %d query framing:\n  C:      %s\n  Python: %s\n",
                   rec, qframe, query_framed);
            failures++;
        }
        int elen; memcpy(&elen, p, 4); p += 4;
        char episode[2048];
        CHECK(elen < (int)sizeof episode);
        memcpy(episode, p, (size_t)elen); episode[elen] = 0; p += elen;

        float expected[3];
        memcpy(expected, p, 12); p += 12;
        int expected_route; memcpy(&expected_route, p, 4); p += 4;

        int n_pieces; memcpy(&n_pieces, p, 4); p += 4;
        CHECK(n_pieces > 0 && n_pieces <= 600);
        char *pieces[600];
        char piece_storage[600 * 80];
        char *storage = piece_storage;
        for (int i = 0; i < n_pieces; i++) {
            int plen; memcpy(&plen, p, 4); p += 4;
            CHECK(plen >= 0 && plen < 80);
            pieces[i] = storage;
            memcpy(pieces[i], p, (size_t)plen); pieces[i][plen] = 0;
            storage += plen + 1;
            p += plen;
        }
        int flen; memcpy(&flen, p, 4); p += 4;
        char framed_expected[4096];
        CHECK(flen < (int)sizeof framed_expected);
        memcpy(framed_expected, p, (size_t)flen); framed_expected[flen] = 0; p += flen;

        /* 1. framing parity: C must rebuild Python's frame byte-for-byte */
        char framed[4096];
        nm_frame_text(episode, framed, sizeof framed);
        if (strcmp(framed, framed_expected) != 0) {
            printf("FAIL record %d framing:\n  C:      %s\n  Python: %s\n",
                   rec, framed, framed_expected);
            failures++;
        }

        /* 2. span reconstruction without a tokenizer */
        int allowed[600];
        int span_start[9600], span_end[9600];
        int n_spans = nm_build_spans(framed, episode, pieces, n_pieces,
                                     allowed, span_start, span_end, 9600);
        if (n_spans < 0) {
            printf("FAIL record %d: nm_build_spans error %d\n", rec, n_spans);
            failures++;
            free(qf); free(sf);
            continue;
        }

        /* 3. route logits parity */
        nm_route_t r = nm_route_decision(ctx, qf, nq, sf, ns, allowed,
                                         n_pieces, span_start, span_end,
                                         n_spans);
        int bad = 0;
        for (int k = 0; k < 3; k++) {
            float got = k == 0 ? r.logit_normal : k == 1 ? r.logit_supported
                                                         : r.logit_insufficient;
            float a = got, b = expected[k];
            if (isfinite(a) && isfinite(b) && fabsf(a - b) > 0.05f) bad = 1;
            if (!isfinite(a) && !isfinite(b)) continue;   /* both -inf */
        }
        if (bad) {
            printf("FAIL record %d logits: got (%.3f %.3f %.3f) want (%.3f %.3f %.3f) "
                   "spans=%d\n", rec, r.logit_normal, r.logit_supported,
                   r.logit_insufficient, expected[0], expected[1], expected[2], n_spans);
            failures++;
        }
        if (r.route != expected_route) {
            printf("FAIL record %d route: got %d want %d\n", rec, r.route,
                   expected_route);
            failures++;
        }
        free(qf); free(sf);
    }

    nm_free(ctx);
    free(blob);
    if (failures) {
        printf("test_route_parity: %d FAILURES\n", failures);
        return 1;
    }
    printf("test_route_parity: all ok (%d records)\n", n_records);
    return 0;
}
