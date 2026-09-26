/* test_memory_state.c — session episode store and .bnstate v1/v2 persistence.
 *
 * Covers the LC-001 failure fixes: capacity (16 -> 512), episode feature-mean
 * cache, and state round-trips (v2 with means, legacy v1 text-only).
 */
#include "memory_state.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; \
} } while (0)

int main(void) {
    setbuf(stdout, NULL);
    /* 1. Unbounded storage: 5000 adds are all kept (no cap) */
    ms_reset_all();
    ms_session_t *s = ms_get_session("cap");
    CHECK(s != NULL);
    for (int i = 0; i < 5000; i++) {
        char buf[64];
        snprintf(buf, sizeof buf, "episode %d", i);
        ms_add_episode(s, buf);
    }
    CHECK(s->n_episodes == 5000);
    CHECK(strcmp(s->episodes[0], "episode 0") == 0);
    CHECK(strcmp(s->episodes[4999], "episode 4999") == 0);

    /* 1b. Unbounded episode LENGTH: a 4000-char episode is kept verbatim */
    char long_text[4001];
    memset(long_text, 'a', 4000);
    long_text[0] = 'L';
    long_text[3999] = 'Z';
    long_text[4000] = 0;
    ms_reset_all();
    ms_session_t *t = ms_get_session("trunc");
    ms_add_episode(t, long_text);
    CHECK(t->n_episodes == 1);
    CHECK(strlen(t->episodes[0]) == 4000);
    CHECK(t->episodes[0][0] == 'L' && t->episodes[0][3999] == 'Z');

    /* 3a. v2 round-trip with a large store and a long episode */
    ms_reset_all();
    ms_session_t *big = ms_get_session("big");
    for (int i = 0; i < 3000; i++) {
        char buf[64];
        snprintf(buf, sizeof buf, "fact number %d", i);
        ms_add_episode(big, buf);
    }
    ms_add_episode(big, long_text);
    CHECK(ms_save("/tmp/ms_test_big.bnstate") == 0);
    ms_reset_all();
    CHECK(ms_load("/tmp/ms_test_big.bnstate") == 0);
    ms_session_t *big2 = ms_find_session("big");
    CHECK(big2 != NULL);
    if (big2) {
        CHECK(big2->n_episodes == 3001);
        CHECK(strcmp(big2->episodes[2999], "fact number 2999") == 0);
        CHECK(strlen(big2->episodes[3000]) == 4000);
    }

    /* 3b. v2 round-trip: episodes AND means survive save/reset/load */
    ms_reset_all();
    ms_session_t *a = ms_get_session("alpha");
    ms_add_episode(a, "Morgan lives in Lima.");
    ms_add_episode(a, "Caius works in Aachen.");
    float m0[MS_FEAT_DIM], m1[MS_FEAT_DIM];
    memset(m0, 0, sizeof m0); memset(m1, 0, sizeof m1);
    m0[0] = 1.5f; m1[7] = -2.25f;
    ms_set_mean(a, 0, m0);
    ms_set_mean(a, 1, m1);
    CHECK(a->means_valid == 1);

    CHECK(ms_save("/tmp/ms_test_v2.bnstate") == 0);
    ms_reset_all();
    CHECK(ms_session_count() == 0);
    CHECK(ms_load("/tmp/ms_test_v2.bnstate") == 0);
    CHECK(ms_session_count() == 1);
    ms_session_t *a2 = ms_find_session("alpha");
    CHECK(a2 != NULL);
    if (a2) {
        CHECK(a2->n_episodes == 2);
        CHECK(strcmp(a2->episodes[0], "Morgan lives in Lima.") == 0);
        CHECK(strcmp(a2->episodes[1], "Caius works in Aachen.") == 0);
        CHECK(a2->means_valid == 1);
        CHECK(a2->ep_means[0][0] == 1.5f);
        CHECK(a2->ep_means[1][7] == -2.25f);
    }

    /* 4. Legacy v1 file (text only) loads with means marked invalid */
    FILE *f = fopen("/tmp/ms_test_v1.bnstate", "wb");
    {
        uint32_t hdr[4] = {0x54534E42u, 1, 1, 0};   /* BNST, version 1 */
        fwrite(hdr, 4, 4, f);
        const char *id = "legacy";
        uint32_t id_len = 6, n_ep = 1, ep_len = 9;
        fwrite(&id_len, 4, 1, f); fwrite(id, 1, 6, f);
        fwrite(&n_ep, 4, 1, f);
        const char *ep = "old fact.";
        fwrite(&ep_len, 4, 1, f); fwrite(ep, 1, 9, f);
    }
    fclose(f);
    ms_reset_all();
    CHECK(ms_load("/tmp/ms_test_v1.bnstate") == 0);
    ms_session_t *l = ms_find_session("legacy");
    CHECK(l != NULL);
    if (l) {
        CHECK(l->n_episodes == 1);
        CHECK(strcmp(l->episodes[0], "old fact.") == 0);
        CHECK(l->means_valid == 0);
    }

    /* 5. Corrupt magic rejected; existing state untouched */
    f = fopen("/tmp/ms_bad.bnstate", "wb");
    fwrite("XXXXXXXX", 1, 8, f);
    fclose(f);
    CHECK(ms_load("/tmp/ms_bad.bnstate") == -1);
    CHECK(ms_session_count() == 1);

    /* 6. Session table grows without a fixed bound */
    ms_reset_all();
    char sid[64];
    for (int i = 0; i < 200; i++) {
        snprintf(sid, sizeof sid, "s%d", i);
        CHECK(ms_get_session(sid) != NULL);
    }
    CHECK(ms_session_count() == 200);

    /* 7. Mean updates after reload stay consistent (idempotent re-save) */
    ms_reset_all();
    ms_session_t *r = ms_get_session("rt");
    ms_add_episode(r, "x");
    float rm[MS_FEAT_DIM];
    memset(rm, 0, sizeof rm); rm[3] = 0.5f;
    ms_set_mean(r, 0, rm);
    CHECK(ms_save("/tmp/ms_test_v2b.bnstate") == 0);
    CHECK(ms_load("/tmp/ms_test_v2b.bnstate") == 0);
    ms_session_t *r2 = ms_find_session("rt");
    CHECK(r2 != NULL);
    if (r2) CHECK(r2->ep_means[0][3] == 0.5f);

    /* 8. Feature-row cache: set/get + v3 round-trip */
    ms_reset_all();
    ms_session_t *rw = ms_get_session("rows");
    ms_add_episode(rw, "Some fact text.");
    float rows[3 * MS_FEAT_DIM];
    for (int i = 0; i < 3 * MS_FEAT_DIM; i++) rows[i] = (float)i * 0.5f;
    ms_set_rows(rw, 0, rows, 3);
    CHECK(rw->ep_row_counts[0] == 3);
    CHECK(rw->ep_rows[0] != NULL);
    CHECK(rw->ep_rows[0][0] == 0.0f);
    CHECK(fabsf(rw->ep_rows[0][3 * MS_FEAT_DIM - 1] - (3 * MS_FEAT_DIM - 1) * 0.5f) < 1e-6f);
    CHECK(ms_save("/tmp/ms_test_rows.bnstate") == 0);
    ms_reset_all();
    CHECK(ms_load("/tmp/ms_test_rows.bnstate") == 0);
    ms_session_t *rw2 = ms_find_session("rows");
    CHECK(rw2 != NULL);
    if (rw2) {
        CHECK(rw2->ep_row_counts[0] == 3);
        CHECK(fabsf(rw2->ep_rows[0][7] - 3.5f) < 1e-6f);
        CHECK(strcmp(rw2->episodes[0], "Some fact text.") == 0);
    }
    /* v2 file (no rows) still loads */
    ms_reset_all();
    CHECK(ms_load("/tmp/ms_test_v2.bnstate") == 0);
    CHECK(ms_find_session("alpha") != NULL);

    if (failures) {
        printf("test_memory_state: %d FAILURES\n", failures);
        return 1;
    }
    printf("test_memory_state: all ok\n");
    return 0;
}
