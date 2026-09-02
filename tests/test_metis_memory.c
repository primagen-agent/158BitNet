#include "metis/metis_file.h"
#include "bitnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); ++failures; } } while (0)

static void test_v5_roundtrip(void) {
    metis_file_model_t m;
    int ids[3] = {0, 15, 31};
    const int D = 2560, KV = 256, Q = 4096, HD = 128, R = 128, NL = 3;
    CHECK(metis_model_alloc(&m, NL, ids, D, KV, Q, HD, 0.9f, 1.0f, 0.9f, 1, 0.9f, 1, R) == 0, "alloc");
    float v = 0.0f;
    float *tensors[] = { m.params.wk, m.params.wv, m.params.w_agg,
                         m.params.gdu_aw, m.params.gdu_bw, m.params.mem_norm,
                         m.params.query_a, m.params.query_b, m.params.query_norm,
                         m.params.gdu_ab_v, m.params.gdu_bb_v, NULL };
    size_t sz[] = { (size_t)NL*KV*D, (size_t)NL*KV*D, (size_t)NL*D,
                    (size_t)NL*D, (size_t)NL*D, (size_t)NL*Q,
                    (size_t)NL*Q*R, (size_t)NL*R*D, (size_t)NL*HD,
                    (size_t)NL, (size_t)NL };
    for (int t = 0; tensors[t]; ++t)
        for (size_t i = 0; i < sz[t]; ++i) tensors[t][i] = 0.001f * (v += 1.0f);
    m.params.mem_norm[7] = 1.5f;
    m.params.gdu_ab_v[0] = -0.027f;
    CHECK(metis_model_save(&m, "/tmp/test_v5.bnmem") == 0, "save");
    char err[128];
    metis_file_model_t *l = metis_model_load("/tmp/test_v5.bnmem", D, Q, KV, HD, 32, err, sizeof err);
    CHECK(l != NULL, "load");
    if (l) {
        CHECK(l->params.n_layers == NL && l->params.layer_ids[2] == 31, "meta");
        CHECK(memcmp(l->params.wk, m.params.wk, (size_t)NL*KV*D*sizeof(float)) == 0, "wk");
        CHECK(l->params.gdu_ab_v[0] == -0.027f, "bias");
        CHECK(l->params.query_rank == R, "rank");
        metis_model_free_loaded(l);
    }
    {
        FILE *f = fopen("/tmp/test_v5.bnmem", "r+b");
        const unsigned char legacy_magic_and_version[12] = {
            'B', 'N', 'M', 'E', 'M', '5', 0, 0, 5, 0, 0, 0
        };
        CHECK(f != NULL, "open legacy fixture");
        if (f != NULL) {
            CHECK(fwrite(legacy_magic_and_version, 1,
                         sizeof legacy_magic_and_version, f) ==
                  sizeof legacy_magic_and_version, "write legacy header");
            fclose(f);
        }
        l = metis_model_load("/tmp/test_v5.bnmem", D, Q, KV, HD, 32,
                             err, sizeof err);
        CHECK(l != NULL, "load legacy BNMEM5 version 5");
        metis_model_free_loaded(l);
    }
    l = metis_model_load("/tmp/test_v5.bnmem", D, Q, KV, HD, 2, err, sizeof err);
    CHECK(l == NULL, "block_count reject");
    remove("/tmp/test_v5.bnmem");
    metis_model_free_arrays(&m);
}

static void test_state_roundtrip(void) {
    metis_file_model_t m;
    const int ids[2] = {1, 3};
    const int NL = 2, KV = 4;
    float source_M[NL * KV * KV], source_S[NL * KV];
    float loaded_M[NL * KV * KV], loaded_S[NL * KV];
    float before_M[NL * KV * KV], before_S[NL * KV];
    char err[128];
    int active = 0;

    CHECK(metis_model_alloc(&m, NL, ids, 8, KV, 8, 4, 0.9f, 1.0f,
                            0.9f, 1, 0.9f, 1, 2) == 0,
          "state model alloc");
    for (size_t i = 0; i < sizeof source_M / sizeof source_M[0]; ++i)
        source_M[i] = (float)i * 0.25f - 2.0f;
    for (size_t i = 0; i < sizeof source_S / sizeof source_S[0]; ++i)
        source_S[i] = (float)i * -0.5f + 1.0f;
    memset(loaded_M, 0, sizeof loaded_M);
    memset(loaded_S, 0, sizeof loaded_S);
    CHECK(metis_state_save(&m.params, source_M, source_S, 1,
                           "/tmp/test_metis_state.bnstate") == 0,
          "state save");
    CHECK(metis_state_load(&m.params, loaded_M, loaded_S, &active,
                           "/tmp/test_metis_state.bnstate", err,
                           sizeof err) == 0,
          "state load");
    CHECK(active == 1, "state active roundtrip");
    CHECK(memcmp(source_M, loaded_M, sizeof source_M) == 0,
          "state M roundtrip");
    CHECK(memcmp(source_S, loaded_S, sizeof source_S) == 0,
          "state S roundtrip");

    memcpy(before_M, loaded_M, sizeof before_M);
    memcpy(before_S, loaded_S, sizeof before_S);
    {
        FILE *f = fopen("/tmp/test_metis_state.bnstate", "r+b");
        CHECK(f != NULL, "open state for corruption");
        if (f != NULL) {
            CHECK(fseek(f, -1, SEEK_END) == 0, "seek state payload");
            if (fseek(f, -1, SEEK_END) == 0) {
                int byte = fgetc(f);
                CHECK(byte != EOF, "read state payload byte");
                CHECK(fseek(f, -1, SEEK_CUR) == 0, "rewind state payload byte");
                if (byte != EOF) fputc(byte ^ 0x01, f);
            }
            fclose(f);
        }
    }
    CHECK(metis_state_load(&m.params, loaded_M, loaded_S, &active,
                           "/tmp/test_metis_state.bnstate", err,
                           sizeof err) != 0,
          "corrupt state rejected");
    CHECK(memcmp(before_M, loaded_M, sizeof before_M) == 0 &&
          memcmp(before_S, loaded_S, sizeof before_S) == 0,
          "failed import leaves state unchanged");
    remove("/tmp/test_metis_state.bnstate");
    metis_model_free_arrays(&m);
}

#if !defined(_WIN32)
static void test_cli_flags(void) {
    int rc = system(BITNET_BINARY_DIR "/minimal_generate "
                    BITNET_SOURCE_DIR "/models/bitcpm4-0.5b-tq2_0.gguf hi 2 "
                    "--memory-model /nonexistent.bnmem >/dev/null 2>&1");
    CHECK(rc != 0, "bad --memory-model rejected");
}
#endif

int main(void) {
    test_v5_roundtrip();
    test_state_roundtrip();
#if !defined(_WIN32)
    test_cli_flags();
#endif
    if (failures) { fprintf(stderr, "%d failures\n", failures); return 1; }
    printf("test_metis_memory: OK\n");
    return 0;
}
