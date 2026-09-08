#include "metis/metis_file.h"
#include "sha256.h"
#include "bitnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); ++failures; } } while (0)

static void test_bound_backbone_roundtrip(void) {
    metis_file_model_t m;
    const int ids[1] = {0};
    const int D = 4, KV = 2, Q = 4, HD = 2, R = 1;
    const uint8_t expected_text_sha256[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    uint8_t digest[32];
    char err[128];
    FILE *text = fopen("/tmp/test_sha256.txt", "wb");
    CHECK(text != NULL, "open SHA-256 fixture");
    if (text != NULL) {
        CHECK(fwrite("abc", 1, 3, text) == 3, "write SHA-256 fixture");
        fclose(text);
    }
    CHECK(bitnet_sha256_file("/tmp/test_sha256.txt", digest) == 0,
          "hash fixture");
    CHECK(memcmp(digest, expected_text_sha256, sizeof digest) == 0,
          "SHA-256 implementation");
    remove("/tmp/test_sha256.txt");

    CHECK(metis_model_alloc(&m, 1, ids, D, KV, Q, HD, 0.9f, 1.0f,
                            0.9f, 1, 0.9f, 1, R) == 0,
          "bound model alloc");
    m.params.has_backbone_sha256 = 1;
    memcpy(m.params.backbone_sha256, expected_text_sha256,
           sizeof expected_text_sha256);
    CHECK(metis_model_save(&m, "/tmp/test_bound.bnmem") == 0,
          "save bound model");
    metis_file_model_t *loaded = metis_model_load(
        "/tmp/test_bound.bnmem", D, Q, KV, HD, 1, err, sizeof err);
    CHECK(loaded != NULL, "load bound model");
    if (loaded != NULL) {
        CHECK(loaded->params.has_backbone_sha256 == 1,
              "bound model flag");
        CHECK(memcmp(loaded->params.backbone_sha256, expected_text_sha256,
                     sizeof expected_text_sha256) == 0,
              "bound model SHA-256");
        metis_model_free_loaded(loaded);
    }
    remove("/tmp/test_bound.bnmem");
    metis_model_free_arrays(&m);
}

static void test_bounded_selection_roundtrip(void) {
    metis_file_model_t m;
    const int ids[1] = {0};
    const int D = 4, KV = 2, Q = 4, HD = 2;
    char err[128];
    CHECK(metis_model_alloc(&m, 1, ids, D, KV, Q, HD, 0.9f, 1.0f,
                            0.9f, 1, 0.9f, 1, 0) == 0,
          "bounded selection alloc");
    m.params.has_backbone_sha256 = 1;
    memset(m.params.backbone_sha256, 0x5a,
           sizeof m.params.backbone_sha256);
    m.params.alpha_max_tokens = 16;
    m.params.alpha_max_fraction = 0.2f;
    CHECK(metis_model_save(&m, "/tmp/test_bounded.bnmem") == 0,
          "save BNMEM6 bounded selection");
    metis_file_model_t *loaded = metis_model_load(
        "/tmp/test_bounded.bnmem", D, Q, KV, HD, 1, err, sizeof err);
    CHECK(loaded != NULL, "load BNMEM6 bounded selection");
    if (loaded != NULL) {
        CHECK(loaded->version == 1, "loaded bounded model runtime version");
        CHECK(loaded->params.alpha_max_tokens == 16,
              "bounded max tokens roundtrip");
        CHECK(fabsf(loaded->params.alpha_max_fraction - 0.2f) < 1e-6f,
              "bounded max fraction roundtrip");
        CHECK(loaded->params.has_backbone_sha256 == 1,
              "bounded model remains backbone bound");
        metis_model_free_loaded(loaded);
    }
    remove("/tmp/test_bounded.bnmem");
    metis_model_free_arrays(&m);
}

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
        CHECK(memcmp(l->params.gdu_ab_v, m.params.gdu_ab_v,
                     (size_t)NL * sizeof(float)) == 0, "all alpha biases");
        CHECK(memcmp(l->params.gdu_bb_v, m.params.gdu_bb_v,
                     (size_t)NL * sizeof(float)) == 0, "all beta biases");
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

static void test_backbone_delta_query_read(void) {
    metis_file_model_t m;
    const int ids[1] = {0};
    const int D = 4, KV = 2, Q = 4, HD = 2, R = 1;
    float h[D], q_base[Q], M[KV * KV], S[KV], out[Q];
    CHECK(metis_model_alloc(&m, 1, ids, D, KV, Q, HD, 0.9f, 1.0f,
                            0.9f, 1, 0.9f, 1, R) == 0,
          "backbone delta alloc");
    m.params.query_add_backbone = 1;
    memset(h, 0, sizeof h);
    memset(m.params.query_a, 0,
           (size_t)Q * R * sizeof(float));
    memset(m.params.query_b, 0,
           (size_t)R * D * sizeof(float));
    for (int i = 0; i < HD; ++i) m.params.query_norm[i] = 1.0f;
    q_base[0] = 1.0f; q_base[1] = 0.0f;
    q_base[2] = 0.0f; q_base[3] = 1.0f;
    M[0] = 1.0f; M[1] = 0.0f;
    M[2] = 0.0f; M[3] = 1.0f;
    S[0] = 0.0f; S[1] = 0.0f;
    metis_read(&m.params, 0, h, q_base, M, S, out);
    CHECK(fabsf(out[0] - 1.0f) < 1e-5f, "base q group 0");
    CHECK(fabsf(out[3] - 1.0f) < 1e-5f, "base q group 1");
    metis_model_free_arrays(&m);
}

static void test_low_rank_kv_roundtrip(void) {
    metis_file_model_t m;
    const int ids[1] = {0};
    const int D = 4, KV = 2, Q = 4, HD = 2, QR = 1, KR = 1;
    char err[128];
    CHECK(metis_model_alloc(&m, 1, ids, D, KV, Q, HD, 0.9f, 1.0f,
                            0.9f, 1, 0.9f, 1, QR) == 0,
          "low-rank K/V alloc");
    free(m.params.wk); m.params.wk = NULL;
    free(m.params.wv); m.params.wv = NULL;
    m.params.kv_rank = KR;
    m.params.wk_a = (float *)calloc((size_t)KV * KR, sizeof(float));
    m.params.wk_b = (float *)calloc((size_t)KR * D, sizeof(float));
    m.params.wv_a = (float *)calloc((size_t)KV * KR, sizeof(float));
    m.params.wv_b = (float *)calloc((size_t)KR * D, sizeof(float));
    CHECK(m.params.wk_a && m.params.wk_b &&
          m.params.wv_a && m.params.wv_b,
          "low-rank K/V factor allocation");
    if (m.params.wk_a && m.params.wk_b &&
        m.params.wv_a && m.params.wv_b) {
        m.params.wk_a[0] = 1.0f;
        m.params.wk_a[1] = -0.5f;
        m.params.wk_b[0] = 0.25f;
        m.params.wk_b[1] = 0.5f;
        m.params.wv_a[0] = 0.75f;
        m.params.wv_a[1] = 1.25f;
        m.params.wv_b[2] = -0.5f;
        m.params.wv_b[3] = 0.125f;
        CHECK(metis_model_save(&m, "/tmp/test_low_rank_kv.bnmem") == 0,
              "save low-rank K/V");
        metis_file_model_t *l = metis_model_load(
            "/tmp/test_low_rank_kv.bnmem", D, Q, KV, HD, 1,
            err, sizeof err);
        CHECK(l != NULL, "load low-rank K/V");
        if (l != NULL) {
            CHECK(l->params.kv_rank == KR, "K/V rank");
            CHECK(l->params.wk == NULL && l->params.wv == NULL,
                  "full K/V omitted");
            CHECK(memcmp(l->params.wk_a, m.params.wk_a,
                         (size_t)KV * KR * sizeof(float)) == 0,
                  "wk_a roundtrip");
            CHECK(memcmp(l->params.wv_b, m.params.wv_b,
                         (size_t)KR * D * sizeof(float)) == 0,
                  "wv_b roundtrip");
            {
                metis_file_model_t full;
                float low_M[4] = {0}, low_S[2] = {0};
                float full_M[4] = {0}, full_S[2] = {0};
                const float raw[4] = {0.5f, -1.0f, 0.25f, 2.0f};
                const float norm[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                CHECK(metis_model_alloc(
                          &full, 1, ids, D, KV, Q, HD,
                          0.9f, 1.0f, 0.9f, 1, 0.9f, 1, QR) == 0,
                      "full K/V comparison alloc");
                for (int o = 0; o < KV; ++o) {
                    for (int i = 0; i < D; ++i) {
                        full.params.wk[(size_t)o * D + i] =
                            m.params.wk_a[o] * m.params.wk_b[i];
                        full.params.wv[(size_t)o * D + i] =
                            m.params.wv_a[o] * m.params.wv_b[i];
                    }
                }
                CHECK(metis_commit_states(
                          &l->params, 0, low_M, low_S, raw, 1,
                          norm, 1e-6f) == 0,
                      "low-rank K/V commit");
                CHECK(metis_commit_states(
                          &full.params, 0, full_M, full_S, raw, 1,
                          norm, 1e-6f) == 0,
                      "full K/V commit");
                for (int i = 0; i < KV * KV; ++i)
                    CHECK(fabsf(low_M[i] - full_M[i]) < 1e-6f,
                          "low-rank K/V M equivalence");
                for (int i = 0; i < KV; ++i)
                    CHECK(fabsf(low_S[i] - full_S[i]) < 1e-6f,
                          "low-rank K/V S equivalence");
                metis_model_free_arrays(&full);
            }
            metis_model_free_loaded(l);
        }
    }
    remove("/tmp/test_low_rank_kv.bnmem");
    metis_model_free_arrays(&m);
}

static void test_commit_uses_standard_rmsnorm_order(void) {
    metis_file_model_t m;
    const int ids[1] = {0};
    const int D = 2, KV = 1, Q = 1, HD = 1;
    const float raw[2] = {1.0f, 3.0f};
    const float norm_w[2] = {2.0f, 1.0f};
    float M[1] = {0.0f};
    float S[1] = {0.0f};

    CHECK(metis_model_alloc(&m, 1, ids, D, KV, Q, HD, 0.9f, 1.0f,
                            0.9f, 1, 0.9f, 1, 1) == 0,
          "standard RMSNorm commit alloc");
    memset(m.params.wk, 0, (size_t)KV * D * sizeof(float));
    memset(m.params.wv, 0, (size_t)KV * D * sizeof(float));
    memset(m.params.w_agg, 0, (size_t)D * sizeof(float));
    memset(m.params.gdu_aw, 0, (size_t)D * sizeof(float));
    memset(m.params.gdu_bw, 0, (size_t)D * sizeof(float));
    m.params.wk[0] = 1.0f;
    m.params.wv[0] = 1.0f;
    m.params.gdu_ab_v[0] = 9.210240f;
    m.params.gdu_bb_v[0] = 9.210240f;

    CHECK(metis_commit_states(&m.params, 0, M, S, raw, 1,
                              norm_w, 1e-6f) == 0,
          "standard RMSNorm commit");
    {
        const float inv = 1.0f / sqrtf(5.0f + 1e-6f);
        const float expected_v = 2.0f * inv;
        const float beta = 0.9f / (1.0f + expf(-9.210240f));
        CHECK(fabsf(M[0] - beta * expected_v) < 5e-5f,
              "commit normalizes raw residual before applying weight");
    }
    metis_model_free_arrays(&m);
}

static void test_full_query_roundtrip(void) {
    metis_file_model_t m;
    const int ids[2] = {0, 1};
    const int D = 8, KV = 4, Q = 8, HD = 4, NL = 2;
    char err[128];
    CHECK(metis_model_alloc(&m, NL, ids, D, KV, Q, HD, 0.9f, 1.0f,
                            0.9f, 1, 0.9f, 1, 0) == 0,
          "full query alloc");
    for (int i = 0; i < NL; ++i) {
        m.params.gdu_ab_v[i] = 0.1f + (float)i;
        m.params.gdu_bb_v[i] = -0.2f - (float)i;
    }
    for (size_t i = 0; i < (size_t)NL * Q * D; ++i)
        m.params.query_proj[i] = (float)i * 0.01f;
    CHECK(metis_model_save(&m, "/tmp/test_full_query.bnmem") == 0,
          "full query save");
    metis_file_model_t *l = metis_model_load(
        "/tmp/test_full_query.bnmem", D, Q, KV, HD, 2, err, sizeof err);
    CHECK(l != NULL, "full query load");
    if (l != NULL) {
        CHECK(l->params.query_rank == 0, "full query rank marker");
        CHECK(memcmp(l->params.query_proj, m.params.query_proj,
                     (size_t)NL * Q * D * sizeof(float)) == 0,
              "full query payload");
        CHECK(memcmp(l->params.gdu_ab_v, m.params.gdu_ab_v,
                     (size_t)NL * sizeof(float)) == 0,
              "full query alpha biases");
        CHECK(memcmp(l->params.gdu_bb_v, m.params.gdu_bb_v,
                     (size_t)NL * sizeof(float)) == 0,
              "full query beta biases");
        metis_model_free_loaded(l);
    }
    remove("/tmp/test_full_query.bnmem");
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
    test_bound_backbone_roundtrip();
    test_bounded_selection_roundtrip();
    test_v5_roundtrip();
    test_full_query_roundtrip();
    test_backbone_delta_query_read();
    test_low_rank_kv_roundtrip();
    test_commit_uses_standard_rmsnorm_order();
    test_state_roundtrip();
#if !defined(_WIN32)
    test_cli_flags();
#endif
    if (failures) { fprintf(stderr, "%d failures\n", failures); return 1; }
    printf("test_metis_memory: OK\n");
    return 0;
}
