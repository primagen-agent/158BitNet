/* test_neural_memory.c — .bnmodel version handling and the EC-001 episode
 * relevance head. Builds minimal .bnmodel files with hand-crafted ep_rel
 * weights so the expected score is computable in closed form:
 * with W1 = row-select identity (h[r] = tanh(x[r])) and W2 = pick h[0],
 * score = sigmoid(tanh(query_mean[0])).
 */
#include "neural_memory.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; \
} } while (0)

typedef struct { const char *name; int rows, cols; } spec_t;

/* v3 fixture with ep_joint.* tensors:
 * pair.weight row r selects input dim r (input layout [q(2048); e(2048);
 * q⊙e(2048); |q−e|(2048)]), so f_r = tanh(x_r); mix picks agg[0]; out is
 * identity → logit = tanh(max_ij q_i[0]) with everything else zeroed. */

/* Two-pass writer: offsets depend only on payload sizes, so the compact JSON
 * index lines' exact lengths are known in the second pass. */
static int write_test_bnmodel(const char *path, uint32_t magic, int version,
                              int with_ep_rel, int bad_ep_rel_shape) {
    spec_t specs[16];
    int n = 0;
    specs[n++] = (spec_t){"query.weight", 1, 1};
    specs[n++] = (spec_t){"route.weight", 1, 1};
    specs[n++] = (spec_t){"branch.encoder.weight", 1, 1};
    if (with_ep_rel) {
        specs[n++] = (spec_t){"ep_joint.pair.weight", bad_ep_rel_shape ? 4 : 64, 8192};
        specs[n++] = (spec_t){"ep_joint.pair.bias", 64, 0};
        specs[n++] = (spec_t){"ep_joint.mix.weight", 64, 4160};
        specs[n++] = (spec_t){"ep_joint.mix.bias", 64, 0};
        specs[n++] = (spec_t){"ep_joint.out.weight", 1, 64};
        specs[n++] = (spec_t){"ep_joint.out.bias", 1, 0};
    }

    long sizes[16];
    for (int i = 0; i < n; i++)
        sizes[i] = (long)specs[i].rows * (specs[i].cols > 0 ? specs[i].cols : 1);
    long index_size = 0;
    for (int i = 0; i < n; i++) {
        long o = 0;
        for (int j = 0; j < i; j++) o += sizes[j] * 4;
        char line[256];
        int len;
        if (specs[i].cols > 0)
            len = snprintf(line, sizeof line, "{\"n\":\"%s\",\"s\":[%d,%d],\"o\":%ld}\n",
                           specs[i].name, specs[i].rows, specs[i].cols, o);
        else
            len = snprintf(line, sizeof line, "{\"n\":\"%s\",\"s\":[%d],\"o\":%ld}\n",
                           specs[i].name, specs[i].rows, o);
        index_size += len;
    }

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t hdr[4] = {magic, (uint32_t)version, (uint32_t)n, (uint32_t)index_size};
    fwrite(hdr, 4, 4, f);
    for (int i = 0; i < n; i++) {
        long o = 0;
        for (int j = 0; j < i; j++) o += sizes[j] * 4;
        if (specs[i].cols > 0)
            fprintf(f, "{\"n\":\"%s\",\"s\":[%d,%d],\"o\":%ld}\n",
                    specs[i].name, specs[i].rows, specs[i].cols, o);
        else
            fprintf(f, "{\"n\":\"%s\",\"s\":[%d],\"o\":%ld}\n",
                    specs[i].name, specs[i].rows, o);
    }
    fputc('\n', f);

    for (int i = 0; i < n; i++) {
        long count = sizes[i];
        if (strcmp(specs[i].name, "ep_joint.pair.weight") == 0 && !bad_ep_rel_shape) {
            /* row r selects input dim r: f[r] = tanh(x[r]) */
            float *w = calloc((size_t)count, sizeof(float));
            for (int r = 0; r < specs[i].rows; r++)
                w[(long)r * 8192 + r] = 1.0f;
            fwrite(w, 4, (size_t)count, f);
            free(w);
        } else if (strcmp(specs[i].name, "ep_joint.mix.weight") == 0) {
            /* mix row 0 selects input dim 0 (agg[0]) */
            float *w = calloc((size_t)count, sizeof(float));
            w[0] = 1.0f;
            fwrite(w, 4, (size_t)count, f);
            free(w);
        } else if (strcmp(specs[i].name, "ep_joint.out.weight") == 0) {
            /* out picks hidden dim 0 */
            float *w = calloc((size_t)count, sizeof(float));
            w[0] = 1.0f;
            fwrite(w, 4, (size_t)count, f);
            free(w);
        } else {
            float *z = calloc((size_t)count, sizeof(float));
            fwrite(z, 4, (size_t)count, f);
            free(z);
        }
    }
    fclose(f);
    return 0;
}

int main(void) {
    setbuf(stdout, NULL);

    /* 1. bad magic rejected */
    CHECK(write_test_bnmodel("/tmp/nm_badmagic.bnmodel", 0x58585858u, 2, 0, 0) == 0);
    CHECK(nm_init("/tmp/nm_badmagic.bnmodel") == NULL);

    /* 2. unsupported future version rejected */
    CHECK(write_test_bnmodel("/tmp/nm_v4.bnmodel", 0x574D4E42u, 4, 1, 0) == 0);
    CHECK(nm_init("/tmp/nm_v4.bnmodel") == NULL);

    /* 3. v2 (no joint head) loads; joint relevance unavailable */
    CHECK(write_test_bnmodel("/tmp/nm_v2.bnmodel", 0x574D4E42u, 2, 0, 0) == 0);
    nm_ctx_t *v2 = nm_init("/tmp/nm_v2.bnmodel");
    CHECK(v2 != NULL);
    if (v2) {
        CHECK(nm_has_episode_joint(v2) == 0);
        float q[2048];
        memset(q, 0, sizeof q);
        CHECK(nm_episode_joint_relevance(v2, q, 1, q, q, 1, q) == -1.0f);
        nm_free(v2);
    }

    /* 4. v3 with ep_joint: score = sigmoid(tanh(tanh(max_i q_i[0])))
     * (pair row r selects input dim r = query dim r; mix/out pick agg[0]) */
    CHECK(write_test_bnmodel("/tmp/nm_v3.bnmodel", 0x574D4E42u, 3, 1, 0) == 0);
    nm_ctx_t *v3 = nm_init("/tmp/nm_v3.bnmodel");
    CHECK(v3 != NULL);
    if (v3) {
        CHECK(nm_has_episode_joint(v3) == 1);
        float qrows[3 * 2048], erows[2 * 2048], qmean[2048], emean[2048];
        memset(qrows, 0, sizeof qrows); memset(erows, 0, sizeof erows);
        memset(qmean, 0, sizeof qmean); memset(emean, 0, sizeof emean);
        qrows[0] = -1.0f; qrows[2048] = 2.0f;   /* max over query rows = 2 */
        erows[100] = 5.0f;                       /* episode dims unpicked */
        float expect = 1.0f / (1.0f + expf(-tanhf(tanhf(2.0f))));
        float got = nm_episode_joint_relevance(v3, qrows, 3, qmean, erows, 2, emean);
        CHECK(got > 0.0f && got < 1.0f);
        CHECK(fabsf(got - expect) < 1e-5f);
        qrows[2048] = -3.0f;
        qrows[0] = -0.5f; qrows[2 * 2048] = -0.9f;   /* max dim-0 = -0.5 */
        float got_neg = nm_episode_joint_relevance(v3, qrows, 3, qmean, erows, 2, emean);
        CHECK(fabsf(got_neg - (1.0f / (1.0f + expf(-tanhf(tanhf(-0.5f)))))) < 1e-5f);
        CHECK(got > got_neg);
        CHECK(nm_episode_joint_relevance(v3, NULL, 3, qmean, erows, 2, emean) == -1.0f);
        nm_free(v3);
    }

    /* 5. v3 declaring wrong ep_joint shapes is corrupt and must fail */
    CHECK(write_test_bnmodel("/tmp/nm_v3_bad.bnmodel", 0x574D4E42u, 3, 1, 1) == 0);
    CHECK(nm_init("/tmp/nm_v3_bad.bnmodel") == NULL);

    /* 6. v3 without any ep_joint entries is corrupt and must fail */
    CHECK(write_test_bnmodel("/tmp/nm_v3_missing.bnmodel", 0x574D4E42u, 3, 0, 0) == 0);
    CHECK(nm_init("/tmp/nm_v3_missing.bnmodel") == NULL);

    /* 7. Dynamic framing: arbitrary-length text, reuse across calls */
    {
        char *buf = NULL;
        size_t cap = 0;
        char big[6000];
        memset(big, 'x', sizeof big - 1);
        big[0] = '"';                       /* needs escaping */
        big[5] = '\\';
        big[10] = '\n';
        big[sizeof big - 1] = 0;
        nm_frame_text_dyn(big, &buf, &cap);
        CHECK(buf != NULL);
        CHECK(strncmp(buf, "{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"", 40) == 0);
        CHECK(strlen(buf) > 6000);          /* escapes only grow it */
        /* escaped bytes present verbatim */
        CHECK(strstr(buf, "\\\"") != NULL && strstr(buf, "\\\\") != NULL &&
              strstr(buf, "\\n") != NULL);
        /* ends with the closing frame */
        CHECK(strcmp(buf + strlen(buf) - 2, "\"}") == 0);
        /* reuse with a longer text keeps working */
        char *bigger = calloc(20000, 1);
        memset(bigger, 'y', 19999);
        nm_frame_text_dyn(bigger, &buf, &cap);
        CHECK(strlen(buf) > 20000);
        nm_frame_query_dyn("hi", &buf, &cap);
        CHECK(strcmp(buf, "[{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"hi\"}]") == 0);
        free(buf);
        free(bigger);
    }

    /* 8. Truncated feature window: pieces reconstruct a PREFIX of a long
     * frame; spans are built inside the window, wrapper tokens disallowed */
    {
        const char *episode = "Caius currently lives in Aachen. "
                              "Darya currently works in Bilbao. Padding follows: zzz.";
        char frame[4096];
        nm_frame_text(episode, frame, sizeof frame);
        size_t flen = strlen(frame);

        /* hand-split the first 60 bytes into fake pieces covering a prefix */
        char pieces_buf[8][64];
        char *pieces[8];
        int n_pieces = 0;
        size_t off = 0;
        while (off < 60) {
            size_t take = 12;
            if (off + take > 60) take = 60 - off;
            memcpy(pieces_buf[n_pieces], frame + off, take);
            pieces_buf[n_pieces][take] = 0;
            pieces[n_pieces] = pieces_buf[n_pieces];
            n_pieces++;
            off += take;
        }
        /* pieces now reconstruct frame[0..60) — a strict prefix */
        (void)flen;

        int allowed[8];
        int sstart[128], send[128];
        int n_spans = nm_build_spans(frame, episode, pieces, n_pieces,
                                     allowed, sstart, send, 128);
        CHECK(n_spans > 0);
        /* wrapper pieces (before the text literal at byte 40) disallowed */
        CHECK(allowed[0] == 0);
        CHECK(allowed[1] == 0);
        CHECK(allowed[2] == 0);
        /* at least one content piece allowed */
        int any_allowed = 0;
        for (int i = 0; i < n_pieces; i++) any_allowed |= allowed[i];
        CHECK(any_allowed == 1);
        /* spans only touch allowed tokens */
        for (int i = 0; i < n_spans; i++) {
            CHECK(sstart[i] >= 0 && send[i] <= n_pieces);
            CHECK(send[i] > sstart[i]);
        }
    }

    if (failures) {
        printf("test_neural_memory: %d FAILURES\n", failures);
        return 1;
    }
    printf("test_neural_memory: all ok\n");
    return 0;
}
