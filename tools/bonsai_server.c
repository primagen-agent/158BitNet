/* One OpenAI-compatible HTTP process for every model family this build runs.
 *
 *   bonsai_server --model cpm05=/m/bitcpm4-0.5b-tq2_0.gguf \
 *                 --model cpm1b=/m/bitcpm4-1b-tq2_0.gguf  \
 *                 --model cpm3b=/m/bitcpm4-3b-tq2_0.gguf  \
 *                 --model cpm8b=/m/bitcpm4-8b-tq2_0.gguf  \
 *                 --model bonsai27b=/m/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
 *                 --draft /m/dspine27.safetensors --block 2
 *
 * The model file decides the family (general.architecture): qwen35 runs the
 * bonsai.c graph, minicpm runs minicpm.c, both through the same bonsai_* API
 * and the same binary.  A draft is attached to the preceding --model: a
 * .safetensors file is the DSpine block drafter (qwen35 targets only) and a
 * .gguf file is an ordinary model used as a draft, so speculative decoding is
 * a property of the registered model rather than of a separate executable.
 *
 * Models load lazily on the first request that names them (or at startup with
 * --preload) and stay resident.  Requests without DSpine use the plain decode
 * path; the draft path is greedy, because acceptance is decided by comparing
 * argmax rows, so a request with temperature > 0 falls back to sampling.
 * "spec": false in a request forces the plain path on a model that has a
 * draft, which is how the two engines are compared through one process.
 *
 * A GGUF draft is worth it only for a large target, and on the Cix board it
 * has to be re-measured whenever the plain path gets faster: the 8B used to
 * gain from a 1B draft (8.3 -> 9.5 tok/s at --model-block 4) and now loses to
 * its own decode (13.0 vs 9.3) because the minicpm engine prepares its head
 * and runs eight threads.  The 27B is the one that still needs a draft:
 * DSpine takes it from 2.0 to 5.0 tok/s, and a GGUF draft cannot (no equal
 * vocab, and a 2-bit target's verify needs the trained drafter).
 *
 * Endpoints: GET /health, GET /v1/models, POST /v1/completions,
 * POST /v1/chat/completions (both with "stream": true SSE), plus per-request
 * statistics in "usage" and "perf". */

#define _GNU_SOURCE
#include "bonsai.h"
#include "dspine.h"

#include "third_party/cJSON/cJSON.h"
#include "third_party/mongoose/mongoose.h"

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_MODELS 16
#define MAX_STOPS 8
#define MAX_SEL 5            /* DSpine reads five residual-stream layers */
#define DEFAULT_CTX 4096
#define MAX_CTX 32768
#define DEFAULT_MAX_TOKENS 256
#define MAX_REQ_TOKENS 4096

typedef struct {
    char id[96];
    char path[1024];
    char draft[1024];        /* DSpine .safetensors, or a .gguf draft model */
    char chat_format[16];    /* "chatml" or "plain" */
    int block;               /* speculative block; 0 = the server default */

    int loaded;              /* 1 when the weights are open */
    int failed;              /* a load was attempted and failed */
    bonsai_model_t *m;
    bonsai_context_t *ctx;
    dspine_model_t *dr;      /* DSpine draft (borrows m and ctx) */
    bonsai_model_t *dm;      /* GGUF draft */
    bonsai_context_t *dctx;
    int sel[MAX_SEL];
    int n_sel;
    int V, H, nl;

    float *vlogits;          /* [16][V] verify rows */
    float *vhid;             /* [16][n_sel*H] residual rows for the draft */
    float *layer_in;         /* [nl][16][H] residual entering every layer */
    float *samp;             /* [V] sampling scratch */
    struct cand *cand;       /* [V] sampling scratch */
    float *st;               /* target state snapshot */
    size_t st_bytes;
    float *dst;              /* GGUF draft state snapshot */
    size_t dst_bytes;
    int ctx_cap;
    unsigned long long last_use;   /* for the resident-set LRU */
} model_t;

struct cand { float v; int id; };

typedef struct {
    char *text;
    size_t len, cap;
    size_t emitted;          /* bytes already streamed to the client */
    int stop_hit;
    const char *stop[MAX_STOPS];
    int n_stops;
    size_t holdback;         /* longest stop minus one, kept back while streaming */
    char pending[4];
    int pending_len, pending_expected;
    struct mg_connection *conn;   /* NULL when the request is not streaming */
    int is_chat;
    const char *id;
    const char *model_id;
} sink_t;

typedef struct {
    int max_tokens;
    float temperature, top_p, repeat_penalty;
    int top_k, repeat_last_n, block;
    int use_spec;               /* "spec": false forces the plain path */
    unsigned long long seed;
} gen_req_t;

typedef struct {
    int prompt_tokens, completion_tokens;
    int hit_eos, stop_sequence;
    int rounds, accepted;
    double prefill_sec, decode_sec, total_sec;
} gen_stats_t;

typedef struct {
    model_t models[MAX_MODELS];
    int n_models;
    int default_index;
    const char *host, *port;
    int threads, ctx_cap, default_max_tokens, max_req_tokens;
    int default_block;
    int preload;
    int max_resident;              /* models kept in memory; 0 = never unload */
    unsigned long long use_clock;
} server_t;

static volatile sig_atomic_t g_stop = 0;
static void handle_signal(int signo) { (void)signo; g_stop = 1; }

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static unsigned long long rng_next(unsigned long long *s) {
    unsigned long long x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x ? x : 88172645463325252ull;
    return *s;
}

static char *dup_n(const char *data, size_t len) {
    char *out = (char *)malloc(len + 1);
    if (out == NULL) return NULL;
    if (len > 0 && data != NULL) memcpy(out, data, len);
    out[len] = '\0';
    return out;
}

static int append_bytes(char **buf, size_t *len, size_t *cap, const char *data, size_t n) {
    if (data == NULL || n == 0) return 0;
    if (*len + n + 1 > *cap) {
        size_t next = *cap ? *cap : 128;
        while (next < *len + n + 1) next *= 2;
        char *p = (char *)realloc(*buf, next);
        if (p == NULL) return -1;
        *buf = p;
        *cap = next;
    }
    memcpy(*buf + *len, data, n);
    *len += n;
    (*buf)[*len] = '\0';
    return 0;
}

static int append_cstr(char **buf, size_t *len, size_t *cap, const char *s) {
    return append_bytes(buf, len, cap, s, strlen(s));
}

/* ---------------- model registry and loading ---------------- */

static model_t *find_model(server_t *srv, const char *id) {
    int i;
    if (srv->n_models == 0) return NULL;
    if (id == NULL || id[0] == '\0') return &srv->models[srv->default_index];
    for (i = 0; i < srv->n_models; ++i)
        if (strcmp(srv->models[i].id, id) == 0) return &srv->models[i];
    return NULL;
}

static int argmax_row(const float *row, int n) {
    int best = 0, i;
    for (i = 1; i < n; ++i)
        if (row[i] > row[best]) best = i;
    return best;
}

static void free_model(model_t *e) {
    free(e->vlogits);
    free(e->vhid);
    free(e->layer_in);
    free(e->samp);
    free(e->cand);
    free(e->st);
    free(e->dst);
    e->vlogits = e->vhid = e->layer_in = e->samp = e->st = e->dst = NULL;
    e->cand = NULL;
    dspine_free(e->dr);
    e->dr = NULL;
    if (e->dctx != NULL) bonsai_free_context(e->dctx);
    e->dctx = NULL;
    if (e->dm != NULL) bonsai_free_model(e->dm);
    e->dm = NULL;
    if (e->ctx != NULL) bonsai_free_context(e->ctx);
    e->ctx = NULL;
    if (e->m != NULL) bonsai_free_model(e->m);
    e->m = NULL;
    e->loaded = 0;
}

/* Open the weights, create the context and the per-model scratch, and attach
 * the draft.  Returns 0 on success; the model stays unloaded on failure. */
static int load_model(model_t *e, const server_t *srv);

/* Keep at most srv->max_resident models open: a request for any other model
 * unloads the least recently used one first.  Weights are the bulk of the
 * footprint (5.95 GB for the 27B, 0.4-5 GB per BitCPM), the board has 28 GB,
 * and the reload costs 2.5-3 s for a BitCPM and 22 s for the 27B - so the
 * resident set is a request-level decision, not a process-level one.  It also
 * keeps the 27B's 5.95 GB mapping from competing with the other models for
 * page cache, which measurably costs it ~5% of decode speed. */
static int ensure_resident(model_t *e, server_t *srv) {
    int i, n = 0;
    if (e->loaded) {
        e->last_use = ++srv->use_clock;
        return 0;
    }
    if (srv->max_resident > 0) {
        for (i = 0; i < srv->n_models; ++i)
            if (srv->models[i].loaded) ++n;
        while (n >= srv->max_resident) {
            model_t *victim = NULL;
            for (i = 0; i < srv->n_models; ++i) {
                model_t *c = &srv->models[i];
                if (!c->loaded || c == e) continue;
                if (victim == NULL || c->last_use < victim->last_use) victim = c;
            }
            if (victim == NULL) break;   /* only e is open: leave it */
            fprintf(stderr, "[unload] %s\n", victim->id);
            free_model(victim);
            --n;
        }
    }
    if (load_model(e, srv) != 0) return -1;
    e->last_use = ++srv->use_clock;
    return 0;
}

static int load_model(model_t *e, const server_t *srv) {
    double t0;
    int is_qwen35;

    if (e->loaded) return 0;
    if (e->failed) return -1;
    t0 = now_sec();
    fprintf(stderr, "[load] %s: %s\n", e->id, e->path);
    e->m = bonsai_load_model(e->path, srv->threads);
    if (e->m == NULL) {
        fprintf(stderr, "[load] %s failed\n", e->id);
        e->failed = 1;
        return -1;
    }
    e->V = bonsai_vocab_size(e->m);
    e->H = bonsai_embedding_length(e->m);
    e->nl = bonsai_layer_count(e->m);
    is_qwen35 = bonsai_model_arch(e->m) == BONSAI_ARCH_QWEN35;
    e->ctx_cap = srv->ctx_cap;
    e->ctx = bonsai_create_context(e->m, e->ctx_cap);
    if (e->ctx != NULL) {
        /* The library clamps the allocation to the model's declared
         * general.architecture.context_length and refuses positions past it,
         * so serve what the model allows rather than what --ctx asked for:
         * prompts are truncated to this, and /v1/models reports it. */
        int cap = bonsai_context_capacity(e->ctx);
        if (cap > 0 && cap < e->ctx_cap) {
            fprintf(stderr, "[load] %s: model allows %d positions, asked for %d\n",
                    e->id, cap, e->ctx_cap);
            e->ctx_cap = cap;
        }
    }
    if (e->ctx == NULL) {
        fprintf(stderr, "[load] %s: context failed\n", e->id);
        free_model(e);
        e->failed = 1;
        return -1;
    }
    e->vlogits = (float *)calloc((size_t)16 * (size_t)e->V, sizeof(float));
    e->samp = (float *)malloc((size_t)e->V * sizeof(float));
    e->cand = (struct cand *)malloc((size_t)e->V * sizeof(struct cand));
    if (e->vlogits == NULL || e->samp == NULL || e->cand == NULL) {
        fprintf(stderr, "[load] %s: out of memory\n", e->id);
        free_model(e);
        e->failed = 1;
        return -1;
    }
    if (e->draft[0] != '\0') {
        size_t n = strlen(e->draft);
        if (n > 12 && strcmp(e->draft + n - 12, ".safetensors") == 0) {
            if (!is_qwen35 || e->nl < 62) {
                fprintf(stderr, "[load] %s: a DSpine draft needs a qwen35 target\n", e->id);
                free_model(e);
                e->failed = 1;
                return -1;
            }
            e->dr = dspine_load(e->draft, e->m, e->ctx);
            if (e->dr == NULL) {
                fprintf(stderr, "[load] %s: draft load failed\n", e->id);
                free_model(e);
                e->failed = 1;
                return -1;
            }
            /* The verify records the residual stream after these layers; they
             * are the DSpine checkpoint's conditioning layers. */
            e->sel[0] = 1; e->sel[1] = 16; e->sel[2] = 31; e->sel[3] = 46; e->sel[4] = 61;
            e->n_sel = MAX_SEL;
            e->vhid = (float *)calloc((size_t)16 * MAX_SEL * (size_t)e->H, sizeof(float));
            e->layer_in = (float *)calloc((size_t)e->nl * 16 * (size_t)e->H, sizeof(float));
            e->st_bytes = bonsai_state_bytes(e->ctx);
            e->st = (float *)malloc(e->st_bytes + 16);
            if (e->vhid == NULL || e->layer_in == NULL || e->st == NULL) {
                fprintf(stderr, "[load] %s: out of memory for the draft\n", e->id);
                free_model(e);
                e->failed = 1;
                return -1;
            }
            if (bonsai_replay_cache_init(e->ctx) != 0)
                fprintf(stderr, "[load] %s: replay cache unavailable, using full replay\n",
                        e->id);
        } else {
            e->dm = bonsai_load_model(e->draft, srv->threads);
            if (e->dm == NULL || bonsai_vocab_size(e->dm) != e->V) {
                fprintf(stderr, "[load] %s: draft model failed to load or has a different vocab\n",
                        e->id);
                free_model(e);
                e->failed = 1;
                return -1;
            }
            e->dctx = bonsai_create_context(e->dm, e->ctx_cap);
            if (e->dctx != NULL) {
                /* The draft walks the same positions as the target, so the
                 * smaller of the two declared lengths is what can be served. */
                int dcap = bonsai_context_capacity(e->dctx);
                if (dcap > 0 && dcap < e->ctx_cap) {
                    fprintf(stderr, "[load] %s: draft allows %d positions, target %d; "
                            "serving %d\n", e->id, dcap, e->ctx_cap, dcap);
                    e->ctx_cap = dcap;
                }
            }
            if (e->dctx == NULL) {
                fprintf(stderr, "[load] %s: draft context failed\n", e->id);
                free_model(e);
                e->failed = 1;
                return -1;
            }
            e->dst_bytes = bonsai_state_bytes(e->dctx);
            e->dst = (float *)malloc(e->dst_bytes + 16);
            if (e->dst == NULL) {
                fprintf(stderr, "[load] %s: out of memory for the draft\n", e->id);
                free_model(e);
                e->failed = 1;
                return -1;
            }
        }
    }
    e->loaded = 1;
    fprintf(stderr,
            "[load] %s: arch %s, %d layers, vocab %d, ctx %d, %s, %.1f s\n", e->id,
            bonsai_arch_name(bonsai_model_arch(e->m)), e->nl, e->V, e->ctx_cap,
            e->dr != NULL ? "DSpine draft" : (e->dm != NULL ? "GGUF draft" : "no draft"),
            now_sec() - t0);
    return 0;
}

/* ---------------- text sink: UTF-8, stop strings, SSE ---------------- */

static int utf8_expected_len(unsigned char b) {
    if (b < 0x80u) return 1;
    if (b >= 0xC2u && b <= 0xDFu) return 2;
    if (b >= 0xE0u && b <= 0xEFu) return 3;
    if (b >= 0xF0u && b <= 0xF4u) return 4;
    return 0;
}

static int utf8_is_cont(unsigned char b) { return (b & 0xC0u) == 0x80u; }

static void json_add_string(cJSON *o, const char *n, const char *v) {
    cJSON_AddItemToObject(o, n, cJSON_CreateString(v == NULL ? "" : v));
}
static void json_add_number(cJSON *o, const char *n, double v) {
    cJSON_AddItemToObject(o, n, cJSON_CreateNumber(v));
}
static void json_add_null(cJSON *o, const char *n) {
    cJSON_AddItemToObject(o, n, cJSON_CreateNull());
}

static void send_sse_json(struct mg_connection *c, cJSON *root) {
    char *body = cJSON_Print(root);
    if (body == NULL) return;
    mg_http_printf_chunk(c, "data: %s\n\n", body);
    free(body);
    mg_mgr_poll(c->mgr, 0);
}

static cJSON *stream_chunk(const sink_t *s, const char *content, const char *finish_reason,
                           int include_role) {
    cJSON *root = cJSON_CreateObject(), *choices = cJSON_CreateArray();
    cJSON *choice = cJSON_CreateObject(), *delta = cJSON_CreateObject();
    if (root == NULL || choices == NULL || choice == NULL || delta == NULL) {
        cJSON_Delete(delta);
        cJSON_Delete(choice);
        cJSON_Delete(choices);
        cJSON_Delete(root);
        return NULL;
    }
    json_add_string(root, "id", s->id);
    json_add_string(root, "object",
                    s->is_chat ? "chat.completion.chunk" : "text_completion");
    json_add_number(root, "created", (double)time(NULL));
    json_add_string(root, "model", s->model_id);
    json_add_number(choice, "index", 0);
    if (s->is_chat) {
        if (include_role) json_add_string(delta, "role", "assistant");
        if (content != NULL && content[0] != '\0') json_add_string(delta, "content", content);
        cJSON_AddItemToObject(choice, "delta", delta);
    } else {
        cJSON_Delete(delta);
        json_add_string(choice, "text", content == NULL ? "" : content);
    }
    if (finish_reason == NULL) json_add_null(choice, "finish_reason");
    else json_add_string(choice, "finish_reason", finish_reason);
    cJSON_AddItemToArray(choices, choice);
    cJSON_AddItemToObject(root, "choices", choices);
    return root;
}

/* Write text[start..end) to the client when streaming. */
static void sink_stream(sink_t *s, size_t start, size_t end) {
    char *tmp;
    cJSON *chunk;
    if (s->conn == NULL || end <= start) return;
    tmp = dup_n(s->text + start, end - start);
    if (tmp == NULL) return;
    chunk = stream_chunk(s, tmp, NULL, 0);
    free(tmp);
    if (chunk != NULL) {
        send_sse_json(s->conn, chunk);
        cJSON_Delete(chunk);
    }
}

/* Flush every byte that cannot be part of a stop string, at a UTF-8 boundary. */
static void sink_flush(sink_t *s, int final) {
    size_t limit, max;
    if (s->conn == NULL) return;
    if (final) {
        limit = s->len;
    } else {
        max = s->holdback < s->len ? s->len - s->holdback : 0;
        if (max <= s->emitted) return;
        limit = max;
        while (limit > s->emitted && utf8_is_cont((unsigned char)s->text[limit])) --limit;
    }
    sink_stream(s, s->emitted, limit);
    s->emitted = limit;
}

static int sink_has_stop(sink_t *s, size_t *pos) {
    int i;
    for (i = 0; i < s->n_stops; ++i) {
        const char *hit = strstr(s->text, s->stop[i]);
        if (hit != NULL) {
            size_t off = (size_t)(hit - s->text);
            if (i == 0 || off < *pos) *pos = off;
        }
    }
    return *pos != (size_t)-1;
}

static int sink_add_bytes(sink_t *s, const char *bytes, int n) {
    int i;
    size_t stop_pos = (size_t)-1;
    if (bytes == NULL || n <= 0) return 0;
    for (i = 0; i < n; ++i) {
        unsigned char b = (unsigned char)bytes[i];
        if (s->pending_len == 0) {
            int expected = utf8_expected_len(b);
            if (expected == 1) {
                if (append_bytes(&s->text, &s->len, &s->cap, (const char *)&b, 1) != 0) return -1;
            } else if (expected > 1) {
                s->pending[0] = (char)b;
                s->pending_len = 1;
                s->pending_expected = expected;
            }
        } else if (utf8_is_cont(b)) {
            s->pending[s->pending_len++] = (char)b;
            if (s->pending_len < s->pending_expected) continue;
            if (append_bytes(&s->text, &s->len, &s->cap, s->pending,
                             (size_t)s->pending_len) != 0) return -1;
            s->pending_len = 0;
            s->pending_expected = 0;
        } else {
            s->pending_len = 0;
            s->pending_expected = 0;
            --i;
        }
    }
    if (s->n_stops > 0 && !s->stop_hit && sink_has_stop(s, &stop_pos)) {
        s->len = stop_pos;
        s->text[s->len] = '\0';
        s->stop_hit = 1;
    }
    sink_flush(s, s->stop_hit);
    return 0;
}

/* ---------------- sampling ---------------- */

static int cmp_cand_desc(const void *a, const void *b) {
    const struct cand *x = (const struct cand *)a, *y = (const struct cand *)b;
    if (x->v < y->v) return 1;
    if (x->v > y->v) return -1;
    return 0;
}

/* Temperature, top-k and top-p over one logits row, with a repetition
 * penalty.  temperature <= 0 is greedy, which is also the spec-decode path. */
static int sample_token(model_t *e, const float *logits, const int *hist, int n_hist,
                        const gen_req_t *p, unsigned long long *rng) {
    const int V = e->V;
    float *x = e->samp;
    int i, nc = 0;
    float inv, mx, thr;
    double sum, acc, r;

    memcpy(x, logits, (size_t)V * sizeof(float));
    if (p->repeat_penalty != 1.0f && p->repeat_last_n > 0 && n_hist > 0) {
        int start = n_hist > p->repeat_last_n ? n_hist - p->repeat_last_n : 0;
        for (i = start; i < n_hist; ++i) {
            int t = hist[i], j, dup = 0;
            if (t < 0 || t >= V) continue;
            for (j = start; j < i; ++j)
                if (hist[j] == t) { dup = 1; break; }
            if (dup) continue;
            x[t] = x[t] > 0.0f ? x[t] / p->repeat_penalty : x[t] * p->repeat_penalty;
        }
    }
    if (p->temperature <= 0.0f) return argmax_row(x, V);

    inv = 1.0f / p->temperature;
    mx = x[0] * inv;
    for (i = 1; i < V; ++i)
        if (x[i] * inv > mx) mx = x[i] * inv;
    thr = mx - 25.0f;
    for (i = 0; i < V; ++i) {
        float v = x[i] * inv;
        if (v >= thr) { e->cand[nc].v = v; e->cand[nc].id = i; ++nc; }
    }
    qsort(e->cand, (size_t)nc, sizeof(struct cand), cmp_cand_desc);
    if (p->top_k > 0 && nc > p->top_k) nc = p->top_k;
    sum = 0.0;
    for (i = 0; i < nc; ++i) {
        e->cand[i].v = expf(e->cand[i].v - e->cand[0].v);
        sum += e->cand[i].v;
    }
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        acc = 0.0;
        for (i = 0; i < nc; ++i) {
            acc += e->cand[i].v;
            if (acc / sum >= p->top_p) { nc = i + 1; break; }
        }
        sum = 0.0;
        for (i = 0; i < nc; ++i) sum += e->cand[i].v;
    }
    r = (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0) * sum;
    acc = 0.0;
    for (i = 0; i < nc; ++i) {
        acc += e->cand[i].v;
        if (r < acc) break;
    }
    if (i >= nc) i = nc - 1;
    return e->cand[i].id;
}

/* ---------------- generation ---------------- */

static int emit_token(model_t *e, int id, sink_t *s, int *count) {
    char buf[256];
    int n = bonsai_decode_token(e->m, id, buf, (int)sizeof(buf));
    if (n < 0) n = 0;
    ++*count;
    return sink_add_bytes(s, buf, n);
}

static int run_plain(model_t *e, const int *prompt, int np, int *hist, gen_req_t *p, sink_t *s,
                     gen_stats_t *st, char *err, size_t errsz) {
    unsigned long long rng = p->seed ? p->seed : 0x9E3779B97F4A7C15ull ^ (unsigned long long)now_sec();
    const float *logits;
    int emitted = 0, n_hist = np;
    double t0;

    if (bonsai_reset_context(e->ctx) != 0) {
        snprintf(err, errsz, "context reset failed");
        return -1;
    }
    memcpy(hist, prompt, (size_t)np * sizeof(int));
    t0 = now_sec();
    if (np > 0 && bonsai_eval(e->ctx, prompt, np) != 0) {
        snprintf(err, errsz, "prefill failed");
        return -1;
    }
    st->prefill_sec = now_sec() - t0;
    logits = bonsai_get_logits(e->ctx);
    if (logits == NULL) {
        snprintf(err, errsz, "no logits after prefill");
        return -1;
    }
    t0 = now_sec();
    for (;;) {
        int id = sample_token(e, logits, hist, n_hist, p, &rng);
        if (bonsai_token_is_eog(e->m, id)) { st->hit_eos = 1; break; }
        if (emit_token(e, id, s, &emitted) != 0) {
            snprintf(err, errsz, "decode failed");
            return -1;
        }
        hist[n_hist++] = id;
        if (emitted >= p->max_tokens || s->stop_hit) break;
        if (bonsai_eval(e->ctx, &id, 1) != 0) {
            snprintf(err, errsz, "decode eval failed");
            return -1;
        }
        logits = bonsai_get_logits(e->ctx);
    }
    st->decode_sec = now_sec() - t0;
    st->completion_tokens = emitted;
    st->stop_sequence = s->stop_hit;
    if (s->stop_hit) st->hit_eos = 0;
    sink_flush(s, 1);
    return 0;
}

/* DSpine drafter: propose a block, verify it in one weight pass, keep the
 * accepted prefix and the bonus token, roll the target's recurrent state back
 * to the accepted length.  Greedy only. */
static int run_spec_dspine(model_t *e, const int *prompt, int np, int *hist, gen_req_t *p,
                           sink_t *s, gen_stats_t *st, char *err, size_t errsz) {
    int block = p->block > 0 ? p->block : e->block;
    int emitted = 0, n_hist = np, anchor, pos, last_n = 0;
    double t0;

    if (block > 16) block = 16;
    if (block < 2) block = 2;
    if (dspine_max_drafts(e->dr) < block - 1) block = dspine_max_drafts(e->dr) + 1;
    if (np <= 0) {
        snprintf(err, errsz, "speculative decoding needs a non-empty prompt");
        return -1;
    }
    dspine_reset(e->dr);
    if (bonsai_reset_context(e->ctx) != 0) {
        snprintf(err, errsz, "context reset failed");
        return -1;
    }
    memcpy(hist, prompt, (size_t)np * sizeof(int));

    /* prefill in verify blocks; every block also feeds the draft's context */
    t0 = now_sec();
    for (pos = 0; pos < np; pos += 16) {
        int n = np - pos < 16 ? np - pos : 16;
        int last = pos + n == np;
        if (bonsai_eval_block(e->ctx, prompt + pos, n, last ? e->vlogits : NULL,
                              last ? e->V : 0, e->sel, e->n_sel, e->vhid, NULL) != 0) {
            snprintf(err, errsz, "prefill failed");
            return -1;
        }
        if (dspine_append_context(e->dr, e->vhid, n) != 0) {
            snprintf(err, errsz, "draft prefill failed");
            return -1;
        }
        last_n = n;
    }
    st->prefill_sec = now_sec() - t0;
    anchor = argmax_row(e->vlogits + (size_t)(last_n - 1) * e->V, e->V);

    t0 = now_sec();
    for (;;) {
        if (bonsai_token_is_eog(e->m, anchor)) { st->hit_eos = 1; break; }
        if (emit_token(e, anchor, s, &emitted) != 0) {
            snprintf(err, errsz, "decode failed");
            return -1;
        }
        hist[n_hist++] = anchor;
        if (emitted >= p->max_tokens || s->stop_hit) break;

        {
            const int remaining = p->max_tokens - emitted;
            const int T = (remaining + 1 < block) ? remaining + 1 : block;
            const int nd = T - 1;
            int drafts[16], vt[16], accept = 0, bonus, i;
            if (dspine_propose(e->dr, anchor, nd, drafts) != nd) {
                snprintf(err, errsz, "draft propose failed");
                return -1;
            }
            vt[0] = anchor;
            for (i = 0; i < nd; ++i) vt[i + 1] = drafts[i];
            bonsai_state_save(e->ctx, e->st);
            if (bonsai_eval_block(e->ctx, vt, T, e->vlogits, e->V, e->sel, e->n_sel, e->vhid,
                                  e->layer_in) != 0) {
                snprintf(err, errsz, "verify failed");
                return -1;
            }
            for (i = 0; i < nd; ++i) {
                if (argmax_row(e->vlogits + (size_t)i * e->V, e->V) == drafts[i]) ++accept;
                else break;
            }
            bonus = argmax_row(e->vlogits + (size_t)accept * e->V, e->V);
            if (accept + 1 < T) {
                bonsai_state_load(e->ctx, e->st);
                bonsai_rewind(e->ctx, T - accept - 1);
                bonsai_replay_state_cached(e->ctx, e->layer_in, T, accept + 1);
            }
            if (dspine_append_context(e->dr, e->vhid, accept + 1) != 0) {
                snprintf(err, errsz, "draft context append failed");
                return -1;
            }
            ++st->rounds;
            st->accepted += accept;
            for (i = 0; i < accept; ++i) {
                if (bonsai_token_is_eog(e->m, drafts[i])) { st->hit_eos = 1; goto finished; }
                if (emit_token(e, drafts[i], s, &emitted) != 0) {
                    snprintf(err, errsz, "decode failed");
                    return -1;
                }
                hist[n_hist++] = drafts[i];
                if (emitted >= p->max_tokens || s->stop_hit) goto finished;
            }
            anchor = bonus;
        }
    }
finished:
    st->decode_sec = now_sec() - t0;
    st->completion_tokens = emitted;
    st->stop_sequence = s->stop_hit;
    if (s->stop_hit) st->hit_eos = 0;
    sink_flush(s, 1);
    return 0;
}

/* An ordinary GGUF model as the draft: the same block verify, driven by the
 * draft's own decode loop.  Works across families (a minicpm draft can drive a
 * qwen35 target) as long as the vocabularies agree. */
static int run_spec_gguf(model_t *e, const int *prompt, int np, int *hist, gen_req_t *p,
                         sink_t *s, gen_stats_t *st, char *err, size_t errsz) {
    int block = p->block > 0 ? p->block : e->block;
    int emitted = 0, n_hist = np, anchor, pos, last_n = 0;
    double t0;

    if (block > 16) block = 16;
    if (block < 2) block = 2;
    if (np <= 0) {
        snprintf(err, errsz, "speculative decoding needs a non-empty prompt");
        return -1;
    }
    if (bonsai_reset_context(e->ctx) != 0 || bonsai_reset_context(e->dctx) != 0) {
        snprintf(err, errsz, "context reset failed");
        return -1;
    }
    memcpy(hist, prompt, (size_t)np * sizeof(int));
    t0 = now_sec();
    for (pos = 0; pos < np; pos += 16) {
        int n = np - pos < 16 ? np - pos : 16;
        int last = pos + n == np;
        if (bonsai_eval_block(e->ctx, prompt + pos, n, last ? e->vlogits : NULL,
                              last ? e->V : 0, NULL, 0, NULL, NULL) != 0) {
            snprintf(err, errsz, "prefill failed");
            return -1;
        }
        last_n = n;
    }
    if (bonsai_eval(e->dctx, prompt, np) != 0) {
        snprintf(err, errsz, "draft prefill failed");
        return -1;
    }
    st->prefill_sec = now_sec() - t0;
    anchor = argmax_row(e->vlogits + (size_t)(last_n - 1) * e->V, e->V);

    t0 = now_sec();
    for (;;) {
        if (bonsai_token_is_eog(e->m, anchor)) { st->hit_eos = 1; break; }
        if (emit_token(e, anchor, s, &emitted) != 0) {
            snprintf(err, errsz, "decode failed");
            return -1;
        }
        hist[n_hist++] = anchor;
        if (emitted >= p->max_tokens || s->stop_hit) break;

        {
            const int remaining = p->max_tokens - emitted;
            const int T = (remaining + 1 < block) ? remaining + 1 : block;
            const int nd = T - 1;
            int vt[16], accept = 0, bonus, i;
            bonsai_state_save(e->dctx, e->dst);
            vt[0] = anchor;
            for (i = 0; i < nd; ++i) {
                if (bonsai_eval(e->dctx, &vt[i], 1) != 0) {
                    snprintf(err, errsz, "draft eval failed");
                    return -1;
                }
                vt[i + 1] = bonsai_sample_greedy(e->dctx);
            }
            bonsai_state_save(e->ctx, e->st);
            if (bonsai_eval_block(e->ctx, vt, T, e->vlogits, e->V, NULL, 0, NULL, NULL) != 0) {
                snprintf(err, errsz, "verify failed");
                return -1;
            }
            for (i = 0; i < nd; ++i) {
                if (argmax_row(e->vlogits + (size_t)i * e->V, e->V) == vt[i + 1]) ++accept;
                else break;
            }
            bonus = argmax_row(e->vlogits + (size_t)accept * e->V, e->V);
            if (accept + 1 < T) {
                bonsai_state_load(e->ctx, e->st);
                bonsai_rewind(e->ctx, T - accept - 1);
            }
            if (nd - accept - 1 > 0) {
                bonsai_state_load(e->dctx, e->dst);
                bonsai_rewind(e->dctx, nd - accept - 1);
            }
            ++st->rounds;
            st->accepted += accept;
            for (i = 0; i < accept; ++i) {
                if (bonsai_token_is_eog(e->m, vt[i + 1])) { st->hit_eos = 1; goto finished; }
                if (emit_token(e, vt[i + 1], s, &emitted) != 0) {
                    snprintf(err, errsz, "decode failed");
                    return -1;
                }
                hist[n_hist++] = vt[i + 1];
                if (emitted >= p->max_tokens || s->stop_hit) goto finished;
            }
            anchor = bonus;
        }
    }
finished:
    st->decode_sec = now_sec() - t0;
    st->completion_tokens = emitted;
    st->stop_sequence = s->stop_hit;
    if (s->stop_hit) st->hit_eos = 0;
    sink_flush(s, 1);
    return 0;
}

/* ---------------- prompts ---------------- */

static const char *json_str(cJSON *o, const char *name, const char *fallback) {
    cJSON *v = cJSON_GetObjectItem(o, name);
    return (v != NULL && cJSON_IsString(v) && v->valuestring != NULL) ? v->valuestring
                                                                     : fallback;
}
static int json_int(cJSON *o, const char *name, int fallback) {
    cJSON *v = cJSON_GetObjectItem(o, name);
    if (v == NULL) return fallback;
    if (cJSON_IsNumber(v)) return (int)(v->valuedouble + (v->valuedouble >= 0 ? 0.5 : -0.5));
    if (cJSON_IsBool(v)) return cJSON_IsTrue(v) ? 1 : 0;
    if (cJSON_IsString(v)) return atoi(v->valuestring);
    return fallback;
}
static double json_num(cJSON *o, const char *name, double fallback) {
    cJSON *v = cJSON_GetObjectItem(o, name);
    return (v != NULL && cJSON_IsNumber(v)) ? v->valuedouble : fallback;
}
static int json_bool(cJSON *o, const char *name, int fallback) {
    cJSON *v = cJSON_GetObjectItem(o, name);
    return (v != NULL && cJSON_IsBool(v)) ? (cJSON_IsTrue(v) ? 1 : 0) : fallback;
}

static char *chat_prompt_chatml(cJSON *req) {
    cJSON *messages = cJSON_GetObjectItem(req, "messages");
    char *out = NULL;
    size_t len = 0, cap = 0;
    int i, n;
    if (messages == NULL || !cJSON_IsArray(messages)) return NULL;
    n = cJSON_GetArraySize(messages);
    if (n <= 0) return NULL;
    for (i = 0; i < n; ++i) {
        cJSON *m = cJSON_GetArrayItem(messages, i);
        const char *role, *content;
        if (m == NULL || !cJSON_IsObject(m)) continue;
        role = json_str(m, "role", "user");
        content = json_str(m, "content", "");
        if (append_cstr(&out, &len, &cap, "<|im_start|>") != 0 ||
            append_cstr(&out, &len, &cap, role) != 0 ||
            append_cstr(&out, &len, &cap, "\n") != 0 ||
            append_cstr(&out, &len, &cap, content) != 0 ||
            append_cstr(&out, &len, &cap, "<|im_end|>\n") != 0) {
            free(out);
            return NULL;
        }
    }
    if (append_cstr(&out, &len, &cap, "<|im_start|>assistant\n") != 0) {
        free(out);
        return NULL;
    }
    return out;
}

/* The BitCPM-CANN turn format used by the existing server's b158 path. */
static char *chat_prompt_plain(cJSON *req) {
    cJSON *messages = cJSON_GetObjectItem(req, "messages");
    char *out = NULL, *sys = NULL;
    size_t len = 0, cap = 0, slen = 0, scap = 0;
    int i, n;
    if (messages == NULL || !cJSON_IsArray(messages)) return NULL;
    n = cJSON_GetArraySize(messages);
    if (n <= 0) return NULL;
    for (i = 0; i < n; ++i) {
        cJSON *m = cJSON_GetArrayItem(messages, i);
        const char *role, *content;
        if (m == NULL || !cJSON_IsObject(m)) continue;
        role = json_str(m, "role", "user");
        content = json_str(m, "content", "");
        if (strcmp(role, "assistant") == 0) {
            if (append_cstr(&out, &len, &cap, content) != 0 ||
                append_cstr(&out, &len, &cap, "<|end_of_text|>") != 0) { free(out); free(sys); return NULL; }
        } else if (strcmp(role, "system") == 0) {
            if (slen > 0 && append_cstr(&sys, &slen, &scap, "\n\n") != 0) { free(out); free(sys); return NULL; }
            if (append_cstr(&sys, &slen, &scap, content) != 0) { free(out); free(sys); return NULL; }
        } else {
            if (append_cstr(&out, &len, &cap, "Human: ") != 0) { free(out); free(sys); return NULL; }
            if (slen > 0 && (append_bytes(&out, &len, &cap, sys, slen) != 0 ||
                             append_cstr(&out, &len, &cap, "\n\n") != 0)) { free(out); free(sys); return NULL; }
            if (append_cstr(&out, &len, &cap, content) != 0 ||
                append_cstr(&out, &len, &cap, "\n\nBITNETAssistant: ") != 0) { free(out); free(sys); return NULL; }
            slen = 0;
            if (sys != NULL) sys[0] = '\0';
        }
    }
    free(sys);
    return out;
}

static char *completion_prompt(cJSON *req) {
    cJSON *p = cJSON_GetObjectItem(req, "prompt");
    char *out = NULL;
    size_t len = 0, cap = 0;
    int i, n;
    if (p == NULL) return NULL;
    if (cJSON_IsString(p)) return dup_n(p->valuestring, strlen(p->valuestring));
    if (cJSON_IsArray(p)) {
        n = cJSON_GetArraySize(p);
        for (i = 0; i < n; ++i) {
            cJSON *e = cJSON_GetArrayItem(p, i);
            if (e != NULL && cJSON_IsString(e) &&
                append_cstr(&out, &len, &cap, e->valuestring) != 0) { free(out); return NULL; }
        }
        if (out == NULL) out = dup_n("", 0);
        return out;
    }
    return NULL;
}

/* ---------------- request handling ---------------- */

static void send_json(struct mg_connection *c, int status, cJSON *root) {
    char *body = cJSON_Print(root);
    if (body == NULL) {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n",
                      "{\"error\":{\"message\":\"failed to serialize json\",\"type\":\"server_error\"}}\n");
        return;
    }
    mg_http_reply(c, status, "Content-Type: application/json\r\n", "%s\n", body);
    free(body);
}

static void send_error(struct mg_connection *c, int status, const char *type, const char *msg) {
    cJSON *root = cJSON_CreateObject();
    cJSON *err = cJSON_CreateObject();
    if (root == NULL || err == NULL) {
        cJSON_Delete(err);
        cJSON_Delete(root);
        mg_http_reply(c, status, "Content-Type: application/json\r\n",
                      "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}\n");
        return;
    }
    json_add_string(err, "message", msg);
    json_add_string(err, "type", type);
    cJSON_AddItemToObject(root, "error", err);
    send_json(c, status, root);
    cJSON_Delete(root);
}

static cJSON *usage_json(const gen_stats_t *st) {
    cJSON *u = cJSON_CreateObject();
    if (u == NULL) return NULL;
    json_add_number(u, "prompt_tokens", st->prompt_tokens);
    json_add_number(u, "completion_tokens", st->completion_tokens);
    json_add_number(u, "total_tokens", st->prompt_tokens + st->completion_tokens);
    return u;
}

static void perf_json(cJSON *root, const model_t *e, const gen_stats_t *st) {
    cJSON *perf = cJSON_CreateObject();
    if (perf == NULL) return;
    json_add_number(perf, "prefill_ms", st->prefill_sec * 1e3);
    json_add_number(perf, "decode_ms", st->decode_sec * 1e3);
    json_add_number(perf, "total_ms", st->total_sec * 1e3);
    {
        /* The first token comes out of the prefill pass, so a one-token answer
         * (a stop string hit immediately, say) has no decode time to divide by
         * and would otherwise report a meaningless rate. */
        double rate = st->decode_sec >= 1e-3 ? st->completion_tokens / st->decode_sec : 0.0;
        json_add_number(perf, "decode_tokens_per_s", rate);
    }
    /* Report the engine that actually ran: a model with a draft still takes
     * the plain path when "spec" is false or the request samples. */
    json_add_string(perf, "engine",
                    st->rounds == 0 ? "decode"
                                    : (e->dr != NULL ? "spec-dspine" : "spec-gguf"));
    if (st->rounds > 0) {
        json_add_number(perf, "spec_rounds", st->rounds);
        json_add_number(perf, "spec_accepted_per_round",
                        (double)st->accepted / (double)st->rounds);
        json_add_number(perf, "spec_tokens_per_round",
                        (double)st->completion_tokens / (double)st->rounds);
    }
    cJSON_AddItemToObject(root, "perf", perf);
}

static void sse_headers(struct mg_connection *c) {
    mg_printf(c, "HTTP/1.1 200 OK\r\n"
                 "Content-Type: text/event-stream\r\n"
                 "Cache-Control: no-cache\r\n"
                 "Connection: close\r\n"
                 "Transfer-Encoding: chunked\r\n\r\n");
    c->is_resp = 1;
}

static void sse_done(struct mg_connection *c) {
    mg_http_printf_chunk(c, "data: [DONE]\n\n");
    mg_http_write_chunk(c, "", 0);
    c->is_draining = 1;
}

static void handle_generation(struct mg_connection *c, server_t *srv, cJSON *req, int is_chat) {
    char err[256] = {0};
    char idbuf[64];
    const char *model_id = json_str(req, "model", NULL);
    model_t *e = find_model(srv, model_id);
    char *prompt_text;
    int *tokens = NULL, np = 0, i;
    gen_req_t p;
    gen_stats_t st;
    sink_t sink;
    int *hist = NULL;
    int stream = json_bool(req, "stream", 0);
    cJSON *stops;

    if (e == NULL) {
        snprintf(err, sizeof err, "unknown model '%s'", model_id == NULL ? "" : model_id);
        send_error(c, 404, "invalid_request_error", err);
        return;
    }
    if (ensure_resident(e, srv) != 0) {
        snprintf(err, sizeof err, "model '%s' failed to load", e->id);
        send_error(c, 503, "server_error", err);
        return;
    }
    prompt_text = is_chat ? (strcmp(e->chat_format, "plain") == 0 ? chat_prompt_plain(req)
                                                                  : chat_prompt_chatml(req))
                          : completion_prompt(req);
    if (prompt_text == NULL) {
        send_error(c, 400, "invalid_request_error",
                   is_chat ? "messages must be a non-empty array" : "prompt must be a string");
        return;
    }
    memset(&p, 0, sizeof p);
    p.max_tokens = json_int(req, "max_tokens", 0);
    if (p.max_tokens <= 0) p.max_tokens = json_int(req, "max_completion_tokens", 0);
    if (p.max_tokens <= 0) p.max_tokens = srv->default_max_tokens;
    if (p.max_tokens > srv->max_req_tokens) p.max_tokens = srv->max_req_tokens;
    p.temperature = (float)json_num(req, "temperature", 0.0);
    if (p.temperature < 0.0f) p.temperature = 0.0f;
    p.use_spec = json_bool(req, "spec", 1);
    p.top_p = (float)json_num(req, "top_p", 1.0);
    p.top_k = json_int(req, "top_k", 0);
    p.repeat_penalty = (float)json_num(req, "repetition_penalty",
                                       json_num(req, "repeat_penalty", 1.0));
    p.repeat_last_n = json_int(req, "repeat_last_n", 64);
    p.block = json_int(req, "spec_block", 0);
    p.seed = (unsigned long long)json_num(req, "seed", 0.0);
    memset(&st, 0, sizeof st);

    tokens = (int *)calloc((size_t)e->ctx_cap, sizeof(int));
    hist = (int *)calloc((size_t)e->ctx_cap, sizeof(int));
    if (tokens == NULL || hist == NULL) {
        free(prompt_text);
        free(tokens);
        free(hist);
        send_error(c, 500, "server_error", "out of memory");
        return;
    }
    np = bonsai_tokenize(e->m, prompt_text, tokens, e->ctx_cap);
    free(prompt_text);
    if (np <= 0) {
        free(tokens);
        free(hist);
        send_error(c, 400, "invalid_request_error", "failed to tokenize the prompt");
        return;
    }
    if (np + p.max_tokens + 1 > e->ctx_cap) {
        p.max_tokens = e->ctx_cap - np - 1;
        if (p.max_tokens <= 0) {
            free(tokens);
            free(hist);
            send_error(c, 400, "invalid_request_error", "prompt fills the context");
            return;
        }
    }

    memset(&sink, 0, sizeof sink);
    stops = cJSON_GetObjectItem(req, "stop");
    if (stops != NULL) {
        int n = cJSON_IsArray(stops) ? cJSON_GetArraySize(stops) : 1;
        for (i = 0; i < n && sink.n_stops < MAX_STOPS; ++i) {
            cJSON *s = cJSON_IsArray(stops) ? cJSON_GetArrayItem(stops, i) : stops;
            if (s != NULL && cJSON_IsString(s) && s->valuestring[0] != '\0') {
                sink.stop[sink.n_stops++] = s->valuestring;
                if (strlen(s->valuestring) > sink.holdback)
                    sink.holdback = strlen(s->valuestring);
            }
        }
        if (sink.holdback > 0) --sink.holdback;
    }
    sink.is_chat = is_chat;
    sink.model_id = e->id;
    snprintf(idbuf, sizeof idbuf, "%s-%lld", is_chat ? "chatcmpl" : "cmpl",
             (long long)time(NULL));
    sink.id = idbuf;
    if (stream) {
        sink.conn = c;
        sse_headers(c);
        if (is_chat) {
            cJSON *chunk = stream_chunk(&sink, NULL, NULL, 1);
            if (chunk != NULL) { send_sse_json(c, chunk); cJSON_Delete(chunk); }
        }
    }

    st.prompt_tokens = np;
    {
        double t0 = now_sec();
        int rc;
        if (e->dr != NULL && p.use_spec && p.temperature <= 0.0f)
            rc = run_spec_dspine(e, tokens, np, hist, &p, &sink, &st, err, sizeof err);
        else if (e->dm != NULL && p.use_spec && p.temperature <= 0.0f)
            rc = run_spec_gguf(e, tokens, np, hist, &p, &sink, &st, err, sizeof err);
        else
            rc = run_plain(e, tokens, np, hist, &p, &sink, &st, err, sizeof err);
        st.total_sec = now_sec() - t0;
        /* The generation is over, so the row pool must stop spinning before
         * the next request (possibly another model) needs the cores. */
        bonsai_park_workers();
        if (rc != 0) {
            free(tokens);
            free(hist);
            if (stream) {
                cJSON *chunk = cJSON_CreateObject();
                json_add_string(chunk, "error", err[0] ? err : "generation failed");
                send_sse_json(c, chunk);
                cJSON_Delete(chunk);
                sse_done(c);
            } else {
                send_error(c, 500, "server_error", err[0] ? err : "generation failed");
            }
            free(sink.text);
            return;
        }
    }
    fprintf(stderr, "[%s] %s: prompt %d, out %d, prefill %.0f ms, decode %.0f ms (%.2f tok/s)" 
                    "%s\n", e->id, st.rounds ? "spec" : "decode", st.prompt_tokens,
            st.completion_tokens, st.prefill_sec * 1e3, st.decode_sec * 1e3,
            st.decode_sec > 0 ? st.completion_tokens / st.decode_sec : 0.0,
            st.rounds ? " [spec]" : "");

    if (stream) {
        const char *finish = st.stop_sequence ? "stop" : (st.hit_eos ? "stop" : "length");
        cJSON *chunk = stream_chunk(&sink, NULL, finish, 0);
        if (chunk != NULL) {
            cJSON_AddItemToObject(chunk, "usage", usage_json(&st));
            perf_json(chunk, e, &st);
            send_sse_json(c, chunk);
            cJSON_Delete(chunk);
        }
        sse_done(c);
    } else {
        cJSON *root = cJSON_CreateObject(), *choices = cJSON_CreateArray();
        cJSON *choice = cJSON_CreateObject();
        const char *finish = st.stop_sequence ? "stop" : (st.hit_eos ? "stop" : "length");
        if (root == NULL || choices == NULL || choice == NULL) {
            cJSON_Delete(choice);
            cJSON_Delete(choices);
            cJSON_Delete(root);
            free(tokens);
            free(hist);
            free(sink.text);
            send_error(c, 500, "server_error", "failed to build response");
            return;
        }
        json_add_string(root, "id", sink.id);
        json_add_string(root, "object", is_chat ? "chat.completion" : "text_completion");
        json_add_number(root, "created", (double)time(NULL));
        json_add_string(root, "model", e->id);
        json_add_number(choice, "index", 0);
        if (is_chat) {
            cJSON *message = cJSON_CreateObject();
            json_add_string(message, "role", "assistant");
            json_add_string(message, "content", sink.text == NULL ? "" : sink.text);
            cJSON_AddItemToObject(choice, "message", message);
        } else {
            json_add_string(choice, "text", sink.text == NULL ? "" : sink.text);
        }
        json_add_string(choice, "finish_reason", finish);
        cJSON_AddItemToArray(choices, choice);
        cJSON_AddItemToObject(root, "choices", choices);
        cJSON_AddItemToObject(root, "usage", usage_json(&st));
        perf_json(root, e, &st);
        send_json(c, 200, root);
        cJSON_Delete(root);
    }
    free(tokens);
    free(hist);
    free(sink.text);
    /* BQ_PROFILE=1 accumulates per-stage timings in the library; dumping them
     * here makes the counters readable request by request (the deltas between
     * two dumps are that request's cost). */
    if (getenv("BQ_PROFILE") != NULL) bonsai_profile_dump();
}

static void handle_models(struct mg_connection *c, server_t *srv) {
    cJSON *root = cJSON_CreateObject(), *data = cJSON_CreateArray();
    int i;
    if (root == NULL || data == NULL) {
        cJSON_Delete(data);
        cJSON_Delete(root);
        send_error(c, 500, "server_error", "failed to build response");
        return;
    }
    json_add_string(root, "object", "list");
    for (i = 0; i < srv->n_models; ++i) {
        model_t *e = &srv->models[i];
        cJSON *m = cJSON_CreateObject(), *meta = cJSON_CreateObject();
        json_add_string(m, "id", e->id);
        json_add_string(m, "object", "model");
        json_add_number(m, "created", (double)time(NULL));
        json_add_string(m, "owned_by", "158bitnet");
        json_add_string(meta, "path", e->path);
        json_add_string(meta, "family", e->loaded ? bonsai_arch_name(bonsai_model_arch(e->m))
                                                  : "unloaded");
        json_add_string(meta, "draft", e->draft[0] ? e->draft : "none");
        json_add_string(meta, "speculation",
                        e->draft[0] == '\0' ? "off"
                                            : (strstr(e->draft, ".safetensors") ? "dspine"
                                                                               : "gguf"));
        /* loaded: the capacity the model actually allows; otherwise the
         * --ctx the server would ask for. */
        json_add_number(meta, "context",
                        e->loaded && e->ctx_cap > 0 ? e->ctx_cap : srv->ctx_cap);
        cJSON_AddItemToObject(meta, "loaded", cJSON_CreateBool(e->loaded ? 1 : 0));
        cJSON_AddItemToObject(m, "meta", meta);
        cJSON_AddItemToArray(data, m);
    }
    cJSON_AddItemToObject(root, "data", data);
    send_json(c, 200, root);
    cJSON_Delete(root);
}

static int is_method(struct mg_http_message *hm, const char *method) {
    return mg_strcmp(hm->method, mg_str(method)) == 0;
}

static void http_handler(struct mg_connection *c, int ev, void *ev_data) {
    server_t *srv = (server_t *)c->fn_data;
    if (ev != MG_EV_HTTP_MSG) return;
    struct mg_http_message *hm = (struct mg_http_message *)ev_data;
    if (is_method(hm, "GET") && mg_match(hm->uri, mg_str("/v1/models"), NULL)) {
        handle_models(c, srv);
    } else if (is_method(hm, "GET") && mg_match(hm->uri, mg_str("/health"), NULL)) {
        mg_http_reply(c, 200, "Content-Type: application/json\r\n",
                      "{\"status\":\"ok\",\"models\":%d}\n", srv->n_models);
    } else if (is_method(hm, "POST") && mg_match(hm->uri, mg_str("/v1/chat/completions"), NULL)) {
        char *body = dup_n(hm->body.buf, hm->body.len);
        cJSON *req = body ? cJSON_Parse(body) : NULL;
        free(body);
        if (req == NULL || !cJSON_IsObject(req)) {
            cJSON_Delete(req);
            send_error(c, 400, "invalid_request_error", "request body must be a JSON object");
            return;
        }
        handle_generation(c, srv, req, 1);
        cJSON_Delete(req);
    } else if (is_method(hm, "POST") && mg_match(hm->uri, mg_str("/v1/completions"), NULL)) {
        char *body = dup_n(hm->body.buf, hm->body.len);
        cJSON *req = body ? cJSON_Parse(body) : NULL;
        free(body);
        if (req == NULL || !cJSON_IsObject(req)) {
            cJSON_Delete(req);
            send_error(c, 400, "invalid_request_error", "request body must be a JSON object");
            return;
        }
        handle_generation(c, srv, req, 0);
        cJSON_Delete(req);
    } else {
        send_error(c, 404, "not_found_error", "endpoint not found");
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--host H] [--port P] [--threads N] [--ctx N] [--preload]\n"
            "          [--default-max-tokens N] [--block N] [--default ID]\n"
            "          [--max-resident N] [--gpu-pct N] [--mm-gpu N] [--no-gpu]\n"
            "  --max-resident keeps at most N models in memory (default 1, 0 = all;\n"
            "  a request for another model unloads the least recently used one).\n"
            "  --gpu-pct/--mm-gpu/--no-gpu are qwen35 env tuning, set before any load.\n"
            "          --model ID=PATH [--draft PATH] [--chat-format chatml|plain] "
            "[--model-block N]\n"
            "  --draft and --chat-format apply to the --model before them.\n"
            "  A .safetensors draft is the DSpine block drafter (qwen35 targets only);\n"
            "  a .gguf draft is an ordinary model whose vocab must match the target.\n",
            argv0);
}

static int parse_model_arg(server_t *srv, const char *spec) {
    const char *eq = strchr(spec, '=');
    model_t *e;
    if (srv->n_models >= MAX_MODELS) {
        fprintf(stderr, "too many models (max %d)\n", MAX_MODELS);
        return -1;
    }
    if (eq == NULL || eq == spec || eq[1] == '\0') {
        fprintf(stderr, "--model needs ID=PATH, got '%s'\n", spec);
        return -1;
    }
    e = &srv->models[srv->n_models];
    memset(e, 0, sizeof *e);
    if ((size_t)(eq - spec) >= sizeof(e->id) || strlen(eq + 1) >= sizeof(e->path)) {
        fprintf(stderr, "model id or path too long\n");
        return -1;
    }
    memcpy(e->id, spec, (size_t)(eq - spec));
    snprintf(e->path, sizeof e->path, "%s", eq + 1);
    snprintf(e->chat_format, sizeof e->chat_format, "chatml");
    e->block = 0;   /* resolved to --block after every model is registered */
    ++srv->n_models;
    return 0;
}

int main(int argc, char **argv) {
    server_t srv;
    struct mg_mgr mgr;
    char listen_url[256];
    const char *default_id = NULL;
    const char *gpu_pct = NULL, *mm_gpu = NULL;
    int gpu_off = 0;
    int i;

    memset(&srv, 0, sizeof srv);
    srv.host = "0.0.0.0";
    srv.port = "8080";
    srv.threads = 8;
    srv.ctx_cap = DEFAULT_CTX;
    srv.default_max_tokens = DEFAULT_MAX_TOKENS;
    srv.max_req_tokens = MAX_REQ_TOKENS;
    srv.default_block = 2;
    srv.max_resident = 1;

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(a, "--host") == 0 && i + 1 < argc) {
            srv.host = argv[++i];
        } else if (strcmp(a, "--port") == 0 && i + 1 < argc) {
            srv.port = argv[++i];
        } else if (strcmp(a, "--threads") == 0 && i + 1 < argc) {
            srv.threads = atoi(argv[++i]);
        } else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) {
            srv.ctx_cap = atoi(argv[++i]);
        } else if (strcmp(a, "--default-max-tokens") == 0 && i + 1 < argc) {
            srv.default_max_tokens = atoi(argv[++i]);
        } else if (strcmp(a, "--max-tokens") == 0 && i + 1 < argc) {
            srv.max_req_tokens = atoi(argv[++i]);
        } else if (strcmp(a, "--block") == 0 && i + 1 < argc) {
            srv.default_block = atoi(argv[++i]);
        } else if (strcmp(a, "--default") == 0 && i + 1 < argc) {
            default_id = argv[++i];
        } else if (strcmp(a, "--preload") == 0) {
            srv.preload = 1;
        } else if (strcmp(a, "--max-resident") == 0 && i + 1 < argc) {
            srv.max_resident = atoi(argv[++i]);
            if (srv.max_resident < 0) srv.max_resident = 0;
        } else if (strcmp(a, "--gpu-pct") == 0 && i + 1 < argc) {
            gpu_pct = argv[++i];
        } else if (strcmp(a, "--mm-gpu") == 0 && i + 1 < argc) {
            mm_gpu = argv[++i];
        } else if (strcmp(a, "--no-gpu") == 0) {
            gpu_off = 1;
        } else if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            if (parse_model_arg(&srv, argv[++i]) != 0) return 1;
        } else if (strcmp(a, "--draft") == 0 && i + 1 < argc) {
            if (srv.n_models == 0) {
                fprintf(stderr, "--draft must follow a --model\n");
                return 1;
            }
            snprintf(srv.models[srv.n_models - 1].draft,
                     sizeof srv.models[srv.n_models - 1].draft, "%s", argv[++i]);
        } else if (strcmp(a, "--chat-format") == 0 && i + 1 < argc) {
            if (srv.n_models == 0) {
                fprintf(stderr, "--chat-format must follow a --model\n");
                return 1;
            }
            snprintf(srv.models[srv.n_models - 1].chat_format,
                     sizeof srv.models[srv.n_models - 1].chat_format, "%s", argv[++i]);
        } else if (strcmp(a, "--model-block") == 0 && i + 1 < argc) {
            if (srv.n_models == 0) {
                fprintf(stderr, "--model-block must follow a --model\n");
                return 1;
            }
            srv.models[srv.n_models - 1].block = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown argument '%s'\n", a);
            usage(argv[0]);
            return 1;
        }
    }
    if (srv.n_models == 0) {
        usage(argv[0]);
        return 1;
    }
    if (default_id != NULL) {
        int found = 0;
        for (i = 0; i < srv.n_models; ++i)
            if (strcmp(srv.models[i].id, default_id) == 0) { srv.default_index = i; found = 1; }
        if (!found) {
            fprintf(stderr, "--default '%s' is not a registered model\n", default_id);
            return 1;
        }
    }
    for (i = 0; i < srv.n_models; ++i)
        if (srv.models[i].block <= 0) srv.models[i].block = srv.default_block;
    if (srv.ctx_cap < 512) srv.ctx_cap = 512;
    if (srv.ctx_cap > MAX_CTX) srv.ctx_cap = MAX_CTX;
    if (srv.default_max_tokens > srv.ctx_cap - 64)
        srv.default_max_tokens = srv.ctx_cap - 64;
    if (srv.max_req_tokens > srv.ctx_cap - 64) srv.max_req_tokens = srv.ctx_cap - 64;
    /* The block verify is where the 2-bit i8mm repack pays off, so switch it
     * on for the whole process unless the caller decided (as the tools do). */
    if (getenv("BQ_P2") == NULL) setenv("BQ_P2", "1", 0);
    /* qwen35 GPU tuning is read when a model is loaded; the minicpm family
     * uses its own knob, so these only steer the qwen35 targets.  On the Cix
     * board the Mali earns nothing once the CPU has the i8mm kernel, and the
     * measured best point is 10% of the matvec rows on the GPU. */
    if (gpu_pct != NULL) setenv("BQ_GPU_PCT", gpu_pct, 1);
    if (mm_gpu != NULL) setenv("BQ_MM_GPU", mm_gpu, 1);
    if (gpu_off) setenv("BQ_GPU", "0", 1);
    /* --threads is the process-wide count, but the BitCPM family has no
     * other way in: src/minicpm.c's projections and the output pool read
     * BITNET_NUM_THREADS once, before their first use, and default to three.
     * Without this a --threads 8 server still decoded the 0.5B on three
     * threads (43 tok/s instead of 50 on the Cix board). */
    {
        char tbuf[16];
        snprintf(tbuf, sizeof(tbuf), "%d", srv.threads);
        setenv("BITNET_NUM_THREADS", tbuf, 1);
    }

    fprintf(stderr, "bonsai_server: %d model(s), ctx %d, threads %d, max-resident %d\n",
            srv.n_models, srv.ctx_cap, srv.threads, srv.max_resident);
    for (i = 0; i < srv.n_models; ++i)
        fprintf(stderr, "  %-12s %s%s%s\n", srv.models[i].id, srv.models[i].path,
                srv.models[i].draft[0] ? "  draft=" : "",
                srv.models[i].draft[0] ? srv.models[i].draft : "");
    if (srv.preload) {
        int budget = srv.max_resident > 0 ? srv.max_resident : srv.n_models;
        int n = 0;
        if (budget > 0) {
            if (ensure_resident(&srv.models[srv.default_index], &srv) == 0) ++n;
            else fprintf(stderr, "preload of %s failed\n", srv.models[srv.default_index].id);
        }
        for (i = 0; i < srv.n_models && n < budget; ++i) {
            if (i == srv.default_index) continue;
            if (load_model(&srv.models[i], &srv) == 0) {
                srv.models[i].last_use = ++srv.use_clock;
                ++n;
            } else {
                fprintf(stderr, "preload of %s failed\n", srv.models[i].id);
            }
        }
        fprintf(stderr, "[preload] %d model(s) resident (budget %d)\n", n, budget);
    }

    snprintf(listen_url, sizeof listen_url, "http://%s:%s", srv.host, srv.port);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    mg_log_set(MG_LL_ERROR);
    mg_mgr_init(&mgr);
    if (mg_http_listen(&mgr, listen_url, http_handler, &srv) == NULL) {
        fprintf(stderr, "failed to listen on %s\n", listen_url);
        mg_mgr_free(&mgr);
        return 1;
    }
    fprintf(stderr, "listening on %s\n", listen_url);
    while (!g_stop) mg_mgr_poll(&mgr, 100);
    mg_mgr_free(&mgr);
    for (i = 0; i < srv.n_models; ++i) free_model(&srv.models[i]);
    return 0;
}
