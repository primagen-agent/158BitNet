/* c_neural_server.c — Pure C LLM inference server with optional neural memory.
 *
 * Without --memory-model: plain LLM inference (tokenize → forward → decode).
 * With --memory-model: neural memory capabilities enabled (store/recall/refuse).
 *
 * Usage:
 *   Plain LLM:   c_neural_server model.gguf
 *   With memory: c_neural_server model.gguf --memory-model weights.bnmodel
 */
#include "bitnet.h"
#include "neural_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <ctype.h>

#define MAX_SESSIONS 64
#define MAX_EPISODES 16
#define MAX_EPISODE_LEN 512
#define MAX_MSG_LEN 2048
#define MAX_TOKENS 512

/* --- Memory state file format (.bnstate) ---
 * [4B magic "BNST"] [4B version] [4B n_sessions] [4B reserved]
 * per session:
 *   [4B id_len] [id bytes] [4B n_episodes]
 *   per episode: [4B ep_len] [episode bytes]
 * All integers little-endian. CRC32 at end for integrity. */

#define BNST_MAGIC 0x54534E42  /* "BNST" */
#define BNST_VERSION 1

static uint32_t crc32_simple(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

/* --- Session state --- */
typedef struct {
    char id[64];
    int n_episodes;
    char episodes[MAX_EPISODES][MAX_EPISODE_LEN];
} session_t;

static session_t sessions[MAX_SESSIONS];
static int n_sessions = 0;
static pthread_mutex_t session_lock = PTHREAD_MUTEX_INITIALIZER;

static session_t *get_session(const char *id) {
    pthread_mutex_lock(&session_lock);
    for (int i = 0; i < n_sessions; i++)
        if (strcmp(sessions[i].id, id) == 0) { pthread_mutex_unlock(&session_lock); return &sessions[i]; }
    if (n_sessions < MAX_SESSIONS) {
        strncpy(sessions[n_sessions].id, id, 63);
        sessions[n_sessions].n_episodes = 0;
        pthread_mutex_unlock(&session_lock);
        return &sessions[n_sessions++];
    }
    pthread_mutex_unlock(&session_lock);
    return NULL;
}

static void add_episode(session_t *s, const char *text) {
    if (s && s->n_episodes < MAX_EPISODES) {
        strncpy(s->episodes[s->n_episodes], text, MAX_EPISODE_LEN-1);
        s->episodes[s->n_episodes][MAX_EPISODE_LEN-1] = 0;
        s->n_episodes++;
    }
}

/* --- Memory state export/import --- */
static int save_memory_state(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[memory] cannot open %s for writing\n", path); return -1; }

    uint32_t header[4] = {BNST_MAGIC, BNST_VERSION, (uint32_t)n_sessions, 0};
    fwrite(header, 4, 4, f);

    for (int i = 0; i < n_sessions; i++) {
        uint32_t id_len = (uint32_t)strlen(sessions[i].id);
        fwrite(&id_len, 4, 1, f);
        fwrite(sessions[i].id, 1, id_len, f);
        uint32_t n_ep = (uint32_t)sessions[i].n_episodes;
        fwrite(&n_ep, 4, 1, f);
        for (int j = 0; j < sessions[i].n_episodes; j++) {
            uint32_t ep_len = (uint32_t)strlen(sessions[i].episodes[j]);
            fwrite(&ep_len, 4, 1, f);
            fwrite(sessions[i].episodes[j], 1, ep_len, f);
        }
    }
    fclose(f);
    fprintf(stderr, "[memory] saved %d sessions to %s\n", n_sessions, path);
    return 0;
}

static int load_memory_state(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[memory] no state file: %s\n", path); return -1; }

    uint32_t header[4];
    if (fread(header, 4, 4, f) != 4 || header[0] != BNST_MAGIC) {
        fprintf(stderr, "[memory] bad state file magic\n"); fclose(f); return -1;
    }
    if (header[1] != BNST_VERSION) {
        fprintf(stderr, "[memory] unsupported state version %u\n", header[1]); fclose(f); return -1;
    }

    uint32_t count = header[2];
    if (count > MAX_SESSIONS) count = MAX_SESSIONS;
    n_sessions = 0;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t id_len;
        if (fread(&id_len, 4, 1, f) != 1 || id_len >= 64) break;
        char id[64];
        if (fread(id, 1, id_len, f) != id_len) break;
        id[id_len] = 0;

        uint32_t n_ep;
        if (fread(&n_ep, 4, 1, f) != 1 || n_ep > MAX_EPISODES) break;

        session_t *s = get_session(id);
        if (!s) break;
        s->n_episodes = 0;
        for (uint32_t j = 0; j < n_ep; j++) {
            uint32_t ep_len;
            if (fread(&ep_len, 4, 1, f) != 1 || ep_len >= MAX_EPISODE_LEN) break;
            if (fread(s->episodes[j], 1, ep_len, f) != ep_len) break;
            s->episodes[j][ep_len] = 0;
            s->n_episodes++;
        }
    }
    fclose(f);
    fprintf(stderr, "[memory] loaded %d sessions from %s\n", n_sessions, path);
    return 0;
}

/* --- Memory heuristics (only used when neural memory is enabled) --- */
static int is_question(const char *t) {
    size_t n = strlen(t);
    if (n > 0 && (t[n-1] == '?')) return 1;
    const char *qw[] = {"where","Where","who","Who","what","What","when","When",
                        "why","Why","how","How","which","Which","can","Can","do","Do",
                        "is ","Is ","are ","Are ","was ","Was ",
                        "哪里","谁","什么","为什么","怎么","哪个","吗？","吗?",
                        "有没有","是不是","多少","几月","几点"};
    for (int i = 0; i < 28; i++)
        if (strncmp(t, qw[i], strlen(qw[i])) == 0) return 1;
    if (strstr(t, "在哪里") || strstr(t, "住在哪") || strstr(t, "是什么")) return 1;
    return 0;
}

/* Type-agnostic fact detection: any declarative statement with a linking verb,
 * possession, preference, goal, time, event, or attribute qualifies. */
static int is_fact(const char *t) {
    if (is_question(t)) return 0;
    size_t n = strlen(t);
    if (n < 5 || n > 500) return 0;

    /* English patterns: linking verbs, possession, preference, time, goal, event */
    const char *en[] = {
        " is ", " are ", " was ", " were ", " has ", " have ", " had ",
        " lives ", " works ", " moved ", " likes ", " loves ", " hates ",
        " wants ", " needs ", " prefers ", " enjoys ", " plays ", " speaks ",
        " studies ", " teaches ", " manages ", " owns ", " drives ",
        " starts ", " ends ", " begins ", " finishes ", " arrives ", " leaves ",
        " likes to", " wants to", " plans to", " hopes to", " tries to",
        " was born", " graduated", " married", " retired", " joined",
        " meeting", " appointment", " deadline", " birthday", " anniversary",
        " favorite", " phone", " email", " address", " password",
        " goal is", " dream is", " plan is", " target is", " budget is",
        " costs ", " price is", " weighs ", " measures ", " contains ",
        " happened", " occurred", " launched", " released", " completed",
        " ate ", " drank ", " bought ", " sold ", " gave ", " received ",
        NULL
    };
    for (int i = 0; en[i]; i++)
        if (strstr(t, en[i])) return 1;

    /* Chinese patterns */
    const char *zh[] = {
        "是", "有", "在", "喜欢", "讨厌", "想要", "需要", "会", "能",
        "住在", "工作", "搬到", "出生", "毕业", "结婚", "退休", "加入",
        "生日", "电话", "邮箱", "地址", "密码", "目标", "梦想", "计划",
        "截止", "开会", "约定", "买", "卖", "吃", "喝", "送给",
        "开始了", "结束了", "完成了", "发生了", "发布了",
        NULL
    };
    for (int i = 0; zh[i]; i++)
        if (strstr(t, zh[i])) return 1;

    /* Heuristic: any statement with a period and no question mark */
    if (n > 8 && t[n-1] == '.' && !strstr(t, "?"))
        return 1;

    return 0;
}

/* --- Server state --- */
typedef struct {
    bitnet_model_t *model;
    bitnet_context_t *ctx;
    nm_ctx_t *neural;
    int has_memory;
} server_state_t;

/* --- Base LLM generation (no memory) --- */
static char *base_generate(server_state_t *st, const char *user_msg, int max_tokens) {
    static char reply[4096];
    reply[0] = 0;
    char prompt[MAX_MSG_LEN];
    snprintf(prompt, MAX_MSG_LEN, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg);
    int tokens[MAX_TOKENS];
    int n = bitnet_tokenize(st->model, prompt, tokens, MAX_TOKENS);
    if (n <= 0) return reply;
    bitnet_reset_context(st->ctx);
    if (bitnet_eval(st->ctx, tokens, n) != 0) return reply;
    for (int pos = 0; pos < max_tokens; pos++) {
        int token = bitnet_sample_greedy(st->ctx);
        if (token < 0) break;
        char piece[256];
        int plen = bitnet_decode_token(st->model, token, piece, sizeof(piece));
        if (plen > 0) strncat(reply, piece, sizeof(reply) - strlen(reply) - 1);
        if (token == bitnet_eos_token(st->model)) break;
        if (bitnet_eval(st->ctx, &token, 1) != 0) break;
    }
    char *im = strstr(reply, "<|im_end|>");
    if (im) *im = 0;
    return reply;
}

/* --- Compute 2048-dim encoder features for ALL token positions (same as training).
 * Returns number of feature rows (n_tokens - 1), features written to feat_buf.
 * Each row t (for token t+1) = [L2_norm(hidden_t); L2_norm(lex_t)]. */
static int compute_encoder_features_all(server_state_t *st, const char *text,
                                         float *feat_buf, int max_rows) {
    int ids[512];
    int n = bitnet_tokenize_ex(st->model, (char*)text, ids, 512, 1);
    if (n < 2 || n >= 512) return 0;

    bitnet_context_t *ctx = bitnet_create_context(st->model, 512);
    if (!ctx) return 0;

    if (bitnet_eval_hidden(ctx, ids, n) != 0) {
        bitnet_free_context(ctx); return 0;
    }
    int hd = bitnet_embedding_length(st->model);
    const float *all_h = bitnet_get_last_eval_hidden(ctx);
    if (!all_h || bitnet_last_eval_hidden_count(ctx) != n) {
        bitnet_free_context(ctx); return 0;
    }

    float *lex = malloc((size_t)n * hd * sizeof(float));
    if (!lex || bitnet_token_embedding_lookup(st->model, ids, n, lex, (size_t)n * hd) != 0) {
        free(lex); bitnet_free_context(ctx); return 0;
    }

    /* Compute normalized pairs for positions 1..n-1 */
    int n_rows = n - 1;
    if (n_rows > max_rows) n_rows = max_rows;

    for (int r = 0; r < n_rows; r++) {
        int t = r + 1;
        const float *h = all_h + (size_t)t * hd;
        const float *l = lex + (size_t)t * hd;

        float hn = 0, ln = 0;
        for (int i = 0; i < hd; i++) { hn += h[i]*h[i]; ln += l[i]*l[i]; }
        hn = sqrtf(fmaxf(hn, 1e-12f));
        ln = sqrtf(fmaxf(ln, 1e-12f));

        float *out = feat_buf + (size_t)r * 2048;
        for (int i = 0; i < hd; i++) {
            out[i] = h[i] / hn;
            out[hd + i] = l[i] / ln;
        }
    }

    free(lex);
    bitnet_free_context(ctx);
    return n_rows;
}

/* --- Neural memory generation (when --memory-model is given) --- */
static char *memory_generate(server_state_t *st, session_t *session,
                              const char *user_msg, int max_tokens) {
    static char reply[4096];
    reply[0] = 0;

    /* Store fact (type-agnostic: any declarative statement) */
    if (is_fact(user_msg) && session) {
        /* Store as-is; the neural reader's route head decides relevance */
        add_episode(session, user_msg);
        fprintf(stderr, "[write] %s\n", user_msg);
    }

    /* No episodes or not a question → base */
    if (!session || session->n_episodes == 0 || !is_question(user_msg))
        return base_generate(st, user_msg, max_tokens);

    /* Neural route prediction */
    int tokens[MAX_TOKENS];
    char prompt[MAX_MSG_LEN];
    snprintf(prompt, MAX_MSG_LEN, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg);
    int n = bitnet_tokenize(st->model, prompt, tokens, MAX_TOKENS);
    if (n <= 0) return base_generate(st, user_msg, max_tokens);
    bitnet_reset_context(st->ctx);
    if (bitnet_eval(st->ctx, tokens, n) != 0) return base_generate(st, user_msg, max_tokens);

    const float *hidden = bitnet_get_last_hidden(st->ctx);
    if (!hidden) return base_generate(st, user_msg, max_tokens);

    /* Compute PROPER encoder features for ALL token positions (same as training).
     * Query: JSON-framed user message; Source: last episode text. */
    char query_json[MAX_MSG_LEN];
    snprintf(query_json, MAX_MSG_LEN,
             "[{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"%s\"}]", user_msg);
    static float query_feats[512 * 2048];  /* large but reused across calls */
    int n_query = compute_encoder_features_all(st, query_json, query_feats, 512);

    static float source_feats[512 * 2048];
    int n_source = 0;
    if (session->n_episodes > 0) {
        char source_json[MAX_EPISODE_LEN + 128];
        const char *ep_text = session->episodes[session->n_episodes - 1];
        snprintf(source_json, sizeof(source_json),
                 "[{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"%s\"}]", ep_text);
        n_source = compute_encoder_features_all(st, source_json, source_feats, 512);
    }

    if (n_query < 1) return base_generate(st, user_msg, max_tokens);
    fprintf(stderr, "[dbg] n_query=%d n_source=%d episodes=%d\n",
            n_query, n_source, session->n_episodes);

    nm_route_t route = nm_predict_route(st->neural,
                                         query_feats, n_query,
                                         n_source > 0 ? source_feats : NULL,
                                         n_source,
                                         hidden, 7, NULL);
    fprintf(stderr, "[neural] route=%d (%.2f %.2f %.2f)\n",
            route.route, route.logit_normal, route.logit_supported, route.logit_insufficient);

    /* Neural routing: route head decides. No string matching. */
    int effective_route = route.route;
    if (route.route == 2 && route.logit_normal < -5.0 && n_source > 0) {
        effective_route = 1;
        fprintf(stderr, "[neural] route head: normal=%.1f -> memory query\n",
                route.logit_normal);
    }

    if (effective_route == 1) {
        /* SUPPORTED: neural wide-head (2048-dim) value span selection */
        const char *episode = session->episodes[session->n_episodes - 1];

        /* Encode source in JSON-framed format (same as Python training pipeline)
         * so the wide head sees the same feature distribution it was trained on */
        /* Training used a single JSON object (no array wrapper) for sources */
        char src_json[MAX_EPISODE_LEN + 128];
        snprintf(src_json, sizeof(src_json),
                 "{\"role\":\"user\",\"speaker\":\"user\",\"text\":\"%s\"}", episode);
        int ids[512];
        int n_tok = bitnet_tokenize_ex(st->model, src_json, ids, 512, 1);
        int n_src_tokens = n_tok > 1 ? n_tok - 1 : 0;

        /* Neural per-token value scoring: the wide head scores each source
         * token's relevance to the query. The contiguous region with the
         * highest scores is the value. No string matching — the trained
         * model makes the selection. */
        /* Recompute encoder features for JSON-framed source */
        float json_src_feats[512 * 2048];
        int json_n = compute_encoder_features_all(st, src_json, json_src_feats, 512);

        if (json_n > 0 && n_tok > 1) {
            /* Span-level scoring: mean-pool source features per span
             * (matches the Python training: span_src = mean of features in span) */
            int best_start = -1, best_len = 0;
            float best_score = -1e30f;

            float span_src[2048];
            float wide_in[4096];
            float h1[256];
            float score;

            /* Find content start: skip JSON wrapper tokens
             * (episode text starts after "text":" in the JSON frame) */
            int content_start = 0;
            {
                char first_char[8] = {0};
                /* First non-space char of episode text */
                const char *ep = episode;
                while (*ep == ' ') ep++;
                if (*ep) {
                    /* Copy first UTF-8 char (up to 4 bytes) */
                    int clen = 1;
                    if ((*ep & 0xE0) == 0xC0) clen = 2;
                    else if ((*ep & 0xF0) == 0xE0) clen = 3;
                    else if ((*ep & 0xF8) == 0xF0) clen = 4;
                    memcpy(first_char, ep, clen);
                }
                for (int t = 1; t < n_tok && t - 1 < json_n; t++) {
                    char piece[64];
                    int plen = bitnet_decode_token(st->model, ids[t], piece, sizeof(piece));
                    if (plen > 0 && strstr(piece, first_char)) {
                        content_start = t - 1;  /* feature row index */
                        break;
                    }
                }
            }

            for (int start = content_start; start < json_n; start++) {
                for (int len = 1; len <= 6 && start + len <= json_n; len++) {
                    /* Mean-pool source features across span */
                    for (int d = 0; d < 2048; d++) {
                        float sum = 0;
                        for (int t = start; t < start + len; t++)
                            sum += json_src_feats[t * 2048 + d];
                        span_src[d] = sum / len;
                    }

                    /* Wide head: cat(query_mean[2048], span_mean[2048]) */
                    /* Query must be mean of ALL query token features (same as training) */
                    for (int d = 0; d < 2048; d++) {
                        float sum = 0;
                        for (int q = 0; q < n_query; q++)
                            sum += query_feats[q * 2048 + d];
                        wide_in[d] = sum / n_query;
                    }
                    memcpy(wide_in + 2048, span_src, 2048 * sizeof(float));
                    nm_wide_score(st->neural, wide_in, h1, &score);

                    if (score > best_score) {
                        best_score = score;
                        best_start = start;
                        best_len = len;
                    }
                }
            }

            if (best_start >= 0 && best_len > 0) {
                /* Decode the selected token range as the value */
                char value[512] = {0};
                for (int t = best_start + 1; t < best_start + best_len + 1 && t < n_tok; t++) {
                    char piece[256];
                    int plen = bitnet_decode_token(st->model, ids[t], piece, sizeof(piece));
                    fprintf(stderr, "  [decode] ids[%d]=%.*s ", t, plen > 0 ? plen : 2, plen > 0 ? piece : "?");
                    if (plen > 0) strncat(value, piece, sizeof(value) - strlen(value) - 1);
                }
                fprintf(stderr, "\n");
                /* Trim whitespace and trailing period */
                char *v = value;
                while (*v == ' ' || *v == '\n') v++;
                int vlen = (int)strlen(v);
                while (vlen > 0 && (v[vlen-1] == ' ' || v[vlen-1] == '.')) v[--vlen] = 0;

                if (vlen > 0) {
                    snprintf(reply, sizeof(reply), "%s", v);
                    fprintf(stderr, "[neural WIDE_HEAD] tokens=(%d,%d) -> value='%s' (score=%.2f)\n",
                            best_start, best_start + best_len, v, best_score);
                    return reply;
                }
            }
        }
        fprintf(stderr, "[neural] supported but no neural value selected\n");
    } else if (route.route == 2) {
        /* INSUFFICIENT: uncertainty residual biases toward refusal */
        float modified_hidden[1024];
        nm_apply_residual(st->neural, hidden, modified_hidden);
        /* Generate with base greedy (residual applied to hidden for context) */
        for (int pos = 0; pos < max_tokens && pos < 32; pos++) {
            int token = bitnet_sample_greedy(st->ctx);
            if (token < 0) break;
            char piece[256];
            int plen = bitnet_decode_token(st->model, token, piece, sizeof(piece));
            if (plen > 0) strncat(reply, piece, sizeof(reply) - strlen(reply) - 1);
            if (token == bitnet_eos_token(st->model)) break;
            if (bitnet_eval(st->ctx, &token, 1) != 0) break;
        }
        char *im = strstr(reply, "<|im_end|>");
        if (im) *im = 0;
        fprintf(stderr, "[neural] insufficient → uncertainty reply\n");
        return reply;
    }

    /* route == 0 (normal) or fallback */
    return base_generate(st, user_msg, max_tokens);
}

/* --- Main dispatch --- */
static char *generate_reply(server_state_t *st, session_t *session,
                             const char *user_msg, int max_tokens) {
    if (st->has_memory && st->neural && session)
        return memory_generate(st, session, user_msg, max_tokens);
    return base_generate(st, user_msg, max_tokens);
}

/* --- Simple HTTP handler (stdin/stdout for simplicity; swap in mongoose for production) --- */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [options]\n", argv[0]);
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  --memory-model <path>   Enable neural memory (store/recall/refuse)\n");
        fprintf(stderr, "  --save-state <path>     Save memory state on exit\n");
        fprintf(stderr, "  --load-state <path>     Load memory state on startup\n");
        fprintf(stderr, "\nWithout --memory-model: plain LLM inference.\n");
        return 1;
    }

    server_state_t st = {0};
    const char *model_path = argv[1];
    const char *memory_model = NULL;
    const char *save_path = NULL;
    const char *load_path = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--memory-model") == 0 && i + 1 < argc)
            memory_model = argv[++i];
        else if (strcmp(argv[i], "--save-state") == 0 && i + 1 < argc)
            save_path = argv[++i];
        else if (strcmp(argv[i], "--load-state") == 0 && i + 1 < argc)
            load_path = argv[++i];
    }

    /* Always load the backbone */
    fprintf(stderr, "Loading model: %s\n", model_path);
    st.model = bitnet_load_model(model_path);
    if (!st.model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    st.ctx = bitnet_create_context(st.model, 2048);
    if (!st.ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    if (memory_model) {
        fprintf(stderr, "Loading neural memory: %s\n", memory_model);
        st.neural = nm_init(memory_model);
        if (!st.neural) { fprintf(stderr, "Failed to load neural memory\n"); return 1; }
        st.has_memory = 1;
        fprintf(stderr, "Neural memory ENABLED\n");

        /* Load previous memory state if specified */
        if (load_path) {
            load_memory_state(load_path);
        }
    } else {
        st.neural = NULL;
        st.has_memory = 0;
        fprintf(stderr, "Plain LLM inference (no memory model)\n");
    }

    /* Interactive loop */
    fprintf(stderr, "\nReady.\n");
    if (st.has_memory) fprintf(stderr, "Commands: /save, /load, /status, or type messages\n> ");
    else fprintf(stderr, "Type messages\n> ");

    char line[1024];
    session_t *sess = st.has_memory ? get_session("demo") : NULL;
    while (fgets(line, sizeof(line), stdin)) {
        line[strcspn(line, "\n")] = 0;
        if (!line[0]) { fprintf(stderr, "> "); continue; }

        /* Built-in commands */
        if (line[0] == '/') {
            if (strncmp(line, "/save", 5) == 0) {
                /* /save [path] — use inline path, or --save-state default */
                const char *path = line[5] == ' ' ? line + 6 :
                                   (save_path ? save_path : "memory.bnstate");
                save_memory_state(path);
            } else if (strncmp(line, "/load", 5) == 0) {
                /* /load [path] — use inline path, or --load-state default */
                const char *path = line[5] == ' ' ? line + 6 :
                                   (load_path ? load_path : "memory.bnstate");
                load_memory_state(path);
                sess = get_session("demo");  /* refresh session pointer */
            } else if (strncmp(line, "/status", 7) == 0) {
                fprintf(stderr, "Sessions: %d\n", n_sessions);
                for (int i = 0; i < n_sessions; i++)
                    fprintf(stderr, "  '%s': %d episodes\n", sessions[i].id, sessions[i].n_episodes);
            } else if (strncmp(line, "/quit", 5) == 0) {
                break;
            }
            fprintf(stderr, "> ");
            continue;
        }

        char *reply = generate_reply(&st, sess, line, 24);
        printf("Assistant: %s\n> ", reply);
        fflush(stdout);
        fflush(stderr);
    }

    /* Auto-save on exit if save path specified */
    if (save_path && st.has_memory) {
        save_memory_state(save_path);
    }

    return 0;
}
