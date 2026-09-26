/* c_neural_server.c — Pure C LLM inference server with optional neural memory.
 *
 * Without --memory-model: plain LLM inference (tokenize → forward → decode).
 * With --memory-model: neural memory capabilities enabled (store/recall/refuse).
 *
 * Memory query path (LC-002 fixes over LC-001):
 *   1. episodes ranked by the trained episode-relevance head (EC-001) over
 *      cached per-episode feature means — not "last episode only";
 *   2. the trained route head decides normal/supported/insufficient on the
 *      top episode (no empirical logit override);
 *   3. supported → wide-head value span selection on the top episode;
 *      insufficient / below tau → uncertainty residual drives refusal through
 *      the backbone's own output projection;
 *   4. query and episode features use the training-matched JSON object frame
 *      (sorted keys, compact separators, escaped text).
 *
 * Usage:
 *   Plain LLM:   c_neural_server model.gguf
 *   With memory: c_neural_server model.gguf --memory-model weights.bnmodel
 */
#include "bitnet.h"
#include "neural_memory.h"
#include "memory_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <ctype.h>

#define MAX_MSG_LEN 2048
#define MAX_TOKENS 512
/* Absolute-score floor for the joint ranking head. EC-007 five-corpus
   world-level holdout Youden point (see reviews/EC-007 tau_analysis). */
#define EP_REL_TAU 0.17f
/* Ranking window (LC-003): plain-C joint scoring costs ~0.5s/episode, so
 * CPU deployment ranks only the most recent episodes. Registered interim
 * limit in reviews/LC-003; remove when the vectorized/Metal kernel lands. */
#define EP_RANK_WINDOW 32

/* --- Memory state: sessions/episodes/means live in memory_state.{c,h} --- */

/* --- Memory heuristics (only used when neural memory is enabled) --- */
static int is_question(const char *t) {
    size_t n = strlen(t);
    if (n > 0 && (t[n-1] == '?')) {
        /* multi-sentence utterances ending in '?' (dialogue turns) still
         * carry facts — only a single-question utterance is a question */
        int has_statement = 0;
        for (size_t i = 0; i + 1 < n; i++) {
            unsigned char c = (unsigned char)t[i];
            if (c == '.' || c == '!' || (c == 0xEF && (unsigned char)t[i+1] == 0xBC)) {
                has_statement = 1; break;
            }
        }
        if (!has_statement) return 1;
    }
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
 * possession, preference, goal, time, event, or attribute qualifies.
 * Storage gating ONLY — never participates in recall. No length cap:
 * memory content is unbounded by design. */
static int is_fact(const char *t) {
    if (is_question(t)) return 0;
    size_t n = strlen(t);
    if (n < 5) return 0;

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

    if (n > 8) {
        /* multi-sentence utterances carry facts regardless of the final
         * punctuation (dialogue turns often end in '?'); single questions
         * were already caught by is_question above */
        int terminators = 0;
        for (size_t i = 0; i < n; i++)
            if (t[i] == '.' || t[i] == '!' || t[i] == '?') terminators++;
        if (terminators >= 2 || t[n-1] == '.')
            return 1;
    }

    return 0;
}

/* --- Server state --- */
typedef struct {
    bitnet_model_t *model;
    bitnet_context_t *ctx;
    nm_ctx_t *neural;
    int has_memory;
} server_state_t;

/* --- JSON framing lives in neural_memory (nm_frame_text): identical to
 * training (json.dumps sort_keys, compact separators, escaped text). --- */

/* Portable arbitrary-length line reader (no getline: MSVC CI). */
static char *read_line(FILE *f, char **buf, size_t *cap) {
    size_t len = 0;
    for (;;) {
        if (*cap - len < 1024) {
            size_t new_cap = *cap ? *cap * 2 : 4096;
            char *grown = realloc(*buf, new_cap);
            if (!grown) return NULL;
            *buf = grown;
            *cap = new_cap;
        }
        if (!fgets(*buf + len, (int)(*cap - len), f))
            return len ? *buf : NULL;
        len += strlen(*buf + len);
        if (len && (*buf)[len - 1] == '\n') {
            (*buf)[len - 1] = 0;
            return *buf;
        }
        if (feof(f)) {
            (*buf)[len] = 0;
            return len ? *buf : NULL;
        }
    }
}

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

static char *frame_buf = NULL;
static size_t frame_cap = 0;

/* --- Compute 2048-dim encoder features for ALL token positions (same as training).
 * Each row t (for token t+1) = [L2_norm(hidden_t); L2_norm(lex_t)].
 * Long texts are truncated at a UTF-8 boundary to the 512-token feature
 * window (memory storage itself stays unbounded); ids_out receives the
 * window's token ids so callers decode pieces from the same tokenization. */
static int compute_encoder_features_all(server_state_t *st, const char *text,
                                         float *feat_buf, int max_rows,
                                         int *ids_out) {
    static char *scratch = NULL;
    static size_t scratch_cap = 0;
    size_t len = strlen(text);
    if (scratch_cap < len + 1) {
        size_t new_cap = scratch_cap ? scratch_cap : 8192;
        while (new_cap < len + 1) new_cap *= 2;
        char *grown = realloc(scratch, new_cap);
        if (!grown) return 0;
        scratch = grown;
        scratch_cap = new_cap;
    }
    memcpy(scratch, text, len + 1);

    int ids[512];
    int n = -1;
    size_t tlen = len;
    while (tlen > 0) {
        n = bitnet_tokenize_ex(st->model, scratch, ids, 512, 1);
        if (n >= 2 && n < 512) break;
        size_t half = tlen / 2;
        while (half > 0 && ((unsigned char)scratch[half] & 0xC0) == 0x80) half--;
        if (half == 0) { n = -1; break; }
        tlen = half;
        scratch[tlen] = 0;
    }
    if (n < 2) return 0;
    if (ids_out) memcpy(ids_out, ids, (size_t)n * sizeof(int));

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

static int argmax_over(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Uncertainty refusal: the trained query-only residual biases generation
 * toward natural refusal. The residual is added to the current position's
 * final-normed hidden and the modified hidden drives the output projection
 * via bitnet_project_hidden_to_logits — the trained branch actually fires. */
static char *uncertainty_reply(server_state_t *st, int max_tokens) {
    static char reply[4096];
    reply[0] = 0;
    int vocab = bitnet_vocab_size(st->model);
    for (int pos = 0; pos < max_tokens && pos < 48; pos++) {
        int token = -1;
        const float *hidden = bitnet_get_last_hidden(st->ctx);
        if (hidden) {
            float modified[1024];
            nm_apply_residual(st->neural, hidden, modified);
            const float *logits = bitnet_project_hidden_to_logits(st->ctx, modified);
            if (logits && vocab > 0)
                token = argmax_over(logits, vocab);
        }
        if (token < 0)
            token = bitnet_sample_greedy(st->ctx);
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

/* --- Neural memory generation (when --memory-model is given) --- */
static char *memory_generate(server_state_t *st, ms_session_t *session,
                              const char *user_msg, int max_tokens) {
    static char reply[4096];
    reply[0] = 0;

    /* Store fact (type-agnostic gate; storage ONLY, never recall).
     * Feature-window rows + derived mean are cached at write time when a
     * ranking head consumes them (joint-pair relevance). */
    if (is_fact(user_msg) && session) {
        ms_add_episode(session, user_msg);
        if (nm_has_episode_joint(st->neural)) {
            static float rows[MS_MAX_ROWS * 2048];
            char frame[8192];
            nm_frame_text_dyn(user_msg, &frame_buf, &frame_cap);
            snprintf(frame, sizeof frame, "%s", frame_buf);
            int n_rows = compute_encoder_features_all(st, frame, rows,
                                                       MS_MAX_ROWS, NULL);
            if (n_rows > 0) {
                int idx = session->n_episodes - 1;
                ms_set_rows(session, idx, rows, n_rows);
                float mean[2048];
                for (int d = 0; d < 2048; d++) {
                    float sum = 0;
                    for (int r = 0; r < n_rows; r++) sum += rows[r * 2048 + d];
                    mean[d] = sum / n_rows;
                }
                ms_set_mean(session, idx, mean);
            }
        }
        fprintf(stderr, "[write] %s\n", user_msg);
    }

    if (!session || session->n_episodes == 0 || !is_question(user_msg))
        return base_generate(st, user_msg, max_tokens);

    /* Prompt forward: prefix hidden for the route head + fallback decoding */
    char prompt[MAX_MSG_LEN];
    snprintf(prompt, MAX_MSG_LEN, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg);
    int ptok[MAX_TOKENS];
    int pn = bitnet_tokenize(st->model, prompt, ptok, MAX_TOKENS);
    if (pn <= 0) return base_generate(st, user_msg, max_tokens);
    bitnet_reset_context(st->ctx);
    if (bitnet_eval(st->ctx, ptok, pn) != 0)
        return base_generate(st, user_msg, max_tokens);

    /* Query features: training frames the query context as a one-element
     * ARRAY; episodes use the single-object frame. Dynamic buffers —
     * memory content is unbounded. */
    static float query_feats[512 * 2048];
    static char *qframe = NULL;
    static size_t qframe_cap = 0;
    nm_frame_query_dyn(user_msg, &qframe, &qframe_cap);
    int n_query = compute_encoder_features_all(st, qframe, query_feats, 512, NULL);
    if (n_query < 1) return base_generate(st, user_msg, max_tokens);

    static float query_mean[2048];
    for (int d = 0; d < 2048; d++) {
        float sum = 0;
        for (int t = 0; t < n_query; t++) sum += query_feats[t * 2048 + d];
        query_mean[d] = sum / n_query;
    }

    /* 1. Episode ranking by the trained joint-pair head (not "last only") */
    int best_idx = session->n_episodes - 1;
    if (nm_has_episode_joint(st->neural)) {
        /* recompute row caches missing (v2 state / failed writes) */
        int missing = 0;
        for (int i = 0; i < session->n_episodes; i++)
            if (session->ep_row_counts[i] == 0) missing++;
        if (missing > 0) {
            fprintf(stderr, "[neural] recomputing rows for %d episodes\n", missing);
            static float rows[MS_MAX_ROWS * 2048];
            for (int i = 0; i < session->n_episodes; i++) {
                if (session->ep_row_counts[i] > 0) continue;
                nm_frame_text_dyn(session->episodes[i], &frame_buf, &frame_cap);
                int n_rows = compute_encoder_features_all(st, frame_buf, rows,
                                                           MS_MAX_ROWS, NULL);
                if (n_rows <= 0) continue;
                ms_set_rows(session, i, rows, n_rows);
                float mean[2048];
                for (int d = 0; d < 2048; d++) {
                    float sum = 0;
                    for (int r = 0; r < n_rows; r++) sum += rows[r * 2048 + d];
                    mean[d] = sum / n_rows;
                }
                ms_set_mean(session, i, mean);
            }
        }
        float best_rel = -1.0f;
        int best = -1, scored = 0;
        static float ep_mean[2048];
        int nq_score = n_query > 64 ? 64 : n_query;
        int rank_begin = session->n_episodes > EP_RANK_WINDOW
                             ? session->n_episodes - EP_RANK_WINDOW : 0;
        for (int i = rank_begin; i < session->n_episodes; i++) {
            int cnt = session->ep_row_counts[i];
            if (cnt <= 0 || !session->ep_rows[i]) continue;
            const float *rows_i = session->ep_rows[i];
            if (session->ep_means[i]) {
                memcpy(ep_mean, session->ep_means[i], sizeof ep_mean);
            } else {
                for (int d = 0; d < 2048; d++) ep_mean[d] = 0;
                for (int r = 0; r < cnt; r++)
                    for (int d = 0; d < 2048; d++)
                        ep_mean[d] += rows_i[r * 2048 + d] / cnt;
            }
            float rel = nm_episode_joint_relevance(st->neural, query_feats,
                                                   nq_score, query_mean,
                                                   rows_i, cnt, ep_mean);
            scored++;
            if (rel > best_rel) { best_rel = rel; best = i; }
        }
        fprintf(stderr, "[neural] episode rank: best=%d rel=%.3f (scored %d, window %d/%d)\n",
                best, best_rel, scored,
                session->n_episodes - rank_begin, session->n_episodes);
        if (scored > 0 && best >= 0 && best_rel < EP_REL_TAU) {
            fprintf(stderr, "[neural] no episode above tau %.2f\n", EP_REL_TAU);
            return uncertainty_reply(st, max_tokens);
        }
        if (best >= 0) best_idx = best;
    }

    /* 2. Route head decides on (query, top episode) — trained decision only */
    const char *episode = session->episodes[best_idx];
    static char *sframe = NULL;
    static size_t sframe_cap = 0;
    nm_frame_text_dyn(episode, &sframe, &sframe_cap);
    static float source_feats[512 * 2048];
    int ids[512];
    int n_source = compute_encoder_features_all(st, sframe, source_feats, 512, ids);
    int n_tok = n_source > 0 ? n_source + 1 : 0;
    static char pieces[512][80];
    static char *piece_ptrs[512];
    int n_pieces = 0;
    for (int t = 1; t < n_tok && n_pieces < 512; t++) {
        char piece[80];
        int plen = bitnet_decode_token(st->model, ids[t], piece, sizeof(piece) - 1);
        if (plen < 0) plen = 0;
        if (plen > (int)sizeof(pieces[0]) - 1) plen = (int)sizeof(pieces[0]) - 1;
        memcpy(pieces[n_pieces], piece, (size_t)plen);
        pieces[n_pieces][plen] = 0;
        piece_ptrs[n_pieces] = pieces[n_pieces];
        n_pieces++;
    }

    static int allowed[512];
    static int span_start[8192], span_end[8192];
    int n_spans = -1;
    if (n_source > 0 && n_pieces == n_source)
        n_spans = nm_build_spans(sframe, episode, piece_ptrs, n_pieces,
                                 allowed, span_start, span_end, 8192);
    if (n_spans < 0) {
        fprintf(stderr, "[neural] span layout unavailable (framing mismatch)\n");
        n_spans = 0;
    }

    nm_route_t route = nm_route_decision(st->neural,
                                         query_feats, n_query,
                                         n_source > 0 ? source_feats : NULL,
                                         n_source,
                                         allowed, n_pieces,
                                         span_start, span_end, n_spans);
    fprintf(stderr, "[neural] route=%d (%.2f %.2f %.2f) spans=%d\n",
            route.route, route.logit_normal, route.logit_supported,
            route.logit_insufficient, n_spans);

    if (route.route == 1 && n_source > 0) {
        /* SUPPORTED: wide-head value span selection on the top episode
         * (ids/n_tok already hold the frame tokenization) */

        /* Find content start: skip the JSON wrapper tokens (episode text
         * begins after "text":" in the frame) */
        int content_start = 0;
        {
            char first_char[8] = {0};
            const char *ep = episode;
            while (*ep == ' ') ep++;
            if (*ep) {
                int clen = 1;
                if ((*ep & 0xE0) == 0xC0) clen = 2;
                else if ((*ep & 0xF0) == 0xE0) clen = 3;
                else if ((*ep & 0xF8) == 0xF0) clen = 4;
                memcpy(first_char, ep, clen);
            }
            for (int t = 1; t < n_tok && t - 1 < n_source; t++) {
                char piece[64];
                int plen = bitnet_decode_token(st->model, ids[t], piece, sizeof(piece));
                if (plen > 0 && strstr(piece, first_char)) {
                    content_start = t - 1;
                    break;
                }
            }
        }

        int best_start = -1, best_len = 0;
        float best_score = -1e30f;
        float span_src[2048];
        float wide_in[4096];
        float h1[256];
        float score;

        for (int start = content_start; start < n_source; start++) {
            for (int len = 1; len <= 6 && start + len <= n_source; len++) {
                for (int d = 0; d < 2048; d++) {
                    float sum = 0;
                    for (int t = start; t < start + len; t++)
                        sum += source_feats[t * 2048 + d];
                    span_src[d] = sum / len;
                }
                memcpy(wide_in, query_mean, 2048 * sizeof(float));
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
            char value[512] = {0};
            for (int t = best_start + 1; t < best_start + best_len + 1 && t < n_tok; t++) {
                char piece[256];
                int plen = bitnet_decode_token(st->model, ids[t], piece, sizeof(piece));
                if (plen > 0) strncat(value, piece, sizeof(value) - strlen(value) - 1);
            }
            char *v = value;
            while (*v == ' ' || *v == '\n') v++;
            int vlen = (int)strlen(v);
            while (vlen > 0 && (v[vlen-1] == ' ' || v[vlen-1] == '.')) v[--vlen] = 0;

            if (vlen > 0) {
                snprintf(reply, sizeof(reply), "%s", v);
                fprintf(stderr, "[neural WIDE_HEAD] ep=%d tokens=(%d,%d) -> value='%s' (score=%.2f)\n",
                        best_idx, best_start, best_start + best_len, v, best_score);
                return reply;
            }
        }
        fprintf(stderr, "[neural] supported but no neural value selected\n");
        return uncertainty_reply(st, max_tokens);
    }

    if (route.route == 2)
        return uncertainty_reply(st, max_tokens);

    /* route == 0 (normal) */
    return base_generate(st, user_msg, max_tokens);
}

/* --- Main dispatch --- */
static char *generate_reply(server_state_t *st, ms_session_t *session,
                             const char *user_msg, int max_tokens) {
    if (st->has_memory && st->neural && session)
        return memory_generate(st, session, user_msg, max_tokens);
    return base_generate(st, user_msg, max_tokens);
}

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
        fprintf(stderr, "Neural memory ENABLED%s\n",
                nm_has_episode_joint(st.neural)
                    ? " (episode ranking)" : "");
        if (load_path) ms_load(load_path);
    } else {
        st.neural = NULL;
        st.has_memory = 0;
        fprintf(stderr, "Plain LLM inference (no memory model)\n");
    }

    fprintf(stderr, "\nReady.\n");
    if (st.has_memory) fprintf(stderr, "Commands: /save, /load, /status, or type messages\n> ");
    else fprintf(stderr, "Type messages\n> ");

    char *line = NULL;
    size_t line_cap = 0;
    ms_session_t *sess = st.has_memory ? ms_get_session("demo") : NULL;
    while (read_line(stdin, &line, &line_cap)) {
        line[strcspn(line, "\n")] = 0;
        if (!line[0]) { fprintf(stderr, "> "); continue; }

        if (line[0] == '/') {
            if (strncmp(line, "/save", 5) == 0) {
                const char *path = line[5] == ' ' ? line + 6 :
                                   (save_path ? save_path : "memory.bnstate");
                if (ms_save(path) == 0)
                    fprintf(stderr, "[memory] saved %d sessions to %s\n",
                            ms_session_count(), path);
            } else if (strncmp(line, "/load", 5) == 0) {
                const char *path = line[5] == ' ' ? line + 6 :
                                   (load_path ? load_path : "memory.bnstate");
                if (ms_load(path) == 0)
                    fprintf(stderr, "[memory] loaded %d sessions from %s\n",
                            ms_session_count(), path);
                else
                    fprintf(stderr, "[memory] load failed: %s\n", path);
                sess = st.has_memory ? ms_find_session("demo") : NULL;
            } else if (strncmp(line, "/status", 7) == 0) {
                fprintf(stderr, "Sessions: %d\n", ms_session_count());
                if (sess)
                    fprintf(stderr, "  '%s': %d episodes (means %s)\n",
                            sess->id, sess->n_episodes,
                            sess->means_valid ? "valid" : "invalid");
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

    if (save_path && st.has_memory) ms_save(save_path);
    free(line);

    return 0;
}
