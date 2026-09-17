#include "bitnet.h"
#include "metis/episodic_store.h"
#include "metis/event_store.h"
#include "metis/memory_snapshot.h"
#include "metis/memory_controller.h"
#include "metis/typed_link_model.h"
#include "metis/typed_pair_encoder.h"
#include "metis/typed_query_activator.h"
#include "metis/typed_writer_model.h"
#include "metis/resident_identity.h"

#include "third_party/cJSON/cJSON.h"
#include "third_party/mongoose/mongoose.h"

#include <signal.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_HOST "127.0.0.1"
#define DEFAULT_PORT "8080"
#define DEFAULT_MODEL_ID "bitnet"
#define DEFAULT_CONTEXT_TOKENS 2048
#define DEFAULT_MAX_TOKENS 1024
#define MAX_REQUEST_TOKENS 4096
#define DEFAULT_REPEAT_LAST_N 64
#define DEFAULT_REPEAT_PENALTY 1.1f
#define MIN_CONTEXT_TOKENS 128

typedef struct server_config {
    const char *host;
    const char *port;
    const char *model_id;
    int max_context_tokens;
    int default_max_tokens;
    int max_request_tokens;
    int repeat_last_n;
    float repeat_penalty;
    const char *lora_path;
    float lora_scale;
    const char *memory_state_dir;
    int episodic_memory;
    const char *memory_controller_path;
    const char *typed_pair_path;
    const char *typed_link_path;
    const char *typed_query_path;
    const char *typed_writer_path;
    const char *resident_path;
} server_config_t;

typedef struct cached_session {
    char *id;
    bitnet_context_t *ctx;
    int *history_tokens;
    int history_count;
    int capacity;
    char *transcript;
    size_t transcript_len;
    size_t transcript_cap;
    metis_episodic_store_t episodic;
    metis_event_store_t events;
    resident_state_t resident;
    size_t resident_selected[4];
    int resident_selected_count;
    time_t last_used;
    struct cached_session *next;
} cached_session_t;

typedef struct server_state {
    bitnet_model_t *model;
    server_config_t cfg;
    cached_session_t *sessions;
    metis_memory_controller_t *memory_controller;
    metis_typed_pair_encoder_t *typed_pair;
    metis_typed_link_model_t *typed_link;
    metis_typed_query_activator_t *typed_query;
    metis_typed_writer_model_t *typed_writer;
    resident_model_t *resident;
} server_state_t;

typedef struct generation_result {
    char *text;
    char *session_id;
    int prompt_tokens;
    int completion_tokens;
    int cached_tokens;
    int reused_tokens;
    int context_tokens;
    int finish_reason_length;
    double prefill_sec;
    double decode_sec;
    double total_sec;
} generation_result_t;

typedef struct generation_state {
    bitnet_context_t *ctx;
    cached_session_t *session;
    int *history_tokens;
    int history_count;
    int capacity;
    int owns_context;
    char *text;
    size_t text_len;
    size_t text_cap;
    char *session_id;
    int prompt_tokens;
    int cached_tokens;
    int reused_tokens;
    int max_tokens;
    int emitted;
    int finish_reason_length;
    char utf8_pending[4];
    int utf8_pending_len;
    int utf8_pending_expected;
    double total_start;
    double prefill_sec;
    double decode_sec;
} generation_state_t;

static volatile sig_atomic_t g_stop = 0;

static void handle_signal(int signo) {
    (void)signo;
    g_stop = 1;
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int parse_int_arg(const char *value, int fallback, int min_value, int max_value) {
    char *end = NULL;
    long parsed = 0;
    if (value == NULL || value[0] == '\0') return fallback;
    parsed = strtol(value, &end, 10);
    if (end == value || parsed < min_value || parsed > max_value) return fallback;
    return (int)parsed;
}

static float parse_float_arg(const char *value, float fallback, float min_value, float max_value) {
    char *end = NULL;
    float parsed = 0.0f;
    if (value == NULL || value[0] == '\0') return fallback;
    parsed = strtof(value, &end);
    if (end == value || parsed < min_value || parsed > max_value) return fallback;
    return parsed;
}

static char *dup_n(const char *data, size_t len) {
    char *out = (char *)malloc(len + 1);
    if (out == NULL) return NULL;
    memcpy(out, data, len);
    out[len] = '\0';
    return out;
}

static int append_bytes(char **buf, size_t *len, size_t *cap, const char *data, size_t data_len) {
    if (data_len == 0) return 0;
    if (*len + data_len + 1 > *cap) {
        size_t next_cap = *cap == 0 ? 256 : *cap;
        char *next = NULL;
        while (*len + data_len + 1 > next_cap) {
            if (next_cap > ((size_t)-1) / 2) return -1;
            next_cap *= 2;
        }
        next = (char *)realloc(*buf, next_cap);
        if (next == NULL) return -1;
        *buf = next;
        *cap = next_cap;
    }
    memcpy(*buf + *len, data, data_len);
    *len += data_len;
    (*buf)[*len] = '\0';
    return 0;
}

static int append_cstr(char **buf, size_t *len, size_t *cap, const char *text) {
    return append_bytes(buf, len, cap, text, text == NULL ? 0 : strlen(text));
}

static int replace_cstr(char **buf, size_t *len, size_t *cap, const char *text) {
    size_t text_len = text == NULL ? 0 : strlen(text);
    char *next = NULL;
    if (text_len + 1 > *cap) {
        next = (char *)realloc(*buf, text_len + 1);
        if (next == NULL) return -1;
        *buf = next;
        *cap = text_len + 1;
    }
    if (text_len > 0) memcpy(*buf, text, text_len);
    (*buf)[text_len] = '\0';
    *len = text_len;
    return 0;
}

static cJSON *json_get_array(cJSON *object, const char *name) {
    cJSON *item = cJSON_GetObjectItem(object, name);
    return item != NULL && cJSON_IsArray(item) ? item : NULL;
}

static const char *json_get_string(cJSON *object, const char *name, const char *fallback) {
    cJSON *item = cJSON_GetObjectItem(object, name);
    return item != NULL && cJSON_IsString(item) && item->valuestring != NULL ?
        item->valuestring : fallback;
}

static int json_get_int(cJSON *object, const char *name, int fallback, int min_value, int max_value) {
    cJSON *item = cJSON_GetObjectItem(object, name);
    int value = fallback;
    if (item != NULL && cJSON_IsNumber(item)) value = item->valueint;
    if (value < min_value) value = min_value;
    if (value > max_value) value = max_value;
    return value;
}

static int json_get_bool(cJSON *object, const char *name, int fallback) {
    cJSON *item = cJSON_GetObjectItem(object, name);
    if (item == NULL) return fallback;
    if (cJSON_IsTrue(item)) return 1;
    if (cJSON_IsFalse(item)) return 0;
    return fallback;
}

static float json_get_float(
    cJSON *object, const char *name, float fallback,
    float min_value, float max_value) {
    cJSON *item = cJSON_GetObjectItem(object, name);
    float value = fallback;
    if (item != NULL && cJSON_IsNumber(item))
        value = (float)item->valuedouble;
    if (!isfinite(value)) value = fallback;
    if (value < min_value) value = min_value;
    if (value > max_value) value = max_value;
    return value;
}

static int token_prefix_equal(const int *tokens, const int *prefix, int n) {
    if (tokens == NULL || prefix == NULL || n < 0) return 0;
    for (int i = 0; i < n; ++i) {
        if (tokens[i] != prefix[i]) return 0;
    }
    return 1;
}

static int utf8_expected_len(unsigned char b) {
    if (b < 0x80u) return 1;
    if (b >= 0xC2u && b <= 0xDFu) return 2;
    if (b >= 0xE0u && b <= 0xEFu) return 3;
    if (b >= 0xF0u && b <= 0xF4u) return 4;
    return 0;
}

static int utf8_is_cont(unsigned char b) {
    return (b & 0xC0u) == 0x80u;
}

static int generation_append_decoded_utf8(generation_state_t *gen,
                                          const char *bytes,
                                          int byte_len,
                                          char *out,
                                          size_t out_size,
                                          char *error,
                                          size_t error_size) {
    char stack_copy[256];
    char *heap_copy = NULL;
    const char *input = bytes;
    size_t out_len = 0;

    if (gen == NULL || bytes == NULL || byte_len <= 0) return 0;

    if (byte_len <= (int)sizeof(stack_copy)) {
        memcpy(stack_copy, bytes, (size_t)byte_len);
        input = stack_copy;
    } else {
        heap_copy = (char *)malloc((size_t)byte_len);
        if (heap_copy == NULL) {
            snprintf(error, error_size, "failed to allocate utf8 buffer");
            return -1;
        }
        memcpy(heap_copy, bytes, (size_t)byte_len);
        input = heap_copy;
    }
    if (out != NULL && out_size > 0) out[0] = '\0';

    for (int i = 0; i < byte_len; ++i) {
        unsigned char b = (unsigned char)input[i];
        char complete[4];
        int complete_len = 0;

        if (gen->utf8_pending_len == 0) {
            int expected = utf8_expected_len(b);
            if (expected == 1) {
                complete[0] = (char)b;
                complete_len = 1;
            } else if (expected > 1) {
                gen->utf8_pending[0] = (char)b;
                gen->utf8_pending_len = 1;
                gen->utf8_pending_expected = expected;
                continue;
            } else {
                continue;
            }
        } else if (utf8_is_cont(b)) {
            gen->utf8_pending[gen->utf8_pending_len++] = (char)b;
            if (gen->utf8_pending_len < gen->utf8_pending_expected) {
                continue;
            }
            memcpy(complete, gen->utf8_pending, (size_t)gen->utf8_pending_expected);
            complete_len = gen->utf8_pending_expected;
            gen->utf8_pending_len = 0;
            gen->utf8_pending_expected = 0;
        } else {
            gen->utf8_pending_len = 0;
            gen->utf8_pending_expected = 0;
            --i;
            continue;
        }

        if (append_bytes(&gen->text, &gen->text_len, &gen->text_cap,
                         complete, (size_t)complete_len) != 0) {
            snprintf(error, error_size, "failed to append decoded text");
            free(heap_copy);
            return -1;
        }
        if (out != NULL && out_size > 0 &&
            out_len + (size_t)complete_len < out_size) {
            memcpy(out + out_len, complete, (size_t)complete_len);
            out_len += (size_t)complete_len;
            out[out_len] = '\0';
        }
    }

    free(heap_copy);
    return 0;
}

static cached_session_t *find_session(server_state_t *state, const char *session_id) {
    if (state == NULL || session_id == NULL || session_id[0] == '\0') return NULL;
    for (cached_session_t *session = state->sessions; session != NULL; session = session->next) {
        if (strcmp(session->id, session_id) == 0) return session;
    }
    return NULL;
}

static void reset_session(cached_session_t *session) {
    if (session == NULL) return;
    bitnet_reset_context(session->ctx);
    session->history_count = 0;
    session->transcript_len = 0;
    if (session->transcript != NULL) session->transcript[0] = '\0';
    metis_episodic_clear(&session->episodic);
    metis_event_store_clear(&session->events);
    resident_state_clear(&session->resident);
}

/* Clear transient conversation/KV state without touching attached memory.
 * Used before importing a committed snapshot into an existing session. */
static void reset_session_context_only(cached_session_t *session) {
    if (session == NULL) return;
    bitnet_reset_context(session->ctx);
    session->history_count = 0;
    session->transcript_len = 0;
    if (session->transcript != NULL) session->transcript[0] = '\0';
}

static cached_session_t *create_session(server_state_t *state, const char *session_id) {
    cached_session_t *session = NULL;
    if (state == NULL || state->model == NULL || session_id == NULL || session_id[0] == '\0') {
        return NULL;
    }
    session = (cached_session_t *)calloc(1, sizeof(*session));
    if (session == NULL) return NULL;
    session->id = dup_n(session_id, strlen(session_id));
    metis_episodic_init(&session->episodic);
    metis_event_store_init(&session->events);
    session->capacity = state->cfg.max_context_tokens;
    session->ctx = bitnet_create_context(state->model, session->capacity);
    session->history_tokens = (int *)calloc((size_t)session->capacity, sizeof(*session->history_tokens));
    if (session->id == NULL || session->ctx == NULL || session->history_tokens == NULL) {
        free(session->history_tokens);
        bitnet_free_context(session->ctx);
        metis_episodic_free(&session->episodic);
        metis_event_store_clear(&session->events);
        resident_state_clear(&session->resident);
        free(session->id);
        free(session);
        return NULL;
    }
    session->last_used = time(NULL);
    session->next = state->sessions;
    state->sessions = session;
    return session;
}

static void free_sessions(server_state_t *state) {
    cached_session_t *session = state == NULL ? NULL : state->sessions;
    while (session != NULL) {
        cached_session_t *next = session->next;
        free(session->history_tokens);
        bitnet_free_context(session->ctx);
        free(session->transcript);
        metis_episodic_free(&session->episodic);
        metis_event_store_clear(&session->events);
        resident_state_clear(&session->resident);
        free(session->id);
        free(session);
        session = next;
    }
    if (state != NULL) state->sessions = NULL;
}

static char *build_chat_prompt_chatml(cJSON *root, int no_think) {
    cJSON *messages = json_get_array(root, "messages");
    char *prompt = NULL;
    size_t len = 0;
    size_t cap = 0;
    int n_messages = messages == NULL ? 0 : cJSON_GetArraySize(messages);

    if (messages == NULL || n_messages <= 0) return NULL;
    for (int i = 0; i < n_messages; ++i) {
        cJSON *msg = cJSON_GetArrayItem(messages, i);
        const char *role = NULL;
        const char *content = NULL;
        if (msg == NULL || !cJSON_IsObject(msg)) continue;
        role = json_get_string(msg, "role", "user");
        content = json_get_string(msg, "content", "");
        if (append_cstr(&prompt, &len, &cap, "<|im_start|>") != 0 ||
            append_cstr(&prompt, &len, &cap, role) != 0 ||
            append_cstr(&prompt, &len, &cap, "\n") != 0 ||
            append_cstr(&prompt, &len, &cap, content) != 0 ||
            append_cstr(&prompt, &len, &cap, "<|im_end|>\n") != 0) {
            free(prompt);
            return NULL;
        }
    }
    /* MiniCPM4-style thinking models: when thinking is disabled, the template
     * injects an empty <think></think> block after the assistant header so the
     * model answers directly instead of reasoning first. */
    if (append_cstr(&prompt, &len, &cap, "<|im_start|>assistant\n") != 0 ||
        (no_think && append_cstr(&prompt, &len, &cap, "<think>\n\n</think>\n") != 0)) {
        free(prompt);
        return NULL;
    }
    return prompt;
}

static char *build_chat_prompt_bitnet_b158(cJSON *root) {
    cJSON *messages = json_get_array(root, "messages");
    char *prompt = NULL;
    size_t len = 0;
    size_t cap = 0;
    char *system_text = NULL;
    size_t system_len = 0;
    size_t system_cap = 0;
    int n_messages = messages == NULL ? 0 : cJSON_GetArraySize(messages);

    if (messages == NULL || n_messages <= 0) return NULL;
    for (int i = 0; i < n_messages; ++i) {
        cJSON *msg = cJSON_GetArrayItem(messages, i);
        const char *role = NULL;
        const char *content = NULL;
        if (msg == NULL || !cJSON_IsObject(msg)) continue;
        role = json_get_string(msg, "role", "user");
        content = json_get_string(msg, "content", "");
        if (strcmp(role, "assistant") == 0) {
            if (append_cstr(&prompt, &len, &cap, content) != 0 ||
                append_cstr(&prompt, &len, &cap, "<|end_of_text|>") != 0) {
                free(prompt);
                free(system_text);
                return NULL;
            }
        } else if (strcmp(role, "system") == 0) {
            if (system_len > 0 &&
                append_cstr(&system_text, &system_len, &system_cap, "\n\n") != 0) {
                free(prompt);
                free(system_text);
                return NULL;
            }
            if (append_cstr(&system_text, &system_len, &system_cap, content) != 0) {
                free(prompt);
                free(system_text);
                return NULL;
            }
        } else {
            if (append_cstr(&prompt, &len, &cap, "Human: ") != 0) {
                free(prompt);
                free(system_text);
                return NULL;
            }
            if (system_len > 0 &&
                (append_bytes(&prompt, &len, &cap, system_text, system_len) != 0 ||
                 append_cstr(&prompt, &len, &cap, "\n\n") != 0)) {
                free(prompt);
                free(system_text);
                return NULL;
            }
            if (append_cstr(&prompt, &len, &cap, content) != 0 ||
                append_cstr(&prompt, &len, &cap, "\n\nBITNETAssistant: ") != 0) {
                free(prompt);
                free(system_text);
                return NULL;
            }
            system_len = 0;
            if (system_text != NULL) system_text[0] = '\0';
        }
    }
    free(system_text);
    return prompt;
}

static char *build_chat_prompt(server_state_t *state, cJSON *root, int enable_thinking) {
    if (state != NULL && bitnet_chat_template_kind(state->model) == 1) {
        return build_chat_prompt_bitnet_b158(root);
    }
    return build_chat_prompt_chatml(root, !enable_thinking);
}

/* enable_thinking: check chat_template_kwargs.enable_thinking then top-level
 * enable_thinking; default true (let the model reason / emit <think>).
 * NOTE: cJSON_IsBool is a macro without a NULL guard -- always check the
 * GetObjectItem result for NULL first. */
static int request_enable_thinking(cJSON *request) {
    cJSON *kw = cJSON_GetObjectItem(request, "chat_template_kwargs");
    if (kw != NULL && cJSON_IsObject(kw)) {
        cJSON *et = cJSON_GetObjectItem(kw, "enable_thinking");
        if (et != NULL && cJSON_IsBool(et)) return cJSON_IsTrue(et) ? 1 : 0;
    }
    {
        cJSON *et = cJSON_GetObjectItem(request, "enable_thinking");
        if (et != NULL && cJSON_IsBool(et)) return cJSON_IsTrue(et) ? 1 : 0;
    }
    return 1;
}

/* Split "<think>REASONING</think>CONTENT" into reasoning + content (in place).
 * If there is no <think>...</think> block, returns text unchanged and
 * *out_reason = NULL. Returned pointers alias into text (valid while text lives). */
static char *split_reasoning(char *text, char **out_reason) {
    char *start;
    char *r;
    char *end;
    *out_reason = NULL;
    start = strstr(text, "<think>");
    if (start == NULL) return text;
    r = start + 7;
    end = strstr(r, "</think>");
    if (end == NULL) return text;          /* unterminated <think>; return whole text */
    *end = '\0';
    if (*r == '\n') r++;                   /* trim leading newline from reasoning */
    *out_reason = r;
    return end + 8 + ((*(end + 8) == '\n') ? 1 : 0);
}

static char *build_completion_prompt(cJSON *root) {
    cJSON *prompt = cJSON_GetObjectItem(root, "prompt");
    if (prompt == NULL) return NULL;
    if (cJSON_IsString(prompt) && prompt->valuestring != NULL) {
        return dup_n(prompt->valuestring, strlen(prompt->valuestring));
    }
    if (cJSON_IsArray(prompt) && cJSON_GetArraySize(prompt) > 0) {
        cJSON *first = cJSON_GetArrayItem(prompt, 0);
        if (first != NULL && cJSON_IsString(first) && first->valuestring != NULL) {
            return dup_n(first->valuestring, strlen(first->valuestring));
        }
    }
    return NULL;
}

static void generation_result_free(generation_result_t *result) {
    if (result == NULL) return;
    free(result->text);
    free(result->session_id);
    memset(result, 0, sizeof(*result));
}

static void generation_state_free(generation_state_t *gen) {
    if (gen == NULL) return;
    if (gen->owns_context) {
        free(gen->history_tokens);
        bitnet_free_context(gen->ctx);
    } else if (gen->session != NULL) {
        gen->session->history_count = gen->history_count;
        gen->session->last_used = time(NULL);
    }
    free(gen->text);
    free(gen->session_id);
    memset(gen, 0, sizeof(*gen));
}

static int prepare_generation(server_state_t *state,
                              const char *prompt,
                              int max_tokens,
                              const char *session_id,
                              int reset_existing_session,
                              generation_state_t *gen,
                              char *error,
                              size_t error_size) {
    int *prompt_tokens = NULL;
    int n_prompt_tokens = 0;
    int prompt_offset = 0;
    const char *session_append_text = NULL;
    size_t session_append_len = 0;
    int replace_session_transcript = 0;
    int context_tokens = 0;
    double prefill_start = 0.0;
    double prefill_end = 0.0;
    cached_session_t *session = NULL;
    int add_bos = 1;

    if (state == NULL || state->model == NULL || prompt == NULL || gen == NULL) {
        snprintf(error, error_size, "invalid generation arguments");
        return -1;
    }
    memset(gen, 0, sizeof(*gen));
    if (max_tokens <= 0) max_tokens = state->cfg.default_max_tokens;
    if (max_tokens > state->cfg.max_request_tokens) max_tokens = state->cfg.max_request_tokens;

    prompt_tokens = (int *)calloc((size_t)state->cfg.max_context_tokens, sizeof(*prompt_tokens));
    if (prompt_tokens == NULL) {
        snprintf(error, error_size, "failed to allocate prompt tokens");
        return -1;
    }

    gen->total_start = monotonic_seconds();
    if (session_id != NULL && session_id[0] != '\0') {
        session = find_session(state, session_id);
        if (session != NULL && reset_existing_session) reset_session(session);
        if (session == NULL) session = create_session(state, session_id);
        if (session == NULL) {
            snprintf(error, error_size, "failed to create session");
            free(prompt_tokens);
            return -1;
        }
        gen->session = session;
        gen->ctx = session->ctx;
        gen->history_tokens = session->history_tokens;
        gen->history_count = session->history_count;
        gen->capacity = session->capacity;
        gen->cached_tokens = session->history_count;
        gen->session_id = dup_n(session->id, strlen(session->id));
        if (gen->session_id == NULL) {
            snprintf(error, error_size, "failed to allocate session id");
            free(prompt_tokens);
            generation_state_free(gen);
            return -1;
        }
        add_bos = session->history_count == 0;
    } else {
        gen->owns_context = 1;
        gen->capacity = state->cfg.max_context_tokens;
    }

    if (session != NULL && gen->history_count > 0 &&
        session->transcript_len > 0 &&
        strncmp(prompt, session->transcript, session->transcript_len) == 0) {
        session_append_text = prompt + session->transcript_len;
        session_append_len = strlen(session_append_text);
        gen->reused_tokens = gen->history_count;
        n_prompt_tokens = bitnet_tokenize_ex(state->model, session_append_text,
                                             prompt_tokens,
                                             state->cfg.max_context_tokens, 0);
    } else if (session != NULL && gen->history_count > 0) {
        int full_prompt_tokens = bitnet_tokenize_ex(state->model, prompt, prompt_tokens,
                                                    state->cfg.max_context_tokens, 1);
        if (full_prompt_tokens >= gen->history_count &&
            token_prefix_equal(prompt_tokens, gen->history_tokens, gen->history_count)) {
            gen->reused_tokens = gen->history_count;
            prompt_offset = gen->history_count;
            n_prompt_tokens = full_prompt_tokens - prompt_offset;
            replace_session_transcript = 1;
        } else {
            n_prompt_tokens = bitnet_tokenize_ex(state->model, prompt, prompt_tokens,
                                                 state->cfg.max_context_tokens, 0);
            session_append_text = prompt;
            session_append_len = strlen(prompt);
        }
    } else {
        n_prompt_tokens = bitnet_tokenize_ex(state->model, prompt, prompt_tokens,
                                             state->cfg.max_context_tokens, add_bos);
        if (session != NULL) {
            session_append_text = prompt;
            session_append_len = strlen(prompt);
        }
    }
    if (n_prompt_tokens < 0 || (n_prompt_tokens == 0 && gen->reused_tokens == 0)) {
        snprintf(error, error_size, "failed to tokenize prompt");
        free(prompt_tokens);
        generation_state_free(gen);
        return -1;
    }

    if (session == NULL) {
        if (n_prompt_tokens + max_tokens + 1 > state->cfg.max_context_tokens) {
            max_tokens = state->cfg.max_context_tokens - n_prompt_tokens - 1;
            if (max_tokens <= 0) {
                snprintf(error, error_size, "no room left for generation");
                free(prompt_tokens);
                generation_state_free(gen);
                return -1;
            }
        }
        context_tokens = n_prompt_tokens + max_tokens + 1;
        if (context_tokens < MIN_CONTEXT_TOKENS) context_tokens = MIN_CONTEXT_TOKENS;
        gen->capacity = context_tokens;
        gen->ctx = bitnet_create_context(state->model, context_tokens);
        gen->history_tokens = (int *)calloc((size_t)context_tokens, sizeof(*gen->history_tokens));
        if (gen->ctx == NULL || gen->history_tokens == NULL) {
            snprintf(error, error_size, "failed to create generation context");
            free(prompt_tokens);
            generation_state_free(gen);
            return -1;
        }
    } else if (gen->history_count + n_prompt_tokens + max_tokens + 1 > gen->capacity) {
        snprintf(error, error_size, "session context is full");
        free(prompt_tokens);
        generation_state_free(gen);
        return -1;
    }

    if (n_prompt_tokens > 0) {
        const int *eval_tokens = prompt_tokens + prompt_offset;
        prefill_start = monotonic_seconds();
        if (bitnet_eval(gen->ctx, eval_tokens, n_prompt_tokens) != 0) {
            snprintf(error, error_size, "prefill eval failed");
            free(prompt_tokens);
            generation_state_free(gen);
            return -1;
        }
        prefill_end = monotonic_seconds();
        memcpy(gen->history_tokens + gen->history_count, eval_tokens,
               (size_t)n_prompt_tokens * sizeof(*gen->history_tokens));
        gen->history_count += n_prompt_tokens;
        if (gen->session != NULL) gen->session->history_count = gen->history_count;
        gen->prefill_sec = prefill_end - prefill_start;
    }
    if (session != NULL) {
        if (replace_session_transcript) {
            if (replace_cstr(&session->transcript, &session->transcript_len,
                             &session->transcript_cap, prompt) != 0) {
                snprintf(error, error_size, "failed to update session transcript");
                free(prompt_tokens);
                generation_state_free(gen);
                return -1;
            }
        } else if (session_append_text != NULL && session_append_len > 0) {
            if (append_bytes(&session->transcript, &session->transcript_len,
                             &session->transcript_cap,
                             session_append_text, session_append_len) != 0) {
                snprintf(error, error_size, "failed to append session transcript");
                free(prompt_tokens);
                generation_state_free(gen);
                return -1;
            }
        }
    }

    gen->prompt_tokens = n_prompt_tokens;
    gen->max_tokens = max_tokens;
    free(prompt_tokens);
    return 0;
}

static int generation_step(server_state_t *state,
                           generation_state_t *gen,
                           char *decoded,
                           size_t decoded_size,
                           int *done,
                           char *error,
                           size_t error_size) {
    int recent_count = 0;
    const int *recent = NULL;
    int next_token = 0;
    double t0 = 0.0;
    double t1 = 0.0;

    if (done != NULL) *done = 0;
    if (decoded != NULL && decoded_size > 0) decoded[0] = '\0';
    if (state == NULL || gen == NULL || gen->ctx == NULL || done == NULL) {
        snprintf(error, error_size, "invalid generation state");
        return -1;
    }
    if (gen->emitted >= gen->max_tokens) {
        gen->finish_reason_length = 1;
        *done = 1;
        return 0;
    }

    recent_count = gen->history_count < state->cfg.repeat_last_n ?
        gen->history_count : state->cfg.repeat_last_n;
    recent = gen->history_tokens + gen->history_count - recent_count;
    next_token = bitnet_sample_greedy_repetition_penalty(gen->ctx, recent,
                                                         recent_count,
                                                         state->cfg.repeat_penalty);
    if (next_token < 0) {
        snprintf(error, error_size, "sampling failed");
        return -1;
    }
    if (bitnet_token_is_eog(state->model, next_token)) {
        gen->finish_reason_length = 0;
        *done = 1;
        return 0;
    }
    if (decoded != NULL && decoded_size > 0) {
        int decoded_len = bitnet_decode_token(state->model, next_token,
                                              decoded, (int)decoded_size);
        if (decoded_len <= 0) decoded[0] = '\0';
        if (generation_append_decoded_utf8(gen, decoded, decoded_len,
                                           decoded, decoded_size,
                                           error, error_size) != 0) {
            return -1;
        }
    }
    if (gen->history_count >= gen->capacity) {
        snprintf(error, error_size, "context capacity exceeded");
        return -1;
    }
    gen->history_tokens[gen->history_count++] = next_token;
    if (gen->session != NULL) gen->session->history_count = gen->history_count;
    t0 = monotonic_seconds();
    if (bitnet_eval(gen->ctx, &next_token, 1) != 0) {
        snprintf(error, error_size, "decode eval failed");
        return -1;
    }
    t1 = monotonic_seconds();
    if (gen->session != NULL && decoded != NULL && decoded[0] != '\0') {
        if (append_cstr(&gen->session->transcript,
                        &gen->session->transcript_len,
                        &gen->session->transcript_cap,
                        decoded) != 0) {
            snprintf(error, error_size, "failed to append session transcript");
            return -1;
        }
    }
    gen->decode_sec += t1 - t0;
    ++gen->emitted;
    if (gen->emitted >= gen->max_tokens) {
        gen->finish_reason_length = 1;
        *done = 1;
    }
    return 0;
}

static int finish_generation(generation_state_t *gen,
                             generation_result_t *result,
                             char *error,
                             size_t error_size) {
    memset(result, 0, sizeof(*result));
    if (gen->text == NULL) {
        gen->text = dup_n("", 0);
        if (gen->text == NULL) {
            snprintf(error, error_size, "failed to allocate empty text");
            return -1;
        }
    }
    result->text = gen->text;
    gen->text = NULL;
    result->session_id = gen->session_id == NULL ? NULL :
        dup_n(gen->session_id, strlen(gen->session_id));
    result->prompt_tokens = gen->prompt_tokens;
    result->completion_tokens = gen->emitted;
    result->cached_tokens = gen->cached_tokens;
    result->reused_tokens = gen->reused_tokens;
    result->context_tokens = gen->history_count;
    result->finish_reason_length = gen->finish_reason_length;
    result->prefill_sec = gen->prefill_sec;
    result->decode_sec = gen->decode_sec;
    result->total_sec = monotonic_seconds() - gen->total_start;
    return 0;
}

static int generate_text(server_state_t *state,
                         const char *prompt,
                         int max_tokens,
                         const char *session_id,
                         int reset_existing_session,
                         generation_result_t *result,
                         char *error,
                         size_t error_size) {
    generation_state_t gen;
    int done = 0;
    char decoded[256];

    if (prepare_generation(
            state, prompt, max_tokens, session_id,
            reset_existing_session,
            &gen, error, error_size) != 0) {
        return -1;
    }
    while (!done) {
        if (generation_step(state, &gen, decoded, sizeof(decoded),
                            &done, error, error_size) != 0) {
            generation_state_free(&gen);
            return -1;
        }
    }
    if (finish_generation(&gen, result, error, error_size) != 0) {
        generation_state_free(&gen);
        return -1;
    }
    generation_state_free(&gen);
    return 0;
}

static int json_add_string(cJSON *object, const char *name, const char *value) {
    return cJSON_AddItemToObject(object, name, cJSON_CreateString(value == NULL ? "" : value));
}

static int json_add_number(cJSON *object, const char *name, double value) {
    return cJSON_AddItemToObject(object, name, cJSON_CreateNumber(value));
}

static int json_add_null(cJSON *object, const char *name) {
    return cJSON_AddItemToObject(object, name, cJSON_CreateNull());
}

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

static void send_error(struct mg_connection *c, int status, const char *type, const char *message) {
    cJSON *root = cJSON_CreateObject();
    cJSON *error = cJSON_CreateObject();
    if (root == NULL || error == NULL) {
        cJSON_Delete(error);
        cJSON_Delete(root);
        mg_http_reply(c, status, "Content-Type: application/json\r\n",
                      "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}\n");
        return;
    }
    json_add_string(error, "message", message);
    json_add_string(error, "type", type);
    cJSON_AddItemToObject(root, "error", error);
    send_json(c, status, root);
    cJSON_Delete(root);
}

static void send_sse_headers(struct mg_connection *c) {
    mg_printf(c,
              "HTTP/1.1 200 OK\r\n"
              "Content-Type: text/event-stream\r\n"
              "Cache-Control: no-cache\r\n"
              "Connection: close\r\n"
              "Transfer-Encoding: chunked\r\n"
              "\r\n");
    c->is_resp = 1;
}

static void send_sse_json(struct mg_connection *c, cJSON *root) {
    char *body = cJSON_Print(root);
    if (body == NULL) return;
    mg_http_printf_chunk(c, "data: %s\n\n", body);
    free(body);
    mg_mgr_poll(c->mgr, 0);
}

static void send_sse_done(struct mg_connection *c) {
    mg_http_printf_chunk(c, "data: [DONE]\n\n");
    mg_http_write_chunk(c, "", 0);
    c->is_draining = 1;
}

static cJSON *build_usage_json(const generation_result_t *result) {
    cJSON *usage = cJSON_CreateObject();
    if (usage == NULL) return NULL;
    json_add_number(usage, "prompt_tokens", result->prompt_tokens);
    json_add_number(usage, "completion_tokens", result->completion_tokens);
    json_add_number(usage, "total_tokens", result->prompt_tokens + result->completion_tokens);
    return usage;
}

static cJSON *build_perf_json(const generation_result_t *result) {
    cJSON *perf = cJSON_CreateObject();
    double decode_tok_s = result->decode_sec > 0.0 ?
        (double)result->completion_tokens / result->decode_sec : 0.0;
    double total_tok_s = result->total_sec > 0.0 ?
        (double)result->completion_tokens / result->total_sec : 0.0;
    if (perf == NULL) return NULL;
    json_add_number(perf, "prefill_sec", result->prefill_sec);
    json_add_number(perf, "decode_sec", result->decode_sec);
    json_add_number(perf, "total_sec", result->total_sec);
    json_add_number(perf, "decode_tok_s", decode_tok_s);
    json_add_number(perf, "total_tok_s", total_tok_s);
    return perf;
}

static cJSON *build_session_json(const generation_result_t *result) {
    cJSON *session = NULL;
    if (result == NULL || result->session_id == NULL) return NULL;
    session = cJSON_CreateObject();
    if (session == NULL) return NULL;
    json_add_string(session, "session_id", result->session_id);
    json_add_number(session, "cached_tokens", result->cached_tokens);
    json_add_number(session, "reused_tokens", result->reused_tokens);
    json_add_number(session, "context_tokens", result->context_tokens);
    return session;
}

static cJSON *build_chat_stream_chunk(server_state_t *state,
                                      const char *id,
                                      const char *content,
                                      const char *finish_reason,
                                      int include_role) {
    cJSON *root = cJSON_CreateObject();
    cJSON *choices = cJSON_CreateArray();
    cJSON *choice = cJSON_CreateObject();
    cJSON *delta = cJSON_CreateObject();
    if (root == NULL || choices == NULL || choice == NULL || delta == NULL) {
        cJSON_Delete(delta);
        cJSON_Delete(choice);
        cJSON_Delete(choices);
        cJSON_Delete(root);
        return NULL;
    }
    json_add_string(root, "id", id);
    json_add_string(root, "object", "chat.completion.chunk");
    json_add_number(root, "created", (double)time(NULL));
    json_add_string(root, "model", state->cfg.model_id);
    json_add_number(choice, "index", 0);
    if (include_role) json_add_string(delta, "role", "assistant");
    if (content != NULL && content[0] != '\0') json_add_string(delta, "content", content);
    cJSON_AddItemToObject(choice, "delta", delta);
    if (finish_reason == NULL) json_add_null(choice, "finish_reason");
    else json_add_string(choice, "finish_reason", finish_reason);
    cJSON_AddItemToArray(choices, choice);
    cJSON_AddItemToObject(root, "choices", choices);
    return root;
}

static cJSON *build_completion_stream_chunk(server_state_t *state,
                                            const char *id,
                                            const char *content,
                                            const char *finish_reason) {
    cJSON *root = cJSON_CreateObject();
    cJSON *choices = cJSON_CreateArray();
    cJSON *choice = cJSON_CreateObject();
    if (root == NULL || choices == NULL || choice == NULL) {
        cJSON_Delete(choice);
        cJSON_Delete(choices);
        cJSON_Delete(root);
        return NULL;
    }
    json_add_string(root, "id", id);
    json_add_string(root, "object", "text_completion");
    json_add_number(root, "created", (double)time(NULL));
    json_add_string(root, "model", state->cfg.model_id);
    json_add_number(choice, "index", 0);
    json_add_string(choice, "text", content == NULL ? "" : content);
    if (finish_reason == NULL) json_add_null(choice, "finish_reason");
    else json_add_string(choice, "finish_reason", finish_reason);
    cJSON_AddItemToArray(choices, choice);
    cJSON_AddItemToObject(root, "choices", choices);
    return root;
}

static void handle_streaming_completion(struct mg_connection *c,
                                        server_state_t *state,
                                        const char *prompt,
                                        int max_tokens,
                                        const char *session_id,
                                        int reset_existing_session,
                                        int is_chat) {
    char error[256];
    char id[64];
    char decoded[256];
    generation_state_t gen;
    int done = 0;
    int failed = 0;

    snprintf(id, sizeof(id), "%s-%lld",
             is_chat ? "chatcmpl" : "cmpl", (long long)time(NULL));
    if (prepare_generation(
            state, prompt, max_tokens, session_id,
            reset_existing_session,
            &gen, error, sizeof(error)) != 0) {
        send_error(c, 500, "server_error", error);
        return;
    }

    send_sse_headers(c);
    if (is_chat) {
        cJSON *role_chunk = build_chat_stream_chunk(state, id, NULL, NULL, 1);
        if (role_chunk != NULL) {
            send_sse_json(c, role_chunk);
            cJSON_Delete(role_chunk);
        }
    }

    while (!done) {
        if (generation_step(state, &gen, decoded, sizeof(decoded),
                            &done, error, sizeof(error)) != 0) {
            cJSON *error_chunk = cJSON_CreateObject();
            if (error_chunk != NULL) {
                json_add_string(error_chunk, "error", error);
                send_sse_json(c, error_chunk);
                cJSON_Delete(error_chunk);
            }
            failed = 1;
            break;
        }
        if (decoded[0] != '\0') {
            cJSON *chunk = is_chat ?
                build_chat_stream_chunk(state, id, decoded, NULL, 0) :
                build_completion_stream_chunk(state, id, decoded, NULL);
            if (chunk != NULL) {
                send_sse_json(c, chunk);
                cJSON_Delete(chunk);
            }
        }
    }
    if (!failed) {
        const char *finish_reason = gen.finish_reason_length ? "length" : "stop";
        cJSON *final_chunk = is_chat ?
            build_chat_stream_chunk(state, id, NULL, finish_reason, 0) :
            build_completion_stream_chunk(state, id, "", finish_reason);
        if (final_chunk != NULL) {
            generation_result_t metrics;
            memset(&metrics, 0, sizeof metrics);
            metrics.session_id = gen.session_id;
            metrics.cached_tokens = gen.cached_tokens;
            metrics.reused_tokens = gen.reused_tokens;
            metrics.context_tokens = gen.history_count;
            metrics.prompt_tokens = gen.prompt_tokens;
            metrics.completion_tokens = gen.emitted;
            if (metrics.session_id != NULL)
                cJSON_AddItemToObject(final_chunk, "bitnet_session", build_session_json(&metrics));
            cJSON_AddItemToObject(final_chunk, "usage", build_usage_json(&metrics));
            send_sse_json(c, final_chunk);
            cJSON_Delete(final_chunk);
        }
    }
    send_sse_done(c);
    generation_state_free(&gen);
}

static void handle_models(struct mg_connection *c, server_state_t *state) {
    cJSON *root = cJSON_CreateObject();
    cJSON *data = cJSON_CreateArray();
    cJSON *model = cJSON_CreateObject();
    if (root == NULL || data == NULL || model == NULL) {
        cJSON_Delete(model);
        cJSON_Delete(data);
        cJSON_Delete(root);
        send_error(c, 500, "server_error", "failed to build response");
        return;
    }
    json_add_string(root, "object", "list");
    json_add_string(model, "id", state->cfg.model_id);
    json_add_string(model, "object", "model");
    json_add_number(model, "created", (double)time(NULL));
    json_add_string(model, "owned_by", "bitnet");
    cJSON_AddItemToArray(data, model);
    cJSON_AddItemToObject(root, "data", data);
    send_json(c, 200, root);
    cJSON_Delete(root);
}

static int memory_session_base_path(
    const server_state_t *state, const char *sid,
    char *out, size_t out_size) {
    size_t n;
    if (state == NULL || state->cfg.memory_state_dir == NULL || sid == NULL)
        return -1;
    n = strlen(sid);
    if (n == 0 || n > 128) return -1;
    for (size_t i = 0; i < n; ++i) {
        unsigned char ch = (unsigned char)sid[i];
        if (!(isalnum(ch) || ch == '-' || ch == '_')) return -1;
    }
    return snprintf(out, out_size, "%s/%s",
                    state->cfg.memory_state_dir, sid) < (int)out_size ? 0 : -1;
}

static const char *last_user_content(cJSON *request) {
    cJSON *messages = cJSON_GetObjectItem(request, "messages");
    if (messages == NULL || !cJSON_IsArray(messages)) return NULL;
    for (int index = cJSON_GetArraySize(messages) - 1; index >= 0; --index) {
        cJSON *message = cJSON_GetArrayItem(messages, index);
        cJSON *role;
        cJSON *content;
        if (message == NULL || !cJSON_IsObject(message)) continue;
        role = cJSON_GetObjectItem(message, "role");
        content = cJSON_GetObjectItem(message, "content");
        if (role != NULL && cJSON_IsString(role) &&
            role->valuestring != NULL &&
            strcmp(role->valuestring, "user") == 0 &&
            content != NULL && cJSON_IsString(content) &&
            content->valuestring != NULL)
            return content->valuestring;
    }
    return NULL;
}

static int contains_ascii_ci(const char *text, const char *needle) {
    size_t needle_length;
    if (text == NULL || needle == NULL) return 0;
    needle_length = strlen(needle);
    if (needle_length == 0) return 1;
    for (; *text != '\0'; ++text) {
        size_t index = 0;
        while (index < needle_length && text[index] != '\0' &&
               tolower((unsigned char)text[index]) ==
               tolower((unsigned char)needle[index]))
            ++index;
        if (index == needle_length) return 1;
    }
    return 0;
}

static int should_store_episodic_record(const char *text) {
    static const char *keywords[] = {
        "remember", "store this", "retain", "record this",
        "update", "replace the old", "delete", "forget",
        "long-term memory", "conversation session",
    };
    if (text == NULL || text[0] == '\0') return 0;
    for (size_t index = 0;
         index < sizeof keywords / sizeof keywords[0]; ++index)
        if (contains_ascii_ci(text, keywords[index])) return 1;
    return 0;
}

typedef enum memory_record_action {
    MEMORY_ACTION_IGNORE = 0,
    MEMORY_ACTION_STORE = 1,
    MEMORY_ACTION_UPDATE = 2,
    MEMORY_ACTION_DELETE = 3,
} memory_record_action_t;

static int controller_classify_action(
    server_state_t *state, const char *text, float *confidence);

static memory_record_action_t inferred_memory_action(const char *text) {
    if (text == NULL || text[0] == '\0') return MEMORY_ACTION_IGNORE;
    if (contains_ascii_ci(text, "forget") ||
        contains_ascii_ci(text, "delete") ||
        contains_ascii_ci(text, "remove from memory") ||
        contains_ascii_ci(text, "unset"))
        return MEMORY_ACTION_DELETE;
    if (contains_ascii_ci(text, "update") ||
        contains_ascii_ci(text, "replace the old") ||
        contains_ascii_ci(text, "is now") ||
        contains_ascii_ci(text, "changed"))
        return MEMORY_ACTION_UPDATE;
    return should_store_episodic_record(text) ?
        MEMORY_ACTION_STORE : MEMORY_ACTION_IGNORE;
}

static int starts_with_ascii_ci(
    const char *text, const char *prefix) {
    if (text == NULL || prefix == NULL) return 0;
    while (*text != '\0' &&
           isspace((unsigned char)*text))
        ++text;
    while (*prefix != '\0') {
        if (*text == '\0' ||
            tolower((unsigned char)*text) !=
            tolower((unsigned char)*prefix))
            return 0;
        ++text;
        ++prefix;
    }
    return *text == '\0' ||
           isspace((unsigned char)*text) ||
           ispunct((unsigned char)*text);
}

static int typed_auto_input_is_query(const char *text) {
    static const char *prefixes[] = {
        "what", "who", "when", "where", "why", "how",
        "which", "is", "are", "was", "were", "do", "does",
        "did", "can", "could", "would", "should", "will",
        "tell me", "give", "show me", "explain",
        "list", "report", "state", "compare", "i need",
        "summarize", "write", "generate", "translate",
        "help", "please",
    };
    const char *end;
    if (text == NULL) return 1;
    end = text + strlen(text);
    while (end > text &&
           isspace((unsigned char)end[-1]))
        --end;
    if (end == text || end[-1] == '?') return 1;
    for (size_t index = 0;
         index < sizeof prefixes / sizeof prefixes[0]; ++index)
        if (starts_with_ascii_ci(text, prefixes[index]))
            return 1;
    return 0;
}

static memory_record_action_t typed_auto_fallback_action(
    const char *text) {
    static const char *fact_cues[] = {
        " is ", " are ", " was ", " has ", " have ",
        " uses ", " prefers ", " likes ", " lives ",
        " works ", " selected ", " chose ", " says ",
        " said ", " appears as ", " represented by ",
        " currently ", " record ", " retain ",
    };
    if (typed_auto_input_is_query(text))
        return MEMORY_ACTION_IGNORE;
    if (contains_ascii_ci(text, "forget") ||
        contains_ascii_ci(text, "delete") ||
        contains_ascii_ci(text, "remove from memory") ||
        contains_ascii_ci(text, "purge"))
        return MEMORY_ACTION_DELETE;
    if (contains_ascii_ci(text, " is now ") ||
        contains_ascii_ci(text, " changed ") ||
        contains_ascii_ci(text, " update ") ||
        contains_ascii_ci(text, " replace ") ||
        contains_ascii_ci(text, " moved on from ") ||
        contains_ascii_ci(text, " no longer ") ||
        contains_ascii_ci(text, " supersed"))
        return MEMORY_ACTION_UPDATE;
    for (size_t index = 0;
         index < sizeof fact_cues / sizeof fact_cues[0]; ++index)
        if (contains_ascii_ci(text, fact_cues[index]))
            return MEMORY_ACTION_STORE;
    return MEMORY_ACTION_IGNORE;
}

static int request_has_memory_action(cJSON *request) {
    cJSON *action = cJSON_GetObjectItem(request, "memory_action");
    return action != NULL && cJSON_IsString(action) &&
           action->valuestring != NULL;
}

static memory_record_action_t request_memory_action(
    server_state_t *state, cJSON *request, const char *text) {
    cJSON *action = cJSON_GetObjectItem(request, "memory_action");
    if (action != NULL && cJSON_IsString(action) &&
        action->valuestring != NULL) {
        if (strcmp(action->valuestring, "store") == 0 ||
            strcmp(action->valuestring, "write") == 0)
            return MEMORY_ACTION_STORE;
        if (strcmp(action->valuestring, "update") == 0)
            return MEMORY_ACTION_UPDATE;
        if (strcmp(action->valuestring, "delete") == 0 ||
            strcmp(action->valuestring, "forget") == 0)
            return MEMORY_ACTION_DELETE;
        if (strcmp(action->valuestring, "ignore") == 0 ||
            strcmp(action->valuestring, "read") == 0)
            return MEMORY_ACTION_IGNORE;
    }
    if (state != NULL && state->memory_controller != NULL &&
        state->memory_controller->action_projection != NULL) {
        int classified = controller_classify_action(
            state, text, NULL);
        if (classified >= MEMORY_ACTION_IGNORE &&
            classified <= MEMORY_ACTION_DELETE)
            return (memory_record_action_t)classified;
        /* A failed learned gate must not silently authorize a heuristic write. */
        fprintf(stderr, "Memory action controller failed; automatic write skipped\n");
        return MEMORY_ACTION_IGNORE;
    }
    return inferred_memory_action(text);
}

static const char *memory_action_name(memory_record_action_t action) {
    switch (action) {
        case MEMORY_ACTION_STORE: return "write";
        case MEMORY_ACTION_UPDATE: return "update";
        case MEMORY_ACTION_DELETE: return "delete";
        case MEMORY_ACTION_IGNORE:
        default: return "ignore";
    }
}

static int controller_classify_action(
    server_state_t *state, const char *text, float *confidence) {
    static const char action_prefix[] =
        "Classify the request as one memory operation: ignore, write, "
        "update, or delete.\nRequest: ";
    static const char action_suffix[] = "\nMemory operation:";
    char *action_text = NULL;
    const char *input = text;
    int tokens[512];
    int count;
    int action;
    bitnet_context_t *context;
    const float *hidden;
    if (state == NULL || state->memory_controller == NULL ||
        state->memory_controller->action_projection == NULL ||
        text == NULL || text[0] == '\0')
        return -1;
    if (state->memory_controller->action_input_projection != NULL) {
        size_t size = strlen(action_prefix) + strlen(text) +
            strlen(action_suffix) + 1u;
        action_text = (char *)malloc(size);
        if (action_text == NULL) return -1;
        snprintf(
            action_text, size, "%s%s%s",
            action_prefix, text, action_suffix);
        input = action_text;
    }
    count = bitnet_tokenize_ex(
        state->model, input, tokens,
        (int)(sizeof tokens / sizeof tokens[0]), 1);
    free(action_text);
    if (count <= 0) return -1;
    context = bitnet_create_context(
        state->model, count < MIN_CONTEXT_TOKENS ?
        MIN_CONTEXT_TOKENS : count + 1);
    if (context == NULL) return -1;
    if (bitnet_eval(context, tokens, count) != 0) {
        bitnet_free_context(context);
        return -1;
    }
    hidden = state->memory_controller->pooling == 2 ?
        bitnet_get_last_pooled_hidden(context) :
        bitnet_get_last_hidden(context);
    action = metis_memory_controller_classify(
        state->memory_controller, hidden, confidence);
    bitnet_free_context(context);
    return action;
}

static int select_typed_active_event(
    server_state_t *state, cached_session_t *session,
    const char *current_text,
    const metis_event_record_t **selected_event,
    float *selected_score, float *exists_score,
    size_t *candidate_count, int query_mode);

typedef struct typed_text_encoding {
    float *hidden;
    float *identity;
    int *tokens;
    size_t token_count;
} typed_text_encoding_t;

static void typed_text_encoding_clear(
    typed_text_encoding_t *encoding) {
    if (encoding == NULL) return;
    free(encoding->hidden);
    free(encoding->identity);
    free(encoding->tokens);
    memset(encoding, 0, sizeof *encoding);
}

static int encode_typed_pair_text(
    server_state_t *state, const char *text,
    typed_text_encoding_t *output) {
    int tokens[256];
    int band_start[256], band_end[256];
    int count;
    int capacity;
    int hidden_dim;
    int bands;
    size_t hidden_count;
    bitnet_context_t *context = NULL;
    const float *hidden;
    if (output != NULL) memset(output, 0, sizeof *output);
    if (state == NULL || state->model == NULL ||
        state->typed_pair == NULL || text == NULL ||
        text[0] == '\0' || output == NULL)
        return -1;
    hidden_dim = state->typed_pair->hidden_dim;
    bands = state->typed_pair->band_count;
    if (bands <= 0 ||
        bands > (int)(sizeof band_start / sizeof band_start[0]))
        return -1;
    for (int band = 0; band < bands; ++band) {
        band_start[band] =
            (int)state->typed_pair->band_start[band];
        band_end[band] =
            (int)state->typed_pair->band_end[band];
    }
    count = bitnet_tokenize_ex(
        state->model, text, tokens,
        (int)(sizeof tokens / sizeof tokens[0]), 1);
    if (count <= 0) return -1;
    capacity = count < MIN_CONTEXT_TOKENS ?
        MIN_CONTEXT_TOKENS : count;
    context = bitnet_create_context(state->model, capacity);
    if (context == NULL ||
        bitnet_context_configure_layer_bands(
            context, band_start, band_end, bands) != 0 ||
        bitnet_eval_hidden(context, tokens, count) != 0)
        goto fail;
    hidden = bitnet_get_last_eval_layer_bands(context);
    if (hidden == NULL ||
        bitnet_last_eval_layer_band_token_count(context) != count ||
        bitnet_layer_band_count(context) != bands)
        goto fail;
    if ((size_t)count > SIZE_MAX / (size_t)bands ||
        (size_t)count * (size_t)bands >
            SIZE_MAX / (size_t)hidden_dim)
        goto fail;
    hidden_count = (size_t)count *
                   (size_t)bands * (size_t)hidden_dim;
    output->hidden = (float *)malloc(
        hidden_count * sizeof(float));
    output->identity = (float *)malloc(
        (size_t)count * (size_t)hidden_dim * sizeof(float));
    output->tokens = (int *)malloc(
        (size_t)count * sizeof(int));
    if (output->hidden == NULL || output->identity == NULL ||
        output->tokens == NULL)
        goto fail;
    memcpy(output->hidden, hidden,
           hidden_count * sizeof(float));
    if (bitnet_token_embedding_lookup(
            state->model, tokens, count, output->identity,
            (size_t)count * (size_t)hidden_dim) != 0)
        goto fail;
    memcpy(
        output->tokens, tokens,
        (size_t)count * sizeof(int));
    output->token_count = (size_t)count;
    bitnet_free_context(context);
    return 0;
fail:
    bitnet_free_context(context);
    typed_text_encoding_clear(output);
    return -1;
}

static int extract_resident_answer(
    server_state_t *state, cached_session_t *session, const char *query,
    char **answer, float *activation_score, float *exists_score,
    float *span_score, size_t *record_index) {
    float scores[RESIDENT_MAX_EVENTS], counts[5]; size_t tokens=0;
    float *features=NULL; char *text=NULL; size_t bytes=1,offset=0;
    if(session->resident.count!=session->events.count)return -1;
    session->resident_selected_count=0;
    features=resident_encode(state->model,query,&tokens);
    if(!features)return -1;
    int rc=resident_read(state->resident,&session->resident,features,tokens,scores,counts);
    free(features);if(rc)return -1;
    int n=resident_select(scores,counts,session->resident.count,session->resident_selected);
    if(n<0)return -1;
    for(int i=0;i<n;i++) {
        const metis_event_record_t *e=&session->events.events[session->resident_selected[i]];
        if(e->raw_record_index>=session->episodic.count)return -1;
        const char *source=session->episodic.records[e->raw_record_index];
        if(e->value_start>=e->value_end||e->value_end>strlen(source)||
           strlen(e->value)!=e->value_end-e->value_start||
           memcmp(source+e->value_start,e->value,e->value_end-e->value_start))return -1;
        bytes+=strlen(e->value)+1;
    }
    text=calloc(bytes,1);if(!text)return -1;
    for(int i=0;i<n;i++) {
        const metis_event_record_t *e=&session->events.events[session->resident_selected[i]];
        if(i)text[offset++]='\n';
        size_t length=strlen(e->value);memcpy(text+offset,e->value,length);offset+=length;
    }
    session->resident_selected_count=n;*answer=text;
    *activation_score=n?scores[session->resident_selected[0]]:0;
    *exists_score=counts[n]-counts[0];*span_score=n?1:0;
    *record_index=n?session->events.events[session->resident_selected[0]].raw_record_index:0;
    return n?1:0;
}

static void resident_recall_metadata(cJSON *root, const cached_session_t *session) {
    cJSON *ids=cJSON_CreateArray();
    if(ids==NULL)return;
    for(int i=0;i<session->resident_selected_count;i++)
        cJSON_AddItemToArray(ids,cJSON_CreateNumber((double)session->resident_selected[i]));
    cJSON_AddItemToObject(root,"selected_event_indices",ids);
    json_add_number(root,"selected_count",session->resident_selected_count);
    json_add_number(root,"resident_events",(double)session->resident.count);
    json_add_number(root,"raw_events_encoded_during_recall",0);
}

static int extract_typed_value_answer(
    server_state_t *state, cached_session_t *session,
    const char *query, char **answer,
    float *activation_score, float *exists_score,
    float *span_score, size_t *record_index) {
    const metis_event_record_t *event = NULL;
    const char *source;
    size_t length;
    size_t candidate_count = 0;
    int selected;
    if (state == NULL || session == NULL ||
        state->typed_pair == NULL || state->typed_link == NULL ||
        query == NULL || answer == NULL ||
        activation_score == NULL || exists_score == NULL ||
        span_score == NULL || record_index == NULL)
        return -1;
    *answer = NULL;
    if (state->resident != NULL)
        return extract_resident_answer(state, session, query, answer,
            activation_score, exists_score, span_score, record_index);
    selected = select_typed_active_event(
        state, session, query, &event,
        activation_score, exists_score, &candidate_count, 1);
    if (selected <= 0) return selected;
    if (event == NULL ||
        event->raw_record_index >=
            session->episodic.count)
        return -1;
    source = session->episodic.records[event->raw_record_index];
    length = strlen(source);
    if (event->value_start >= event->value_end ||
        event->value_end > length ||
        strlen(event->value) !=
            event->value_end - event->value_start ||
        memcmp(
            source + event->value_start, event->value,
            event->value_end - event->value_start) != 0)
        return -1;
    *answer = dup_n(
        source + event->value_start,
        event->value_end - event->value_start);
    if (*answer == NULL) return -1;
    *span_score = 1.0f;
    *record_index = event->raw_record_index;
    return 1;
}

static int event_spans_are_unknown(
    const metis_event_record_t *event) {
    return event->subject_start == METIS_EVENT_SPAN_UNKNOWN &&
           event->subject_end == METIS_EVENT_SPAN_UNKNOWN &&
           event->value_start == METIS_EVENT_SPAN_UNKNOWN &&
           event->value_end == METIS_EVENT_SPAN_UNKNOWN;
}

static int event_spans_fit_source(
    const metis_event_record_t *event, const char *source) {
    size_t length;
    if (event == NULL || source == NULL) return 0;
    if (event_spans_are_unknown(event)) return 1;
    length = strlen(source);
    return event->subject_end > event->subject_start &&
           event->value_end > event->value_start &&
           event->subject_end <= length &&
           event->value_end <= length && event->value != NULL &&
           strlen(event->value) == event->value_end - event->value_start &&
           memcmp(source + event->value_start, event->value,
                  event->value_end - event->value_start) == 0;
}

static void handle_memory_state(
    struct mg_connection *c, server_state_t *state,
    cJSON *request, int do_import) {
    const char *sid = json_get_string(request, "session_id", NULL);
    cached_session_t *session;
    char base_path[1024];
    char episodic_path[1200];
    char event_path[1200];
    metis_episodic_store_t imported_episodic;
    metis_event_store_t imported_events;
    resident_state_t imported_resident = {0};
    char resident_path[1200];
    cJSON *root;

    metis_episodic_init(&imported_episodic);
    metis_event_store_init(&imported_events);
    if (!state->cfg.episodic_memory ||
        state->cfg.memory_state_dir == NULL) {
        send_error(c, 403, "memory_state_disabled",
                   "typed memory state is not configured");
        return;
    }
    if (memory_session_base_path(state, sid, base_path, sizeof base_path) != 0) {
        send_error(c, 400, "invalid_request_error",
                   "invalid session_id");
        return;
    }
    session = find_session(state, sid);
    if (do_import && session == NULL)
        session = create_session(state, sid);
    if (session == NULL) {
        send_error(c, 404, "not_found_error", "session not found");
        return;
    }
    if (do_import) {
        if (metis_memory_snapshot_paths_resident(base_path,
                episodic_path, sizeof episodic_path, event_path, sizeof event_path,
                state->resident == NULL ? NULL : resident_path, sizeof resident_path) != 0 ||
            metis_episodic_load(
                &imported_episodic, episodic_path) != 0 ||
            metis_event_store_load(
                &imported_events, event_path) != 0 ||
            (state->resident != NULL &&
             (resident_state_load(&imported_resident, state->resident, resident_path) != 0 ||
              imported_resident.count != imported_events.count))) {
            resident_state_clear(&imported_resident);
            metis_episodic_free(&imported_episodic);
            metis_event_store_clear(&imported_events);
            send_error(c, 500, "server_error",
                       "failed to import typed memory state");
            return;
        }
        for (size_t index = 0;
             index < imported_events.count; ++index) {
            const metis_event_record_t *event =
                &imported_events.events[index];
            if (event->raw_record_index >= imported_episodic.count ||
                !event_spans_fit_source(
                    event,
                    imported_episodic.records[
                        event->raw_record_index])) {
                metis_episodic_free(&imported_episodic);
                metis_event_store_clear(&imported_events);
                resident_state_clear(&imported_resident);
                send_error(c, 500, "server_error",
                           "typed event has invalid source evidence");
                return;
            }
        }
        reset_session_context_only(session);
        metis_episodic_clear(&session->episodic);
        metis_event_store_clear(&session->events);
        session->episodic = imported_episodic;
        session->events = imported_events;
        resident_state_clear(&session->resident);
        session->resident = imported_resident;
        metis_episodic_init(&imported_episodic);
        metis_event_store_init(&imported_events);
    } else if (metis_memory_snapshot_save_resident(base_path,
                   &session->episodic, &session->events,
                   state->resident == NULL ? NULL : &session->resident,
                   state->resident) != 0) {
        send_error(c, 500, "server_error",
                   "failed to export typed memory state");
        return;
    }
    root = cJSON_CreateObject();
    if (root == NULL) {
        send_error(c, 500, "server_error",
                   "failed to build response");
        return;
    }
    json_add_string(root, "status", "ok");
    json_add_string(root, "session_id", sid);
    json_add_number(
        root, "episodic_records",
        (double)metis_episodic_count(&session->episodic));
    json_add_number(
        root, "typed_events", (double)session->events.count);
    if (state->resident != NULL) {
        json_add_number(root,"resident_events",(double)session->resident.count);
        json_add_number(root,"resident_state_restored",do_import);
        json_add_number(root,"raw_events_reencoded",0);
    }
    send_json(c, 200, root);
    cJSON_Delete(root);
}

static int parse_event_operation(const char *name,
                                 metis_event_operation_t *operation) {
    if (name == NULL || operation == NULL) return -1;
    if (strcmp(name, "assert") == 0) {
        *operation = METIS_EVENT_ASSERT;
    } else if (strcmp(name, "supersede") == 0) {
        *operation = METIS_EVENT_SUPERSEDE;
    } else if (strcmp(name, "retract") == 0) {
        *operation = METIS_EVENT_RETRACT;
    } else {
        return -1;
    }
    return 0;
}

static int parse_memory_kind(const char *name,
                             metis_memory_kind_t *kind) {
    if (name == NULL || kind == NULL) return -1;
    if (strcmp(name, "property") == 0) {
        *kind = METIS_MEMORY_PROPERTY;
    } else if (strcmp(name, "set") == 0) {
        *kind = METIS_MEMORY_SET;
    } else if (strcmp(name, "event") == 0) {
        *kind = METIS_MEMORY_EVENT;
    } else {
        return -1;
    }
    return 0;
}

static int parse_event_polarity(const char *name,
                                metis_event_polarity_t *polarity) {
    if (name == NULL || polarity == NULL) return -1;
    if (strcmp(name, "positive") == 0) {
        *polarity = METIS_EVENT_POSITIVE;
    } else if (strcmp(name, "negative") == 0) {
        *polarity = METIS_EVENT_NEGATIVE;
    } else {
        return -1;
    }
    return 0;
}

static int parse_event_modality(const char *name,
                                metis_event_modality_t *modality) {
    if (name == NULL || modality == NULL) return -1;
    if (strcmp(name, "actual") == 0) {
        *modality = METIS_EVENT_ACTUAL;
    } else if (strcmp(name, "planned") == 0) {
        *modality = METIS_EVENT_PLANNED;
    } else if (strcmp(name, "possible") == 0) {
        *modality = METIS_EVENT_POSSIBLE;
    } else {
        return -1;
    }
    return 0;
}

static const char *memory_kind_name(metis_memory_kind_t kind) {
    return kind == METIS_MEMORY_PROPERTY ? "property" :
           kind == METIS_MEMORY_SET ? "set" : "event";
}

static const char *event_polarity_name(
    metis_event_polarity_t polarity) {
    return polarity == METIS_EVENT_NEGATIVE ? "negative" : "positive";
}

static const char *event_modality_name(
    metis_event_modality_t modality) {
    return modality == METIS_EVENT_PLANNED ? "planned" :
           modality == METIS_EVENT_POSSIBLE ? "possible" : "actual";
}

static int json_get_span_offset(
    cJSON *request, const char *name, size_t *value) {
    cJSON *item = cJSON_GetObjectItem(request, name);
    double number;
    if (item == NULL || !cJSON_IsNumber(item) || value == NULL)
        return -1;
    number = item->valuedouble;
    if (!isfinite(number) || number < 0.0 ||
        number > (double)UINT32_MAX || floor(number) != number)
        return -1;
    *value = (size_t)number;
    return 0;
}

static int find_unique_evidence_span(
    const char *source, const char *text, size_t *start, size_t *end) {
    const char *match;
    if (source == NULL || text == NULL || text[0] == '\0' ||
        start == NULL || end == NULL)
        return -1;
    match = strstr(source, text);
    if (match == NULL || strstr(match + 1, text) != NULL) return -1;
    *start = (size_t)(match - source);
    *end = *start + strlen(text);
    return 0;
}

static int exact_evidence_span(
    const char *source, const char *text, size_t start, size_t end) {
    size_t text_length;
    size_t source_length;
    if (source == NULL || text == NULL) return 0;
    text_length = strlen(text);
    source_length = strlen(source);
    return end > start && end <= source_length &&
           end - start == text_length &&
           memcmp(source + start, text, text_length) == 0;
}

static int resolve_event_evidence(
    cJSON *request, metis_event_record_t *event, const char *source) {
    const char *subject_text = json_get_string(
        request, "subject_text", event->entity);
    const char *value_text = json_get_string(
        request, "value_text", event->value);
    cJSON *subject_start = cJSON_GetObjectItem(request, "subject_start");
    cJSON *subject_end = cJSON_GetObjectItem(request, "subject_end");
    cJSON *value_start = cJSON_GetObjectItem(request, "value_start");
    cJSON *value_end = cJSON_GetObjectItem(request, "value_end");
    int provided = subject_start != NULL || subject_end != NULL ||
                   value_start != NULL || value_end != NULL;
    if (subject_text == NULL || value_text == NULL) return -1;
    if (provided) {
        if (subject_start == NULL || subject_end == NULL ||
            value_start == NULL || value_end == NULL ||
            json_get_span_offset(
                request, "subject_start", &event->subject_start) != 0 ||
            json_get_span_offset(
                request, "subject_end", &event->subject_end) != 0 ||
            json_get_span_offset(
                request, "value_start", &event->value_start) != 0 ||
            json_get_span_offset(
                request, "value_end", &event->value_end) != 0)
            return -1;
    } else if (
        find_unique_evidence_span(
            source, subject_text,
            &event->subject_start, &event->subject_end) != 0 ||
        find_unique_evidence_span(
            source, value_text,
            &event->value_start, &event->value_end) != 0) {
        return -1;
    }
    return exact_evidence_span(
               source, subject_text,
               event->subject_start, event->subject_end) &&
           exact_evidence_span(
               source, value_text,
               event->value_start, event->value_end) ? 0 : -1;
}

static int typed_word_byte(unsigned char value) {
    return isalnum(value) || value == '_' ||
           value == '-' || value == '\'' || value >= 0x80u;
}

/* Latin word completion must not swallow adjacent unsegmented CJK text. */
static size_t typed_value_word_width(const char *text) {
    unsigned char first = (unsigned char)text[0];
    if (first < 0x80u)
        return first != 0 && typed_word_byte(first) ? 1u : 0u;
    if (first >= 0xC3u && first <= 0xCAu &&
        ((unsigned char)text[1] & 0xC0u) == 0x80u) {
        unsigned code = ((first & 0x1fu) << 6) | ((unsigned char)text[1] & 0x3fu);
        if (code <= 0x02afu && code != 0x00d7u && code != 0x00f7u) return 2u;
    }
    return 0;
}

static int decode_typed_source(
    server_state_t *state, const typed_text_encoding_t *encoding,
    const char *source, char **decoded_output,
    size_t **offsets_output, size_t *source_offset_output) {
    char *decoded = NULL;
    size_t decoded_length = 0, decoded_capacity = 0;
    size_t *offsets = NULL;
    char *source_match;
    if (state == NULL || encoding == NULL ||
        encoding->tokens == NULL || source == NULL ||
        decoded_output == NULL || offsets_output == NULL ||
        source_offset_output == NULL)
        return -1;
    offsets = (size_t *)malloc(
        (encoding->token_count + 1u) * sizeof(size_t));
    if (offsets == NULL) return -1;
    offsets[0] = 0;
    for (size_t token = 0;
         token < encoding->token_count; ++token) {
        char piece[256];
        int length = bitnet_decode_token(
            state->model, encoding->tokens[token],
            piece, sizeof piece);
        if (length < 0 ||
            append_bytes(
                &decoded, &decoded_length, &decoded_capacity,
                piece, (size_t)length) != 0) {
            free(decoded);
            free(offsets);
            return -1;
        }
        offsets[token + 1u] = decoded_length;
    }
    if (decoded == NULL) {
        free(offsets);
        return -1;
    }
    source_match = strstr(decoded, source);
    if (source_match == NULL ||
        strlen(source_match) != strlen(source)) {
        free(decoded);
        free(offsets);
        return -1;
    }
    *source_offset_output = (size_t)(source_match - decoded);
    *decoded_output = decoded;
    *offsets_output = offsets;
    return 0;
}

static int typed_token_span_to_source(
    const char *decoded, const size_t *offsets,
    size_t token_count, size_t source_offset,
    const char *source, size_t token_start, size_t token_end,
    int expand_entity, size_t *start_output, size_t *end_output) {
    size_t start, end, source_length;
    if (decoded == NULL || offsets == NULL || source == NULL ||
        start_output == NULL || end_output == NULL ||
        token_start > token_end || token_end >= token_count)
        return -1;
    start = offsets[token_start];
    end = offsets[token_end + 1u];
    while (start < end &&
           isspace((unsigned char)decoded[start]))
        ++start;
    while (end > start &&
           isspace((unsigned char)decoded[end - 1u]))
        --end;
    if (expand_entity == 2) {
        while (start > source_offset && ((unsigned char)decoded[start] & 0xC0u) == 0x80u) --start;
        while (((unsigned char)decoded[end] & 0xC0u) == 0x80u) ++end;
        while (start > source_offset) {
            size_t previous = start - 1u;
            while (previous > source_offset && ((unsigned char)decoded[previous] & 0xC0u) == 0x80u) --previous;
            if (typed_value_word_width(decoded + previous) != start - previous) break;
            start = previous;
        }
        size_t width;
        while ((width = typed_value_word_width(decoded + end)) != 0) end += width;
    } else if (expand_entity) {
        while (start < end &&
               !typed_word_byte((unsigned char)decoded[start]))
            ++start;
        while (start > source_offset &&
               typed_word_byte((unsigned char)decoded[start - 1u]))
            --start;
        while (typed_word_byte((unsigned char)decoded[end]))
            ++end;
        if (expand_entity == 1 && end >= start + 2u &&
            decoded[end - 2u] == '\'' &&
            decoded[end - 1u] == 's')
            end -= 2u;
        else if (expand_entity == 1 && end >= start + 4u &&
                 memcmp(decoded + end - 4u, "\xE2\x80\x99s", 4) == 0)
            end -= 4u;
    }
    source_length = strlen(source);
    if (start < source_offset || end <= start ||
        end - source_offset > source_length)
        return -1;
    *start_output = start - source_offset;
    *end_output = end - source_offset;
    return 0;
}

static char *duplicate_source_span(
    const char *source, size_t start, size_t end) {
    char *value;
    if (source == NULL || end <= start || end > strlen(source))
        return NULL;
    value = (char *)malloc(end - start + 1u);
    if (value == NULL) return NULL;
    memcpy(value, source + start, end - start);
    value[end - start] = '\0';
    return value;
}

static char *normalize_predicate_span(
    const char *source, size_t start, size_t end) {
    char *value;
    size_t written = 0;
    int separator = 0;
    if (source == NULL || end <= start || end > strlen(source))
        return NULL;
    value = (char *)malloc(end - start + 1u);
    if (value == NULL) return NULL;
    for (size_t index = start; index < end; ++index) {
        unsigned char byte = (unsigned char)source[index];
        if (isalnum(byte)) {
            if (separator && written > 0)
                value[written++] = '_';
            value[written++] = (char)tolower(byte);
            separator = 0;
        } else if (written > 0) {
            separator = 1;
        }
    }
    value[written] = '\0';
    if (written == 0 ||
        value[0] < 'a' || value[0] > 'z') {
        free(value);
        return NULL;
    }
    return value;
}

static int source_span_is_iso_date(
    const char *source, size_t start, size_t end) {
    if (source == NULL || end - start != 10u ||
        end > strlen(source))
        return 0;
    for (size_t index = 0; index < 10u; ++index) {
        unsigned char byte = (unsigned char)source[start + index];
        if (index == 4u || index == 7u) {
            if (byte != '-') return 0;
        } else if (!isdigit(byte)) {
            return 0;
        }
    }
    return 1;
}

static void expand_leading_value_article(
    const char *source, size_t *start) {
    size_t word_end, word_start, length;
    if (source == NULL || start == NULL || *start == 0)
        return;
    word_end = *start;
    while (word_end > 0 &&
           isspace((unsigned char)source[word_end - 1u]))
        --word_end;
    if (word_end == *start || word_end == 0)
        return;
    word_start = word_end;
    while (word_start > 0 &&
           isalpha((unsigned char)source[word_start - 1u]))
        --word_start;
    length = word_end - word_start;
    if ((length == 1u &&
         tolower((unsigned char)source[word_start]) == 'a') ||
        (length == 2u &&
         tolower((unsigned char)source[word_start]) == 'a' &&
         tolower((unsigned char)source[word_start + 1u]) == 'n') ||
        (length == 3u &&
         tolower((unsigned char)source[word_start]) == 't' &&
         tolower((unsigned char)source[word_start + 1u]) == 'h' &&
         tolower((unsigned char)source[word_start + 2u]) == 'e'))
        *start = word_start;
}

static void add_event_json(
    cJSON *root, const metis_event_record_t *event);

static int select_typed_active_event(
    server_state_t *state, cached_session_t *session,
    const char *current_text,
    const metis_event_record_t **selected_event,
    float *selected_score, float *exists_score,
    size_t *candidate_count, int query_mode);

typedef struct autonomous_memory_result {
    typed_text_encoding_t encoding;
    metis_typed_writer_result_t prediction;
    float target_score;
    float exists_score;
    size_t candidate_count;
    size_t event_index;
    int target_status;
    int deduplicated;
    int predecessor_missing;
    int operation_predicted;
    int predicted_operation;
} autonomous_memory_result_t;

/* Linguistic change is not a database UPDATE: a newly observed current fact
 * remains useful when no old version has ever been stored. Never invent a
 * target or deactivate an unrelated event. Explicit updates stay strict. */
static int compile_autonomous_operation(
    metis_event_record_t *event, const metis_event_record_t *target,
    int target_status, memory_record_action_t operation_hint) {
    if (event == NULL || target_status < 0) return -1;
    if (target_status == 0) {
        if (operation_hint == MEMORY_ACTION_UPDATE) return -2;
        event->operation = METIS_EVENT_ASSERT;
        event->target_event_id = "";
        return 0;
    }
    if (target == NULL) return -1;
    event->operation = METIS_EVENT_SUPERSEDE;
    event->target_event_id = target->event_id;
    event->entity = target->entity;
    event->predicate = target->predicate;
    return 0;
}

static int remember_autonomous_event(
    server_state_t *state, cached_session_t *session,
    const char *text, float priority, cJSON *request,
    memory_record_action_t operation_hint,
    autonomous_memory_result_t *result,
    char *error, size_t error_size) {
    metis_event_record_t event;
    char *decoded = NULL, *entity = NULL;
    char *predicate = NULL, *value = NULL, *valid_time = NULL;
    size_t *offsets = NULL;
    size_t source_offset = 0;
    size_t field_start[METIS_TYPED_WRITER_FIELD_COUNT];
    size_t field_end[METIS_TYPED_WRITER_FIELD_COUNT];
    char event_id[96], episode_id[96], source_id[96];
    const char *requested;
    int time_known = 1;
    int status = -1;
    size_t source_count = session == NULL ? 0 : session->episodic.count;
    resident_slot_t resident_pending = {0};
    memset(&event, 0, sizeof event);
    if (result != NULL) memset(result, 0, sizeof *result);
    if (error != NULL && error_size > 0) error[0] = '\0';
#define AUTO_FAIL(code, ...) do { \
    status = (code); \
    if (error != NULL && error_size > 0) \
        snprintf(error, error_size, __VA_ARGS__); \
    goto cleanup; \
} while (0)
    if (state == NULL || session == NULL ||
        text == NULL || text[0] == '\0' ||
        result == NULL || state->typed_writer == NULL)
        AUTO_FAIL(-1, "invalid autonomous writer arguments");
    if (operation_hint == MEMORY_ACTION_IGNORE &&
        (request == NULL || cJSON_GetObjectItem(request, "event_id") == NULL)) {
        for (size_t index = 0; index < session->events.count; ++index) {
            const metis_event_record_t *previous = &session->events.events[index];
            /* In resident mode every version is retained. Only an immediate
             * replay is idempotent: A -> B -> A must append a new observation. */
            if (state->resident != NULL && index + 1 != session->events.count)
                continue;
            if (previous->active && previous->raw_record_index < source_count &&
                strcmp(session->episodic.records[previous->raw_record_index], text) == 0) {
                result->event_index = index;
                result->deduplicated = 1;
                result->prediction.operation = previous->operation == METIS_EVENT_SUPERSEDE;
                return 0;
            }
        }
    }
    if (
        encode_typed_pair_text(
            state, text, &result->encoding) != 0 ||
        metis_typed_writer_predict(
            state->typed_writer,
            result->encoding.hidden, result->encoding.identity,
            result->encoding.token_count,
            &result->prediction) != 0 ||
        decode_typed_source(
            state, &result->encoding, text, &decoded,
            &offsets, &source_offset) != 0)
        AUTO_FAIL(
            -1, "autonomous typed writer inference failed");
    result->operation_predicted = 1;
    result->predicted_operation = result->prediction.operation;
    if (operation_hint == MEMORY_ACTION_STORE)
        result->prediction.operation = 0;
    else if (operation_hint == MEMORY_ACTION_UPDATE)
        result->prediction.operation = 1;
    for (int field = 0;
         field < METIS_TYPED_WRITER_FIELD_COUNT; ++field) {
        int span_status = typed_token_span_to_source(
                decoded, offsets, result->encoding.token_count,
                source_offset, text,
                result->prediction.span_start[field],
                result->prediction.span_end[field],
                field == METIS_TYPED_WRITER_ENTITY ? 1 :
                    (field == METIS_TYPED_WRITER_VALUE ? 2 : 0),
                &field_start[field], &field_end[field]);
        if (span_status != 0 && field == METIS_TYPED_WRITER_ENTITY) {
            /* A malformed span may still have a valid learned entity anchor. */
            span_status = typed_token_span_to_source(
                decoded, offsets, result->encoding.token_count, source_offset, text,
                result->prediction.anchor[field], result->prediction.anchor[field], 1,
                &field_start[field], &field_end[field]);
        }
        if (field == METIS_TYPED_WRITER_TIME &&
            (span_status != 0 ||
             !source_span_is_iso_date(
                 text, field_start[field], field_end[field]))) {
            time_known = 0;
            field_start[field] = 0;
            field_end[field] = 0;
            continue;
        }
        if (span_status != 0)
            AUTO_FAIL(
                -3,
                "autonomous writer produced an invalid field %d source span",
                field);
    }
    entity = duplicate_source_span(
        text, field_start[METIS_TYPED_WRITER_ENTITY],
        field_end[METIS_TYPED_WRITER_ENTITY]);
    predicate = normalize_predicate_span(
        text, field_start[METIS_TYPED_WRITER_PREDICATE],
        field_end[METIS_TYPED_WRITER_PREDICATE]);
    expand_leading_value_article(
        text, &field_start[METIS_TYPED_WRITER_VALUE]);
    value = duplicate_source_span(
        text, field_start[METIS_TYPED_WRITER_VALUE],
        field_end[METIS_TYPED_WRITER_VALUE]);
    valid_time = time_known ?
        duplicate_source_span(
            text, field_start[METIS_TYPED_WRITER_TIME],
            field_end[METIS_TYPED_WRITER_TIME]) :
        (char *)calloc(1, 1);
    if (entity == NULL || predicate == NULL ||
        value == NULL || valid_time == NULL)
        AUTO_FAIL(
            -1, "failed to compile autonomous writer spans");
    requested = request == NULL ? NULL :
        json_get_string(request, "event_id", NULL);
    if (requested == NULL || requested[0] == '\0') {
        snprintf(
            event_id, sizeof event_id,
            "auto-event-%zu", session->events.count);
        requested = event_id;
    }
    event.event_id = (char *)requested;
    requested = request == NULL ? NULL :
        json_get_string(request, "episode_id", NULL);
    if (requested == NULL || requested[0] == '\0') {
        snprintf(
            episode_id, sizeof episode_id,
            "auto-episode-%zu", session->episodic.count);
        requested = episode_id;
    }
    event.episode_id = (char *)requested;
    requested = request == NULL ? NULL :
        json_get_string(request, "source_id", NULL);
    if (requested == NULL || requested[0] == '\0') {
        snprintf(
            source_id, sizeof source_id,
            "auto-source-%zu", session->episodic.count);
        requested = source_id;
    }
    event.source_id = (char *)requested;
    event.entity = entity;
    event.predicate = predicate;
    event.value = value;
    event.valid_time = valid_time;
    event.target_event_id = "";
    event.operation = result->prediction.operation == 0 ?
        METIS_EVENT_ASSERT : METIS_EVENT_SUPERSEDE;
    event.memory_kind = METIS_MEMORY_PROPERTY;
    event.polarity = METIS_EVENT_POSITIVE;
    event.modality = METIS_EVENT_ACTUAL;
    event.active = 1;
    event.subject_start =
        field_start[METIS_TYPED_WRITER_ENTITY];
    event.subject_end =
        field_end[METIS_TYPED_WRITER_ENTITY];
    event.value_start =
        field_start[METIS_TYPED_WRITER_VALUE];
    event.value_end =
        field_end[METIS_TYPED_WRITER_VALUE];
    if (state->resident != NULL) {
        size_t token_count=0;
        if(session->resident.count!=session->events.count ||
           session->resident.count>=RESIDENT_MAX_EVENTS)
            AUTO_FAIL(-1,"resident memory capacity or state mismatch");
        float *features=resident_encode(state->model,text,&token_count);
        if(features==NULL)AUTO_FAIL(-1,"resident encoding failed or exceeds 128 tokens");
        int write_status=resident_write(state->resident,features,token_count,&resident_pending);
        free(features);
        if(write_status!=0)AUTO_FAIL(-1,"resident address write failed");
        /* All observed versions remain resident. The learned version gate,
         * not the legacy predecessor linker, determines current vs history. */
        event.operation=METIS_EVENT_ASSERT;
    }
    if (event.operation == METIS_EVENT_SUPERSEDE) {
        const metis_event_record_t *selected_event = NULL;
        result->target_status = select_typed_active_event(
            state, session, text, &selected_event,
            &result->target_score, &result->exists_score,
            &result->candidate_count, 0);
        int operation_status = compile_autonomous_operation(
            &event, selected_event, result->target_status, operation_hint);
        if (operation_status == -1)
            AUTO_FAIL(
                -1, "neural predecessor selection failed");
        if (operation_status == -2)
            AUTO_FAIL(
                -2,
                "explicit update rejected all active candidates");
        result->predecessor_missing = result->target_status == 0;
    }
    if (metis_episodic_add_indexed(
            &session->episodic, text, NULL, priority,
            &event.raw_record_index) != 0)
        AUTO_FAIL(
            -1, "failed to store autonomous memory record");
    if (metis_event_store_apply(&session->events, &event) != 0)
        AUTO_FAIL(
            -1, "failed to apply autonomous typed event");
    result->event_index = session->events.count - 1u;
    if(state->resident != NULL) {
        session->resident.slots[session->resident.count++]=resident_pending;
        memset(&resident_pending,0,sizeof resident_pending);
    }
    status = 0;
cleanup:
    free(resident_pending.data);
    free(decoded);
    free(offsets);
    free(entity);
    free(predicate);
    free(value);
    free(valid_time);
    if (status != 0) {
        if (session != NULL) metis_episodic_truncate(&session->episodic, source_count);
        if (result != NULL) typed_text_encoding_clear(&result->encoding);
    }
#undef AUTO_FAIL
    return status;
}

static void add_autonomous_memory_json(
    cJSON *root, const autonomous_memory_result_t *result,
    const metis_event_store_t *events) {
    const metis_event_record_t *event;
    if (root == NULL || result == NULL || events == NULL ||
        result->event_index >= events->count)
        return;
    event = &events->events[result->event_index];
    json_add_string(
        root, "writer_fields_source", "neural_autonomous_writer");
    json_add_number(root, "autonomous_writer", 1.0);
    json_add_number(root, "deduplicated", result->deduplicated);
    if (result->operation_predicted) {
        json_add_string(root, "writer_predicted_operation",
            result->predicted_operation ? "supersede" : "assert");
        json_add_number(root, "writer_assert_logit", result->prediction.operation_logits[0]);
        json_add_number(root, "writer_supersede_logit", result->prediction.operation_logits[1]);
    }
    json_add_string(
        root, "target_resolution",
        result->target_status > 0 ? "neural_activation" :
        result->predecessor_missing ? "no_predecessor_asserted" :
                                    "not_required");
    if (result->target_status > 0 || result->predecessor_missing) {
        json_add_number(
            root, "target_candidate_count",
            (double)result->candidate_count);
        json_add_number(
            root, "target_exists_score", result->exists_score);
        json_add_number(
            root, "target_activation_score", result->target_score);
    }
    for (int field = 0;
         field < METIS_TYPED_WRITER_FIELD_COUNT; ++field) {
        static const char *names[] = {
            "entity_anchor_token", "predicate_anchor_token",
            "value_anchor_token", "time_anchor_token"
        };
        json_add_number(
            root, names[field],
            (double)result->prediction.anchor[field]);
    }
    add_event_json(root, event);
}

static void handle_memory_remember(
    struct mg_connection *c, server_state_t *state, cJSON *request) {
    const char *sid = json_get_string(request, "session_id", NULL);
    const char *text = json_get_string(
        request, "memory_record", NULL);
    float priority;
    cached_session_t *session;
    autonomous_memory_result_t result;
    char error[256];
    int status;
    cJSON *root = NULL;
    memset(&result, 0, sizeof result);
    if (!state->cfg.episodic_memory ||
        state->typed_writer == NULL) {
        send_error(c, 403, "typed_writer_disabled",
                   "autonomous typed writer is not enabled");
        return;
    }
    if (sid == NULL || sid[0] == '\0' ||
        text == NULL || text[0] == '\0') {
        send_error(c, 400, "invalid_request_error",
                   "session_id and memory_record are required");
        return;
    }
    session = find_session(state, sid);
    if (session == NULL) session = create_session(state, sid);
    if (session == NULL) {
        send_error(c, 500, "server_error",
                   "failed to create memory session");
        return;
    }
    priority = json_get_float(
        request, "memory_priority", 0.0f, 0.0f, 1.0f);
    if (state->resident != NULL &&
        typed_auto_fallback_action(text) == MEMORY_ACTION_DELETE) {
        send_error(c,400,"unsupported_memory_operation",
            "experimental resident memory does not support deletion");
        return;
    }
    status = remember_autonomous_event(
        state, session, text, priority, request,
        MEMORY_ACTION_IGNORE,
        &result, error, sizeof error);
    if (status != 0) {
        send_error(
            c,
            status == -2 ? 409 :
            status == -3 ? 422 : 500,
            status == -2 ? "memory_target_not_found" :
            status == -3 ? "writer_span_error" :
                           "server_error",
            error);
        return;
    }
    root = cJSON_CreateObject();
    if (root == NULL) {
        send_error(c, 500, "server_error",
                   "failed to build autonomous writer response");
        typed_text_encoding_clear(&result.encoding);
        return;
    }
    json_add_string(root, "status", "ok");
    json_add_string(root, "session_id", sid);
    json_add_string(
        root, "writer_fields_source", "neural_autonomous_writer");
    json_add_number(root, "autonomous_writer", 1.0);
    json_add_number(root, "kv_cache_touched", 0.0);
    json_add_number(
        root, "typed_events", (double)session->events.count);
    json_add_number(
        root, "episodic_records",
        (double)session->episodic.count);
    add_autonomous_memory_json(root, &result, &session->events);
    send_json(c, 200, root);
    cJSON_Delete(root);
    typed_text_encoding_clear(&result.encoding);
}

static void add_event_json(cJSON *root,
                           const metis_event_record_t *event) {
    json_add_string(root, "event_id", event->event_id);
    json_add_string(root, "episode_id", event->episode_id);
    json_add_string(root, "source_id", event->source_id);
    json_add_string(root, "entity", event->entity);
    json_add_string(root, "predicate", event->predicate);
    json_add_string(root, "value", event->value);
    json_add_string(root, "valid_time", event->valid_time);
    json_add_string(root, "target_event_id", event->target_event_id);
    json_add_string(
        root, "operation",
        event->operation == METIS_EVENT_ASSERT ? "assert" :
        event->operation == METIS_EVENT_SUPERSEDE ? "supersede" :
                                                    "retract");
    json_add_string(root, "memory_kind",
                    memory_kind_name(event->memory_kind));
    json_add_string(root, "polarity",
                    event_polarity_name(event->polarity));
    json_add_string(root, "modality",
                    event_modality_name(event->modality));
    json_add_number(root, "active", event->active ? 1.0 : 0.0);
    json_add_number(
        root, "raw_record_index", (double)event->raw_record_index);
    if (event_spans_are_unknown(event)) {
        json_add_null(root, "subject_start");
        json_add_null(root, "subject_end");
        json_add_null(root, "value_start");
        json_add_null(root, "value_end");
    } else {
        json_add_number(
            root, "subject_start", (double)event->subject_start);
        json_add_number(root, "subject_end", (double)event->subject_end);
        json_add_number(root, "value_start", (double)event->value_start);
        json_add_number(root, "value_end", (double)event->value_end);
    }
}

static int select_typed_active_event(
    server_state_t *state, cached_session_t *session,
    const char *current_text,
    const metis_event_record_t **selected_event,
    float *selected_score, float *exists_score,
    size_t *candidate_count, int query_mode) {
    typed_text_encoding_t current;
    typed_text_encoding_t candidate;
    const metis_event_record_t **events = NULL;
    float *scores = NULL;
    float *entity_feature = NULL;
    float *predicate_feature = NULL;
    float *query_features = NULL;
    float *entity_logits = NULL;
    float *predicate_logits = NULL;
    size_t count = 0;
    size_t selected = SIZE_MAX;
    size_t feature_dim;
    int status = -1;
    memset(&current, 0, sizeof current);
    memset(&candidate, 0, sizeof candidate);
    if (selected_event != NULL) *selected_event = NULL;
    if (selected_score != NULL) *selected_score = 0.0f;
    if (exists_score != NULL) *exists_score = 0.0f;
    if (candidate_count != NULL) *candidate_count = 0;
    if (state == NULL || session == NULL ||
        state->typed_pair == NULL || state->typed_link == NULL ||
        current_text == NULL || current_text[0] == '\0' ||
        selected_event == NULL || selected_score == NULL ||
        exists_score == NULL ||
        candidate_count == NULL)
        return -1;
    for (size_t index = 0;
         index < session->events.count; ++index)
        count += session->events.events[index].active ? 1u : 0u;
    *candidate_count = count;
    if (count == 0) return 0;
    feature_dim =
        (size_t)state->typed_link->pair_feature_dim;
    events = (const metis_event_record_t **)calloc(
        count, sizeof(*events));
    scores = (float *)malloc(count * sizeof(float));
    entity_feature = (float *)malloc(
        feature_dim * sizeof(float));
    predicate_feature = (float *)malloc(
        feature_dim * sizeof(float));
    if (query_mode && state->typed_query != NULL) {
        query_features = (float *)malloc(
            count * feature_dim * 2u * sizeof(float));
        entity_logits = (float *)malloc(
            count * sizeof(float));
        predicate_logits = (float *)malloc(
            count * sizeof(float));
    }
    if (events == NULL || scores == NULL ||
        entity_feature == NULL || predicate_feature == NULL ||
        (query_mode && state->typed_query != NULL &&
         (query_features == NULL || entity_logits == NULL ||
          predicate_logits == NULL)) ||
        encode_typed_pair_text(
            state, current_text, &current) != 0)
        goto cleanup;
    count = 0;
    for (size_t index = 0;
         index < session->events.count; ++index) {
        const metis_event_record_t *event =
            &session->events.events[index];
        float entity_logit;
        float predicate_logit;
        int pair_status;
        if (!event->active) continue;
        if (event->raw_record_index >=
            session->episodic.count)
            goto cleanup;
        pair_status = encode_typed_pair_text(
                state,
                session->episodic.records[
                    event->raw_record_index],
                &candidate);
        if (pair_status == 0)
            pair_status = metis_typed_pair_encoder_score(
                state->typed_pair,
                current.hidden, current.token_count,
                current.identity,
                candidate.hidden, candidate.token_count,
                candidate.identity,
                entity_feature, predicate_feature,
                &entity_logit, &predicate_logit);
        if (pair_status != 0)
            goto cleanup;
        if (query_mode && state->typed_query != NULL) {
            memcpy(
                query_features +
                    count * feature_dim * 2u,
                entity_feature,
                feature_dim * sizeof(float));
            memcpy(
                query_features +
                    count * feature_dim * 2u +
                    feature_dim,
                predicate_feature,
                feature_dim * sizeof(float));
            entity_logits[count] = entity_logit;
            predicate_logits[count] = predicate_logit;
        } else if (metis_typed_link_score_pairs(
                       state->typed_link,
                       entity_feature, predicate_feature,
                       &entity_logit, &predicate_logit, 1,
                       &scores[count]) != 0) {
            goto cleanup;
        }
        typed_text_encoding_clear(&candidate);
        events[count++] = event;
    }
    if (count != *candidate_count)
        goto cleanup;
    if (query_mode && state->typed_query != NULL) {
        if (metis_typed_query_select(
                state->typed_query,
                query_features,
                entity_logits,
                predicate_logits,
                count, &selected,
                selected_score,
                exists_score) != 0)
            goto cleanup;
    } else if (metis_typed_link_select_predecessor(
                   state->typed_link, scores, count,
                   &selected, exists_score) != 0) {
        goto cleanup;
    }
    if (selected == SIZE_MAX) {
        status = 0;
        goto cleanup;
    }
    if (selected >= count) goto cleanup;
    *selected_event = events[selected];
    if (!(query_mode && state->typed_query != NULL))
        *selected_score = scores[selected];
    status = 1;
cleanup:
    typed_text_encoding_clear(&candidate);
    typed_text_encoding_clear(&current);
    free(events);
    free(scores);
    free(entity_feature);
    free(predicate_feature);
    free(query_features);
    free(entity_logits);
    free(predicate_logits);
    return status;
}

static void handle_memory_event(
    struct mg_connection *c, server_state_t *state, cJSON *request) {
    const char *sid = json_get_string(request, "session_id", NULL);
    const char *memory_record = json_get_string(
        request, "memory_record", NULL);
    cached_session_t *session;
    metis_event_record_t event;
    cJSON *root;
    int raw_index;
    int neural_target_status = 0;
    float neural_target_score = 0.0f;
    float neural_exists_score = 0.0f;
    size_t neural_candidate_count = 0;
    size_t source_index = 0;
    size_t source_count;
    const char *source;
    memset(&event, 0, sizeof event);
    if (!state->cfg.episodic_memory) {
        send_error(c, 403, "episodic_memory_disabled",
                   "episodic memory is not enabled");
        return;
    }
    if (sid == NULL || sid[0] == '\0') {
        send_error(c, 400, "invalid_request_error",
                   "session_id is required");
        return;
    }
    session = find_session(state, sid);
    if (session == NULL && memory_record != NULL)
        session = create_session(state, sid);
    if (session == NULL || (memory_record == NULL && session->episodic.count == 0)) {
        send_error(c, 404, "not_found_error",
                   "session or source episode not found");
        return;
    }
    source_count = session->episodic.count;
    source_index = source_count;
    if (memory_record != NULL) {
        for (size_t i = 0; i < source_count; ++i)
            if (strcmp(session->episodic.records[i], memory_record) == 0) {
                source_index = i;
                break;
            }
    }
    event.event_id = (char *)json_get_string(
        request, "event_id", NULL);
    event.episode_id = (char *)json_get_string(
        request, "episode_id", NULL);
    event.source_id = (char *)json_get_string(
        request, "source_id", NULL);
    event.entity = (char *)json_get_string(
        request, "entity", NULL);
    event.predicate = (char *)json_get_string(
        request, "predicate", NULL);
    event.value = (char *)json_get_string(
        request, "value", NULL);
    event.valid_time = (char *)json_get_string(
        request, "valid_time", "");
    event.target_event_id = (char *)json_get_string(
        request, "target_event_id", "");
    if (parse_event_operation(
            json_get_string(request, "operation", NULL),
            &event.operation) != 0) {
        send_error(c, 400, "invalid_request_error",
                   "operation must be assert, supersede, or retract");
        return;
    }
    if (parse_memory_kind(
            json_get_string(request, "memory_kind", NULL),
            &event.memory_kind) != 0) {
        send_error(c, 400, "invalid_request_error",
                   "memory_kind must be property, set, or event");
        return;
    }
    if (parse_event_polarity(
            json_get_string(request, "polarity", "positive"),
            &event.polarity) != 0) {
        send_error(c, 400, "invalid_request_error",
                   "polarity must be positive or negative");
        return;
    }
    if (parse_event_modality(
            json_get_string(request, "modality", "actual"),
            &event.modality) != 0) {
        send_error(c, 400, "invalid_request_error",
                   "modality must be actual, planned, or possible");
        return;
    }
    raw_index = json_get_int(
        request, "raw_record_index",
        memory_record != NULL ? (int)source_index : (int)session->episodic.count - 1, 0,
        memory_record != NULL ? (int)source_index : (int)session->episodic.count - 1);
    if (memory_record != NULL && (size_t)raw_index != source_index) {
        send_error(c, 400, "invalid_request_error", "source record index does not match memory_record");
        return;
    }
    event.raw_record_index = (size_t)raw_index;
    source = memory_record != NULL ? memory_record : session->episodic.records[event.raw_record_index];
    event.active = 1;
    if (resolve_event_evidence(
            request, &event,
            source) != 0) {
        send_error(
            c, 400, "invalid_request_error",
            "subject/value evidence must be exact, unambiguous UTF-8 byte spans");
        return;
    }
    if (event.operation != METIS_EVENT_ASSERT &&
        event.target_event_id[0] == '\0' &&
        state->typed_pair != NULL &&
        state->typed_link != NULL) {
        const char *selected_target = NULL;
        const metis_event_record_t *selected_event = NULL;
        neural_target_status = select_typed_active_event(
            state, session,
            source,
            &selected_event, &neural_target_score,
            &neural_exists_score,
            &neural_candidate_count, 0);
        if (neural_target_status < 0) {
            send_error(c, 500, "server_error",
                       "neural predecessor selection failed");
            return;
        }
        if (neural_target_status == 0) {
            send_error(c, 409, "memory_target_not_found",
                       "neural predecessor activation rejected all candidates");
            return;
        }
        selected_target = selected_event->event_id;
        event.target_event_id = (char *)selected_target;
    }
    if (memory_record != NULL && metis_episodic_add_indexed(
            &session->episodic, memory_record, NULL, 0.0f, &event.raw_record_index) != 0) {
        send_error(c, 500, "server_error", "failed to store event source evidence");
        return;
    }
    if (metis_event_store_apply(&session->events, &event) != 0) {
        metis_episodic_truncate(&session->episodic, source_count);
        send_error(c, 400, "invalid_request_error",
                   "invalid, duplicate, or unresolved typed event");
        return;
    }
    root = cJSON_CreateObject();
    if (root == NULL) {
        send_error(c, 500, "server_error",
                   "failed to build event response");
        return;
    }
    json_add_string(root, "status", "ok");
    json_add_string(root, "session_id", sid);
    json_add_string(
        root, "writer_fields_source", "structured_request");
    json_add_number(root, "autonomous_writer", 0.0);
    json_add_number(root, "typed_events",
                    (double)session->events.count);
    json_add_string(
        root, "target_resolution",
        neural_target_status > 0 ? "neural_activation" :
                                  "provided_or_not_required");
    if (neural_target_status > 0) {
        json_add_number(
            root, "target_candidate_count",
            (double)neural_candidate_count);
        json_add_number(
            root, "target_exists_score",
            neural_exists_score);
        json_add_number(
            root, "target_activation_score",
            neural_target_score);
    }
    add_event_json(
        root, &session->events.events[session->events.count - 1]);
    send_json(c, 200, root);
    cJSON_Delete(root);
}

static void handle_memory_event_current(
    struct mg_connection *c, server_state_t *state, cJSON *request) {
    const char *sid = json_get_string(request, "session_id", NULL);
    const char *entity = json_get_string(request, "entity", NULL);
    const char *predicate = json_get_string(
        request, "predicate", NULL);
    cached_session_t *session = find_session(state, sid);
    const metis_event_record_t *event;
    cJSON *root;
    if (session == NULL || entity == NULL || predicate == NULL) {
        send_error(c, 400, "invalid_request_error",
                   "session_id, entity, and predicate are required");
        return;
    }
    event = metis_event_store_current(
        &session->events, entity, predicate);
    root = cJSON_CreateObject();
    if (root == NULL) {
        send_error(c, 500, "server_error",
                   "failed to build event response");
        return;
    }
    json_add_string(root, "status", event == NULL ? "not_found" : "ok");
    json_add_string(root, "session_id", sid);
    json_add_number(
        root, "active_count",
        (double)metis_event_store_active_count(
            &session->events, entity, predicate));
    if (event != NULL) add_event_json(root, event);
    send_json(c, 200, root);
    cJSON_Delete(root);
}

static void handle_memory_event_active(
    struct mg_connection *c, server_state_t *state, cJSON *request) {
    const char *sid = json_get_string(request, "session_id", NULL);
    const char *entity = json_get_string(request, "entity", NULL);
    const char *predicate = json_get_string(
        request, "predicate", NULL);
    cached_session_t *session = find_session(state, sid);
    cJSON *root;
    cJSON *events;
    size_t count;
    (void)state;
    if (session == NULL || entity == NULL || predicate == NULL) {
        send_error(c, 400, "invalid_request_error",
                   "session_id, entity, and predicate are required");
        return;
    }
    root = cJSON_CreateObject();
    events = cJSON_CreateArray();
    if (root == NULL || events == NULL) {
        cJSON_Delete(events);
        cJSON_Delete(root);
        send_error(c, 500, "server_error",
                   "failed to build event response");
        return;
    }
    count = metis_event_store_active_count(
        &session->events, entity, predicate);
    json_add_string(root, "status", count == 0 ? "not_found" : "ok");
    json_add_string(root, "session_id", sid);
    json_add_number(root, "active_count", (double)count);
    for (size_t index = 0; index < count; ++index) {
        const metis_event_record_t *event =
            metis_event_store_active_at(
                &session->events, entity, predicate, index);
        cJSON *item = cJSON_CreateObject();
        if (event == NULL || item == NULL) {
            cJSON_Delete(item);
            cJSON_Delete(events);
            cJSON_Delete(root);
            send_error(c, 500, "server_error",
                       "failed to build active events");
            return;
        }
        add_event_json(item, event);
        cJSON_AddItemToArray(events, item);
    }
    cJSON_AddItemToObject(root, "events", events);
    send_json(c, 200, root);
    cJSON_Delete(root);
}

static void handle_memory_extract(
    struct mg_connection *c, server_state_t *state,
    cJSON *request) {
    const char *sid = json_get_string(request, "session_id", NULL);
    const char *query = json_get_string(request, "query", NULL);
    cached_session_t *session;
    char *answer = NULL;
    float confidence = 0.0f;
    float null_score = 0.0f;
    float span_score = 0.0f;
    size_t record_index = 0;
    int extracted;
    cJSON *root;

    if (state->typed_pair == NULL || state->typed_link == NULL ||
        state->typed_query == NULL) {
        send_error(c, 403, "typed_memory_disabled",
                   "typed neural recall is not configured");
        return;
    }
    if (sid == NULL || query == NULL || query[0] == '\0') {
        send_error(c, 400, "invalid_request_error",
                   "session_id and query are required");
        return;
    }
    session = find_session(state, sid);
    if (session == NULL) {
        send_error(c, 404, "not_found_error", "session not found");
        return;
    }
    extracted = extract_typed_value_answer(
        state, session, query, &answer,
        &confidence, &null_score,
        &span_score, &record_index);
    if (extracted < 0) {
        send_error(c, 500, "server_error",
                   "typed memory extraction failed");
        return;
    }
    root = cJSON_CreateObject();
    if (root == NULL) {
        free(answer);
        send_error(c, 500, "server_error",
                   "failed to build response");
        return;
    }
    json_add_string(
        root, "status", extracted ? "extracted" : "fallback");
    json_add_string(root, "session_id", sid);
    json_add_string(root, "answer", answer == NULL ? "" : answer);
    json_add_number(root, "confidence", confidence);
    json_add_number(root, "activation_exists_score", null_score);
    json_add_number(root, "span_score", span_score);
    json_add_number(root, "kv_cache_touched", 0);
    json_add_string(
        root, "recall_mode",
        state->resident == NULL ?
        "neural_typed_query_activation_then_compiled_pointer" :
        "neural_resident_activation_then_compiled_pointer");
    if (state->resident != NULL) resident_recall_metadata(root,session);
    if (extracted)
        json_add_number(root, "record_index", (double)record_index);
    send_json(c, 200, root);
    cJSON_Delete(root);
    free(answer);
}

static void handle_chat_completions(struct mg_connection *c, server_state_t *state,
                                    cJSON *request) {
    char error[256];
    char id[64];
    char *prompt = NULL;
    char *pointer_answer = NULL;
    const char *session_id = NULL;
    const char *user_content = NULL;
    const char *memory_record = NULL;
    float pointer_confidence = 0.0f;
    float pointer_null_score = 0.0f;
    float pointer_span_score = 0.0f;
    float memory_priority = 0.0f;
    size_t pointer_record_index = 0;
    int pointer_used = 0;
    int memory_auto_requested = 0;
    int auto_typed_handled = 0;
    int auto_memory_stored = 0;
    int auto_memory_status = 0;
    char auto_memory_error[256];
    autonomous_memory_result_t auto_memory_result;
    memory_record_action_t memory_action = MEMORY_ACTION_IGNORE;
    int max_tokens = 0;
    int reset_existing_session = 0;
    int reset_context_only = 0;
    int generation_reset = 0;
    cached_session_t *episodic_session = NULL;
    generation_result_t result;
    cJSON *root = NULL;
    cJSON *choices = NULL;
    cJSON *choice = NULL;
    cJSON *message = NULL;
    memset(&auto_memory_result, 0, sizeof auto_memory_result);
    auto_memory_error[0] = '\0';

    session_id = json_get_string(request, "session_id", NULL);
    reset_existing_session = json_get_bool(request, "reset_session", 0);
    reset_context_only = json_get_bool(request, "reset_context", 0);
    generation_reset = reset_existing_session;
    max_tokens = json_get_int(request, "max_tokens", state->cfg.default_max_tokens,
                              1, state->cfg.max_request_tokens);
    user_content = last_user_content(request);
    memory_record = json_get_string(
        request, "memory_record", user_content);
    memory_action = request_memory_action(state, request, user_content);
    if (state->resident != NULL && memory_action == MEMORY_ACTION_DELETE) {
        send_error(c,400,"unsupported_memory_operation",
            "experimental resident memory does not support deletion");
        return;
    }
    memory_auto_requested = json_get_bool(
        request, "memory_auto",
        state->typed_writer != NULL ? 1 : 0);
    {
        cJSON *automatic = cJSON_GetObjectItem(
            request, "memory_auto");
        if (automatic != NULL &&
            cJSON_IsBool(automatic) &&
            !cJSON_IsTrue(automatic) &&
            !request_has_memory_action(request))
            memory_action = MEMORY_ACTION_IGNORE;
    }
    if (memory_auto_requested &&
        !request_has_memory_action(request) &&
        state->memory_controller == NULL) {
        memory_record_action_t typed_action =
            typed_auto_fallback_action(user_content);
        if (typed_action != MEMORY_ACTION_IGNORE ||
            typed_auto_input_is_query(user_content))
            memory_action = typed_action;
    }
    if (state->resident != NULL && memory_action == MEMORY_ACTION_DELETE) {
        send_error(c,400,"unsupported_memory_operation",
            "experimental resident memory does not support deletion");
        return;
    }
    memory_priority = json_get_float(
        request, "memory_priority", 0.0f, 0.0f, 1.0f);
    if (session_id != NULL && session_id[0] != '\0') {
        cached_session_t *existing_session =
            find_session(state, session_id);
        if (state->cfg.episodic_memory)
            episodic_session = existing_session;
        if (existing_session != NULL &&
            (state->cfg.episodic_memory || reset_context_only)) {
            if (reset_existing_session) reset_session(existing_session);
            else reset_session_context_only(existing_session);
            generation_reset = 0;
        }
    }
    {
        int enable_thinking = request_enable_thinking(request);
        prompt = build_chat_prompt(state, request, enable_thinking);
    }
    if (prompt == NULL) {
        send_error(c, 400, "invalid_request_error", "messages must be a non-empty array");
        return;
    }
    if (state->resident != NULL && memory_auto_requested &&
        memory_action == MEMORY_ACTION_IGNORE && episodic_session == NULL &&
        session_id != NULL && session_id[0] != '\0') {
        episodic_session = create_session(state, session_id);
        if (episodic_session == NULL) {
            free(prompt);
            send_error(c,500,"server_error","failed to create resident memory session");
            return;
        }
    }
    if (memory_auto_requested &&
        (memory_action == MEMORY_ACTION_STORE ||
         memory_action == MEMORY_ACTION_UPDATE)) {
        cached_session_t *session = NULL;
        auto_typed_handled = 1;
        if (state->typed_writer == NULL) {
            auto_memory_status = -1;
            snprintf(
                auto_memory_error, sizeof auto_memory_error,
                "typed writer is not loaded");
        } else if (session_id == NULL || session_id[0] == '\0') {
            auto_memory_status = -1;
            snprintf(
                auto_memory_error, sizeof auto_memory_error,
                "session_id is required for automatic memory");
        } else {
            session = find_session(state, session_id);
            if (session == NULL)
                session = create_session(state, session_id);
            if (session == NULL) {
                auto_memory_status = -1;
                snprintf(
                    auto_memory_error, sizeof auto_memory_error,
                    "failed to create automatic memory session");
            } else {
                auto_memory_status = remember_autonomous_event(
                    state, session, memory_record,
                    memory_priority, request,
                    request_has_memory_action(request) ? memory_action : MEMORY_ACTION_IGNORE,
                    &auto_memory_result,
                    auto_memory_error,
                    sizeof auto_memory_error);
                auto_memory_stored =
                    auto_memory_status == 0;
                if (auto_memory_stored && !request_has_memory_action(request))
                    memory_action = session->events.events[auto_memory_result.event_index].operation == METIS_EVENT_ASSERT ?
                        MEMORY_ACTION_STORE : MEMORY_ACTION_UPDATE;
                typed_text_encoding_clear(
                    &auto_memory_result.encoding);
                episodic_session = session;
            }
        }
    }
    if (state->cfg.episodic_memory &&
        state->typed_pair != NULL &&
        state->typed_link != NULL &&
        state->typed_query != NULL &&
        episodic_session != NULL &&
        (episodic_session->events.count > 0 || state->resident != NULL) &&
        memory_action == MEMORY_ACTION_IGNORE &&
        (json_get_bool(request, "memory_copy", 0) ||
         memory_auto_requested)) {
        int extracted = extract_typed_value_answer(
            state, episodic_session, user_content,
            &pointer_answer, &pointer_confidence,
            &pointer_null_score, &pointer_span_score,
            &pointer_record_index);
        if (state->resident != NULL && extracted < 0) {
            free(pointer_answer);free(prompt);
            send_error(c,500,"resident_recall_failed","resident neural recall failed");
            return;
        }
        pointer_used = extracted > 0 ||
            (state->resident != NULL && extracted == 0);
    }
    if (json_get_bool(request, "stream", 0) && !pointer_used) {
        handle_streaming_completion(
            c, state, prompt, max_tokens, session_id,
            generation_reset, 1);
        free(prompt);
        return;
    }

    memset(&result, 0, sizeof(result));
    if (pointer_used) {
        int answer_tokens[192];
        result.text = pointer_answer;
        pointer_answer = NULL;
        result.session_id = session_id == NULL ? NULL :
            dup_n(session_id, strlen(session_id));
        result.completion_tokens = bitnet_tokenize_ex(
            state->model, result.text, answer_tokens,
            (int)(sizeof answer_tokens / sizeof answer_tokens[0]), 0);
        if (result.completion_tokens < 0) result.completion_tokens = 0;
    } else {
        if (generate_text(
                state, prompt, max_tokens, session_id, generation_reset,
                &result, error, sizeof(error)) != 0) {
            free(pointer_answer);
            free(prompt);
            send_error(c, 500, "server_error", error);
            return;
        }
    }
    free(prompt);

    snprintf(id, sizeof(id), "chatcmpl-%lld", (long long)time(NULL));
    root = cJSON_CreateObject();
    choices = cJSON_CreateArray();
    choice = cJSON_CreateObject();
    message = cJSON_CreateObject();
    if (root == NULL || choices == NULL || choice == NULL || message == NULL) {
        cJSON_Delete(message);
        cJSON_Delete(choice);
        cJSON_Delete(choices);
        cJSON_Delete(root);
        generation_result_free(&result);
        send_error(c, 500, "server_error", "failed to build response");
        return;
    }

    json_add_string(root, "id", id);
    json_add_string(root, "object", "chat.completion");
    json_add_number(root, "created", (double)time(NULL));
    json_add_string(root, "model", state->cfg.model_id);
    json_add_string(root, "memory_action", memory_action_name(memory_action));
    json_add_number(root, "memory_priority", memory_priority);
    if (memory_auto_requested) {
        cJSON *automatic = cJSON_CreateObject();
        if (automatic != NULL) {
            json_add_number(
                automatic, "enabled",
                state->typed_writer != NULL ? 1.0 : 0.0);
            json_add_string(
                automatic, "decision",
                memory_action_name(memory_action));
            json_add_number(
                automatic, "attempted",
                auto_typed_handled ? 1.0 : 0.0);
            json_add_number(
                automatic, "stored",
                auto_memory_stored ? 1.0 : 0.0);
            json_add_string(
                automatic, "status",
                auto_memory_stored ? "stored" :
                auto_typed_handled ? "failed" :
                memory_action == MEMORY_ACTION_IGNORE ?
                    "ignored" : "not_typed");
            if (auto_memory_error[0] != '\0')
                json_add_string(
                    automatic, "error", auto_memory_error);
            if (auto_memory_stored) {
                cached_session_t *session =
                    find_session(state, session_id);
                if (session != NULL)
                    add_autonomous_memory_json(
                        automatic,
                        &auto_memory_result,
                        &session->events);
            }
            cJSON_AddItemToObject(root, "memory_auto", automatic);
        }
    }
    json_add_number(choice, "index", 0);
    json_add_string(message, "role", "assistant");
    {
        char *reasoning = NULL;
        char *content = split_reasoning(result.text, &reasoning);
        json_add_string(message, "content", content);
        if (reasoning != NULL) {
            json_add_string(message, "reasoning_content", reasoning);
        }
    }
    cJSON_AddItemToObject(choice, "message", message);
    json_add_string(choice, "finish_reason", result.finish_reason_length ? "length" : "stop");
    cJSON_AddItemToArray(choices, choice);
    cJSON_AddItemToObject(root, "choices", choices);
    cJSON_AddItemToObject(root, "usage", build_usage_json(&result));
    cJSON_AddItemToObject(root, "bitnet_perf", build_perf_json(&result));
    if (pointer_used) {
        cJSON *copy = cJSON_CreateObject();
        if (copy != NULL) {
            json_add_string(
                copy, "mode",
                state->resident == NULL ?
                "neural_typed_query_activation_then_compiled_pointer" :
                "neural_resident_activation_then_compiled_pointer");
            if (state->resident != NULL)
                resident_recall_metadata(copy,episodic_session);
            json_add_number(copy, "confidence", pointer_confidence);
            json_add_number(
                copy, "span_score", pointer_span_score);
            json_add_number(
                copy, "activation_exists_score",
                pointer_null_score);
            json_add_number(
                copy, "record_index", (double)pointer_record_index);
            cJSON_AddItemToObject(root, "memory_copy", copy);
        }
    }
    if (result.session_id != NULL) {
        cJSON_AddItemToObject(root, "bitnet_session", build_session_json(&result));
    }
    if (pointer_used && json_get_bool(request, "stream", 0)) {
        cJSON *chunk = build_chat_stream_chunk(state, id, result.text, NULL, 1);
        cJSON *final_chunk = build_chat_stream_chunk(state, id, NULL, "stop", 0);
        if (chunk == NULL || final_chunk == NULL) {
            send_error(c, 500, "server_error", "failed to build memory stream");
        } else {
            const char *fields[] = {"memory_action", "memory_auto", "memory_copy",
                                    "bitnet_session", "usage"};
            for (size_t i = 0; i < sizeof fields / sizeof fields[0]; ++i) {
                cJSON *value = cJSON_GetObjectItem(root, fields[i]);
                if (value != NULL) {
                    char *encoded = cJSON_Print(value);
                    cJSON *copy = encoded == NULL ? NULL : cJSON_Parse(encoded);
                    free(encoded);
                    if (copy != NULL) cJSON_AddItemToObject(final_chunk, fields[i], copy);
                }
            }
            send_sse_headers(c);
            send_sse_json(c, chunk);
            send_sse_json(c, final_chunk);
            send_sse_done(c);
        }
        cJSON_Delete(chunk);
        cJSON_Delete(final_chunk);
    } else {
        send_json(c, 200, root);
    }
    cJSON_Delete(root);
    generation_result_free(&result);
}

static void handle_completions(struct mg_connection *c, server_state_t *state,
                               cJSON *request) {
    char error[256];
    char id[64];
    char *prompt = NULL;
    const char *session_id = NULL;
    int max_tokens = 0;
    int reset_existing_session = 0;
    generation_result_t result;
    cJSON *root = NULL;
    cJSON *choices = NULL;
    cJSON *choice = NULL;

    prompt = build_completion_prompt(request);
    if (prompt == NULL) {
        send_error(c, 400, "invalid_request_error", "prompt must be a string");
        return;
    }
    session_id = json_get_string(request, "session_id", NULL);
    reset_existing_session = json_get_bool(request, "reset_session", 0);
    max_tokens = json_get_int(request, "max_tokens", state->cfg.default_max_tokens,
                              1, state->cfg.max_request_tokens);
    if (json_get_bool(request, "stream", 0)) {
        handle_streaming_completion(
            c, state, prompt, max_tokens, session_id,
            reset_existing_session, 0);
        free(prompt);
        return;
    }

    memset(&result, 0, sizeof(result));
    if (generate_text(
            state, prompt, max_tokens, session_id, reset_existing_session,
            &result, error, sizeof(error)) != 0) {
        free(prompt);
        send_error(c, 500, "server_error", error);
        return;
    }
    free(prompt);

    snprintf(id, sizeof(id), "cmpl-%lld", (long long)time(NULL));
    root = cJSON_CreateObject();
    choices = cJSON_CreateArray();
    choice = cJSON_CreateObject();
    if (root == NULL || choices == NULL || choice == NULL) {
        cJSON_Delete(choice);
        cJSON_Delete(choices);
        cJSON_Delete(root);
        generation_result_free(&result);
        send_error(c, 500, "server_error", "failed to build response");
        return;
    }
    json_add_string(root, "id", id);
    json_add_string(root, "object", "text_completion");
    json_add_number(root, "created", (double)time(NULL));
    json_add_string(root, "model", state->cfg.model_id);
    json_add_number(choice, "index", 0);
    json_add_string(choice, "text", result.text);
    json_add_string(choice, "finish_reason", result.finish_reason_length ? "length" : "stop");
    cJSON_AddItemToArray(choices, choice);
    cJSON_AddItemToObject(root, "choices", choices);
    cJSON_AddItemToObject(root, "usage", build_usage_json(&result));
    cJSON_AddItemToObject(root, "bitnet_perf", build_perf_json(&result));
    if (result.session_id != NULL) {
        cJSON_AddItemToObject(root, "bitnet_session", build_session_json(&result));
    }
    send_json(c, 200, root);
    cJSON_Delete(root);
    generation_result_free(&result);
}

static void handle_post_json(struct mg_connection *c,
                             struct mg_http_message *hm,
                             server_state_t *state,
                             int action) {
    char *body = dup_n(hm->body.buf, hm->body.len);
    cJSON *request = NULL;
    if (body == NULL) {
        send_error(c, 400, "invalid_request_error", "empty or invalid request body");
        return;
    }
    request = cJSON_Parse(body);
    free(body);
    if (request == NULL || !cJSON_IsObject(request)) {
        cJSON_Delete(request);
        send_error(c, 400, "invalid_request_error", "request body must be a JSON object");
        return;
    }
    if (state->resident != NULL && (action == 6 || action == 7 || action == 8)) {
        send_error(c,400,"unsupported_memory_operation",
            "resident mode supports autonomous writes and neural recall, not typed event APIs");
    } else if (action == 2) {
        handle_memory_state(c, state, request, 0);
    } else if (action == 3) {
        handle_memory_state(c, state, request, 1);
    } else if (action == 5) {
        handle_memory_extract(c, state, request);
    } else if (action == 6) {
        handle_memory_event(c, state, request);
    } else if (action == 7) {
        handle_memory_event_current(c, state, request);
    } else if (action == 8) {
        handle_memory_event_active(c, state, request);
    } else if (action == 11) {
        handle_memory_remember(c, state, request);
    } else if (action == 1) {
        handle_chat_completions(c, state, request);
    } else {
        handle_completions(c, state, request);
    }
    cJSON_Delete(request);
}

static int is_method(struct mg_http_message *hm, const char *method) {
    return mg_strcmp(hm->method, mg_str(method)) == 0;
}

static void http_handler(struct mg_connection *c, int ev, void *ev_data) {
    server_state_t *state = (server_state_t *)c->fn_data;
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *)ev_data;
        if (is_method(hm, "GET") && mg_match(hm->uri, mg_str("/v1/models"), NULL)) {
            handle_models(c, state);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/chat/completions"), NULL)) {
            handle_post_json(c, hm, state, 1);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/completions"), NULL)) {
            handle_post_json(c, hm, state, 0);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/memory/export"), NULL)) {
            handle_post_json(c, hm, state, 2);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/memory/import"), NULL)) {
            handle_post_json(c, hm, state, 3);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/memory/remember"), NULL)) {
            handle_post_json(c, hm, state, 11);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/memory/extract"), NULL)) {
            handle_post_json(c, hm, state, 5);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/memory/event"), NULL)) {
            handle_post_json(c, hm, state, 6);
        } else if (is_method(hm, "POST") &&
                   mg_match(
                       hm->uri, mg_str("/v1/memory/event/current"), NULL)) {
            handle_post_json(c, hm, state, 7);
        } else if (is_method(hm, "POST") &&
                   mg_match(
                       hm->uri, mg_str("/v1/memory/event/active"), NULL)) {
            handle_post_json(c, hm, state, 8);
        } else if (is_method(hm, "GET") && mg_match(hm->uri, mg_str("/health"), NULL)) {
            mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"status\":\"ok\"}\n");
        } else {
            send_error(c, 404, "not_found_error", "endpoint not found");
        }
    }
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s <model.gguf> [--host HOST] [--port PORT] [--model-id ID]\n"
            "       [--ctx TOKENS] [--max-tokens TOKENS] [--default-max-tokens TOKENS]\n"
            "       [--repeat-last-n N]\n"
            "       [--repeat-penalty PENALTY]\n"
            "       [--lora PATH] [--lora-scale SCALE]\n"
            "       [--memory-state-dir DIR] [--episodic-memory]\n"
            "       [--memory-controller PATH]"
            " [--typed-pair-model PATH]"
            " [--typed-link-model PATH]"
            " [--typed-query-model PATH]"
            " [--typed-writer-model PATH] [--resident-model PATH]\n",
            argv0);
}

static void free_memory_sidecars(server_state_t *state) {
    if (state == NULL) return;
    metis_typed_link_model_free(state->typed_link);
    metis_typed_query_activator_free(state->typed_query);
    metis_typed_pair_encoder_free(state->typed_pair);
    metis_typed_writer_model_free(state->typed_writer);
    metis_memory_controller_free(state->memory_controller);
    resident_model_free(state->resident);
    state->resident = NULL;
    state->typed_link = NULL;
    state->typed_query = NULL;
    state->typed_pair = NULL;
    state->typed_writer = NULL;
    state->memory_controller = NULL;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    char listen_url[256];
    struct mg_mgr mgr;
    server_state_t state;

    memset(&state, 0, sizeof(state));
    state.cfg.host = DEFAULT_HOST;
    state.cfg.port = DEFAULT_PORT;
    state.cfg.model_id = DEFAULT_MODEL_ID;
    state.cfg.max_context_tokens = DEFAULT_CONTEXT_TOKENS;
    state.cfg.default_max_tokens = DEFAULT_MAX_TOKENS;
    state.cfg.max_request_tokens = MAX_REQUEST_TOKENS;
    state.cfg.repeat_last_n = DEFAULT_REPEAT_LAST_N;
    state.cfg.repeat_penalty = DEFAULT_REPEAT_PENALTY;
    state.cfg.lora_scale = 1.0f;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    model_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        if (strcmp(argv[index], "--host") == 0 && index + 1 < argc) {
            state.cfg.host = argv[++index];
        } else if (strcmp(argv[index], "--port") == 0 &&
                   index + 1 < argc) {
            state.cfg.port = argv[++index];
        } else if (strcmp(argv[index], "--model-id") == 0 &&
                   index + 1 < argc) {
            state.cfg.model_id = argv[++index];
        } else if (strcmp(argv[index], "--ctx") == 0 &&
                   index + 1 < argc) {
            state.cfg.max_context_tokens = parse_int_arg(
                argv[++index], DEFAULT_CONTEXT_TOKENS,
                MIN_CONTEXT_TOKENS, 32768);
        } else if (strcmp(argv[index], "--max-tokens") == 0 &&
                   index + 1 < argc) {
            state.cfg.max_request_tokens = parse_int_arg(
                argv[++index], MAX_REQUEST_TOKENS,
                1, MAX_REQUEST_TOKENS);
        } else if (strcmp(argv[index], "--default-max-tokens") == 0 &&
                   index + 1 < argc) {
            state.cfg.default_max_tokens = parse_int_arg(
                argv[++index], DEFAULT_MAX_TOKENS,
                1, MAX_REQUEST_TOKENS);
        } else if (strcmp(argv[index], "--repeat-last-n") == 0 &&
                   index + 1 < argc) {
            state.cfg.repeat_last_n = parse_int_arg(
                argv[++index], DEFAULT_REPEAT_LAST_N, 0, 4096);
        } else if (strcmp(argv[index], "--repeat-penalty") == 0 &&
                   index + 1 < argc) {
            state.cfg.repeat_penalty = parse_float_arg(
                argv[++index], DEFAULT_REPEAT_PENALTY, 1.0f, 10.0f);
        } else if (strcmp(argv[index], "--lora") == 0 &&
                   index + 1 < argc) {
            state.cfg.lora_path = argv[++index];
        } else if (strcmp(argv[index], "--lora-scale") == 0 &&
                   index + 1 < argc) {
            state.cfg.lora_scale = parse_float_arg(
                argv[++index], 1.0f, 0.0f, 100.0f);
        } else if (strcmp(argv[index], "--memory-state-dir") == 0 &&
                   index + 1 < argc) {
            state.cfg.memory_state_dir = argv[++index];
        } else if (strcmp(argv[index], "--episodic-memory") == 0) {
            state.cfg.episodic_memory = 1;
        } else if (strcmp(argv[index], "--memory-controller") == 0 &&
                   index + 1 < argc) {
            state.cfg.memory_controller_path = argv[++index];
        } else if (strcmp(argv[index], "--typed-pair-model") == 0 &&
                   index + 1 < argc) {
            state.cfg.typed_pair_path = argv[++index];
        } else if (strcmp(argv[index], "--typed-link-model") == 0 &&
                   index + 1 < argc) {
            state.cfg.typed_link_path = argv[++index];
        } else if (strcmp(argv[index], "--typed-query-model") == 0 &&
                   index + 1 < argc) {
            state.cfg.typed_query_path = argv[++index];
        } else if (strcmp(argv[index], "--resident-model") == 0 &&
                   index + 1 < argc) {
            state.cfg.resident_path = argv[++index];
        } else if (strcmp(argv[index], "--typed-writer-model") == 0 &&
                   index + 1 < argc) {
            state.cfg.typed_writer_path = argv[++index];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (state.cfg.default_max_tokens > state.cfg.max_request_tokens)
        state.cfg.default_max_tokens = state.cfg.max_request_tokens;
    if (state.cfg.max_context_tokens <
        state.cfg.max_request_tokens + MIN_CONTEXT_TOKENS)
        state.cfg.max_context_tokens =
            state.cfg.max_request_tokens + MIN_CONTEXT_TOKENS;

    if (state.cfg.episodic_memory &&
        state.cfg.memory_state_dir == NULL) {
        fprintf(stderr,
                "--episodic-memory requires --memory-state-dir\n");
        return 1;
    }
    if (state.cfg.memory_controller_path != NULL &&
        !state.cfg.episodic_memory) {
        fprintf(stderr,
                "--memory-controller requires --episodic-memory\n");
        return 1;
    }
    if(state.cfg.resident_path != NULL && !state.cfg.episodic_memory) {
        fprintf(stderr,"--resident-model requires complete typed writer configuration\n");
        return 1;
    }
    {
        int typed_count =
            (state.cfg.typed_pair_path != NULL) +
            (state.cfg.typed_link_path != NULL) +
            (state.cfg.typed_query_path != NULL) +
            (state.cfg.typed_writer_path != NULL);
        if (typed_count != 0 && typed_count != 4) {
            fprintf(stderr,
                    "typed memory requires pair, link, query, and writer "
                    "models together\n");
            return 1;
        }
        if (state.cfg.episodic_memory && typed_count != 4) {
            fprintf(stderr,
                    "--episodic-memory requires the complete typed-memory "
                    "model set\n");
            return 1;
        }
        if (typed_count != 0 && !state.cfg.episodic_memory) {
            fprintf(stderr,
                    "typed memory models require --episodic-memory\n");
            return 1;
        }
        if (typed_count != 0 && state.cfg.lora_path != NULL) {
            fprintf(stderr,
                    "typed memory cannot use LoRA because its hidden-state "
                    "domain is bound to the unmodified backbone\n");
            return 1;
        }
    }

    fprintf(stderr, "Loading model: %s\n", model_path);
    state.model = bitnet_load_model(model_path);
    if (state.model == NULL) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    if (state.cfg.lora_path != NULL) {
        if (bitnet_load_lora(
                state.model, state.cfg.lora_path,
                state.cfg.lora_scale) != 0) {
            fprintf(stderr, "Failed to load LoRA: %s\n",
                    state.cfg.lora_path);
            bitnet_free_model(state.model);
            return 1;
        }
        fprintf(stderr, "Loaded LoRA: %s (tensors=%d, scale=%.4f)\n",
                state.cfg.lora_path,
                bitnet_lora_count(state.model),
                state.cfg.lora_scale);
    }
    if (state.cfg.memory_controller_path != NULL) {
        char controller_error[256];
        state.memory_controller = metis_memory_controller_load(
            state.cfg.memory_controller_path, model_path,
            bitnet_embedding_length(state.model),
            controller_error, sizeof controller_error);
        if (state.memory_controller == NULL) {
            fprintf(stderr, "Failed to load memory controller: %s\n",
                    controller_error);
            bitnet_free_model(state.model);
            return 1;
        }
        fprintf(stderr,
                "Memory action controller loaded: %s "
                "(rank=%d pooling=%d)\n",
                state.cfg.memory_controller_path,
                state.memory_controller->rank,
                state.memory_controller->pooling);
    }
    if (state.cfg.typed_pair_path != NULL) {
        char typed_error[256];
        int compatible;
        state.typed_pair = metis_typed_pair_encoder_load(
            state.cfg.typed_pair_path, model_path,
            bitnet_embedding_length(state.model),
            typed_error, sizeof typed_error);
        if (state.typed_pair == NULL) {
            fprintf(stderr, "Failed to load typed pair model: %s\n",
                    typed_error);
            free_memory_sidecars(&state);
            bitnet_free_model(state.model);
            return 1;
        }
        state.typed_link = metis_typed_link_model_load(
            state.cfg.typed_link_path, model_path,
            typed_error, sizeof typed_error);
        compatible =
            state.typed_link != NULL &&
            state.typed_link->rank == state.typed_pair->rank &&
            state.typed_link->pair_feature_dim ==
                state.typed_pair->head_width &&
            memcmp(
                state.typed_link->backbone_sha256,
                state.typed_pair->backbone_sha256,
                sizeof state.typed_link->backbone_sha256) == 0;
        if (!compatible) {
            fprintf(stderr, "Failed to load typed link model: %s%s\n",
                    typed_error,
                    state.typed_link != NULL ?
                        "geometry does not match pair encoder" : "");
            free_memory_sidecars(&state);
            bitnet_free_model(state.model);
            return 1;
        }
        state.typed_writer = metis_typed_writer_model_load(
            state.cfg.typed_writer_path, model_path,
            bitnet_embedding_length(state.model),
            typed_error, sizeof typed_error);
        compatible =
            state.typed_writer != NULL &&
            state.typed_writer->hidden_dim ==
                state.typed_pair->hidden_dim &&
            state.typed_writer->rank == state.typed_pair->rank &&
            state.typed_writer->band_count ==
                state.typed_pair->band_count &&
            memcmp(
                state.typed_writer->backbone_sha256,
                state.typed_pair->backbone_sha256,
                sizeof state.typed_writer->backbone_sha256) == 0;
        for (int band = 0;
             compatible && band < state.typed_writer->band_count;
             ++band) {
            compatible =
                state.typed_writer->band_start[band] ==
                    state.typed_pair->band_start[band] &&
                state.typed_writer->band_end[band] ==
                    state.typed_pair->band_end[band];
        }
        if (!compatible) {
            fprintf(stderr, "Failed to load typed writer model: %s%s\n",
                    typed_error,
                    state.typed_writer != NULL ?
                        "geometry does not match pair encoder" : "");
            free_memory_sidecars(&state);
            bitnet_free_model(state.model);
            return 1;
        }
        state.typed_query = metis_typed_query_activator_load(
            state.cfg.typed_query_path, model_path,
            state.cfg.typed_pair_path,
            state.typed_pair->head_width * 2,
            typed_error, sizeof typed_error);
        if (state.typed_query == NULL) {
            fprintf(stderr,
                    "Failed to load typed query activator: %s\n",
                    typed_error);
            free_memory_sidecars(&state);
            bitnet_free_model(state.model);
            return 1;
        }
        fprintf(stderr,
                "Typed memory loaded: writer=%s pair=%s link=%s query=%s "
                "(rank=%d bands=%d)\n",
                state.cfg.typed_writer_path,
                state.cfg.typed_pair_path,
                state.cfg.typed_link_path,
                state.cfg.typed_query_path,
                state.typed_pair->rank,
                state.typed_pair->band_count);
    }

    if(state.cfg.resident_path != NULL) {
        state.resident=resident_model_load(state.cfg.resident_path,model_path);
        if(!state.resident) {
            fprintf(stderr,"Invalid resident model or backbone binding\n");
            free_memory_sidecars(&state);bitnet_free_model(state.model);return 1;
        }
        fprintf(stderr,"Experimental neural resident recall enabled (32 events, 128 tokens; no delete)\n");
    }
    snprintf(listen_url, sizeof listen_url, "http://%s:%s",
             state.cfg.host, state.cfg.port);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    mg_log_set(MG_LL_ERROR);
    mg_mgr_init(&mgr);
    if (mg_http_listen(
            &mgr, listen_url, http_handler, &state) == NULL) {
        fprintf(stderr, "Failed to listen on %s\n", listen_url);
        mg_mgr_free(&mgr);
        free_memory_sidecars(&state);
        bitnet_free_model(state.model);
        return 1;
    }
    fprintf(stderr, "OpenAI-compatible server listening on %s\n",
            listen_url);
    fprintf(stderr, "Model id: %s\n", state.cfg.model_id);
    fprintf(stderr,
            "Default max_tokens: %d, request max_tokens cap: %d, ctx: %d\n",
            state.cfg.default_max_tokens,
            state.cfg.max_request_tokens,
            state.cfg.max_context_tokens);
    while (!g_stop)
        mg_mgr_poll(&mgr, 100);
    mg_mgr_free(&mgr);
    free_sessions(&state);
    free_memory_sidecars(&state);
    bitnet_free_model(state.model);
    return 0;
}
