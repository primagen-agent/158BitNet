#include "bitnet.h"
#include "metis/episodic_store.h"
#include "metis/memory_controller.h"
#include "metis/memory_pointer.h"

#include "third_party/cJSON/cJSON.h"
#include "third_party/mongoose/mongoose.h"

#include <signal.h>
#include <ctype.h>
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
    const char *memory_model_path;
    const char *memory_state_dir;
    int episodic_memory;
    int episodic_top_k;
    const char *memory_controller_path;
    const char *memory_pointer_path;
    float episodic_lexical_weight;
    float memory_address_threshold;
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
    time_t last_used;
    struct cached_session *next;
} cached_session_t;

typedef struct server_state {
    bitnet_model_t *model;
    server_config_t cfg;
    cached_session_t *sessions;
    metis_memory_controller_t *memory_controller;
    metis_memory_pointer_t *memory_pointer;
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
    int episodic_key_dim;
    if (session == NULL) return;
    episodic_key_dim = session->episodic.key_dim;
    bitnet_reset_context(session->ctx);
    bitnet_memory_reset(session->ctx);   /* no-op without metis attached */
    session->history_count = 0;
    session->transcript_len = 0;
    if (session->transcript != NULL) session->transcript[0] = '\0';
    metis_episodic_clear(&session->episodic);
    if (episodic_key_dim > 0)
        (void)metis_episodic_configure_keys(
            &session->episodic, episodic_key_dim);
}

/* Clear transient conversation/KV state without touching attached memory.
 * Used before importing a committed snapshot into an existing session. */
static void reset_session_context_only(cached_session_t *session) {
    if (session == NULL) return;
    bitnet_reset_context(session->ctx);
    bitnet_memory_discard_captured(session->ctx);
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
    if (state->memory_controller != NULL &&
        metis_episodic_configure_keys(
            &session->episodic,
            state->memory_controller->rank) != 0) {
        free(session->id);
        free(session);
        return NULL;
    }
    session->capacity = state->cfg.max_context_tokens;
    session->ctx = bitnet_create_context(state->model, session->capacity);
    session->history_tokens = (int *)calloc((size_t)session->capacity, sizeof(*session->history_tokens));
    if (session->id == NULL || session->ctx == NULL || session->history_tokens == NULL) {
        free(session->history_tokens);
        bitnet_free_context(session->ctx);
        free(session->id);
        free(session);
        return NULL;
    }
    session->last_used = time(NULL);
    session->next = state->sessions;
    state->sessions = session;

    /* Memory model (if one was loaded in main): bind it to this session's
     * context. bitnet_model_t is opaque here; cfg.memory_model_path != NULL
     * implies the model carries a loaded memory model because main exits on
     * load failure. bitnet_context_attach_memory needs the same model the
     * context was created from (checked there). */
    if (state->cfg.memory_model_path != NULL) {
        if (bitnet_context_attach_memory(session->ctx, state->model) != 0) {
            state->sessions = session->next;
            free(session->history_tokens);
            bitnet_free_context(session->ctx);
            free(session->id);
            free(session);
            return NULL;
        }
    }
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
    /* Memory-model sessions use the reference Metis protocol rendering:
     * plain assistant header, NO injected think block. Probe2 evidence:
     * the trained memory models collapse (<answer>/<think> fragments)
     * when a think block precedes the answer; without it recall works.
     * Ignore enable_thinking in this mode — the memory training
     * distribution defines the template. */
    if (state != NULL && state->cfg.memory_model_path != NULL) {
        return build_chat_prompt_chatml(root, 0);
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
        /* v9b protocol (best measured, 52% on v9): the exchange is
         * committed AFTER generation (user prefill + assistant ack rows
         * accumulate and commit in finish_generation / the streaming
         * tail) — matching the reference TRAINER, which commits each
         * full chunk incl. acks. Nothing to do here; decode rows
         * accumulate. bitnet_memory_commit is a no-op without memory. */
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
    /* Commit before transferring result ownership so a failure can be
     * reported cleanly. Reset the memory state because a failed multi-layer
     * commit may already have updated an earlier layer. */
    if (gen->session != NULL && gen->emitted > 0 &&
        bitnet_memory_active(gen->ctx) && bitnet_memory_commit(gen->ctx) != 0) {
        bitnet_memory_reset(gen->ctx);
        snprintf(error, error_size, "failed to commit memory model states");
        return -1;
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

    if (prepare_generation(state, prompt, max_tokens, session_id,
                           reset_existing_session, &gen, error, error_size) != 0) {
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
    if (prepare_generation(state, prompt, max_tokens, session_id,
                           reset_existing_session, &gen, error, sizeof(error)) != 0) {
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
    if (!failed && gen.session != NULL && gen.emitted > 0 &&
        bitnet_memory_active(gen.ctx) && bitnet_memory_commit(gen.ctx) != 0) {
        cJSON *error_chunk = cJSON_CreateObject();
        bitnet_memory_reset(gen.ctx);
        if (error_chunk != NULL) {
            json_add_string(error_chunk, "error", "failed to commit memory model states");
            send_sse_json(c, error_chunk);
            cJSON_Delete(error_chunk);
        }
        failed = 1;
    }
    if (!failed) {
        const char *finish_reason = gen.finish_reason_length ? "length" : "stop";
        cJSON *final_chunk = is_chat ?
            build_chat_stream_chunk(state, id, NULL, finish_reason, 0) :
            build_completion_stream_chunk(state, id, "", finish_reason);
        if (final_chunk != NULL) {
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

static int memory_state_path(const server_state_t *state, const char *sid,
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
    return snprintf(out, out_size, "%s/%s.bnstate",
                    state->cfg.memory_state_dir, sid) < (int)out_size ? 0 : -1;
}

static int episodic_state_path(const server_state_t *state, const char *sid,
                               char *out, size_t out_size) {
    char base[1024];
    size_t length;
    if (memory_state_path(state, sid, base, sizeof base) != 0) return -1;
    length = strlen(base);
    if (length < strlen(".bnstate")) return -1;
    base[length - strlen(".bnstate")] = '\0';
    return snprintf(out, out_size, "%s.bnepisodic", base) < (int)out_size
        ? 0 : -1;
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

static char *prepend_episodic_context(
    const char *prompt, const char *records) {
    static const char prefix[] =
        "<|im_start|>system\n"
        "The following are exact long-term memory records. Prefer newer "
        "records when values conflict. A delete or forget record invalidates "
        "the matching older value. Answer from these records exactly.\n";
    static const char suffix[] = "<|im_end|>\n";
    size_t size;
    char *result;
    if (prompt == NULL || records == NULL) return NULL;
    size = strlen(prefix) + strlen(records) + strlen(suffix) +
           strlen(prompt) + 1;
    result = (char *)malloc(size);
    if (result == NULL) return NULL;
    snprintf(result, size, "%s%s%s%s", prefix, records, suffix, prompt);
    return result;
}

static int controller_encode_key(
    server_state_t *state, const char *text, int is_query, float *output) {
    static const char query_prefix[] =
        "Find the stored conversation facts needed to answer this question: ";
    char *encoded_text = NULL;
    const char *input = text;
    int tokens[512];
    int count;
    bitnet_context_t *context;
    const float *hidden;
    if (state == NULL || state->memory_controller == NULL ||
        text == NULL || output == NULL) return -1;
    if (is_query) {
        size_t size = strlen(query_prefix) + strlen(text) + 1;
        encoded_text = (char *)malloc(size);
        if (encoded_text == NULL) return -1;
        snprintf(encoded_text, size, "%s%s", query_prefix, text);
        input = encoded_text;
    }
    count = bitnet_tokenize_ex(
        state->model, input, tokens,
        (int)(sizeof tokens / sizeof tokens[0]), 1);
    free(encoded_text);
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
    if (hidden == NULL ||
        metis_memory_controller_project(
            state->memory_controller, hidden, is_query, output) != 0) {
        bitnet_free_context(context);
        return -1;
    }
    bitnet_free_context(context);
    return 0;
}

static int controller_classify_action(
    server_state_t *state, const char *text, float *confidence) {
    int tokens[512];
    int count;
    int action;
    bitnet_context_t *context;
    const float *hidden;
    if (state == NULL || state->memory_controller == NULL ||
        state->memory_controller->action_projection == NULL ||
        text == NULL || text[0] == '\0')
        return -1;
    count = bitnet_tokenize_ex(
        state->model, text, tokens,
        (int)(sizeof tokens / sizeof tokens[0]), 1);
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

static int store_episodic_record(
    server_state_t *state, cached_session_t *session, const char *text,
    memory_record_action_t action) {
    float *key = NULL;
    int result;
    if (state == NULL || session == NULL || text == NULL) return -1;
    if (action == MEMORY_ACTION_IGNORE) return 0;
    if (state->memory_controller != NULL) {
        if (session->episodic.key_dim == 0 &&
            metis_episodic_configure_keys(
                &session->episodic,
                state->memory_controller->rank) != 0)
            return -10;
        key = (float *)malloc(
            (size_t)state->memory_controller->rank * sizeof(float));
        if (key != NULL &&
            controller_encode_key(state, text, 0, key) != 0) {
            free(key);
            key = NULL;
        }
        if (key == NULL && action != MEMORY_ACTION_STORE) return -11;
    }
    if (action == MEMORY_ACTION_STORE) {
        result = metis_episodic_add_with_key(
            &session->episodic, text, key);
    } else if (key == NULL) {
        result = -1;
    } else if (action == MEMORY_ACTION_UPDATE) {
        float threshold = (
            state->memory_controller != NULL &&
            state->memory_controller->action_projection != NULL) ?
            state->memory_controller->address_threshold :
            state->cfg.memory_address_threshold;
        result = metis_episodic_upsert_with_key(
            &session->episodic, text, key, threshold, NULL);
    } else {
        float threshold = (
            state->memory_controller != NULL &&
            state->memory_controller->action_projection != NULL) ?
            state->memory_controller->address_threshold :
            state->cfg.memory_address_threshold;
        result = metis_episodic_tombstone_with_key(
            &session->episodic, text, key, threshold, NULL);
    }
    free(key);
    return result;
}

static char *retrieve_episodic_context(
    server_state_t *state, cached_session_t *session,
    const char *query, int top_k) {
    float *query_key = NULL;
    char *context;
    if (state == NULL || session == NULL || query == NULL) return NULL;
    if (state->memory_controller != NULL) {
        query_key = (float *)malloc(
            (size_t)state->memory_controller->rank * sizeof(float));
        if (query_key != NULL &&
            controller_encode_key(state, query, 1, query_key) == 0) {
            context = metis_episodic_build_hybrid_context(
                &session->episodic, query, query_key,
                state->cfg.episodic_lexical_weight, top_k);
            free(query_key);
            return context;
        }
        free(query_key);
    }
    return metis_episodic_build_context(
        &session->episodic, query, top_k);
}

static int rank_episodic_records(
    server_state_t *state, cached_session_t *session, const char *query,
    size_t *indices, int max_results) {
    float *query_key = NULL;
    int result;
    if (state == NULL || session == NULL || query == NULL ||
        indices == NULL || max_results <= 0)
        return -1;
    if (state->memory_controller != NULL) {
        query_key = (float *)malloc(
            (size_t)state->memory_controller->rank * sizeof(float));
        if (query_key == NULL) return -1;
        if (controller_encode_key(state, query, 1, query_key) != 0) {
            free(query_key);
            return -1;
        }
    }
    result = metis_episodic_rank_hybrid(
        &session->episodic, query, query_key,
        state->cfg.episodic_lexical_weight, indices, max_results);
    free(query_key);
    return result;
}

static int pointer_extract_record(
    server_state_t *state, const char *question, const char *record,
    char **answer, float *confidence) {
    static const char question_label[] = "Question: ";
    static const char record_label[] = "\nMemory record:\n";
    int prefix_tokens[320], record_tokens[192], tokens[512];
    char *prefix = NULL;
    size_t prefix_size, answer_len = 0, answer_cap = 0;
    int prefix_count, record_count, total_count;
    bitnet_context_t *context = NULL;
    const float *hidden;
    size_t start, end;
    int selected;
    if (state == NULL || state->memory_pointer == NULL ||
        question == NULL || record == NULL || answer == NULL ||
        confidence == NULL)
        return -1;
    *answer = NULL;
    prefix_size = strlen(question_label) + strlen(question) +
                  strlen(record_label) + 1;
    prefix = (char *)malloc(prefix_size);
    if (prefix == NULL) return -1;
    snprintf(prefix, prefix_size, "%s%s%s",
             question_label, question, record_label);
    prefix_count = bitnet_tokenize_ex(
        state->model, prefix, prefix_tokens,
        (int)(sizeof prefix_tokens / sizeof prefix_tokens[0]), 1);
    free(prefix);
    record_count = bitnet_tokenize_ex(
        state->model, record, record_tokens,
        (int)(sizeof record_tokens / sizeof record_tokens[0]), 0);
    if (prefix_count <= 0 || record_count <= 0 ||
        prefix_count + record_count > (int)(sizeof tokens / sizeof tokens[0]))
        return -1;
    memcpy(tokens, prefix_tokens, (size_t)prefix_count * sizeof(int));
    memcpy(tokens + prefix_count, record_tokens,
           (size_t)record_count * sizeof(int));
    total_count = prefix_count + record_count;
    context = bitnet_create_context(
        state->model, total_count < MIN_CONTEXT_TOKENS ?
        MIN_CONTEXT_TOKENS : total_count + 1);
    if (context == NULL ||
        bitnet_eval(context, tokens, total_count) != 0) {
        bitnet_free_context(context);
        return -1;
    }
    hidden = bitnet_get_last_eval_hidden(context);
    if (hidden == NULL ||
        bitnet_last_eval_hidden_count(context) != total_count) {
        bitnet_free_context(context);
        return -1;
    }
    selected = metis_memory_pointer_select(
        state->memory_pointer,
        hidden + (size_t)prefix_count *
            (size_t)state->memory_pointer->hidden_dim,
        (size_t)record_count, &start, &end, confidence);
    bitnet_free_context(context);
    if (selected <= 0) return selected;
    for (size_t index = start; index <= end; ++index) {
        char piece[256];
        int length = bitnet_decode_token(
            state->model, record_tokens[index], piece, sizeof piece);
        if (length < 0 ||
            append_bytes(
                answer, &answer_len, &answer_cap,
                piece, (size_t)length) != 0) {
            free(*answer);
            *answer = NULL;
            return -1;
        }
    }
    if (*answer == NULL) return 0;
    {
        char *begin = *answer;
        char *finish = begin + strlen(begin);
        while (*begin != '\0' && isspace((unsigned char)*begin)) ++begin;
        while (finish > begin &&
               isspace((unsigned char)finish[-1])) --finish;
        if (begin != *answer)
            memmove(*answer, begin, (size_t)(finish - begin));
        (*answer)[finish - begin] = '\0';
        if ((*answer)[0] == '\0') {
            free(*answer);
            *answer = NULL;
            return 0;
        }
    }
    return 1;
}

static int extract_episodic_answer(
    server_state_t *state, cached_session_t *session, const char *query,
    char **answer, float *confidence, size_t *record_index) {
    size_t indices[3];
    char *best_answer = NULL;
    float best_confidence = -1.0f;
    int count;
    if (answer == NULL || confidence == NULL || record_index == NULL)
        return -1;
    *answer = NULL;
    count = rank_episodic_records(
        state, session, query, indices,
        (int)(sizeof indices / sizeof indices[0]));
    if (count <= 0) return count;
    for (int i = 0; i < count; ++i) {
        char *candidate = NULL;
        float candidate_confidence = 0.0f;
        int result = pointer_extract_record(
            state, query, session->episodic.records[indices[i]],
            &candidate, &candidate_confidence);
        if (result > 0 && candidate_confidence > best_confidence) {
            free(best_answer);
            best_answer = candidate;
            best_confidence = candidate_confidence;
            *record_index = indices[i];
        } else {
            free(candidate);
        }
    }
    if (best_answer == NULL) return 0;
    *answer = best_answer;
    *confidence = best_confidence;
    return 1;
}

static void rebuild_episodic_keys(
    server_state_t *state, cached_session_t *session) {
    if (state == NULL || session == NULL ||
        state->memory_controller == NULL) return;
    if (metis_episodic_configure_keys(
            &session->episodic,
            state->memory_controller->rank) != 0) return;
    for (size_t index = 0; index < session->episodic.count; ++index) {
        float *key = (float *)malloc(
            (size_t)state->memory_controller->rank * sizeof(float));
        if (key == NULL) return;
        if (controller_encode_key(
                state, session->episodic.records[index], 0, key) == 0)
            (void)metis_episodic_add_with_key(
                &session->episodic,
                session->episodic.records[index], key);
        free(key);
    }
}

static void handle_memory_state(struct mg_connection *c, server_state_t *state,
                                cJSON *request, int do_import) {
    const char *sid = json_get_string(request, "session_id", NULL);
    cached_session_t *session;
    char path[1024];
    char episodic_path[1024];
    metis_episodic_store_t imported_episodic;
    int has_imported_episodic = 0;
    cJSON *root;
    metis_episodic_init(&imported_episodic);
    if (state->cfg.memory_state_dir == NULL) {
        send_error(c, 403, "memory_state_disabled", "memory state directory is not configured");
        return;
    }
    if (memory_state_path(state, sid, path, sizeof path) != 0) {
        send_error(c, 400, "invalid_request_error", "invalid session_id");
        return;
    }
    session = find_session(state, sid);
    if (do_import && session == NULL) session = create_session(state, sid);
    if (session == NULL) {
        send_error(c, 404, "not_found_error", "session not found");
        return;
    }
    if (state->cfg.episodic_memory &&
        episodic_state_path(
            state, sid, episodic_path, sizeof episodic_path) != 0) {
        send_error(c, 500, "server_error",
                   "failed to resolve episodic memory path");
        return;
    }
    if (state->cfg.episodic_memory && do_import) {
        if (metis_episodic_load(
                &imported_episodic, episodic_path) != 0) {
            send_error(c, 500, "server_error",
                       "failed to import episodic memory records");
            return;
        }
        has_imported_episodic = 1;
    }
    if (do_import) reset_session_context_only(session);
    if (state->cfg.memory_model_path != NULL) {
        if ((do_import ? bitnet_memory_import(session->ctx, path) :
                         bitnet_memory_export(session->ctx, path)) != 0) {
            metis_episodic_free(&imported_episodic);
            send_error(
                c, 500, "server_error", do_import ?
                "failed to import memory state" :
                "failed to export memory state");
            return;
        }
    }
    if (state->cfg.episodic_memory) {
        if (do_import) {
            metis_episodic_clear(&session->episodic);
            session->episodic = imported_episodic;
            metis_episodic_init(&imported_episodic);
            has_imported_episodic = 0;
            rebuild_episodic_keys(state, session);
        } else if (metis_episodic_save(
                       &session->episodic, episodic_path) != 0) {
            send_error(c, 500, "server_error",
                       "failed to export episodic memory records");
            return;
        }
    }
    if (has_imported_episodic)
        metis_episodic_free(&imported_episodic);
    root = cJSON_CreateObject();
    if (root == NULL) {
        send_error(c, 500, "server_error", "failed to build response");
        return;
    }
    json_add_string(root, "status", "ok");
    json_add_string(root, "session_id", sid);
    json_add_number(
        root, "episodic_records",
        (double)metis_episodic_count(&session->episodic));
    json_add_number(
        root, "matrix_memory",
        state->cfg.memory_model_path != NULL ? 1.0 : 0.0);
    send_json(c, 200, root);
    cJSON_Delete(root);
}

static void handle_memory_search(
    struct mg_connection *c, server_state_t *state, cJSON *request) {
    const char *sid = json_get_string(request, "session_id", NULL);
    const char *query = json_get_string(request, "query", NULL);
    int top_k = json_get_int(
        request, "top_k", state->cfg.episodic_top_k, 1, 32);
    cached_session_t *session;
    char *context;
    cJSON *root;
    if (!state->cfg.episodic_memory) {
        send_error(c, 403, "episodic_memory_disabled",
                   "episodic memory is not enabled");
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
    context = retrieve_episodic_context(
        state, session, query, top_k);
    root = cJSON_CreateObject();
    if (root == NULL) {
        free(context);
        send_error(c, 500, "server_error", "failed to build response");
        return;
    }
    json_add_string(root, "status", "ok");
    json_add_string(root, "session_id", sid);
    json_add_number(
        root, "episodic_records",
        (double)metis_episodic_count(&session->episodic));
    json_add_string(root, "context", context == NULL ? "" : context);
    send_json(c, 200, root);
    cJSON_Delete(root);
    free(context);
}

static void handle_memory_extract(
    struct mg_connection *c, server_state_t *state, cJSON *request) {
    const char *sid = json_get_string(request, "session_id", NULL);
    const char *query = json_get_string(request, "query", NULL);
    cached_session_t *session;
    char *answer = NULL;
    float confidence = 0.0f;
    size_t record_index = 0;
    int extracted;
    cJSON *root;
    if (state->memory_pointer == NULL) {
        send_error(c, 403, "memory_pointer_disabled",
                   "memory pointer is not configured");
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
    extracted = extract_episodic_answer(
        state, session, query, &answer, &confidence, &record_index);
    if (extracted < 0) {
        send_error(c, 500, "server_error",
                   "memory pointer extraction failed");
        return;
    }
    root = cJSON_CreateObject();
    if (root == NULL) {
        free(answer);
        send_error(c, 500, "server_error", "failed to build response");
        return;
    }
    json_add_string(root, "status", extracted ? "extracted" : "fallback");
    json_add_string(root, "session_id", sid);
    json_add_string(root, "answer", answer == NULL ? "" : answer);
    json_add_number(root, "confidence", confidence);
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
    char *episodic_context = NULL;
    char *augmented_prompt = NULL;
    char *pointer_answer = NULL;
    const char *session_id = NULL;
    const char *user_content = NULL;
    float pointer_confidence = 0.0f;
    size_t pointer_record_index = 0;
    int pointer_used = 0;
    memory_record_action_t memory_action = MEMORY_ACTION_IGNORE;
    int max_tokens = 0;
    int reset_existing_session = 0;
    int generation_reset = 0;
    cached_session_t *episodic_session = NULL;
    generation_result_t result;
    cJSON *root = NULL;
    cJSON *choices = NULL;
    cJSON *choice = NULL;
    cJSON *message = NULL;

    session_id = json_get_string(request, "session_id", NULL);
    reset_existing_session = json_get_bool(request, "reset_session", 0);
    generation_reset = reset_existing_session;
    max_tokens = json_get_int(request, "max_tokens", state->cfg.default_max_tokens,
                              1, state->cfg.max_request_tokens);
    user_content = last_user_content(request);
    memory_action = request_memory_action(state, request, user_content);
    if (state->cfg.episodic_memory &&
        session_id != NULL && session_id[0] != '\0') {
        episodic_session = find_session(state, session_id);
        if (episodic_session != NULL) {
            if (reset_existing_session) reset_session(episodic_session);
            else reset_session_context_only(episodic_session);
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
    if (state->cfg.episodic_memory && episodic_session != NULL &&
        memory_action == MEMORY_ACTION_IGNORE) {
        episodic_context = retrieve_episodic_context(
            state, episodic_session, user_content,
            state->cfg.episodic_top_k);
        if (episodic_context != NULL) {
            augmented_prompt = prepend_episodic_context(
                prompt, episodic_context);
            if (augmented_prompt == NULL) {
                free(episodic_context);
                free(prompt);
                send_error(c, 500, "server_error",
                           "failed to build episodic memory prompt");
                return;
            }
            free(prompt);
            prompt = augmented_prompt;
            if (strstr(
                    episodic_context,
                    "[memory 1] DELETED MEMORY") != NULL) {
                pointer_answer = dup_n(
                    "No information available.",
                    strlen("No information available."));
                if (pointer_answer == NULL) {
                    free(episodic_context);
                    free(prompt);
                    send_error(c, 500, "server_error",
                               "failed to build tombstone answer");
                    return;
                }
                pointer_used = 2;
            }
        }
        if (pointer_used == 0 &&
            !json_get_bool(request, "stream", 0) &&
            json_get_bool(request, "memory_copy", 0) &&
            state->memory_pointer != NULL &&
            extract_episodic_answer(
                state, episodic_session, user_content,
                &pointer_answer, &pointer_confidence,
                &pointer_record_index) > 0)
            pointer_used = 1;
    }
    if (json_get_bool(request, "stream", 0)) {
        handle_streaming_completion(c, state, prompt, max_tokens,
                                    session_id, generation_reset, 1);
        if (state->cfg.episodic_memory &&
            memory_action != MEMORY_ACTION_IGNORE) {
            cached_session_t *session = find_session(state, session_id);
            if (session != NULL)
                (void)store_episodic_record(
                    state, session, user_content, memory_action);
        }
        free(episodic_context);
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
            free(episodic_context);
            free(prompt);
            send_error(c, 500, "server_error", error);
            return;
        }
    }
    if (state->cfg.episodic_memory &&
        memory_action != MEMORY_ACTION_IGNORE) {
        cached_session_t *session = find_session(state, session_id);
        int store_status = session == NULL ? -12 :
            store_episodic_record(
                state, session, user_content, memory_action);
        if (store_status != 0) {
            generation_result_free(&result);
            free(episodic_context);
            free(prompt);
            snprintf(
                error, sizeof(error),
                "failed to store episodic memory record "
                "(action=%s, status=%d, key_dim=%d, count=%zu)",
                memory_action_name(memory_action), store_status,
                session == NULL ? -1 : session->episodic.key_dim,
                session == NULL ? 0 : session->episodic.count);
            send_error(c, 500, "server_error", error);
            return;
        }
    }
    free(episodic_context);
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
                pointer_used == 2 ? "deletion_tombstone" :
                                    "extractive_pointer");
            json_add_number(copy, "confidence", pointer_confidence);
            json_add_number(
                copy, "record_index", (double)pointer_record_index);
            cJSON_AddItemToObject(root, "memory_copy", copy);
        }
    }
    if (result.session_id != NULL) {
        cJSON_AddItemToObject(root, "bitnet_session", build_session_json(&result));
    }
    send_json(c, 200, root);
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
        handle_streaming_completion(c, state, prompt, max_tokens,
                                    session_id, reset_existing_session, 0);
        free(prompt);
        return;
    }

    memset(&result, 0, sizeof(result));
    if (generate_text(state, prompt, max_tokens, session_id, reset_existing_session,
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
    if (action == 2) {
        handle_memory_state(c, state, request, 0);
    } else if (action == 3) {
        handle_memory_state(c, state, request, 1);
    } else if (action == 4) {
        handle_memory_search(c, state, request);
    } else if (action == 5) {
        handle_memory_extract(c, state, request);
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
                   mg_match(hm->uri, mg_str("/v1/memory/search"), NULL)) {
            handle_post_json(c, hm, state, 4);
        } else if (is_method(hm, "POST") &&
                   mg_match(hm->uri, mg_str("/v1/memory/extract"), NULL)) {
            handle_post_json(c, hm, state, 5);
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
            "       [--memory-model PATH] [--memory-state-dir DIR]\n"
            "       [--episodic-memory] [--episodic-top-k N]\n"
            "       [--memory-controller PATH]"
            " [--memory-pointer PATH]"
            " [--episodic-lexical-weight WEIGHT]"
            " [--memory-address-threshold SCORE]\n",
            argv0);
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
    state.cfg.episodic_top_k = 3;
    state.cfg.episodic_lexical_weight = 0.75f;
    state.cfg.memory_address_threshold = 0.80f;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    model_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            state.cfg.host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            state.cfg.port = argv[++i];
        } else if (strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            state.cfg.model_id = argv[++i];
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            state.cfg.max_context_tokens = parse_int_arg(argv[++i], DEFAULT_CONTEXT_TOKENS,
                                                         MIN_CONTEXT_TOKENS, 32768);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            state.cfg.max_request_tokens = parse_int_arg(argv[++i], MAX_REQUEST_TOKENS,
                                                         1, MAX_REQUEST_TOKENS);
        } else if (strcmp(argv[i], "--default-max-tokens") == 0 && i + 1 < argc) {
            state.cfg.default_max_tokens = parse_int_arg(argv[++i], DEFAULT_MAX_TOKENS,
                                                         1, MAX_REQUEST_TOKENS);
        } else if (strcmp(argv[i], "--repeat-last-n") == 0 && i + 1 < argc) {
            state.cfg.repeat_last_n = parse_int_arg(argv[++i], DEFAULT_REPEAT_LAST_N,
                                                    0, 4096);
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            state.cfg.repeat_penalty = parse_float_arg(argv[++i], DEFAULT_REPEAT_PENALTY,
                                                       1.0f, 10.0f);
        } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
            state.cfg.lora_path = argv[++i];
        } else if (strcmp(argv[i], "--lora-scale") == 0 && i + 1 < argc) {
            state.cfg.lora_scale = parse_float_arg(argv[++i], 1.0f, 0.0f, 100.0f);
        } else if (strcmp(argv[i], "--memory-model") == 0 && i + 1 < argc) {
            state.cfg.memory_model_path = argv[++i];
        } else if (strcmp(argv[i], "--memory-state-dir") == 0 && i + 1 < argc) {
            state.cfg.memory_state_dir = argv[++i];
        } else if (strcmp(argv[i], "--episodic-memory") == 0) {
            state.cfg.episodic_memory = 1;
        } else if (strcmp(argv[i], "--episodic-top-k") == 0 &&
                   i + 1 < argc) {
            state.cfg.episodic_top_k = parse_int_arg(
                argv[++i], 3, 1, 32);
        } else if (strcmp(argv[i], "--memory-controller") == 0 &&
                   i + 1 < argc) {
            state.cfg.memory_controller_path = argv[++i];
        } else if (strcmp(argv[i], "--memory-pointer") == 0 &&
                   i + 1 < argc) {
            state.cfg.memory_pointer_path = argv[++i];
        } else if (strcmp(argv[i], "--episodic-lexical-weight") == 0 &&
                   i + 1 < argc) {
            state.cfg.episodic_lexical_weight = parse_float_arg(
                argv[++i], 0.75f, 0.0f, 1.0f);
        } else if (strcmp(argv[i], "--memory-address-threshold") == 0 &&
                   i + 1 < argc) {
            state.cfg.memory_address_threshold = parse_float_arg(
                argv[++i], 0.80f, -1.0f, 1.0f);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (state.cfg.default_max_tokens > state.cfg.max_request_tokens) {
        state.cfg.default_max_tokens = state.cfg.max_request_tokens;
    }
    if (state.cfg.max_context_tokens < state.cfg.max_request_tokens + MIN_CONTEXT_TOKENS) {
        state.cfg.max_context_tokens = state.cfg.max_request_tokens + MIN_CONTEXT_TOKENS;
    }
    if (state.cfg.episodic_memory &&
        state.cfg.memory_state_dir == NULL) {
        fprintf(stderr,
                "--episodic-memory requires --memory-state-dir\n");
        return 1;
    }
    if (state.cfg.episodic_memory &&
        state.cfg.memory_model_path == NULL &&
        state.cfg.memory_controller_path == NULL) {
        fprintf(stderr,
                "--episodic-memory without --memory-model requires "
                "--memory-controller\n");
        return 1;
    }
    if (state.cfg.memory_controller_path != NULL &&
        !state.cfg.episodic_memory) {
        fprintf(stderr,
                "--memory-controller requires --episodic-memory\n");
        return 1;
    }
    if (state.cfg.memory_pointer_path != NULL &&
        !state.cfg.episodic_memory) {
        fprintf(stderr,
                "--memory-pointer requires --episodic-memory\n");
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", model_path);
    state.model = bitnet_load_model(model_path);
    if (state.model == NULL) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    if (state.cfg.lora_path != NULL) {
        if (bitnet_load_lora(state.model, state.cfg.lora_path, state.cfg.lora_scale) != 0) {
            fprintf(stderr, "Failed to load LoRA: %s\n", state.cfg.lora_path);
            bitnet_free_model(state.model);
            return 1;
        }
        fprintf(stderr, "Loaded LoRA: %s (tensors=%d, scale=%.4f)\n",
                state.cfg.lora_path,
                bitnet_lora_count(state.model),
                state.cfg.lora_scale);
    }
    if (state.cfg.memory_model_path != NULL) {
        if (bitnet_load_memory_model(state.model, state.cfg.memory_model_path) != 0) {
            fprintf(stderr, "Failed to load memory model: %s\n",
                    state.cfg.memory_model_path);
            bitnet_free_model(state.model);
            return 1;
        }
        fprintf(stderr, "Memory model loaded: %s\n",
                state.cfg.memory_model_path);
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
                "Memory controller loaded: %s "
                "(rank=%d pooling=%d lexical_weight=%.2f)\n",
                state.cfg.memory_controller_path,
                state.memory_controller->rank,
                state.memory_controller->pooling,
                state.cfg.episodic_lexical_weight);
    }
    if (state.cfg.memory_pointer_path != NULL) {
        char pointer_error[256];
        state.memory_pointer = metis_memory_pointer_load(
            state.cfg.memory_pointer_path, model_path,
            bitnet_embedding_length(state.model),
            pointer_error, sizeof pointer_error);
        if (state.memory_pointer == NULL) {
            fprintf(stderr, "Failed to load memory pointer: %s\n",
                    pointer_error);
            metis_memory_controller_free(state.memory_controller);
            bitnet_free_model(state.model);
            return 1;
        }
        fprintf(stderr,
                "Memory pointer loaded: %s "
                "(max_span=%d threshold=%.4f)\n",
                state.cfg.memory_pointer_path,
                state.memory_pointer->max_span,
                state.memory_pointer->threshold);
    }
    if (state.cfg.episodic_memory) {
        fprintf(stderr,
                "Episodic memory enabled: top_k=%d, KV reuse disabled "
                "per request\n",
                state.cfg.episodic_top_k);
    }

    snprintf(listen_url, sizeof(listen_url), "http://%s:%s", state.cfg.host, state.cfg.port);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    mg_log_set(MG_LL_ERROR);
    mg_mgr_init(&mgr);
    if (mg_http_listen(&mgr, listen_url, http_handler, &state) == NULL) {
        fprintf(stderr, "Failed to listen on %s\n", listen_url);
        mg_mgr_free(&mgr);
        metis_memory_pointer_free(state.memory_pointer);
        metis_memory_controller_free(state.memory_controller);
        bitnet_free_model(state.model);
        return 1;
    }
    fprintf(stderr, "OpenAI-compatible server listening on %s\n", listen_url);
    fprintf(stderr, "Model id: %s\n", state.cfg.model_id);
    fprintf(stderr, "Default max_tokens: %d, request max_tokens cap: %d, ctx: %d\n",
            state.cfg.default_max_tokens,
            state.cfg.max_request_tokens,
            state.cfg.max_context_tokens);
    while (!g_stop) {
        mg_mgr_poll(&mgr, 100);
    }
    mg_mgr_free(&mgr);
    free_sessions(&state);
    metis_memory_pointer_free(state.memory_pointer);
    metis_memory_controller_free(state.memory_controller);
    bitnet_free_model(state.model);
    return 0;
}
