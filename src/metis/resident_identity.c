#include "resident_identity.h"
#include "sha256.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = RESIDENT_WIDTH, H = RESIDENT_HIDDEN, S = RESIDENT_STRIDE };
static const size_t elements[25] = {
    1, 1,      4 * D, D *H,      D, D * 4 * D, D, D,         1, D * (3 * D + 1),
    D, 5 * D,  5,     D * 4 * D, D, 2 * D,     1, D * 2 * D, D, D,
    1, 64 * H, 64,    64,        1};
static uint32_t crc32(const void *data, size_t size) {
    const unsigned char *b = data;
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < size; i++) {
        c ^= b[i];
        for (int j = 0; j < 8; j++)
            c = (c >> 1) ^ (0xedb88320u & (uint32_t)-(int32_t)(c & 1));
    }
    return ~c;
}
static int u32read(FILE *f, uint32_t *n) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4)
        return -1;
    *n = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}
static int u32write(FILE *f, uint32_t n) {
    unsigned char b[4] = {(unsigned char)n, (unsigned char)(n >> 8), (unsigned char)(n >> 16),
                          (unsigned char)(n >> 24)};
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}
static int floats_read(FILE *f, float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        if (u32read(f, &u))
            return -1;
        memcpy(v + i, &u, 4);
        if (!isfinite(v[i]))
            return -1;
    }
    return 0;
}
static int floats_write(FILE *f, const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        if (!isfinite(v[i]))
            return -1;
        memcpy(&u, v + i, 4);
        if (u32write(f, u))
            return -1;
    }
    return 0;
}
void resident_model_free(resident_model_t *m) {
    if (m) {
        for (int i = 0; i < 25; i++)
            free(m->tensor[i]);
        free(m);
    }
}
resident_model_t *resident_model_load(const char *path, const char *backbone) {
    if (!path || !backbone)
        return NULL;
    FILE *f = fopen(path, "rb");
    resident_model_t *m = NULL;
    unsigned char magic[8], sha[32], actual[32], checkpoint[32];
    uint32_t v, h, d, n, sizes[25], crcs[25];
    if (!f)
        return NULL;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "BNRESID1", 8) || u32read(f, &v) || v != 1 ||
        u32read(f, &h) || h != H || u32read(f, &d) || d != D || u32read(f, &n) || n != 25 ||
        fread(sha, 1, 32, f) != 32 || fread(checkpoint, 1, 32, f) != 32 ||
        bitnet_sha256_file(backbone, actual) || memcmp(sha, actual, 32))
        goto fail;
    for (int i = 0; i < 25; i++)
        if (u32read(f, &sizes[i]) || sizes[i] != 4 * elements[i] || u32read(f, &crcs[i]))
            goto fail;
    m = calloc(1, sizeof *m);
    if (!m)
        goto fail;
    for (int i = 0; i < 25; i++) {
        unsigned char *bytes = malloc(sizes[i]);
        if (!bytes)
            goto fail;
        if (fread(bytes, 1, sizes[i], f) != sizes[i] || crc32(bytes, sizes[i]) != crcs[i]) {
            free(bytes);
            goto fail;
        }
        m->tensor[i] = malloc(sizes[i]);
        if (!m->tensor[i]) {
            free(bytes);
            goto fail;
        }
        for (size_t j = 0; j < elements[i]; j++) {
            uint32_t u = (uint32_t)bytes[4 * j] | ((uint32_t)bytes[4 * j + 1] << 8) |
                         ((uint32_t)bytes[4 * j + 2] << 16) | ((uint32_t)bytes[4 * j + 3] << 24);
            memcpy(m->tensor[i] + j, &u, 4);
            if (!isfinite(m->tensor[i][j])) {
                free(bytes);
                goto fail;
            }
        }
        free(bytes);
    }
    if (fgetc(f) != EOF || ferror(f) || bitnet_sha256_file(path, m->sha256))
        goto fail;
    fclose(f);
    return m;
fail:
    fclose(f);
    resident_model_free(m);
    return NULL;
}
static float dot(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}
static void normalize(float *v, int n) {
    float a = sqrtf(dot(v, v, n));
    a = fmaxf(a, 1e-12f);
    for (int i = 0; i < n; i++)
        v[i] /= a;
}
static float softplus(float x) {
    return fmaxf(x, 0) + log1pf(expf(-fabsf(x)));
}
static float logsig(float x) {
    return -softplus(-x);
}
static float logadd(float a, float b) {
    float m = fmaxf(a, b);
    return m + logf(expf(a - m) + expf(b - m));
}
static float logit(float p) {
    p = fminf(p, -1e-6f);
    return p - logf(-expm1f(p));
}
static void linear(float *out, const float *x, const float *w, const float *b, int rows, int cols,
                   int gelu) {
    for (int i = 0; i < rows; i++) {
        float v = dot(x, w + i * cols, cols) + (b ? b[i] : 0);
        out[i] = gelu ? v * .5f * (1 + erff(v * .7071067811865475f)) : v;
    }
}
static void softmax(float *v, size_t n) {
    float max = v[0], sum = 0;
    for (size_t i = 1; i < n; i++)
        max = fmaxf(max, v[i]);
    for (size_t i = 0; i < n; i++) {
        v[i] = expf(v[i] - max);
        sum += v[i];
    }
    for (size_t i = 0; i < n; i++)
        v[i] /= sum;
}
static void project(const resident_model_t *m, const float *x, float *out) {
    linear(out, x, m->tensor[3], m->tensor[4], D, H, 0);
    for (int i = 0; i < D; i++)
        out[i] /= 1 + expf(-out[i]);
    normalize(out, D);
}
static float role(const resident_model_t *m, const float *x) {
    float v[64];
    linear(v, x, m->tensor[21], m->tensor[22], 64, H, 1);
    return dot(v, m->tensor[23], 64) + m->tensor[24][0];
}
int resident_write(const resident_model_t *m, const float *x, size_t n, resident_slot_t *out) {
    if (!m || !x || !out || n < 2 || n > RESIDENT_MAX_TOKENS)
        return -1;
    float *v = calloc(n * S, sizeof(float));
    if (!v)
        return -1;
    for (size_t t = 1; t < n; t++) {
        project(m, x + t * 2 * H, v + t * S);
        memcpy(v + t * S + D, x + t * 2 * H + H, H * sizeof(float));
        v[t * S + S - 2] = role(m, x + t * 2 * H);
        v[t * S + S - 1] = 1;
    }
    for (size_t i = 0; i < n * S; i++)
        if (!isfinite(v[i])) {
            free(v);
            return -1;
        }
    out->tokens = n;
    out->data = v;
    return 0;
}
static void attend(const float *query, const resident_slot_t *slot, float *out) {
    float weights[RESIDENT_MAX_TOKENS];
    size_t n = slot->tokens;
    for (size_t t = 1; t < n; t++)
        weights[t - 1] = 8 * dot(query, slot->data + t * S, D);
    softmax(weights, n - 1);
    memset(out, 0, D * sizeof(float));
    for (size_t t = 1; t < n; t++)
        for (int j = 0; j < D; j++)
            out[j] += weights[t - 1] * slot->data[t * S + j];
}
static void pair(float *out, const float *a, const float *b, const float *w, const float *bias) {
    float x[4 * D];
    for (int j = 0; j < D; j++) {
        x[j] = a[j];
        x[D + j] = b[j];
        x[2 * D + j] = a[j] * b[j];
        x[3 * D + j] = fabsf(a[j] - b[j]);
    }
    linear(out, x, w, bias, D, 4 * D, 1);
}
static float link_direction(const resident_model_t *m, const resident_slot_t *a,
                            const resident_slot_t *b) {
    float aggregate[2 * D] = {0}, attended[D], v[D];
    for (int j = 0; j < D; j++)
        aggregate[D + j] = -1e4f;
    for (size_t t = 1; t < a->tokens; t++) {
        attend(a->data + t * S, b, attended);
        pair(v, a->data + t * S, attended, m->tensor[13], m->tensor[14]);
        for (int j = 0; j < D; j++) {
            aggregate[j] += v[j] / (float)(a->tokens - 1);
            aggregate[D + j] = fmaxf(aggregate[D + j], v[j]);
        }
    }
    return dot(aggregate, m->tensor[15], 2 * D) + m->tensor[16][0];
}
int resident_read(const resident_model_t *m, const resident_state_t *state, const float *query,
                  size_t nt, float *scores, float counts[5]) {
    if (!m || !state || !query || !scores || !counts || nt < 2 || nt > RESIDENT_MAX_TOKENS ||
        state->count > RESIDENT_MAX_EVENTS)
        return -1;
    size_t n = state->count;
    if (!n) {
        for (int k = 0; k < 5; k++)
            counts[k] = k ? -1e4f : 0;
        return 0;
    }
    for (size_t i = 0; i < n; i++)
        if (!state->slots[i].data || state->slots[i].tokens < 2 ||
            state->slots[i].tokens > RESIDENT_MAX_TOKENS)
            return -1;
    float qt[RESIDENT_MAX_TOKENS][D], qr[RESIDENT_MAX_TOKENS];
    float pool[4][RESIDENT_MAX_TOKENS], q[4][D] = {{0}}, scope_input[2 * D] = {0}, scope_hidden[D];
    float identities[RESIDENT_MAX_EVENTS][H], address[RESIDENT_MAX_EVENTS], cnt[3 * D + 1] = {0};
    for (int j = 0; j < D; j++) {
        scope_input[D + j] = -1e4f;
        cnt[D + j] = -1e4f;
    }
    for (size_t t = 1; t < nt; t++) {
        project(m, query + t * 2 * H, qt[t]);
        qr[t] = role(m, query + t * 2 * H);
        for (int j = 0; j < D; j++) {
            scope_input[j] += qt[t][j] / (float)(nt - 1);
            scope_input[D + j] = fmaxf(scope_input[D + j], qt[t][j]);
        }
    }
    for (int h = 0; h < 4; h++) {
        for (size_t t = 1; t < nt; t++)
            pool[h][t - 1] = dot(qt[t], m->tensor[2] + h * D, D);
        softmax(pool[h], nt - 1);
        for (size_t t = 1; t < nt; t++)
            for (int j = 0; j < D; j++)
                q[h][j] += pool[h][t - 1] * qt[t][j];
        normalize(q[h], D);
        for (int j = 0; j < D; j++)
            cnt[2 * D + j] += q[h][j] / 4;
    }
    linear(scope_hidden, scope_input, m->tensor[17], m->tensor[18], D, 2 * D, 1);
    float history = dot(scope_hidden, m->tensor[19], D) + m->tensor[20][0];
    float scale = softplus(m->tensor[0][0]), bias = m->tensor[1][0];
    for (size_t i = 0; i < n; i++) {
        const resident_slot_t *slot = &state->slots[i];
        float weights[RESIDENT_MAX_TOKENS];
        for (size_t t = 1; t < slot->tokens; t++)
            weights[t - 1] = slot->data[t * S + S - 2];
        softmax(weights, slot->tokens - 1);
        memset(identities[i], 0, H * sizeof(float));
        for (size_t t = 1; t < slot->tokens; t++)
            for (int j = 0; j < H; j++)
                identities[i][j] += weights[t - 1] * slot->data[t * S + D + j];
        normalize(identities[i], H);
        float features[4][D] = {{0}}, attended[D], v[D], head[4], maximum[D];
        for (size_t t = 1; t < nt; t++) {
            attend(qt[t], slot, attended);
            pair(v, qt[t], attended, m->tensor[5], m->tensor[6]);
            for (int h = 0; h < 4; h++)
                for (int j = 0; j < D; j++)
                    features[h][j] += pool[h][t - 1] * v[j];
        }
        for (int h = 0; h < 4; h++)
            head[h] = dot(features[h], m->tensor[7], D) + m->tensor[8][0];
        float a = logadd(logadd(head[0], head[1]), logadd(head[2], head[3])) - logf(4);
        for (int j = 0; j < D; j++) {
            maximum[j] =
                fmaxf(fmaxf(features[0][j], features[1][j]), fmaxf(features[2][j], features[3][j]));
            cnt[j] += maximum[j] / (float)n;
            cnt[D + j] = fmaxf(cnt[D + j], maximum[j]);
        }
        for (size_t t = 1; t < nt; t++)
            weights[t - 1] = 8 * dot(identities[i], query + t * 2 * H + H, H) + logsig(qr[t]);
        softmax(weights, nt - 1);
        float lexical[H] = {0};
        for (size_t t = 1; t < nt; t++)
            for (int j = 0; j < H; j++)
                lexical[j] += weights[t - 1] * query[t * 2 * H + H + j];
        normalize(lexical, H);
        float entity = scale * dot(identities[i], lexical, H) + bias;
        address[i] = logit(logsig(a) + logsig(entity));
    }
    cnt[3 * D] = (float)n / 32;
    float count_hidden[D];
    linear(count_hidden, cnt, m->tensor[9], m->tensor[10], D, 3 * D + 1, 1);
    linear(counts, count_hidden, m->tensor[11], m->tensor[12], 5, D, 0);
    for (size_t i = 0; i < n; i++) {
        float current = 0;
        for (size_t j = i + 1; j < n; j++) {
            float old = .5f * (link_direction(m, &state->slots[i], &state->slots[j]) +
                               link_direction(m, &state->slots[j], &state->slots[i]));
            float ent = scale * dot(identities[i], identities[j], H) + bias;
            float link = logit(logsig(old) + logsig(ent));
            current += logsig(-link);
        }
        float gate = logadd(logsig(history), logsig(-history) + current);
        scores[i] = logit(logsig(address[i]) + gate);
        if (!isfinite(scores[i]))
            return -1;
    }
    for (int k = 0; k < 5; k++)
        if (!isfinite(counts[k]))
            return -1;
    return 0;
}
int resident_select(const float *scores, const float counts[5], size_t n, size_t out[4]) {
    int k = 0;
    for (int i = 1; i < 5; i++)
        if (counts[i] > counts[k])
            k = i;
    if ((size_t)k > n)
        k = (int)n;
    int used[RESIDENT_MAX_EVENTS] = {0};
    if (n > RESIDENT_MAX_EVENTS)
        return -1;
    for (int j = 0; j < k; j++) {
        size_t best = n;
        for (size_t i = 0; i < n; i++)
            if (!used[i] && (best == n || scores[i] > scores[best]))
                best = i;
        out[j] = best;
        used[best] = 1;
    }
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++)
            if (out[j] < out[i]) {
                size_t t = out[i];
                out[i] = out[j];
                out[j] = t;
            }
    return k;
}
void resident_state_clear(resident_state_t *s) {
    if (s) {
        for (size_t i = 0; i < s->count; i++)
            free(s->slots[i].data);
        memset(s, 0, sizeof *s);
    }
}
float *resident_encode(bitnet_model_t *model, const char *text, size_t *count) {
    int ids[RESIDENT_MAX_TOKENS + 1];
    float *out = NULL, *lex = NULL;
    bitnet_context_t *ctx = NULL;
    if (!model || !text || !count || bitnet_embedding_length(model) != H)
        return NULL;
    int n = bitnet_tokenize_ex(model, text, ids, RESIDENT_MAX_TOKENS + 1, 1);
    if (n < 2 || n > RESIDENT_MAX_TOKENS)
        return NULL;
    ctx = bitnet_create_context(model, RESIDENT_MAX_TOKENS);
    if (!ctx)
        goto done;
    if (bitnet_eval_hidden(ctx, ids, n) || bitnet_last_eval_hidden_count(ctx) != n)
        goto done;
    const float *h = bitnet_get_last_eval_hidden(ctx);
    if (!h)
        goto done;
    lex = malloc((size_t)n * H * sizeof(float));
    out = malloc((size_t)n * 2 * H * sizeof(float));
    if (!lex || !out || bitnet_token_embedding_lookup(model, ids, n, lex, (size_t)n * H)) {
        free(out);
        out = NULL;
        goto done;
    }
    for (int t = 0; t < n; t++) {
        memcpy(out + t * 2 * H, h + t * H, H * sizeof(float));
        memcpy(out + t * 2 * H + H, lex + t * H, H * sizeof(float));
        normalize(out + t * 2 * H, H);
        normalize(out + t * 2 * H + H, H);
    }
    *count = (size_t)n;
done:
    free(lex);
    bitnet_free_context(ctx);
    return out;
}
int resident_state_save(const resident_state_t *s, const resident_model_t *m, const char *path) {
    if (!s || !m || !path || s->count > RESIDENT_MAX_EVENTS)
        return -1;
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    int rc = -1;
    if (fwrite("BNRSTAT1", 1, 8, f) != 8 || fwrite(m->sha256, 1, 32, f) != 32 ||
        u32write(f, (uint32_t)s->count))
        goto done;
    for (size_t i = 0; i < s->count; i++)
        if (s->slots[i].tokens < 2 || s->slots[i].tokens > RESIDENT_MAX_TOKENS ||
            !s->slots[i].data || u32write(f, (uint32_t)s->slots[i].tokens) ||
            floats_write(f, s->slots[i].data, s->slots[i].tokens * S))
            goto done;
    rc = 0;
done:
    if (fclose(f))
        rc = -1;
    return rc;
}
int resident_state_load(resident_state_t *s, const resident_model_t *m, const char *path) {
    if (!s || !m || !path)
        return -1;
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    resident_state_t temp = {0};
    unsigned char magic[8], sha[32];
    uint32_t n, t;
    int rc = -1;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "BNRSTAT1", 8) || fread(sha, 1, 32, f) != 32 ||
        memcmp(sha, m->sha256, 32) || u32read(f, &n) || n > RESIDENT_MAX_EVENTS)
        goto done;
    for (uint32_t i = 0; i < n; i++) {
        if (u32read(f, &t) || t < 2 || t > RESIDENT_MAX_TOKENS)
            goto done;
        temp.slots[i].tokens = t;
        temp.slots[i].data = malloc((size_t)t * S * sizeof(float));
        temp.count = i + 1;
        if (!temp.slots[i].data || floats_read(f, temp.slots[i].data, (size_t)t * S))
            goto done;
        for (uint32_t j = 0; j < t; j++)
            if (temp.slots[i].data[j * S + S - 1] != (j ? 1.f : 0.f))
                goto done;
    }
    if (fgetc(f) != EOF || ferror(f))
        goto done;
    resident_state_clear(s);
    *s = temp;
    memset(&temp, 0, sizeof temp);
    rc = 0;
done:
    resident_state_clear(&temp);
    fclose(f);
    return rc;
}
