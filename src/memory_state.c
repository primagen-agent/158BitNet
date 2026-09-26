/* memory_state.c — unbounded session episode store + .bnstate persistence.
 * Contract defined by tests/test_memory_state.c.
 */
#include "memory_state.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ms_session_t *sessions = NULL;
static int n_sessions = 0;
static int session_capacity = 0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/* Sanity bounds against corrupt files, NOT storage caps. */
#define MS_MAX_SESSIONS_HARD (1 << 20)
#define MS_MAX_EPISODE_HARD  (100u * 1024 * 1024)

void ms_reset_all(void) {
    pthread_mutex_lock(&lock);
    for (int i = 0; i < n_sessions; i++) {
        ms_session_t *s = &sessions[i];
        for (int j = 0; j < s->n_episodes; j++) {
            free(s->episodes[j]);
            free(s->ep_means ? s->ep_means[j] : NULL);
            free(s->ep_rows ? s->ep_rows[j] : NULL);
        }
        free(s->episodes);
        free(s->ep_means);
        free(s->ep_rows);
        free(s->ep_row_counts);
    }
    free(sessions);
    sessions = NULL;
    n_sessions = 0;
    session_capacity = 0;
    pthread_mutex_unlock(&lock);
}

ms_session_t *ms_find_session(const char *id) {
    pthread_mutex_lock(&lock);
    ms_session_t *found = NULL;
    for (int i = 0; i < n_sessions; i++)
        if (strcmp(sessions[i].id, id) == 0) { found = &sessions[i]; break; }
    pthread_mutex_unlock(&lock);
    return found;
}

static void ensure_episode_slot(ms_session_t *s) {
    if (s->n_episodes < s->ep_capacity) return;
    int new_cap = s->ep_capacity ? s->ep_capacity * 2 : 8;
    char **eps = realloc(s->episodes, (size_t)new_cap * sizeof(char *));
    if (!eps) return;
    s->episodes = eps;
    float **ms = realloc(s->ep_means, (size_t)new_cap * sizeof(float *));
    if (!ms) return;
    s->ep_means = ms;
    float **rs = realloc(s->ep_rows, (size_t)new_cap * sizeof(float *));
    if (!rs) return;
    s->ep_rows = rs;
    int *rc = realloc(s->ep_row_counts, (size_t)new_cap * sizeof(int));
    if (!rc) return;
    s->ep_row_counts = rc;
    for (int i = s->ep_capacity; i < new_cap; i++) {
        s->episodes[i] = NULL;
        s->ep_means[i] = NULL;
        s->ep_rows[i] = NULL;
        s->ep_row_counts[i] = 0;
    }
    s->ep_capacity = new_cap;
}

ms_session_t *ms_get_session(const char *id) {
    pthread_mutex_lock(&lock);
    ms_session_t *found = NULL;
    for (int i = 0; i < n_sessions; i++)
        if (strcmp(sessions[i].id, id) == 0) { found = &sessions[i]; break; }
    if (!found) {
        if (n_sessions >= session_capacity) {
            int new_cap = session_capacity ? session_capacity * 2 : 16;
            ms_session_t *grown = realloc(sessions,
                (size_t)new_cap * sizeof(ms_session_t));
            if (grown) {
                sessions = grown;
                session_capacity = new_cap;
            }
        }
        if (n_sessions < session_capacity) {
            found = &sessions[n_sessions++];
            memset(found, 0, sizeof *found);
            strncpy(found->id, id, sizeof found->id - 1);
        }
    }
    pthread_mutex_unlock(&lock);
    return found;
}

void ms_add_episode(ms_session_t *s, const char *text) {
    if (!s || !text) return;
    pthread_mutex_lock(&lock);
    ensure_episode_slot(s);
    if (s->n_episodes >= s->ep_capacity) {   /* OOM: drop, don't crash */
        pthread_mutex_unlock(&lock);
        return;
    }
    size_t len = strlen(text);
    char *copy = malloc(len + 1);
    if (!copy) { pthread_mutex_unlock(&lock); return; }
    memcpy(copy, text, len + 1);
    s->episodes[s->n_episodes] = copy;
    s->ep_means[s->n_episodes] = NULL;
    s->n_episodes++;
    pthread_mutex_unlock(&lock);
}

void ms_set_rows(ms_session_t *s, int index, const float *rows, int n_rows) {
    if (!s || index < 0 || index >= s->n_episodes || !rows) return;
    if (n_rows <= 0) n_rows = 0;
    if (n_rows > MS_MAX_ROWS) n_rows = MS_MAX_ROWS;
    free(s->ep_rows[index]);
    s->ep_rows[index] = NULL;
    s->ep_row_counts[index] = 0;
    if (n_rows == 0) return;
    float *buf = malloc((size_t)n_rows * MS_FEAT_DIM * sizeof(float));
    if (!buf) return;
    memcpy(buf, rows, (size_t)n_rows * MS_FEAT_DIM * sizeof(float));
    s->ep_rows[index] = buf;
    s->ep_row_counts[index] = n_rows;
}

void ms_set_mean(ms_session_t *s, int index, const float *mean) {
    if (!s || index < 0 || index >= s->n_episodes || !mean) return;
    if (!s->ep_means[index]) {
        float *row = malloc(MS_FEAT_DIM * sizeof(float));
        if (!row) return;
        s->ep_means[index] = row;
    }
    memcpy(s->ep_means[index], mean, MS_FEAT_DIM * sizeof(float));
    s->means_valid = 1;
}

int ms_session_count(void) {
    return n_sessions;
}

int ms_save(const char *path) {
    if (!path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { perror("ms_save"); return -1; }
    pthread_mutex_lock(&lock);
    uint32_t hdr[4] = {0x54534E42u, 3, (uint32_t)n_sessions, 0};
    int ok = fwrite(hdr, 4, 4, f) == 4;
    for (int i = 0; ok && i < n_sessions; i++) {
        ms_session_t *s = &sessions[i];
        uint32_t id_len = (uint32_t)strlen(s->id);
        ok = fwrite(&id_len, 4, 1, f) == 1 && fwrite(s->id, 1, id_len, f) == id_len;
        uint32_t n_ep = (uint32_t)s->n_episodes;
        ok = ok && fwrite(&n_ep, 4, 1, f) == 1;
        for (uint32_t j = 0; ok && j < n_ep; j++) {
            uint32_t ep_len = (uint32_t)strlen(s->episodes[j]);
            ok = fwrite(&ep_len, 4, 1, f) == 1 &&
                 fwrite(s->episodes[j], 1, ep_len, f) == ep_len;
        }
        uint32_t means_flag = (s->means_valid && s->ep_means) ? 1u : 0u;
        ok = ok && fwrite(&means_flag, 4, 1, f) == 1;
        if (ok && means_flag)
            for (uint32_t j = 0; ok && j < n_ep; j++)
                ok = fwrite(s->ep_means[j], sizeof(float), MS_FEAT_DIM, f)
                     == MS_FEAT_DIM;
        uint32_t rows_flag = (s->ep_rows && s->ep_row_counts) ? 1u : 0u;
        ok = ok && fwrite(&rows_flag, 4, 1, f) == 1;
        if (ok && rows_flag)
            for (uint32_t j = 0; ok && j < n_ep; j++) {
                uint32_t n_rows = (uint32_t)s->ep_row_counts[j];
                ok = fwrite(&n_rows, 4, 1, f) == 1;
                if (ok && n_rows > 0)
                    ok = fwrite(s->ep_rows[j], sizeof(float),
                                (size_t)n_rows * MS_FEAT_DIM, f)
                         == (size_t)n_rows * MS_FEAT_DIM;
            }
    }
    pthread_mutex_unlock(&lock);
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

int ms_load(const char *path) {
    if (!path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != 0x54534E42u) {
        fclose(f); return -1;
    }
    if (hdr[1] < 1 || hdr[1] > 3) {   /* v1 text, v2 means, v3 rows */
        fclose(f); return -1;
    }
    int version = (int)hdr[1];
    uint32_t count = hdr[2];
    if (count > MS_MAX_SESSIONS_HARD) { fclose(f); return -1; }

    ms_reset_all();
    pthread_mutex_lock(&lock);
    int ok = 1;
    for (uint32_t i = 0; ok && i < count; i++) {
        uint32_t id_len = 0;
        ok = fread(&id_len, 4, 1, f) == 1 && id_len < sizeof sessions[0].id;
        char id[64] = {0};
        if (ok) ok = fread(id, 1, id_len, f) == id_len;
        uint32_t n_ep = 0;
        if (ok) ok = fread(&n_ep, 4, 1, f) == 1 && n_ep <= MS_MAX_EPISODE_HARD;

        ms_session_t *s = NULL;
        if (ok) {   /* grow table under lock (no fixed session bound) */
            if (n_sessions >= session_capacity) {
                int new_cap = session_capacity ? session_capacity * 2 : 16;
                ms_session_t *grown = realloc(sessions,
                    (size_t)new_cap * sizeof(ms_session_t));
                if (grown) { sessions = grown; session_capacity = new_cap; }
            }
            if (n_sessions < session_capacity) {
                s = &sessions[n_sessions++];
                memset(s, 0, sizeof *s);
                memcpy(s->id, id, id_len);
            } else ok = 0;
        }
        for (uint32_t j = 0; ok && s && j < n_ep; j++) {
            uint32_t ep_len = 0;
            ok = fread(&ep_len, 4, 1, f) == 1 && ep_len <= MS_MAX_EPISODE_HARD;
            if (ok) {
                ensure_episode_slot(s);
                if (s->n_episodes >= s->ep_capacity) { ok = 0; break; }
                char *buf = malloc(ep_len + 1);
                if (!buf) { ok = 0; break; }
                if (fread(buf, 1, ep_len, f) != ep_len) { free(buf); ok = 0; break; }
                buf[ep_len] = 0;
                s->episodes[s->n_episodes++] = buf;
            }
        }
        if (ok && version >= 2) {
            uint32_t means_flag = 0;
            ok = fread(&means_flag, 4, 1, f) == 1;
            if (ok && means_flag && s) {
                for (int j = 0; ok && j < s->n_episodes; j++) {
                    if (!s->ep_means[j]) {
                        s->ep_means[j] = malloc(MS_FEAT_DIM * sizeof(float));
                        if (!s->ep_means[j]) { ok = 0; break; }
                    }
                    ok = fread(s->ep_means[j], sizeof(float), MS_FEAT_DIM, f)
                         == MS_FEAT_DIM;
                }
                if (ok) s->means_valid = 1;
            }
        }
        if (ok && version >= 3) {
            uint32_t rows_flag = 0;
            ok = fread(&rows_flag, 4, 1, f) == 1;
            if (ok && rows_flag && s) {
                for (int j = 0; ok && j < s->n_episodes; j++) {
                    uint32_t n_rows = 0;
                    ok = fread(&n_rows, 4, 1, f) == 1 && n_rows <= MS_MAX_ROWS;
                    if (ok && n_rows > 0) {
                        float *buf = malloc((size_t)n_rows * MS_FEAT_DIM * sizeof(float));
                        if (!buf) { ok = 0; break; }
                        ok = fread(buf, sizeof(float),
                                   (size_t)n_rows * MS_FEAT_DIM, f)
                             == (size_t)n_rows * MS_FEAT_DIM;
                        if (ok) {
                            s->ep_rows[j] = buf;
                            s->ep_row_counts[j] = (int)n_rows;
                        } else free(buf);
                    }
                }
            }
        }
    }
    pthread_mutex_unlock(&lock);
    fclose(f);
    if (!ok) {
        ms_reset_all();
        return -1;
    }
    return 0;
}
