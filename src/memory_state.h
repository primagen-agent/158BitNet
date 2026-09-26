/* memory_state.h — session episode store and .bnstate persistence.
 *
 * Storage is unbounded by design (a memory system must not cap content):
 * the session table grows dynamically and episodes are arbitrary-length
 * strings. Per-episode 2048-dim feature means are OPTIONAL ( callers that
 * have no ranking head skip them entirely; .bnstate stores them only when
 * computed).
 *
 * .bnstate layout (all integers little-endian):
 *   [4B magic "BNST"] [4B version] [4B n_sessions] [4B reserved]
 *   per session:
 *     [4B id_len] [id bytes] [4B n_episodes]
 *     per episode: [4B ep_len] [episode bytes]
 *     if version >= 2: [4B means_flag] and, when nonzero,
 *                      n_episodes x 2048 float32 mean rows
 */
#ifndef MEMORY_STATE_H
#define MEMORY_STATE_H

#define MS_FEAT_DIM 2048

#define MS_MAX_ROWS 128      /* cached feature-window rows per episode */

typedef struct {
    char id[64];
    int n_episodes;
    int ep_capacity;          /* allocated episode slots */
    char **episodes;          /* arbitrary-length strings */
    float **ep_means;         /* per-episode [MS_FEAT_DIM] or NULL */
    float **ep_rows;          /* per-episode [n_rows x MS_FEAT_DIM] or NULL */
    int *ep_row_counts;       /* cached row count per episode (0 = none) */
    int means_valid;          /* 1 when every episode has a mean */
} ms_session_t;

/* Find without creating; get creates on demand. NULL only on OOM. */
ms_session_t *ms_find_session(const char *id);
ms_session_t *ms_get_session(const char *id);

/* Append an episode of ANY length (stored verbatim). */
void ms_add_episode(ms_session_t *s, const char *text);

/* Cache one episode's 2048-dim feature mean; flags the session valid. */
void ms_set_mean(ms_session_t *s, int index, const float *mean);

/* Cache an episode's feature WINDOW rows (n_rows <= MS_MAX_ROWS) for the
 * joint-pair relevance head. Rows subsume the mean (derived on demand). */
void ms_set_rows(ms_session_t *s, int index, const float *rows, int n_rows);

int ms_session_count(void);
void ms_reset_all(void);   /* frees everything; test/reload helper */

int ms_save(const char *path);
int ms_load(const char *path);

#endif
