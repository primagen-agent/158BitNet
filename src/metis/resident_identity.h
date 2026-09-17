#ifndef BITNET_RESIDENT_IDENTITY_H
#define BITNET_RESIDENT_IDENTITY_H
#include "bitnet.h"
#include <stddef.h>
#include <stdint.h>

#define RESIDENT_MAX_EVENTS 32
#define RESIDENT_MAX_TOKENS 128
#define RESIDENT_WIDTH 128
#define RESIDENT_HIDDEN 1024
#define RESIDENT_STRIDE (RESIDENT_WIDTH + RESIDENT_HIDDEN + 2)
typedef struct resident_model {
    float *tensor[25];
    uint8_t sha256[32];
} resident_model_t;
typedef struct resident_slot {
    size_t tokens;
    float *data;
} resident_slot_t;
typedef struct resident_state {
    size_t count;
    resident_slot_t slots[RESIDENT_MAX_EVENTS];
} resident_state_t;

resident_model_t *resident_model_load(const char *path, const char *backbone);
void resident_model_free(resident_model_t *model);
void resident_state_clear(resident_state_t *state);
/* Input [token][contextual hidden | lexical embedding], normalized identically
 * to training. BOS is excluded by the operator; no strings or labels enter it. */
int resident_write(const resident_model_t *model, const float *features, size_t tokens,
                   resident_slot_t *out);
int resident_read(const resident_model_t *model, const resident_state_t *state, const float *query,
                  size_t tokens, float *scores, float counts[5]);
int resident_select(const float *scores, const float counts[5], size_t n, size_t selected[4]);
/* Fresh-context C backbone prefill. No vocabulary decode or session KV reuse. */
float *resident_encode(bitnet_model_t *backbone, const char *text, size_t *tokens);
int resident_state_save(const resident_state_t *state, const resident_model_t *model,
                        const char *path);
int resident_state_load(resident_state_t *state, const resident_model_t *model, const char *path);
#endif
