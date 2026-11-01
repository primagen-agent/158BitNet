#ifndef BITNET_NEURAL_SLOTS_H
#define BITNET_NEURAL_SLOTS_H
#include "bitnet.h"
#include "neural_read_head.h"
#include <stddef.h>
#include <stdint.h>

#define NS_HIDDEN 2048
#define NS_LEXICAL 2048
#define NS_INPUT (NS_HIDDEN + NS_LEXICAL)
#define NS_BOUNDARY_INPUT (3 * NS_INPUT)
#define NS_WIDTH 64
#define NS_KIND_CONTEXT_INPUT (2 * NS_HIDDEN)
#define NS_READ_CONTEXT_WIDTH 16
#define NS_MEMBERSHIP_INPUT (NS_WIDTH + NS_BOUNDARY_INPUT)
#define NS_SOURCE_CONTEXT_WIDTH 32
#define NS_SOURCE_CONTEXT_INPUT (2 * NS_WIDTH + NS_HIDDEN)
#define NS_RELATION_HEADS 4
#define NS_SEMANTIC (NS_RELATION_HEADS * NS_WIDTH)
/* Four independent semantic attention heads followed by temporal intent. */
#define NS_QUERY (NS_SEMANTIC + NS_WIDTH)
#define NS_IDENTITY (NS_LEXICAL + NS_WIDTH)
/* Per-forward resource limit. Overlength input is rejected, never truncated. */
#define NS_TOKEN_LIMIT 4096
/* Activation operating point: a live record enters the active set when its
 * learned joint probability exceeds this floor. The neutral point of the
 * reader's score transform is 0.5; the runtime reads lower because evidence
 * that the reader considered plausible is worth surfacing while a withheld
 * record costs the answer. The authored development bank reaches 76% recall
 * at 0.5 and 80% at 0.1 with zero false activation throughout, so the floor
 * is the point where the recall target is met without weakening rejection. */
#define NS_ACTIVATION_PROBABILITY 0.1f
/* The active set carries at most this many records per pass, highest learned
 * score first. Later rounds append those records to their context and the
 * server places them in the prompt, so the budget is what keeps both a fixed
 * small share of the context window instead of growing with the bank. */
#define NS_EVIDENCE_BUDGET 8

typedef enum { NS_IGNORE, NS_WRITE, NS_UPDATE, NS_DELETE } ns_action_t;
typedef enum { NS_CURRENT, NS_SUPERSEDED, NS_DELETED, NS_TENTATIVE } ns_status_t;
typedef struct ns_model ns_model_t;
typedef struct {
    size_t count;
    float *rows;           /* normalized C hidden+lexical features [count,NS_INPUT] */
    size_t *starts, *ends; /* exact UTF-8 source byte offsets for each row */
    size_t word_count;
    float *word_rows; /* structural word aggregation; entity choice is neural */
} ns_features_t;
typedef struct {
    uint64_t id, supersedes, created_at;
    ns_status_t status;
    char *content, *source; /* Exact selected fragment; unselected content is not stored. */
    char *referent; /* model-selected source fragments from prior dialogue */
    char *speaker, *observed_at, *source_id; /* literal provenance, not subject guesses */
    size_t source_start, source_end; /* Absolute byte offsets in the original observation body. */
    size_t count;
    float *encoded;              /* learned write representation [count,64] */
    float identity[NS_IDENTITY]; /* learned entity-focused identity representation
                                  */
    float read_context[NS_READ_CONTEXT_WIDTH]; /* frozen raw-hidden read projection */
    /* Non-persistent activation cache: normalized lexical token rows of the
     * exact stored text (referent and content). Never serialized. */
    float *lexical_rows;
    size_t lexical_count;
} ns_slot_t;
/* A bank is owned by one scope. Callers serialize reads and mutations. */
typedef struct {
    ns_slot_t **slots;
    size_t count, capacity;
    uint64_t next_id;
} ns_bank_t;
typedef struct {
    ns_action_t action;
    size_t written, updated, deleted;
    int unresolved;    /* model requires a reference but selected no source */
    size_t candidates; /* candidates selected by the joint plan */
} ns_write_result_t;
typedef struct {
    uint64_t *ids;
    float *scores;
    size_t count;
    unsigned rounds;
    int complete; /* false when the configured read budget was exhausted */
} ns_activation_t;

ns_model_t *ns_model_load(const char *path, const char *backbone);
/* Frozen backbone identity the model file is bound to. */
const unsigned char *ns_model_backbone_sha(const ns_model_t *model);
/* Aligned source-word rows: encoded[count][NS_WIDTH] and mandatory raw
 * features[count][NS_INPUT]. The learned raw residual uses previous/current/next
 * rows, normalizes each hidden/lexical half, and pads document edges with zero.
 * Both inputs remain unchanged. */
int ns_plan_boundary_logits(const ns_model_t *model, const float *encoded,
                            const float *raw_word_rows, size_t count,
                            float *begin_logits, float *end_logits);
/* Complete residual source heads share one normalized raw three-word window.
 * Membership prepends the unchanged current encoded[NS_WIDTH] source row. */
int ns_plan_source_logits(const ns_model_t *model, const float *encoded,
                          const float *raw_word_rows, size_t count,
                          float *begin_logits, float *end_logits, float *membership_logits);
/* Learned source-context residual: [body, full-body, normalized current-word
 * hidden] passes through a NS_SOURCE_CONTEXT_WIDTH-wide tanh layer and a
 * linear scalar output. Raw rows have stride NS_INPUT; the complete raw
 * representation is validated. All aligned inputs stay unchanged. */
int ns_context_membership_logits(const ns_model_t *model, const float *full,
                                  const float *body, const float *raw_word_rows,
                                  size_t count, float *residual);
/* Frozen action logits plus learned begin and source-membership residuals.
 * Membership adds to all source tags1..4 equally; outside0 stays unchanged. */
int ns_plan_emission_logits(const ns_model_t *model, const float *encoded,
                            const float *begin_logits, const float *membership_logits,
                            size_t count, float *logits);
int ns_decode_plan_logits(const ns_model_t *model, const float *logits, const float *end_logits,
                          size_t count, int *tags);
void ns_model_free(ns_model_t *model);
int ns_encode(bitnet_model_t *backbone, const char *text, ns_features_t *out);
int ns_features_words(ns_features_t *features, const char *source);
void ns_features_clear(ns_features_t *features);
/* Learned heads. No text, labels, gold fields or operation hints enter them.
 * Kind classification combines the 64-wide memory mean with the normalized
 * original hidden last/mean context; memory token geometry stays unchanged. */
int ns_predict_kind(const ns_model_t *model, const ns_features_t *input);
int ns_kind_logits(const ns_model_t *model, const ns_features_t *input, float logits[3]);
typedef struct {
    size_t start, end;
} ns_span_t;
int ns_prepare(const ns_model_t *model, const ns_features_t *input, ns_slot_t *slot);
int ns_query(const ns_model_t *model, const ns_features_t *input, float query[NS_QUERY],
             float identity[NS_IDENTITY]);
/* A single-pass learned scope query for UPDATE/DELETE; no read feedback. */
int ns_scope_query(const ns_model_t *model, const ns_features_t *input, float query[NS_QUERY],
                   float identity[NS_IDENTITY]);
int ns_advance_query(const ns_model_t *model, const float original[NS_QUERY],
                     float current[NS_QUERY]);
/* Use raw queries before advancing the relation channels. Identity is mixed
 * by a learned scalar gate, then normalized in its lexical/projected blocks. */
int ns_advance_identity(const ns_model_t *model, const float original_query[NS_QUERY],
                        const float current_query[NS_QUERY],
                        const float original_identity[NS_IDENTITY],
                        float current_identity[NS_IDENTITY]);
float ns_score(const ns_model_t *model, const float query[NS_QUERY],
               const float identity[NS_IDENTITY], const ns_slot_t *slot, size_t index,
               size_t bank_count, int scope);
void ns_bank_clear(ns_bank_t *bank);
const ns_slot_t *ns_bank_find(const ns_bank_t *bank, uint64_t id);
int ns_ingest(const ns_model_t *model, bitnet_model_t *backbone, ns_bank_t *bank, const char *text,
              ns_write_result_t *result);
/* When path is configured, publish the staged snapshot before replacing RAM.
 * Slot pointers may change on a durable commit; persistent IDs remain stable.
 */
int ns_ingest_persist(const ns_model_t *model, bitnet_model_t *backbone, ns_bank_t *bank,
                      const char *text, const char *path, ns_write_result_t *result);
char *ns_dialogue_plain(const char *chatml);
char *ns_slot_text(const ns_slot_t *slot);
typedef struct {
    const char *body, *speaker, *timestamp, *source_id, *past;
} ns_observation_t;
typedef struct {
    ns_span_t span;
    ns_action_t action;
    char *reference;
    int unresolved;
} ns_plan_item_t;
typedef struct {
    ns_plan_item_t *items;
    size_t count;
    char *interaction_reference; /* whole-message read reference, chosen by model */
    int interaction_unresolved;
} ns_write_plan_t;
int ns_plan_observation(const ns_model_t *model, bitnet_model_t *backbone,
                        const ns_observation_t *observation, ns_write_plan_t *plan);
void ns_write_plan_clear(ns_write_plan_t *plan);
/* Return the model-generated joint plan while committing its writes. The
 * caller owns the returned plan and clears it on success or failure. */
int ns_ingest_observation_with_plan(const ns_model_t *model, bitnet_model_t *backbone,
                                    ns_bank_t *bank, const ns_observation_t *observation,
                                    const char *path, ns_write_result_t *result,
                                    ns_write_plan_t *plan);
int ns_ingest_observation(const ns_model_t *model, bitnet_model_t *backbone, ns_bank_t *bank,
                          const ns_observation_t *observation, const char *path,
                          ns_write_result_t *result);
char *ns_observation_interaction(const ns_model_t *model, bitnet_model_t *backbone,
                                 const ns_observation_t *observation, char **reference,
                                 int *unresolved);
/* Full-bank learned activation, then additional model passes over selected
 * evidence. Every pass replaces the set and can retract prior selections.
 * Addresses are decoded from gates; no keyword/ANN candidate stage.
 */
int ns_activate(const ns_model_t *model, bitnet_model_t *backbone, const ns_bank_t *bank,
                const char *interaction, unsigned max_rounds, ns_activation_t *out);
/* The context model selected these reference source bytes. Their separately
 * pooled identity anchors the first pass and the learned later-round update. */
int ns_activate_context(const ns_model_t *model, bitnet_model_t *backbone, const ns_bank_t *bank,
                        const char *interaction, const char *reference, unsigned max_rounds,
                        ns_activation_t *out);
/* Optional learned read-decision head. For activation it replaces the reader
 * logit with a decision over the reader logit and frozen lexical token
 * alignment between the original interaction and each live slot's stored text.
 * Alignment features are normalized over the complete live bank, and later
 * rounds keep that original query alignment while re-scoring their
 * evidence-augmented context. Write, reference and update decisions keep the
 * reader logit. Slots cache their normalized lexical rows as a non-persistent
 * activation cache; imported snapshots rebuild it from the frozen backbone on
 * first read. */
int ns_activate_context_head(const ns_model_t *model, const ns_read_head_t *head,
                             bitnet_model_t *backbone, const ns_bank_t *bank,
                             const char *interaction, const char *reference,
                             unsigned max_rounds, ns_activation_t *out);
/* Diagnostic factors from the same scoring pass, before any thresholding. */
typedef struct {
    float semantic_logit, identity_logit, version_logit;
    float intent_logits[3];
    unsigned intent_index;
    float semantic_log_probability, identity_log_probability;
    float version_log_probability, intent_log_probability;
    float joint_log_probability, clamped_log_probability, score;
} ns_score_trace_t;
typedef struct {
    uint64_t id;
    ns_status_t status;
    ns_score_trace_t factors;
    /* Filled only when a read head scored this slot in this pass. */
    int has_head;
    float head_logit;
    float head_features[NS_READ_HEAD_FEATURES];
} ns_slot_trace_t;
typedef struct {
    unsigned round;
    int stable; /* set convergence, including an initially empty set */
    size_t slot_count, active_count;
    const ns_slot_trace_t *slots;
    const uint64_t *active_ids;
    const float *query, *identity, *original_query, *original_identity;
} ns_round_trace_t;
/* Borrowed const arrays and vectors are valid only during the callback.
 * Return nonzero to fail this diagnostic call. No trace buffers are allocated
 * without a callback. Normal activation, HTTP and snapshots do not use traces. */
typedef int (*ns_trace_callback_t)(const ns_round_trace_t *round, void *user);
int ns_activate_context_trace(const ns_model_t *model, bitnet_model_t *backbone,
                              const ns_bank_t *bank, const char *interaction,
                              const char *reference, unsigned max_rounds,
                              ns_activation_t *out, ns_trace_callback_t callback, void *user);
int ns_activate_context_trace_head(const ns_model_t *model, const ns_read_head_t *head,
                                   bitnet_model_t *backbone, const ns_bank_t *bank,
                                   const char *interaction, const char *reference,
                                   unsigned max_rounds, ns_activation_t *out,
                                   ns_trace_callback_t callback, void *user);
void ns_activation_clear(ns_activation_t *activation);
/* Escape role delimiters in evidence/content, preserving stored raw bytes. */
char *ns_escape_prompt_text(const char *text);
char *ns_memory_prompt(const ns_bank_t *bank, const ns_activation_t *activation,
                       const char *recent_dialogue, const char *user_message,
                       const ns_write_result_t *write_result);
char *ns_memory_prompt_context(const ns_bank_t *bank, const ns_activation_t *activation,
                               const char *recent, const char *current,
                               const ns_write_result_t *write, const char *reference);
/* CRC and model-identity bound; atomic publication and transactional reload. */
int ns_bank_save(const ns_bank_t *bank, const ns_model_t *model, const char *path);
int ns_bank_load(ns_bank_t *bank, const ns_model_t *model, const char *path);
#endif
