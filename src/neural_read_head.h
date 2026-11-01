/* Learned read-decision head over frozen lexical token alignment.
 *
 * The artifact holds feature standardization statistics and a small MLP. The
 * C runtime composes the same eleven inputs the Python trainer composes:
 * reader logit, query/fact token coverage, the single best token-pair cosine,
 * their within-query normalization, and three interactions. Serving applies
 * the head only to activation; write, reference and update decisions keep the
 * reader logit unchanged.
 */
#ifndef NEURAL_READ_HEAD_H
#define NEURAL_READ_HEAD_H

#include <stddef.h>
#include <stdint.h>

#define NS_READ_HEAD_FEATURES 11
#define NS_READ_HEAD_MAX_LAYERS 3
#define NS_READ_HEAD_HIDDEN 32

typedef struct ns_read_head ns_read_head_t;

struct ns_model;
struct ns_slot;

/* Load and validate an artifact; the stored backbone identity must match the
 * model's frozen backbone. Returns NULL on any malformed or non-finite input. */
ns_read_head_t *ns_read_head_load(const char *path, const struct ns_model *model);
void ns_read_head_free(ns_read_head_t *head);
int ns_read_head_logit(const ns_read_head_t *head,
                       const float features[NS_READ_HEAD_FEATURES], float *out);
const unsigned char *ns_read_head_backbone_sha(const ns_read_head_t *head);
const unsigned char *ns_read_head_reader_sha(const ns_read_head_t *head);

#endif
