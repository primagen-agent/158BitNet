#include "bitnet.h"
#include "metis/typed_link_model.h"
#include "metis/typed_pair_encoder.h"
#include "sha256.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct encoded_episode {
    int *tokens;
    int token_count;
    float *expected_hidden;
    float *expected_identity;
    float *hidden;
    float *identity;
} encoded_episode_t;

static int read_exact(FILE *file, void *output, size_t size) {
    return fread(output, 1, size, file) == size ? 0 : -1;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (read_exact(file, bytes, sizeof bytes) != 0) return -1;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 0;
}

static int read_floats(
    FILE *file, size_t count, float **output) {
    if (count == 0 || count > SIZE_MAX / sizeof(float))
        return -1;
    *output = (float *)malloc(count * sizeof(float));
    if (*output == NULL ||
        read_exact(file, *output,
                   count * sizeof(float)) != 0) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

static void episode_free(encoded_episode_t *episode) {
    if (episode == NULL) return;
    free(episode->tokens);
    free(episode->expected_hidden);
    free(episode->expected_identity);
    free(episode->hidden);
    free(episode->identity);
    memset(episode, 0, sizeof *episode);
}

static int episode_read(
    FILE *file, uint32_t hidden, uint32_t bands,
    encoded_episode_t *episode) {
    uint32_t count;
    size_t hidden_count;
    size_t identity_count;
    if (read_u32(file, &count) != 0 ||
        count == 0 || count > INT32_MAX ||
        (size_t)count > SIZE_MAX / (size_t)bands ||
        (size_t)count * (size_t)bands >
            SIZE_MAX / (size_t)hidden)
        return -1;
    episode->token_count = (int)count;
    episode->tokens = (int *)malloc(
        (size_t)count * sizeof(int));
    if (episode->tokens == NULL ||
        read_exact(file, episode->tokens,
                   (size_t)count * sizeof(int)) != 0)
        return -1;
    hidden_count =
        (size_t)count * (size_t)bands * (size_t)hidden;
    identity_count =
        (size_t)count * (size_t)hidden;
    if (read_floats(
            file, hidden_count,
            &episode->expected_hidden) != 0 ||
        read_floats(
            file, identity_count,
            &episode->expected_identity) != 0)
        return -1;
    return 0;
}

static int episode_encode(
    bitnet_model_t *backbone,
    const int *band_start, const int *band_end,
    int bands, int hidden_dim,
    encoded_episode_t *episode) {
    bitnet_context_t *context = NULL;
    const float *hidden;
    size_t hidden_count =
        (size_t)episode->token_count *
        (size_t)bands * (size_t)hidden_dim;
    size_t identity_count =
        (size_t)episode->token_count * (size_t)hidden_dim;
    context = bitnet_create_context(
        backbone, episode->token_count);
    episode->hidden = (float *)malloc(
        hidden_count * sizeof(float));
    episode->identity = (float *)malloc(
        identity_count * sizeof(float));
    if (context == NULL || episode->hidden == NULL ||
        episode->identity == NULL ||
        bitnet_context_configure_layer_bands(
            context, band_start, band_end, bands) != 0 ||
        bitnet_eval_hidden(
            context, episode->tokens,
            episode->token_count) != 0)
        goto fail;
    hidden = bitnet_get_last_eval_layer_bands(context);
    if (hidden == NULL ||
        bitnet_last_eval_layer_band_token_count(context)
            != episode->token_count ||
        bitnet_token_embedding_lookup(
            backbone, episode->tokens,
            episode->token_count,
            episode->identity, identity_count) != 0)
        goto fail;
    memcpy(episode->hidden, hidden,
           hidden_count * sizeof(float));
    bitnet_free_context(context);
    return 0;
fail:
    bitnet_free_context(context);
    return -1;
}

static float max_error(
    const float *left, const float *right, size_t count) {
    float maximum = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        const float value = fabsf(left[index] - right[index]);
        if (value > maximum) maximum = value;
    }
    return maximum;
}

static float top_gap(const float *values, size_t count) {
    float top1 = -INFINITY;
    float top2 = -INFINITY;
    for (size_t index = 0; index < count; ++index) {
        if (values[index] > top1) {
            top2 = top1;
            top1 = values[index];
        } else if (values[index] > top2) {
            top2 = values[index];
        }
    }
    return count > 1 ? top1 - top2 : 0.0f;
}

int main(int argc, char **argv) {
    static const unsigned char magic[8] =
        {'B', 'N', 'T', 'R', 'L', 'N', 'K', '1'};
    unsigned char actual_magic[8];
    unsigned char expected_sha[32];
    unsigned char actual_sha[32];
    uint32_t version;
    uint32_t hidden;
    uint32_t bands;
    uint32_t layers;
    uint32_t feature_dim;
    uint32_t candidate_count = 0;
    uint32_t current_count;
    uint32_t expected_selected;
    int *band_start = NULL;
    int *band_end = NULL;
    encoded_episode_t current = {0};
    encoded_episode_t *candidates = NULL;
    float *expected_entity = NULL;
    float *expected_predicate = NULL;
    float *expected_entity_logits = NULL;
    float *expected_predicate_logits = NULL;
    float *expected_joint = NULL;
    float expected_exists;
    float *actual_entity = NULL;
    float *actual_predicate = NULL;
    float *actual_entity_logits = NULL;
    float *actual_predicate_logits = NULL;
    float *actual_joint = NULL;
    float actual_exists;
    size_t actual_selected = SIZE_MAX;
    float feature_error;
    float logit_error;
    float joint_error;
    FILE *file = NULL;
    bitnet_model_t *backbone = NULL;
    metis_typed_pair_encoder_t *encoder = NULL;
    metis_typed_link_model_t *link = NULL;
    char error[160];
    int status = 1;

    if (argc != 5) {
        fprintf(stderr,
                "usage: %s MODEL.bntpair MODEL.bntlink "
                "BACKBONE.gguf SAMPLE.bin\n",
                argv[0]);
        return 2;
    }
    file = fopen(argv[4], "rb");
    if (file == NULL ||
        read_exact(file, actual_magic, sizeof actual_magic) != 0 ||
        memcmp(actual_magic, magic, sizeof magic) != 0 ||
        read_u32(file, &version) != 0 || version != 1 ||
        read_u32(file, &hidden) != 0 ||
        read_u32(file, &bands) != 0 ||
        read_u32(file, &layers) != 0 ||
        read_u32(file, &feature_dim) != 0 ||
        read_u32(file, &candidate_count) != 0 ||
        read_u32(file, &current_count) != 0 ||
        hidden == 0 || bands < 2 || layers < bands ||
        feature_dim == 0 || candidate_count < 2 ||
        current_count == 0)
        goto cleanup;
    band_start = (int *)malloc(bands * sizeof(int));
    band_end = (int *)malloc(bands * sizeof(int));
    candidates = (encoded_episode_t *)calloc(
        candidate_count, sizeof(*candidates));
    if (band_start == NULL || band_end == NULL ||
        candidates == NULL)
        goto cleanup;
    for (uint32_t band = 0; band < bands; ++band) {
        uint32_t start;
        uint32_t end;
        if (read_u32(file, &start) != 0 ||
            read_u32(file, &end) != 0 ||
            start > INT32_MAX || end > INT32_MAX)
            goto cleanup;
        band_start[band] = (int)start;
        band_end[band] = (int)end;
    }
    if (read_exact(file, expected_sha, sizeof expected_sha) != 0 ||
        episode_read(file, hidden, bands, &current) != 0 ||
        current.token_count != (int)current_count)
        goto cleanup;
    for (uint32_t index = 0;
         index < candidate_count; ++index)
        if (episode_read(
                file, hidden, bands,
                &candidates[index]) != 0)
            goto cleanup;
    {
        const size_t feature_count =
            (size_t)candidate_count * feature_dim;
        if (read_floats(
                file, feature_count,
                &expected_entity) != 0 ||
            read_floats(
                file, feature_count,
                &expected_predicate) != 0 ||
            read_floats(
                file, candidate_count,
                &expected_entity_logits) != 0 ||
            read_floats(
                file, candidate_count,
                &expected_predicate_logits) != 0 ||
            read_floats(
                file, candidate_count,
                &expected_joint) != 0 ||
            read_exact(file, &expected_exists,
                       sizeof expected_exists) != 0 ||
            read_u32(file, &expected_selected) != 0 ||
            fgetc(file) != EOF)
            goto cleanup;
        actual_entity = (float *)malloc(
            feature_count * sizeof(float));
        actual_predicate = (float *)malloc(
            feature_count * sizeof(float));
    }
    fclose(file);
    file = NULL;
    if (bitnet_sha256_file(argv[3], actual_sha) != 0 ||
        memcmp(actual_sha, expected_sha, sizeof actual_sha) != 0)
        goto cleanup;
    backbone = bitnet_load_model(argv[3]);
    encoder = metis_typed_pair_encoder_load(
        argv[1], argv[3], (int)hidden,
        error, sizeof error);
    link = metis_typed_link_model_load(
        argv[2], argv[3], error, sizeof error);
    actual_entity_logits = (float *)malloc(
        candidate_count * sizeof(float));
    actual_predicate_logits = (float *)malloc(
        candidate_count * sizeof(float));
    actual_joint = (float *)malloc(
        candidate_count * sizeof(float));
    if (backbone == NULL || encoder == NULL || link == NULL ||
        encoder->band_count != (int)bands ||
        encoder->layer_count != (int)layers ||
        encoder->head_width != (int)feature_dim ||
        link->pair_feature_dim != (int)feature_dim ||
        actual_entity == NULL || actual_predicate == NULL ||
        actual_entity_logits == NULL ||
        actual_predicate_logits == NULL ||
        actual_joint == NULL ||
        episode_encode(
            backbone, band_start, band_end,
            (int)bands, (int)hidden, &current) != 0)
        goto cleanup;
    for (uint32_t index = 0;
         index < candidate_count; ++index) {
        if (episode_encode(
                backbone, band_start, band_end,
                (int)bands, (int)hidden,
                &candidates[index]) != 0 ||
            metis_typed_pair_encoder_score(
                encoder,
                current.hidden,
                (size_t)current.token_count,
                current.identity,
                candidates[index].hidden,
                (size_t)candidates[index].token_count,
                candidates[index].identity,
                actual_entity +
                    (size_t)index * feature_dim,
                actual_predicate +
                    (size_t)index * feature_dim,
                &actual_entity_logits[index],
                &actual_predicate_logits[index]) != 0)
            goto cleanup;
    }
    if (metis_typed_link_score_pairs(
            link, actual_entity, actual_predicate,
            actual_entity_logits, actual_predicate_logits,
            candidate_count, actual_joint) != 0 ||
        metis_typed_link_select_predecessor(
            link, actual_joint, candidate_count,
            &actual_selected, &actual_exists) != 0)
        goto cleanup;
    feature_error = fmaxf(
        max_error(
            actual_entity, expected_entity,
            (size_t)candidate_count * feature_dim),
        max_error(
            actual_predicate, expected_predicate,
            (size_t)candidate_count * feature_dim));
    logit_error = fmaxf(
        max_error(
            actual_entity_logits,
            expected_entity_logits, candidate_count),
        max_error(
            actual_predicate_logits,
            expected_predicate_logits, candidate_count));
    joint_error = max_error(
        actual_joint, expected_joint, candidate_count);
    printf(
        "test_typed_runtime_link_parity: "
        "candidates=%u feature_error=%.9g "
        "pair_logit_error=%.9g joint_error=%.9g "
        "exists=%.9g/%.9g "
        "top_gap=%.9g/%.9g selected=%zu/%u\n",
        candidate_count, feature_error,
        logit_error, joint_error,
        actual_exists, expected_exists,
        top_gap(actual_joint, candidate_count),
        top_gap(expected_joint, candidate_count),
        actual_selected, expected_selected);
    if (
        feature_error > 2.5e-2f ||
        logit_error > 7.5e-2f ||
        joint_error > 7.5e-2f ||
        fabsf(actual_exists - expected_exists) > 1e-2f ||
        (actual_exists > 0.0f) != (expected_exists > 0.0f) ||
        actual_selected != (
            expected_selected == UINT32_MAX
                ? SIZE_MAX : (size_t)expected_selected))
        goto cleanup;
    status = 0;
cleanup:
    if (file != NULL) fclose(file);
    bitnet_free_model(backbone);
    metis_typed_pair_encoder_free(encoder);
    metis_typed_link_model_free(link);
    episode_free(&current);
    if (candidates != NULL)
        for (uint32_t index = 0;
             index < candidate_count; ++index)
            episode_free(&candidates[index]);
    free(candidates);
    free(band_start);
    free(band_end);
    free(expected_entity);
    free(expected_predicate);
    free(expected_entity_logits);
    free(expected_predicate_logits);
    free(expected_joint);
    free(actual_entity);
    free(actual_predicate);
    free(actual_entity_logits);
    free(actual_predicate_logits);
    free(actual_joint);
    return status;
}
