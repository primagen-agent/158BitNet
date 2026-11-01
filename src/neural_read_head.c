#include "neural_read_head.h"
#include "neural_slots.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ns_read_head {
    size_t width, layers;
    size_t in[NS_READ_HEAD_MAX_LAYERS], out[NS_READ_HEAD_MAX_LAYERS];
    float *weight[NS_READ_HEAD_MAX_LAYERS], *bias[NS_READ_HEAD_MAX_LAYERS];
    float *mean, *std;
    unsigned char backbone[32], reader[32];
};

void ns_read_head_free(ns_read_head_t *h) {
    if (!h)
        return;
    for (size_t i = 0; i < NS_READ_HEAD_MAX_LAYERS; i++) {
        free(h->weight[i]);
        free(h->bias[i]);
    }
    free(h->mean);
    free(h->std);
    free(h);
}

const unsigned char *ns_read_head_backbone_sha(const ns_read_head_t *h) {
    return h ? h->backbone : NULL;
}
const unsigned char *ns_read_head_reader_sha(const ns_read_head_t *h) {
    return h ? h->reader : NULL;
}

static int read32(FILE *f, uint32_t *v) {
    return fread(v, 4, 1, f) == 1;
}

ns_read_head_t *ns_read_head_load(const char *path, const ns_model_t *model) {
    const unsigned char *expected = ns_model_backbone_sha(model);
    if (!path || !expected)
        return NULL;
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char magic[8];
    uint32_t version = 0, width = 0, layers = 0;
    ns_read_head_t *h = calloc(1, sizeof *h);
    if (!h || fread(magic, 1, 8, f) != 8 || memcmp(magic, "NSRDHEAD", 8) ||
        !read32(f, &version) || version != 1 || !read32(f, &width) ||
        width != NS_READ_HEAD_FEATURES || !read32(f, &layers) ||
        layers != NS_READ_HEAD_MAX_LAYERS)
        goto fail;
    h->width = width;
    h->layers = layers;
    for (size_t i = 0; i < layers; i++) {
        uint32_t in = 0, out = 0;
        if (!read32(f, &out) || !read32(f, &in) || !in || !out ||
            (i + 1 < layers && out != NS_READ_HEAD_HIDDEN) ||
            (i + 1 == layers && out != 1))
            goto fail;
        h->in[i] = in;
        h->out[i] = out;
        h->weight[i] = malloc((size_t)in * out * sizeof(float));
        h->bias[i] = malloc((size_t)out * sizeof(float));
        if (!h->weight[i] || !h->bias[i] ||
            fread(h->weight[i], sizeof(float), (size_t)in * out, f) != (size_t)in * out ||
            fread(h->bias[i], sizeof(float), out, f) != out)
            goto fail;
        for (size_t j = 0; j < (size_t)in * out; j++)
            if (!isfinite(h->weight[i][j]))
                goto fail;
        for (size_t j = 0; j < out; j++)
            if (!isfinite(h->bias[i][j]))
                goto fail;
    }
    if (h->in[0] != width)
        goto fail;
    h->mean = malloc(width * sizeof(float));
    h->std = malloc(width * sizeof(float));
    if (!h->mean || !h->std || fread(h->mean, sizeof(float), width, f) != width ||
        fread(h->std, sizeof(float), width, f) != width ||
        fread(h->reader, 1, 32, f) != 32 || fread(h->backbone, 1, 32, f) != 32 ||
        fgetc(f) != EOF || ferror(f))
        goto fail;
    for (size_t j = 0; j < width; j++) {
        if (!isfinite(h->mean[j]) || !isfinite(h->std[j]) || h->std[j] <= 0.f)
            goto fail;
    }
    for (size_t i = 0; i + 1 < layers; i++)
        if (h->in[i + 1] != h->out[i])
            goto fail;
    if (memcmp(h->backbone, expected, 32))
        goto fail;
    fclose(f);
    return h;
fail:
    fclose(f);
    ns_read_head_free(h);
    return NULL;
}

int ns_read_head_logit(const ns_read_head_t *h,
                       const float features[NS_READ_HEAD_FEATURES], float *out) {
    if (!h || !features || !out || h->layers != NS_READ_HEAD_MAX_LAYERS ||
        h->in[0] != h->width)
        return -1;
    float a[NS_READ_HEAD_HIDDEN > NS_READ_HEAD_FEATURES ? NS_READ_HEAD_HIDDEN
                                                       : NS_READ_HEAD_FEATURES];
    float b[NS_READ_HEAD_HIDDEN];
    for (size_t j = 0; j < h->width; j++) {
        a[j] = (features[j] - h->mean[j]) / h->std[j];
        if (!isfinite(a[j]))
            return -1;
    }
    float *src = a, *dst = b;
    for (size_t layer = 0; layer < h->layers; layer++) {
        size_t in = h->in[layer], outn = h->out[layer];
        for (size_t o = 0; o < outn; o++) {
            float sum = h->bias[layer][o];
            const float *row = h->weight[layer] + o * in;
            for (size_t j = 0; j < in; j++)
                sum += row[j] * src[j];
            dst[o] = layer + 1 < h->layers ? (sum > 0.f ? sum : 0.f) : sum;
        }
        float *swap = src;
        src = dst;
        dst = swap;
    }
    if (!isfinite(src[0]))
        return -1;
    *out = src[0];
    return 0;
}
