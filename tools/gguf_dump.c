/* Print a GGUF's metadata and tensor table; with a name filter, also the
 * first values of matching F32 tensors.  Used to pin down a model file's
 * exact hyperparameters before wiring its graph into the runtime.
 *
 *   gguf_dump model.gguf [name-substring] [count]
 */
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    gguf_file_t file, *f = &file;
    uint64_t i, j;
    const char *filter = argc > 2 ? argv[2] : NULL;
    long count = argc > 3 ? atol(argv[3]) : 8;

    if (argc < 2) { fprintf(stderr, "usage: gguf_dump model.gguf [name-substring] [n]\n"); return 1; }
    if (gguf_open(argv[1], &file) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    printf("version %u tensors %llu kv %llu data at %llu\n",
           (unsigned)f->version, (unsigned long long)f->tensor_count,
           (unsigned long long)f->metadata_count, (unsigned long long)f->data_offset);
    for (i = 0; i < f->metadata_count; ++i) {
        const gguf_metadata_t *m = &f->metadata[i];
        if (strncmp(m->key, "tokenizer.", 10) == 0 && m->type != GGUF_TYPE_STRING)
            continue;   /* arrays of 70k tokens: skip, not informative here */
        if (strcmp(m->key, "tokenizer.chat_template") == 0) continue;
        switch (m->type) {
        case GGUF_TYPE_UINT32: case GGUF_TYPE_UINT64:
            printf("%s = %llu\n", m->key, (unsigned long long)m->value.u64); break;
        case GGUF_TYPE_INT32: case GGUF_TYPE_INT64:
            printf("%s = %lld\n", m->key, (long long)m->value.i64); break;
        case GGUF_TYPE_FLOAT32: case GGUF_TYPE_FLOAT64:
            printf("%s = %g\n", m->key, m->value.f64); break;
        case GGUF_TYPE_STRING:
            printf("%s = %s\n", m->key, m->string_value ? m->string_value : ""); break;
        case GGUF_TYPE_BOOL:
            printf("%s = %d\n", m->key, m->value.boolean); break;
        default: printf("%s = <type %d>\n", m->key, (int)m->type); break;
        }
    }
    printf("--- tensors ---\n");
    for (i = 0; i < f->tensor_count; ++i) {
        const gguf_tensor_t *t = &f->tensors[i];
        if (filter != NULL && strstr(t->name, filter) == NULL) continue;
        printf("%-34s [", t->name);
        for (j = 0; j < t->n_dims; ++j) printf("%s%llu", j ? ", " : "", (unsigned long long)t->dims[j]);
        printf("] type %s off %llu\n", gguf_tensor_type_name(t->type), (unsigned long long)t->offset);
        if (t->type == 0 && count > 0) {   /* F32: show the first values */
            const float *p = (const float *)gguf_get_tensor_ptr(f, t);
            uint64_t n = t->dims[0] < (uint64_t)count ? t->dims[0] : (uint64_t)count;
            printf("    ");
            for (j = 0; j < n; ++j) printf("%g ", (double)p[j]);
            printf("\n");
        }
    }
    gguf_close(&file);
    return 0;
}
