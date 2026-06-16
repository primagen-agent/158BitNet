#include "gguf.h"

#include <stdio.h>

static void print_metadata_value(const gguf_metadata_t *entry) {
    switch (entry->type) {
        case GGUF_TYPE_STRING:
            printf("%s", entry->string_value != NULL ? entry->string_value : "");
            break;
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_UINT64:
            printf("%llu", (unsigned long long)entry->value.u64);
            break;
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_INT64:
            printf("%lld", (long long)entry->value.i64);
            break;
        case GGUF_TYPE_FLOAT32:
        case GGUF_TYPE_FLOAT64:
            printf("%.8g", entry->value.f64);
            break;
        case GGUF_TYPE_BOOL:
            printf("%s", entry->value.boolean ? "true" : "false");
            break;
        case GGUF_TYPE_ARRAY:
            printf("array<%s>[%llu]", gguf_type_name(entry->array_type), (unsigned long long)entry->array_count);
            break;
        default:
            printf("unsupported");
            break;
    }
}

int main(int argc, char **argv) {
    gguf_file_t file;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    if (gguf_open(argv[1], &file) != 0) {
        fprintf(stderr, "Failed to open GGUF file\n");
        return 1;
    }

    printf("GGUF version: %u\n", file.version);
    printf("Metadata count: %llu\n", (unsigned long long)file.metadata_count);
    printf("Tensor count: %llu\n", (unsigned long long)file.tensor_count);
    printf("Alignment: %llu\n", (unsigned long long)file.alignment);
    printf("Data offset: %llu\n", (unsigned long long)file.data_offset);

    for (unsigned long long i = 0; i < (unsigned long long)file.metadata_count; ++i) {
        const gguf_metadata_t *entry = &file.metadata[i];
        printf("Metadata: %s type=%s value=", entry->key, gguf_type_name(entry->type));
        print_metadata_value(entry);
        printf("\n");
    }

    for (unsigned long long i = 0; i < (unsigned long long)file.tensor_count; ++i) {
        const gguf_tensor_t *tensor = &file.tensors[i];
        printf("Tensor: %s type=%s(%u) dims=", tensor->name, gguf_tensor_type_name(tensor->type), tensor->type);
        for (uint32_t dim = 0; dim < tensor->n_dims; ++dim) {
            if (dim > 0) {
                printf("x");
            }
            printf("%llu", (unsigned long long)tensor->dims[dim]);
        }
        printf(" offset=%llu\n", (unsigned long long)tensor->offset);
    }

    gguf_close(&file);
    return 0;
}
