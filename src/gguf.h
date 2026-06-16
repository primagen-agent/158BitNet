#ifndef GGUF_H
#define GGUF_H

#include <stddef.h>
#include <stdint.h>

#define GGUF_MAX_DIMS 4

typedef enum gguf_value_type {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12
} gguf_value_type_t;

typedef struct gguf_metadata {
    char *key;
    uint32_t type;
    uint32_t array_type;
    uint64_t array_count;
    char *string_value;
    char **array_strings;
    float *array_floats;
    int32_t *array_ints;
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        int boolean;
    } value;
} gguf_metadata_t;

typedef struct gguf_tensor {
    char *name;
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
} gguf_tensor_t;

typedef struct gguf_file {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t alignment;
    uint64_t data_offset;
    void *data_ptr;
    size_t file_size;
    gguf_metadata_t *metadata;
    gguf_tensor_t *tensors;
} gguf_file_t;

int gguf_open(const char *path, gguf_file_t *file);
void gguf_close(gguf_file_t *file);
const void *gguf_get_tensor_ptr(const gguf_file_t *file, const gguf_tensor_t *tensor);

const gguf_metadata_t *gguf_find_metadata(const gguf_file_t *file, const char *key);
const char *gguf_type_name(uint32_t type);
const char *gguf_tensor_type_name(uint32_t type);

#endif
