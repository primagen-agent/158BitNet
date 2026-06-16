#include "tensor.h"

#include <stdio.h>
#include <string.h>

const gguf_tensor_t *gguf_find_tensor(const gguf_file_t *file, const char *name) {
    if (file == NULL || name == NULL || file->tensors == NULL) {
        return NULL;
    }

    for (uint64_t i = 0; i < file->tensor_count; ++i) {
        const gguf_tensor_t *tensor = &file->tensors[i];
        if (tensor->name != NULL && strcmp(tensor->name, name) == 0) {
            return tensor;
        }
    }

    return NULL;
}

int gguf_tensor_has_shape(const gguf_tensor_t *tensor, const uint64_t *dims, uint32_t n_dims) {
    if (tensor == NULL || dims == NULL) {
        return 0;
    }

    if (tensor->n_dims != n_dims) {
        return 0;
    }

    for (uint32_t i = 0; i < n_dims; ++i) {
        if (tensor->dims[i] != dims[i]) {
            return 0;
        }
    }

    return 1;
}

int gguf_read_tensor_data(const char *gguf_path, uint64_t data_offset, const gguf_tensor_t *tensor, void *dst, size_t dst_size) {
    FILE *fp = NULL;

    if (gguf_path == NULL || tensor == NULL || dst == NULL) {
        return -1;
    }

    fp = fopen(gguf_path, "rb");
    if (fp == NULL) {
        return -1;
    }

    if (fseek(fp, (long)(data_offset + tensor->offset), SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }

    if (fread(dst, 1, dst_size, fp) != dst_size) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}
