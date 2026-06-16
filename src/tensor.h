#ifndef TENSOR_H
#define TENSOR_H

#include "gguf.h"

#include <stddef.h>

const gguf_tensor_t *gguf_find_tensor(const gguf_file_t *file, const char *name);
int gguf_tensor_has_shape(const gguf_tensor_t *tensor, const uint64_t *dims, uint32_t n_dims);
int gguf_read_tensor_data(const char *gguf_path, uint64_t data_offset, const gguf_tensor_t *tensor, void *dst, size_t dst_size);

#endif
