#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u

static void gguf_reset(gguf_file_t *file) {
    if (file != NULL) {
        memset(file, 0, sizeof(*file));
        file->alignment = 32;
    }
}

static int read_u32(FILE *fp, uint32_t *value) {
    return fread(value, sizeof(*value), 1, fp) == 1 ? 0 : -1;
}

static int read_u64(FILE *fp, uint64_t *value) {
    return fread(value, sizeof(*value), 1, fp) == 1 ? 0 : -1;
}

static int read_string(FILE *fp, char **out) {
    uint64_t len = 0;
    char *buf = NULL;

    if (read_u64(fp, &len) != 0) {
        return -1;
    }

    buf = (char *)malloc((size_t)len + 1);
    if (buf == NULL) {
        return -1;
    }

    if (len > 0 && fread(buf, 1, (size_t)len, fp) != len) {
        free(buf);
        return -1;
    }

    buf[len] = '\0';
    *out = buf;
    return 0;
}

static int skip_bytes(FILE *fp, uint64_t count) {
    if (count == 0) {
        return 0;
    }
    return fseek(fp, (long)count, SEEK_CUR) == 0 ? 0 : -1;
}

static int skip_value(FILE *fp, uint32_t type);

static int skip_array(FILE *fp) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    uint64_t i = 0;

    if (read_u32(fp, &elem_type) != 0 || read_u64(fp, &count) != 0) {
        return -1;
    }

    if (elem_type == GGUF_TYPE_STRING) {
        for (i = 0; i < count; ++i) {
            char *tmp = NULL;
            if (read_string(fp, &tmp) != 0) {
                return -1;
            }
            free(tmp);
        }
        return 0;
    }

    if (elem_type == GGUF_TYPE_ARRAY) {
        return -1;
    }

    for (i = 0; i < count; ++i) {
        if (skip_value(fp, elem_type) != 0) {
            return -1;
        }
    }

    return 0;
}

static int skip_value(FILE *fp, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return skip_bytes(fp, 1);
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return skip_bytes(fp, 2);
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return skip_bytes(fp, 4);
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return skip_bytes(fp, 8);
        case GGUF_TYPE_STRING: {
            char *tmp = NULL;
            int rc = read_string(fp, &tmp);
            free(tmp);
            return rc;
        }
        case GGUF_TYPE_ARRAY:
            return skip_array(fp);
        default:
            return -1;
    }
}

static int read_scalar_value(FILE *fp, gguf_metadata_t *entry) {
    switch (entry->type) {
        case GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            if (fread(&v, sizeof(v), 1, fp) != 1) return -1;
            entry->value.u64 = v;
            return 0;
        }
        case GGUF_TYPE_INT8: {
            int8_t v = 0;
            if (fread(&v, sizeof(v), 1, fp) != 1) return -1;
            entry->value.i64 = v;
            return 0;
        }
        case GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (fread(&v, sizeof(v), 1, fp) != 1) return -1;
            entry->value.u64 = v;
            return 0;
        }
        case GGUF_TYPE_INT16: {
            int16_t v = 0;
            if (fread(&v, sizeof(v), 1, fp) != 1) return -1;
            entry->value.i64 = v;
            return 0;
        }
        case GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (read_u32(fp, &v) != 0) return -1;
            entry->value.u64 = v;
            return 0;
        }
        case GGUF_TYPE_INT32: {
            int32_t v = 0;
            if (fread(&v, sizeof(v), 1, fp) != 1) return -1;
            entry->value.i64 = v;
            return 0;
        }
        case GGUF_TYPE_FLOAT32: {
            float v = 0.0f;
            if (fread(&v, sizeof(v), 1, fp) != 1) return -1;
            entry->value.f64 = v;
            return 0;
        }
        case GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            if (fread(&v, sizeof(v), 1, fp) != 1) return -1;
            entry->value.boolean = v != 0;
            return 0;
        }
        case GGUF_TYPE_STRING:
            return read_string(fp, &entry->string_value);
        case GGUF_TYPE_UINT64:
            return read_u64(fp, &entry->value.u64);
        case GGUF_TYPE_INT64:
            return fread(&entry->value.i64, sizeof(entry->value.i64), 1, fp) == 1 ? 0 : -1;
        case GGUF_TYPE_FLOAT64:
            return fread(&entry->value.f64, sizeof(entry->value.f64), 1, fp) == 1 ? 0 : -1;
        default:
            return -1;
    }
}

static int read_metadata_value(FILE *fp, gguf_metadata_t *entry) {
    if (entry->type == GGUF_TYPE_ARRAY) {
        if (read_u32(fp, &entry->array_type) != 0 || read_u64(fp, &entry->array_count) != 0) {
            return -1;
        }

        if (entry->array_type == GGUF_TYPE_STRING) {
            uint64_t i = 0;
            entry->array_strings = (char **)calloc((size_t)entry->array_count, sizeof(char *));
            if (entry->array_strings == NULL) return -1;
            for (i = 0; i < entry->array_count; ++i) {
                if (read_string(fp, &entry->array_strings[i]) != 0) {
                    return -1;
                }
            }
            return 0;
        }

        if (entry->array_type == GGUF_TYPE_FLOAT32) {
            uint64_t i = 0;
            entry->array_floats = (float *)calloc((size_t)entry->array_count, sizeof(float));
            if (entry->array_floats == NULL) return -1;
            for (i = 0; i < entry->array_count; ++i) {
                if (fread(&entry->array_floats[i], sizeof(float), 1, fp) != 1) {
                    return -1;
                }
            }
            return 0;
        }

        if (entry->array_type == GGUF_TYPE_INT32) {
            uint64_t i = 0;
            entry->array_ints = (int32_t *)calloc((size_t)entry->array_count, sizeof(int32_t));
            if (entry->array_ints == NULL) return -1;
            for (i = 0; i < entry->array_count; ++i) {
                if (fread(&entry->array_ints[i], sizeof(int32_t), 1, fp) != 1) {
                    return -1;
                }
            }
            return 0;
        }

        if (entry->array_type == GGUF_TYPE_ARRAY) {
            return -1;
        }

        for (uint64_t i = 0; i < entry->array_count; ++i) {
            if (skip_value(fp, entry->array_type) != 0) {
                return -1;
            }
        }
        return 0;
    }

    return read_scalar_value(fp, entry);
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t mask = alignment - 1;
    return (value + mask) & ~mask;
}

const gguf_metadata_t *gguf_find_metadata(const gguf_file_t *file, const char *key) {
    if (file == NULL || key == NULL) {
        return NULL;
    }

    for (uint64_t i = 0; i < file->metadata_count; ++i) {
        if (file->metadata[i].key != NULL && strcmp(file->metadata[i].key, key) == 0) {
            return &file->metadata[i];
        }
    }

    return NULL;
}

const char *gguf_type_name(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8: return "uint8";
        case GGUF_TYPE_INT8: return "int8";
        case GGUF_TYPE_UINT16: return "uint16";
        case GGUF_TYPE_INT16: return "int16";
        case GGUF_TYPE_UINT32: return "uint32";
        case GGUF_TYPE_INT32: return "int32";
        case GGUF_TYPE_FLOAT32: return "float32";
        case GGUF_TYPE_BOOL: return "bool";
        case GGUF_TYPE_STRING: return "string";
        case GGUF_TYPE_ARRAY: return "array";
        case GGUF_TYPE_UINT64: return "uint64";
        case GGUF_TYPE_INT64: return "int64";
        case GGUF_TYPE_FLOAT64: return "float64";
        default: return "unknown";
    }
}

const char *gguf_tensor_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 33: return "Q6_K_ALT";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        default: return "UNKNOWN";
    }
}

int gguf_open(const char *path, gguf_file_t *file) {
    FILE *fp = NULL;
    uint32_t magic = 0;
    long end_pos = 0;

    if (path == NULL || file == NULL) {
        return -1;
    }

    gguf_reset(file);

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    if (read_u32(fp, &magic) != 0 || magic != GGUF_MAGIC) {
        fclose(fp);
        return -1;
    }

    if (read_u32(fp, &file->version) != 0 || read_u64(fp, &file->tensor_count) != 0 || read_u64(fp, &file->metadata_count) != 0) {
        fclose(fp);
        return -1;
    }

    file->metadata = (gguf_metadata_t *)calloc((size_t)file->metadata_count, sizeof(*file->metadata));
    file->tensors = (gguf_tensor_t *)calloc((size_t)file->tensor_count, sizeof(*file->tensors));
    if ((file->metadata_count > 0 && file->metadata == NULL) || (file->tensor_count > 0 && file->tensors == NULL)) {
        gguf_close(file);
        fclose(fp);
        return -1;
    }

    for (uint64_t i = 0; i < file->metadata_count; ++i) {
        gguf_metadata_t *entry = &file->metadata[i];
        if (read_string(fp, &entry->key) != 0 || read_u32(fp, &entry->type) != 0) {
            gguf_close(file);
            fclose(fp);
            return -1;
        }

        if (read_metadata_value(fp, entry) != 0) {
            gguf_close(file);
            fclose(fp);
            return -1;
        }

        if (strcmp(entry->key, "general.alignment") == 0) {
            if (entry->type == GGUF_TYPE_UINT32 || entry->type == GGUF_TYPE_UINT64) {
                file->alignment = entry->value.u64;
            }
        }
    }

    for (uint64_t i = 0; i < file->tensor_count; ++i) {
        gguf_tensor_t *tensor = &file->tensors[i];
        if (read_string(fp, &tensor->name) != 0 || read_u32(fp, &tensor->n_dims) != 0) {
            gguf_close(file);
            fclose(fp);
            return -1;
        }

        if (tensor->n_dims > GGUF_MAX_DIMS) {
            gguf_close(file);
            fclose(fp);
            return -1;
        }

        for (uint32_t dim = 0; dim < tensor->n_dims; ++dim) {
            if (read_u64(fp, &tensor->dims[dim]) != 0) {
                gguf_close(file);
                fclose(fp);
                return -1;
            }
        }

        if (read_u32(fp, &tensor->type) != 0 || read_u64(fp, &tensor->offset) != 0) {
            gguf_close(file);
            fclose(fp);
            return -1;
        }
    }

    end_pos = ftell(fp);
    if (end_pos < 0) {
        gguf_close(file);
        fclose(fp);
        return -1;
    }

    if (file->alignment == 0) {
        file->alignment = 32;
    }

    file->data_offset = align_up((uint64_t)end_pos, file->alignment);

    /* mmap the entire file for zero-I/O tensor access */
    {
        int fd = fileno(fp);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            gguf_close(file);
            fclose(fp);
            return -1;
        }
        file->file_size = (size_t)st.st_size;
        file->data_ptr = mmap(NULL, file->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (file->data_ptr == MAP_FAILED) {
            file->data_ptr = NULL;
            file->file_size = 0;
            gguf_close(file);
            fclose(fp);
            return -1;
        }
        /* Advise kernel: will need random access to this mapping */
        madvise(file->data_ptr, file->file_size, MADV_RANDOM);
    }

    fclose(fp);
    return 0;
}

void gguf_close(gguf_file_t *file) {
    if (file == NULL) {
        return;
    }

    if (file->data_ptr != NULL) {
        munmap(file->data_ptr, file->file_size);
    }

    if (file->metadata != NULL) {
        for (uint64_t i = 0; i < file->metadata_count; ++i) {
            uint64_t j = 0;
            free(file->metadata[i].key);
            free(file->metadata[i].string_value);
            if (file->metadata[i].array_strings != NULL) {
                for (j = 0; j < file->metadata[i].array_count; ++j) {
                    free(file->metadata[i].array_strings[j]);
                }
                free(file->metadata[i].array_strings);
            }
            free(file->metadata[i].array_floats);
            free(file->metadata[i].array_ints);
        }
        free(file->metadata);
    }

    if (file->tensors != NULL) {
        for (uint64_t i = 0; i < file->tensor_count; ++i) {
            free(file->tensors[i].name);
        }
        free(file->tensors);
    }

    gguf_reset(file);
}

const void *gguf_get_tensor_ptr(const gguf_file_t *file, const gguf_tensor_t *tensor) {
    if (file == NULL || file->data_ptr == NULL || tensor == NULL) {
        return NULL;
    }
    return (const uint8_t *)file->data_ptr + file->data_offset + tensor->offset;
}
