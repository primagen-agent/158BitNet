#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include "cJSON.h"

static void *(*cjson_malloc)(size_t sz) = malloc;
static void (*cjson_free)(void *ptr) = free;

void cJSON_InitHooks(cJSON_Hooks *hooks) {
    if (!hooks) return;
    cjson_malloc = hooks->malloc_fn ? hooks->malloc_fn : malloc;
    cjson_free = hooks->free_fn ? hooks->free_fn : free;
}

static cJSON *cjson_new_item(void) {
    cJSON *node = (cJSON *)cjson_malloc(sizeof(cJSON));
    if (node) memset(node, 0, sizeof(cJSON));
    return node;
}

void cJSON_Delete(cJSON *c) {
    cJSON *next;
    while (c) {
        next = c->next;
        if (c->child) cJSON_Delete(c->child);
        if (c->valuestring) cjson_free(c->valuestring);
        if (c->string) cjson_free(c->string);
        cjson_free(c);
        c = next;
    }
}

static char *cjson_strdup(const char *str) {
    size_t len = strlen(str) + 1;
    char *copy = (char *)cjson_malloc(len);
    if (copy) memcpy(copy, str, len);
    return copy;
}

typedef struct {
    const char *content;
    int length;
    int offset;
} parse_buffer;

static int buffer_skip_whitespace(parse_buffer *buf) {
    while (buf->offset < buf->length && isspace((unsigned char)buf->content[buf->offset]))
        buf->offset++;
    return 1;
}

static int parse_value(cJSON *item, parse_buffer *buf);
static int parse_string(cJSON *item, parse_buffer *buf);
static int parse_number(cJSON *item, parse_buffer *buf);
static int parse_array(cJSON *item, parse_buffer *buf);
static int parse_object(cJSON *item, parse_buffer *buf);

static int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

static int parse_hex4(const char *src, int *codepoint) {
    int value = 0;
    for (int i = 0; i < 4; ++i) {
        int hex = hex_value(src[i]);
        if (hex < 0) return 0;
        value = (value << 4) | hex;
    }
    *codepoint = value;
    return 1;
}

static int append_utf8(char *dst, int *out, int codepoint) {
    if (codepoint < 0) return 0;
    if (codepoint <= 0x7f) {
        dst[(*out)++] = (char)codepoint;
    } else if (codepoint <= 0x7ff) {
        dst[(*out)++] = (char)(0xc0 | (codepoint >> 6));
        dst[(*out)++] = (char)(0x80 | (codepoint & 0x3f));
    } else if (codepoint <= 0xffff) {
        dst[(*out)++] = (char)(0xe0 | (codepoint >> 12));
        dst[(*out)++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        dst[(*out)++] = (char)(0x80 | (codepoint & 0x3f));
    } else if (codepoint <= 0x10ffff) {
        dst[(*out)++] = (char)(0xf0 | (codepoint >> 18));
        dst[(*out)++] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
        dst[(*out)++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        dst[(*out)++] = (char)(0x80 | (codepoint & 0x3f));
    } else {
        return 0;
    }
    return 1;
}

static int parse_value(cJSON *item, parse_buffer *buf) {
    buffer_skip_whitespace(buf);
    if (buf->offset >= buf->length) return 0;
    switch (buf->content[buf->offset]) {
        case '"': return parse_string(item, buf);
        case '-': case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            return parse_number(item, buf);
        case '[': return parse_array(item, buf);
        case '{': return parse_object(item, buf);
        case 't':
            if (buf->offset + 3 < buf->length && strncmp(buf->content + buf->offset, "true", 4) == 0) {
                item->type = (1 << 2);
                item->valueint = 1;
                item->valuedouble = 1.0;
                buf->offset += 4;
                return 1;
            }
            return 0;
        case 'f':
            if (buf->offset + 4 < buf->length && strncmp(buf->content + buf->offset, "false", 5) == 0) {
                item->type = (1 << 1);
                item->valueint = 0;
                item->valuedouble = 0.0;
                buf->offset += 5;
                return 1;
            }
            return 0;
        case 'n':
            if (buf->offset + 3 < buf->length && strncmp(buf->content + buf->offset, "null", 4) == 0) {
                item->type = (1 << 3);
                buf->offset += 4;
                return 1;
            }
            return 0;
        default: return 0;
    }
}

static int parse_string(cJSON *item, parse_buffer *buf) {
    char *str = NULL;
    int len = 0;
    int out = 0;
    if (buf->content[buf->offset] != '"') return 0;
    buf->offset++;
    int start = buf->offset;
    while (buf->offset < buf->length && buf->content[buf->offset] != '"') {
        if (buf->content[buf->offset] == '\\') buf->offset++;
        buf->offset++;
    }
    if (buf->offset >= buf->length) return 0;
    len = buf->offset - start;
    str = (char *)cjson_malloc(len + 1);
    if (!str) return 0;
    for (int i = start; i < buf->offset; ++i) {
        unsigned char ch = (unsigned char)buf->content[i];
        if (ch != '\\') {
            str[out++] = (char)ch;
            continue;
        }
        if (++i >= buf->offset) {
            cjson_free(str);
            return 0;
        }
        switch (buf->content[i]) {
            case '"': str[out++] = '"'; break;
            case '\\': str[out++] = '\\'; break;
            case '/': str[out++] = '/'; break;
            case 'b': str[out++] = '\b'; break;
            case 'f': str[out++] = '\f'; break;
            case 'n': str[out++] = '\n'; break;
            case 'r': str[out++] = '\r'; break;
            case 't': str[out++] = '\t'; break;
            case 'u': {
                int codepoint = 0;
                if (i + 4 >= buf->offset ||
                    !parse_hex4(buf->content + i + 1, &codepoint)) {
                    cjson_free(str);
                    return 0;
                }
                i += 4;
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    int low = 0;
                    if (i + 6 >= buf->offset ||
                        buf->content[i + 1] != '\\' ||
                        buf->content[i + 2] != 'u' ||
                        !parse_hex4(buf->content + i + 3, &low) ||
                        low < 0xdc00 || low > 0xdfff) {
                        cjson_free(str);
                        return 0;
                    }
                    i += 6;
                    codepoint = 0x10000 + (((codepoint - 0xd800) << 10) |
                                           (low - 0xdc00));
                }
                if (!append_utf8(str, &out, codepoint)) {
                    cjson_free(str);
                    return 0;
                }
                break;
            }
            default:
                cjson_free(str);
                return 0;
        }
    }
    str[out] = '\0';
    item->type = (1 << 5);
    item->valuestring = str;
    buf->offset++;
    return 1;
}

static int parse_number(cJSON *item, parse_buffer *buf) {
    int start = buf->offset;
    if (buf->content[buf->offset] == '-') buf->offset++;
    while (buf->offset < buf->length && buf->content[buf->offset] >= '0' && buf->content[buf->offset] <= '9')
        buf->offset++;
    if (buf->offset < buf->length && buf->content[buf->offset] == '.') {
        buf->offset++;
        while (buf->offset < buf->length && buf->content[buf->offset] >= '0' && buf->content[buf->offset] <= '9')
            buf->offset++;
    }
    if (buf->offset < buf->length && (buf->content[buf->offset] == 'e' || buf->content[buf->offset] == 'E')) {
        buf->offset++;
        if (buf->offset < buf->length && (buf->content[buf->offset] == '+' || buf->content[buf->offset] == '-'))
            buf->offset++;
        while (buf->offset < buf->length && buf->content[buf->offset] >= '0' && buf->content[buf->offset] <= '9')
            buf->offset++;
    }
    int len = buf->offset - start;
    char *numstr = (char *)cjson_malloc(len + 1);
    if (!numstr) return 0;
    memcpy(numstr, buf->content + start, len);
    numstr[len] = '\0';
    item->type = (1 << 4);
    item->valuedouble = strtod(numstr, NULL);
    item->valueint = (int)item->valuedouble;
    cjson_free(numstr);
    return 1;
}

static int parse_array(cJSON *item, parse_buffer *buf) {
    if (buf->content[buf->offset] != '[') return 0;
    buf->offset++;
    item->type = (1 << 6);
    buffer_skip_whitespace(buf);
    if (buf->offset < buf->length && buf->content[buf->offset] == ']') {
        buf->offset++;
        return 1;
    }
    cJSON *child = cjson_new_item();
    if (!child) return 0;
    item->child = child;
    if (!parse_value(child, buf)) { cjson_free(child); item->child = NULL; return 0; }
    buffer_skip_whitespace(buf);
    while (buf->offset < buf->length && buf->content[buf->offset] == ',') {
        buf->offset++;
        cJSON *new_item = cjson_new_item();
        if (!new_item) return 0;
        child->next = new_item;
        new_item->prev = child;
        child = new_item;
        if (!parse_value(child, buf)) return 0;
        buffer_skip_whitespace(buf);
    }
    if (buf->offset < buf->length && buf->content[buf->offset] == ']') {
        buf->offset++;
        return 1;
    }
    return 0;
}

static int parse_object(cJSON *item, parse_buffer *buf) {
    if (buf->content[buf->offset] != '{') return 0;
    buf->offset++;
    item->type = (1 << 7);
    buffer_skip_whitespace(buf);
    if (buf->offset < buf->length && buf->content[buf->offset] == '}') {
        buf->offset++;
        return 1;
    }
    cJSON *child = cjson_new_item();
    if (!child) return 0;
    item->child = child;
    buffer_skip_whitespace(buf);
    if (!parse_string(child, buf)) { cjson_free(child); item->child = NULL; return 0; }
    child->string = child->valuestring;
    child->valuestring = NULL;
    buffer_skip_whitespace(buf);
    if (buf->offset >= buf->length || buf->content[buf->offset] != ':') return 0;
    buf->offset++;
    if (!parse_value(child, buf)) return 0;
    buffer_skip_whitespace(buf);
    while (buf->offset < buf->length && buf->content[buf->offset] == ',') {
        buf->offset++;
        cJSON *new_item = cjson_new_item();
        if (!new_item) return 0;
        child->next = new_item;
        new_item->prev = child;
        child = new_item;
        buffer_skip_whitespace(buf);
        if (!parse_string(child, buf)) return 0;
        child->string = child->valuestring;
        child->valuestring = NULL;
        buffer_skip_whitespace(buf);
        if (buf->offset >= buf->length || buf->content[buf->offset] != ':') return 0;
        buf->offset++;
        if (!parse_value(child, buf)) return 0;
        buffer_skip_whitespace(buf);
    }
    if (buf->offset < buf->length && buf->content[buf->offset] == '}') {
        buf->offset++;
        return 1;
    }
    return 0;
}

cJSON *cJSON_Parse(const char *value) {
    if (!value) return NULL;
    parse_buffer buf = {value, (int)strlen(value), 0};
    cJSON *item = cjson_new_item();
    if (!item) return NULL;
    if (!parse_value(item, &buf)) {
        cJSON_Delete(item);
        return NULL;
    }
    return item;
}

cJSON *cJSON_GetObjectItem(cJSON *object, const char *string) {
    if (!object || !string) return NULL;
    cJSON *child = object->child;
    while (child) {
        if (child->string && strcmp(child->string, string) == 0) return child;
        child = child->next;
    }
    return NULL;
}

int cJSON_GetArraySize(cJSON *array) {
    if (!array) return 0;
    int size = 0;
    cJSON *child = array->child;
    while (child) { size++; child = child->next; }
    return size;
}

cJSON *cJSON_GetArrayItem(cJSON *array, int index) {
    if (!array) return NULL;
    cJSON *child = array->child;
    while (child && index > 0) { child = child->next; index--; }
    return child;
}

static int print_value(cJSON *item, char **out, int *out_len, int *out_size);

static int ensure_capacity(char **out, int *out_len, int *out_size, int needed) {
    if (*out_len + needed >= *out_size) {
        int new_size = (*out_size + needed) * 2;
        char *new_out = (char *)realloc(*out, new_size);
        if (!new_out) return 0;
        *out = new_out;
        *out_size = new_size;
    }
    return 1;
}

static int print_string(const char *str, char **out, int *out_len, int *out_size) {
    static const char hex[] = "0123456789abcdef";
    int len = (int)strlen(str);
    if (!ensure_capacity(out, out_len, out_size, len + 3)) return 0;
    (*out)[(*out_len)++] = '"';
    for (int i = 0; i < len; ++i) {
        unsigned char ch = (unsigned char)str[i];
        if (ch == '"' || ch == '\\') {
            if (!ensure_capacity(out, out_len, out_size, 3)) return 0;
            (*out)[(*out_len)++] = '\\';
            (*out)[(*out_len)++] = (char)ch;
        } else if (ch == '\b' || ch == '\f' || ch == '\n' || ch == '\r' || ch == '\t') {
            char esc = ch == '\b' ? 'b' :
                       ch == '\f' ? 'f' :
                       ch == '\n' ? 'n' :
                       ch == '\r' ? 'r' : 't';
            if (!ensure_capacity(out, out_len, out_size, 3)) return 0;
            (*out)[(*out_len)++] = '\\';
            (*out)[(*out_len)++] = esc;
        } else if (ch < 0x20) {
            if (!ensure_capacity(out, out_len, out_size, 7)) return 0;
            (*out)[(*out_len)++] = '\\';
            (*out)[(*out_len)++] = 'u';
            (*out)[(*out_len)++] = '0';
            (*out)[(*out_len)++] = '0';
            (*out)[(*out_len)++] = hex[(ch >> 4) & 0x0f];
            (*out)[(*out_len)++] = hex[ch & 0x0f];
        } else {
            if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
            (*out)[(*out_len)++] = (char)ch;
        }
    }
    (*out)[(*out_len)++] = '"';
    (*out)[*out_len] = '\0';
    return 1;
}

static int print_number(double num, char **out, int *out_len, int *out_size) {
    if (!ensure_capacity(out, out_len, out_size, 64)) return 0;
    int n;
    if (num == (double)(int)num && num <= INT_MAX && num >= INT_MIN)
        n = sprintf(*out + *out_len, "%d", (int)num);
    else
        n = sprintf(*out + *out_len, "%.17g", num);
    *out_len += n;
    return 1;
}

static int print_value(cJSON *item, char **out, int *out_len, int *out_size) {
    if (!item) return 0;
    if (cJSON_IsString(item) && item->valuestring)
        return print_string(item->valuestring, out, out_len, out_size);
    if (cJSON_IsNumber(item))
        return print_number(item->valuedouble, out, out_len, out_size);
    if (cJSON_IsBool(item)) {
        if (!ensure_capacity(out, out_len, out_size, 6)) return 0;
        int n;
        if (cJSON_IsTrue(item)) n = sprintf(*out + *out_len, "true");
        else n = sprintf(*out + *out_len, "false");
        *out_len += n;
        return 1;
    }
    if (cJSON_IsNull(item)) {
        if (!ensure_capacity(out, out_len, out_size, 5)) return 0;
        memcpy(*out + *out_len, "null", 4);
        *out_len += 4;
        (*out)[*out_len] = '\0';
        return 1;
    }
    if (cJSON_IsArray(item)) {
        if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
        (*out)[(*out_len)++] = '[';
        cJSON *child = item->child;
        while (child) {
            if (!print_value(child, out, out_len, out_size)) return 0;
            child = child->next;
            if (child) {
                if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
                (*out)[(*out_len)++] = ',';
            }
        }
        if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
        (*out)[(*out_len)++] = ']';
        (*out)[*out_len] = '\0';
        return 1;
    }
    if (cJSON_IsObject(item)) {
        if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
        (*out)[(*out_len)++] = '{';
        cJSON *child = item->child;
        while (child) {
            if (!print_string(child->string ? child->string : "", out, out_len, out_size)) return 0;
            if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
            (*out)[(*out_len)++] = ':';
            if (!print_value(child, out, out_len, out_size)) return 0;
            child = child->next;
            if (child) {
                if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
                (*out)[(*out_len)++] = ',';
            }
        }
        if (!ensure_capacity(out, out_len, out_size, 2)) return 0;
        (*out)[(*out_len)++] = '}';
        (*out)[*out_len] = '\0';
        return 1;
    }
    return 0;
}

char *cJSON_Print(cJSON *item) {
    int out_size = 256;
    int out_len = 0;
    char *out = (char *)cjson_malloc(out_size);
    if (!out) return NULL;
    out[0] = '\0';
    if (!print_value(item, &out, &out_len, &out_size)) {
        cjson_free(out);
        return NULL;
    }
    return out;
}

cJSON *cJSON_CreateObject(void) {
    cJSON *item = cjson_new_item();
    if (item) item->type = (1 << 7);
    return item;
}

cJSON *cJSON_CreateArray(void) {
    cJSON *item = cjson_new_item();
    if (item) item->type = (1 << 6);
    return item;
}

cJSON *cJSON_CreateString(const char *string) {
    cJSON *item = cjson_new_item();
    if (item) {
        item->type = (1 << 5);
        item->valuestring = cjson_strdup(string);
    }
    return item;
}

cJSON *cJSON_CreateNumber(double num) {
    cJSON *item = cjson_new_item();
    if (item) {
        item->type = (1 << 4);
        item->valuedouble = num;
        item->valueint = (int)num;
    }
    return item;
}

cJSON *cJSON_CreateBool(int b) {
    cJSON *item = cjson_new_item();
    if (item) {
        item->type = b ? (1 << 2) : (1 << 1);
        item->valueint = b ? 1 : 0;
        item->valuedouble = b ? 1.0 : 0.0;
    }
    return item;
}

cJSON *cJSON_CreateNull(void) {
    cJSON *item = cjson_new_item();
    if (item) item->type = (1 << 3);
    return item;
}

int cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item) {
    if (!object || !item) return 0;
    item->string = cjson_strdup(string);
    cJSON *child = object->child;
    if (!child) {
        object->child = item;
    } else {
        while (child->next) child = child->next;
        child->next = item;
        item->prev = child;
    }
    return 1;
}

int cJSON_AddItemToArray(cJSON *array, cJSON *item) {
    if (!array || !item) return 0;
    cJSON *child = array->child;
    if (!child) {
        array->child = item;
    } else {
        while (child->next) child = child->next;
        child->next = item;
        item->prev = child;
    }
    return 1;
}
