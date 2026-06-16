#ifndef CJSON_H
#define CJSON_H

#include <stddef.h>

typedef struct cJSON {
    struct cJSON *next;
    struct cJSON *prev;
    struct cJSON *child;
    int type;
    char *valuestring;
    int valueint;
    double valuedouble;
    char *string;
} cJSON;

typedef struct cJSON_Hooks {
    void *(*malloc_fn)(size_t sz);
    void (*free_fn)(void *ptr);
} cJSON_Hooks;

void cJSON_InitHooks(cJSON_Hooks *hooks);
cJSON *cJSON_Parse(const char *value);
void cJSON_Delete(cJSON *c);
char *cJSON_Print(cJSON *item);
cJSON *cJSON_GetObjectItem(cJSON *object, const char *string);
int cJSON_GetArraySize(cJSON *array);
cJSON *cJSON_GetArrayItem(cJSON *array, int index);
cJSON *cJSON_CreateObject(void);
cJSON *cJSON_CreateArray(void);
cJSON *cJSON_CreateString(const char *string);
cJSON *cJSON_CreateNumber(double num);
cJSON *cJSON_CreateBool(int b);
cJSON *cJSON_CreateNull(void);
int cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item);
int cJSON_AddItemToArray(cJSON *array, cJSON *item);

#define cJSON_IsInvalid(item) ((item)->type & (1 << 0))
#define cJSON_IsFalse(item)   ((item)->type & (1 << 1))
#define cJSON_IsTrue(item)    ((item)->type & (1 << 2))
#define cJSON_IsBool(item)    ((item)->type & ((1 << 1) | (1 << 2)))
#define cJSON_IsNull(item)    ((item)->type & (1 << 3))
#define cJSON_IsNumber(item)  ((item)->type & (1 << 4))
#define cJSON_IsString(item)  ((item)->type & (1 << 5))
#define cJSON_IsArray(item)   ((item)->type & (1 << 6))
#define cJSON_IsObject(item)  ((item)->type & (1 << 7))

#endif
