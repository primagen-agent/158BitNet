#define main openai_server_main
#include "../examples/openai_server.c"
#undef main

#include <stdio.h>
#include <string.h>

static int expect_string(const char *label, const char *actual, const char *expected) {
    if (strcmp(actual, expected) != 0) {
        fprintf(stderr, "%s: expected '%s', got '%s'\n", label, expected, actual);
        return 1;
    }
    return 0;
}

static int test_bitnet_b158_prompt_system_folds_into_next_user(void) {
    static const char json[] =
        "{\"messages\":["
        "{\"role\":\"system\",\"content\":\"You are concise.\"},"
        "{\"role\":\"user\",\"content\":\"What is the capital of France?\"}"
        "]}";
    cJSON *root = cJSON_Parse(json);
    char *prompt = NULL;
    int failures = 0;

    if (root == NULL) {
        fprintf(stderr, "failed to parse prompt test json\n");
        return 1;
    }
    prompt = build_chat_prompt_bitnet_b158(root);
    if (prompt == NULL) {
        fprintf(stderr, "failed to build bitnet-b1.58 prompt\n");
        cJSON_Delete(root);
        return 1;
    }
    failures += expect_string(
        "bitnet-b1.58 system prompt",
        prompt,
        "Human: You are concise.\n\nWhat is the capital of France?\n\nBITNETAssistant: ");
    free(prompt);
    cJSON_Delete(root);
    return failures;
}

int main(void) {
    generation_state_t gen;
    char decoded[32];
    char error[128];
    int failures = 0;

    memset(&gen, 0, sizeof(gen));
    memset(error, 0, sizeof(error));

    strcpy(decoded, "Paris");
    if (generation_append_decoded_utf8(&gen, decoded, 5,
                                       decoded, sizeof(decoded),
                                       error, sizeof(error)) != 0) {
        fprintf(stderr, "ascii append failed: %s\n", error);
        return 1;
    }
    failures += expect_string("ascii chunk", decoded, "Paris");
    failures += expect_string("ascii text", gen.text, "Paris");
    free(gen.text);

    memset(&gen, 0, sizeof(gen));
    memset(error, 0, sizeof(error));

    decoded[0] = (char)0xE4;
    decoded[1] = '\0';
    if (generation_append_decoded_utf8(&gen, decoded, 1,
                                       decoded, sizeof(decoded),
                                       error, sizeof(error)) != 0) {
        fprintf(stderr, "utf8 lead append failed: %s\n", error);
        return 1;
    }
    failures += expect_string("utf8 pending chunk", decoded, "");

    decoded[0] = (char)0xB8;
    decoded[1] = (char)0xAD;
    decoded[2] = '\0';
    if (generation_append_decoded_utf8(&gen, decoded, 2,
                                       decoded, sizeof(decoded),
                                       error, sizeof(error)) != 0) {
        fprintf(stderr, "utf8 continuation append failed: %s\n", error);
        return 1;
    }
    failures += expect_string("utf8 complete chunk", decoded, "中");
    failures += expect_string("utf8 complete text", gen.text, "中");
    free(gen.text);

    failures += test_bitnet_b158_prompt_system_folds_into_next_user();

    return failures == 0 ? 0 : 1;
}
