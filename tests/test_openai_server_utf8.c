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
    {
        server_state_t state = {0};
        metis_memory_controller_t controller = {0};
        float projection = 0.0f;
        cJSON *request = cJSON_CreateObject();
        controller.action_projection = &projection;
        state.memory_controller = &controller;
        /* A missing backbone forces classification failure before any matmul. */
        if (request == NULL) return 1;
        if (request_memory_action(&state, request, "My name is Alex.") != MEMORY_ACTION_IGNORE ||
            request_memory_action(&state, request, "Please forget my address.") != MEMORY_ACTION_IGNORE) {
            fprintf(stderr, "failed learned gate fell back to a heuristic mutation\n");
            ++failures;
        }
        if (!cJSON_AddItemToObject(request, "memory_action", cJSON_CreateString("store"))) {
            cJSON_Delete(request);
            return 1;
        }
        if (request_memory_action(&state, request, "My name is Alex.") != MEMORY_ACTION_STORE) {
            fprintf(stderr, "explicit memory action no longer overrides the gate\n");
            ++failures;
        }
        cJSON_Delete(request);
    }
    {
        metis_event_record_t event = {0}, target = {0};
        event.entity = "new-entity";
        event.predicate = "new_property";
        event.operation = METIS_EVENT_SUPERSEDE;
        if (compile_autonomous_operation(&event, NULL, 0, MEMORY_ACTION_IGNORE) ||
            event.operation != METIS_EVENT_ASSERT || strcmp(event.target_event_id, "") ||
            strcmp(event.entity, "new-entity") || strcmp(event.predicate, "new_property")) {
            fprintf(stderr, "a first observed change was not preserved as a new fact\n");
            ++failures;
        }
        if (compile_autonomous_operation(&event, NULL, 0, MEMORY_ACTION_UPDATE) != -2 ||
            compile_autonomous_operation(&event, NULL, -1, MEMORY_ACTION_IGNORE) != -1 ||
            compile_autonomous_operation(&event, NULL, 1, MEMORY_ACTION_IGNORE) != -1) {
            fprintf(stderr, "explicit update or inference failure was silently accepted\n");
            ++failures;
        }
        target.event_id = "activated-old-version";
        target.entity = "old-entity";
        target.predicate = "old_property";
        if (compile_autonomous_operation(&event, &target, 1, MEMORY_ACTION_IGNORE) ||
            event.operation != METIS_EVENT_SUPERSEDE ||
            strcmp(event.target_event_id, target.event_id) ||
            strcmp(event.entity, target.entity) || strcmp(event.predicate, target.predicate)) {
            fprintf(stderr, "activated predecessor was not linked\n");
            ++failures;
        }
    }
    {
        const char *word = "caf\xC3\xA9teria";
        size_t offsets[] = {0, 3};
        size_t start = 99, end = 99;
        if (typed_token_span_to_source(word, offsets, 1, 0, word,
                0, 0, 2, &start, &end) || start != 0 || end != strlen(word)) {
            fprintf(stderr, "value boundary truncated a UTF-8 word\n");
            ++failures;
        }
        word = "Mira's";
        offsets[1] = 4;
        if (typed_token_span_to_source(word, offsets, 1, 0, word,
                0, 0, 2, &start, &end) || end != 6 ||
            typed_token_span_to_source(word, offsets, 1, 0, word,
                0, 0, 1, &start, &end) || end != 4) {
            fprintf(stderr, "entity and value possessive handling diverged\n");
            ++failures;
        }
        word = "北京天气";
        offsets[1] = 6;
        if (typed_token_span_to_source(word, offsets, 1, 0, word,
                0, 0, 2, &start, &end) || start != 0 || end != 6) {
            fprintf(stderr, "word completion consumed unrelated CJK characters\n");
            ++failures;
        }
    }

    {
        const char *queries[] = {"Give both entries for Morgan.",
            "List the recorded preferences.", "Compare earlier and current cities.",
            "Report the address.", "State the current name.", "I need the old and new values."};
        for (size_t i=0;i<sizeof queries/sizeof queries[0];i++) {
            if(!typed_auto_input_is_query(queries[i]) ||
               typed_auto_fallback_action(queries[i])!=MEMORY_ACTION_IGNORE) {
                fprintf(stderr,"imperative query was classified as a write\n");++failures;
            }
        }
        if(typed_auto_input_is_query("Morgan's workplace is Kestrel Labs."))++failures;
    }
    return failures == 0 ? 0 : 1;
}
