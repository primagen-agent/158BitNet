#include "metis/memory_snapshot.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed line %d: %s\n", __LINE__, #x);                                 \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
int main(void) {
    const char *base = "test-resident-snapshot";
    char ep[1200], ev[1200], rs[1200];
    resident_model_t model = {0}, wrong = {0};
    model.sha256[0] = 12;
    wrong.sha256[0] = 13;
    resident_state_t state = {0}, loaded = {0};
    metis_episodic_store_t episodes;
    metis_event_store_t events;
    metis_episodic_init(&episodes);
    metis_event_store_init(&events);
    CHECK(metis_episodic_add_with_priority(&episodes, "Morgan lives in Lima", NULL, 0) == 0);
    metis_event_record_t e = {0};
    e.event_id = "one";
    e.episode_id = "episode";
    e.source_id = "source";
    e.entity = "Morgan";
    e.predicate = "city";
    e.value = "Lima";
    e.valid_time = "";
    e.target_event_id = "";
    e.subject_start = 0;
    e.subject_end = 6;
    e.value_start = 16;
    e.value_end = 20;
    CHECK(metis_event_store_apply(&events, &e) == 0);
    state.count = 1;
    state.slots[0].tokens = 2;
    state.slots[0].data = calloc(2 * RESIDENT_STRIDE, sizeof(float));
    CHECK(state.slots[0].data != NULL);
    state.slots[0].data[2 * RESIDENT_STRIDE - 1] = 1;
    CHECK(metis_memory_snapshot_save_resident(base, &episodes, &events, &state, &model) == 0);
    CHECK(metis_memory_snapshot_paths(base, ep, sizeof ep, ev, sizeof ev) != 0);
    CHECK(metis_memory_snapshot_paths_resident(base, ep, sizeof ep, ev, sizeof ev, rs, sizeof rs) ==
          0);
    CHECK(resident_state_load(&loaded, &model, rs) == 0 && loaded.count == 1);
    float *previous = loaded.slots[0].data;
    CHECK(resident_state_load(&loaded, &wrong, rs) != 0 && loaded.slots[0].data == previous);
    state.slots[0].data[RESIDENT_STRIDE] = NAN;
    CHECK(metis_memory_snapshot_save_resident(base, &episodes, &events, &state, &model) != 0);
    CHECK(metis_memory_snapshot_paths_resident(base, ep, sizeof ep, ev, sizeof ev, rs, sizeof rs) ==
          0);
    CHECK(resident_state_load(&loaded, &model, rs) == 0);
    FILE *f = fopen(rs, "ab");
    CHECK(f != NULL);
    CHECK(fputc(42, f) != EOF);
    CHECK(fclose(f) == 0);
    previous = loaded.slots[0].data;
    CHECK(resident_state_load(&loaded, &model, rs) != 0 && loaded.slots[0].data == previous);
    CHECK(metis_memory_snapshot_paths_resident(base, ep, sizeof ep, ev, sizeof ev, rs, sizeof rs) !=
          0);
    remove(ep);
    remove(ev);
    remove(rs);
    remove("test-resident-snapshot.bnsnapshot");
    resident_state_clear(&state);
    resident_state_clear(&loaded);
    metis_episodic_free(&episodes);
    metis_event_store_clear(&events);
    puts("resident snapshot integrity passed");
    return 0;
}
