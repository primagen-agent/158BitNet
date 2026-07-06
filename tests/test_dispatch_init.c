#include "bitnet_dispatch.h"
#include "ops.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    /* Before any dispatch-routed call, the pointer is NULL. */
    assert(g_bitnet_dispatch == NULL);

    /* Calling any kernel that goes through dispatch triggers init. For
     * Phase 1, ops.c functions are not yet trampolined (Phase 2), so we
     * call bitnet_dispatch_init() explicitly here. */
    bitnet_dispatch_init();

    assert(g_bitnet_dispatch != NULL);
    assert(g_bitnet_dispatch->rms_norm_eps != NULL);
    assert(g_bitnet_dispatch->tq2_quantize_vec_i8 != NULL);

    printf("test_dispatch_init: OK (tier=%d)\n", g_bitnet_dispatch->tier);
    return 0;
}
