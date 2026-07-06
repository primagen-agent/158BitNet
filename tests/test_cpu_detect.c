#include "cpu_detect.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

int main(void) {
#if defined(__x86_64__) || defined(_M_X64)
    /* All x86-64 CPUs since 2003 support SSE3 — the table-lookup primitive. */
    bitnet_cpu_features_t f = bitnet_cpu_detect();
    assert(f.has_sse3 && "SSE3 must be present on x86-64");
    (void)f;
#endif

    bitnet_cpu_tier_t t = bitnet_cpu_pick_tier();
    assert(t >= BITNET_TIER_SCALAR && t <= BITNET_TIER_AVX512_VNNI);

    /* Tier name round-trips through from_string. */
    const char *name = bitnet_cpu_tier_name(t);
    assert(name != NULL && strlen(name) > 0);
    assert(bitnet_cpu_tier_from_string(name) == t);

    /* Invalid override string returns -1 (caller falls back to auto). */
    assert(bitnet_cpu_tier_from_string("nonsense") == (bitnet_cpu_tier_t)-1);

    /* Specific valid strings map to their tiers. */
    assert(bitnet_cpu_tier_from_string("scalar") == BITNET_TIER_SCALAR);
    assert(bitnet_cpu_tier_from_string("avx2") == BITNET_TIER_AVX2);
    assert(bitnet_cpu_tier_from_string("avx_vnni") == BITNET_TIER_AVX_VNNI);
    assert(bitnet_cpu_tier_from_string("avx512_vnni") == BITNET_TIER_AVX512_VNNI);

    printf("test_cpu_detect: OK (tier=%s)\n", name);
    return 0;
}
