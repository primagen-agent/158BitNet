#include "bitnet_dispatch.h"
#include "cpu_detect.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Per-tier tables — defined in kernel_registry.c. */
extern bitnet_dispatch_t g_dispatch_scalar;
#if defined(__ARM_NEON)
extern bitnet_dispatch_t g_dispatch_arm_neon;
#endif
#if defined(__x86_64__) || defined(_M_X64)
extern bitnet_dispatch_t g_dispatch_avx2;
extern bitnet_dispatch_t g_dispatch_avx_vnni;
extern bitnet_dispatch_t g_dispatch_avx512_vnni;
#endif

bitnet_dispatch_t *g_bitnet_dispatch = NULL;

static pthread_once_t g_dispatch_once = PTHREAD_ONCE_INIT;

static bitnet_cpu_tier_t resolve_tier_with_override(int *quiet) {
    *quiet = 0;
    const char *q = getenv("BITNET_QUIET");
    if (q && strcmp(q, "1") == 0) *quiet = 1;

    bitnet_cpu_tier_t auto_tier = bitnet_cpu_pick_tier();
    const char *override = getenv("BITNET_CPU_TIER");
    if (override == NULL || override[0] == '\0') return auto_tier;

    bitnet_cpu_tier_t requested = bitnet_cpu_tier_from_string(override);
    if (requested == BITNET_TIER_INVALID) {
        fprintf(stderr,
                "[bitnet] unrecognized BITNET_CPU_TIER='%s'; valid: "
                "scalar|avx2|avx_vnni|avx512_vnni. Falling back to auto-detect.\n",
                override);
        return auto_tier;
    }
    /* Safety: don't allow forcing a tier the CPU can't actually run. */
    bitnet_cpu_tier_t max_safe = auto_tier;
    if (requested > max_safe) {
        fprintf(stderr,
                "[bitnet] BITNET_CPU_TIER='%s' requested but CPU only supports "
                "tier '%s'. Falling back to auto-detect.\n",
                override, bitnet_cpu_tier_name(max_safe));
        return max_safe;
    }
    return requested;
}

static void dispatch_init_impl(void) {
    int quiet = 0;
    bitnet_cpu_tier_t tier = resolve_tier_with_override(&quiet);

#if defined(__ARM_NEON)
    /* ARM builds always use the NEON table regardless of the detected
     * "x86 tier" — the scalar tier is the only one available on ARM
     * from the bitnet_cpu_pick_tier perspective, but we override here. */
    g_bitnet_dispatch = &g_dispatch_arm_neon;
    if (!quiet) {
        fprintf(stderr, "[bitnet] cpu tier: arm_neon\n");
    }
    (void)tier;
    return;
#else
    switch (tier) {
        case BITNET_TIER_SCALAR:
            g_bitnet_dispatch = &g_dispatch_scalar;
            break;
        case BITNET_TIER_AVX2:
            g_bitnet_dispatch = &g_dispatch_avx2;
            break;
        case BITNET_TIER_AVX_VNNI:
            g_bitnet_dispatch = &g_dispatch_avx_vnni;
            break;
        case BITNET_TIER_AVX512_VNNI:
            g_bitnet_dispatch = &g_dispatch_avx512_vnni;
            break;
        default:
            g_bitnet_dispatch = &g_dispatch_scalar;
            break;
    }
    if (!quiet) {
        fprintf(stderr, "[bitnet] cpu tier: %s\n",
                bitnet_cpu_tier_name(g_bitnet_dispatch->tier));
    }
#endif
}

void bitnet_dispatch_init(void) {
    pthread_once(&g_dispatch_once, dispatch_init_impl);
}
