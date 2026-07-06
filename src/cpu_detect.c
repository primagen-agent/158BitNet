#include "cpu_detect.h"
#include <string.h>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
/* GCC/Clang — use the built-in CPU detection, which already handles the
 * OS-level XGETBV check that confirms YMM/ZMM state is saved. */
bitnet_cpu_features_t bitnet_cpu_detect(void) {
    __builtin_cpu_init();
    bitnet_cpu_features_t f;
    memset(&f, 0, sizeof(f));
    f.has_sse3        = __builtin_cpu_supports("sse3");
    f.has_avx2        = __builtin_cpu_supports("avx2");
    f.has_fma         = __builtin_cpu_supports("fma");
    f.has_avx_vnni    = __builtin_cpu_supports("avxvnni");
    f.has_avx512f     = __builtin_cpu_supports("avx512f");
    f.has_avx512bw    = __builtin_cpu_supports("avx512bw");
    f.has_avx512vnni  = __builtin_cpu_supports("avx512vnni");
    return f;
}
#elif defined(_MSC_VER)
#include <intrin.h>
bitnet_cpu_features_t bitnet_cpu_detect(void) {
    bitnet_cpu_features_t f;
    memset(&f, 0, sizeof(f));
    int cpuinfo[4];
    __cpuid(cpuinfo, 0);
    int max_leaf = cpuinfo[0];
    if (max_leaf < 1) return f;

    __cpuid(cpuinfo, 1);
    f.has_sse3 = (cpuinfo[2] & (1 << 0)) != 0;       /* ECX bit 0 */
    int osxmm_ymm = ((_xgetbv(_XCR_XFEATURE_ENABLED_MASK) & 0x6) == 0x6);

    if (max_leaf >= 7) {
        __cpuidex(cpuinfo, 7, 0);
        f.has_avx2 = osxmm_ymm && (cpuinfo[1] & (1 << 5));    /* EBX bit 5 */
        f.has_avx512f    = osxmm_ymm && (cpuinfo[1] & (1 << 16));
        f.has_avx512bw   = osxmm_ymm && (cpuinfo[1] & (1 << 30));
        f.has_avx512vnni = osxmm_ymm && (cpuinfo[2] & (1 << 11)); /* ECX bit 11 */

        if (max_leaf >= 7) {
            __cpuidex(cpuinfo, 7, 1);
            f.has_avx_vnni = osxmm_ymm && (cpuinfo[0] & (1 << 4)); /* EAX bit 4 */
        }
    }
    /* FMA — leaf 1 ECX bit 12. */
    __cpuid(cpuinfo, 1);
    f.has_fma = osxmm_ymm && (cpuinfo[2] & (1 << 12));
    return f;
}
#else
bitnet_cpu_features_t bitnet_cpu_detect(void) {
    bitnet_cpu_features_t f;
    memset(&f, 0, sizeof(f));
    return f;
}
#endif

bitnet_cpu_tier_t bitnet_cpu_pick_tier(void) {
#if defined(__x86_64__) || defined(_M_X64)
    bitnet_cpu_features_t f = bitnet_cpu_detect();
    if (f.has_avx512f && f.has_avx512bw && f.has_avx512vnni) {
        return BITNET_TIER_AVX512_VNNI;
    }
    if (f.has_avx2 && f.has_avx_vnni && f.has_fma) {
        return BITNET_TIER_AVX_VNNI;
    }
    if (f.has_avx2 && f.has_fma) {
        return BITNET_TIER_AVX2;
    }
    return BITNET_TIER_SCALAR;
#else
    return BITNET_TIER_SCALAR;
#endif
}

const char *bitnet_cpu_tier_name(bitnet_cpu_tier_t tier) {
    switch (tier) {
        case BITNET_TIER_SCALAR:      return "scalar";
        case BITNET_TIER_AVX2:        return "avx2";
        case BITNET_TIER_AVX_VNNI:    return "avx_vnni";
        case BITNET_TIER_AVX512_VNNI: return "avx512_vnni";
        default: return NULL;
    }
}

bitnet_cpu_tier_t bitnet_cpu_tier_from_string(const char *name) {
    if (name == NULL) return BITNET_TIER_INVALID;
    if (strcmp(name, "scalar") == 0)      return BITNET_TIER_SCALAR;
    if (strcmp(name, "avx2") == 0)        return BITNET_TIER_AVX2;
    if (strcmp(name, "avx_vnni") == 0)    return BITNET_TIER_AVX_VNNI;
    if (strcmp(name, "avx512_vnni") == 0) return BITNET_TIER_AVX512_VNNI;
    return BITNET_TIER_INVALID;
}
