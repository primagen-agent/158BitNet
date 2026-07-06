#ifndef BITNET_CPU_DETECT_H
#define BITNET_CPU_DETECT_H

#include <stdint.h>

typedef enum {
    BITNET_TIER_SCALAR      = 0,
    BITNET_TIER_AVX2        = 1,
    BITNET_TIER_AVX_VNNI    = 2,
    BITNET_TIER_AVX512_VNNI = 3,
} bitnet_cpu_tier_t;

/* Sentinel returned by bitnet_cpu_tier_from_string on unrecognized input. */
#define BITNET_TIER_INVALID ((bitnet_cpu_tier_t)-1)

typedef struct {
    int has_sse3;
    int has_avx2;
    int has_fma;
    int has_avx_vnni;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512vnni;
} bitnet_cpu_features_t;

/* Detect CPU features available with current OS+process context. */
bitnet_cpu_features_t bitnet_cpu_detect(void);

/* Pick the best tier the detected features support. */
bitnet_cpu_tier_t bitnet_cpu_pick_tier(void);

/* Stable human-readable name for a tier ("scalar", "avx2", "avx_vnni",
 * "avx512_vnni"). Returns NULL for invalid tiers. */
const char *bitnet_cpu_tier_name(bitnet_cpu_tier_t tier);

/* Parse a tier name string. Returns BITNET_TIER_INVALID if unrecognized. */
bitnet_cpu_tier_t bitnet_cpu_tier_from_string(const char *name);

#endif
