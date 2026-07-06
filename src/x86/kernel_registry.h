#ifndef BITNET_KERNEL_REGISTRY_H
#define BITNET_KERNEL_REGISTRY_H

#include "bitnet_dispatch.h"

extern bitnet_dispatch_t g_dispatch_scalar;
#if defined(__ARM_NEON)
extern bitnet_dispatch_t g_dispatch_arm_neon;
#endif
#if defined(__x86_64__) || defined(_M_X64)
extern bitnet_dispatch_t g_dispatch_avx2;
extern bitnet_dispatch_t g_dispatch_avx_vnni;
extern bitnet_dispatch_t g_dispatch_avx512_vnni;
#endif

#endif
