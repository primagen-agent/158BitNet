#!/bin/sh
# Build the Qwen3.5-hybrid "Ternary-Bonsai" runtime and its CLI on the target
# machine (used on the phone, where cmake is not installed).
#
#   scripts/build_bonsai.sh [outdir]
#
# Extra sources (e.g. the Vulkan backend) are picked up automatically.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/build-bonsai}
CC=${CC:-gcc}
# The dotprod flag is not cosmetic: without it the TQ2_0 kernels compile to
# their scalar tier and the same model decodes about 5x slower.  Cross-building
# from an x86 host would otherwise lose it, so look at the compiler target too.
ARCH_FLAGS=${ARCH_FLAGS:-}
if [ -z "$ARCH_FLAGS" ]; then
    case "$(uname -m) $CC" in
        *aarch64*|*arm64*) ARCH_FLAGS="-march=armv8.2-a+dotprod" ;;
    esac
fi

# One binary, both families: bonsai.c is the qwen35 graph and hands minicpm
# files to minicpm.c, which needs the neutral kernels as well.
SRC="$ROOT/src/bonsai.c $ROOT/src/minicpm.c $ROOT/src/gguf.c $ROOT/src/tensor.c
     $ROOT/src/tokenizer.c $ROOT/src/quant_ptq1_0.c $ROOT/src/quant_tq2_0.c
     $ROOT/src/quant_q4k.c $ROOT/src/quant_q6k.c $ROOT/src/quant_q8k.c
     $ROOT/src/ops.c $ROOT/src/sampler.c $ROOT/src/thread_config.c
     $ROOT/src/bitnet.c $ROOT/src/bitnet_dispatch.c $ROOT/src/cpu_detect.c
     $ROOT/src/x86/kernel_registry.c $ROOT/src/x86/ops_x86.c
     $ROOT/src/x86/bitnet_hotpath_x86.c $ROOT/src/x86/quant_tq2_0_x86.c
     $ROOT/src/x86/quant_q6k_x86.c $ROOT/src/unicode_data.c"
VK_SRC=""
VK_LIBS=""
VK_DEF=""
# bionic has no separate libpthread; cross-building with an NDK clang would
# fail the link over a library that does not exist there.
LIBS="-lpthread"
case "$CC" in
    *android*) LIBS="" ;;
esac

# The Vulkan backend carries its shaders as generated SPIR-V headers, so the
# build refreshes any that are missing or older than their .comp source.
# Only the matvec shader is required: the attention and LUT kernels are picked
# up when their headers exist, so a stray .comp that fails to compile must not
# disable the whole backend.
VK_SHADERS="ptq_matvec"
shaders_ready() {
    for s in $VK_SHADERS; do
        [ -f "$ROOT/src/vk/${s}_spv.h" ] || return 1
    done
    return 0
}

if command -v glslangValidator >/dev/null 2>&1; then
    for f in "$ROOT"/src/vk/*.comp; do
        [ -f "$f" ] || continue
        base=$(basename "$f" .comp)
        hdr="$ROOT/src/vk/${base}_spv.h"
        if [ ! -f "$hdr" ] || [ "$f" -nt "$hdr" ]; then
            # A shader that fails here must not take the whole build with it:
            # report it and let the shader check below decide about the backend.
            "$ROOT/scripts/gen_shader_header.sh" "$f" >/dev/null 2>&1 ||
                echo "warning: $f failed to compile" >&2
        fi
    done
fi

if [ -f "$ROOT/src/vk/vk_backend.c" ] && [ -f /usr/include/vulkan/vulkan.h ]; then
    if shaders_ready; then
        VK_SRC="$ROOT/src/vk/vk_backend.c"
        VK_LIBS="-lvulkan"
        VK_DEF="-DBITNET_ENABLE_VULKAN=1"
    else
        echo "warning: shader headers missing, building without the Vulkan backend" >&2
    fi
fi

mkdir -p "$OUT"
$CC -O3 $ARCH_FLAGS -ffp-contract=fast -fno-math-errno -fomit-frame-pointer \
    -Wall -Wextra -Wno-unused-parameter $VK_DEF \
    -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/src/vk" \
    -o "$OUT/bonsai_generate" \
    "$ROOT/tools/bonsai_generate.c" $SRC $VK_SRC \
    $LIBS -lm $VK_LIBS
echo "built $OUT/bonsai_generate"
