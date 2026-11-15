#!/bin/sh
# Cross-build the speculative-decoding driver on the host for an Android device.
#
#   scripts/build_spec_host.sh [output]        # default build-bonsai/bonsai_spec
#
# The board has no compiler, so this is the usual way to get the binary onto it:
#   adb push build-bonsai/bonsai_spec /data/local/tmp/
#   adb shell 'cd /data/local/tmp && ./bonsai_spec -m M.gguf --spec -d D.safetensors -B 3 -n 32 -t 6'
#
# Set CC to another aarch64-linux-android<api>-clang, or to a bionic zig cc.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-/tmp/dl/android-ndk-r27b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}
OUT=${1:-$ROOT/build-bonsai/bonsai_spec}
mkdir -p "$(dirname "$OUT")"
# -lpthread is not a separate library on bionic; -DBITNET_ENABLE_VULKAN pulls in
# the matvec/matmat pipelines and the PTQ block matmul's GPU share.
# One binary, both families: bonsai.c carries qwen35 itself and dispatches
# minicpm files to minicpm.c, which needs the GGUF/tokenizer/quant layers and
# the *neutral* bitnet kernels (ops, q4k/q6k/q8k/tq2_0, dispatch registry).
# bitnet.c is linked for the *_impl bodies the registry shims bind to.
SRC="$ROOT/src/bonsai.c $ROOT/src/minicpm.c $ROOT/src/quant_ptq1_0.c
     $ROOT/src/quant_tq2_0.c $ROOT/src/quant_q4k.c $ROOT/src/quant_q6k.c
     $ROOT/src/quant_q8k.c $ROOT/src/ops.c $ROOT/src/sampler.c
     $ROOT/src/bitnet.c $ROOT/src/bitnet_dispatch.c $ROOT/src/cpu_detect.c
     $ROOT/src/x86/kernel_registry.c $ROOT/src/x86/ops_x86.c
     $ROOT/src/x86/bitnet_hotpath_x86.c $ROOT/src/x86/quant_tq2_0_x86.c
     $ROOT/src/x86/quant_q6k_x86.c $ROOT/src/tokenizer.c $ROOT/src/gguf.c
     $ROOT/src/tensor.c $ROOT/src/thread_config.c $ROOT/src/unicode_data.c
     $ROOT/src/dspine.c $ROOT/src/vk/vk_backend.c"
$CC -O3 -march=armv8.2-a+dotprod -ffp-contract=fast -fno-math-errno \
    -fomit-frame-pointer -Wall -Wextra -DBITNET_ENABLE_VULKAN=1 \
    -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/src/vk" \
    -o "$OUT" "$ROOT/tools/bonsai_spec.c" $SRC \
    -lvulkan -lm
echo "built $OUT"
