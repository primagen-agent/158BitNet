#!/bin/sh
# Cross-build the unified OpenAI server for an Android device (bionic).
#
#   scripts/build_server_host.sh [output]        # default build-bonsai/bonsai_server
#   adb -s PHONE push build-bonsai/bonsai_server /data/local/tmp/
#
# The board builds natively (scripts/build_server_board.sh); a phone has no
# compiler, so this uses the NDK clang.  -lpthread is not a separate library
# on bionic and the NDK sysroot carries the Vulkan headers, so the link line
# is the on-board one minus pthread.  Keep -march=armv8.2-a+dotprod: without
# it the TQ2_0 kernels fall back to their scalar tier (~5x slower decode).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-/tmp/dl/android-ndk-r27b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}
OUT=${1:-$ROOT/build-bonsai/bonsai_server}
mkdir -p "$(dirname "$OUT")"
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
    -fomit-frame-pointer -Wall -Wextra -Wno-unused-parameter \
    -DBITNET_ENABLE_VULKAN=1 \
    -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/src/vk" \
    -o "$OUT" "$ROOT/tools/bonsai_server.c" \
    "$ROOT/src/third_party/mongoose/mongoose.c" "$ROOT/src/third_party/cJSON/cJSON.c" \
    $SRC -lvulkan -lm
echo "built $OUT"
