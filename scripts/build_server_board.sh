#!/bin/sh
# On-board build of the unified OpenAI-compatible server and its context probe.
# The board has gcc, so nothing is cross-compiled: this script assumes the
# source tree lives at /data/bq (scripts/sync_board_src.sh refreshes it) and
# writes $ROOT/bonsai_server and $ROOT/ctx_probe.
#
#   adb -s 192.168.70.120:5555 push scripts/build_server_board.sh /data/bq/scripts/
#   adb -s 192.168.70.120:5555 shell sh /data/bq/scripts/build_server_board.sh
set -e
ROOT=${ROOT:-/data/bq}
OUT=${1:-bonsai_server}
SRC="$ROOT/src/bonsai.c $ROOT/src/minicpm.c $ROOT/src/quant_ptq1_0.c
     $ROOT/src/quant_tq2_0.c $ROOT/src/quant_q4k.c $ROOT/src/quant_q6k.c
     $ROOT/src/quant_q8k.c $ROOT/src/ops.c $ROOT/src/sampler.c
     $ROOT/src/bitnet.c $ROOT/src/bitnet_dispatch.c $ROOT/src/cpu_detect.c
     $ROOT/src/x86/kernel_registry.c $ROOT/src/x86/ops_x86.c
     $ROOT/src/x86/bitnet_hotpath_x86.c $ROOT/src/x86/quant_tq2_0_x86.c
     $ROOT/src/x86/quant_q6k_x86.c $ROOT/src/tokenizer.c $ROOT/src/gguf.c
     $ROOT/src/tensor.c $ROOT/src/thread_config.c $ROOT/src/unicode_data.c
     $ROOT/src/dspine.c $ROOT/src/vk/vk_backend.c"
CFLAGS="-O3 -march=armv8.2-a+dotprod -ffp-contract=fast -fno-math-errno
        -fomit-frame-pointer -Wall -Wextra -Wno-unused-parameter -DBITNET_ENABLE_VULKAN=1
        -I$ROOT/include -I$ROOT/src -I$ROOT/src/vk"
gcc $CFLAGS -o "$ROOT/$OUT" "$ROOT/tools/bonsai_server.c" \
    "$ROOT/src/third_party/mongoose/mongoose.c" "$ROOT/src/third_party/cJSON/cJSON.c" \
    $SRC -lvulkan -lm -lpthread
echo "built $OUT"
# Same sources, one more main: reports the context a model really allows.
gcc $CFLAGS -o "$ROOT/ctx_probe" "$ROOT/tools/ctx_probe.c" $SRC -lvulkan -lm -lpthread
echo "built ctx_probe"
