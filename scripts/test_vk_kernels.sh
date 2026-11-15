#!/bin/sh
# Build and run the Vulkan kernel tests on the machine you are sitting at.
#
#   scripts/test_vk_kernels.sh
#
# Both tests compare every GPU kernel against an independent CPU reference and
# exit non-zero on a mismatch, so they need a working Vulkan device but not a
# model file.  With no GPU in the box, Mesa's lavapipe is enough:
#
#   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json scripts/test_vk_kernels.sh
#
# The build needs Vulkan headers and a loader.  If they are not in the default
# search path, point CFLAGS at them and LOADER at the library:
#
#   CC="zig cc" CFLAGS="-I/tmp/vkhdr" \
#     LOADER=/usr/lib/x86_64-linux-gnu/libvulkan.so.1 scripts/test_vk_kernels.sh
#
# For a cross build, set CC to the cross compiler and OUT to a directory you
# can copy to the device, then run the binaries there instead.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-cc}
OUT=${OUT:-${TMPDIR:-/tmp}/bq_vk_tests}
CFLAGS=${CFLAGS:-}
LOADER=${LOADER:-}

mkdir -p "$OUT"

# Prefer the unversioned symlink; Debian ships only .so.1 without libvulkan-dev.
if [ -z "$LOADER" ]; then
    if echo 'int main(void){return 0;}' | $CC -x c - -o /dev/null -lvulkan 2>/dev/null; then
        LOADER=-lvulkan
    else
        for l in /usr/lib/*/libvulkan.so.1 /lib/*/libvulkan.so.1; do
            [ -e "$l" ] && { LOADER=$l; break; }
        done
    fi
fi
[ -n "$LOADER" ] || { echo "no Vulkan loader; set LOADER=/path/to/libvulkan.so" >&2; exit 1; }

INC="-I$ROOT/include -I$ROOT/src -I$ROOT/src/vk"
DEFS="-DBITNET_ENABLE_VULKAN=1"
# gnu11 rather than c11: bonsai.c uses clock_gettime, which POSIX c11 hides.
STD="-std=gnu11"
COMMON="-O2 -Wall -Wextra $STD $DEFS $CFLAGS $INC"

build() {
    name=$1; src=$2; shift 2
    echo "--- build $name"
    # shellcheck disable=SC2086
    $CC $COMMON -o "$OUT/$name" "$src" "$@"
}

build test_vk_kernels "$ROOT/tests/test_vk_kernels.c" \
      "$ROOT/src/vk/vk_backend.c" -lm "$LOADER"
build test_vk_hybrid "$ROOT/tests/test_vk_hybrid.c" \
      "$ROOT/src/vk/vk_backend.c" "$ROOT/src/quant_ptq1_0.c" \
      "$ROOT/src/unicode_data.c" "$ROOT/src/gguf.c" -lm "$LOADER"

run() {
    echo "=== $1"
    "$OUT/$1" || { echo "FAIL: $1" >&2; exit 1; }
}

run test_vk_kernels
run test_vk_hybrid
echo "all Vulkan kernel tests passed"
