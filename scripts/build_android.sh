#!/usr/bin/env sh
set -eu

if [ -z "${ANDROID_NDK:-}" ]; then
    echo "ERROR: ANDROID_NDK is not set" >&2
    echo "Example: ANDROID_NDK=/path/to/android-ndk ./scripts/build_android.sh" >&2
    exit 1
fi

TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "ERROR: Android toolchain not found: $TOOLCHAIN_FILE" >&2
    exit 1
fi

ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_API="${ANDROID_API:-24}"
BUILD_DIR="${BUILD_DIR:-build/android-$ANDROID_ABI}"

if [ "$#" -gt 0 ]; then
    TARGETS="$*"
else
    TARGETS="minimal_generate gguf_inspect"
fi

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="android-$ANDROID_API" \
    -DBITNET_BUILD_TESTS=OFF

for target in $TARGETS; do
    cmake --build "$BUILD_DIR" --target "$target"
done

echo "Android binaries are in: $BUILD_DIR"
