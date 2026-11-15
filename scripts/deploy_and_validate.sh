#!/bin/sh
# Host side: push the tree to the target board and run the Vulkan validation
# there.  The host has no compiler, so every build happens on the device.
#
#   scripts/deploy_and_validate.sh [host:port] [model]
set -e
ADDR=${1:-192.168.70.125:5555}
MODEL=${2:-/data/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE=/workdir/AI/158BitNet

adb connect "$ADDR" >/dev/null 2>&1 || true
adb -s "$ADDR" wait-for-device
adb -s "$ADDR" shell 'mkdir -p '"$REMOTE"
tar -C "$ROOT" -cf - \
    --exclude=build-bonsai --exclude=.git --exclude='*.gguf' \
    . | adb -s "$ADDR" shell "tar -C $REMOTE -xf -"
echo "pushed to $ADDR:$REMOTE"
exec adb -s "$ADDR" shell "cd $REMOTE && sh scripts/validate_vk.sh '$MODEL'"
