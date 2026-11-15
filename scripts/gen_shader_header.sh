#!/bin/sh
# Rebuild an embedded SPIR-V header after editing a compute shader.
#
#   scripts/gen_shader_header.sh src/vk/ptq_matvec.comp
#
# Writes src/vk/<name>_spv.h next to the shader; the C side includes it as a
# byte array so the runtime carries no external shader files.
set -e
src=${1:?usage: gen_shader_header.sh <shader.comp>}
dir=$(cd "$(dirname "$src")" && pwd)
base=$(basename "$src" .comp)
spv=${TMPDIR:-/tmp}/$$_$base.spv
hdr="$dir/${base}_spv.h"

# Either front end produces consumable SPIR-V; glslc also runs spirv-opt,
# which trims the loop epilogues the ternary decoders generate.
if command -v glslangValidator >/dev/null 2>&1; then
    glslangValidator -V "$src" -o "$spv" >/dev/null
elif command -v glslc >/dev/null 2>&1; then
    glslc -fshader-stage=compute -O "$src" -o "$spv"
elif [ -x /tmp/dl/glslc ]; then
    /tmp/dl/glslc -fshader-stage=compute -O "$src" -o "$spv"
else
    echo "gen_shader_header: need glslangValidator or glslc" >&2
    exit 1
fi

{
    printf '/* Generated from %s with glslangValidator -V.\n' "$src"
    printf ' * Do not edit; run scripts/gen_shader_header.sh after changing the shader. */\n'
    printf '#ifndef BITNET_VK_%s_SPV_H\n' "$(echo "$base" | tr 'a-z' 'A-Z')"
    printf '#define BITNET_VK_%s_SPV_H\n\n' "$(echo "$base" | tr 'a-z' 'A-Z')"
    printf '#include <stddef.h>\n#include <stdint.h>\n\n'
    printf 'static const uint8_t bq_vk_%s_spv[] = {\n' "$base"
    od -An -v -tu1 "$spv" | awk '{ for (i=1;i<=NF;i++) { printf "%s0x%02x,", (++n%12==1?"    ":" "), $i; if (n%12==0) printf "\n" } }'
    printf '\n};\n'
    printf 'static const size_t bq_vk_%s_spv_len = sizeof(bq_vk_%s_spv);\n\n' "$base" "$base"
    printf '#endif\n'
} > "$hdr"
rm -f "$spv"
echo "wrote $hdr"
