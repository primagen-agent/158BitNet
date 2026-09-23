/* Isolated instrumented build of the actual runtime, not a reimplementation.
 * bitnet.c's observers compile away in libbitnet and production executables. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *trace_stream;
static int trace_error;
static uint32_t trace_case = UINT32_MAX;

static void trace_float(int stage, int layer, int token, const float *data, int size) {
    if (layer || trace_error) return;
    if (stage == 1 && token == 0) ++trace_case;
    uint32_t header[4] = {trace_case, (uint32_t)stage, (uint32_t)token, (uint32_t)size};
    if (size <= 0 || size > 65536 || fwrite(header, 4, 4, trace_stream) != 4 ||
        fwrite(data, 4, (size_t)size, trace_stream) != (size_t)size) trace_error = 1;
}

static void trace_q8(int stage, int layer, int token, const int8_t *data, int size, float scale) {
    if (layer || trace_error || (stage != 17 && stage != 18)) return;
    if (size <= 0 || size > 65536) { trace_error = 1; return; }
    float *values = malloc((size_t)size * sizeof(float));
    if (!values) { trace_error = 1; return; }
    for (int i = 0; i < size; ++i) values[i] = (float)data[i] * scale;
    trace_float(stage, layer, token, values, size);
    free(values);
}

#define BITNET_TRACE_FLOAT trace_float
#define BITNET_TRACE_Q8 trace_q8
#include "../src/bitnet.c"
#define main feature_probe_main
#include "memory_feature_probe.c"
#undef main

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    trace_stream = fopen(argv[5], "wbx");
    if (!trace_stream) return 2;
    if (fwrite("BNTR0001", 1, 8, trace_stream) != 8) trace_error = 1;
    int rc = feature_probe_main(argc - 1, argv);
    if (fclose(trace_stream)) trace_error = 1;
    return rc || trace_error ? 1 : 0;
}
