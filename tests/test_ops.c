#include "ops.h"

#include <math.h>
#include <stdio.h>

void bitnet_silu_mul(float *gate, const float *up, int n);
float bitnet_silu_mul_max_abs(float *gate, const float *up, int n);

static int close_enough(float a, float b, float tol) {
    float diff = fabsf(a - b);
    return diff <= tol;
}

int main(void) {
    float x[8] = { -102.0f, -101.0f, -100.0f, -99.0f,
                   -103.0f, -104.0f, -105.0f, -106.0f };
    float ref[8];
    float max_val = x[0];
    float sum = 0.0f;

    for (int i = 0; i < 8; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    for (int i = 0; i < 8; ++i) {
        ref[i] = expf(x[i] - max_val);
        sum += ref[i];
    }
    for (int i = 0; i < 8; ++i) {
        ref[i] /= sum;
    }

    bitnet_softmax(x, 8);

    for (int i = 0; i < 8; ++i) {
        if (!close_enough(x[i], ref[i], 0.00001f)) {
            fprintf(stderr, "softmax[%d]=%.8f expected=%.8f\n", i, x[i], ref[i]);
            return 1;
        }
    }

    {
        float gate[8] = { -4.0f, -1.25f, -0.5f, 0.0f, 0.5f, 1.25f, 4.0f, 8.0f };
        float up[8] = { 0.25f, -0.5f, 1.5f, -2.0f, 3.0f, -4.0f, 0.75f, -1.25f };
        float expected[8];

        for (int i = 0; i < 8; ++i) {
            expected[i] = gate[i];
        }
        bitnet_silu(expected, 8);
        for (int i = 0; i < 8; ++i) {
            expected[i] *= up[i];
        }

        bitnet_silu_mul(gate, up, 8);

        for (int i = 0; i < 8; ++i) {
            if (!close_enough(gate[i], expected[i], 0.00001f)) {
                fprintf(stderr, "silu_mul[%d]=%.8f expected=%.8f\n", i, gate[i], expected[i]);
                return 2;
            }
        }

        {
            float gate_with_max[8] = { -4.0f, -1.25f, -0.5f, 0.0f, 0.5f, 1.25f, 4.0f, 8.0f };
            float max_abs = bitnet_silu_mul_max_abs(gate_with_max, up, 8);
            float expected_max = 0.0f;
            for (int i = 0; i < 8; ++i) {
                float a = fabsf(expected[i]);
                if (a > expected_max) expected_max = a;
                if (!close_enough(gate_with_max[i], expected[i], 0.00001f)) {
                    fprintf(stderr, "silu_mul_max[%d]=%.8f expected=%.8f\n",
                            i, gate_with_max[i], expected[i]);
                    return 3;
                }
            }
            if (!close_enough(max_abs, expected_max, 0.00001f)) {
                fprintf(stderr, "silu_mul_max_abs=%.8f expected=%.8f\n", max_abs, expected_max);
                return 4;
            }
        }
    }

    return 0;
}
