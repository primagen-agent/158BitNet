#include "ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void bitnet_silu_mul(float *gate, const float *up, int n);
float bitnet_silu_mul_max_abs(float *gate, const float *up, int n);

static int close_enough(float a, float b, float tol) {
    float diff = fabsf(a - b);
    return diff <= tol;
}

static int test_rms_norm_lengths(void) {
    /* Verify RMSNorm at every length we expect to encounter. Hidden dim
     * is typically 256..4096 for these models. */
    int lengths[] = {1, 4, 7, 8, 16, 31, 32, 100, 256, 1023, 1024, 4096};
    for (size_t i = 0; i < sizeof(lengths)/sizeof(lengths[0]); ++i) {
        int n = lengths[i];
        float *x = malloc((size_t)n * sizeof(float));
        float *w = malloc((size_t)n * sizeof(float));
        if (!x || !w) { fprintf(stderr, "oom\n"); free(x); free(w); return 10; }
        for (int j = 0; j < n; ++j) { x[j] = (float)(j % 7) - 3.0f; w[j] = 1.0f; }
        bitnet_rms_norm_eps(x, w, n, 1e-6f);
        /* Sanity: result norms to ~1.0 after weighting by all-ones. */
        float sum = 0;
        for (int j = 0; j < n; ++j) sum += x[j] * x[j];
        float rms = sqrtf(sum / (float)n);
        if (!(fabsf(rms - 1.0f) < 0.01f)) {
            fprintf(stderr, "rms_norm_lengths n=%d rms=%.6f (expected ~1.0)\n", n, rms);
            free(x); free(w);
            return 5;
        }
        free(x); free(w);
    }
    return 0;
}

static int test_softmax_lengths(void) {
    /* Softmax invariants: outputs are non-negative, sum to 1.0, and the
     * argmax of the input maps to the argmax of the output. The existing
     * inline softmax check at the top of main() covers the n=8 case
     * numerically; here we sweep a range of lengths to exercise tail
     * handling in each SIMD variant. */
    int lengths[] = {1, 4, 7, 8, 16, 31, 32, 100, 256, 1024};
    for (size_t i = 0; i < sizeof(lengths)/sizeof(lengths[0]); ++i) {
        int n = lengths[i];
        float *x = malloc((size_t)n * sizeof(float));
        if (!x) { fprintf(stderr, "oom\n"); free(x); return 30; }
        for (int j = 0; j < n; ++j) x[j] = (float)((j % 11) - 5);

        int expected_argmax = 0;
        for (int j = 1; j < n; ++j) {
            if (x[j] > x[expected_argmax]) expected_argmax = j;
        }

        bitnet_softmax(x, n);

        /* Check 1: outputs sum to 1.0 within tolerance. */
        float sum = 0;
        for (int j = 0; j < n; ++j) sum += x[j];
        if (fabsf(sum - 1.0f) > 1e-4f) {
            fprintf(stderr, "softmax_lengths n=%d sum=%.8f (expected ~1.0)\n", n, sum);
            free(x);
            return 31;
        }
        /* Check 2: all values in [0, 1]. */
        for (int j = 0; j < n; ++j) {
            if (!(x[j] >= 0.0f && x[j] <= 1.0f)) {
                fprintf(stderr, "softmax_lengths n=%d [%d]=%.8f out of [0,1]\n",
                        n, j, x[j]);
                free(x);
                return 32;
            }
        }
        /* Check 3: argmax preserved. */
        int actual_argmax = 0;
        for (int j = 1; j < n; ++j) {
            if (x[j] > x[actual_argmax]) actual_argmax = j;
        }
        if (actual_argmax != expected_argmax) {
            fprintf(stderr, "softmax_lengths n=%d argmax %d != expected %d\n",
                    n, actual_argmax, expected_argmax);
            free(x);
            return 33;
        }
        free(x);
    }
    return 0;
}

static int test_rope_apply_basic(void) {
    /* Single head, head_dim=4, rope_dim=4 -> 2 pairs.
     * Pair 0: rotate (1,0) by (cos=1, sin=0) -> (1,0)
     * Pair 1: rotate (0,1) by (cos=0, sin=1) -> (-1,0) */
    int n_heads = 1, head_dim = 4, rope_dim = 4;
    float x[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float cos_tab[2] = {1.0f, 0.0f};
    float sin_tab[2] = {0.0f, 1.0f};
    bitnet_rope_apply(x, n_heads, head_dim, rope_dim, cos_tab, sin_tab);
    if (fabsf(x[0] - 1.0f) > 1e-6f) {
        fprintf(stderr, "rope_apply x[0]=%.8f expected=1.0\n", x[0]);
        return 40;
    }
    if (fabsf(x[1] - 0.0f) > 1e-6f) {
        fprintf(stderr, "rope_apply x[1]=%.8f expected=0.0\n", x[1]);
        return 41;
    }
    if (fabsf(x[2] - (-1.0f)) > 1e-6f) {
        fprintf(stderr, "rope_apply x[2]=%.8f expected=-1.0\n", x[2]);
        return 42;
    }
    if (fabsf(x[3] - 0.0f) > 1e-6f) {
        fprintf(stderr, "rope_apply x[3]=%.8f expected=0.0\n", x[3]);
        return 43;
    }
    return 0;
}

static int test_rope_apply_multi_head(void) {
    /* 2 heads, head_dim=4, rope_dim=4 -> 2 pairs per head.
     * Use 45-degree rotations: cos=sin=sqrt(2)/2 ~ 0.7071.
     * Pair rotation (x0, x1) by angle theta:
     *   x0' = x0*c - x1*s
     *   x1' = x0*s + x1*c
     * Input (1, 0) rotated 45deg -> (c, s) = (0.7071, 0.7071)
     * Input (0, 1) rotated 45deg -> (-s, c) = (-0.7071, 0.7071)
     * Input (1, 1) rotated 45deg -> (1*c - 1*s, 1*s + 1*c) = (0, 2c)
     * Input (2,-1) rotated 45deg -> (2c - (-1)s, 2s + (-1)c) = (2c+s, 2s-c)
     *                              = (3*0.7071, 0.7071) = (2.121, 0.7071) */
    int n_heads = 2, head_dim = 4, rope_dim = 4;
    float x[8] = {1.0f, 0.0f, 0.0f, 1.0f,
                  1.0f, 1.0f, 2.0f, -1.0f};
    float c = 0.70710678118f; /* sqrt(2)/2 */
    float s = 0.70710678118f;
    float cos_tab[2] = {c, c};
    float sin_tab[2] = {s, s};
    float expected[8] = {
        c,   s,         /* (1,0) rotated */
        -s,  c,         /* (0,1) rotated */
        0.0f, c + s,    /* (1,1) rotated -> (0, 2c) */
        2.0f * c + s, 2.0f * s - c   /* (2,-1) rotated */
    };
    bitnet_rope_apply(x, n_heads, head_dim, rope_dim, cos_tab, sin_tab);
    for (int j = 0; j < 8; ++j) {
        if (fabsf(x[j] - expected[j]) > 1e-5f) {
            fprintf(stderr, "rope_apply_multi [%d]=%.8f expected=%.8f\n",
                    j, x[j], expected[j]);
            return 50;
        }
    }
    return 0;
}

static int test_rope_apply_null_safe(void) {
    /* NULL/invalid inputs should be a no-op, not a crash. */
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float cos_tab[2] = {1.0f, 1.0f};
    float sin_tab[2] = {0.0f, 0.0f};
    /* x NULL */
    bitnet_rope_apply(NULL, 1, 4, 4, cos_tab, sin_tab);
    /* cos NULL */
    bitnet_rope_apply(x, 1, 4, 4, NULL, sin_tab);
    /* sin NULL */
    bitnet_rope_apply(x, 1, 4, 4, cos_tab, NULL);
    /* zero heads */
    bitnet_rope_apply(x, 0, 4, 4, cos_tab, sin_tab);
    /* x should be untouched after the no-op calls that referenced it. */
    if (x[0] != 1.0f || x[1] != 2.0f || x[2] != 3.0f || x[3] != 4.0f) {
        fprintf(stderr, "rope_apply_null_safe modified input unexpectedly\n");
        return 60;
    }
    return 0;
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

    {
        int rc = test_rms_norm_lengths();
        if (rc) return rc;
    }

    {
        int rc = test_softmax_lengths();
        if (rc) return rc;
    }

    {
        int rc = test_rope_apply_basic();
        if (rc) return rc;
    }

    {
        int rc = test_rope_apply_multi_head();
        if (rc) return rc;
    }

    {
        int rc = test_rope_apply_null_safe();
        if (rc) return rc;
    }

    /* Length-sweep correctness for residual_add, silu, silu_mul.
     *
     * Note on tolerance: on ARM the dispatched silu path uses the NEON
     * polynomial exp approximation (neon_exp_approx_f32 in ops.c), which
     * has up to ~2% relative error vs libm at certain points (e.g. x=1.0).
     * So the silu sweep compares silu_mul output against the result of
     * calling silu on the same gate (i.e. internal consistency between
     * two kernels that share the approximation), not against libm directly.
     * residual_add is exact arithmetic, so it gets a tight tolerance. */
    {
        int lengths[] = {1, 4, 7, 8, 16, 31, 32, 100, 256, 1024};
        size_t n_lengths = sizeof(lengths) / sizeof(lengths[0]);

        /* residual_add: out = a + b. */
        for (size_t i = 0; i < n_lengths; ++i) {
            int n = lengths[i];
            float *a = malloc((size_t)n * sizeof(float));
            float *b = malloc((size_t)n * sizeof(float));
            float *out = malloc((size_t)n * sizeof(float));
            for (int j = 0; j < n; ++j) {
                a[j] = (float)(j % 5) - 2.0f;
                b[j] = (float)(j % 3) - 1.0f;
            }
            bitnet_residual_add(out, a, b, n);
            for (int j = 0; j < n; ++j) {
                if (fabsf(out[j] - (a[j] + b[j])) > 1e-6f) {
                    fprintf(stderr, "residual_add n=%d [%d]=%.8f expected=%.8f\n",
                            n, j, out[j], a[j] + b[j]);
                    free(a); free(b); free(out);
                    return 20;
                }
            }
            free(a); free(b); free(out);
        }

        /* silu + silu_mul internal consistency: silu_mul(gate, up) should
         * equal silu(gate) * up elementwise, regardless of which exp
         * approximation the dispatched tier uses. */
        for (size_t i = 0; i < n_lengths; ++i) {
            int n = lengths[i];
            float *gate_a = malloc((size_t)n * sizeof(float));
            float *gate_b = malloc((size_t)n * sizeof(float));
            float *up = malloc((size_t)n * sizeof(float));
            for (int j = 0; j < n; ++j) {
                float g = ((float)(j % 7) - 3.0f) * 0.5f;
                gate_a[j] = g;
                gate_b[j] = g;
                up[j] = ((float)(j % 5) - 2.0f);
            }
            bitnet_silu(gate_a, n);
            bitnet_silu_mul(gate_b, up, n);
            for (int j = 0; j < n; ++j) {
                float expected = gate_a[j] * up[j];
                if (fabsf(gate_b[j] - expected) > 1e-5f) {
                    fprintf(stderr, "silu_mul n=%d [%d]=%.8f expected=%.8f "
                            "(silu(gate)=%.8f up=%.8f)\n",
                            n, j, gate_b[j], expected, gate_a[j], up[j]);
                    free(gate_a); free(gate_b); free(up);
                    return 22;
                }
            }
            free(gate_a); free(gate_b); free(up);
        }
    }

    return 0;
}
