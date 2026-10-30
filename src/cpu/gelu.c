/*
 * gelu.c
 * Serial (CPU) GELU activation, tanh approximation (as in Pythia):
 *   gelu(x) = 0.5 * x * (1 + tanh(s * (x + 0.044715 * x^3))),  s = sqrt(2/pi)
 * Applied elementwise inside the MLP. Backward accumulates (+=).
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"

#define GELU_SCALE 0.7978845608f
#define GELU_COEF  0.044715f

/*
 * gelu_forward
 * out[i] = gelu(inp[i]) for all N elements (N = B * T * 4C in the MLP).
 */
void gelu_forward(float *out, const float *inp, size_t N) {
    for (size_t i = 0; i < N; i++) {
        float x = inp[i];
        float u = GELU_SCALE * (x + GELU_COEF * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(u));
    }
}

/*
 * gelu_backward
 * dinp[i] += dout[i] * gelu'(inp[i]), where
 *   gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * s * (1 + 3 * 0.044715 * x^2)
 * and t = tanh(s * (x + 0.044715 * x^3)). inp is the input saved from forward.
 */
void gelu_backward(float *dinp, const float *inp, const float *dout, size_t N) {
    for (size_t i = 0; i < N; i++) {
        float x = inp[i];
        float u = GELU_SCALE * (x + GELU_COEF * x * x * x);
        float t = tanhf(u);
        float du = GELU_SCALE * (1.0f + 3.0f * GELU_COEF * x * x);
        float grad = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
        dinp[i] += dout[i] * grad;
    }
}
