// Owner: Marcus
#include "layers.h"

// out[i] = 0.5 * x * (1 + erf(x / sqrt(2))), x = inp[i]
// (Pythia uses this exact GELU; the CUDA version must use the same formula)
void gelu_forward(float *out, const float *inp, int N) {
    // TODO(Marcus)
}

// dinp[i] += dout[i] * gelu'(inp[i])
void gelu_backward(float *dinp, const float *dout, const float *inp, int N) {
    // TODO(Marcus)
}

// out[i] = inp[i] + a[i] + b[i]
void residual_forward(float *out, const float *inp, const float *a, const float *b, int N) {
    // TODO(Marcus)
}

// dinp[i] += dout[i];  da[i] += dout[i];  db[i] += dout[i]
void residual_backward(float *dinp, float *da, float *db, const float *dout, int N) {
    // TODO(Marcus)
}
