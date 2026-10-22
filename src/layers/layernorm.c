// Owner: Andrew
#include "layers.h"

// For each (b,t), over the C values x = inp[b,t,:]:
//   mean[b,t] = average of x
//   rstd[b,t] = 1 / sqrt(variance of x + 1e-5)
//   out[b,t,c] = (x[c] - mean) * rstd * weight[c] + bias[c]
void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                       const float *weight, const float *bias, int B, int T, int C) {
    // TODO(Andrew)
}

// Gradients of the above, using the saved mean and rstd.
void layernorm_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                        const float *inp, const float *weight, const float *mean,
                        const float *rstd, int B, int T, int C) {
    // TODO(Andrew)
}
