// Owner: Rishik
#include "layers.h"

// out[b,t,o] = bias[o] + sum_c inp[b,t,c] * weight[o,c]
// bias may be NULL (no bias)
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {
    // TODO(Rishik)
}

// dinp[b,t,c]  += sum_o dout[b,t,o] * weight[o,c]
// dweight[o,c] += sum_{b,t} dout[b,t,o] * inp[b,t,c]
// dbias[o]     += sum_{b,t} dout[b,t,o]        (skip if dbias is NULL)
void matmul_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                     const float *inp, const float *weight, int B, int T, int C, int OC) {
    // TODO(Rishik)
}
