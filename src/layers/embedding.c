// Owner: Manny
#include "layers.h"

// out[b,t,:] = wte[inp[b,t], :] + wpe[t, :]
void embedding_forward(float *out, const int *inp, const float *wte, const float *wpe,
                       int B, int T, int C) {
    // TODO(Manny)
}

// dwte[inp[b,t], :] += dout[b,t,:]
// dwpe[t, :]        += dout[b,t,:]
void embedding_backward(float *dwte, float *dwpe, const float *dout, const int *inp,
                        int B, int T, int C) {
    // TODO(Manny)
}
