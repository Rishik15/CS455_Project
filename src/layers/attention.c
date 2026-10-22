// Owner: Rishik
#include "layers.h"

// Causal multi-head self-attention. hs = C / NH is the head size.
// qkv[b,t,:] holds [q (C) | k (C) | v (C)]; head h uses channels h*hs .. (h+1)*hs - 1
// of each.
//
// att[b,h,t,t2] = softmax over t2 <= t of (q[b,t,h] . k[b,t2,h]) / sqrt(hs)
//                 and 0 for t2 > t (a token can't look at later tokens)
// out[b,t,h*hs + i] = sum_t2 att[b,h,t,t2] * v[b,t2,h*hs + i]
void attention_forward(float *out, float *att, const float *qkv, int B, int T, int C, int NH) {
    // TODO(Rishik)
}

// Gradients of the above. datt has the same shape as att and can be used
// as scratch space (it starts at zero).
void attention_backward(float *dqkv, float *datt, const float *dout, const float *qkv,
                        const float *att, int B, int T, int C, int NH) {
    // TODO(Rishik)
}
