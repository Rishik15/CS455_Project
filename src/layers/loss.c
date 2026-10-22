// Owner: Andrew
#include "layers.h"

// probs[b,t,:] = softmax(logits[b,t,:])
// (subtract the max logit before exp so it can't overflow)
void softmax_forward(float *probs, const float *logits, int B, int T, int V) {
    // TODO(Andrew)
}

// losses[b,t] = -log(probs[b,t,targets[b,t]])
void crossentropy_forward(float *losses, const float *probs, const int *targets,
                          int B, int T, int V) {
    // TODO(Andrew)
}

// dlogits[b,t,v] += dlosses[b,t] * (probs[b,t,v] - (v == targets[b,t] ? 1 : 0))
void crossentropy_softmax_backward(float *dlogits, const float *dlosses, const float *probs,
                                   const int *targets, int B, int T, int V) {
    // TODO(Andrew)
}
