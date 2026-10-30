/*
 * crossentropy.c
 * Serial (CPU) fused softmax + cross-entropy loss over the vocabulary.
 * Forward turns logits into probabilities, saves them, and returns the mean
 * loss over all B*T positions. Backward starts the whole backward pass:
 *   dlogits = (probs - one_hot(target)) / (B * T)
 * Backward accumulates (+=).
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"

/*
 * crossentropy_forward
 * probs:   [B, T, V]  softmax of logits, saved for backward
 * losses:  [B, T]     -log(probs[target]) per position
 * logits:  [B, T, V]
 * targets: [B, T]     the correct next token at each position
 * Returns the mean of losses. Subtracting each row's max before exp keeps
 * the softmax from overflowing.
 */
float crossentropy_forward(float *probs, float *losses, const float *logits,
                           const int *targets, int B, int T, int V) {
    size_t N = (size_t)B * T;
    double total = 0.0;
    for (size_t i = 0; i < N; i++) {
        const float *z = logits + i * V;
        float *p = probs + i * V;
        float maxv = z[0];
        for (int v = 1; v < V; v++) {
            if (z[v] > maxv) maxv = z[v];
        }
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            p[v] = expf(z[v] - maxv);
            sum += p[v];
        }
        float inv = 1.0f / sum;
        for (int v = 0; v < V; v++) {
            p[v] *= inv;
        }
        losses[i] = -logf(p[targets[i]]);
        total += losses[i];
    }
    return (float)(total / N);
}

/*
 * crossentropy_backward
 * Gradient of the mean loss with respect to the logits:
 *   dlogits[i, v] += (probs[i, v] - (v == targets[i])) / (B * T)
 * probs is the softmax saved from the forward pass.
 */
void crossentropy_backward(float *dlogits, const float *probs, const int *targets,
                           int B, int T, int V) {
    size_t N = (size_t)B * T;
    float scale = 1.0f / N;
    for (size_t i = 0; i < N; i++) {
        const float *p = probs + i * V;
        float *dz = dlogits + i * V;
        for (int v = 0; v < V; v++) {
            float indicator = v == targets[i] ? 1.0f : 0.0f;
            dz[v] += (p[v] - indicator) * scale;
        }
    }
}
