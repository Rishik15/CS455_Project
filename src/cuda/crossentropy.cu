/*
 * crossentropy.cu
 * GPU (CUDA) fused softmax + cross-entropy, one thread per position.
 * Same math as src/cpu/crossentropy.c. Backward accumulates (+=).
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"
#include "cuda_utils.h"

/* crossentropy_forward_kernel: thread i softmaxes row i and writes its loss. */
__global__ void crossentropy_forward_kernel(float *probs, float *losses, const float *logits,
                                            const int *targets, size_t N, int V) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
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
    for (int v = 0; v < V; v++) p[v] *= inv;
    losses[i] = -logf(p[targets[i]]);
}

/* crossentropy_sum_kernel: a single thread adds the N losses into *total. */
__global__ void crossentropy_sum_kernel(float *total, const float *losses, size_t N) {
    double s = 0.0;
    for (size_t i = 0; i < N; i++) s += losses[i];
    *total = (float)s;
}

/* crossentropy_backward_kernel: thread (i, v) adds (probs - one_hot) / N into dlogits. */
__global__ void crossentropy_backward_kernel(float *dlogits, const float *probs,
                                             const int *targets, size_t N, int V) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * V) return;
    size_t i = idx / V;
    int v = idx % V;
    float indicator = v == targets[i] ? 1.0f : 0.0f;
    dlogits[idx] += (probs[idx] - indicator) / N;
}

/*
 * crossentropy_forward
 * probs: [B, T, V], losses: [B, T], logits: [B, T, V], targets: [B, T] (all device).
 * Returns the mean loss on the host.
 * shortcut: the loss sum runs on one GPU thread, replace with a reduction if it shows in profiling.
 */
float crossentropy_forward(float *probs, float *losses, const float *logits,
                           const int *targets, int B, int T, int V) {
    size_t N = (size_t)B * T;
    crossentropy_forward_kernel<<<blocks(N), BLOCK>>>(probs, losses, logits, targets, N, V);
    float *d_total;
    cudaMalloc(&d_total, sizeof(float));
    crossentropy_sum_kernel<<<1, 1>>>(d_total, losses, N);
    float total;
    cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_total);
    return total / N;
}

/*
 * crossentropy_backward
 * dlogits[i, v] += (probs[i, v] - (v == targets[i])) / (B * T)
 */
void crossentropy_backward(float *dlogits, const float *probs, const int *targets,
                           int B, int T, int V) {
    size_t N = (size_t)B * T;
    crossentropy_backward_kernel<<<blocks(N * V), BLOCK>>>(dlogits, probs, targets, N, V);
}
