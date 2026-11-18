/*
 * gelu.cu
 * GPU (CUDA) GELU activation, tanh approximation, one thread per element.
 * Same math as src/cpu/gelu.c. Backward accumulates (+=).
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"
#include "cuda_utils.h"

#define GELU_SCALE 0.7978845608f
#define GELU_COEF  0.044715f

/* gelu_forward_kernel: out[i] = gelu(inp[i]). */
__global__ void gelu_forward_kernel(float *out, const float *inp, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = inp[i];
    float u = GELU_SCALE * (x + GELU_COEF * x * x * x);
    out[i] = 0.5f * x * (1.0f + tanhf(u));
}

/* gelu_backward_kernel: dinp[i] += dout[i] * gelu'(inp[i]). */
__global__ void gelu_backward_kernel(float *dinp, const float *inp, const float *dout, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = inp[i];
    float u = GELU_SCALE * (x + GELU_COEF * x * x * x);
    float t = tanhf(u);
    float du = GELU_SCALE * (1.0f + 3.0f * GELU_COEF * x * x);
    float grad = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
    dinp[i] += dout[i] * grad;
}

/* gelu_forward: out[i] = gelu(inp[i]) for all N elements. */
void gelu_forward(float *out, const float *inp, size_t N) {
    gelu_forward_kernel<<<blocks(N), BLOCK>>>(out, inp, N);
}

/* gelu_backward: dinp[i] += dout[i] * gelu'(inp[i]); inp is the input saved from forward. */
void gelu_backward(float *dinp, const float *inp, const float *dout, size_t N) {
    gelu_backward_kernel<<<blocks(N), BLOCK>>>(dinp, inp, dout, N);
}
