/*
 * layernorm.cu
 * GPU (CUDA) layer normalization over the channels of each token, one thread
 * per token. Same math as src/cpu/layernorm.c. Backward accumulates (+=);
 * dweight and dbias use atomicAdd because every token adds into the same C values.
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"
#include "cuda_utils.h"

#define LN_EPS 1e-5f

/* layernorm_forward_kernel: thread i normalizes token i and saves its mean and rstd. */
__global__ void layernorm_forward_kernel(float *out, float *mean, float *rstd, const float *inp,
                                         const float *weight, const float *bias, size_t N, int C) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float *x = inp + i * C;
    float *y = out + i * C;
    float m = 0.0f;
    for (int c = 0; c < C; c++) m += x[c];
    m /= C;
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float d = x[c] - m;
        v += d * d;
    }
    v /= C;
    float s = 1.0f / sqrtf(v + LN_EPS);
    for (int c = 0; c < C; c++) y[c] = (x[c] - m) * s * weight[c] + bias[c];
    mean[i] = m;
    rstd[i] = s;
}

/* layernorm_backward_kernel: thread i handles the backward pass of token i. */
__global__ void layernorm_backward_kernel(float *dinp, float *dweight, float *dbias,
                                          const float *dout, const float *inp, const float *weight,
                                          const float *mean, const float *rstd, size_t N, int C) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float *x = inp + i * C;
    const float *dy = dout + i * C;
    float *dx = dinp + i * C;
    float m = mean[i];
    float s = rstd[i];
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int c = 0; c < C; c++) {
        float norm = (x[c] - m) * s;
        float dnorm = dy[c] * weight[c];
        dnorm_mean += dnorm;
        dnorm_norm_mean += dnorm * norm;
    }
    dnorm_mean /= C;
    dnorm_norm_mean /= C;
    for (int c = 0; c < C; c++) {
        float norm = (x[c] - m) * s;
        float dnorm = dy[c] * weight[c];
        atomicAdd(&dbias[c], dy[c]);
        atomicAdd(&dweight[c], dy[c] * norm);
        dx[c] += s * (dnorm - dnorm_mean - norm * dnorm_norm_mean);
    }
}

/*
 * layernorm_forward
 * out: [B, T, C], mean: [B, T], rstd: [B, T], inp: [B, T, C], weight: [C], bias: [C]
 */
void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                       const float *weight, const float *bias, int B, int T, int C) {
    size_t N = (size_t)B * T;
    layernorm_forward_kernel<<<blocks(N), BLOCK>>>(out, mean, rstd, inp, weight, bias, N, C);
}

/*
 * layernorm_backward
 * Accumulates dinp, dweight, dbias from dout; inp, mean, rstd are saved from forward.
 */
void layernorm_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                        const float *inp, const float *weight, const float *mean,
                        const float *rstd, int B, int T, int C) {
    size_t N = (size_t)B * T;
    layernorm_backward_kernel<<<blocks(N), BLOCK>>>(dinp, dweight, dbias, dout, inp, weight,
                                                    mean, rstd, N, C);
}
