/*
 * matmul.cu
 * GPU (CUDA) linear layer: out = inp @ weight^T + bias, applied to every token.
 * Same math and layout as src/cpu/matmul.c, but the loops over tokens and
 * outputs become one GPU thread per output element. All pointers are device
 * pointers. Backward accumulates (+=).
 */
#include <stddef.h>
#include "ops.h"
#include "cuda_utils.h"

/*
 * matmul_forward_kernel
 * One thread per output element (i, o): out[i, o] = dot(inp[i, :], weight[o, :]) + bias[o].
 * N is B*T, the number of tokens.
 */
__global__ void matmul_forward_kernel(float *out, const float *inp, const float *weight,
                                      const float *bias, size_t N, int C_in, int C_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out) return;
    size_t i = idx / C_out;
    int o = idx % C_out;
    const float *x = inp + i * C_in;
    const float *w = weight + (size_t)o * C_in;
    float sum = bias != NULL ? bias[o] : 0.0f;
    for (int k = 0; k < C_in; k++) {
        sum += x[k] * w[k];
    }
    out[idx] = sum;
}

/*
 * matmul_forward
 * out:    [B*T, C_out]
 * inp:    [B*T, C_in]
 * weight: [C_out, C_in]
 * bias:   [C_out], or NULL for no bias
 */
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C_in, int C_out) {
    size_t N = (size_t)B * T;
    size_t total = N * C_out;
    matmul_forward_kernel<<<blocks(total), BLOCK>>>(out, inp, weight, bias,
                                                                  N, C_in, C_out);
}

/*
 * matmul_dinp_kernel
 * One thread per element (i, k): dinp[i, k] += sum over o of dout[i, o] * weight[o, k].
 */
__global__ void matmul_dinp_kernel(float *dinp, const float *dout, const float *weight,
                                   size_t N, int C_in, int C_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_in) return;
    size_t i = idx / C_in;
    int k = idx % C_in;
    float sum = 0.0f;
    for (int o = 0; o < C_out; o++) {
        sum += dout[i * C_out + o] * weight[(size_t)o * C_in + k];
    }
    dinp[idx] += sum;
}

/*
 * matmul_dweight_kernel
 * One thread per element (o, k): dweight[o, k] += sum over i of dout[i, o] * inp[i, k].
 * Also, for the threads with k == 0, dbias[o] += sum over i of dout[i, o].
 */
__global__ void matmul_dweight_kernel(float *dweight, float *dbias, const float *dout,
                                      const float *inp, size_t N, int C_in, int C_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)C_out * C_in) return;
    int o = idx / C_in;
    int k = idx % C_in;
    float sum = 0.0f;
    float bsum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float g = dout[i * C_out + o];
        sum += g * inp[i * C_in + k];
        bsum += g;
    }
    dweight[idx] += sum;
    if (dbias != NULL && k == 0) {
        dbias[o] += bsum;
    }
}

/*
 * matmul_backward
 * Given dout = dLoss/dout, accumulates the three gradients:
 *   dinp[i, :]    += sum over o of dout[i, o] * weight[o, :]
 *   dweight[o, :] += sum over i of dout[i, o] * inp[i, :]
 *   dbias[o]      += sum over i of dout[i, o]
 * dinp:    [B*T, C_in]
 * dweight: [C_out, C_in]
 * dbias:   [C_out], or NULL when the layer has no bias
 * dout:    [B*T, C_out]
 * inp:     [B*T, C_in], the input saved from the forward pass
 * weight:  [C_out, C_in]
 */
void matmul_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                     const float *inp, const float *weight,
                     int B, int T, int C_in, int C_out) {
    size_t N = (size_t)B * T;
    size_t n_dinp = N * C_in;
    size_t n_dw = (size_t)C_out * C_in;
    matmul_dinp_kernel<<<blocks(n_dinp), BLOCK>>>(dinp, dout, weight,
                                                                N, C_in, C_out);
    matmul_dweight_kernel<<<blocks(n_dw), BLOCK>>>(dweight, dbias, dout, inp,
                                                                 N, C_in, C_out);
}
