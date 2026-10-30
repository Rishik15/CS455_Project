/*
 * matmul.c
 * Serial (CPU) linear layer: out = inp @ weight^T + bias, applied to every token.
 * Weights are stored [C_out, C_in] so the output layer can reuse the token
 * embedding table [V, C] directly (tied weights). Backward accumulates (+=).
 */
#include <stddef.h>
#include "ops.h"

/*
 * matmul_forward
 * For each of the B*T tokens, computes out[i, o] = dot(inp[i, :], weight[o, :]) + bias[o].
 * out:    [B*T, C_out]
 * inp:    [B*T, C_in]
 * weight: [C_out, C_in]
 * bias:   [C_out], or NULL for no bias
 */
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C_in, int C_out) {
    size_t N = (size_t)B * T;
    for (size_t i = 0; i < N; i++) {
        const float *x = inp + i * C_in;
        float *y = out + i * C_out;
        for (int o = 0; o < C_out; o++) {
            const float *w = weight + (size_t)o * C_in;
            float sum = bias != NULL ? bias[o] : 0.0f;
            for (int k = 0; k < C_in; k++) {
                sum += x[k] * w[k];
            }
            y[o] = sum;
        }
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
    for (size_t i = 0; i < N; i++) {
        const float *x = inp + i * C_in;
        const float *dy = dout + i * C_out;
        float *dx = dinp + i * C_in;
        for (int o = 0; o < C_out; o++) {
            float g = dy[o];
            const float *w = weight + (size_t)o * C_in;
            float *dw = dweight + (size_t)o * C_in;
            for (int k = 0; k < C_in; k++) {
                dx[k] += g * w[k];
                dw[k] += g * x[k];
            }
            if (dbias != NULL) {
                dbias[o] += g;
            }
        }
    }
}
