/*
 * layernorm.c
 * Serial (CPU) layer normalization over the channel dimension of each token:
 *   out = (x - mean) * rstd * weight + bias,  rstd = 1 / sqrt(var + eps)
 * Forward saves mean and rstd per token for the backward pass.
 * Backward accumulates (+=).
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"

#define LN_EPS 1e-5f

/*
 * layernorm_forward
 * out:    [B, T, C]
 * mean:   [B, T]  saved per-token mean
 * rstd:   [B, T]  saved per-token 1 / standard deviation
 * inp:    [B, T, C]
 * weight: [C]  learned scale (initialize to 1)
 * bias:   [C]  learned shift (initialize to 0)
 */
void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                       const float *weight, const float *bias, int B, int T, int C) {
    size_t N = (size_t)B * T;
    for (size_t i = 0; i < N; i++) {
        const float *x = inp + i * C;
        float *y = out + i * C;
        float m = 0.0f;
        for (int c = 0; c < C; c++) {
            m += x[c];
        }
        m /= C;
        float v = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = x[c] - m;
            v += d * d;
        }
        v /= C;
        float s = 1.0f / sqrtf(v + LN_EPS);
        for (int c = 0; c < C; c++) {
            y[c] = (x[c] - m) * s * weight[c] + bias[c];
        }
        mean[i] = m;
        rstd[i] = s;
    }
}

/*
 * layernorm_backward
 * With norm = (x - mean) * rstd and dnorm = dout * weight:
 *   dbias   += dout
 *   dweight += dout * norm
 *   dinp    += rstd * (dnorm - mean(dnorm) - norm * mean(dnorm * norm))
 * The two mean terms account for every channel affecting the mean and
 * variance that all other channels were normalized with.
 * inp, mean, and rstd are the values saved from the forward pass.
 */
void layernorm_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                        const float *inp, const float *weight, const float *mean,
                        const float *rstd, int B, int T, int C) {
    size_t N = (size_t)B * T;
    for (size_t i = 0; i < N; i++) {
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
            dbias[c] += dy[c];
            dweight[c] += dy[c] * norm;
            dx[c] += s * (dnorm - dnorm_mean - norm * dnorm_norm_mean);
        }
    }
}
