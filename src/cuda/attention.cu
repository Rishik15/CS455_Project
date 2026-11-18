/*
 * attention.cu
 * GPU (CUDA) causal multi-head self-attention, one thread per (batch, token,
 * head). Same math and layout as src/cpu/attention.c: input is the QKV matmul
 * output [B, T, 3C], forward saves att [B, NH, T, T]. Backward accumulates (+=)
 * with atomicAdd, because many tokens add into the same key and value rows.
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"
#include "cuda_utils.h"

/* dot: dot product of two length-n vectors. */
__device__ static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/*
 * attention_forward_kernel
 * Thread idx = (b * T + t) * NH + h computes the attention weights and output
 * for token t, head h of batch b.
 */
__global__ void attention_forward_kernel(float *out, float *att, const float *inp,
                                         int B, int T, int C, int NH) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * T * NH) return;
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / ((size_t)NH * T);
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    const float *qkv_b = inp + (size_t)b * T * 3 * C;
    float *att_row = att + (((size_t)b * NH + h) * T + t) * T;
    float *out_row = out + ((size_t)b * T + t) * C + h * hs;
    const float *q = qkv_b + (size_t)t * 3 * C + h * hs;
    float maxv = -INFINITY;
    for (int t2 = 0; t2 <= t; t2++) {
        const float *k = qkv_b + (size_t)t2 * 3 * C + C + h * hs;
        att_row[t2] = dot(q, k, hs) * scale;
        if (att_row[t2] > maxv) maxv = att_row[t2];
    }
    float sum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        att_row[t2] = expf(att_row[t2] - maxv);
        sum += att_row[t2];
    }
    float inv = 1.0f / sum;
    for (int t2 = 0; t2 < T; t2++) att_row[t2] = t2 <= t ? att_row[t2] * inv : 0.0f;
    for (int i = 0; i < hs; i++) out_row[i] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float *v = qkv_b + (size_t)t2 * 3 * C + 2 * C + h * hs;
        for (int i = 0; i < hs; i++) out_row[i] += att_row[t2] * v[i];
    }
}

/*
 * attention_backward_kernel
 * Thread idx = (b * T + t) * NH + h runs the backward pass for token t, head h.
 * With datt[t2] = dout . v_t2:
 *   dv_t2 += att[t2] * dout,  dpre = att[t2] * (datt[t2] - sum_j att[j] * datt[j]),
 *   dq += dpre * k_t2 * scale,  dk_t2 += dpre * q * scale
 */
__global__ void attention_backward_kernel(float *dinp, const float *dout, const float *inp,
                                          const float *att, int B, int T, int C, int NH) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * T * NH) return;
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / ((size_t)NH * T);
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    const float *qkv_b = inp + (size_t)b * T * 3 * C;
    float *dqkv_b = dinp + (size_t)b * T * 3 * C;
    const float *att_row = att + (((size_t)b * NH + h) * T + t) * T;
    const float *dout_row = dout + ((size_t)b * T + t) * C + h * hs;
    size_t q_off = (size_t)t * 3 * C + h * hs;
    float weighted = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float *v = qkv_b + (size_t)t2 * 3 * C + 2 * C + h * hs;
        weighted += att_row[t2] * dot(dout_row, v, hs);
    }
    for (int t2 = 0; t2 <= t; t2++) {
        size_t k_off = (size_t)t2 * 3 * C + C + h * hs;
        size_t v_off = (size_t)t2 * 3 * C + 2 * C + h * hs;
        float datt = dot(dout_row, qkv_b + v_off, hs);
        float dpre = att_row[t2] * (datt - weighted) * scale;
        for (int i = 0; i < hs; i++) {
            atomicAdd(&dqkv_b[v_off + i], att_row[t2] * dout_row[i]);
            atomicAdd(&dqkv_b[q_off + i], dpre * qkv_b[k_off + i]);
            atomicAdd(&dqkv_b[k_off + i], dpre * qkv_b[q_off + i]);
        }
    }
}

/*
 * attention_forward
 * out: [B, T, C], att: [B, NH, T, T], inp: [B, T, 3C]
 */
void attention_forward(float *out, float *att, const float *inp,
                       int B, int T, int C, int NH) {
    attention_forward_kernel<<<blocks((size_t)B * T * NH), BLOCK>>>(out, att, inp, B, T, C, NH);
}

/*
 * attention_backward
 * dinp: [B, T, 3C], dout: [B, T, C], inp: [B, T, 3C], att: [B, NH, T, T]
 */
void attention_backward(float *dinp, const float *dout, const float *inp, const float *att,
                        int B, int T, int C, int NH) {
    attention_backward_kernel<<<blocks((size_t)B * T * NH), BLOCK>>>(dinp, dout, inp, att,
                                                                    B, T, C, NH);
}
