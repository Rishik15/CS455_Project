/*
 * attention.c
 * Serial (CPU) causal multi-head self-attention.
 * Input is the QKV matmul output [B, T, 3C]: each token's row holds its query,
 * key, and value vectors side by side, each C wide and split into NH heads of
 * size hs = C / NH. For every token t and head h:
 *   att[t2] = softmax over t2 <= t of (q_t . k_t2) / sqrt(hs)
 *   out_t   = sum over t2 <= t of att[t2] * v_t2
 * Positions after t are masked out, so a token never sees the future.
 * Forward saves att [B, NH, T, T]. Backward accumulates (+=).
 */
#include <stddef.h>
#include <math.h>
#include "ops.h"

/*
 * dot
 * Dot product of two length-n vectors.
 */
static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

/*
 * attention_row_forward
 * Computes the attention weights and output for one token t and one head h.
 * qkv_b points at batch b's [T, 3C] block; out_row at this head's hs outputs.
 */
static void attention_row_forward(float *out_row, float *att_row, const float *qkv_b,
                                  int t, int h, int T, int C, int hs, float scale) {
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
    for (int t2 = 0; t2 < T; t2++) {
        att_row[t2] = t2 <= t ? att_row[t2] * inv : 0.0f;
    }
    for (int i = 0; i < hs; i++) {
        out_row[i] = 0.0f;
    }
    for (int t2 = 0; t2 <= t; t2++) {
        const float *v = qkv_b + (size_t)t2 * 3 * C + 2 * C + h * hs;
        for (int i = 0; i < hs; i++) {
            out_row[i] += att_row[t2] * v[i];
        }
    }
}

/*
 * attention_forward
 * out: [B, T, C]       heads concatenated back into C channels
 * att: [B, NH, T, T]   saved attention weights (zero above the diagonal)
 * inp: [B, T, 3C]      output of the QKV matmul
 */
void attention_forward(float *out, float *att, const float *inp,
                       int B, int T, int C, int NH) {
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    for (int b = 0; b < B; b++) {
        const float *qkv_b = inp + (size_t)b * T * 3 * C;
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float *att_row = att + (((size_t)b * NH + h) * T + t) * T;
                float *out_row = out + ((size_t)b * T + t) * C + h * hs;
                attention_row_forward(out_row, att_row, qkv_b, t, h, T, C, hs, scale);
            }
        }
    }
}

/*
 * attention_row_backward
 * Backward for one token t and one head h. With datt[t2] = dout . v_t2:
 *   dv_t2   += att[t2] * dout
 *   dpre    =  att[t2] * (datt[t2] - sum over j of att[j] * datt[j])
 *   dq      += dpre * k_t2 * scale
 *   dk_t2   += dpre * q * scale
 * The first pass computes the softmax correction sum; the second applies it.
 */
static void attention_row_backward(float *dqkv_b, const float *qkv_b, const float *att_row,
                                   const float *dout_row, int t, int h, int C, int hs,
                                   float scale) {
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
            dqkv_b[v_off + i] += att_row[t2] * dout_row[i];
            dqkv_b[q_off + i] += dpre * qkv_b[k_off + i];
            dqkv_b[k_off + i] += dpre * qkv_b[q_off + i];
        }
    }
}

/*
 * attention_backward
 * dinp: [B, T, 3C]     gradient for the QKV matmul output
 * dout: [B, T, C]
 * inp:  [B, T, 3C]     the QKV values saved from forward
 * att:  [B, NH, T, T]  the attention weights saved from forward
 */
void attention_backward(float *dinp, const float *dout, const float *inp, const float *att,
                        int B, int T, int C, int NH) {
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    for (int b = 0; b < B; b++) {
        const float *qkv_b = inp + (size_t)b * T * 3 * C;
        float *dqkv_b = dinp + (size_t)b * T * 3 * C;
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float *att_row = att + (((size_t)b * NH + h) * T + t) * T;
                const float *dout_row = dout + ((size_t)b * T + t) * C + h * hs;
                attention_row_backward(dqkv_b, qkv_b, att_row, dout_row, t, h, C, hs, scale);
            }
        }
    }
}
