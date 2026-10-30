/*
 * ops.h
 * Interface for every model op. Each op is a stateless forward/backward pair
 * working on raw float arrays. src/cpu/ and src/cuda/ both implement this file.
 * Layout conventions:
 *   activations are [B, T, C] row-major
 *   linear weights are [C_out, C_in]
 *   every backward function accumulates (+=) into its gradient arrays
 */
#ifndef OPS_H
#define OPS_H

#include <stddef.h>

/* Token + position embedding: out[b,t,:] = wte[tokens[b,t],:] + wpe[t,:]. */
void encoder_forward(float *out, const int *tokens, const float *wte, const float *wpe,
                     int B, int T, int C);

/* Accumulates embedding gradients into dwte [V, C] and dwpe [T_max, C]. */
void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *tokens,
                      int B, int T, int C);

/* Linear layer: out = inp @ weight^T + bias; bias may be NULL. */
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C_in, int C_out);

/* Accumulates gradients for the input, weight, and bias (dbias may be NULL). */
void matmul_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                     const float *inp, const float *weight,
                     int B, int T, int C_in, int C_out);

/* Residual connection: out = a + b over N elements. */
void residual_forward(float *out, const float *a, const float *b, size_t N);

/* Adds dout into both da (skip path) and db (branch path). */
void residual_backward(float *da, float *db, const float *dout, size_t N);

/* GELU activation (tanh approximation) over N elements. */
void gelu_forward(float *out, const float *inp, size_t N);

/* Accumulates dout * gelu'(inp) into dinp. */
void gelu_backward(float *dinp, const float *inp, const float *dout, size_t N);

/* LayerNorm over C channels per token; saves mean [B,T] and rstd [B,T]. */
void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                       const float *weight, const float *bias, int B, int T, int C);

/* Accumulates gradients for the input, weight, and bias using saved mean/rstd. */
void layernorm_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                        const float *inp, const float *weight, const float *mean,
                        const float *rstd, int B, int T, int C);

/* Causal multi-head attention on QKV input [B,T,3C]; saves att [B,NH,T,T]. */
void attention_forward(float *out, float *att, const float *inp,
                       int B, int T, int C, int NH);

/* Accumulates the gradient for the QKV input using the saved att weights. */
void attention_backward(float *dinp, const float *dout, const float *inp, const float *att,
                        int B, int T, int C, int NH);

/* Softmax + cross-entropy; saves probs [B,T,V] and losses [B,T]; returns mean loss. */
float crossentropy_forward(float *probs, float *losses, const float *logits,
                           const int *targets, int B, int T, int V);

/* Accumulates (probs - one_hot(targets)) / (B*T) into dlogits. */
void crossentropy_backward(float *dlogits, const float *probs, const int *targets,
                           int B, int T, int V);

#endif
