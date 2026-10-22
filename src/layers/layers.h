#ifndef LAYERS_H
#define LAYERS_H

// Every layer of the model, as a forward and a backward function.
// transformer.c calls these in order; each owner fills in their own file.
//
// Rules (so the MPI and CUDA versions can reuse the same design):
//  - All tensors are float arrays, row-major, with the shapes listed below.
//  - Forward functions overwrite their outputs.
//  - Backward functions ADD (+=) into every gradient they write, never overwrite.
//    transformer.c zeroes the gradients before the backward pass starts.
//  - No malloc and no global variables inside layer functions.
//  - Argument order: outputs, then inputs, then sizes. For backward: the
//    gradients being computed, then dout, then values saved by the forward pass.
//
// Sizes: B = batch, T = sequence length, C = hidden size, V = vocab size,
// NH = number of attention heads, OC = output channels.

// ---- embedding.c (Manny) ----
// inp (B, T) token ids -> out (B, T, C)
void embedding_forward(float *out, const int *inp, const float *wte, const float *wpe,
                       int B, int T, int C);
void embedding_backward(float *dwte, float *dwpe, const float *dout, const int *inp,
                        int B, int T, int C);

// ---- layernorm.c (Andrew) ----
// inp (B, T, C) -> out (B, T, C); mean and rstd (B, T) are saved for backward
void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                       const float *weight, const float *bias, int B, int T, int C);
void layernorm_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                        const float *inp, const float *weight, const float *mean,
                        const float *rstd, int B, int T, int C);

// ---- matmul.c (Rishik) ----
// inp (B, T, C), weight (OC, C), bias (OC) or NULL -> out (B, T, OC)
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC);
void matmul_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                     const float *inp, const float *weight, int B, int T, int C, int OC);

// ---- attention.c (Rishik) ----
// qkv (B, T, 3C) -> out (B, T, C); att (B, NH, T, T) is saved for backward
// datt is scratch space for backward, same shape as att
void attention_forward(float *out, float *att, const float *qkv, int B, int T, int C, int NH);
void attention_backward(float *dqkv, float *datt, const float *dout, const float *qkv,
                        const float *att, int B, int T, int C, int NH);

// ---- gelu.c (Marcus) ----
// elementwise on N values
void gelu_forward(float *out, const float *inp, int N);
void gelu_backward(float *dinp, const float *dout, const float *inp, int N);

// out = inp + a + b, elementwise on N values (Pythia's parallel residual)
void residual_forward(float *out, const float *inp, const float *a, const float *b, int N);
void residual_backward(float *dinp, float *da, float *db, const float *dout, int N);

// ---- loss.c (Andrew) ----
// logits (B, T, V) -> probs (B, T, V)
void softmax_forward(float *probs, const float *logits, int B, int T, int V);
// probs (B, T, V), targets (B, T) -> losses (B, T)
void crossentropy_forward(float *losses, const float *probs, const int *targets,
                          int B, int T, int V);
// backward through cross-entropy and softmax together
void crossentropy_softmax_backward(float *dlogits, const float *dlosses, const float *probs,
                                   const int *targets, int B, int T, int V);

#endif
