/*
 * model.h
 * The GPT model object. All parameters live in one contiguous buffer and all
 * gradients in a second buffer with the same layout, so MPI can allreduce and
 * CUDA can copy them in a single call. ParameterTensors and
 * ActivationTensors are named pointers into those buffers.
 */
#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>
#include "config.h"

#define NUM_PARAM_TENSORS 16
#define NUM_ACT_TENSORS 22

/*
 * ParameterTensors
 * Shapes (V vocab, Tm max_seq_len, L layers, C channels); per-layer tensors
 * are stacked, layer l starts at l times the per-layer size:
 *   wte [V, C]          wpe [Tm, C]
 *   ln1w, ln1b [L, C]   qkvw [L, 3C, C]     qkvb [L, 3C]
 *   attprojw [L, C, C]  attprojb [L, C]     ln2w, ln2b [L, C]
 *   fcw [L, 4C, C]      fcb [L, 4C]         fcprojw [L, C, 4C]   fcprojb [L, C]
 *   lnfw, lnfb [C]
 */
typedef struct {
    float *wte;
    float *wpe;
    float *ln1w;
    float *ln1b;
    float *qkvw;
    float *qkvb;
    float *attprojw;
    float *attprojb;
    float *ln2w;
    float *ln2b;
    float *fcw;
    float *fcb;
    float *fcprojw;
    float *fcprojb;
    float *lnfw;
    float *lnfb;
} ParameterTensors;

/*
 * ActivationTensors
 * Everything the forward pass saves for the backward pass (NH heads):
 *   encoded [B, T, C]
 *   per layer, stacked over L: ln1 [B,T,C], ln1_mean/ln1_rstd [B,T],
 *     qkv [B,T,3C], atty [B,T,C], att [B,NH,T,T], attproj [B,T,C],
 *     residual2 [B,T,C], ln2 [B,T,C], ln2_mean/ln2_rstd [B,T],
 *     fch/fch_gelu [B,T,4C], fcproj [B,T,C], residual3 [B,T,C]
 *   lnf [B,T,C], lnf_mean/lnf_rstd [B,T], logits/probs [B,T,V], losses [B,T]
 * The same struct is reused for the gradients of these activations.
 */
typedef struct {
    float *encoded;
    float *ln1;
    float *ln1_mean;
    float *ln1_rstd;
    float *qkv;
    float *atty;
    float *att;
    float *attproj;
    float *residual2;
    float *ln2;
    float *ln2_mean;
    float *ln2_rstd;
    float *fch;
    float *fch_gelu;
    float *fcproj;
    float *residual3;
    float *lnf;
    float *lnf_mean;
    float *lnf_rstd;
    float *logits;
    float *probs;
    float *losses;
} ActivationTensors;

/*
 * GPT
 * params/grads point into params_memory/grads_memory (num_parameters floats
 * each). acts/grads_acts point into acts_memory/grads_acts_memory, sized for
 * max_batch x max_seq. B and T are the sizes used by the most recent forward.
 */
typedef struct {
    GPTConfig config;
    ParameterTensors params;
    ParameterTensors grads;
    float *params_memory;
    float *grads_memory;
    size_t param_sizes[NUM_PARAM_TENSORS];
    size_t num_parameters;
    ActivationTensors acts;
    ActivationTensors grads_acts;
    float *acts_memory;
    float *grads_acts_memory;
    size_t act_sizes[NUM_ACT_TENSORS];
    size_t num_activations;
    int max_batch;
    int max_seq;
    int B;
    int T;
    int *inputs;
    int *targets;
    int has_targets;
    float mean_loss;
} GPT;

/* Allocates every buffer for up to max_batch x max_seq tokens and initializes weights from seed. */
void gpt_build(GPT *model, GPTConfig config, int max_batch, int max_seq, uint64_t seed);

/* Runs the forward pass; returns the mean loss, or -1 when targets is NULL (inference). */
float gpt_forward(GPT *model, const int *inputs, const int *targets, int B, int T);

/* Sets every parameter gradient to zero; call once per step before gpt_backward. */
void gpt_zero_grad(GPT *model);

/* Backward pass for the most recent gpt_forward call that had targets. */
void gpt_backward(GPT *model);

/* Prints the configuration, parameter count, and memory use. */
void gpt_print_summary(const GPT *model);

/* Frees every buffer owned by the model. */
void gpt_free(GPT *model);

#endif
