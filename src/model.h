#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>

// The model has Pythia-14M's shape with a character vocabulary:
//   token embedding + position embedding
//   L layers of:  x = x + attention(layernorm1(x)) + mlp(layernorm2(x))
//   final layernorm, then a linear layer to vocab-size logits
// Unlike Pythia, we use learned position embeddings instead of rotary ones.

// Model shape. model_default_config() fills it from config.h plus the vocab
// size from data/vocab.bin; tests can use a tiny one (e.g. 2 layers, hidden
// size 8) so they run fast.
typedef struct {
    int vocab_size;   // V
    int max_seq_len;  // maxT, longest sequence the model accepts
    int n_layer;      // L
    int n_head;       // NH
    int n_embd;       // C, hidden size
    int n_ff;         // FF, MLP hidden size
} ModelConfig;

// Every parameter lives in one float array (Model.params_memory); these
// pointers point into it, in this order. That lets MPI_Allreduce, cudaMemcpy,
// AdamW and checkpoint save/load treat all parameters as a single array.
// "L," in a shape means one slice per layer.
// To add a tensor: add it here, bump the count, and add it to
// param_layout() in transformer.c.
#define NUM_PARAM_TENSORS 17
typedef struct {
    float *wte;       // (V, C)        token embedding
    float *wpe;       // (maxT, C)     position embedding
    float *ln1w;      // (L, C)        layernorm before attention
    float *ln1b;      // (L, C)
    float *qkvw;      // (L, 3C, C)    attention input projection
    float *qkvb;      // (L, 3C)
    float *attprojw;  // (L, C, C)     attention output projection
    float *attprojb;  // (L, C)
    float *ln2w;      // (L, C)        layernorm before MLP
    float *ln2b;      // (L, C)
    float *fcw;       // (L, FF, C)    MLP first linear layer
    float *fcb;       // (L, FF)
    float *fcprojw;   // (L, C, FF)    MLP second linear layer
    float *fcprojb;   // (L, C)
    float *lnfw;      // (C)           final layernorm
    float *lnfb;      // (C)
    float *lm_head;   // (V, C)        hidden state -> logits (no bias)
} ParameterTensors;

// Values saved during the forward pass for use in the backward pass.
// Also one float array (Model.acts_memory). The same struct holds the
// gradient of each activation (Model.grads_acts).
#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    float *encoded;   // (B, T, C)          embedding output = input to layer 0
    float *ln1;       // (L, B, T, C)
    float *ln1_mean;  // (L, B, T)
    float *ln1_rstd;  // (L, B, T)
    float *qkv;       // (L, B, T, 3C)
    float *att;       // (L, B, NH, T, T)   attention weights after softmax
    float *atty;      // (L, B, T, C)       attention output, before projection
    float *attproj;   // (L, B, T, C)
    float *ln2;       // (L, B, T, C)
    float *ln2_mean;  // (L, B, T)
    float *ln2_rstd;  // (L, B, T)
    float *fch;       // (L, B, T, FF)      MLP hidden, before GELU
    float *fch_gelu;  // (L, B, T, FF)
    float *fcproj;    // (L, B, T, C)
    float *residual;  // (L, B, T, C)       output of each layer
    float *lnf;       // (B, T, C)
    float *lnf_mean;  // (B, T)
    float *lnf_rstd;  // (B, T)
    float *logits;    // (B, T, V)
    float *probs;     // (B, T, V)
    float *losses;    // (B, T)
} ActivationTensors;

typedef struct {
    ModelConfig config;

    // parameters and their gradients: same layout, num_parameters floats each
    ParameterTensors params;
    ParameterTensors grads;
    float *params_memory;
    float *grads_memory;
    size_t num_parameters;

    // activations and their gradients, allocated for batch_size x seq_len
    ActivationTensors acts;
    ActivationTensors grads_acts;
    float *acts_memory;
    float *grads_acts_memory;
    size_t num_activations;
    int batch_size;
    int seq_len;

    // the last forward pass (the backward pass needs these)
    int *inputs;      // (B, T) token ids
    int *targets;     // (B, T) next token ids
    int cur_B, cur_T;
    float mean_loss;  // -1 if the last forward pass had no targets
} Model;

// vocab_size comes from vocab_load() (see vocab.h)
ModelConfig model_default_config(int vocab_size);

// Allocates parameters and gradients and initializes the weights from seed.
void model_init(Model *m, ModelConfig config, uint64_t seed);

// Allocates activations for up to B sequences of up to T tokens.
void model_alloc_activations(Model *m, int B, int T);

// Runs the model on inputs (B, T). If targets is not NULL, also computes
// the loss into m->mean_loss. Next-token probabilities end up in m->acts.probs.
void model_forward(Model *m, const int *inputs, const int *targets, int B, int T);

// Sets all parameter gradients to zero. Call once per training step.
void model_zero_grad(Model *m);

// Adds the gradient of the last forward pass's loss into m->grads.
// The last model_forward call must have had targets.
void model_backward(Model *m);

void model_free(Model *m);

#endif
