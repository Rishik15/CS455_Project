/*
 * model.c
 * The GPT model: allocates one contiguous buffer each for parameters,
 * parameter gradients, activations, and activation gradients, initializes the
 * weights, and wires the ops from ops.h into the forward and backward passes.
 * This file is the same for every training version; only the linked ops change.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "ops.h"
#include "utils.h"

#define INIT_STD 0.02f

typedef struct {
    float *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb;
    float *ln2w, *ln2b, *fcw, *fcb, *fcprojw, *fcprojb;
} LayerParams;

typedef struct {
    float *ln1, *ln1_mean, *ln1_rstd, *qkv, *atty, *att, *attproj, *residual2;
    float *ln2, *ln2_mean, *ln2_rstd, *fch, *fch_gelu, *fcproj, *residual3;
} LayerActs;

/*
 * fatal
 * Prints an error message and exits.
 */
static void fatal(const char *message) {
    fprintf(stderr, "gpt: %s\n", message);
    exit(1);
}

/*
 * check_config
 * Rejects shapes the ops cannot handle before anything is allocated.
 */
static void check_config(GPTConfig c, int max_batch, int max_seq) {
    if (c.vocab_size < 1 || c.n_layers < 1 || c.n_heads < 1 || c.channels < 1) {
        fatal("every config value must be positive");
    }
    if (c.channels % c.n_heads != 0) {
        fatal("channels must be divisible by n_heads");
    }
    if (max_batch < 1 || max_seq < 1 || max_seq > c.max_seq_len) {
        fatal("need max_batch >= 1 and 1 <= max_seq <= config.max_seq_len");
    }
}

/*
 * param_sizes
 * Number of floats in each parameter tensor, in ParameterTensors order.
 */
static void param_sizes(size_t *s, GPTConfig c) {
    size_t V = c.vocab_size, Tm = c.max_seq_len, L = c.n_layers, C = c.channels;
    s[0] = V * C;
    s[1] = Tm * C;
    s[2] = L * C;
    s[3] = L * C;
    s[4] = L * 3 * C * C;
    s[5] = L * 3 * C;
    s[6] = L * C * C;
    s[7] = L * C;
    s[8] = L * C;
    s[9] = L * C;
    s[10] = L * 4 * C * C;
    s[11] = L * 4 * C;
    s[12] = L * C * 4 * C;
    s[13] = L * C;
    s[14] = C;
    s[15] = C;
}

/*
 * act_sizes
 * Number of floats in each activation tensor for B x T tokens,
 * in ActivationTensors order.
 */
static void act_sizes(size_t *s, GPTConfig c, int B, int T) {
    size_t BT = (size_t)B * T, L = c.n_layers, C = c.channels;
    size_t V = c.vocab_size, NH = c.n_heads;
    s[0] = BT * C;
    s[1] = L * BT * C;
    s[2] = L * BT;
    s[3] = L * BT;
    s[4] = L * BT * 3 * C;
    s[5] = L * BT * C;
    s[6] = L * B * NH * T * T;
    s[7] = L * BT * C;
    s[8] = L * BT * C;
    s[9] = L * BT * C;
    s[10] = L * BT;
    s[11] = L * BT;
    s[12] = L * BT * 4 * C;
    s[13] = L * BT * 4 * C;
    s[14] = L * BT * C;
    s[15] = L * BT * C;
    s[16] = BT * C;
    s[17] = BT;
    s[18] = BT;
    s[19] = BT * V;
    s[20] = BT * V;
    s[21] = BT;
}

/*
 * param_list
 * Addresses of every pointer field in ParameterTensors, in declaration order.
 */
static void param_list(ParameterTensors *p, float **list[NUM_PARAM_TENSORS]) {
    float **fields[NUM_PARAM_TENSORS] = {
        &p->wte, &p->wpe, &p->ln1w, &p->ln1b, &p->qkvw, &p->qkvb,
        &p->attprojw, &p->attprojb, &p->ln2w, &p->ln2b, &p->fcw, &p->fcb,
        &p->fcprojw, &p->fcprojb, &p->lnfw, &p->lnfb};
    memcpy(list, fields, sizeof(fields));
}

/*
 * act_list
 * Addresses of every pointer field in ActivationTensors, in declaration order.
 */
static void act_list(ActivationTensors *a, float **list[NUM_ACT_TENSORS]) {
    float **fields[NUM_ACT_TENSORS] = {
        &a->encoded, &a->ln1, &a->ln1_mean, &a->ln1_rstd, &a->qkv, &a->atty,
        &a->att, &a->attproj, &a->residual2, &a->ln2, &a->ln2_mean, &a->ln2_rstd,
        &a->fch, &a->fch_gelu, &a->fcproj, &a->residual3, &a->lnf, &a->lnf_mean,
        &a->lnf_rstd, &a->logits, &a->probs, &a->losses};
    memcpy(list, fields, sizeof(fields));
}

/*
 * alloc_tensors
 * Allocates one zeroed buffer holding all n tensors back to back and points
 * each field at its slice. Writes the total float count to *total.
 */
static float *alloc_tensors(float **fields[], const size_t *sizes, int n,
                            size_t *total, const char *what) {
    size_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += sizes[i];
    }
    float *memory = xcalloc(sum, sizeof(float), what);
    float *p = memory;
    for (int i = 0; i < n; i++) {
        *fields[i] = p;
        p += sizes[i];
    }
    *total = sum;
    return memory;
}

/*
 * fill_normal
 * Fills n floats with Gaussian noise of mean 0 and the given std.
 */
static void fill_normal(float *x, size_t n, float std) {
    for (size_t i = 0; i < n; i++) {
        x[i] = rng_normal(0.0f, std);
    }
}

/*
 * fill_value
 * Sets n floats to value.
 */
static void fill_value(float *x, size_t n, float value) {
    for (size_t i = 0; i < n; i++) {
        x[i] = value;
    }
}

/*
 * init_params
 * GPT-2 style init: weights ~ N(0, 0.02), the two projections that write into
 * the residual stream scaled down by sqrt(2L) so the stream does not grow with
 * depth, layernorm scales 1, all biases 0 (already zero from xcalloc).
 */
static void init_params(GPT *m, uint64_t seed) {
    ParameterTensors *p = &m->params;
    const size_t *s = m->param_sizes;
    float proj_std = INIT_STD / sqrtf(2.0f * m->config.n_layers);
    rng_seed(seed);
    fill_normal(p->wte, s[0], INIT_STD);
    fill_normal(p->wpe, s[1], INIT_STD);
    fill_value(p->ln1w, s[2], 1.0f);
    fill_normal(p->qkvw, s[4], INIT_STD);
    fill_normal(p->attprojw, s[6], proj_std);
    fill_value(p->ln2w, s[8], 1.0f);
    fill_normal(p->fcw, s[10], INIT_STD);
    fill_normal(p->fcprojw, s[12], proj_std);
    fill_value(p->lnfw, s[14], 1.0f);
}

/*
 * gpt_build
 * Allocates parameters, gradients, activations, activation gradients, and the
 * saved inputs/targets, then initializes the weights.
 */
void gpt_build(GPT *model, GPTConfig config, int max_batch, int max_seq, uint64_t seed) {
    float **pfields[NUM_PARAM_TENSORS], **gfields[NUM_PARAM_TENSORS];
    float **afields[NUM_ACT_TENSORS], **gafields[NUM_ACT_TENSORS];
    size_t tokens = (size_t)max_batch * max_seq;
    check_config(config, max_batch, max_seq);
    memset(model, 0, sizeof(*model));
    model->config = config;
    model->max_batch = max_batch;
    model->max_seq = max_seq;
    param_sizes(model->param_sizes, config);
    param_list(&model->params, pfields);
    param_list(&model->grads, gfields);
    model->params_memory = alloc_tensors(pfields, model->param_sizes, NUM_PARAM_TENSORS,
                                         &model->num_parameters, "parameters");
    model->grads_memory = alloc_tensors(gfields, model->param_sizes, NUM_PARAM_TENSORS,
                                        &model->num_parameters, "parameter gradients");
    act_sizes(model->act_sizes, config, max_batch, max_seq);
    act_list(&model->acts, afields);
    act_list(&model->grads_acts, gafields);
    model->acts_memory = alloc_tensors(afields, model->act_sizes, NUM_ACT_TENSORS,
                                       &model->num_activations, "activations");
    model->grads_acts_memory = alloc_tensors(gafields, model->act_sizes, NUM_ACT_TENSORS,
                                             &model->num_activations, "activation gradients");
    model->inputs = xmalloc(tokens * sizeof(int), "inputs");
    model->targets = xmalloc(tokens * sizeof(int), "targets");
    init_params(model, seed);
}

/*
 * layer_params
 * Pointers to layer l's slice of every per-layer parameter tensor.
 * Works on both params and grads since they share a layout.
 */
static LayerParams layer_params(const ParameterTensors *p, int l, int C) {
    size_t i = l, c = C;
    LayerParams lp;
    lp.ln1w = p->ln1w + i * c;
    lp.ln1b = p->ln1b + i * c;
    lp.qkvw = p->qkvw + i * 3 * c * c;
    lp.qkvb = p->qkvb + i * 3 * c;
    lp.attprojw = p->attprojw + i * c * c;
    lp.attprojb = p->attprojb + i * c;
    lp.ln2w = p->ln2w + i * c;
    lp.ln2b = p->ln2b + i * c;
    lp.fcw = p->fcw + i * 4 * c * c;
    lp.fcb = p->fcb + i * 4 * c;
    lp.fcprojw = p->fcprojw + i * c * 4 * c;
    lp.fcprojb = p->fcprojb + i * c;
    return lp;
}

/*
 * layer_acts
 * Pointers to layer l's slice of every per-layer activation tensor for the
 * current B x T. Works on both acts and grads_acts.
 */
static LayerActs layer_acts(const ActivationTensors *a, int l, int B, int T, int C, int NH) {
    size_t i = l, BT = (size_t)B * T, BTC = BT * C;
    LayerActs la;
    la.ln1 = a->ln1 + i * BTC;
    la.ln1_mean = a->ln1_mean + i * BT;
    la.ln1_rstd = a->ln1_rstd + i * BT;
    la.qkv = a->qkv + i * 3 * BTC;
    la.atty = a->atty + i * BTC;
    la.att = a->att + i * (size_t)B * NH * T * T;
    la.attproj = a->attproj + i * BTC;
    la.residual2 = a->residual2 + i * BTC;
    la.ln2 = a->ln2 + i * BTC;
    la.ln2_mean = a->ln2_mean + i * BT;
    la.ln2_rstd = a->ln2_rstd + i * BT;
    la.fch = a->fch + i * 4 * BTC;
    la.fch_gelu = a->fch_gelu + i * 4 * BTC;
    la.fcproj = a->fcproj + i * BTC;
    la.residual3 = a->residual3 + i * BTC;
    return la;
}

/*
 * check_tokens
 * Exits if any token id is outside [0, V), which would read past wte.
 */
static void check_tokens(const int *tokens, size_t n, int V) {
    for (size_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= V) {
            fatal("token id out of range for vocab_size");
        }
    }
}

/*
 * block_forward
 * One transformer block (pre-LayerNorm):
 *   x2 = x + attproj(attention(qkv(ln1(x))))
 *   x3 = x2 + fcproj(gelu(fc(ln2(x2))))
 * residual is x, the block's input from the residual stream.
 */
static void block_forward(GPT *m, int l, const float *residual, int B, int T) {
    int C = m->config.channels, NH = m->config.n_heads;
    size_t BTC = (size_t)B * T * C;
    LayerParams w = layer_params(&m->params, l, C);
    LayerActs a = layer_acts(&m->acts, l, B, T, C, NH);
    layernorm_forward(a.ln1, a.ln1_mean, a.ln1_rstd, residual, w.ln1w, w.ln1b, B, T, C);
    matmul_forward(a.qkv, a.ln1, w.qkvw, w.qkvb, B, T, C, 3 * C);
    attention_forward(a.atty, a.att, a.qkv, B, T, C, NH);
    matmul_forward(a.attproj, a.atty, w.attprojw, w.attprojb, B, T, C, C);
    residual_forward(a.residual2, residual, a.attproj, BTC);
    layernorm_forward(a.ln2, a.ln2_mean, a.ln2_rstd, a.residual2, w.ln2w, w.ln2b, B, T, C);
    matmul_forward(a.fch, a.ln2, w.fcw, w.fcb, B, T, C, 4 * C);
    gelu_forward(a.fch_gelu, a.fch, 4 * BTC);
    matmul_forward(a.fcproj, a.fch_gelu, w.fcprojw, w.fcprojb, B, T, 4 * C, C);
    residual_forward(a.residual3, a.residual2, a.fcproj, BTC);
}

/*
 * gpt_forward
 * Embeds the tokens, runs every block, applies the final layernorm and the
 * tied output layer (logits = lnf @ wte^T). With targets, also computes the
 * loss and saves what gpt_backward needs. B and T may be smaller than the
 * sizes the model was built with (useful for generation).
 */
float gpt_forward(GPT *model, const int *inputs, const int *targets, int B, int T) {
    GPTConfig c = model->config;
    int C = c.channels, L = c.n_layers, V = c.vocab_size;
    size_t BT = (size_t)B * T, BTC = BT * C;
    ActivationTensors *a = &model->acts;
    ParameterTensors *w = &model->params;
    if (B < 1 || T < 1 || B > model->max_batch || T > model->max_seq) {
        fatal("B or T is outside the sizes the model was built for");
    }
    check_tokens(inputs, BT, V);
    model->B = B;
    model->T = T;
    memcpy(model->inputs, inputs, BT * sizeof(int));
    encoder_forward(a->encoded, model->inputs, w->wte, w->wpe, B, T, C);
    for (int l = 0; l < L; l++) {
        const float *residual = l == 0 ? a->encoded : a->residual3 + (size_t)(l - 1) * BTC;
        block_forward(model, l, residual, B, T);
    }
    const float *last = a->residual3 + (size_t)(L - 1) * BTC;
    layernorm_forward(a->lnf, a->lnf_mean, a->lnf_rstd, last, w->lnfw, w->lnfb, B, T, C);
    matmul_forward(a->logits, a->lnf, w->wte, NULL, B, T, C, V);
    model->has_targets = targets != NULL;
    if (!model->has_targets) {
        model->mean_loss = -1.0f;
        return -1.0f;
    }
    check_tokens(targets, BT, V);
    memcpy(model->targets, targets, BT * sizeof(int));
    model->mean_loss = crossentropy_forward(a->probs, a->losses, a->logits,
                                            model->targets, B, T, V);
    return model->mean_loss;
}

/*
 * gpt_zero_grad
 * Parameter gradients accumulate (+=), so they are cleared once per step.
 */
void gpt_zero_grad(GPT *model) {
    memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
}

/*
 * block_backward
 * Reverse of block_forward. Expects the gradient for this block's output
 * (residual3) to already be in grads_acts; adds the gradient for the block's
 * input into dresidual and every parameter gradient into grads.
 */
static void block_backward(GPT *m, int l, const float *residual, float *dresidual,
                           int B, int T) {
    int C = m->config.channels, NH = m->config.n_heads;
    size_t BTC = (size_t)B * T * C;
    LayerParams w = layer_params(&m->params, l, C);
    LayerParams dw = layer_params(&m->grads, l, C);
    LayerActs a = layer_acts(&m->acts, l, B, T, C, NH);
    LayerActs g = layer_acts(&m->grads_acts, l, B, T, C, NH);
    residual_backward(g.residual2, g.fcproj, g.residual3, BTC);
    matmul_backward(g.fch_gelu, dw.fcprojw, dw.fcprojb, g.fcproj, a.fch_gelu, w.fcprojw,
                    B, T, 4 * C, C);
    gelu_backward(g.fch, a.fch, g.fch_gelu, 4 * BTC);
    matmul_backward(g.ln2, dw.fcw, dw.fcb, g.fch, a.ln2, w.fcw, B, T, C, 4 * C);
    layernorm_backward(g.residual2, dw.ln2w, dw.ln2b, g.ln2, a.residual2, w.ln2w,
                       a.ln2_mean, a.ln2_rstd, B, T, C);
    residual_backward(dresidual, g.attproj, g.residual2, BTC);
    matmul_backward(g.atty, dw.attprojw, dw.attprojb, g.attproj, a.atty, w.attprojw,
                    B, T, C, C);
    attention_backward(g.qkv, g.atty, a.qkv, a.att, B, T, C, NH);
    matmul_backward(g.ln1, dw.qkvw, dw.qkvb, g.qkv, a.ln1, w.qkvw, B, T, C, 3 * C);
    layernorm_backward(dresidual, dw.ln1w, dw.ln1b, g.ln1, residual, w.ln1w,
                       a.ln1_mean, a.ln1_rstd, B, T, C);
}

/*
 * gpt_backward
 * Clears the activation gradients, then runs every op's backward in reverse:
 * loss, output layer, final layernorm, blocks L-1..0, and the embeddings.
 * dwte receives gradient twice (output layer and embedding), both added.
 */
void gpt_backward(GPT *model) {
    GPTConfig c = model->config;
    int B = model->B, T = model->T, C = c.channels, L = c.n_layers, V = c.vocab_size;
    size_t BTC = (size_t)B * T * C, last = (size_t)(L - 1) * BTC;
    ActivationTensors *a = &model->acts, *g = &model->grads_acts;
    ParameterTensors *w = &model->params, *dw = &model->grads;
    if (!model->has_targets) {
        fatal("gpt_backward needs a gpt_forward call with targets first");
    }
    memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
    crossentropy_backward(g->logits, a->probs, model->targets, B, T, V);
    matmul_backward(g->lnf, dw->wte, NULL, g->logits, a->lnf, w->wte, B, T, C, V);
    layernorm_backward(g->residual3 + last, dw->lnfw, dw->lnfb, g->lnf, a->residual3 + last,
                       w->lnfw, a->lnf_mean, a->lnf_rstd, B, T, C);
    for (int l = L - 1; l >= 0; l--) {
        size_t prev = (size_t)(l - 1) * BTC;
        const float *residual = l == 0 ? a->encoded : a->residual3 + prev;
        float *dresidual = l == 0 ? g->encoded : g->residual3 + prev;
        block_backward(model, l, residual, dresidual, B, T);
    }
    encoder_backward(dw->wte, dw->wpe, g->encoded, model->inputs, B, T, C);
}

/*
 * gpt_print_summary
 * Shape, parameter count, and memory for params+grads and acts+grads.
 */
void gpt_print_summary(const GPT *model) {
    GPTConfig c = model->config;
    double mb = 1024.0 * 1024.0;
    printf("GPT: vocab %d, context %d, layers %d, heads %d, channels %d\n",
           c.vocab_size, c.max_seq_len, c.n_layers, c.n_heads, c.channels);
    printf("parameters: %zu (%.1f MB with gradients)\n", model->num_parameters,
           2.0 * model->num_parameters * sizeof(float) / mb);
    printf("activations for B=%d, T=%d: %zu floats (%.1f MB with gradients)\n",
           model->max_batch, model->max_seq, model->num_activations,
           2.0 * model->num_activations * sizeof(float) / mb);
}

/*
 * gpt_free
 * Releases every buffer and clears the struct.
 */
void gpt_free(GPT *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
    memset(model, 0, sizeof(*model));
}
