// Model setup, forward pass and backward pass.
// The math for each step is in src/layers/; this file allocates memory and
// calls the layers in the right order.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "model.h"
#include "rng.h"
#include "layers/layers.h"

#define INIT_STD 0.02f

static void fail(const char *msg) {
    fprintf(stderr, "model error: %s\n", msg);
    exit(1);
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (p == NULL) fail("out of memory");
    return p;
}

ModelConfig model_default_config(int vocab_size) {
    ModelConfig c;
    c.vocab_size = vocab_size;
    c.max_seq_len = MAX_SEQ_LEN;
    c.n_layer = N_LAYER;
    c.n_head = N_HEAD;
    c.n_embd = N_EMBD;
    c.n_ff = N_FF;
    return c;
}

// ---------------------------------------------------------------------------
// memory layout

// Lists every parameter tensor and its size, in memory order.
static void param_layout(ParameterTensors *t, const ModelConfig *c,
                         float **ptrs[], size_t sizes[]) {
    size_t V = c->vocab_size, maxT = c->max_seq_len, L = c->n_layer;
    size_t C = c->n_embd, FF = c->n_ff;
    int i = 0;
#define TENSOR(field, n) (ptrs[i] = &t->field, sizes[i] = (n), i++)
    TENSOR(wte, V * C);
    TENSOR(wpe, maxT * C);
    TENSOR(ln1w, L * C);
    TENSOR(ln1b, L * C);
    TENSOR(qkvw, L * 3 * C * C);
    TENSOR(qkvb, L * 3 * C);
    TENSOR(attprojw, L * C * C);
    TENSOR(attprojb, L * C);
    TENSOR(ln2w, L * C);
    TENSOR(ln2b, L * C);
    TENSOR(fcw, L * FF * C);
    TENSOR(fcb, L * FF);
    TENSOR(fcprojw, L * C * FF);
    TENSOR(fcprojb, L * C);
    TENSOR(lnfw, C);
    TENSOR(lnfb, C);
    TENSOR(lm_head, V * C);
#undef TENSOR
    if (i != NUM_PARAM_TENSORS) fail("param_layout does not match NUM_PARAM_TENSORS");
}

// Lists every activation tensor and its size, in memory order.
static void activation_layout(ActivationTensors *t, const ModelConfig *c, int B, int T,
                              float **ptrs[], size_t sizes[]) {
    size_t L = c->n_layer, C = c->n_embd, NH = c->n_head, FF = c->n_ff, V = c->vocab_size;
    size_t BT = (size_t)B * T;
    int i = 0;
#define TENSOR(field, n) (ptrs[i] = &t->field, sizes[i] = (n), i++)
    TENSOR(encoded, BT * C);
    TENSOR(ln1, L * BT * C);
    TENSOR(ln1_mean, L * BT);
    TENSOR(ln1_rstd, L * BT);
    TENSOR(qkv, L * BT * 3 * C);
    TENSOR(att, L * B * NH * T * T);
    TENSOR(atty, L * BT * C);
    TENSOR(attproj, L * BT * C);
    TENSOR(ln2, L * BT * C);
    TENSOR(ln2_mean, L * BT);
    TENSOR(ln2_rstd, L * BT);
    TENSOR(fch, L * BT * FF);
    TENSOR(fch_gelu, L * BT * FF);
    TENSOR(fcproj, L * BT * C);
    TENSOR(residual, L * BT * C);
    TENSOR(lnf, BT * C);
    TENSOR(lnf_mean, BT);
    TENSOR(lnf_rstd, BT);
    TENSOR(logits, BT * V);
    TENSOR(probs, BT * V);
    TENSOR(losses, BT);
#undef TENSOR
    if (i != NUM_ACTIVATION_TENSORS) fail("activation_layout does not match NUM_ACTIVATION_TENSORS");
}

// Allocates one zeroed block for n tensors and points each tensor at its slice.
static float *alloc_tensors(float **ptrs[], const size_t sizes[], int n, size_t *total_out) {
    size_t total = 0;
    for (int i = 0; i < n; i++) total += sizes[i];
    float *memory = xcalloc(total, sizeof(float));
    float *p = memory;
    for (int i = 0; i < n; i++) {
        *ptrs[i] = p;
        p += sizes[i];
    }
    *total_out = total;
    return memory;
}

// One layer's slice of the per-layer parameters.
typedef struct {
    float *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb;
    float *ln2w, *ln2b, *fcw, *fcb, *fcprojw, *fcprojb;
} LayerParams;

static LayerParams layer_params(const ParameterTensors *p, const ModelConfig *c, int l) {
    size_t C = c->n_embd, FF = c->n_ff;
    LayerParams w;
    w.ln1w = p->ln1w + l * C;
    w.ln1b = p->ln1b + l * C;
    w.qkvw = p->qkvw + l * 3 * C * C;
    w.qkvb = p->qkvb + l * 3 * C;
    w.attprojw = p->attprojw + l * C * C;
    w.attprojb = p->attprojb + l * C;
    w.ln2w = p->ln2w + l * C;
    w.ln2b = p->ln2b + l * C;
    w.fcw = p->fcw + l * FF * C;
    w.fcb = p->fcb + l * FF;
    w.fcprojw = p->fcprojw + l * C * FF;
    w.fcprojb = p->fcprojb + l * C;
    return w;
}

// One layer's slice of the per-layer activations for a B x T batch.
typedef struct {
    float *ln1, *ln1_mean, *ln1_rstd, *qkv, *att, *atty, *attproj;
    float *ln2, *ln2_mean, *ln2_rstd, *fch, *fch_gelu, *fcproj, *residual;
} LayerActs;

static LayerActs layer_acts(const ActivationTensors *a, const ModelConfig *c, int l, int B, int T) {
    size_t C = c->n_embd, NH = c->n_head, FF = c->n_ff;
    size_t BT = (size_t)B * T;
    LayerActs x;
    x.ln1 = a->ln1 + l * BT * C;
    x.ln1_mean = a->ln1_mean + l * BT;
    x.ln1_rstd = a->ln1_rstd + l * BT;
    x.qkv = a->qkv + l * BT * 3 * C;
    x.att = a->att + l * (size_t)B * NH * T * T;
    x.atty = a->atty + l * BT * C;
    x.attproj = a->attproj + l * BT * C;
    x.ln2 = a->ln2 + l * BT * C;
    x.ln2_mean = a->ln2_mean + l * BT;
    x.ln2_rstd = a->ln2_rstd + l * BT;
    x.fch = a->fch + l * BT * FF;
    x.fch_gelu = a->fch_gelu + l * BT * FF;
    x.fcproj = a->fcproj + l * BT * C;
    x.residual = a->residual + l * BT * C;
    return x;
}

// The input to layer l: the embedding for layer 0, otherwise the previous layer's output.
static float *layer_input(const ActivationTensors *a, const ModelConfig *c, int l, int B, int T) {
    size_t BT = (size_t)B * T, C = c->n_embd;
    return l == 0 ? a->encoded : a->residual + (l - 1) * BT * C;
}

// ---------------------------------------------------------------------------
// setup

static void fill_normal(float *x, size_t n, float std, Rng *rng) {
    for (size_t i = 0; i < n; i++) x[i] = rng_normal(rng, 0.0f, std);
}

static void fill_value(float *x, size_t n, float value) {
    for (size_t i = 0; i < n; i++) x[i] = value;
}

// Weights are normal(0, 0.02). The two projections that write into the
// residual stream are scaled down by 1/sqrt(2L) so the residual doesn't grow
// with depth. Layernorm weights start at 1; all biases start at 0.
static void init_weights(Model *m, uint64_t seed) {
    const ModelConfig *c = &m->config;
    size_t V = c->vocab_size, maxT = c->max_seq_len, L = c->n_layer;
    size_t C = c->n_embd, FF = c->n_ff;
    ParameterTensors *p = &m->params;
    float resid_std = INIT_STD / sqrtf(2.0f * L);
    Rng rng;
    rng_seed(&rng, seed);

    fill_normal(p->wte, V * C, INIT_STD, &rng);
    fill_normal(p->wpe, maxT * C, INIT_STD, &rng);
    fill_value(p->ln1w, L * C, 1.0f);
    fill_normal(p->qkvw, L * 3 * C * C, INIT_STD, &rng);
    fill_normal(p->attprojw, L * C * C, resid_std, &rng);
    fill_value(p->ln2w, L * C, 1.0f);
    fill_normal(p->fcw, L * FF * C, INIT_STD, &rng);
    fill_normal(p->fcprojw, L * C * FF, resid_std, &rng);
    fill_value(p->lnfw, C, 1.0f);
    fill_normal(p->lm_head, V * C, INIT_STD, &rng);
}

void model_init(Model *m, ModelConfig config, uint64_t seed) {
    memset(m, 0, sizeof(*m));
    if (config.n_embd % config.n_head != 0) fail("n_embd must be divisible by n_head");
    m->config = config;

    float **ptrs[NUM_PARAM_TENSORS];
    size_t sizes[NUM_PARAM_TENSORS];
    param_layout(&m->params, &config, ptrs, sizes);
    m->params_memory = alloc_tensors(ptrs, sizes, NUM_PARAM_TENSORS, &m->num_parameters);
    param_layout(&m->grads, &config, ptrs, sizes);
    m->grads_memory = alloc_tensors(ptrs, sizes, NUM_PARAM_TENSORS, &m->num_parameters);

    init_weights(m, seed);
    m->mean_loss = -1.0f;
}

void model_alloc_activations(Model *m, int B, int T) {
    if (B < 1 || T < 1) fail("batch size and sequence length must be at least 1");
    if (T > m->config.max_seq_len) fail("sequence length is longer than max_seq_len");

    free(m->acts_memory);
    free(m->grads_acts_memory);
    free(m->inputs);
    free(m->targets);

    float **ptrs[NUM_ACTIVATION_TENSORS];
    size_t sizes[NUM_ACTIVATION_TENSORS];
    activation_layout(&m->acts, &m->config, B, T, ptrs, sizes);
    m->acts_memory = alloc_tensors(ptrs, sizes, NUM_ACTIVATION_TENSORS, &m->num_activations);
    activation_layout(&m->grads_acts, &m->config, B, T, ptrs, sizes);
    m->grads_acts_memory = alloc_tensors(ptrs, sizes, NUM_ACTIVATION_TENSORS, &m->num_activations);

    m->inputs = xcalloc((size_t)B * T, sizeof(int));
    m->targets = xcalloc((size_t)B * T, sizeof(int));
    m->batch_size = B;
    m->seq_len = T;
}

// ---------------------------------------------------------------------------
// forward and backward

void model_forward(Model *m, const int *inputs, const int *targets, int B, int T) {
    const ModelConfig *c = &m->config;
    int L = c->n_layer, C = c->n_embd, NH = c->n_head, FF = c->n_ff, V = c->vocab_size;
    size_t BT = (size_t)B * T;

    if (m->acts_memory == NULL) fail("call model_alloc_activations before model_forward");
    if (B < 1 || T < 1 || B > m->batch_size || T > m->seq_len)
        fail("B x T is bigger than the allocated activations");
    for (size_t i = 0; i < BT; i++) {
        if (inputs[i] < 0 || inputs[i] >= V) fail("input token id out of range");
        if (targets && (targets[i] < 0 || targets[i] >= V)) fail("target token id out of range");
    }

    memcpy(m->inputs, inputs, BT * sizeof(int));
    if (targets) memcpy(m->targets, targets, BT * sizeof(int));
    m->cur_B = B;
    m->cur_T = T;

    const ParameterTensors *p = &m->params;
    const ActivationTensors *a = &m->acts;

    embedding_forward(a->encoded, m->inputs, p->wte, p->wpe, B, T, C);

    for (int l = 0; l < L; l++) {
        LayerParams w = layer_params(p, c, l);
        LayerActs x = layer_acts(a, c, l, B, T);
        float *inp = layer_input(a, c, l, B, T);

        // attention branch
        layernorm_forward(x.ln1, x.ln1_mean, x.ln1_rstd, inp, w.ln1w, w.ln1b, B, T, C);
        matmul_forward(x.qkv, x.ln1, w.qkvw, w.qkvb, B, T, C, 3 * C);
        attention_forward(x.atty, x.att, x.qkv, B, T, C, NH);
        matmul_forward(x.attproj, x.atty, w.attprojw, w.attprojb, B, T, C, C);

        // MLP branch (reads the same input as attention: Pythia's parallel residual)
        layernorm_forward(x.ln2, x.ln2_mean, x.ln2_rstd, inp, w.ln2w, w.ln2b, B, T, C);
        matmul_forward(x.fch, x.ln2, w.fcw, w.fcb, B, T, C, FF);
        gelu_forward(x.fch_gelu, x.fch, BT * FF);
        matmul_forward(x.fcproj, x.fch_gelu, w.fcprojw, w.fcprojb, B, T, FF, C);

        residual_forward(x.residual, inp, x.attproj, x.fcproj, BT * C);
    }

    float *last = layer_input(a, c, L, B, T);
    layernorm_forward(a->lnf, a->lnf_mean, a->lnf_rstd, last, p->lnfw, p->lnfb, B, T, C);
    matmul_forward(a->logits, a->lnf, p->lm_head, NULL, B, T, C, V);
    softmax_forward(a->probs, a->logits, B, T, V);

    if (targets) {
        crossentropy_forward(a->losses, a->probs, m->targets, B, T, V);
        double sum = 0.0;
        for (size_t i = 0; i < BT; i++) sum += a->losses[i];
        m->mean_loss = (float)(sum / BT);
    } else {
        m->mean_loss = -1.0f;
    }
}

void model_zero_grad(Model *m) {
    memset(m->grads_memory, 0, m->num_parameters * sizeof(float));
}

void model_backward(Model *m) {
    if (m->mean_loss < 0.0f) fail("model_backward needs a forward pass with targets first");

    const ModelConfig *c = &m->config;
    int L = c->n_layer, C = c->n_embd, NH = c->n_head, FF = c->n_ff, V = c->vocab_size;
    int B = m->cur_B, T = m->cur_T;
    size_t BT = (size_t)B * T;

    const ParameterTensors *p = &m->params;
    const ParameterTensors *g = &m->grads;
    const ActivationTensors *a = &m->acts;
    const ActivationTensors *da = &m->grads_acts;

    // activation gradients start at zero; every layer adds into them
    memset(m->grads_acts_memory, 0, m->num_activations * sizeof(float));

    // the loss is the mean over all B*T positions
    for (size_t i = 0; i < BT; i++) da->losses[i] = 1.0f / BT;

    crossentropy_softmax_backward(da->logits, da->losses, a->probs, m->targets, B, T, V);
    matmul_backward(da->lnf, g->lm_head, NULL, da->logits, a->lnf, p->lm_head, B, T, C, V);
    float *last = layer_input(a, c, L, B, T);
    float *dlast = layer_input(da, c, L, B, T);
    layernorm_backward(dlast, g->lnfw, g->lnfb, da->lnf, last, p->lnfw, a->lnf_mean, a->lnf_rstd, B, T, C);

    for (int l = L - 1; l >= 0; l--) {
        LayerParams w = layer_params(p, c, l);
        LayerParams dw = layer_params(g, c, l);
        LayerActs x = layer_acts(a, c, l, B, T);
        LayerActs dx = layer_acts(da, c, l, B, T);
        float *inp = layer_input(a, c, l, B, T);
        float *dinp = layer_input(da, c, l, B, T);

        residual_backward(dinp, dx.attproj, dx.fcproj, dx.residual, BT * C);

        // MLP branch
        matmul_backward(dx.fch_gelu, dw.fcprojw, dw.fcprojb, dx.fcproj, x.fch_gelu, w.fcprojw, B, T, FF, C);
        gelu_backward(dx.fch, dx.fch_gelu, x.fch, BT * FF);
        matmul_backward(dx.ln2, dw.fcw, dw.fcb, dx.fch, x.ln2, w.fcw, B, T, C, FF);
        layernorm_backward(dinp, dw.ln2w, dw.ln2b, dx.ln2, inp, w.ln2w, x.ln2_mean, x.ln2_rstd, B, T, C);

        // attention branch
        matmul_backward(dx.atty, dw.attprojw, dw.attprojb, dx.attproj, x.atty, w.attprojw, B, T, C, C);
        attention_backward(dx.qkv, dx.att, dx.atty, x.qkv, x.att, B, T, C, NH);
        matmul_backward(dx.ln1, dw.qkvw, dw.qkvb, dx.qkv, x.ln1, w.qkvw, B, T, C, 3 * C);
        layernorm_backward(dinp, dw.ln1w, dw.ln1b, dx.ln1, inp, w.ln1w, x.ln1_mean, x.ln1_rstd, B, T, C);
    }

    embedding_backward(g->wte, g->wpe, da->encoded, m->inputs, B, T, C);
}

void model_free(Model *m) {
    free(m->params_memory);
    free(m->grads_memory);
    free(m->acts_memory);
    free(m->grads_acts_memory);
    free(m->inputs);
    free(m->targets);
    memset(m, 0, sizeof(*m));
}
