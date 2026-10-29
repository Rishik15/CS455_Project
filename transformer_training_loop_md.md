# transformer training loop:
## 1. tokenize input
- prep_data.c
## 2. forward pass
- **embedding forward** (done in embedding.c)
- **LayerNormforward**: normalizes each token vector, applies scale and bias, and save mean and inverse std for backward
- **matrix mult**: input * weight plus bias (heaviest)
- **attention forward**
- **GELU forward**: MLP activation
- **Residual add forward**: element-wise addition
- **transformer block forward**: layernorm, attention, projection, residual, layernorm, MLP up, GELU, MLP down, residual
- **Model forward**: embedding, N blocks, final layernorm, output projection to vocab logits
- **Loss**
	- softmax + cross-entropy: turns logits into probabilities and computes mean negative log-likelihood of the targets
## 4. **backward pass**
- **cross-entropy+softmax backward**: gradient of the logits is simply the probabilities minus the one-hot target (one-hot encoding)
- **matmul backward**: produces two gradients, one for the input and one for the weight (plus bias)
- **GELU backward**
- **Attention backward**: goes back through the weighted sum, the softmax, and the score computation to get gradients for Q, K, V
- **LayerNorm backward**: uses the saved mean and std
- **Residual backward**: passes the gradient through to both branches
- **Embedding backward**: accumulates gradients into the rows of tokens that were used (embedding.c)
- **model backward**: calls everything above in exact order of the forward pass
## 5. optimizer
- **AdamW update**: for each parameter, updates the moment estimates, applies bias correction, applies weight decay, and updates the weights
- **Optional**: gradient norm clippling and a learning-rate schedule function
## 6. reset
- **zero gradients**: sets the gradient block to zero

# **training loop**:
- **train step**: get batch, model forward, loss, zero grad, model backward, optimizer update
- **main training loop**: runs trains steps, logs the loss, occasionally evaluates on validation data
- **Sampling/generation** (optional but very useful): runs forward on a prompt and samples the next token repeatedly so you can see if its learning

## One function to save a lot of pain:
- gradient check: for a few random params, nudge the values slightly, return forward, and compare the numerical slope to your analytic gradient. Test each backward function this way as you write it, especially attention and layer norm
