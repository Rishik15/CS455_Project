# Model walkthrough: one vector through every op

This follows a single token's vector through the whole forward pass, then back through the backward pass, and says exactly where tokens interact and which code does each step. Shapes and parameter counts are in [01-model.md](01-model.md); where the files live is in [02-structure.md](02-structure.md).

## Setup

Take the text `the cat sat on` as 4 tokens. Say the ids are `[12, 87, 301, 45]` (made up). One sequence, so `B = 1`, `T = 4`.

The task: at each position, predict the **next** token. Position 0 (`the`) must predict `cat`, position 1 (`cat`) must predict `sat`, and so on. So `inputs = [12, 87, 301, 45]` and `targets = [87, 301, 45, next]`: the same ids shifted by one.

The model is a stack of steps that keep one **vector of C = 128 floats per token**. Call the vector at position t `x_t`. Everything below is "what happens to `x_t`". The full state at any point is a `[T, 128]` table, one row per token. In code that is `[B, T, C]` row-major, so token t's vector is the 128 floats starting at `(b*T + t) * C`.

Two kinds of op:

- **Per-token ops** touch each row alone: encoder, layernorm, matmul, gelu, residual, crossentropy. Row t never sees row t2.
- **Attention** is the only op where rows read each other.

## Forward pass, in order

### 1. Encoder: id to vector

`encoder_forward` (`src/cpu/encoder.c`).

```
x_t = wte[token_t] + wpe[t]
```

`wte` is a `[4096, 128]` table: one learned 128-float row per token id. `wpe` is a `[256, 128]` table: one learned row per position. Row `token_t` of `wte` says which word it is; row `t` of `wpe` says where it is. They are added element by element. Without `wpe` the model could not tell `cat sat` from `sat cat`, because attention itself has no sense of order.

Output `encoded`, `[B, T, C]`. This is `x` entering block 0.

### 2. The block (repeated 6 times)

`block_forward` (`src/model.c`) runs two half-blocks. Each half-block is: normalize, do work in a branch, add the branch result back to `x` (the residual).

#### 2a. ln1: layernorm

`layernorm_forward` (`src/cpu/layernorm.c`). Parameters `ln1w`, `ln1b` (128 each).

Rescales one token's 128 numbers so they have mean 0 and variance 1, then applies a learned scale and shift:

```
mean = sum(x) / C
var  = sum((x - mean)^2) / C
rstd = 1 / sqrt(var + 1e-5)
out[c] = (x[c] - mean) * rstd * w[c] + b[c]
```

Tiny example with C = 4 and `x = [1, 2, 3, 4]`: mean 2.5, var 1.25, rstd 0.8944, normalized `[-1.342, -0.447, 0.447, 1.342]`. With `w = 1`, `b = 0` that is the output.

Why: the matmuls after it work best when their input has a steady scale, however large `x` has grown. `mean` and `rstd` are saved per token (`ln1_mean`, `ln1_rstd`, each `[B, T]`) for backward.

Per-token. Output `ln1`, `[B, T, C]`. The original `x` is untouched; it waits for the residual add.

#### 2b. QKV matmul: make query, key, value

`matmul_forward` (`src/cpu/matmul.c`). Parameters `qkvw` `[384, 128]`, `qkvb` `[384]`.

Matmul is the workhorse. For one token it computes 384 dot products, each between the 128-float input and one 128-float row of the weight, plus a bias:

```
out[o] = dot(x, weight[o, :]) + bias[o]      for o = 0..383
```

So each output number is a learned weighted sum of **all 128 input channels**. This is how channels mix within a token. (Weights are `[C_out, C_in]`, so row `o` of the weight is the recipe for output `o`.)

The 384 outputs are cut into three 128-float pieces:

| Slice | Name | Meaning |
|---|---|---|
| `[0, 128)` | **q**, query | what this token is looking for |
| `[128, 256)` | **k**, key | what this token offers to others |
| `[256, 384)` | **v**, value | the content this token hands over when picked |

Per-token. Output `qkv`, `[B, T, 3C]`.

#### 2c. Attention: the only place tokens talk

`attention_forward` (`src/cpu/attention.c`). No parameters.

Each of q, k, v is split into `NH = 4` heads of `hs = 32` floats. Head h uses floats `h*32 .. h*32+31` of each slice. Heads run independently, so every token gets 4 separate chances to look at the past in 4 different ways.

For token t and head h:

1. **Scores.** Dot `q_t` against the key of every token t2 up to and including t, scaled:
   `score[t2] = dot(q_t, k_t2) / sqrt(32)`
2. **Causal mask.** Tokens after t are skipped (set to zero weight), so a token never sees the future. Without this, position 0 could read the answer `cat` straight from position 1.
3. **Softmax** over those scores, so they become weights that are positive and sum to 1. Example with 3 visible tokens and scores `[2, 1, 0]`: weights `[0.665, 0.245, 0.090]`. The max score is subtracted first for numerical safety.
4. **Mix the values.** The head's output is the weighted sum of the visible tokens' value vectors:
   `out_t = sum over t2 <= t of weight[t2] * v_t2`
5. The 4 heads' 32-float outputs sit side by side and become 128 floats again.

This is the **only** step where row t reads other rows (the k and v of earlier rows). The weights are saved as `att`, `[B, NH, T, T]` (zero above the diagonal), because backward needs them.

For the example, token 2 (`sat`) forms its q, scores it against the keys of `the`, `cat` and `sat`, and takes a weighted blend of their values. A head might learn that a verb should pay most attention to the noun before it.

Output `atty`, `[B, T, C]`.

#### 2d. Attention projection matmul

`matmul_forward` with `attprojw` `[128, 128]`, `attprojb`. Mixes the four heads' outputs together and maps back into the space the residual stream uses. Per-token. Output `attproj`.

#### 2e. First residual

`residual_forward` (`src/cpu/residual.c`).

```
residual2 = x + attproj
```

`x` is the block's input, `attproj` is what attention produced. The branch only has to say "how should x change", not rebuild x. The sum goes to its own buffer `residual2`.

#### 2f. ln2, MLP up, GELU, MLP down

The second half-block works on each token alone:

| Step | Op | Shape of one token's vector |
|---|---|---|
| ln2 | `layernorm_forward`, `ln2w`, `ln2b` | 128 to 128 |
| MLP up | `matmul_forward`, `fcw` `[512, 128]`, `fcb` | 128 to 512 |
| GELU | `gelu_forward` (`src/cpu/gelu.c`) | 512 to 512 |
| MLP down | `matmul_forward`, `fcprojw` `[128, 512]`, `fcprojb` | 512 to 128 |

GELU is a smooth bend applied to every number separately: `gelu(x) = 0.5 * x * (1 + tanh(0.7979 * (x + 0.044715 * x^3)))`. Roughly: large positive numbers pass through (`gelu(2)` is 1.95), large negative ones are squashed toward 0 (`gelu(-1)` is -0.159), `gelu(0)` is 0. Without this bend, two matmuls in a row would collapse into one matmul and the MLP could not compute anything interesting. Going up to 512 gives it room to compute many small features, and going down mixes them back.

Where attention gathers information from other tokens, the MLP is where each token thinks about what it gathered.

#### 2g. Second residual

```
residual3 = residual2 + fcproj
```

`residual3` is the output of the block and the input `x` of the next block (or of the final layernorm after block 5).

After 6 blocks, `x_t` has been repeatedly nudged by attention (information from earlier tokens) and MLPs (per-token processing). Because of the residual adds, `x_t` is a running sum of all these nudges on top of the original embedding.

### 3. Final layernorm

`layernorm_forward` with `lnfw`, `lnfb`. Same op as ln1/ln2, one more time on the last block's output. Output `lnf`, `[B, T, C]`.

### 4. Logits: vector back to vocabulary

`matmul_forward` with weight `wte` and **no bias** (`NULL`).

```
logits[v] = dot(lnf_t, wte[v, :])       for v = 0..4095
```

One dot product per vocabulary entry: how well does the final vector match token v's embedding row? 4096 scores per token, `[B, T, V]`. This reuses the same `wte` that the encoder used to turn ids into vectors: tied weights. The model reads a word in and writes a word out with the same table, so the learned meaning of each word is shared in both directions. (This works directly because linear weights are stored `[C_out, C_in]` and `wte` is `[V, C]`.)

### 5. Cross-entropy: score the prediction

`crossentropy_forward` (`src/cpu/crossentropy.c`).

1. Softmax the 4096 logits into probabilities `probs`, `[B, T, V]` (saved).
2. Loss for this position is `-log(probs[target])`: the model's probability of the correct next token, as a penalty.
3. The reported loss is the mean over all `B*T` positions.

Anchors: if the model gave the right token probability 0.25, the loss is 1.386; if it guessed uniformly over 4096 tokens, the loss is `ln(4096) = 8.318`; probability 1 gives loss 0. Training pushes the loss down.

Note all T positions give a loss from one pass. One 4-token sequence is 4 training examples at once, which the causal mask makes valid: position t only ever used tokens up to t.

## Where tokens interact

| Op | Mixes channels in a token | Mixes across tokens |
|---|---|---|
| encoder | no | no |
| layernorm | yes (mean and variance over the 128 channels) | no |
| matmul | yes | no |
| gelu, residual | no | no |
| attention | no (heads use their own slices) | **yes, only earlier tokens** |
| crossentropy | over the vocabulary | no |

This split is why the parallel versions are easy to reason about: everything except attention is independent per row, and attention's reads only go backward in position.

## One token's vector, start to finish

| Where | Vector (one token) | Code |
|---|---|---|
| input | token id (int) | `inputs` |
| after encoder | 128 | `encoded` |
| ln1 | 128 | `ln1` |
| qkv | 384 = q, k, v | `qkv` |
| attention | 128 (4 heads x 32) | `atty` |
| attproj | 128 | `attproj` |
| residual2 | 128 | `residual2` |
| ln2 | 128 | `ln2` |
| fc | 512 | `fch` |
| gelu | 512 | `fch_gelu` |
| fcproj | 128 | `fcproj` |
| residual3 | 128 | `residual3` |
| (the 6 blocks repeat) | | |
| lnf | 128 | `lnf` |
| logits | 4096 | `logits` |
| probs | 4096 | `probs` |
| loss | 1 float | `losses` |

## Backward pass

The loss is one number. Backward asks, for every number in the model, "if I nudged this number up a little, how much would the loss change?" That is its **gradient**. Walk the forward pass in reverse; each op turns the gradient of its output into the gradient of its input (and of its own parameters), using the chain rule. `gpt_backward` (`src/model.c`) does the walk.

The training step then moves every parameter a small amount against its gradient (the optimizer, planned).

### Per-op rules

Starting at the end:

| Op | What backward does |
|---|---|
| `crossentropy_backward` | `dlogits = (probs - one_hot(target)) / (B*T)`. Starts the whole chain. For the correct token the gradient is `prob - 1` (negative, so raising that logit lowers the loss); for every other token it is `prob` (positive, so lowering those logits lowers the loss). The `1/(B*T)` comes from the mean. |
| `matmul_backward` | Three gradients from `dout`: `dinp[i, :] += sum_o dout[i,o] * weight[o, :]`, `dweight[o, :] += sum_i dout[i,o] * inp[i, :]`, `dbias[o] += sum_i dout[i,o]`. The weight gradient needs the saved input `inp`; the input gradient needs the weight. |
| `layernorm_backward` | Uses the saved `mean` and `rstd`. A channel's change moves the mean and variance that all 128 channels were normalized with, so the formula has two extra mean terms: `dx = rstd * (dnorm - mean(dnorm) - norm * mean(dnorm * norm))`. |
| `gelu_backward` | `dinp += dout * gelu'(inp)`, using the saved pre-GELU input `fch`. |
| `residual_backward` | The derivative of `a + b` is 1 for both, so `dout` is added unchanged into both `da` (skip) and `db` (branch). |
| `attention_backward` | Reverse of the four attention steps, using the saved `att`: gradient flows to `v` of every earlier token (weighted by `att`), through the softmax, and to both `q` of the current token and `k` of the earlier tokens. |
| `encoder_backward` | Adds each token's gradient into its row of `dwte` and its position's row of `dwpe`. |

### Gradients also run along the skip path

In a block's backward pass, the gradient arriving at `residual3` goes to **both** the branch and straight to `residual2`. Same again at `residual2`. That is `residual_backward` splitting. So each block passes gradient to the one below it along two routes, the branch and the skip. Because the skip route is just an addition, the gradient of the loss reaches layer 0 undamped. In `block_backward`:

```
residual_backward(g.residual2, g.fcproj, g.residual3, BTC);   // split into skip and branch
...branch backward: fcproj, gelu, fc, ln2 (adds into g.residual2)...
residual_backward(dresidual, g.attproj, g.residual2, BTC);    // split again
...branch backward: attproj, attention, qkv, ln1 (adds into dresidual)...
```

`dresidual` is the gradient for the block's input, i.e. the previous block's `residual3` gradient, so the next iteration of the loop picks it up.

### Tokens connect again in backward

Attention's backward is the only place a token's gradient lands on **another token's row**: token 2's loss pushes gradient into the `k` and `v` of tokens 0 and 1. That is how an early token learns "I should have offered something more useful to later tokens".

### Tied `wte` gets two gradients

`dwte` is filled twice and both are added (`+=`):

1. From the logits matmul: `dwte[v, :] += sum_i dlogits[i, v] * lnf[i, :]` (every token in the batch affects every row, through the 4096 scores).
2. From `encoder_backward`: only the rows of the tokens that actually appeared.

### Why forward saves things

Backward is a function of values from forward. So forward keeps these (all in `ActivationTensors`, `include/model.h`):

| Saved | Needed by |
|---|---|
| each matmul's input (`ln1`, `atty`, `ln2`, `fch_gelu`, `lnf`) | the weight gradient |
| `ln*_mean`, `ln*_rstd` and the layernorm input | `layernorm_backward` |
| `fch` (pre-GELU) | `gelu_backward` |
| `qkv` and `att` | `attention_backward` |
| `probs` | `crossentropy_backward` |
| `inputs`, `targets` (copied into the `GPT` struct) | `encoder_backward`, `crossentropy_backward` |

The activations stay allocated through the whole step, which is why `acts_memory` is sized for the full batch.

### Accumulate, never overwrite

Every `_backward` function uses `+=`. A value that feeds two places (like `wte`, or `residual2` which feeds the branch and the skip) receives two contributions, and `+=` sums them. The cost is that gradients must start at zero: `gpt_zero_grad` for parameters once per step, and `gpt_backward` clears `grads_acts_memory` itself at the start.

## Reading the code in this order

1. `gpt_forward` and `block_forward` in `src/model.c`: the order above, as calls.
2. `encoder.c`, `residual.c`, `gelu.c`: one loop each.
3. `matmul.c`, `layernorm.c`: per-token loops.
4. `crossentropy.c`.
5. `attention.c`: last, since it has the only cross-token logic.
6. `gpt_backward` and `block_backward`: the same sequence mirrored.
