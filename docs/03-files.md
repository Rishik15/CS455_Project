# File by file

What each existing source file does. Model shapes are in [01-model.md](01-model.md), the layers and include rules in [02-structure.md](02-structure.md), and a worked trace of one vector in [model-walkthrough.md](model-walkthrough.md). Files that are still empty are not covered here.

Order: headers, `src/`, `src/cpu/` in forward-pass order, then `tools/`.

## Headers

### `include/config.h`

- **Purpose:** `GPTConfig`, the model shape: `vocab_size`, `max_seq_len`, `n_layers`, `n_heads`, `channels`. Batch size and learning rate are not here; they belong to the training loop.
- **Used by:** `include/model.h` (so `src/model.c` and the training loops).
- **Functions:**

| Function | What it does | Notes |
|---|---|---|
| `gpt_config_default` | Returns `{4096, 256, 6, 4, 128}` | `static inline` in the header, so any file can call it without a `.c` file |

### `include/ops.h`

- **Purpose:** the contract between the model and the op implementations. `src/cpu/` implements every prototype here, and `src/cuda/` will implement the same names. The doc comment at the top fixes the conventions: activations `[B, T, C]` row-major, linear weights `[C_out, C_in]`, every backward accumulates (`+=`).
- **Used by:** `src/model.c` and every file in `src/cpu/`.
- **Include guard:** `#ifndef OPS_H ... #endif`. If two files both include `ops.h` and a third includes both of them, the guard makes the compiler read the contents once, not twice. Twice would be a redefinition error.
- **Prototypes:** 14 functions, a forward and a backward for each of `encoder`, `matmul`, `residual`, `gelu`, `layernorm`, `attention`, `crossentropy`. Each is documented in its `.c` file below.

### `include/utils.h` and `src/utils.c`

- **Purpose:** checked allocation, a seedable random number generator and a timer.
- **Used by:** `src/model.c` (`xmalloc`, `xcalloc`, `rng_seed`, `rng_normal`). `time_seconds` is for the training loops.
- **Functions:**

| Function | What it does | Notes |
|---|---|---|
| `xmalloc(bytes, what)` | `malloc` that prints `out of memory allocating <what>` and exits instead of returning `NULL` | |
| `xcalloc(count, size, what)` | same for `calloc`, zeroed memory | |
| `rng_seed(seed)` | sets the generator state | zero is replaced, since xorshift never leaves the all-zero state |
| `rng_next` (internal) | advances xorshift64* and returns 64 bits | state is a file-level `static uint64_t` |
| `rng_uniform()` | uniform float strictly between 0 and 1 | |
| `rng_normal(mean, std)` | Gaussian sample | Box-Muller from two uniforms |
| `time_seconds()` | wall-clock seconds | `CLOCK_MONOTONIC` |

- **Details that matter:**
  - The RNG is xorshift64*, not `rand()`. `rand()` gives different numbers on different systems; we need identical weights on every machine to compare the four versions.
  - `rng_uniform` takes the top 24 bits and adds half a step, so every value is exactly representable as a float and the result is never 0 or 1. This protects `log(u1)` in Box-Muller from `log(0)`.
  - `time_seconds` and `clock_gettime` need `#define _POSIX_C_SOURCE 199309L` at the top of `utils.c`, because `-std=c99` hides POSIX functions otherwise.

### `include/model.h`

- **Purpose:** the `GPT` object and its API.
- **Used by:** `src/model.c`, and the training loops, `generate.c` and the tests later.
- **Structs:**

| Struct | What it is |
|---|---|
| `ParameterTensors` | 16 named `float *`: `wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb, lnfw, lnfb`. Shapes are in the doc comment, e.g. `qkvw [L, 3C, C]`. Per-layer tensors are stacked, so layer l starts at l times the per-layer size. |
| `ActivationTensors` | 22 named `float *`: `encoded`, then per layer (stacked over L) `ln1, ln1_mean, ln1_rstd, qkv, atty, att, attproj, residual2, ln2, ln2_mean, ln2_rstd, fch, fch_gelu, fcproj, residual3`, then `lnf, lnf_mean, lnf_rstd, logits, probs, losses`. The same struct type holds the gradients of the activations. |
| `GPT` | `config`; `params` and `grads` with their backing buffers `params_memory` and `grads_memory`; `acts` and `grads_acts` with `acts_memory` and `grads_acts_memory`; the size arrays `param_sizes` and `act_sizes`; `num_parameters`, `num_activations`; `max_batch`, `max_seq`; `B`, `T` from the latest forward; copies of `inputs` and `targets`; `has_targets`; `mean_loss`. |

- **Constants:** `NUM_PARAM_TENSORS` 16, `NUM_ACT_TENSORS` 22.
- **Public API:** see `src/model.c` below.

## `src/`

### `src/model.c`

- **Purpose:** allocates the four big buffers, initializes the weights, and wires the ops from `ops.h` into the forward and backward passes. The same file is used by every version; only the linked ops change.
- **Used by:** the training loops.

**Public API:**

| Function | What it does | When to call |
|---|---|---|
| `gpt_build(model, config, max_batch, max_seq, seed)` | validates the config, allocates everything for up to `max_batch x max_seq` tokens, initializes the weights from `seed` | once at start |
| `gpt_forward(model, inputs, targets, B, T)` | forward pass; returns the mean loss, or `-1` when `targets` is `NULL` | every step; with `NULL` targets for generation |
| `gpt_zero_grad(model)` | zeroes every parameter gradient | once per step, before `gpt_backward` |
| `gpt_backward(model)` | backward pass for the latest forward that had targets | every step after `gpt_forward` |
| `gpt_print_summary(model)` | prints config, parameter count, memory | at start |
| `gpt_free(model)` | frees every buffer and clears the struct | once at end |

**Internal helpers:**

| Function | What it does |
|---|---|
| `fatal(message)` | prints `gpt: <message>` and exits |
| `check_config(c, max_batch, max_seq)` | rejects non-positive values, `channels % n_heads != 0`, and `max_seq` outside `1..config.max_seq_len` |
| `param_sizes(s, c)` | number of floats in each of the 16 parameter tensors, in struct order |
| `act_sizes(s, c, B, T)` | number of floats in each of the 22 activation tensors for `B x T` |
| `param_list(p, list)` | addresses of the 16 pointer fields of a `ParameterTensors`, in declaration order |
| `act_list(a, list)` | the same for `ActivationTensors` |
| `alloc_tensors(fields, sizes, n, total, what)` | one zeroed buffer for n tensors back to back; sets each field to its slice; writes the total |
| `fill_normal(x, n, std)`, `fill_value(x, n, value)` | fill with Gaussian noise or a constant |
| `init_params(m, seed)` | seeds the RNG and sets all weights (see [01-model.md](01-model.md#initialization)) |
| `layer_params(p, l, C)` | `LayerParams`: pointers to layer l's slice of each per-layer parameter tensor |
| `layer_acts(a, l, B, T, C, NH)` | `LayerActs`: pointers to layer l's slice of each per-layer activation tensor |
| `check_tokens(tokens, n, V)` | exits if any id is outside `[0, V)`, which would read past `wte` |
| `block_forward(m, l, residual, B, T)` | one transformer block |
| `block_backward(m, l, residual, dresidual, B, T)` | reverse of `block_forward` |

**Details that matter:**

- `alloc_tensors` is what makes the four buffers contiguous (see [02-structure.md](02-structure.md#one-contiguous-buffer-per-kind)). It uses `xcalloc`, so parameters and gradients start at zero; `init_params` then fills the weights, and biases stay zero.
- `layer_params` and `layer_acts` take any `ParameterTensors` or `ActivationTensors`, so they work for both values and gradients.
- `gpt_forward` accepts `B` and `T` smaller than the sizes given to `gpt_build` (generation uses short sequences), but exits if they are larger. It exits on out-of-range token ids, copies the inputs and targets into the model (backward needs them), and returns `-1` when `targets` is `NULL`.
- `gpt_backward` zeroes the activation gradients itself (`grads_acts_memory`). `gpt_zero_grad` only clears parameter gradients. Parameter gradients must be zeroed once per step because every backward adds into them. `gpt_backward` exits if the last forward had no targets.

**`block_forward`** (`x` is the `residual` argument):

| Call | Diagram box ([01-model.md](01-model.md#architecture)) |
|---|---|
| `layernorm_forward(a.ln1, ..., residual, ...)` | layernorm (ln1) |
| `matmul_forward(a.qkv, a.ln1, w.qkvw, w.qkvb, C, 3C)` | matmul QKV |
| `attention_forward(a.atty, a.att, a.qkv, ...)` | attention |
| `matmul_forward(a.attproj, a.atty, w.attprojw, ..., C, C)` | matmul attention output |
| `residual_forward(a.residual2, residual, a.attproj)` | residual `x = x + attn` |
| `layernorm_forward(a.ln2, ..., a.residual2, ...)` | layernorm (ln2) |
| `matmul_forward(a.fch, a.ln2, w.fcw, w.fcb, C, 4C)` | matmul MLP up |
| `gelu_forward(a.fch_gelu, a.fch, 4*BTC)` | gelu |
| `matmul_forward(a.fcproj, a.fch_gelu, w.fcprojw, ..., 4C, C)` | matmul MLP down |
| `residual_forward(a.residual3, a.residual2, a.fcproj)` | residual `x = x + mlp` |

**`block_backward`** (`g` is the activation gradients, `dw` the parameter gradients, `dresidual` the gradient for the block's input):

| Call | What it does |
|---|---|
| `residual_backward(g.residual2, g.fcproj, g.residual3)` | splits the output gradient into the skip (`g.residual2`) and the MLP branch (`g.fcproj`) |
| `matmul_backward(g.fch_gelu, dw.fcprojw, dw.fcprojb, g.fcproj, ...)` | MLP down |
| `gelu_backward(g.fch, a.fch, g.fch_gelu)` | gelu, uses the saved pre-GELU `a.fch` |
| `matmul_backward(g.ln2, dw.fcw, dw.fcb, g.fch, ...)` | MLP up |
| `layernorm_backward(g.residual2, dw.ln2w, dw.ln2b, g.ln2, a.residual2, ...)` | ln2, adds into `g.residual2` |
| `residual_backward(dresidual, g.attproj, g.residual2)` | splits into the skip (`dresidual`) and the attention branch (`g.attproj`) |
| `matmul_backward(g.atty, dw.attprojw, dw.attprojb, g.attproj, ...)` | attention output projection |
| `attention_backward(g.qkv, g.atty, a.qkv, a.att, ...)` | attention, writes the QKV gradient |
| `matmul_backward(g.ln1, dw.qkvw, dw.qkvb, g.qkv, ...)` | QKV |
| `layernorm_backward(dresidual, dw.ln1w, dw.ln1b, g.ln1, residual, ...)` | ln1, adds into `dresidual` |

- `g.residual2` receives gradient twice: from the second residual (skip) and from ln2's backward. `dresidual` also receives it twice: from the first residual (skip) and from ln1's backward. Both are `+=`, so the two sums are the full gradient. `dresidual` is the previous block's `g.residual3` (or `g.encoded` for layer 0), so the next iteration of the loop continues from it.

**`gpt_forward` order:** `encoder_forward`; `block_forward` for each layer (block 0 reads `a->encoded`, later blocks read the previous `residual3`); `layernorm_forward` into `lnf`; `matmul_forward(a->logits, a->lnf, w->wte, NULL, ...)`; `crossentropy_forward` if targets exist.

**`gpt_backward` order:** `crossentropy_backward` into `g->logits`; `matmul_backward` for the logits layer (adds into `dw->wte`); `layernorm_backward` for `lnf`, writing into the last `g->residual3`; `block_backward` for L-1 down to 0; `encoder_backward` (adds into `dw->wte` and `dw->wpe`).

**Memory:** `gpt_print_summary` at `B=4, T=64` reports 6,101,760 activation floats, **46.6 MB** with gradients. At full size (`B=8, T=256`) the activations are about 444 MiB with gradients. The largest buffers are `att` `[L,B,NH,T,T]` (48 MiB), and `logits` and `probs` `[B,T,V]` (32 MiB each). The `fch` and `fch_gelu` buffers are 24 MiB each.

## `src/cpu/`

All files here are stateless, allocate nothing, and include only `ops.h`. Every backward adds (`+=`) into its gradient arrays.

### `src/cpu/encoder.c`

- **Purpose:** token plus position embedding.
- **Used by:** `gpt_forward`, `gpt_backward`.

| Function | What it does | Shapes |
|---|---|---|
| `encoder_forward(out, tokens, wte, wpe, B, T, C)` | adds the token row and the position row | out `[B,T,C]`, tokens `[B,T]`, wte `[V,C]`, wpe `[Tmax,C]` |
| `encoder_backward(dwte, dwpe, dout, tokens, B, T, C)` | sends each token's gradient to its row of `dwte` and its position's row of `dwpe` | dout `[B,T,C]` |

```
forward:  out[b,t,:] = wte[tokens[b,t],:] + wpe[t,:]
backward: dwte[tokens[b,t],:] += dout[b,t,:]
          dwpe[t,:]           += dout[b,t,:]
```

- **Saved for backward:** only `tokens` (the model's copy `model->inputs`).
- **Details:** `dwte` also receives the gradient of the tied output layer, so this function must only ever add. A token that appears twice in the batch adds into the same row twice.

### `src/cpu/layernorm.c`

- **Purpose:** normalizes each token's C values to mean 0 and variance 1, then applies a learned scale and shift. `LN_EPS` is `1e-5`.
- **Used by:** `block_forward` (ln1, ln2), `gpt_forward` (lnf) and their backwards.

| Function | What it does | Shapes |
|---|---|---|
| `layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C)` | normalize, scale, shift; save `mean` and `rstd` | out, inp `[B,T,C]`, mean, rstd `[B,T]`, weight, bias `[C]` |
| `layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C)` | gradients for input, weight and bias | |

```
forward:  mean = sum(x) / C
          var  = sum((x - mean)^2) / C
          rstd = 1 / sqrt(var + eps)
          out  = (x - mean) * rstd * weight + bias
backward: norm  = (x - mean) * rstd
          dnorm = dout * weight
          dbias   += dout
          dweight += dout * norm
          dinp    += rstd * (dnorm - mean(dnorm) - norm * mean(dnorm * norm))
```

- **Saved for backward:** `inp`, `mean`, `rstd` (`ln*_mean`, `ln*_rstd` and the layer's input buffer).
- **Details:**
  - The two mean terms exist because every channel affects the mean and variance that every other channel was normalized with. A change in one channel moves all 128 outputs.
  - The gradient per token sums to zero across channels. Shifting every input channel by the same amount changes nothing, so the gradient of such a shift must be zero.
  - Weight starts at 1 and bias at 0 (`init_params`).

### `src/cpu/matmul.c`

- **Purpose:** linear layer applied to every token.
- **Used by:** 4 calls per block, plus the output layer, in `model.c`.

| Function | What it does | Shapes |
|---|---|---|
| `matmul_forward(out, inp, weight, bias, B, T, C_in, C_out)` | one dot product per output channel, plus bias | out `[B*T,C_out]`, inp `[B*T,C_in]`, weight `[C_out,C_in]`, bias `[C_out]` or `NULL` |
| `matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C_in, C_out)` | all three gradients in one pass over tokens | `dbias` may be `NULL` |

```
forward:  out[i,o]     = dot(inp[i,:], weight[o,:]) + bias[o]
backward: dinp[i,:]    += sum over o of dout[i,o] * weight[o,:]
          dweight[o,:] += sum over i of dout[i,o] * inp[i,:]
          dbias[o]     += sum over i of dout[i,o]
```

- **Saved for backward:** the forward input `inp` (needed for `dweight`) and the weight (needed for `dinp`).
- **Details:**
  - Weights are `[C_out, C_in]`. The tied output layer then uses `wte` `[V, C]` directly, with `C_out = V`. Row `o` of the weight is contiguous, so the inner loop walks memory in order.
  - `bias` is `NULL` for the logits layer, and `dbias` is `NULL` in its backward.
  - Mixes channels within a token and never across tokens. About 90% of compute, so the main target for CUDA optimization.

### `src/cpu/attention.c`

- **Purpose:** causal multi-head self-attention. It is the only op that mixes information across tokens.
- **Used by:** `block_forward`, `block_backward`.
- **Input layout:** the QKV matmul output `[B,T,3C]`. Each token's row is `[q | k | v]`, each C wide. Each is split into `NH` heads of size `hs = C / NH`; head h uses floats `h*hs .. h*hs+hs-1` of each part.

| Function | What it does | Shapes |
|---|---|---|
| `attention_forward(out, att, inp, B, T, C, NH)` | loops over batch, token, head | out `[B,T,C]`, att `[B,NH,T,T]`, inp `[B,T,3C]` |
| `attention_backward(dinp, dout, inp, att, B, T, C, NH)` | same loops for backward | dinp `[B,T,3C]`, dout `[B,T,C]` |
| `dot(a, b, n)` (internal) | dot product of two vectors | |
| `attention_row_forward(...)` (internal) | scores, softmax and output for one token and head | |
| `attention_row_backward(...)` (internal) | backward for one token and head | |

```
forward (token t, head h):
  score[t2] = dot(q_t, k_t2) / sqrt(hs)          for t2 <= t
  att[t2]   = softmax(score)[t2]                 (0 for t2 > t)
  out_t     = sum over t2 <= t of att[t2] * v_t2
backward (datt[t2] = dout . v_t2):
  w         = sum over j of att[j] * datt[j]
  dv_t2    += att[t2] * dout
  dpre      = att[t2] * (datt[t2] - w) * scale
  dq       += dpre * k_t2
  dk_t2    += dpre * q
```

- **Saved for backward:** `att` (the whole `[B,NH,T,T]`, zero above the diagonal) and the forward `qkv`.
- **Details:**
  - Causal: only `t2 <= t` is scored, so no token sees the future.
  - The softmax subtracts the row max before `expf`, so large scores do not overflow.
  - Backward runs two passes per row. The first computes the softmax correction sum `w = sum(att[j] * datt[j])`. The second applies the formulas above. Writes go into the QKV gradient buffer `g.qkv`, which feeds the QKV matmul's backward.
  - Backward is where gradient lands on other tokens' rows: `dk` and `dv` go to earlier tokens.
  - The helpers keep each function short. Nothing is allocated.

### `src/cpu/residual.c`

- **Purpose:** `out = a + b`. `a` is the residual stream (the skip), `b` is the branch output.
- **Used by:** twice per block, in `block_forward` and `block_backward`.

| Function | What it does | Shapes |
|---|---|---|
| `residual_forward(out, a, b, N)` | elementwise add | `N = B*T*C` elements |
| `residual_backward(da, db, dout, N)` | adds `dout` into both | |

```
forward:  out[i] = a[i] + b[i]
backward: da[i] += dout[i]      (skip path)
          db[i] += dout[i]      (branch path)
```

- **Saved for backward:** nothing. The derivative of a sum is 1.
- **Details:** `out` must not alias `a` or `b`, because backward needs `a` later (the input `residual` of the block is also read by ln1's backward). That is why each residual has its own buffer.

### `src/cpu/gelu.c`

- **Purpose:** the MLP activation, tanh approximation.
- **Used by:** `block_forward`, `block_backward`.

| Function | What it does | Shapes |
|---|---|---|
| `gelu_forward(out, inp, N)` | applies gelu elementwise | `N = B*T*4C` in the MLP |
| `gelu_backward(dinp, inp, dout, N)` | `dinp += dout * gelu'(inp)` | |

```
forward:  u = sqrt(2/pi) * (x + 0.044715 * x^3)
          out = 0.5 * x * (1 + tanh(u))
backward: t = tanh(u)
          gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
          dinp += dout * gelu'(x)
```

- **Saved for backward:** the forward **input** `fch`, not the output.
- **Details:** because backward needs the input, `model.c` keeps both `fch` and `fch_gelu`. The first is the matmul output and the second the activation output.

### `src/cpu/crossentropy.c`

- **Purpose:** fused softmax and cross-entropy loss over the vocabulary.
- **Used by:** `gpt_forward`, `gpt_backward`.

| Function | What it does | Shapes |
|---|---|---|
| `crossentropy_forward(probs, losses, logits, targets, B, T, V)` | softmax per position, loss per position, returns the mean | probs, logits `[B,T,V]`, losses, targets `[B,T]` |
| `crossentropy_backward(dlogits, probs, targets, B, T, V)` | gradient of the mean loss with respect to the logits | |

```
forward:  probs  = softmax(logits)
          losses = -log(probs[target])
          return mean(losses)
backward: dlogits[i,v] += (probs[i,v] - (v == target[i])) / (B*T)
```

- **Saved for backward:** `probs` and `targets`.
- **Details:**
  - The row max is subtracted before `expf`, so large logits cannot overflow.
  - The mean is accumulated in a `double`.
  - `crossentropy_backward` is where `gpt_backward` starts.
  - `probs` is large: `B*T*V` floats, 32 MiB (about 33.5 MB) at `B=8, T=256, V=4096`.

## `tools/`

### `tools/prep_data.c`

- **Purpose:** offline data prep, run once. **Current state: only the counting half is done.** The file is one `main` plus helpers; it is not yet split into `pretokenize` and `bpe_train`.
- **Reads:** `data/TinyStories-train.txt`. It reads the first 25 MiB (`READ_BYTES`).
- **What it does now:**
  1. splits stories on `<|endoftext|>` (`SEP`) and trims whitespace; empty stories are skipped;
  2. drops stories containing a character outside 32-126 other than `'\n'`;
  3. splits text into pieces with `token_length`;
  4. counts pieces in a hash table, until `TARGET_TOKENS` (5,000,000) tokens, counting one end-of-story token per story;
  5. sorts pieces by count and gives the most common ones ids, up to `VOCAB_SIZE` 4096, with id 0 = end-of-story and id 1 = unknown;
  6. prints the number of stories, tokens, distinct pieces, coverage by the vocab and the 10 most common pieces.
- **What it does not do yet:** encode the stories as ids, or write `train.bin` and `vocab.bin`. It frees everything and exits after the printout.
- **Functions:**

| Function | What it does |
|---|---|
| `token_length(p)` | length of the piece starting at `p`: a word (letters), number (digits) or single punctuation mark, each with an optional single leading space; `'\n'` is its own piece; a lone space is a piece |
| `hash(s, len)` (internal) | FNV-1a hash of `len` bytes |
| `find_slot(table, s, len)` (internal) | slot holding this piece, or the empty slot where it belongs; linear probing on collision |
| `compare_entries(a, b)` (internal) | `qsort` order: most common first, ties broken by the text |
| `main` | everything above |

- **Details that matter:**
  - The pieces glue back into the original text, which is what lets BPE be applied piece by piece.
  - The table has `TABLE_SIZE` 65,536 slots and exits if more than half fills.
  - `Entry` points into the original buffer instead of copying the text.
  - `EOS_ID` and `UNK_ID` are defined but not used yet.
  - The text after the last `<|endoftext|>` is ignored, since stories are found by searching for the separator.
  - The distinct piece count is about 15K (the team's last count).
- **What changes:** it is moving to BPE. The piece table becomes the input to BPE training. Merges never cross piece boundaries. Ids become 0 = end-of-story, 1-96 = base characters, the rest learned merges up to 4096, with no unknown token. The remaining work is in `04-remaining.md`.

## Verification

Results from running the code during development:

| Check | Result |
|---|---|
| each op's backward vs finite differences | matched |
| full-model gradient check | all 16 parameter tensors within float rounding, about 4e-5 absolute |
| initial loss on random tokens | about 8.38, close to ln(4096) = 8.32 |
| gradient descent on one tiny batch | loss 3.0 to 0.32 in 60 steps |
| one forward+backward over 256 tokens | about 0.8 s, single-threaded, `-O2` |

`tests/test_ops.c` is still empty; the repeatable version of these checks goes there.
