/*
 * encoder.cu
 * GPU (CUDA) token + position embedding, one thread per output element.
 * Same math as src/cpu/encoder.c. Backward accumulates (+=) with atomicAdd,
 * because many tokens share a wte row and every batch row shares a wpe row.
 */
#include <stddef.h>
#include "ops.h"
#include "cuda_utils.h"

/* encoder_forward_kernel: thread (b, t, c) writes wte[tokens[b, t], c] + wpe[t, c]. */
__global__ void encoder_forward_kernel(float *out, const int *tokens, const float *wte,
                                       const float *wpe, int B, int T, int C) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * T * C) return;
    size_t bt = idx / C;
    int c = idx % C;
    int t = bt % T;
    out[idx] = wte[(size_t)tokens[bt] * C + c] + wpe[(size_t)t * C + c];
}

/* encoder_backward_kernel: thread (b, t, c) adds dout into the dwte and dwpe it came from. */
__global__ void encoder_backward_kernel(float *dwte, float *dwpe, const float *dout,
                                        const int *tokens, int B, int T, int C) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * T * C) return;
    size_t bt = idx / C;
    int c = idx % C;
    int t = bt % T;
    atomicAdd(&dwte[(size_t)tokens[bt] * C + c], dout[idx]);
    atomicAdd(&dwpe[(size_t)t * C + c], dout[idx]);
}

/*
 * encoder_forward
 * out: [B, T, C], tokens: [B, T], wte: [V, C], wpe: [T_max, C]
 */
void encoder_forward(float *out, const int *tokens, const float *wte, const float *wpe,
                     int B, int T, int C) {
    encoder_forward_kernel<<<blocks((size_t)B * T * C), BLOCK>>>(out, tokens, wte, wpe, B, T, C);
}

/*
 * encoder_backward
 * dwte: [V, C], dwpe: [T_max, C], dout: [B, T, C], tokens: [B, T]
 */
void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *tokens,
                      int B, int T, int C) {
    encoder_backward_kernel<<<blocks((size_t)B * T * C), BLOCK>>>(dwte, dwpe, dout, tokens, B, T, C);
}
