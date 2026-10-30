/*
 * encoder.c
 * Serial (CPU) token + position embedding. Stateless: the embedding tables and
 * their gradients are views into the model's single params/grads buffers,
 * which model.c allocates and initializes. Backward accumulates (+=).
 */
#include <stddef.h>
#include "ops.h"

/*
 * encoder_forward
 * out[b, t, :] = wte[tokens[b, t], :] + wpe[t, :]
 * out:    [B, T, C]
 * tokens: [B, T]
 * wte:    [V, C] token embedding table
 * wpe:    [T_max, C] position embedding table
 */
void encoder_forward(float *out, const int *tokens, const float *wte, const float *wpe,
                     int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *o = out + ((size_t)b * T + t) * C;
            const float *tok = wte + (size_t)tokens[b * T + t] * C;
            const float *pos = wpe + (size_t)t * C;
            for (int c = 0; c < C; c++) {
                o[c] = tok[c] + pos[c];
            }
        }
    }
}

/*
 * encoder_backward
 * Sends each token's gradient to the row of wte it came from and to the row
 * of wpe for its position. dwte also receives gradient from the tied output
 * layer, so this only ever adds into it.
 * dwte:   [V, C]
 * dwpe:   [T_max, C]
 * dout:   [B, T, C]
 * tokens: [B, T]
 */
void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *tokens,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *d = dout + ((size_t)b * T + t) * C;
            float *dtok = dwte + (size_t)tokens[b * T + t] * C;
            float *dpos = dwpe + (size_t)t * C;
            for (int c = 0; c < C; c++) {
                dtok[c] += d[c];
                dpos[c] += d[c];
            }
        }
    }
}
