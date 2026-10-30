/*
 * residual.c
 * Serial (CPU) residual connection: out = a + b, elementwise.
 * a is the residual stream that skips the branch, b is the branch output.
 * Backward accumulates (+=).
 */
#include <stddef.h>
#include "ops.h"

/*
 * residual_forward
 * out[i] = a[i] + b[i] for all N elements (N = B * T * C).
 * out may not alias a or b, since the backward pass needs a later.
 */
void residual_forward(float *out, const float *a, const float *b, size_t N) {
    for (size_t i = 0; i < N; i++) {
        out[i] = a[i] + b[i];
    }
}

/*
 * residual_backward
 * The derivative of a + b is 1 for both inputs, so the incoming gradient is
 * added unchanged into both: da carries it up the skip connection, db carries
 * it back through the branch.
 */
void residual_backward(float *da, float *db, const float *dout, size_t N) {
    for (size_t i = 0; i < N; i++) {
        da[i] += dout[i];
        db[i] += dout[i];
    }
}
