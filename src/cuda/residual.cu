/*
 * residual.cu
 * GPU (CUDA) residual connection: out = a + b, elementwise, one thread per
 * element. Same math as src/cpu/residual.c. Backward accumulates (+=).
 */
#include <stddef.h>
#include "ops.h"
#include "cuda_utils.h"

/* residual_forward_kernel: out[i] = a[i] + b[i]. */
__global__ void residual_forward_kernel(float *out, const float *a, const float *b, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

/* residual_backward_kernel: adds dout[i] into both da[i] and db[i]. */
__global__ void residual_backward_kernel(float *da, float *db, const float *dout, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        da[i] += dout[i];
        db[i] += dout[i];
    }
}

/* residual_forward: out[i] = a[i] + b[i] for all N elements; out must not alias a or b. */
void residual_forward(float *out, const float *a, const float *b, size_t N) {
    residual_forward_kernel<<<blocks(N), BLOCK>>>(out, a, b, N);
}

/* residual_backward: adds the incoming gradient unchanged into da and db. */
void residual_backward(float *da, float *db, const float *dout, size_t N) {
    residual_backward_kernel<<<blocks(N), BLOCK>>>(da, db, dout, N);
}
