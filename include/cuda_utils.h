/*
 * cuda_utils.h
 * Launch settings shared by the kernels in src/cuda/: a fixed block size and
 * the number of blocks needed to give every element one thread.
 */
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stddef.h>

#define BLOCK 256

/*
 * blocks
 * Number of BLOCK-thread blocks needed to cover n elements (one thread each).
 */
static inline unsigned int blocks(size_t n) {
    return (unsigned int)((n + BLOCK - 1) / BLOCK);
}

#endif
