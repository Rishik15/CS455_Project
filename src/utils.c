/*
 * utils.c
 * Implementation of utils.h: checked allocation, a portable xorshift64*
 * random number generator with Box-Muller Gaussian sampling, and a
 * monotonic wall-clock timer.
 */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "utils.h"

#define TWO_PI 6.283185307179586

static uint64_t rng_state = 0x853c49e6748fea9bULL;

/*
 * xmalloc
 * Allocates bytes or exits with an error naming what was being allocated.
 */
void *xmalloc(size_t bytes, const char *what) {
    void *p = malloc(bytes);
    if (p == NULL) {
        fprintf(stderr, "out of memory allocating %s (%zu bytes)\n", what, bytes);
        exit(1);
    }
    return p;
}

/*
 * xcalloc
 * Allocates count * size zeroed bytes or exits with an error naming what was
 * being allocated.
 */
void *xcalloc(size_t count, size_t size, const char *what) {
    void *p = calloc(count, size);
    if (p == NULL) {
        fprintf(stderr, "out of memory allocating %s (%zu bytes)\n", what, count * size);
        exit(1);
    }
    return p;
}

/*
 * rng_seed
 * Sets the generator state. A zero seed is replaced because xorshift
 * never leaves the all-zero state.
 */
void rng_seed(uint64_t seed) {
    rng_state = seed != 0 ? seed : 0x853c49e6748fea9bULL;
}

/*
 * rng_next
 * Advances xorshift64* and returns 64 random bits.
 */
static uint64_t rng_next(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

/*
 * rng_uniform
 * Uses the top 24 bits so every value is exactly representable as a float,
 * offset by half a step so the result is never 0 or 1 (log(0) in Box-Muller).
 */
float rng_uniform(void) {
    return ((float)(rng_next() >> 40) + 0.5f) / 16777216.0f;
}

/*
 * rng_normal
 * Box-Muller transform of two uniform samples.
 */
float rng_normal(float mean, float std) {
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    double z = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
    return mean + std * (float)z;
}

/*
 * time_seconds
 * Monotonic clock, unaffected by system clock changes.
 */
double time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
