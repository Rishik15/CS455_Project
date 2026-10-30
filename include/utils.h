/*
 * utils.h
 * Shared helpers: allocation that exits with a message on failure, a
 * seedable random number generator that gives identical results on every
 * machine (needed so serial, MPI, and CUDA runs can be compared), and a timer.
 */
#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

/* malloc that prints what failed and exits instead of returning NULL. */
void *xmalloc(size_t bytes, const char *what);

/* calloc (zeroed memory) that prints what failed and exits instead of returning NULL. */
void *xcalloc(size_t count, size_t size, const char *what);

/* Seeds the global random number generator. */
void rng_seed(uint64_t seed);

/* Uniform random float strictly between 0 and 1. */
float rng_uniform(void);

/* Gaussian random float with the given mean and standard deviation. */
float rng_normal(float mean, float std);

/* Wall-clock time in seconds, for timing training steps. */
double time_seconds(void);

#endif
