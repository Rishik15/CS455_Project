#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <math.h>

// Small deterministic random number generator (xorshift64*).
// We use our own instead of rand() so every machine and every version
// gets exactly the same starting weights from the same seed.

#define RNG_PI 3.14159265358979323846f

typedef struct {
    uint64_t state;
} Rng;

static inline void rng_seed(Rng *r, uint64_t seed) {
    r->state = seed ? seed : 0x9E3779B97F4A7C15ULL; // state must never be 0
}

static inline uint32_t rng_u32(Rng *r) {
    r->state ^= r->state >> 12;
    r->state ^= r->state << 25;
    r->state ^= r->state >> 27;
    return (uint32_t)((r->state * 0x2545F4914F6CDD1DULL) >> 32);
}

// uniform float in [0, 1)
static inline float rng_uniform(Rng *r) {
    return (rng_u32(r) >> 8) / 16777216.0f;
}

// normal(mean, std) using the Box-Muller transform
static inline float rng_normal(Rng *r, float mean, float std) {
    float u1 = rng_uniform(r);
    float u2 = rng_uniform(r);
    if (u1 < 1e-7f) u1 = 1e-7f; // avoid log(0)
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * RNG_PI * u2);
    return mean + std * z;
}

#endif
