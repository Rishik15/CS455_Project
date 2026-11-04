/*
* optimizer.h
 * AdamW optimizer: updates every parameter from its gradient using running
 * averages of the gradient (m) and squared gradient (v).
 */
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stddef.h>

typedef struct {
    float *m; // running avg of gradients
    float *v; // running squared avg of gradients
    size_t n; //number of parameters
    int t; // step counter
    float beta1;
    float beta2;
    float weight_decay;
    float eps; //epsilon
} AdamW;

/* Initiates vectors to store running gradients and squared running gradients*/
void adamw_init(AdamW *ad, size_t n, float weight_decay, float eps );

/* Update parameter weights*/
void adamw_update(AdamW *ad, float *grads, float *params, float lr);

/*Free the running gradient vectors*/
void adamw_free(AdamW *ad);

#endif