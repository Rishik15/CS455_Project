#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "optimizer.h"

void adamw_init(AdamW *ad, size_t n, float weight_decay, float eps ) {
    ad->m = calloc(n, sizeof(float));
    ad->v = calloc(n, sizeof(float));
    if (ad->m == NULL || ad->v == NULL) {
        fprintf(stderr, "failed to allocate memory for adamW\n");
        exit(1);
    }
    ad->n = n;
    ad->t=0;
    ad->beta1 = 0.9f;
    ad->beta2 = 0.95f;
    ad->weight_decay = weight_decay;
    ad->eps = eps;
}

void adamw_update(AdamW *ad, float *grads, float *params, float lr) {
    float bias_correction1 = 0.0f;
    float bias_correction2 = 0.0f;
    ad->t++;
    float beta1_t = powf(ad->beta1, ad->t);
    float beta2_t = powf(ad->beta2, ad->t);
    for (size_t i=0; i<ad->n; i++) {
        //update moments
        ad->m[i] = ad->beta1*ad->m[i] + (1-ad->beta1)*grads[i];
        ad->v[i] = ad->beta2*ad->v[i] + (1-ad->beta2)*powf(grads[i], 2);
        //calculate bias corrections
        bias_correction1 = ad->m[i]/(1-beta1_t);
        bias_correction2 = ad->v[i]/(1-beta2_t);
        float initial_param = params[i];
        //compute update
        params[i] = params[i]-lr*(bias_correction1/(sqrtf(bias_correction2)+ad->eps));
        //apply weight decay and update param
        params[i] = params[i]-lr*ad->weight_decay*initial_param;
    }
}

void adamw_free(AdamW *ad) {
    free(ad->m);
    free(ad->v);
}