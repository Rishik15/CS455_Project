/*
 * eval.c
 * Validation loss over a fixed set of batches. Forward passes only: no
 * backward pass and no optimizer step, so training is not affected.
 */
#include "eval.h"

float eval_loss(GPT *model, DataLoader *val, int batches, int total_B, int rank) {
    double sum = 0.0;
    dataloader_reset(val);
    for (int i = 0; i < batches; i++) {
        dataloader_next_MPI(val, total_B, rank);
        sum += gpt_forward(model, val->inputs, val->targets, val->B, val->T);
    }
    return (float)(sum / batches);
}
