/*
 * eval.h
 * Validation loss: the mean loss over a fixed set of batches from val.bin,
 * using forward passes only, so the weights are not changed. Every version
 * evaluates the same batches, so their final validation losses can be compared.
 */
#ifndef EVAL_H
#define EVAL_H

#include "dataloader.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Batches evaluated at the end of every run, the same in every version. */
#define VAL_BATCHES 20

/*
 * Moves val back to the start of the file and returns the mean loss over its
 * first `batches` batches. total_B is the batch size over all ranks; each rank
 * evaluates its share of every batch, like dataloader_next_MPI in training.
 * Serial passes total_B = B and rank = 0. val must be built with the batch
 * size the model was built for (B, or local_B under MPI).
 */
float eval_loss(GPT *model, DataLoader *val, int batches, int total_B, int rank);

#ifdef __cplusplus
}
#endif

#endif
