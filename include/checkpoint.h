#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "model.h"

/* Versioned weights-only checkpoint; no optimizer or training progress.
 * Return 0 on success, -1 on I/O/format error (diagnostic on stderr).
 * Allocation failure follows gpt_build's fatal-error behavior. */
int checkpoint_save(const char *path, const GPT *model);

/* model must be zero-initialized and unallocated. On failure it stays empty.
 * Runtime batch/context capacity may differ from training. */
int checkpoint_load(const char *path, GPT *model, int max_batch, int max_seq);

#endif
