/*
 * dataloader.h
 * Serves batches of inputs and targets (inputs shifted by one) from a
 * .bin file of uint16 token ids.
 */
#ifndef Dataloader_H
#define Dataloader_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint16_t *tokens;     // the whole .bin file in memory
    size_t num_tokens;
    size_t start, end;    // the part of the file this loader reads (serial: 0 to num_tokens)
    size_t pos;           // where the next batch starts
    int B, T;
    int *inputs;          // B*T ints, passed to gpt_forward
    int *targets;         // B*T ints
} DataLoader;

/* Loads the whole file into memory and allocates one B x T batch. */
void dataloader_init(DataLoader *dl, const char *path, int B, int T);

/* Fills inputs/targets with the next batch, wrapping to start when out of room. */
void dataloader_next(DataLoader *dl);

/* Moves back to the start, so validation sees the same batches each time. */
void dataloader_reset(DataLoader *dl);

/* Frees the token, input and target buffers. */
void dataloader_free(DataLoader *dl);

#endif