#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "dataloader.h"

void dataloader_init(DataLoader *dl, const char* path, int B, int T) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror("File open failed\n");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    dl->num_tokens = filesize/2;
    if (dl->num_tokens < B*T+1) {
        fprintf(stderr, "Not enough tokens read for \nbatch size: %d\ntoken per batch: %d\n", B, T);
        exit(1);
    }
    dl->start = 0;
    dl->pos = 0;
    dl->end = dl->num_tokens;
    dl->B = B;
    dl->T = T;

    dl->tokens = malloc(sizeof(uint16_t) * dl->num_tokens);
    if (dl->tokens == NULL) {
        perror("Memory allocation failed\n");
        exit(1);
    }
    size_t tokens_read = fread(dl->tokens, sizeof(uint16_t), dl->num_tokens, f);
    if (tokens_read != dl->num_tokens) {
        perror("Failed to read all token\n");
        exit(1);
    }

    dl->inputs = malloc(sizeof(int) * B*T);
    dl->targets = malloc(sizeof(int) * B*T);
    if (dl->inputs == NULL || dl->targets == NULL) {
        perror("Memory allocation failed\n");
        exit(1);
    }
    fclose(f);
}

void dataloader_next(DataLoader *dl) {
    int B = dl->B;
    int T = dl->T;

    // if the current position is past the endpoint of the file, go back to the start
    if (dl->pos+B*T+1 > dl->end) {
        dl->pos = dl->start;
    }

    for (int i = 0; i < B*T; i++) {
        dl->inputs[i] = dl->tokens[dl->pos+i];
        dl->targets[i] = dl->tokens[dl->pos+i+1];
    }
    // Progress pos forward for next read
    dl->pos += B*T;
}

void dataloader_reset(DataLoader *dl) {
    dl->pos = dl->start;
}

void dataloader_free(DataLoader *dl) {
    free(dl->tokens);
    free(dl->targets);
    free(dl->inputs);
}

