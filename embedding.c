#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "embedding.h"

#define PI 3.1415926535

static double random_uniform() {
        return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
}

static float random_gausse(double mean, double stddev) {
        //uses mean=0 for initalization
        //uses stddev=0.02 for initalization
        float u1 = random_uniform();
        float u2 = random_uniform();
        return stddev*(sqrt(-2.0*log(u1))*cos(2.0*PI*u2))+mean;
}

void init_embeddings(Embedding *emb, int vocab_size, int embedding_dimensions, int max_seq_len) {
        emb->vocab_size = vocab_size;
        emb->embedding_dimensions = embedding_dimensions;
        emb->max_seq_len = max_seq_len;

        size_t embedding_size = (size_t) vocab_size * embedding_dimensions;
        size_t position_size = (size_t) embedding_dimensions * max_seq_len;

        emb->token_emb = malloc(embedding_size * sizeof(float));
        emb->pos_emb = malloc(position_size * sizeof(float));
        emb->d_token_emb = calloc(embedding_size, sizeof(float));
        emb->d_pos_emb = calloc(position_size, sizeof(float));

        if (emb->pos_emb == NULL || emb->token_emb == NULL || emb->d_token_emb == NULL || emb->d_pos_emb == NULL) {
                perror("Failed to allocated space for token_emb or pos_emb");
                free(emb->token_emb);
                free(emb->pos_emb);
                free(emb->d_token_emb);
                free(emb->d_pos_emb);
                exit(1);
        }

        //initialize vectors
        float *t_emb = emb->token_emb;
        float *t_pos_emb = emb->pos_emb;

        for (size_t i = 0; i < embedding_size; i++) {
                t_emb[i] = random_gausse(0, 0.02);
        }

        for (size_t i = 0; i < position_size; i++) {
                t_pos_emb[i] = random_gausse(0, 0.02);
        }
}

void embedding_forward(Embedding *emb, int *tokens, int batch_size, int seq_length, float *output) {
        int embedding_dimensions = emb->embedding_dimensions;
        int n = batch_size*seq_length;
        for (int i =0; i<n; i++) {
                int t = tokens[i];
                for (int j =0; j<embedding_dimensions; j++) {
                        output[i*embedding_dimensions+j] = emb->pos_emb[i%seq_length*embedding_dimensions+j]+emb->token_emb[t*embedding_dimensions+j];
                }
        }
}

void embedding_backward(Embedding *emb, int *tokens, int batch_size, int seq_length, float *d_output) {
        //This doesnt include any heavy computatoin, the point is to map gradients to tokens
        //so they can be updated during training using the optimizer.
        int embedding_dimensions = emb->embedding_dimensions;
        int n = batch_size*seq_length;
        for (int i =0; i<n; i++) {
                int t = tokens[i];
                for (int j =0; j<embedding_dimensions; j++) {
                        emb->d_pos_emb[(i%seq_length)*embedding_dimensions+j] += d_output[i*embedding_dimensions+j];
                        emb->d_token_emb[t*embedding_dimensions+j] += d_output[i*embedding_dimensions+j];
                }

        }
}

void embedding_zero_gradient(Embedding *emb) {
        memset(emb->d_token_emb, 0, sizeof(float)*emb->embedding_dimensions*emb->vocab_size);
        memset(emb->d_pos_emb, 0, sizeof(float)*emb->embedding_dimensions*emb->max_seq_len);
}

void embedding_free(Embedding *emb) {
        free(emb->d_token_emb);
        free(emb->d_pos_emb);
        free(emb->token_emb);
        free(emb->pos_emb);
}
