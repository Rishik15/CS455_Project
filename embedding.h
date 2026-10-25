#ifndef EMBEDDING_H
#define EMBEDDING_H

typedef struct {
    int vocab_size;
    int embedding_dimensions;
    int max_seq_len;

    float *token_emb;
    float *pos_emb;
    float *d_token_emb;
    float *d_pos_emb;
} Embedding;

void init_embeddings(Embedding *emb, int vocab_size, int embedding_dimensions, int max_seq_len);
void embedding_forward(Embedding *emb, int *tokens, int batch_size, int seq_length, float *output);
void embedding_backward(Embedding *emb, int *tokens, int batch_size, int seq_length, float *d_output);
void embedding_zero_gradient(Embedding *emb);
void embedding_free(Embedding *emb);

#endif