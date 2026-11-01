/*
 * bpe_train.h
 * Byte pair encoding. Starts with one token per character and repeatedly
 * merges the most frequent adjacent pair of tokens into a new token, counting
 * pairs only inside pieces (see pretokenize.h). Also encodes new text with
 * the learned merges.
 *
 * Ids: 0 = end-of-story, 1 = '\n', 2-96 = ' ' to '~' (ASCII 32-126),
 * 97 and up = merges in the order they were learned.
 */
#ifndef BPE_TRAIN_H
#define BPE_TRAIN_H

#include <stdint.h>
#include "pretokenize.h"

#define BPE_EOS_ID 0
#define BPE_NUM_BASE 97

/*
 * BpeToken
 * text/len: the characters the token stands for ("<eos>" for id 0).
 * left/right: the two tokens merged to make it, -1 for the 97 base tokens.
 */
typedef struct {
    char *text;
    int len;
    int left;
    int right;
} BpeToken;

/*
 * PairTable
 * Hash table from a pair of token ids to an int: pair counts while
 * training, the merged token's id when encoding.
 */
typedef struct {
    uint32_t *keys;
    int *values;
    int capacity;
} PairTable;

/*
 * Bpe
 * tokens[vocab_size], indexed by id. merged maps (left, right) to the id
 * of the token they merge into.
 */
typedef struct {
    int vocab_size;
    BpeToken *tokens;
    PairTable merged;
} Bpe;

/* Id of a single character: '\n' is 1, ' ' to '~' are 2-96, anything else -1. */
int bpe_char_id(char c);

/*
 * Learns merges until there are vocab_size tokens (fewer if no pairs are
 * left). Afterwards each piece's ids hold its final encoding.
 */
void bpe_train(Bpe *bpe, PieceTable *pieces, int vocab_size);

/* Encodes one piece into out (room for len ids) and returns the number of ids. */
int bpe_encode(const Bpe *bpe, const char *s, int len, int *out);

/* Frees the token texts, the token array and the merge table. */
void bpe_free(Bpe *bpe);

#endif
