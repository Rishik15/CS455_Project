#ifndef VOCAB_H
#define VOCAB_H

// The character vocabulary, as written by prep_data.c.
//
// data/vocab.bin format:
//   int32 size, then size bytes: the character for each token id.
//   Id 0 is the end-of-story token (stored as '\0'); the characters follow in
//   ASCII order.
//
// data/train.bin and data/val.bin format:
//   one byte per token id. Every story is preceded by the end-of-story token,
//   and the file ends with one.

#define EOS_ID 0

typedef struct {
    int size;               // number of token ids, including end-of-story
    char id_to_char[256];   // character for each id
    int char_to_id[256];    // id for each character, -1 if not in the vocab
} Vocab;

// Reads path into v. Returns 0 on success, -1 on error (and prints why).
int vocab_load(Vocab *v, const char *path);

#endif
