/*
 * pretokenize.h
 * Splits text into pieces (a word, number or punctuation mark with its
 * leading space, or '\n') and counts each distinct piece in a hash table.
 * BPE merges never cross a piece boundary, so BPE only has to work on the
 * ~15K distinct pieces instead of millions of characters of text.
 */
#ifndef PRETOKENIZE_H
#define PRETOKENIZE_H

/*
 * Piece
 * start/len: the piece's text, pointing into the original buffer (not copied).
 * count: how many times it appears in the counted text.
 * ids/n_ids: the piece's tokens, filled in and kept up to date by bpe_train.
 */
typedef struct {
    const char *start;
    int len;
    int count;
    int *ids;
    int n_ids;
} Piece;

/*
 * PieceTable
 * Open-addressing hash table of pieces. capacity is a power of 2 and the
 * table grows so it is never more than half full. Empty slots have start NULL.
 */
typedef struct {
    Piece *slots;
    int capacity;
    int count;
} PieceTable;

/* Length of the piece starting at p. The text must be '\0'-terminated. */
int piece_length(const char *p);

/* Creates an empty table; capacity must be a power of 2. */
void piece_table_init(PieceTable *t, int capacity);

/* The slot holding this piece, or the empty slot (start NULL) where it would go. */
Piece *piece_table_find(const PieceTable *t, const char *s, int len);

/* Splits len bytes of text into pieces and adds one to each piece's count. */
void piece_table_count(PieceTable *t, const char *text, int len);

/* Frees the table and every piece's ids. */
void piece_table_free(PieceTable *t);

#endif
