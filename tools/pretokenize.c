/*
 * pretokenize.c
 * Implementation of pretokenize.h: the piece splitting rule and a hash table
 * keyed by piece text (FNV-1a hash, linear probing).
 */
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "pretokenize.h"
#include "utils.h"

/*
 * piece_length
 * A word (letters), number (digits) or single punctuation mark, each with an
 * optional single leading space, so " said" and " \"" are pieces. '\n' is a
 * piece on its own, and so is a space with nothing after it. Gluing the
 * pieces back together gives the original text.
 */
int piece_length(const char *p) {
    int i = 0;
    if (p[0] == '\n') return 1;
    if (p[0] == ' ') i = 1;
    if (isalpha((unsigned char)p[i])) {
        while (isalpha((unsigned char)p[i])) i++;
    } else if (isdigit((unsigned char)p[i])) {
        while (isdigit((unsigned char)p[i])) i++;
    } else if (p[i] != ' ' && p[i] != '\n' && p[i] != '\0') {
        i++;
    }
    return i > 0 ? i : 1;
}

/*
 * hash_bytes
 * FNV-1a hash of len bytes.
 */
static uint32_t hash_bytes(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

/*
 * piece_table_init
 * Zeroed slots: start NULL, count 0, ids NULL.
 */
void piece_table_init(PieceTable *t, int capacity) {
    t->slots = xcalloc(capacity, sizeof(Piece), "piece table");
    t->capacity = capacity;
    t->count = 0;
}

/*
 * piece_table_find
 * Starts at the piece's hash and steps to the next slot while the slot holds
 * a different piece.
 */
Piece *piece_table_find(const PieceTable *t, const char *s, int len) {
    uint32_t i = hash_bytes(s, len) & (t->capacity - 1);
    while (t->slots[i].start != NULL &&
           !(t->slots[i].len == len && memcmp(t->slots[i].start, s, len) == 0)) {
        i = (i + 1) & (t->capacity - 1);
    }
    return &t->slots[i];
}

/*
 * grow
 * Moves every piece into a table twice the size.
 */
static void grow(PieceTable *t) {
    PieceTable bigger;
    piece_table_init(&bigger, t->capacity * 2);
    for (int i = 0; i < t->capacity; i++) {
        if (t->slots[i].start != NULL) {
            *piece_table_find(&bigger, t->slots[i].start, t->slots[i].len) = t->slots[i];
        }
    }
    bigger.count = t->count;
    free(t->slots);
    *t = bigger;
}

/*
 * piece_table_count
 * New pieces are added with count 0 before the increment. The table grows
 * before it would pass half full, so probing stays short.
 */
void piece_table_count(PieceTable *t, const char *text, int len) {
    const char *p = text;
    const char *end = text + len;
    while (p < end) {
        int n = piece_length(p);
        Piece *e = piece_table_find(t, p, n);
        if (e->start == NULL) {
            if (2 * (t->count + 1) > t->capacity) {
                grow(t);
                e = piece_table_find(t, p, n);
            }
            e->start = p;
            e->len = n;
            t->count++;
        }
        e->count++;
        p += n;
    }
}

/*
 * piece_table_free
 * Frees each piece's ids (NULL if bpe_train never ran), then the slots.
 */
void piece_table_free(PieceTable *t) {
    for (int i = 0; i < t->capacity; i++) {
        free(t->slots[i].ids);
    }
    free(t->slots);
    t->slots = NULL;
    t->capacity = 0;
    t->count = 0;
}
