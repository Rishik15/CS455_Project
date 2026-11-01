/*
 * bpe_train.c
 * Implementation of bpe_train.h. Each training step recounts every adjacent
 * pair across the distinct pieces (weighted by how often each piece appears),
 * picks the most frequent pair and merges it in every piece. Simple rather
 * than fast: about 4000 steps over ~100K tokens takes a few seconds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe_train.h"
#include "utils.h"

#define EMPTY_KEY 0xFFFFFFFFu

/*
 * pair_key
 * Packs two ids (each below 65536) into one key; never equals EMPTY_KEY
 * because ids stay below 4096.
 */
static uint32_t pair_key(int a, int b) {
    return ((uint32_t)a << 16) | (uint32_t)b;
}

/*
 * mix
 * Scrambles all key bits into the low bits (murmur3 finalizer), so pairs
 * sharing a right id do not all land in the same slots.
 */
static uint32_t mix(uint32_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6bu;
    k ^= k >> 13;
    k *= 0xc2b2ae35u;
    k ^= k >> 16;
    return k;
}

/*
 * pair_table_clear
 * Marks every slot empty.
 */
static void pair_table_clear(PairTable *t) {
    memset(t->keys, 0xFF, sizeof(uint32_t) * t->capacity);
}

/*
 * pair_table_init
 * Picks a power-of-2 capacity at least twice max_entries, so the table is
 * never more than half full.
 */
static void pair_table_init(PairTable *t, int max_entries) {
    int capacity = 1024;
    while (capacity < 2 * max_entries) capacity *= 2;
    t->keys = xmalloc(sizeof(uint32_t) * capacity, "pair table keys");
    t->values = xmalloc(sizeof(int) * capacity, "pair table values");
    t->capacity = capacity;
    pair_table_clear(t);
}

/*
 * pair_table_free
 * Frees both arrays.
 */
static void pair_table_free(PairTable *t) {
    free(t->keys);
    free(t->values);
    t->keys = NULL;
    t->values = NULL;
    t->capacity = 0;
}

/*
 * pair_index
 * Slot holding (a, b), or the empty slot where it would go (linear probing).
 */
static uint32_t pair_index(const PairTable *t, int a, int b) {
    uint32_t key = pair_key(a, b);
    uint32_t i = mix(key) & (t->capacity - 1);
    while (t->keys[i] != EMPTY_KEY && t->keys[i] != key) {
        i = (i + 1) & (t->capacity - 1);
    }
    return i;
}

/*
 * pair_value
 * Pointer to the value for (a, b), adding the pair with value 0 if missing.
 */
static int *pair_value(PairTable *t, int a, int b) {
    uint32_t i = pair_index(t, a, b);
    if (t->keys[i] == EMPTY_KEY) {
        t->keys[i] = pair_key(a, b);
        t->values[i] = 0;
    }
    return &t->values[i];
}

/*
 * pair_get
 * Value for (a, b), or -1 if the pair is not in the table.
 */
static int pair_get(const PairTable *t, int a, int b) {
    uint32_t i = pair_index(t, a, b);
    return t->keys[i] == EMPTY_KEY ? -1 : t->values[i];
}

/*
 * bpe_char_id
 * '\n' first, then the printable characters in ASCII order.
 */
int bpe_char_id(char c) {
    if (c == '\n') return 1;
    if (c >= 32 && c <= 126) return c - 30;
    return -1;
}

/*
 * set_token
 * Copies the text (with a '\0' after it, for printing) and records the pair.
 */
static void set_token(BpeToken *tok, const char *text, int len, int left, int right) {
    tok->text = xmalloc(len + 1, "token text");
    memcpy(tok->text, text, len);
    tok->text[len] = '\0';
    tok->len = len;
    tok->left = left;
    tok->right = right;
}

/*
 * init_base_tokens
 * Id 0 is end-of-story, then one token per allowed character.
 */
static void init_base_tokens(Bpe *bpe) {
    set_token(&bpe->tokens[BPE_EOS_ID], "<eos>", 5, -1, -1);
    set_token(&bpe->tokens[bpe_char_id('\n')], "\n", 1, -1, -1);
    for (int c = 32; c <= 126; c++) {
        char ch = (char)c;
        set_token(&bpe->tokens[bpe_char_id(ch)], &ch, 1, -1, -1);
    }
}

/*
 * init_piece_ids
 * Gives every piece one id per character and returns the total number of
 * ids, an upper bound on the number of distinct pairs. Exits on a character
 * outside the vocab, since prep_data should have filtered those stories out.
 */
static int init_piece_ids(PieceTable *pieces) {
    int total = 0;
    for (int i = 0; i < pieces->capacity; i++) {
        Piece *p = &pieces->slots[i];
        if (p->start == NULL) continue;
        p->ids = xmalloc(sizeof(int) * p->len, "piece ids");
        p->n_ids = p->len;
        for (int k = 0; k < p->len; k++) {
            p->ids[k] = bpe_char_id(p->start[k]);
            if (p->ids[k] < 0) {
                fprintf(stderr, "character %d has no token id\n", (unsigned char)p->start[k]);
                exit(1);
            }
        }
        total += p->len;
    }
    return total;
}

/*
 * count_pairs
 * counts[(a, b)] = number of times b follows a anywhere in the text. A pair
 * inside a piece that appears 500 times counts 500 times.
 */
static void count_pairs(PairTable *counts, const PieceTable *pieces) {
    pair_table_clear(counts);
    for (int i = 0; i < pieces->capacity; i++) {
        const Piece *p = &pieces->slots[i];
        for (int k = 0; k + 1 < p->n_ids; k++) {
            *pair_value(counts, p->ids[k], p->ids[k + 1]) += p->count;
        }
    }
}

/*
 * best_pair
 * Key of the most frequent pair, ties broken by the smaller key so every
 * machine learns the same merges. EMPTY_KEY if there are no pairs.
 */
static uint32_t best_pair(const PairTable *counts) {
    uint32_t best = EMPTY_KEY;
    int best_count = 0;
    for (int i = 0; i < counts->capacity; i++) {
        uint32_t key = counts->keys[i];
        if (key == EMPTY_KEY) continue;
        if (counts->values[i] > best_count ||
            (counts->values[i] == best_count && key < best)) {
            best = key;
            best_count = counts->values[i];
        }
    }
    return best;
}

/*
 * merge_ids
 * Replaces each (a, b) in ids with new_id, scanning left to right, and
 * returns the new length. "a a a" with pair (a, a) becomes "new a".
 */
static int merge_ids(int *ids, int n, int a, int b, int new_id) {
    int w = 0;
    for (int r = 0; r < n; r++) {
        if (r + 1 < n && ids[r] == a && ids[r + 1] == b) {
            ids[w++] = new_id;
            r++;
        } else {
            ids[w++] = ids[r];
        }
    }
    return w;
}

/*
 * add_merge
 * Creates token id from (a, b): its text is a's text followed by b's.
 */
static void add_merge(Bpe *bpe, int id, int a, int b) {
    const BpeToken *x = &bpe->tokens[a];
    const BpeToken *y = &bpe->tokens[b];
    char *text = xmalloc(x->len + y->len, "merged text");
    memcpy(text, x->text, x->len);
    memcpy(text + x->len, y->text, y->len);
    set_token(&bpe->tokens[id], text, x->len + y->len, a, b);
    free(text);
}

/*
 * bpe_train
 * One merge per step: count pairs, take the most frequent, merge it in every
 * piece. Then builds the (left, right) -> id table that bpe_encode uses.
 */
void bpe_train(Bpe *bpe, PieceTable *pieces, int vocab_size) {
    bpe->tokens = xcalloc(vocab_size, sizeof(BpeToken), "bpe tokens");
    init_base_tokens(bpe);
    PairTable counts;
    pair_table_init(&counts, init_piece_ids(pieces));
    int id = BPE_NUM_BASE;
    for (; id < vocab_size; id++) {
        count_pairs(&counts, pieces);
        uint32_t key = best_pair(&counts);
        if (key == EMPTY_KEY) break;
        int a = (int)(key >> 16);
        int b = (int)(key & 0xFFFF);
        add_merge(bpe, id, a, b);
        for (int i = 0; i < pieces->capacity; i++) {
            Piece *p = &pieces->slots[i];
            p->n_ids = merge_ids(p->ids, p->n_ids, a, b, id);
        }
    }
    bpe->vocab_size = id;
    pair_table_free(&counts);
    pair_table_init(&bpe->merged, bpe->vocab_size);
    for (int m = BPE_NUM_BASE; m < bpe->vocab_size; m++) {
        *pair_value(&bpe->merged, bpe->tokens[m].left, bpe->tokens[m].right) = m;
    }
}

/*
 * bpe_encode
 * Starts from one id per character and keeps merging the pair that was
 * learned earliest (lowest id) until no learned pair is left. This gives
 * the same ids as replaying the merges in training order.
 */
int bpe_encode(const Bpe *bpe, const char *s, int len, int *out) {
    for (int k = 0; k < len; k++) {
        out[k] = bpe_char_id(s[k]);
    }
    int n = len;
    while (n > 1) {
        int best = -1;
        for (int k = 0; k + 1 < n; k++) {
            int id = pair_get(&bpe->merged, out[k], out[k + 1]);
            if (id >= 0 && (best < 0 || id < best)) best = id;
        }
        if (best < 0) break;
        n = merge_ids(out, n, bpe->tokens[best].left, bpe->tokens[best].right, best);
    }
    return n;
}

/*
 * bpe_free
 * Token texts first, then the arrays.
 */
void bpe_free(Bpe *bpe) {
    for (int id = 0; id < bpe->vocab_size; id++) {
        free(bpe->tokens[id].text);
    }
    free(bpe->tokens);
    pair_table_free(&bpe->merged);
    bpe->tokens = NULL;
    bpe->vocab_size = 0;
}
