/*
 * prep_data.c
 * Offline data prep, run once from the repo root. Reads TinyStories, keeps
 * the stories made only of printable ASCII (32-126) and '\n', learns a
 * VOCAB_SIZE-token BPE vocab from the first TOKENIZER_CHARS characters of
 * training stories, and writes:
 *   data/train.bin    whole stories until there are TARGET_TOKENS + 1 ids
 *   data/val.bin      every clean story in TinyStories-valid.txt
 *   data/vocab.bin    int32 vocab_size, then for each id in order: 1 byte length,
 *                     then the token's characters (id 0 is stored as "<eos>").
 *                     Printing each id's characters back to back gives the text.
 *   data/merges.bin   int32 n_merges, then per merge uint16 left, uint16 right.
 *                     Merge i makes id 97 + i. Needed to encode prompts.
 * The .bin token files are uint16 ids: EOS story EOS story ... story EOS.
 * Build: gcc -std=c99 -O2 -Iinclude tools/prep_data.c tools/pretokenize.c
 *        tools/bpe_train.c src/utils.c -o prep_data
 */
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe_train.h"
#include "pretokenize.h"
#include "utils.h"


#define TRAIN_TEXT "data/TinyStories-train.txt"
#define VAL_TEXT "data/TinyStories-valid.txt"
#define TRAIN_FILE "data/train.bin"
#define VAL_FILE "data/val.bin"
#define VOCAB_FILE "data/vocab.bin"
#define MERGES_FILE "data/merges.bin"

/*	These macros are redfined for testing
	#define TRAIN_READ_BYTES (25L * 1024 * 1024)
	#define TOKENIZER_CHARS 20000000L
	#define TARGET_TOKENS 5000000L 
*/

#define TRAIN_READ_BYTES (2L * 1024 * 1024)
#define TOKENIZER_CHARS 500000L
#define TARGET_TOKENS 10000L

#define VOCAB_SIZE 4096
#define CHUNK_TOKENS (1 << 20)
#define SEP "<|endoftext|>"

typedef struct {
    int length;
    char *start;
} Story;

/*
 * StorySet
 * The text buffer and the kept stories, which point into it ('\0'-terminated).
 * longest is the longest story's length.
 */
typedef struct {
    char *buffer;
    Story *stories;
    int count;
    int longest;
} StorySet;

/*
 * read_text
 * Reads up to max_bytes of the file (all of it if max_bytes < 0) into a
 * '\0'-terminated buffer.
 */
static char *read_text(const char *path, long max_bytes) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        exit(1);
    }
    if (max_bytes < 0) {
        fseek(file, 0, SEEK_END);
        max_bytes = ftell(file);
        fseek(file, 0, SEEK_SET);
    }
    char *buffer = xmalloc(max_bytes + 1, "text buffer");
    size_t n = fread(buffer, 1, max_bytes, file);
    fclose(file);
    buffer[n] = '\0';
    printf("read %zu bytes from %s\n", n, path);
    return buffer;
}

/*
 * split_stories
 * Cuts the buffer at each SEP, trims whitespace around each story and skips
 * empty ones. Text after the last SEP is ignored, since it may be cut off.
 * Returns a growing array of stories and sets *count.
 */
static Story *split_stories(char *buffer, int *count) {
    int capacity = 1024;
    Story *stories = xmalloc(sizeof(Story) * capacity, "stories");
    char *curr = buffer;
    char *story_end;
    *count = 0;
    while ((story_end = strstr(curr, SEP))) {
        *story_end = '\0';
        char *p = curr;
        char *end = story_end - 1;
        while (*p && isspace((unsigned char)*p)) p++;
        while (end >= p && isspace((unsigned char)*end)) {
            *end = '\0';
            end--;
        }
        int length = (int)(end - p + 1);
        if (length > 0) {
            if (*count == capacity) {
                Story *bigger = xmalloc(sizeof(Story) * capacity * 2, "stories");
                memcpy(bigger, stories, sizeof(Story) * capacity);
                free(stories);
                stories = bigger;
                capacity *= 2;
            }
            stories[*count].length = length;
            stories[*count].start = p;
            (*count)++;
        }
        curr = story_end + strlen(SEP);
    }
    return stories;
}

/*
 * is_clean
 * True if every character has a token id (printable ASCII or '\n').
 */
static int is_clean(const Story *s) {
    for (int i = 0; i < s->length; i++) {
        if (bpe_char_id(s->start[i]) < 0) return 0;
    }
    return 1;
}

/*
 * load_stories
 * Reads (all of the file if max_bytes < 0), splits and filters, keeping the
 * clean stories in their original order.
 */
static StorySet load_stories(const char *path, long max_bytes) {
    StorySet set = {0};
    int found;
    set.buffer = read_text(path, max_bytes);
    set.stories = split_stories(set.buffer, &found);
    for (int i = 0; i < found; i++) {
        if (!is_clean(&set.stories[i])) continue;
        set.stories[set.count++] = set.stories[i];
        if (set.stories[i].length > set.longest) set.longest = set.stories[i].length;
    }
    printf("%d stories, %d kept (printable ASCII and newlines only)\n", found, set.count);
    return set;
}

/*
 * encode_story
 * Appends the story's ids to out and returns how many were added. Pieces seen
 * while training the tokenizer already hold their ids; other pieces are
 * encoded with the learned merges. scratch needs room for the longest piece.
 */
static long encode_story(const Story *s, const PieceTable *pieces, const Bpe *bpe,
                         uint16_t *out, int *scratch) {
    long t = 0;
    const char *p = s->start;
    const char *end = p + s->length;
    while (p < end) {
        int len = piece_length(p);
        const Piece *piece = piece_table_find(pieces, p, len);
        const int *ids = piece->ids;
        int n = piece->n_ids;
        if (piece->start == NULL) {
            n = bpe_encode(bpe, p, len, scratch);
            ids = scratch;
        }
        for (int k = 0; k < n; k++) out[t++] = (uint16_t)ids[k];
        p += len;
    }
    return t;
}

/*
 * write_or_die
 * fwrite that exits with the file name on a short write.
 */
static void write_or_die(const void *data, size_t size, size_t count, FILE *f,
                         const char *path) {
    if (fwrite(data, size, count, f) != count) {
        perror(path);
        exit(1);
    }
}

/*
 * open_or_die
 * fopen for writing that exits with the file name on failure.
 */
static FILE *open_or_die(const char *path) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        perror(path);
        exit(1);
    }
    return f;
}

/*
 * close_or_die
 * fclose that exits if buffered data could not be written.
 */
static void close_or_die(FILE *f, const char *path) {
    if (fclose(f) != 0) {
        perror(path);
        exit(1);
    }
}

/*
 * write_stories
 * Writes EOS story EOS story ... story EOS, adding whole stories until there
 * are at least max_tokens + 1 ids (the +1 because each target is the next
 * id), or every story if max_tokens < 0. Ids are written in chunks of about
 * CHUNK_TOKENS. Exits if the stories run out first.
 */
static void write_stories(const char *path, const StorySet *set, const PieceTable *pieces,
                          const Bpe *bpe, long max_tokens) {
    FILE *f = open_or_die(path);
    uint16_t *buf = xmalloc(sizeof(uint16_t) * (CHUNK_TOKENS + set->longest + 2), "ids");
    int *scratch = xmalloc(sizeof(int) * set->longest, "scratch ids");
    long written = 0;
    long t = 0;
    int used = 0;
    while (used < set->count && (max_tokens < 0 || written + t < max_tokens)) {
        buf[t++] = BPE_EOS_ID;
        t += encode_story(&set->stories[used], pieces, bpe, buf + t, scratch);
        used++;
        if (t >= CHUNK_TOKENS) {
            write_or_die(buf, sizeof(uint16_t), t, f, path);
            written += t;
            t = 0;
        }
    }
    buf[t++] = BPE_EOS_ID;
    write_or_die(buf, sizeof(uint16_t), t, f, path);
    written += t;
    close_or_die(f, path);
    free(scratch);
    free(buf);
    if (max_tokens >= 0 && written < max_tokens + 1) {
        fprintf(stderr, "only %ld tokens from %d stories, read more bytes\n", written, used);
        exit(1);
    }
    printf("wrote %ld tokens (%d stories) to %s\n", written, used, path);
}

/*
 * write_vocab
 * Format described at the top of this file.
 */
static void write_vocab(const char *path, const Bpe *bpe) {
    FILE *f = open_or_die(path);
    int32_t size = bpe->vocab_size;
    write_or_die(&size, sizeof(int32_t), 1, f, path);
    for (int id = 0; id < bpe->vocab_size; id++) {
        const BpeToken *tok = &bpe->tokens[id];
        if (tok->len > 255) {
            fprintf(stderr, "token %d is too long for vocab.bin (%d chars)\n", id, tok->len);
            exit(1);
        }
        unsigned char len8 = (unsigned char)tok->len;
        write_or_die(&len8, 1, 1, f, path);
        write_or_die(tok->text, 1, tok->len, f, path);
    }
    close_or_die(f, path);
    printf("wrote %d tokens to %s\n", bpe->vocab_size, path);
}

/*
 * write_merges
 * Format described at the top of this file.
 */
static void write_merges(const char *path, const Bpe *bpe) {
    FILE *f = open_or_die(path);
    int32_t n = bpe->vocab_size - BPE_NUM_BASE;
    write_or_die(&n, sizeof(int32_t), 1, f, path);
    for (int id = BPE_NUM_BASE; id < bpe->vocab_size; id++) {
        uint16_t pair[2] = {(uint16_t)bpe->tokens[id].left, (uint16_t)bpe->tokens[id].right};
        write_or_die(pair, sizeof(uint16_t), 2, f, path);
    }
    close_or_die(f, path);
    printf("wrote %d merges to %s\n", n, path);
}

/*
 * print_tokens
 * Prints count tokens starting at id first, with '\n' shown as [\n].
 */
static void print_tokens(const char *label, const Bpe *bpe, int first, int count) {
    printf("%s:", label);
    for (int id = first; id < first + count && id < bpe->vocab_size; id++) {
        printf(" [");
        for (int k = 0; k < bpe->tokens[id].len; k++) {
            char c = bpe->tokens[id].text[k];
            if (c == '\n') printf("\\n");
            else putchar(c);
        }
        printf("]");
    }
    printf("\n");
}

/*
 * train_tokenizer
 * Counts the pieces in the first training stories, up to TOKENIZER_CHARS
 * characters, then learns the merges. A fixed amount of text means the vocab
 * stays the same if more training text is read later (e.g. a larger run).
 */
static void train_tokenizer(const StorySet *train, PieceTable *pieces, Bpe *bpe) {
    piece_table_init(pieces, 65536);
    long chars = 0;
    int used = 0;
    while (used < train->count && chars < TOKENIZER_CHARS) {
        piece_table_count(pieces, train->stories[used].start, train->stories[used].length);
        chars += train->stories[used].length;
        used++;
    }
    printf("tokenizer text: %d stories, %ld chars, %d distinct pieces\n", used, chars,
           pieces->count);
    bpe_train(bpe, pieces, VOCAB_SIZE);
    printf("vocab: %d tokens (%d base + %d merges)\n", bpe->vocab_size, BPE_NUM_BASE,
           bpe->vocab_size - BPE_NUM_BASE);
    print_tokens("first merges", bpe, BPE_NUM_BASE, 12);
    print_tokens("last merges", bpe, bpe->vocab_size - 8, 8);
}

/*
 * main
 * Train the tokenizer, then encode and write everything.
 */
int main(void) {
    StorySet train = load_stories(TRAIN_TEXT, TRAIN_READ_BYTES);
    PieceTable pieces;
    Bpe bpe;
    train_tokenizer(&train, &pieces, &bpe);
    write_stories(TRAIN_FILE, &train, &pieces, &bpe, TARGET_TOKENS);

    StorySet val = load_stories(VAL_TEXT, -1);
    write_stories(VAL_FILE, &val, &pieces, &bpe, -1);
    write_vocab(VOCAB_FILE, &bpe);
    write_merges(MERGES_FILE, &bpe);

    bpe_free(&bpe);
    piece_table_free(&pieces);
    free(train.buffer);
    free(train.stories);
    free(val.buffer);
    free(val.stories);
    return 0;
}
