/*
data_prep: turns the TinyStories text into token files for training.

1. read the first 20MB of TinyStories-train.txt
2. split it into stories at <|endoftext|> and trim leading/trailing whitespace
3. keep only stories made of characters 32-126 and '\n'
4. take stories in order until there are TRAIN_TOKENS + 1 tokens
   (+1 because each target is the next token)
5. build the vocab from the characters in those stories: id 0 is the
   end-of-story token, then the characters in ASCII order
6. encode those stories and write data/train.bin
7. write the vocab to data/vocab.bin
8. do the same for TinyStories-valid.txt -> data/val.bin, using the train vocab

The file formats are described in src/vocab.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#include "src/config.h"
#include "src/vocab.h"

#define TRAIN_TEXT  "data/TinyStories-train.txt"
#define VALID_TEXT  "data/TinyStories-valid.txt"
#define READ_BYTES  (20 * 1024 * 1024)   // first 20 MB of each file
#define MAX_STORIES 100000
#define SEP "<|endoftext|>"

typedef struct {
    int length;
    char *start;
} Story;

// Reads the first READ_BYTES of path into *buffer_out and splits it into
// stories. If skip_first is set, the text before the first separator is
// dropped (the validation file starts in the middle of a story). The text
// after the last separator is always dropped, since it may be cut off.
// Returns the number of stories, or -1 on error.
static int read_stories(const char *path, int skip_first, char **buffer_out, Story *stories) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        return -1;
    }

    char *buffer = malloc(sizeof(char) * READ_BYTES + 1);
    if (buffer == NULL) {
        perror("Failed to allocate memory for buffer");
        fclose(file);
        return -1;
    }

    size_t data_read = fread(buffer, sizeof(char), READ_BYTES, file);
    fclose(file);
    buffer[data_read] = '\0'; // null terminate end of data
    printf("read %zu bytes from %s\n", data_read, path);

    char *story_end;
    int story_count = 0;
    char *curr = buffer;

    if (skip_first && (story_end = strstr(curr, SEP))) {
        curr = story_end + strlen(SEP);
    }

    //separate stories, remove trailing and leading white space
    while ((story_end = strstr(curr, SEP)) && story_count < MAX_STORIES) {
        *story_end = '\0'; // replace <|endoftext|> with null terminator
        char *p = curr;
        char *end = story_end - 1;

        while (*p && isspace((unsigned char)*p)) {p++;}
        while (end >= p && isspace((unsigned char)*end)) {
            *end = '\0';
            end--;
        }

        //skips empty stories
        int length = end - p + 1;
        if (length > 0) {
            stories[story_count].length = length;
            stories[story_count].start = p;
            story_count++;
        }

        curr = story_end + strlen(SEP);
    }

    *buffer_out = buffer;
    return story_count;
}

// 1 if the story only has characters 32-126 and '\n'
static int is_clean(const Story *s) {
    for (int i = 0; i < s->length; i++) {
        unsigned char c = s->start[i];
        if (c != '\n' && (c < 32 || c > 126)) return 0;
    }
    return 1;
}

// 1 if every character in the story is in the vocab
static int in_vocab(const Story *s, const Vocab *v) {
    for (int i = 0; i < s->length; i++) {
        if (v->char_to_id[(unsigned char)s->start[i]] < 0) return 0;
    }
    return 1;
}

// Number of stories (taken in order) needed to reach at least min_tokens
// tokens. Each story costs its length + 1 (the end-of-story token before it),
// plus 1 for the end-of-story token that ends the file.
// Returns -1 if all the stories together are not enough.
static int stories_for_tokens(const Story *stories, int count, long min_tokens) {
    long tokens = 1;
    int n = 0;
    while (n < count && tokens < min_tokens) {
        tokens += stories[n].length + 1;
        n++;
    }
    return tokens >= min_tokens ? n : -1;
}

// Writes the stories as token ids: EOS story EOS story ... story EOS.
// Returns the number of tokens written, or -1 on error.
static long write_tokens(const char *path, const Story *stories, int count, const Vocab *v) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror(path);
        return -1;
    }
    long tokens = 0;
    for (int i = 0; i < count; i++) {
        fputc(EOS_ID, file);
        for (int j = 0; j < stories[i].length; j++) {
            fputc(v->char_to_id[(unsigned char)stories[i].start[j]], file);
        }
        tokens += stories[i].length + 1;
    }
    fputc(EOS_ID, file);
    tokens++;
    if (fclose(file) != 0) {
        perror(path);
        return -1;
    }
    return tokens;
}

static int write_vocab(const char *path, const Vocab *v) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror(path);
        return -1;
    }
    int32_t size = v->size;
    fwrite(&size, sizeof(size), 1, file);
    fwrite(v->id_to_char, 1, v->size, file);
    if (fclose(file) != 0) {
        perror(path);
        return -1;
    }
    return 0;
}

int main(void) {
    Story *stories = malloc(sizeof(Story) * MAX_STORIES); // array to store stories
    if (stories == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // ---- training data ----
    char *train_text;
    int story_count = read_stories(TRAIN_TEXT, 0, &train_text, stories);
    if (story_count < 0) {
        free(stories);
        return 1;
    }

    // keep the clean stories, moving them to the front of the array
    int kept_count = 0;
    for (int i = 0; i < story_count; i++) {
        if (is_clean(&stories[i])) stories[kept_count++] = stories[i];
    }

    int train_count = stories_for_tokens(stories, kept_count, TRAIN_TOKENS + 1);
    if (train_count < 0) {
        printf("not enough clean stories for %d tokens, increase READ_BYTES\n", TRAIN_TOKENS);
        free(train_text);
        free(stories);
        return 1;
    }

    // vocab: id 0 is end-of-story, then every character used, in ASCII order
    int seen[256] = {0};
    for (int i = 0; i < train_count; i++) {
        for (int j = 0; j < stories[i].length; j++) seen[(unsigned char)stories[i].start[j]] = 1;
    }
    Vocab vocab;
    vocab.size = 1;
    vocab.id_to_char[EOS_ID] = '\0';
    for (int c = 0; c < 256; c++) {
        vocab.char_to_id[c] = -1;
        if (seen[c]) {
            vocab.char_to_id[c] = vocab.size;
            vocab.id_to_char[vocab.size] = (char)c;
            vocab.size++;
        }
    }

    long train_tokens = write_tokens(TRAIN_PATH, stories, train_count, &vocab);
    if (train_tokens < 0 || write_vocab(VOCAB_PATH, &vocab) < 0) {
        free(train_text);
        free(stories);
        return 1;
    }
    printf("train: %d stories, %d clean, %d used -> %ld tokens in %s\n",
           story_count, kept_count, train_count, train_tokens, TRAIN_PATH);
    printf("vocab: %d tokens (%d characters + end-of-story) in %s\n",
           vocab.size, vocab.size - 1, VOCAB_PATH);
    free(train_text);

    // ---- validation data ----
    char *valid_text;
    story_count = read_stories(VALID_TEXT, 1, &valid_text, stories);
    if (story_count < 0) {
        free(stories);
        return 1;
    }

    // keep stories the train vocab can encode
    kept_count = 0;
    for (int i = 0; i < story_count; i++) {
        if (in_vocab(&stories[i], &vocab)) stories[kept_count++] = stories[i];
    }

    int val_count = stories_for_tokens(stories, kept_count, VAL_TOKENS);
    if (val_count < 0) val_count = kept_count; // use everything there is

    long val_tokens = write_tokens(VAL_PATH, stories, val_count, &vocab);
    if (val_tokens < 0) {
        free(valid_text);
        free(stories);
        return 1;
    }
    printf("val: %d stories, %d encodable, %d used -> %ld tokens in %s\n",
           story_count, kept_count, val_count, val_tokens, VAL_PATH);

    free(valid_text);
    free(stories);
    return 0;
}
