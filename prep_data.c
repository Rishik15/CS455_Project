/*
data_prep

1. read the training.txt file (20MB) (DONE)
2. split up the stories in the 20MB and change the delimeter to '\0' (DONE)
3. check that the stories contain only valid characters 32-126 + \n (DONE)
4. split the stories into word tokens and count each token, stopping once there are TARGET_TOKENS tokens (DONE)
5. pick the vocab: end-of-story, unknown, then the most common tokens; give each an id (DONE)
6. encode the stories used in step 4 as token ids (unknown id for rare words)
7. write the encoded stories into train.bin (2 bytes per token)
8. write the vocab into vocab.bin

 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define READ_BYTES    (25 * 1024 * 1024)   // first 25 MB of the file (20 MB is just short of 5M word tokens)
#define MAX_STORIES   100000
#define TARGET_TOKENS 5000000
#define VOCAB_SIZE    4096
#define TABLE_SIZE    65536                // hash table slots: power of 2, well above the ~15K distinct tokens
#define EOS_ID 0                           // end-of-story token
#define UNK_ID 1                           // unknown token, for words not in the vocab
#define SEP "<|endoftext|>"

typedef struct {
    int index;
    int length;
    char *start;
} Story;

// one distinct token in the hash table
typedef struct {
    const char *start;  // points into buffer
    int len;
    int count;          // how many times it appears
    int id;             // token id, -1 if it didn't make the vocab
} Entry;

// Length of the token that starts at p. Words, numbers and punctuation marks
// keep the single space in front of them (" said", " \""), and '\n' is its own
// token, so gluing the tokens back together gives the original text.
static int token_length(const char *p) {
    int i = 0;
    if (p[0] == '\n') return 1;
    if (p[0] == ' ') i = 1;  // optional leading space
    if (isalpha((unsigned char)p[i])) {
        while (isalpha((unsigned char)p[i])) i++;
    } else if (isdigit((unsigned char)p[i])) {
        while (isdigit((unsigned char)p[i])) i++;
    } else if (p[i] != ' ' && p[i] != '\n' && p[i] != '\0') {
        i++;  // one punctuation mark
    }
    return i > 0 ? i : 1;  // a space with nothing after it is a token on its own
}

// FNV-1a hash of len bytes
static uint32_t hash(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

// Returns the slot holding this token, or the empty slot where it belongs.
// If the slot is taken by a different token, try the next one (linear probing).
static Entry *find_slot(Entry *table, const char *s, int len) {
    uint32_t i = hash(s, len) & (TABLE_SIZE - 1);
    while (table[i].start != NULL &&
           !(table[i].len == len && memcmp(table[i].start, s, len) == 0)) {
        i = (i + 1) & (TABLE_SIZE - 1);
    }
    return &table[i];
}

// qsort order for the vocab: most common first, ties broken by the token text
static int compare_entries(const void *a, const void *b) {
    const Entry *x = *(Entry *const *)a;
    const Entry *y = *(Entry *const *)b;
    if (x->count != y->count) return y->count - x->count;
    int n = x->len < y->len ? x->len : y->len;
    int c = memcmp(x->start, y->start, n);
    if (c != 0) return c;
    return x->len - y->len;
}

int main(void) {
    FILE *file = fopen("data/TinyStories-train.txt", "rb");
    if (file == NULL) {
        perror("failed to read training data from file");
        return 1;
    }

    char *buffer = malloc(sizeof(char) * READ_BYTES+1);
    if (buffer == NULL) {
        perror("Failed to allocate memory for buffer");
        fclose(file);
        return 1;
    }

    size_t data_read = fread(buffer, sizeof(char), READ_BYTES, file); // get number of data read
    fclose(file);

    printf("Successfully read %zu bytes into memory.\n", data_read);
    buffer[data_read] = '\0'; // null terminate end of data

    char *story_end;
    Story *stories = (Story *)malloc(sizeof(Story) * MAX_STORIES); // array to store stories
    int story_count = 0;
    char *curr = buffer;

    if (stories == NULL) {
        printf("Memory allocation failed\n");
        free(buffer);
        return 1;
    }

    //separate stories, remove trailing and leading white space
    while ((story_end = strstr(curr, SEP)) && story_count < MAX_STORIES){
        *story_end = '\0'; // replace <|endoftext|> with null terminator
        char *p = curr;
        char *end = story_end-1;

        while (*p && isspace((unsigned char)*p)){p++;}
        while (end >= p && isspace((unsigned char)*end)) {
            *end='\0';
            end--;
        }

        //skips empty stories
        int length=end-p+1;
        if (length>0) {
            stories[story_count].index = story_count;
            stories[story_count].length = length;
            stories[story_count].start = p;
            story_count++;
        }

        curr = story_end+strlen(SEP);
    }

    Story *kept_stories = malloc(sizeof(Story) * MAX_STORIES);
    if (kept_stories == NULL) {
        printf("Memory allocation failed\n");
        free(buffer);
        free(stories);
        return 1;
    }
    int kept_count=0;
    int store_flag;
    for (int i=0; i<story_count; i++) {
        store_flag = 1;
        char *c=stories[i].start;
        while (*c) {
            if (*c != '\n' && (*c<32 || *c>126)) {
                store_flag = 0;
                break;
            }
            c++;
        }
        if (store_flag) {
            kept_stories[kept_count] = stories[i];
            kept_count++;
        }
    }

    Entry *table = calloc(TABLE_SIZE, sizeof(Entry));
    if (table == NULL) {
        printf("Memory allocation failed\n");
        free(buffer);
        free(stories);
        free(kept_stories);
        return 1;
    }

    //count every token in the stories, until there are TARGET_TOKENS + 1 tokens
    //(+1 because each training target is the next token)
    long total_tokens = 1; // end-of-story token at the very end of train.bin
    int used_count = 0;    // stories needed to reach the target
    int distinct = 0;
    for (int i=0; i<kept_count && total_tokens < TARGET_TOKENS+1; i++) {
        char *p = kept_stories[i].start;
        char *end = p + kept_stories[i].length;
        while (p < end) {
            int len = token_length(p);
            Entry *e = find_slot(table, p, len);
            if (e->start == NULL) {
                if (distinct >= TABLE_SIZE/2) { // keep the table at most half full so probing stays fast
                    printf("too many distinct tokens, increase TABLE_SIZE\n");
                    return 1;
                }
                e->start = p;
                e->len = len;
                e->id = -1;
                distinct++;
            }
            e->count++;
            total_tokens++;
            p += len;
        }
        total_tokens++; // end-of-story token before each story
        used_count++;
    }
    if (total_tokens < TARGET_TOKENS+1) {
        printf("only %ld tokens, increase READ_BYTES\n", total_tokens);
        return 1;
    }

    //sort the distinct tokens, most common first
    Entry **sorted = malloc(sizeof(Entry *) * distinct);
    if (sorted == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    int n = 0;
    for (int i=0; i<TABLE_SIZE; i++) {
        if (table[i].start != NULL) sorted[n++] = &table[i];
    }
    qsort(sorted, distinct, sizeof(Entry *), compare_entries);

    //ids: 0 = end-of-story, 1 = unknown, then the most common tokens.
    //the table is now word_to_id (find_slot, then read id), and sorted[id-2] is id_to_word
    int vocab_size = 2;
    long covered = 0;
    for (int i=0; i<distinct && vocab_size<VOCAB_SIZE; i++) {
        sorted[i]->id = vocab_size;
        covered += sorted[i]->count;
        vocab_size++;
    }

    long word_tokens = total_tokens - used_count - 1; // without the end-of-story tokens
    printf("%d stories used, %ld tokens, %d distinct tokens\n", used_count, total_tokens, distinct);
    printf("vocab: %d ids, covering %.2f%% of word tokens (the rest become unknown)\n",
           vocab_size, 100.0 * covered / word_tokens);
    printf("most common:");
    for (int i=0; i<10; i++) {
        if (sorted[i]->start[0] == '\n') printf(" [\\n]");
        else printf(" [%.*s]", sorted[i]->len, sorted[i]->start);
    }
    printf("\n");

    free(sorted);
    free(table);
    free(buffer);
    free(stories);
    free(kept_stories);
    return 0;
}