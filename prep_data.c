/*
data_prep

1. read the training.txt file (20MB) (DONE)
2. split up the stories in the 20MB and change the delimeter to '\0' (DONE)
3. check that the stories contain only valid characters 32-126 + \n (DONE)
4. iterate through the stories, tracking all the characters that show up, keep track of vocab_size + eos
5. create a token map doing char_to_id and id_to_char for each vocab character, to make encoding and decoding easier in the future
6. tokenize the stories that were kept using char_to_id and id_to_char
7. write the encoded stories into train.bin
8. write the vocab into vocab.bin

 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define READ_BYTES  (20 * 1024 * 1024)   // first 20 MB of the file
#define MAX_STORIES 100000
#define TARGET_CHAR 5000000
#define SEP "<|endoftext|>"

typedef struct {
    int index;
    int length;
    char *start;
} Story;

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

    int vocab[128]={0};
    int vocab_size =0;
    for (int i=0; i<kept_count; i++) {
        char *c=kept_stories[i].start;
        while (*c) {
            if (!vocab[(unsigned char)*c]) {
                vocab[(unsigned char)*c]=1;
                vocab_size++;
            }
            c++;
        }
    }
    vocab_size++; // eos token

    free(buffer);
    free(stories);
    free(kept_stories);
    return 0;
}