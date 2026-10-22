#include <stdio.h>
#include <stdint.h>

#include "vocab.h"

int vocab_load(Vocab *v, const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        return -1;
    }

    int32_t size;
    if (fread(&size, sizeof(size), 1, file) != 1 || size < 2 || size > 256) {
        fprintf(stderr, "%s: bad vocab size\n", path);
        fclose(file);
        return -1;
    }
    if (fread(v->id_to_char, 1, size, file) != (size_t)size || v->id_to_char[EOS_ID] != '\0') {
        fprintf(stderr, "%s: bad vocab table\n", path);
        fclose(file);
        return -1;
    }
    fclose(file);

    v->size = size;
    for (int c = 0; c < 256; c++) v->char_to_id[c] = -1;
    for (int id = 1; id < size; id++) v->char_to_id[(unsigned char)v->id_to_char[id]] = id;
    return 0;
}
