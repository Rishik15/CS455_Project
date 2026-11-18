#define _POSIX_C_SOURCE 200809L
#include "checkpoint.h"
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* V1: 8-byte magic, LE uint32 version, five LE uint32 config fields,
 * LE uint64 parameter count, then LE IEEE-754 binary32 parameters.
 * Parameter order is param_list() in model.c. Header is 40 bytes. */
static const unsigned char magic[8] = {'C','S','4','5','5','G','P','T'};

static int put_uint(FILE *f, uint64_t x, int bytes) {
    for (int i = 0; i < bytes; i++, x >>= 8)
        if (fputc((int)(x & 255), f) == EOF) return -1;
    return 0;
}

static int get_uint(FILE *f, uint64_t *x, int bytes) {
    *x = 0;
    for (int i = 0; i < bytes; i++) {
        int c = fgetc(f);
        if (c == EOF) return -1;
        *x |= (uint64_t)c << (8 * i);
    }
    return 0;
}

static int supported_float(void) {
    return sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128;
}

int checkpoint_save(const char *path, const GPT *model) {
    if (!supported_float() || !model->params_memory) return -1;
    char *tmp = malloc(strlen(path) + 12);
    if (!tmp) return -1;
    sprintf(tmp, "%s.tmpXXXXXX", path);
    int fd = mkstemp(tmp);
    FILE *f = fd < 0 ? NULL : fdopen(fd, "wb");
    int failed = !f;
    if (!f && fd >= 0) close(fd);
    if (f) {
        GPTConfig c = model->config;
        int fields[] = {c.vocab_size, c.max_seq_len, c.n_layers, c.n_heads, c.channels};
        failed = fwrite(magic, 1, 8, f) != 8 || put_uint(f, 1, 4);
        for (int i = 0; i < 5 && !failed; i++) failed = put_uint(f, fields[i], 4);
        if (!failed) failed = put_uint(f, model->num_parameters, 8);
        for (size_t i = 0; i < model->num_parameters && !failed; i++) {
            uint32_t bits;
            memcpy(&bits, &model->params_memory[i], 4);
            failed = put_uint(f, bits, 4);
        }
        if (fclose(f)) failed = 1;
    }
    if (!failed && rename(tmp, path)) failed = 1;
    if (failed) {
        fprintf(stderr, "checkpoint: could not save %s\n", path);
        unlink(tmp);
    }
    free(tmp);
    return failed ? -1 : 0;
}

int checkpoint_load(const char *path, GPT *model, int max_batch, int max_seq) {
    FILE *f = fopen(path, "rb");
    GPT loaded = {0};
    unsigned char signature[8];
    uint64_t version, fields[5], count;
    if (!f) goto fail;
    if (!supported_float() || fread(signature, 1, 8, f) != 8 ||
        memcmp(signature, magic, 8) || get_uint(f, &version, 4) || version != 1) goto fail;
    for (int i = 0; i < 5; i++) {
        if (get_uint(f, &fields[i], 4) || fields[i] == 0 || fields[i] > 65536) goto fail;
    }
    GPTConfig c = {(int)fields[0], (int)fields[1], (int)fields[2],
                   (int)fields[3], (int)fields[4]};
    if (c.channels % c.n_heads || max_batch < 1 || max_seq < 1 ||
        max_seq > c.max_seq_len || max_batch > INT_MAX / max_seq) goto fail;
    /* Same tensor sizes as model.c, computed in wide arithmetic. */
    uint64_t C = fields[4], L = fields[2];
    uint64_t expected = fields[0]*C + fields[1]*C + L*(12*C*C + 13*C) + 2*C;
    if (get_uint(f, &count, 8) || count != expected || count > (SIZE_MAX - 40) / 4 ||
        count > (LONG_MAX - 40) / 4) goto fail;
    if (fseek(f, 0, SEEK_END) || ftell(f) != (long)(40 + 4*count) ||
        fseek(f, 40, SEEK_SET)) goto fail;
    gpt_build(&loaded, c, max_batch, max_seq, 1);
    for (size_t i = 0; i < loaded.num_parameters; i++) {
        uint64_t value;
        if (get_uint(f, &value, 4)) goto fail;
        uint32_t bits = (uint32_t)value;
        memcpy(&loaded.params_memory[i], &bits, 4);
    }
    if (fclose(f)) { f = NULL; goto fail; }
    *model = loaded;
    return 0;
fail:
    if (f) fclose(f);
    gpt_free(&loaded);
    fprintf(stderr, "checkpoint: could not load %s (I/O error or invalid format)\n", path);
    return -1;
}
