#define _POSIX_C_SOURCE 200809L
#include "checkpoint.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char path[] = "/tmp/cs455-checkpoint-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    GPT a = {0}, b = {0};
    GPTConfig c = {32, 8, 2, 2, 8};
    gpt_build(&a, c, 2, 8, 123);
    /* Exercise all parameters, including biases normally initialized to zero. */
    for (size_t i = 0; i < a.num_parameters; i++) a.params_memory[i] += 0.001f;
    assert(checkpoint_save(path, &a) == 0);
    assert(checkpoint_load(path, &b, 1, 4) == 0);
    assert(b.config.vocab_size == c.vocab_size && b.config.max_seq_len == c.max_seq_len);
    assert(b.config.n_layers == c.n_layers && b.config.n_heads == c.n_heads &&
           b.config.channels == c.channels);
    assert(a.num_parameters == b.num_parameters);
    assert(memcmp(a.params_memory, b.params_memory, a.num_parameters*sizeof(float)) == 0);
    int tokens[] = {1, 2, 3, 4}, targets[] = {2, 3, 4, 5};
    assert(gpt_forward(&a, tokens, targets, 1, 4) == gpt_forward(&b, tokens, targets, 1, 4));
    assert(memcmp(a.acts.logits, b.acts.logits, 4*32*sizeof(float)) == 0);
    gpt_free(&b);
    /* Unsupported version. */
    FILE *f = fopen(path, "r+b");
    assert(f); assert(fseek(f, 8, SEEK_SET) == 0); assert(fputc(2, f) != EOF); fclose(f);
    assert(checkpoint_load(path, &b, 1, 4) == -1 && !b.params_memory);
    /* Mismatched parameter count. */
    assert(checkpoint_save(path, &a) == 0);
    f = fopen(path, "r+b"); assert(f); assert(fseek(f, 32, SEEK_SET) == 0);
    int byte = fgetc(f); assert(fseek(f, 32, SEEK_SET) == 0);
    assert(fputc(byte ^ 1, f) != EOF); fclose(f);
    assert(checkpoint_load(path, &b, 1, 4) == -1 && !b.params_memory);
    /* Truncated file. */
    assert(checkpoint_save(path, &a) == 0);
    assert(truncate(path, 44) == 0);
    assert(checkpoint_load(path, &b, 1, 4) == -1 && !b.params_memory);
    unlink(path);
    gpt_free(&a);
    puts("Checkpoint tests passed.");
    return 0;
}
