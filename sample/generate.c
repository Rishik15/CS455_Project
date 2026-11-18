#include "checkpoint.h"
#include "utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Unconditional sampling: EOS starts a story; no prompt encoding needed. */
int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "checkpoints/serial-trial";
    char *end;
    long count = 64, seed = 42;
    if (argc > 4) goto usage;
    if (argc > 2) {
        count = strtol(argv[2], &end, 10);
        if (!*argv[2] || *end || count < 1 || count > 256) goto usage;
    }
    if (argc > 3) {
        seed = strtol(argv[3], &end, 10);
        if (!*argv[3] || *end || seed < 1 || seed > 2147483647) goto usage;
    }
    char path[4096];
    GPT model = {0};
    char (*vocab)[256] = NULL;
    FILE *f = NULL;
    int result = 1;
    if (snprintf(path, sizeof(path), "%s/model.bin", dir) >= (int)sizeof(path)) goto done;
    if (checkpoint_load(path, &model, 1, (int)count)) goto done;
    if (snprintf(path, sizeof(path), "%s/vocab.bin", dir) >= (int)sizeof(path)) goto done;
    f = fopen(path, "rb");
    int32_t size;
    if (!f || fread(&size, sizeof(size), 1, f) != 1 || size != model.config.vocab_size) goto done;
    vocab = xcalloc((size_t)size, sizeof(*vocab), "vocabulary");
    for (int i = 0; i < size; i++) {
        int len = fgetc(f);
        if (len < 1 || fread(vocab[i], 1, (size_t)len, f) != (size_t)len) goto done;
    }
    if (fgetc(f) != EOF || ferror(f)) goto done;
    int ids[257] = {0};
    rng_seed((uint64_t)seed);
    for (int t = 1; t <= count; t++) {
        gpt_forward(&model, ids, NULL, 1, t);
        const float *logits = model.acts.logits + (size_t)(t - 1) * size;
        float max = logits[0];
        for (int i = 0; i < size; i++) {
            if (!isfinite(logits[i])) goto done;
            if (logits[i] > max) max = logits[i];
        }
        double sum = 0;
        for (int i = 0; i < size; i++) sum += exp((double)logits[i] - max);
        double draw = rng_uniform() * sum;
        int next = size - 1;
        for (int i = 0; i < size; i++) {
            draw -= exp((double)logits[i] - max);
            if (draw <= 0) { next = i; break; }
        }
        if (next == 0) break;
        ids[t] = next;
        fputs(vocab[next], stdout);
        fflush(stdout);
    }
    putchar('\n');
    result = 0;
done:
    if (result) fprintf(stderr, "Generation failed: check checkpoint, vocabulary, and context length.\n");
    if (f) fclose(f);
    free(vocab);
    gpt_free(&model);
    return result;
usage:
    fprintf(stderr, "Usage: %s [checkpoint-dir [tokens:1-256 [seed:1-2147483647]]]\n", argv[0]);
    return 1;
}
