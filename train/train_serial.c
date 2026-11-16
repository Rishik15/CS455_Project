#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include "checkpoint.h"
#include "config.h"
#include "dataloader.h"
#include "model.h"
#include "optimizer.h"
#include "utils.h"

#define RUN_DIR "checkpoints/serial-trial"
#define CHECKPOINT_PATH RUN_DIR "/model.bin"

static int make_directory(const char *path) {
    if (mkdir(path, 0755) == 0) return 0;
    struct stat st;
    return errno == EEXIST && stat(path, &st) == 0 && S_ISDIR(st.st_mode) ? 0 : -1;
}

static int copy_file(const char *source, const char *destination) {
    FILE *in = fopen(source, "rb");
    if (!in) return -1;
    FILE *out = fopen(destination, "wb");
    if (!out) { fclose(in); return -1; }
    unsigned char buffer[8192];
    size_t n;
    int failed = 0;
    while ((n = fread(buffer, 1, sizeof(buffer), in)) != 0) {
        if (fwrite(buffer, 1, n, out) != n) { failed = 1; break; }
    }
    if (ferror(in)) failed = 1;
    if (fclose(in)) failed = 1;
    if (fclose(out)) failed = 1;
    return failed ? -1 : 0;
}

int main(void) {
    const int B = 32, T = 256;
    const double training_seconds = 20.0;
    const float learning_rate = 0.001f;
    const uint64_t seed = 0xbad;
    const float weight_decay = 0.01f, epsilon = 1e-8f;
    int status = EXIT_FAILURE;
    setvbuf(stdout, NULL, _IOLBF, 0);

    if (access(CHECKPOINT_PATH, F_OK) == 0) {
        fprintf(stderr, "%s already exists; move the run directory before another trial.\n",
                CHECKPOINT_PATH);
        return EXIT_FAILURE;
    }
    if (make_directory("checkpoints") || make_directory(RUN_DIR) ||
        copy_file("data/vocab.bin", RUN_DIR "/vocab.bin") ||
        copy_file("data/merges.bin", RUN_DIR "/merges.bin")) {
        fprintf(stderr, "Could not prepare %s and its tokenizer files.\n", RUN_DIR);
        return EXIT_FAILURE;
    }

    GPT model = {0};
    gpt_build(&model, gpt_config_default(), B, T, seed);
    gpt_print_summary(&model);
    AdamW optimizer = {0};
    adamw_init(&optimizer, model.num_parameters, weight_decay, epsilon);
    DataLoader loader = {0};
    dataloader_init(&loader, "data/train.bin", B, T);
    printf("Training: B=%d T=%d lr=%g tokens=%zu budget=%.1fs\n",
           B, T, learning_rate, loader.num_tokens, training_seconds);
    printf("Budget excludes setup and saving; a complete step may exceed the budget.\n");

    int steps = 0;
    float loss = 0;
    double start = time_seconds();
    do {
        dataloader_next(&loader);
        loss = gpt_forward(&model, loader.inputs, loader.targets, B, T);
        if (!isfinite(loss)) {
            fprintf(stderr, "Non-finite loss at step %d; checkpoint not saved.\n", steps + 1);
            goto cleanup;
        }
        gpt_zero_grad(&model);
        gpt_backward(&model);
        adamw_update(&optimizer, model.grads_memory, model.params_memory, learning_rate);
        steps++;
        printf("Step %d | Loss: %.6f | Elapsed: %.3fs\n", steps, loss, time_seconds() - start);
    } while (time_seconds() - start < training_seconds);
    double elapsed = time_seconds() - start;
    for (size_t i = 0; i < model.num_parameters; i++) {
        if (!isfinite(model.params_memory[i])) {
            fprintf(stderr, "Non-finite parameter after update; checkpoint not saved.\n");
            goto cleanup;
        }
    }
    if (checkpoint_save(CHECKPOINT_PATH, &model)) goto cleanup;
    printf("Saved weights: %s\n", CHECKPOINT_PATH);
    printf("Completed %d steps in %.3fs (%.1f tokens/s). Final batch loss: %.6f\n",
           steps, elapsed, (double)steps * B * T / elapsed, loss);
    status = EXIT_SUCCESS;
cleanup:
    dataloader_free(&loader);
    adamw_free(&optimizer);
    gpt_free(&model);
    return status;
}
