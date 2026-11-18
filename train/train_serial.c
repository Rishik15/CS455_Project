#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "checkpoint.h"
#include "config.h"
#include "dataloader.h"
#include "model.h"
#include "optimizer.h"
#include "utils.h"

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
    const int max_steps = 500;
    const float learning_rate = 0.001f;
    const uint64_t seed = 0xbad;
    const float weight_decay = 0.01f, epsilon = 1e-8f;
    int status = EXIT_FAILURE;
    setvbuf(stdout, NULL, _IOLBF, 0);

    time_t started_at = time(NULL);
    struct tm local_start;
    char run_dir[128], checkpoint_path[160], vocab_path[160], merges_path[160];
    if (started_at == (time_t)-1 || !localtime_r(&started_at, &local_start) ||
        !strftime(run_dir, sizeof(run_dir),
                  "checkpoints/serial-%Y%m%d-%H%M%S-XXXXXX", &local_start)) {
        fprintf(stderr, "Could not determine the run start time.\n");
        return EXIT_FAILURE;
    }
    if (make_directory("checkpoints") || !mkdtemp(run_dir)) {
        fprintf(stderr, "Could not create run directory: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/model.bin", run_dir);
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.bin", run_dir);
    snprintf(merges_path, sizeof(merges_path), "%s/merges.bin", run_dir);
    if (copy_file("data/vocab.bin", vocab_path) ||
        copy_file("data/merges.bin", merges_path)) {
        fprintf(stderr, "Could not prepare %s and its tokenizer files.\n", run_dir);
        return EXIT_FAILURE;
    }
    printf("Run directory: %s\n", run_dir);

    GPT model = {0};
    gpt_build(&model, gpt_config_default(), B, T, seed);
    gpt_print_summary(&model);
    AdamW optimizer = {0};
    adamw_init(&optimizer, model.num_parameters, weight_decay, epsilon);
    DataLoader loader = {0};
    dataloader_init(&loader, "data/train.bin", B, T);
    printf("Training: B=%d T=%d lr=%g tokens=%zu budget=%d steps\n",
               B, T, learning_rate, loader.num_tokens, max_steps);

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
    } while (steps < max_steps);
    double elapsed = time_seconds() - start;
    for (size_t i = 0; i < model.num_parameters; i++) {
        if (!isfinite(model.params_memory[i])) {
            fprintf(stderr, "Non-finite parameter after update; checkpoint not saved.\n");
            goto cleanup;
        }
    }
    if (checkpoint_save(checkpoint_path, &model)) goto cleanup;
    printf("Saved weights: %s\n", checkpoint_path);
    printf("Completed %d steps in %.3fs (%.1f tokens/s). Final batch loss: %.6f\n",
           steps, elapsed, (double)steps * B * T / elapsed, loss);
    status = EXIT_SUCCESS;
cleanup:
    dataloader_free(&loader);
    adamw_free(&optimizer);
    gpt_free(&model);
    return status;
}
