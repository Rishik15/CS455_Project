#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <mpi.h>

#include "checkpoint.h"
#include "config.h"
#include "dataloader.h"
#include "model.h"
#include "optimizer.h"
#include "utils.h"

#define MASTER 0

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

int main(int argc, char **argv) {
    /*
     * Initial training settings
     */
    const int max_steps = 500;
    const int B = 32, T = 256;
    const double training_seconds = 30.0 * 60.0;
    const float learning_rate = 0.001f;
    const uint64_t seed = 0xbad;
    const float weight_decay = 0.01f, epsilon = 1e-8f;
    int status = EXIT_FAILURE;


    /*
     * Set Up the MPI communication world for the training
     */
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int exit_early = 0;

    if (B%num_procs != 0) {
        fprintf(stderr, "Could not evenly split batches\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int local_B = B/num_procs; //split the batches evenly through the available process

    char run_dir[128], checkpoint_path[160], vocab_path[160], merges_path[160];
    time_t started_at = time(NULL);
    struct tm local_start;
    if (my_rank == MASTER) {
        if (started_at == (time_t)-1 || !localtime_r(&started_at, &local_start) ||
            !strftime(run_dir, sizeof(run_dir),
                      "checkpoints/MPI-%Y%m%d-%H%M%S-XXXXXX", &local_start)) {
            fprintf(stderr, "Could not determine the run start time.\n");
            exit_early = 1;
            goto early_exit;
                      }
        if (make_directory("checkpoints") || !mkdtemp(run_dir)) {
            fprintf(stderr, "Could not create run directory: %s\n", strerror(errno));
            exit_early = 1;
            goto early_exit;
        }
        snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/model.bin", run_dir);
        snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.bin", run_dir);
        snprintf(merges_path, sizeof(merges_path), "%s/merges.bin", run_dir);
        if (copy_file("data/vocab.bin", vocab_path) ||
            copy_file("data/merges.bin", merges_path)) {
            fprintf(stderr, "Could not prepare %s and its tokenizer files.\n", run_dir);
            exit_early = 1;
            goto early_exit;
            }
        printf("Run directory: %s\n", run_dir);
    }

early_exit:
    MPI_Bcast(&exit_early, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    if (exit_early) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    GPT model = {0};
    gpt_build(&model, gpt_config_default(), local_B, T, seed);
    AdamW optimizer = {0};
    adamw_init(&optimizer, model.num_parameters, weight_decay, epsilon);
    DataLoader loader = {0};
    dataloader_init(&loader, "data/train.bin", local_B, T);

    if (my_rank == MASTER) {
        gpt_print_summary(&model);
        printf("Training: B=%d T=%d lr=%g tokens=%zu budget=%.1fs\n",
               B, T, learning_rate, loader.num_tokens, training_seconds);
        printf("Budget excludes setup and saving; a complete step may exceed the budget.\n");
    }

    int steps = 0;
    float loss = 0;
    double start = time_seconds();
    int flag;
    int local_flag;

    do {
        local_flag = 0;
        flag = 0;
        dataloader_next_MPI(&loader, B, my_rank);
        loss = gpt_forward(&model, loader.inputs, loader.targets, local_B, T);

        if (!isfinite(loss)) {
            local_flag = 1;
        }

        //sync cleanups if one were to happen
        MPI_Allreduce(&local_flag, &flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (flag) {
            if (my_rank == MASTER) {
                fprintf(stderr, "Non-finite loss at step %d; checkpoint not saved.\n", steps + 1);
            }
            goto cleanup;
        }

        gpt_zero_grad(&model);
        gpt_backward(&model);

        MPI_Allreduce(MPI_IN_PLACE, model.grads_memory, model.num_parameters, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i=0; i<model.num_parameters; i++) {model.grads_memory[i] /= num_procs;}

        adamw_update(&optimizer, model.grads_memory, model.params_memory, learning_rate);

        //TODO validation

        steps++;
        if (my_rank==MASTER){
            printf("Step %d | Loss: %.6f | Elapsed: %.3fs\n", steps, loss, time_seconds() - start);
            flag = steps<=max_steps;
        }
        MPI_Bcast(&flag, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    } while (flag);

    if (my_rank == MASTER) {
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
    }
    status = EXIT_SUCCESS;

cleanup:
    dataloader_free(&loader);
    adamw_free(&optimizer);
    gpt_free(&model);
    MPI_Finalize();
    return status;
}
