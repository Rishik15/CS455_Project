#ifndef CONFIG_H
#define CONFIG_H

// One place for the model size and training settings.
// All four versions (serial, MPI, CUDA, MPI+CUDA) use these same values
// so the benchmark comparison is fair.

// ---- data (made by prep_data.c) ----
#define TRAIN_PATH   "data/train.bin"
#define VAL_PATH     "data/val.bin"
#define VOCAB_PATH   "data/vocab.bin"   // the vocab size comes from this file

// ---- model: Pythia-14M shape with a character vocabulary ----
#define MAX_SEQ_LEN  256   // context length, in characters
#define N_LAYER      6
#define N_HEAD       4
#define N_EMBD       128   // hidden size
#define N_FF         512   // MLP hidden size (4 * N_EMBD)

// ---- training ----
#define BATCH_SIZE    32        // sequences per step, total across all processes/GPUs
#define TRAIN_TOKENS  5000000   // train on this many tokens (~610 steps of 32 x 256)
#define VAL_TOKENS    250000    // size of the validation set
#define LEARNING_RATE 1e-3f     // AdamW: starting values, tune with the serial version
#define ADAM_BETA1    0.9f
#define ADAM_BETA2    0.95f
#define ADAM_EPS      1e-8f
#define WEIGHT_DECAY  0.0f
#define SEED          455       // weight initialization seed

#endif
