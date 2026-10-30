/*
 * config.h
 * Model hyperparameters. GPTConfig describes the shape of the network only;
 * batch size and learning rate belong to the training loop.
 * Defaults follow the Pythia-14M shape with a 4096-token BPE vocab.
 */
#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    int vocab_size;
    int max_seq_len;
    int n_layers;
    int n_heads;
    int channels;
} GPTConfig;

/*
 * gpt_config_default
 * vocab 4096, context 256, 6 layers, 4 heads, 128 channels (head size 32).
 */
static inline GPTConfig gpt_config_default(void) {
    GPTConfig config = {4096, 256, 6, 4, 128};
    return config;
}

#endif
