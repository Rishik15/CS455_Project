# CS455 Project

A small language model written from scratch in C. It learns to write short children's stories by reading the TinyStories dataset and predicting the next piece of text.

We train it four ways and compare the speed: on one CPU, across several machines with MPI, on one GPU with CUDA, and across several machines with GPUs (MPI + CUDA).


## Running the Serial Implementation Locally ##

download tinystories training and test data to data/
	Use this syntax: TinyStories-(train/valid).txt

compile and run the prep_data program from root 
	(info in tools/prep_data.c)

run make train_serial; ./train_serial
	this uses a preset data path that can be changed via macro in train/train_serial.c
	
## 20-second serial trial

From the repository root:

```sh
make prep_data train_serial
./prep_data
./train_serial
```

Preparation regenerates the matching train/validation/vocabulary/merge files:
25 MiB of training text read, 20 million tokenizer characters, and approximately
five million training tokens. Run preparation once; subsequent trials can reuse
these files. It is excluded from the training time budget.

The trial uses the default model (4096 vocabulary entries, context 256, six
layers, four heads, 128 channels), batch size 32, sequence length 256, and the
existing learning rate 0.001. Training stops after the first complete optimizer
step that reaches 20 seconds. A single serial step can exceed 20 seconds;
this is a soft training limit, not a process timeout. Setup and checkpoint
writing are additional time. The model allocates roughly 1.8 GiB of parameter,
optimizer, activation, and gradient buffers at this batch/context size.

Successful completion writes `checkpoints/serial-trial/model.bin`, alongside
copies of the matching `vocab.bin` and `merges.bin`. The checkpoint preserves
model configuration and weights, not AdamW state or training progress. Move
the run directory before another trial; an existing model is not overwritten.
The printed final batch loss was measured before the last optimizer update.

`make test_checkpoint` checks exact parameter and forward-output round trips,
unsupported versions, inconsistent parameter counts, and truncated files.

## Try the saved model

```sh
make generate
./generate checkpoints/serial-trial 64 42
```

This loads the saved weights and matching vocabulary and samples up to 64 new
tokens, starting with the end-of-story token. It stops early if the model emits
another end-of-story token. Change `42` (the random seed) for a different sample.
The token limit can be 1–256 and must fit the saved model's context length.
There is no prompt input yet, so `merges.bin` is not needed for this command.
Generation does not train or modify the checkpoint. A 20-second trial will
usually produce incoherent text; this command lets you inspect its output.
