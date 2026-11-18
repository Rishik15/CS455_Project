# CS455 Project

A small language model written from scratch in C. It learns to write short children's stories by reading the TinyStories dataset and predicting the next piece of text.

The project aims to compare training on one CPU, across machines with MPI, on one GPU with CUDA, and across machines with GPUs (MPI + CUDA).

## Install dependencies (Fedora)

```sh
sudo dnf install gcc make glibc-devel curl
```
## Download TinyStories (from the repository root)

```sh
mkdir -p data
test -s data/TinyStories-train.txt || curl -fL --retry 3 \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt \
  -o data/TinyStories-train.txt
test -s data/TinyStories-valid.txt || curl -fL --retry 3 \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt \
  -o data/TinyStories-valid.txt
```

## Preprocess data

```sh
make prep_data
./prep_data
```

## Run the serial training session (30 minute budget, batch 32, sequence 256, learning rate 0.001)`
## budget can be changed in train/train-serial
```sh
make train_serial
./train_serial
```

Each run automatically creates `checkpoints/serial-YYYYMMDD-HHMMSS-XXXXXX/`
using its local start date and time, with a unique suffix to prevent collisions.
Weights are saved there when training finishes, alongside the tokenizer files.
The directory is printed at startup; previous runs are preserved.

To generate text, pass the printed run directory:

```sh
make generate
./generate checkpoints/serial-YYYYMMDD-HHMMSS-XXXXXX 128
```
