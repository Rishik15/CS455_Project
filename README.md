# CS455 Project

A small language model written from scratch in C. It learns to write short children's stories by reading the TinyStories dataset and predicting the next piece of text.

The project aims to compare training on one CPU, across machines with MPI, on one GPU with CUDA, and across machines with GPUs (MPI + CUDA).

## Running the Serial Implementation Locally

```sh
sudo dnf install gcc make glibc-devel curl
```

```sh
mkdir -p data
curl -fL --retry 3 \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt \
  -o data/TinyStories-train.txt
curl -fL --retry 3 \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt \
  -o data/TinyStories-valid.txt
```

```sh
make prep_data
./prep_data
```

## 20-second serial trial

```sh
make train_serial
./train_serial
```

## Try the saved model

```sh
make generate
./generate checkpoints/serial-trial 64 42
```
